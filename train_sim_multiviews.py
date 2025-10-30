import os
import random
import sys
from argparse import ArgumentParser
from typing import Dict, List, Optional, Tuple

import torch
import wandb
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from arguments.camera_presets import apply_profile
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.general_utils import safe_state
from utils.image_utils import heteroscedastic_noise, quantize_14bit
from utils.loss_utils import l1_loss, ssim
from utils.train_utils import (
    TrainingConfig,
    TrainingReportContext,
    WandbImageConfig,
    compose_run_name,
    group_train_cameras,
    log_initial_scene_summary,
    prepare_output_and_logger,
    render_multiplexed_view,
    render_single_view,
    training_report,
    tv_2d,
    update_densification_stats,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.set_device(0)
    print(f"Using device: {device}")
    print(f"GPU Name: {torch.cuda.get_device_name(device)}")
else:
    print("Using device: cpu")


def compute_noise_lambdas(
    opt: OptimizationParams, iteration: int, total_train_views: int
) -> Tuple[float, float, float]:
    """
    Return the configured heteroscedastic noise coefficients without attenuation.
    Keeping the noise constant preserves the desired camera simulation.
    """
    return opt.lambda_read, opt.lambda_shot, 1.0


def training(config: TrainingConfig) -> None:
    dataset = config.dataset
    opt = config.opt
    pipe = config.pipe

    testing_iterations = list(config.testing_iterations)
    saving_iterations = list(config.saving_iterations)

    tb_writer, _ = prepare_output_and_logger(dataset)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, include_test_cameras=config.include_test_cameras)
    gaussians.training_setup(opt)

    print("lambda_dssim", opt.lambda_dssim)

    initial_log: Dict[str, float] = {}
    if getattr(scene, "avg_angle", None) is not None:
        initial_log["average_angle"] = float(scene.avg_angle)
    if initial_log:
        wandb.log(initial_log, step=0)
    log_initial_scene_summary(scene, opt)

    H = W = 0
    multiplexing_args = None
    if dataset.use_multiplexing:
        print(f"Using multiplexing with {scene.n_multiplexed_images} sub-images")
        if config.resolution and config.resolution > 0:
            scaled_res = max(1, int(800 / float(config.resolution)))
            H = W = scaled_res
        else:
            H = W = 800
        scene.init_multiplexing(config.dls, H, W)
        multiplexing_args = (
            scene.comap_yx,
            scene.dim_lens_lf_yx,
            scene.n_multiplexed_images,
            H,
            W,
            scene.max_overlap,
        )
    multiplexed_gt = scene.multiplexed_gt

    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32,
        device=device,
    )
    iter_start, iter_end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )

    raw_train_cameras = scene.getTrainCameras()
    all_train_cameras = group_train_cameras(raw_train_cameras)
    total_train_views = len(all_train_cameras)

    profile_gpu = config.profile_memory and torch.cuda.is_available()
    subimage_budget = None
    if dataset.use_multiplexing and config.multiplex_max_subimages > 0:
        subimage_budget = max(
            config.multiplex_max_subimages, scene.n_multiplexed_images
        )

    print(
        "Sampling at most {} view groups per iteration; total iterations {}".format(
            "all" if not dataset.use_multiplexing else "limited",
            opt.iterations,
        )
    )

    view_indices = list(all_train_cameras.keys())
    available_view_indices: List[int] = []
    # Decide how many distinct main views to include per iteration (sampling with coverage)
    progress_bar = tqdm(range(0, opt.iterations), desc="Training progress")
    ema_loss_for_log = 0.0

    report_ctx = TrainingReportContext(
        writer=tb_writer,
        testing_iterations=list(testing_iterations),
        scene=scene,
        pipe=pipe,
        background=background,
        device=device,
        opt=opt,
        wandb_images=config.wandb_images,
        wandb=wandb,
        multiplexing_args=multiplexing_args,
    )

    for iteration in range(1, opt.iterations + 1):
        if profile_gpu:
            torch.cuda.reset_peak_memory_stats()

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Promote SH degree occasionally (legacy cadence: every 500 iterations)
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()

        if iteration - 1 == config.debug_from:
            pipe.debug = True

        bg = torch.rand((3), device=device) if opt.random_background else background

        accumulated_loss = 0.0
        accumulated_train_loss = 0.0
        accumulated_train_tv = 0.0
        last_radii: Optional[torch.Tensor] = None
        total_sampled_views = 0

        lambda_read, lambda_shot, noise_scale = compute_noise_lambdas(
            opt, iteration, total_train_views
        )

        extra_log: Dict[str, float] = {
            "noise_scale": noise_scale,
            "lambda_read": lambda_read,
            "lambda_shot": lambda_shot,
        }

        iteration_loss_sum = 0.0
        iteration_tv_sum = 0.0
        iteration_view_count = 0
        last_gt_image: Optional[torch.Tensor] = None

        cameras_to_train: Dict[int, List]
        if dataset.use_multiplexing:
            if subimage_budget:
                max_groups = subimage_budget // scene.n_multiplexed_images
                max_groups = max(1, max_groups)
            else:
                max_groups = len(all_train_cameras)
            num_cameras_to_sample = min(len(view_indices), max_groups)
        else:
            num_cameras_to_sample = min(len(view_indices), 120)

        if len(view_indices) > num_cameras_to_sample:
            if not available_view_indices:
                available_view_indices = view_indices.copy()
                random.shuffle(available_view_indices)

            num_to_sample = min(num_cameras_to_sample, len(available_view_indices))
            sampled_indices = [
                available_view_indices.pop() for _ in range(num_to_sample)
            ]

            if not available_view_indices:
                available_view_indices = view_indices.copy()
                random.shuffle(available_view_indices)

            cameras_to_train = {idx: all_train_cameras[idx] for idx in sampled_indices}
        else:
            cameras_to_train = all_train_cameras

        actual_sampled = len(cameras_to_train)
        total_sampled_views += actual_sampled
        if dataset.use_multiplexing and subimage_budget:
            sampled_subimages = actual_sampled * scene.n_multiplexed_images
            extra_log["sampled_subimages"] = float(sampled_subimages)
            extra_log["subimage_budget"] = float(subimage_budget)

        total_train_loss = torch.zeros((), device=device, dtype=torch.float32)
        mean_train_tv_loss = 0.0
        cameras_processed = 0
        all_render_pkgs: List[Dict[str, torch.Tensor]] = []

        for _, viewpoint_cam in cameras_to_train.items():
            if not viewpoint_cam:
                continue

            if dataset.use_multiplexing:
                rendered_image, tv_train_loss, render_pkgs = render_multiplexed_view(
                    viewpoint_cam, gaussians, pipe, bg, scene, H, W, device
                )
                all_render_pkgs.extend(render_pkgs)
                viewpoint_index = int(viewpoint_cam[0].image_name.split("_")[1])
                gt_image = multiplexed_gt[viewpoint_index].to(
                    device, dtype=torch.float32
                )
            else:
                rendered_image, tv_train_loss, render_pkgs = render_single_view(
                    viewpoint_cam, gaussians, pipe, bg, device
                )
                all_render_pkgs.extend(render_pkgs)
                gt_image = viewpoint_cam[0].original_image.to(
                    device, dtype=torch.float32
                )

            if lambda_read > 0.0 or lambda_shot > 0.0:
                rendered_for_loss = heteroscedastic_noise(
                    rendered_image, lambda_read, lambda_shot
                )
                rendered_for_loss = quantize_14bit(rendered_for_loss)
            else:
                rendered_for_loss = rendered_image

            L_l1 = (1.0 - opt.lambda_dssim) * l1_loss(rendered_for_loss, gt_image)
            ssim_term = opt.lambda_dssim * (1.0 - ssim(rendered_for_loss, gt_image))
            train_tv = opt.tv_weight * tv_train_loss

            total_train_loss = total_train_loss + L_l1 + ssim_term + train_tv
            mean_train_tv_loss += float(train_tv.item())
            cameras_processed += 1
            last_gt_image = gt_image
            if render_pkgs:
                last_radii = render_pkgs[-1]["radii"].detach()

        if cameras_processed == 0:
            continue

        total_train_loss = total_train_loss / cameras_processed
        mean_train_tv_loss = mean_train_tv_loss / cameras_processed
        train_loss_value = total_train_loss.detach().item()
        iteration_loss_sum = train_loss_value * cameras_processed
        iteration_tv_sum = mean_train_tv_loss * cameras_processed
        iteration_view_count = cameras_processed

        tv_candidates = scene.getFullTestCameras()
        if tv_candidates:
            tv_viewpoints = random.choices(tv_candidates, k=min(3, len(tv_candidates)))
            tv_loss = sum(
                tv_2d(render(vp, gaussians, pipe, bg)["render"]) for vp in tv_viewpoints
            )
            unseen_tv = (tv_loss / len(tv_viewpoints)) * opt.tv_unseen_weight
        else:
            unseen_tv = torch.zeros((), device=device, dtype=torch.float32)

        loss = total_train_loss + unseen_tv
        loss.backward()

        iteration_avg_train_loss = iteration_loss_sum / max(iteration_view_count, 1)
        iteration_avg_train_tv = iteration_tv_sum / max(iteration_view_count, 1)
        loss_value = loss.detach().item()
        unseen_tv_value = unseen_tv.detach().item()

        accumulated_loss = loss_value
        accumulated_train_loss = iteration_avg_train_loss
        accumulated_train_tv = iteration_avg_train_tv
        gt_image = last_gt_image

        if profile_gpu:
            torch.cuda.synchronize()
            extra_log["gpu_max_mem_mb"] = torch.cuda.max_memory_allocated(
                device=device
            ) / (1024**2)
            if hasattr(torch.cuda, "max_memory_reserved"):
                extra_log["gpu_max_reserved_mb"] = torch.cuda.max_memory_reserved(
                    device=device
                ) / (1024**2)

        iter_end.record()

        with torch.no_grad():
            if all_render_pkgs:
                update_densification_stats(gaussians, all_render_pkgs)

            # Progress bar
            ema_loss_for_log = 0.4 * accumulated_loss + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(
                report_ctx,
                iteration=iteration,
                loss=torch.tensor(accumulated_loss),
                elapsed=iter_start.elapsed_time(iter_end),
                unseen_tv_loss=torch.tensor(unseen_tv_value),
                train_loss=torch.tensor(accumulated_train_loss),
                mean_train_tv_loss=float(accumulated_train_tv),
                gt_image=gt_image,
                extra_log=extra_log,
                noise_params={"lambda_read": lambda_read, "lambda_shot": lambda_shot},
            )
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter and total_sampled_views > 0:
                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                    and last_radii is not None
                ):
                    size_threshold = (
                        config.size_threshold
                        if iteration > opt.opacity_reset_interval
                        else None
                    )

                    if scene.cameras_extent < 0.05:
                        # Hotfix for when single-view case is not able to set cameras_extent
                        scene.cameras_extent = 4.8

                    gaussians.densify_and_prune(
                        max_grad=opt.densify_grad_threshold,
                        min_opacity=0.005,
                        extent=scene.cameras_extent * config.extent_multiplier,
                        max_screen_size=size_threshold,
                        radii=last_radii,
                    )

                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

                last_radii = None

            elif iteration % opt.opacity_reset_interval == 0 or (
                dataset.white_background and iteration == opt.densify_from_iter
            ):
                gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

        # If densification was not triggered, ensure temporary radii state is cleared
        last_radii = None


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6004)
    parser.add_argument("--debug_from", type=int, default=3000)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations",
        nargs="+",
        type=int,
        default=[10] + list(range(250, 30_000 + 1, 250)),
    )
    parser.add_argument(
        "--save_iterations",
        nargs="+",
        type=int,
        default=list(range(2000, 30_000 + 1, 3000)),
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--checkpoint_iterations",
        nargs="+",
        type=int,
        default=[4000, 6000, 8000, 10000, 20000, 30000],
    )
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--dls", type=int, default=20)
    parser.add_argument("--size_threshold", type=int, default=150)
    parser.add_argument("--extent_multiplier", type=float, default=1.0)
    parser.add_argument("--output-id", type=str, default="3")
    parser.add_argument(
        "--wandb_image_interval",
        type=int,
        default=500,
        help="Log W&B evaluation images every N test iterations (0 logs only final iteration).",
    )
    parser.add_argument(
        "--wandb_max_images",
        type=int,
        default=5,
        help="Maximum number of images per validation set to upload when enabled.",
    )
    parser.add_argument(
        "--wandb_disable_eval_images",
        action="store_true",
        help="Disable logging evaluation render images to W&B.",
    )
    parser.add_argument(
        "--multiplex_max_subimages",
        type=int,
        default=160,
        help="Maximum number of multiplexed sub-images to render per iteration (<=0 disables the limit).",
    )
    parser.add_argument(
        "--profile_memory",
        action="store_true",
        help="Enable per-iteration GPU memory profiling.",
    )
    parser.add_argument(
        "--use_camera_profile",
        action="store_true",
        help="Apply camera-specific hyperparameter presets (matches legacy behaviour when disabled).",
    )
    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="Generate camera trajectories and configuration without running training",
    )
    args = parser.parse_args(sys.argv[1:])

    args.save_iterations.append(args.iterations)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    if args.use_camera_profile:
        opt, camera_profile = apply_profile(dataset, opt, args.dls)
        print(f"Applied '{camera_profile}' hyperparameter preset")
    else:
        camera_profile = "none"

    if opt.iterations not in args.save_iterations:
        args.save_iterations.append(opt.iterations)

    run_name = compose_run_name(dataset, opt, args.dls)

    if not dataset.model_path:
        dataset.model_path = (
            f"/share/monakhova/shamus_data/multiplexed_pixels/output{args.output_id}/"
            + run_name
        )

    if args.skip_train:
        tb_writer, _ = prepare_output_and_logger(dataset)
        if tb_writer:
            tb_writer.close()
        print("DEBUG: Creating GaussianModel...")
        gaussians = GaussianModel(dataset.sh_degree)
        print("DEBUG: Creating Scene...")
        Scene(
            dataset,
            gaussians,
        )
        print("DEBUG: Scene created successfully")
        transforms_path = os.path.join(dataset.model_path, "transforms.json")
        print(f"Transforms written to {transforms_path}")
        sys.exit(0)

    wandb.login()
    wandb.init(
        name=run_name, save_code=False, settings=wandb.Settings(_disable_stats=True)
    )

    wandb_images = WandbImageConfig(
        interval=args.wandb_image_interval,
        max_images=args.wandb_max_images,
        enable_eval_images=not args.wandb_disable_eval_images,
    )

    train_config = TrainingConfig(
        dataset=dataset,
        opt=opt,
        pipe=pipe,
        testing_iterations=list(args.test_iterations),
        saving_iterations=list(args.save_iterations),
        debug_from=args.debug_from,
        resolution=args.resolution,
        dls=args.dls,
        size_threshold=args.size_threshold,
        extent_multiplier=args.extent_multiplier,
        wandb_images=wandb_images,
        multiplex_max_subimages=args.multiplex_max_subimages,
        profile_memory=args.profile_memory,
    )

    training(train_config)

    # All done
    print("\nTraining complete.")
