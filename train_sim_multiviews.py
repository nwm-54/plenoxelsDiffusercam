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


def training(
    *,
    scene: Scene,
    gaussians: GaussianModel,
    dataset: ModelParams,
    opt: OptimizationParams,
    pipe: PipelineParams,
    testing_iterations: List[int],
    saving_iterations: List[int],
    debug_from: int,
    resolution: int,
    dls: float,
    size_threshold: int,
    extent_multiplier: float,
    wandb_images: WandbImageConfig,
    profile_memory: bool,
    tb_writer,
    wandb_module,
) -> None:
    if device.type != "cuda":
        raise RuntimeError(
            "CUDA device is required for training; no compatible GPU was detected."
        )
    testing_iterations = list(testing_iterations)
    saving_iterations = list(saving_iterations)

    print("lambda_dssim", opt.lambda_dssim)

    initial_log: Dict[str, float] = {}
    if getattr(scene, "avg_angle", None) is not None:
        initial_log["average_angle"] = float(scene.avg_angle)
    if initial_log:
        wandb_module.log(initial_log, step=0)
    log_initial_scene_summary(scene, opt)

    H = W = 0
    multiplexing_args = None
    if dataset.use_multiplexing:
        print(f"Using multiplexing with {scene.n_multiplexed_images} sub-images")
        if resolution and resolution > 0:
            scaled_res = max(1, int(800 / float(resolution)))
            H = W = scaled_res
        else:
            H = W = 800
        scene.init_multiplexing(dls, H, W)
        multiplexing_args = (
            scene.comap_yx,
            scene.dim_lens_lf_yx,
            scene.n_multiplexed_images,
            H,
            W,
            scene.microlens_weights,
            scene.throughput_map,
        )
    multiplexed_gt = scene.multiplexed_gt

    metrics_summary_path = os.path.join(dataset.model_path, "metrics_summary.json")

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

    profile_gpu = profile_memory and torch.cuda.is_available()
    # Progress tracking
    progress_bar = tqdm(range(0, opt.iterations), desc="Training progress")
    ema_loss_for_log = 0.0

    unseen_tv_queue: List = []

    for iteration in range(1, opt.iterations + 1):
        if profile_gpu:
            torch.cuda.reset_peak_memory_stats()

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Promote SH degree occasionally (legacy cadence: every 500 iterations)
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()

        if iteration - 1 == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device=device) if opt.random_background else background

        accumulated_loss = 0.0
        accumulated_train_loss = 0.0
        accumulated_train_tv = 0.0
        last_radii: Optional[torch.Tensor] = None
        last_group_scale: Optional[float] = None

        lambda_read, lambda_shot, noise_scale = compute_noise_lambdas(
            opt, iteration, total_train_views
        )

        extra_log: Dict[str, float] = {
            "noise_scale": noise_scale,
            "lambda_read": lambda_read,
            "lambda_shot": lambda_shot,
        }
        # keep logging minimal to avoid confusion

        iteration_loss_sum = 0.0
        iteration_tv_sum = 0.0
        iteration_view_count = 0
        last_gt_image: Optional[torch.Tensor] = None

        cameras_to_train: Dict[int, List] = all_train_cameras
        group_items = list(cameras_to_train.items())
        random.shuffle(group_items)
        total_groups = len(group_items)
        if total_groups == 0:
            continue

        total_sampled_views = total_groups

        # Compute unseen TV once per iteration
        tv_candidates = scene.getFullTestCameras()
        if tv_candidates:
            if not unseen_tv_queue:
                unseen_tv_queue = tv_candidates.copy()
                random.shuffle(unseen_tv_queue)
            sample_count = min(3, len(tv_candidates))
            tv_viewpoints: List = []
            while len(tv_viewpoints) < sample_count:
                if not unseen_tv_queue:
                    unseen_tv_queue = tv_candidates.copy()
                    random.shuffle(unseen_tv_queue)
                candidate = unseen_tv_queue.pop()
                if candidate in tv_viewpoints:
                    continue
                tv_viewpoints.append(candidate)
            tv_loss = sum(
                tv_2d(render(vp, gaussians, pipe, bg)["render"]) for vp in tv_viewpoints
            )
            unseen_tv = (tv_loss / len(tv_viewpoints)) * opt.tv_unseen_weight
        else:
            unseen_tv = torch.zeros((), device=device, dtype=torch.float32)

        # Backprop unseen TV first, then accumulate per-group losses
        if unseen_tv.requires_grad:
            unseen_tv.backward(retain_graph=False)

        mean_train_tv_loss = 0.0
        iteration_loss_sum = 0.0
        iteration_tv_sum = 0.0
        iteration_view_count = 0
        cameras_processed = 0

        # Start with a conservative micro-batch size and adapt on OOM.
        # Empirically, 12 multiplexed groups keeps peak usage ~32GB on a 48GB card.
        micro_bs = 12 if dataset.use_multiplexing else 32
        micro_bs = min(micro_bs, total_groups)
        start_idx = 0
        while start_idx < total_groups:
            end_idx = min(total_groups, start_idx + micro_bs)
            batch = group_items[start_idx:end_idx]
            try:
                micro_sum = torch.zeros((), device=device, dtype=torch.float32)
                micro_tv_acc = 0.0
                micro_count = 0
                micro_render_pkgs: List[Dict[str, torch.Tensor]] = []
                for group_id, viewpoint_cam in batch:
                    if not viewpoint_cam:
                        continue

                    group_scale = getattr(scene, "group_pruning_scales", {}).get(
                        group_id, getattr(scene, "pruning_extent_scale", 1.0)
                    )

                    if dataset.use_multiplexing:
                        rendered_image, tv_train_loss, render_pkgs = (
                            render_multiplexed_view(
                                viewpoint_cam, gaussians, pipe, bg, scene, H, W, device
                            )
                        )
                        micro_render_pkgs.extend(render_pkgs)
                        gt_tensor = multiplexed_gt.get(group_id)
                        if gt_tensor is None:
                            raise KeyError(
                                f"Missing multiplexed ground truth for group {group_id}"
                            )
                        gt_image = gt_tensor.to(device, dtype=torch.float32)

                        rendered_for_loss = (
                            quantize_14bit(
                                heteroscedastic_noise(
                                    rendered_image, lambda_read, lambda_shot
                                )
                            )
                            if (lambda_read > 0.0 or lambda_shot > 0.0)
                            else rendered_image
                        )
                        L_l1 = (1.0 - opt.lambda_dssim) * l1_loss(
                            rendered_for_loss, gt_image
                        )
                        ssim_term = opt.lambda_dssim * (
                            1.0 - ssim(rendered_for_loss, gt_image)
                        )
                        train_tv = opt.tv_weight * tv_train_loss
                    else:
                        # For stereo/iPhone, render each camera independently
                        rendered_images, tv_losses, render_pkgs = render_single_view(
                            viewpoint_cam, gaussians, pipe, bg, device
                        )
                        micro_render_pkgs.extend(render_pkgs)

                        # Compute loss separately for each camera and average
                        L_l1_per_cam = []
                        ssim_per_cam = []
                        tv_per_cam = []

                        for rendered_image, tv_loss, cam in zip(
                            rendered_images, tv_losses, viewpoint_cam
                        ):
                            gt_image = cam.original_image.to(
                                device, dtype=torch.float32
                            )

                            rendered_for_loss = (
                                quantize_14bit(
                                    heteroscedastic_noise(
                                        rendered_image, lambda_read, lambda_shot
                                    )
                                )
                                if (lambda_read > 0.0 or lambda_shot > 0.0)
                                else rendered_image
                            )

                            L_l1_per_cam.append(
                                (1.0 - opt.lambda_dssim)
                                * l1_loss(rendered_for_loss, gt_image)
                            )
                            ssim_per_cam.append(
                                opt.lambda_dssim
                                * (1.0 - ssim(rendered_for_loss, gt_image))
                            )
                            tv_per_cam.append(opt.tv_weight * tv_loss)

                        # Average losses across all cameras
                        L_l1 = torch.stack(L_l1_per_cam).mean()
                        ssim_term = torch.stack(ssim_per_cam).mean()
                        train_tv = torch.stack(tv_per_cam).mean()

                    micro_sum = micro_sum + L_l1 + ssim_term + train_tv
                    micro_tv_acc += float(train_tv.item())
                    micro_count += 1
                    cameras_processed += 1
                    last_gt_image = gt_image
                    if render_pkgs:
                        last_radii = render_pkgs[-1]["radii"].detach()
                        last_group_scale = float(group_scale)

                if micro_count > 0:
                    # Accumulate gradients: scale by total number of groups for a true mean
                    (micro_sum / total_groups).backward(retain_graph=False)
                    mean_train_tv_loss += micro_tv_acc
                    iteration_loss_sum += micro_sum.detach().item()
                    iteration_tv_sum += micro_tv_acc
                    iteration_view_count += micro_count
                    # Update densification stats incrementally to keep memory bounded
                    if micro_render_pkgs:
                        update_densification_stats(gaussians, micro_render_pkgs)

                start_idx = end_idx
                # Free cached memory aggressively between micro-batches
                torch.cuda.empty_cache()
            except RuntimeError as e:
                if "out of memory" in str(e) and micro_bs > 1:
                    torch.cuda.empty_cache()
                    micro_bs = max(1, micro_bs // 2)
                    continue  # retry same range with smaller batch
                raise

        if iteration_view_count == 0:
            continue

        mean_train_tv_loss = mean_train_tv_loss / iteration_view_count
        train_loss_value = iteration_loss_sum / iteration_view_count
        # accumulated_loss here mirrors final scalar used in training_report
        accumulated_loss = train_loss_value + unseen_tv.detach().item()
        accumulated_train_loss = train_loss_value
        accumulated_train_tv = mean_train_tv_loss
        gt_image = last_gt_image
        unseen_tv_value = unseen_tv.detach().item()

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
        torch.cuda.synchronize()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * accumulated_loss + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(
                iteration=iteration,
                loss=torch.tensor(accumulated_loss),
                elapsed=iter_start.elapsed_time(iter_end),
                unseen_tv_loss=torch.tensor(unseen_tv_value),
                train_loss=torch.tensor(accumulated_train_loss),
                mean_train_tv_loss=float(accumulated_train_tv),
                gt_image=gt_image,
                extra_log=extra_log,
                scene=scene,
                gaussians=gaussians,
                pipe=pipe,
                background=background,
                device=device,
                wandb_images=wandb_images,
                testing_iterations=testing_iterations,
                tb_writer=tb_writer,
                wandb_module=wandb_module,
                multiplexing_args=multiplexing_args,
                summary_path=metrics_summary_path,
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
                        size_threshold
                        if iteration > opt.opacity_reset_interval
                        else None
                    )

                    if scene.cameras_extent < 0.05:
                        # Hotfix for when single-view case is not able to set cameras_extent
                        scene.cameras_extent = 4.8

                    extent_scale = (
                        last_group_scale
                        if last_group_scale is not None
                        else getattr(scene, "pruning_extent_scale", 1.0)
                    )
                    gaussians.densify_and_prune(
                        max_grad=opt.densify_grad_threshold,
                        min_opacity=0.005,
                        extent=scene.cameras_extent * extent_multiplier * extent_scale,
                        max_screen_size=size_threshold,
                        radii=last_radii,
                    )

                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

                last_radii = None
                last_group_scale = None

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
        last_group_scale = None


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
        "--profile_memory",
        action="store_true",
        help="Enable per-iteration GPU memory profiling.",
    )
    parser.add_argument(
        "--use_camera_profile",
        dest="use_camera_profile",
        action="store_true",
        default=True,
        help="Apply camera-specific hyperparameter presets (enabled by default).",
    )
    parser.add_argument(
        "--skip_camera_profile",
        dest="use_camera_profile",
        action="store_false",
        help="Disable camera-specific hyperparameter presets.",
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

    tb_writer, _ = prepare_output_and_logger(dataset)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    if args.skip_train:
        if tb_writer:
            tb_writer.close()
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

    training(
        scene=scene,
        gaussians=gaussians,
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
        profile_memory=args.profile_memory,
        tb_writer=tb_writer,
        wandb_module=wandb,
    )

    if tb_writer:
        tb_writer.close()

    # All done
    print("\nTraining complete.")
