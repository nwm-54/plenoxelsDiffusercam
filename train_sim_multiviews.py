import os
import random
import sys
import uuid
from argparse import ArgumentParser, Namespace
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

import wandb
from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from lpipsPyTorch import lpips
from scene import GaussianModel, Scene, multiplexing
from scene.cameras import Camera
from utils.general_utils import get_dataset_name, safe_state
from utils.image_utils import heteroscedastic_noise, psnr, quantize_14bit
from utils.loss_utils import l1_loss, ssim

try:
    from torch.utils.tensorboard.writer import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
    SummaryWriter = None  # type: ignore

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.set_device(0)
    print(f"Using device: {device}")
    print(f"GPU Name: {torch.cuda.get_device_name(device)}")
else:
    print("Using device: cpu")


def training(
    dataset: ModelParams,
    opt: OptimizationParams,
    pipe: PipelineParams,
    testing_iterations: List[int],
    saving_iterations: List[int],
    checkpoint_iterations: List[int],
    checkpoint: int,
    debug_from: int,
    resolution: int,
    dls: float,
    size_threshold_arg: int,
    extent_multiplier: float,
    include_test_cameras: bool = False,
):
    tb_writer, model_path = prepare_output_and_logger(dataset)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, include_test_cameras=include_test_cameras)
    gaussians.training_setup(opt)

    initial_log: Dict[str, float] = {}
    if getattr(scene, "avg_angle", None) is not None:
        initial_log["average_angle"] = float(scene.avg_angle)
    group_metrics = getattr(scene, "group_metrics", {}) or {}
    for gid, values in group_metrics.items():
        angle_val = values.get("angle_deg")
        dist_val = values.get("distance_m")
        if angle_val is not None:
            initial_log[f"camera_groups/{gid}/angle_deg"] = float(angle_val)
        if dist_val is not None:
            initial_log[f"camera_groups/{gid}/distance_m"] = float(dist_val)
    if initial_log:
        wandb.log(initial_log, step=0)
    print("Tv weight ", opt.tv_weight)
    print("TV unseen weight ", opt.tv_unseen_weight)
    print("Train cameras:", sum(len(c) for c in scene.getTrainCameras().values()))
    print("Adjacent test cameras:", len(scene.getTestCameras()))
    print("Full test cameras:", len(scene.getFullTestCameras()))

    if dataset.use_multiplexing:
        print(f"Using multiplexing with {scene.n_multiplexed_images} sub-images")
        H = W = 800 // resolution
        scene.init_multiplexing(dls, H, W)
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
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(0, opt.iterations), desc="Training progress")

    all_train_cameras = scene.getTrainCameras()
    view_indices = list(all_train_cameras.keys())
    available_view_indices: List[int] = []

    for iteration in range(1, opt.iterations + 1):
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # TODO make this more frequent because we are only using 3000 iterations
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device=device) if opt.random_background else background

        total_train_loss = 0.0
        mean_train_tv_loss = 0.0
        all_render_pkgs: List[Dict[str, torch.Tensor]] = []

        cameras_to_train: Dict[int, List[Camera]]
        num_cameras_to_sample = 10 if dataset.use_multiplexing else 120
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

        for _, viewpoint_cam in cameras_to_train.items():
            tv_train_loss = 0.0
            gt_image = None
            rendered_image = None

            if dataset.use_multiplexing:
                rendered_sub_images = []
                for single_viewpoint in viewpoint_cam:
                    render_pkg = render(
                        single_viewpoint, gaussians, pipe, bg, scaling_modifier=1.0
                    )
                    all_render_pkgs.append(render_pkg)
                    rendered_sub_image = render_pkg["render"]
                    if single_viewpoint.mask is not None:
                        mask = single_viewpoint.mask.to(device)
                        rendered_sub_image *= mask
                    tv_train_loss += tv_2d(rendered_sub_image)
                    rendered_sub_images.append(rendered_sub_image)
                tv_train_loss /= len(viewpoint_cam)
                rendered_image = multiplexing.generate(
                    rendered_sub_images,
                    scene.comap_yx,
                    scene.dim_lens_lf_yx,
                    scene.n_multiplexed_images,
                    H,
                    W,
                    scene.max_overlap,
                )
                viewpoint_index = int(viewpoint_cam[0].image_name.split("_")[1])
                gt_image = multiplexed_gt[viewpoint_index].to(device)
            else:  # single view case
                assert len(viewpoint_cam) == 1, (
                    "There should only be one camera in viewpoint_cam"
                )
                cam: Camera = viewpoint_cam[0]
                render_pkg = render(cam, gaussians, pipe, bg, scaling_modifier=1.0)
                all_render_pkgs.append(render_pkg)
                tv_train_loss = tv_2d(render_pkg["render"])
                rendered_image = render_pkg["render"]
                if cam.mask is not None:
                    mask = cam.mask.to(device)
                    rendered_image *= mask
                gt_image = cam.original_image.to(device)

            # add heteroscedastic noise
            rendered_image = heteroscedastic_noise(
                rendered_image, opt.lambda_read, opt.lambda_shot
            )
            # quantize rendered image to 14 bit depth
            rendered_image = quantize_14bit(rendered_image)
            L_l1 = (1.0 - opt.lambda_dssim) * l1_loss(rendered_image, gt_image)
            _ssim = opt.lambda_dssim * (1.0 - ssim(rendered_image, gt_image))
            train_tv = opt.tv_weight * tv_train_loss
            mean_train_tv_loss += train_tv.item()
            total_train_loss += L_l1 + _ssim + train_tv

        total_train_loss /= len(all_train_cameras)
        mean_train_tv_loss /= len(all_train_cameras)
        # TV viewpoint
        tv_viewpoints = random.choices(scene.getFullTestCameras(), k=3)
        tv_loss = 0.0
        for tv_viewpoint in tv_viewpoints:
            tv_image = render(tv_viewpoint, gaussians, pipe, bg)["render"]
            tv_loss += tv_2d(tv_image)

        # Loss
        unseen_tv = tv_loss / len(tv_viewpoints) * opt.tv_unseen_weight
        loss = total_train_loss + unseen_tv

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(
                tb_writer,
                iteration,
                loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                (pipe, background),
                unseen_tv,
                model_path,
                total_train_loss,
                mean_train_tv_loss,
                opt,
                (dataset.use_multiplexing, dataset.use_stereo, dataset.use_iphone),
                (
                    scene.comap_yx,
                    scene.dim_lens_lf_yx,
                    scene.n_multiplexed_images,
                    H,
                    W,
                    scene.max_overlap,
                )
                if dataset.use_multiplexing
                else None,
                gt_image,
            )
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                for render_pkg in all_render_pkgs:
                    visibility_filter, radii, viewspace_point_tensor = (
                        render_pkg["visibility_filter"],
                        render_pkg["radii"],
                        render_pkg["viewspace_points"],
                    )
                    gaussians.max_radii2D[visibility_filter] = torch.max(
                        gaussians.max_radii2D[visibility_filter],
                        radii[visibility_filter],
                    )
                    gaussians.add_densification_stats(
                        viewspace_point_tensor, visibility_filter
                    )

                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    size_threshold = (
                        size_threshold_arg
                        if iteration > opt.opacity_reset_interval
                        else None
                    )
                    if scene.cameras_extent < 0.05:
                        scene.cameras_extent = 4.8  # hotfix for when single-view case is not able to set cameras_extent
                    gaussians.densify_and_prune(
                        max_grad=opt.densify_grad_threshold,
                        min_opacity=0.005,
                        extent=scene.cameras_extent * extent_multiplier,
                        max_screen_size=size_threshold,
                        radii=radii,
                    )

                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)


def tv_2d(image: torch.Tensor) -> torch.Tensor:
    return (
        torch.square(image[:, 1:, :] - image[:, :-1, :]).mean()
        + torch.square(image[:, :, 1:] - image[:, :, :-1]).mean()
    )


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer, args.model_path


def training_report(
    tb_writer: SummaryWriter,  # type: ignore
    iteration: int,
    loss: torch.Tensor,
    elapsed: float,
    testing_iterations: List[int],
    scene: Scene,
    renderArgs: Tuple[PipelineParams, torch.Tensor],
    unseen_tv_loss: torch.Tensor,
    model_path: str,
    train_loss: torch.Tensor,
    mean_train_tv_loss: float,
    opt: OptimizationParams,
    mode_flags: Tuple[bool, bool, bool],
    multiplexing_args: Optional[
        Tuple[torch.Tensor, List[int], int, int, int, int]
    ] = None,
    gt_image: Optional[torch.Tensor] = None,
):
    subset_cameras = random.sample(scene.getFullTestCameras(), 10)
    l1_subset, psnr_subset = 0.0, 0.0
    for viewpoint in subset_cameras:
        out = render(viewpoint, scene.gaussians, *renderArgs)["render"]
        gt = viewpoint.original_image.to(device)
        l1_subset += l1_loss(out, gt).mean().double()
        psnr_subset += psnr(out, gt).mean().double()

    total_pts = scene.gaussians.get_xyz.shape[0]

    log_dict = {
        "total_loss": loss.item(),
        "unseen_tv_loss": unseen_tv_loss.item(),
        "train_tv_loss": mean_train_tv_loss,
        "train_loss": train_loss.item(),
        "eval_l1": l1_subset.item() / len(subset_cameras),
        "eval_psnr": psnr_subset.item() / len(subset_cameras),
        "total_points": total_pts,
        "iter_time": elapsed,
    }

    if tb_writer:
        for key, value in log_dict.items():
            tb_writer.add_scalar(key, value, iteration)

    wandb.log(log_dict, step=iteration)
    if iteration in testing_iterations:
        img_dict = {}

        validation_configs = (
            {"name": "adjacent test camera", "cameras": scene.getTestCameras()},
            {"name": "full test camera", "cameras": scene.getFullTestCameras()},
            {
                "name": "train camera",
                "cameras": [
                    cam
                    for cam_list in scene.getTrainCameras().values()
                    for cam in cam_list
                ],
            },
        )

        for config in validation_configs:
            cams = config["cameras"]
            if not cams:
                continue

            l1_test, psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0, 0.0
            for idx, viewpoint in enumerate(cams):
                out = render(viewpoint, scene.gaussians, *renderArgs)["render"]
                gt = viewpoint.original_image.to(device)

                if idx < 3:
                    out_render = heteroscedastic_noise(
                        out, opt.lambda_read, opt.lambda_shot
                    )
                    out_render = quantize_14bit(out_render)
                    if tb_writer:
                        tb_writer.add_images(
                            f"render/{config['name']}_{viewpoint.image_name}/",
                            out_render[None],
                            global_step=iteration,
                        )
                        tb_writer.add_images(
                            f"ground_truth/{config['name']}_{viewpoint.image_name}/",
                            gt[None],
                            global_step=iteration,
                        )
                    img_dict[f"render/{config['name']}_{viewpoint.image_name}"] = (
                        wandb.Image(out_render.cpu())
                    )

                l1_test += l1_loss(out, gt).mean().double()
                psnr_test += psnr(out, gt).mean().double()
                ssim_test += ssim(out, gt)
                lpips_test += (
                    lpips(out.unsqueeze(0), gt.unsqueeze(0), net_type="vgg")
                    .mean()
                    .double()
                )

            l1_test /= len(cams)
            psnr_test /= len(cams)
            ssim_test /= len(cams)
            lpips_test /= len(cams)

            msg = f"[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test:.4f} PSNR {psnr_test:.4f} SSIM {ssim_test:.4f} LPIPS {lpips_test:.4f}"
            print(msg)

            if tb_writer:
                tb_writer.add_scalar(f"l1_loss/{config['name']}", l1_test, iteration)
                tb_writer.add_scalar(f"psnr/{config['name']}", psnr_test, iteration)
                tb_writer.add_scalar(f"ssim/{config['name']}", ssim_test, iteration)
                tb_writer.add_scalar(f"lpips/{config['name']}", lpips_test, iteration)
            log_dict[f"l1_loss/{config['name']}"] = l1_test.item()
            log_dict[f"psnr/{config['name']}"] = psnr_test.item()
            log_dict[f"ssim/{config['name']}"] = ssim_test.item()
            log_dict[f"lpips/{config['name']}"] = lpips_test.item()

        if multiplexing_args is not None:
            train_cameras = list(scene.getTrainCameras().values())[0]
            image_list = [
                render(single_viewpoint, scene.gaussians, *renderArgs)["render"]
                for single_viewpoint in train_cameras
            ]
            multiplexed_image = multiplexing.generate(image_list, *multiplexing_args)
            if tb_writer:
                tb_writer.add_images(
                    "render/trained_multiplex",
                    multiplexed_image.unsqueeze(0),
                    global_step=iteration,
                )
            img_dict["render/trained_multiplex"] = wandb.Image(multiplexed_image.cpu())

        if gt_image is not None:
            if tb_writer:
                tb_writer.add_images(
                    "render/gt_image", gt_image.unsqueeze(0), global_step=iteration
                )
            img_dict["render/gt_image"] = wandb.Image(gt_image.cpu())
        wandb.log({**log_dict, **img_dict}, step=iteration)


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

    if opt.iterations not in args.save_iterations:
        args.save_iterations.append(opt.iterations)

    if dataset.use_multiplexing:
        multiplexing_str = "multiplexing"
    elif dataset.use_stereo:
        multiplexing_str = "stereo"
    elif dataset.use_iphone:
        multiplexing_str = "iphone"
    else:
        multiplexing_str = "singleview"
    run_name = f"{get_dataset_name(dataset.source_path)}_{dataset.n_train_images}views_{multiplexing_str}_dls{args.dls}"
    if opt.tv_weight > 0:
        run_name += f"_tv{opt.tv_weight}"
    if opt.tv_unseen_weight > 0:
        run_name += f"_unseen{opt.tv_unseen_weight}"
    if dataset.camera_offset != 0:
        run_name += f"_offset{dataset.camera_offset}"
    if dataset.use_iphone and not dataset.iphone_same_focal_length:
        run_name += "_multifocal"
    if dataset.angle_deg:
        run_name += f"_{dataset.angle_deg}deg"

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
    wandb.init(name=run_name)

    training(
        dataset=dataset,
        opt=opt,
        pipe=pipe,
        testing_iterations=args.test_iterations,
        saving_iterations=args.save_iterations,
        checkpoint_iterations=args.checkpoint_iterations,
        checkpoint=args.start_checkpoint,
        debug_from=args.debug_from,
        resolution=args.resolution,
        dls=args.dls,
        size_threshold_arg=args.size_threshold,
        extent_multiplier=args.extent_multiplier,
    )

    # All done
    print("\nTraining complete.")
