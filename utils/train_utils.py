import os
import random
import uuid
from argparse import Namespace
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    from torch.utils.tensorboard.writer import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
    SummaryWriter = None  # type: ignore

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from lpipsPyTorch import lpips
from scene import GaussianModel, Scene, multiplexing
from scene.cameras import Camera
from utils.general_utils import get_dataset_name
from utils.image_utils import heteroscedastic_noise, psnr, quantize_14bit
from utils.loss_utils import l1_loss, ssim


__all__ = [
    "TENSORBOARD_FOUND",
    "SummaryWriter",
    "WandbImageConfig",
    "TrainingReportContext",
    "TrainingConfig",
    "compose_run_name",
    "log_initial_scene_summary",
    "collect_validation_configs",
    "group_train_cameras",
    "sample_camera_views",
    "render_multiplexed_view",
    "render_single_view",
    "update_densification_stats",
    "tv_2d",
    "prepare_output_and_logger",
    "training_report",
]


def compose_run_name(dataset: ModelParams, opt: OptimizationParams, dls: int) -> str:
    """Build a descriptive run name that mirrors previous inline construction."""
    if dataset.use_multiplexing:
        multiplexing_str = "multiplexing"
    elif dataset.use_stereo:
        multiplexing_str = "stereo"
    elif dataset.use_iphone:
        multiplexing_str = "iphone"
    else:
        multiplexing_str = "singleview"

    run_name = (
        f"{get_dataset_name(dataset.source_path)}_"
        f"{dataset.n_train_images}views_{multiplexing_str}_dls{dls}"
    )

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
    return run_name


def log_initial_scene_summary(scene: Scene, opt: OptimizationParams) -> None:
    """Mirror the original print statements that describe the scene."""
    print("Tv weight ", opt.tv_weight)
    print("TV unseen weight ", opt.tv_unseen_weight)
    print("Train cameras:", sum(len(c) for c in scene.getTrainCameras().values()))
    print("Adjacent test cameras:", len(scene.getTestCameras()))
    print("Full test cameras:", len(scene.getFullTestCameras()))


def collect_validation_configs(scene: Scene) -> List[Dict[str, Any]]:
    """Centralize creation of validation configurations."""
    def _representative_cameras(cams: List[Camera]) -> List[Camera]:
        return cams

    return [
        {
            "name": "adjacent test camera",
            "cameras": _representative_cameras(scene.getTestCameras()),
            "multiplexed_groups": {},
            "multiplexed_gt": {},
        },
        {
            "name": "full test camera",
            "cameras": _representative_cameras(scene.getFullTestCameras()),
            "multiplexed_groups": {},
            "multiplexed_gt": {},
        },
        {
            "name": "train camera",
            "cameras": [
                cam for cam_list in scene.getTrainCameras().values() for cam in cam_list
            ],
            "multiplexed_groups": scene.getTrainCameras(),
            "multiplexed_gt": getattr(scene, "multiplexed_gt", {}),
        },
    ]


@dataclass
class WandbImageConfig:
    interval: int
    max_images: int
    enable_eval_images: bool

    def should_log(self, iteration: int, testing_iterations: List[int]) -> bool:
        """Return True if this iteration should emit W&B image artifacts."""
        if not self.enable_eval_images or self.max_images <= 0 or not testing_iterations:
            return False
        if self.interval <= 0:
            return iteration == testing_iterations[-1]
        return iteration % self.interval == 0 or iteration == testing_iterations[-1]


@dataclass
class TrainingReportContext:
    writer: Optional[SummaryWriter]  # type: ignore[valid-type]
    testing_iterations: List[int]
    scene: Scene
    pipe: PipelineParams
    background: torch.Tensor
    device: torch.device
    opt: OptimizationParams
    wandb_images: WandbImageConfig
    wandb: Any
    multiplexing_args: Optional[
        Tuple[torch.Tensor, List[int], int, int, int, int]
    ] = None


@dataclass
class TrainingConfig:
    dataset: ModelParams
    opt: OptimizationParams
    pipe: PipelineParams
    testing_iterations: List[int]
    saving_iterations: List[int]
    debug_from: int
    resolution: int
    dls: float
    size_threshold: int
    extent_multiplier: float
    wandb_images: WandbImageConfig
    include_test_cameras: bool = False
    multiplex_max_subimages: int = 160
    profile_memory: bool = False


def group_train_cameras(
    all_train_cameras: Dict[int, List[Camera]],
) -> Dict[int, List[Camera]]:
    """
    Collapse camera entries that share the same ``groupid`` so multi-lens rigs
    are sampled together.
    """
    grouped: Dict[int, List[Camera]] = {}
    for view_idx, cam_list in all_train_cameras.items():
        for cam in cam_list:
            group_id = getattr(cam, "groupid", None)
            if group_id is None:
                group_id = getattr(cam, "uid", view_idx)
                setattr(cam, "groupid", group_id)
            if group_id not in grouped:
                grouped[group_id] = []
            grouped[group_id].append(cam)
    return grouped


def sample_camera_views(
    all_train_cameras: Dict[int, List[Camera]],
    num_cameras_to_sample: int,
) -> Dict[int, List[Camera]]:
    """
    Randomly sample up to ``num_cameras_to_sample`` camera groups for this iteration.
    """
    if num_cameras_to_sample <= 0 or not all_train_cameras:
        return {}

    group_ids = list(all_train_cameras.keys())
    random.shuffle(group_ids)

    if num_cameras_to_sample >= len(group_ids):
        return {idx: all_train_cameras[idx] for idx in group_ids}

    selected = group_ids[:num_cameras_to_sample]
    return {idx: all_train_cameras[idx] for idx in selected}


def render_multiplexed_view(
    viewpoint_cam: List[Camera],
    gaussians: GaussianModel,
    pipe: PipelineParams,
    bg: torch.Tensor,
    scene: Scene,
    height: int,
    width: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, torch.Tensor]]]:
    """
    Render a multiplexed view from multiple sub-images.

    Renders each sub-image separately, applies masks, and combines them into
    a single multiplexed image using the scene's multiplexing parameters.

    Returns (rendered_image, tv_loss, render_packages).
    """
    rendered_sub_images = []
    all_render_pkgs = []
    tv_loss = 0.0

    for single_viewpoint in viewpoint_cam:
        render_pkg = render(single_viewpoint, gaussians, pipe, bg, scaling_modifier=1.0)
        all_render_pkgs.append(render_pkg)

        rendered_sub_image = render_pkg["render"]
        if single_viewpoint.mask is not None:
            rendered_sub_image *= single_viewpoint.mask.to(device)

        tv_loss += tv_2d(rendered_sub_image)
        rendered_sub_images.append(rendered_sub_image)

    tv_loss /= len(viewpoint_cam)

    rendered_image = multiplexing.generate(
        rendered_sub_images,
        scene.comap_yx,
        scene.dim_lens_lf_yx,
        scene.n_multiplexed_images,
        height,
        width,
        scene.max_overlap,
    )

    return rendered_image, tv_loss, all_render_pkgs


def render_single_view(
    viewpoint_cam: List[Camera],
    gaussians: GaussianModel,
    pipe: PipelineParams,
    bg: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, torch.Tensor]]]:
    """Render a single view. Returns (rendered_image, tv_loss, render_packages)."""
    assert len(viewpoint_cam) == 1, "Single view should have exactly one camera"

    cam = viewpoint_cam[0]
    render_pkg = render(cam, gaussians, pipe, bg, scaling_modifier=1.0)

    rendered_image = render_pkg["render"]
    if cam.mask is not None:
        rendered_image *= cam.mask.to(device)

    tv_loss = tv_2d(rendered_image)

    return rendered_image, tv_loss, [render_pkg]


def update_densification_stats(
    gaussians: GaussianModel,
    render_packages: List[Dict[str, torch.Tensor]],
) -> None:
    """Update Gaussian densification statistics from render packages."""
    for render_pkg in render_packages:
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        viewspace_points = render_pkg["viewspace_points"]

        gaussians.max_radii2D[visibility_filter] = torch.max(
            gaussians.max_radii2D[visibility_filter],
            radii[visibility_filter],
        )
        gaussians.add_densification_stats(viewspace_points, visibility_filter)


def tv_2d(image: torch.Tensor) -> torch.Tensor:
    return (
        torch.square(image[:, 1:, :] - image[:, :-1, :]).mean()
        + torch.square(image[:, :, 1:] - image[:, :, :-1]).mean()
    )


def prepare_output_and_logger(args: ModelParams):
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
        tb_writer = SummaryWriter(args.model_path)  # type: ignore[call-arg]
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer, args.model_path


def _synchronize_cuda_or_raise(
    tag: str, camera: Optional[Camera] = None, enable_env: str = "GS_ENABLE_CUDA_SYNC"
) -> None:
    """
    Synchronize CUDA to surface asynchronous errors closer to their origin.
    When CUDA raises, append the current tag (and camera info if available) to aid debugging.
    Set environment variable GS_ENABLE_CUDA_SYNC=1 to turn this diagnostic on.
    """
    if not torch.cuda.is_available() or os.environ.get(enable_env, "") not in {"1", "true", "True"}:
        return

    try:
        torch.cuda.synchronize()
    except RuntimeError as err:
        cam_name = getattr(camera, "image_name", None)
        cam_uid = getattr(camera, "uid", None)
        context = tag
        if cam_name or cam_uid is not None:
            context += f" (camera={cam_name or 'unknown'}, uid={cam_uid})"
        raise RuntimeError(f"CUDA failure detected during {context}") from err


def training_report(
    ctx: TrainingReportContext,
    iteration: int,
    loss: torch.Tensor,
    elapsed: float,
    unseen_tv_loss: torch.Tensor,
    train_loss: torch.Tensor,
    mean_train_tv_loss: float,
    gt_image: Optional[torch.Tensor] = None,
    extra_log: Optional[Dict[str, Any]] = None,
    noise_params: Optional[Dict[str, float]] = None,
) -> None:
    scene = ctx.scene
    tb_writer = ctx.writer
    testing_iterations = ctx.testing_iterations
    device = ctx.device
    wandb = ctx.wandb

    _synchronize_cuda_or_raise("training_report_start")

    lambda_read = (
        noise_params["lambda_read"]
        if noise_params is not None and "lambda_read" in noise_params
        else ctx.opt.lambda_read
    )
    lambda_shot = (
        noise_params["lambda_shot"]
        if noise_params is not None and "lambda_shot" in noise_params
        else ctx.opt.lambda_shot
    )
    apply_noise_for_logging = (lambda_read > 0.0) or (lambda_shot > 0.0)

    full_test_cameras = scene.getFullTestCameras()
    subset_size = min(10, len(full_test_cameras))
    l1_subset, psnr_subset = 0.0, 0.0

    if subset_size > 0:
        subset_cameras = random.sample(full_test_cameras, subset_size)
        for viewpoint in subset_cameras:
            out = render(viewpoint, scene.gaussians, ctx.pipe, ctx.background)["render"]
            _synchronize_cuda_or_raise("eval_subset_render", viewpoint)
            gt = viewpoint.original_image.to(device)
            l1_subset += l1_loss(out, gt).mean().double().item()
            psnr_subset += psnr(out, gt).mean().double().item()

        l1_subset = l1_subset / subset_size
        psnr_subset = psnr_subset / subset_size
    else:
        l1_subset = 0.0
        psnr_subset = 0.0

    total_pts = scene.gaussians.get_xyz.shape[0]

    log_dict = {
        "total_loss": loss.item(),
        "unseen_tv_loss": unseen_tv_loss.item(),
        "train_tv_loss": mean_train_tv_loss,
        "train_loss": train_loss.item(),
        "eval_l1": l1_subset,
        "eval_psnr": psnr_subset,
        "total_points": total_pts,
        "iter_time": elapsed,
    }

    if extra_log:
        log_dict.update(extra_log)

    if tb_writer:
        for key, value in log_dict.items():
            tb_writer.add_scalar(key, value, iteration)

    img_dict: Dict[str, Any] = {}
    if iteration in testing_iterations:
        log_images_this_iter = ctx.wandb_images.should_log(iteration, testing_iterations)

        validation_configs = collect_validation_configs(scene)

        for config in validation_configs:
            cams = config["cameras"]
            if not cams:
                continue

            render_cache: Dict[str, torch.Tensor] = {}
            l1_test, psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0, 0.0
            for idx, viewpoint in enumerate(cams):
                out = render(viewpoint, scene.gaussians, ctx.pipe, ctx.background)[
                    "render"
                ]
                _synchronize_cuda_or_raise(f"{config['name']}_render", viewpoint)
                gt = viewpoint.original_image.to(device)
                render_cache[viewpoint.image_name] = out.detach()

                eval_render = torch.clamp(out, 0.0, 1.0)
                if apply_noise_for_logging:
                    eval_render = heteroscedastic_noise(
                        eval_render, lambda_read, lambda_shot
                    )
                    eval_render = quantize_14bit(eval_render)

                if log_images_this_iter and idx < ctx.wandb_images.max_images:
                    if tb_writer:
                        tb_writer.add_images(
                            f"render/{config['name']}_{viewpoint.image_name}/",
                            eval_render[None],
                            global_step=iteration,
                        )
                        tb_writer.add_images(
                            f"ground_truth/{config['name']}_{viewpoint.image_name}/",
                            gt[None],
                            global_step=iteration,
                        )
                    img_dict[f"render/{config['name']}_{viewpoint.image_name}"] = (
                        wandb.Image(eval_render.cpu())
                    )

                l1_test += l1_loss(eval_render, gt).mean().double()
                psnr_test += psnr(eval_render, gt).mean().double()
                ssim_test += ssim(eval_render, gt)
                lpips_test += (
                    lpips(eval_render.unsqueeze(0), gt.unsqueeze(0), net_type="vgg")
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

            mp_groups = config.get("multiplexed_groups")
            mp_gt = config.get("multiplexed_gt")
            if mp_groups and mp_gt:
                mp_l1_sum = 0.0
                mp_psnr_sum = 0.0
                mp_count = 0

                for group_id, cam_group in mp_groups.items():
                    if group_id not in mp_gt or not cam_group:
                        continue

                    rendered_sub_images: List[torch.Tensor] = []
                    for cam in cam_group:
                        cached = render_cache.get(cam.image_name)
                        if cached is None:
                            cached = render(cam, scene.gaussians, ctx.pipe, ctx.background)[
                                "render"
                            ]
                        rendered = cached.clone()
                        if cam.mask is not None:
                            rendered = rendered * cam.mask.to(device)
                        rendered_sub_images.append(rendered)

                    if not rendered_sub_images:
                        continue

                    gt_image = mp_gt[group_id].to(device, dtype=torch.float32)
                    mp_render = multiplexing.generate(
                        rendered_sub_images,
                        scene.comap_yx,
                        scene.dim_lens_lf_yx,
                        scene.n_multiplexed_images,
                        gt_image.shape[1],
                        gt_image.shape[2],
                        scene.max_overlap,
                    )

                    mp_l1_sum += (
                        l1_loss(mp_render, gt_image).mean().double().item()
                    )
                    mp_psnr_sum += psnr(mp_render, gt_image).mean().double().item()
                    mp_count += 1

                if mp_count > 0:
                    mp_l1 = mp_l1_sum / mp_count
                    mp_psnr_value = mp_psnr_sum / mp_count
                    log_name = f"multiplexed {config['name']}"
                    log_dict[f"l1_loss/{log_name}"] = mp_l1
                    log_dict[f"psnr/{log_name}"] = mp_psnr_value
                    if tb_writer:
                        tb_writer.add_scalar(f"l1_loss/{log_name}", mp_l1, iteration)
                        tb_writer.add_scalar(f"psnr/{log_name}", mp_psnr_value, iteration)

        if ctx.multiplexing_args is not None:
            train_cameras = list(scene.getTrainCameras().values())[0]
            image_list: List[torch.Tensor] = []
            for single_viewpoint in train_cameras:
                render_pkg = render(
                    single_viewpoint, scene.gaussians, ctx.pipe, ctx.background
                )
                _synchronize_cuda_or_raise("train_camera_render", single_viewpoint)
                image_list.append(render_pkg["render"])
            multiplexed_image = multiplexing.generate(
                image_list, *ctx.multiplexing_args
            )
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
