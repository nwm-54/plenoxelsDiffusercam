import itertools
import os
import random
import uuid
from argparse import Namespace
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim

__all__ = [
    "TENSORBOARD_FOUND",
    "SummaryWriter",
    "WandbImageConfig",
    "compose_run_name",
    "log_initial_scene_summary",
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


@dataclass
class WandbImageConfig:
    interval: int
    max_images: int
    enable_eval_images: bool

    def should_log(self, iteration: int, testing_iterations: List[int]) -> bool:
        """Return True if this iteration should emit W&B image artifacts."""
        if (
            not self.enable_eval_images
            or self.max_images <= 0
            or not testing_iterations
        ):
            return False
        if self.interval <= 0:
            return iteration == testing_iterations[-1]
        return iteration % self.interval == 0 or iteration == testing_iterations[-1]

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


def training_report(
    *,
    iteration: int,
    loss: torch.Tensor,
    elapsed: float,
    unseen_tv_loss: torch.Tensor,
    train_loss: torch.Tensor,
    mean_train_tv_loss: float,
    scene: Scene,
    gaussians: GaussianModel,
    pipe: PipelineParams,
    background: torch.Tensor,
    device: torch.device,
    wandb_images: WandbImageConfig,
    testing_iterations: List[int],
    tb_writer,
    wandb_module,
    gt_image: Optional[torch.Tensor] = None,
    extra_log: Optional[Dict[str, Any]] = None,
    multiplexing_args: Optional[
        Tuple[torch.Tensor, List[int], int, int, int, int]
    ] = None,
) -> None:
    full_test_cameras = scene.getFullTestCameras()
    subset_size = min(10, len(full_test_cameras))
    l1_subset, psnr_subset = 0.0, 0.0

    if subset_size > 0:
        subset_cameras = random.sample(full_test_cameras, subset_size)
        for viewpoint in subset_cameras:
            out = render(viewpoint, gaussians, pipe, background)["render"]
            gt = viewpoint.original_image.to(device)
            l1_subset += l1_loss(out, gt).mean().double().item()
            psnr_subset += psnr(out, gt).mean().double().item()

        l1_subset = l1_subset / subset_size
        psnr_subset = psnr_subset / subset_size
    else:
        l1_subset = 0.0
        psnr_subset = 0.0

    log_dict = {
        "total_loss": loss.item(),
        "unseen_tv_loss": unseen_tv_loss.item(),
        "train_tv_loss": mean_train_tv_loss,
        "train_loss": train_loss.item(),
        "eval_l1": l1_subset,
        "eval_psnr": psnr_subset,
        "total_points": gaussians.get_xyz.shape[0],
        "iter_time": elapsed,
    }

    if extra_log:
        log_dict.update(extra_log)

    if tb_writer:
        for key, value in log_dict.items():
            tb_writer.add_scalar(key, value, iteration)

    img_dict: Dict[str, Any] = {}
    if iteration in testing_iterations:
        log_images_this_iter = wandb_images.should_log(
            iteration, testing_iterations
        )

        def _init_metric_accumulator() -> Dict[str, float]:
            return {"l1": 0.0, "psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "count": 0.0}

        def _accumulate_metrics(
            acc: Dict[str, float], pred: torch.Tensor, gt: torch.Tensor
        ) -> None:
            pred = pred.clamp(0.0, 1.0)
            acc["l1"] += float(l1_loss(pred, gt).mean().item())
            acc["psnr"] += float(psnr(pred, gt).mean().item())
            acc["ssim"] += float(ssim(pred, gt).mean().item())
            acc["lpips"] += float(
                lpips(pred.unsqueeze(0), gt.unsqueeze(0), net_type="vgg").mean().item()
            )
            acc["count"] += 1.0

        def _evaluate_split(name: str, samples: Iterable[Dict[str, torch.Tensor]]) -> None:
            metrics = _init_metric_accumulator()
            logged = 0

            with torch.no_grad():
                for idx, sample in enumerate(samples):
                    pred = sample["pred"]
                    gt = sample["gt"]
                    label = sample.get("label", str(idx))

                    _accumulate_metrics(metrics, pred, gt)

                    if log_images_this_iter and logged < wandb_images.max_images:
                        pred_img = pred.detach().clamp(0.0, 1.0).cpu()
                        gt_img = gt.detach().clamp(0.0, 1.0).cpu()
                        if tb_writer:
                            tb_writer.add_images(
                                f"render/{name}_{label}",
                                pred_img.unsqueeze(0),
                                global_step=iteration,
                            )
                            tb_writer.add_images(
                                f"ground_truth/{name}_{label}",
                                gt_img.unsqueeze(0),
                                global_step=iteration,
                            )
                        img_dict[f"render/{name}_{label}"] = wandb_module.Image(pred_img)
                        logged += 1

            if metrics["count"] == 0:
                return

            l1_avg = metrics["l1"] / metrics["count"]
            psnr_avg = metrics["psnr"] / metrics["count"]
            ssim_avg = metrics["ssim"] / metrics["count"]
            lpips_avg = metrics["lpips"] / metrics["count"]

            msg = (
                f"[ITER {iteration}] Evaluating {name}: "
                f"L1 {l1_avg:.4f} PSNR {psnr_avg:.4f} "
                f"SSIM {ssim_avg:.4f} LPIPS {lpips_avg:.4f}"
            )
            print(msg)

            if tb_writer:
                tb_writer.add_scalar(f"l1_loss/{name}", l1_avg, iteration)
                tb_writer.add_scalar(f"psnr/{name}", psnr_avg, iteration)
                tb_writer.add_scalar(f"ssim/{name}", ssim_avg, iteration)
                tb_writer.add_scalar(f"lpips/{name}", lpips_avg, iteration)
            log_dict[f"l1_loss/{name}"] = l1_avg
            log_dict[f"psnr/{name}"] = psnr_avg
            log_dict[f"ssim/{name}"] = ssim_avg
            log_dict[f"lpips/{name}"] = lpips_avg

        def _iterate_single_cameras(cameras: Iterable[Camera]):
            for cam in cameras:
                pred = render(cam, gaussians, pipe, background)["render"]
                yield {
                    "pred": torch.clamp(pred, 0.0, 1.0),
                    "gt": cam.original_image.to(device),
                    "label": cam.image_name or f"camera_{cam.uid}",
                }

        def _iterate_multiplexed(groups: Dict[int, List[Camera]]):
            mp_gt = getattr(scene, "multiplexed_gt", {}) or {}
            for group_id, cam_group in groups.items():
                if not cam_group:
                    continue
                gt_image = mp_gt.get(group_id)
                if gt_image is None:
                    continue
                gt_tensor = gt_image.to(device, dtype=torch.float32)
                rendered, _, _ = render_multiplexed_view(
                    cam_group,
                    gaussians,
                    pipe,
                    background,
                    scene,
                    gt_tensor.shape[1],
                    gt_tensor.shape[2],
                    device,
                )
                yield {
                    "pred": torch.clamp(rendered, 0.0, 1.0),
                    "gt": gt_tensor,
                    "label": cam_group[0].image_name
                    if cam_group[0].image_name
                    else f"group_{group_id}",
                }

        _evaluate_split(
            "adjacent test camera", _iterate_single_cameras(scene.getTestCameras())
        )
        _evaluate_split(
            "full test camera", _iterate_single_cameras(scene.getFullTestCameras())
        )

        train_groups = scene.getTrainCameras()
        if getattr(scene, "multiplexed_gt", None):
            _evaluate_split("train camera", _iterate_multiplexed(train_groups))
        else:
            flattened = itertools.chain.from_iterable(train_groups.values())
            _evaluate_split("train camera", _iterate_single_cameras(flattened))

        if multiplexing_args is not None:
            train_cameras = list(scene.getTrainCameras().values())[0]
            image_list: List[torch.Tensor] = []
            for single_viewpoint in train_cameras:
                render_pkg = render(single_viewpoint, gaussians, pipe, background)
                image_list.append(render_pkg["render"])
            multiplexed_image = multiplexing.generate(
                image_list, *multiplexing_args
            )
            if tb_writer:
                tb_writer.add_images(
                    "render/trained_multiplex",
                    multiplexed_image.unsqueeze(0),
                    global_step=iteration,
                )
            img_dict["render/trained_multiplex"] = wandb_module.Image(
                multiplexed_image.cpu()
            )

        if gt_image is not None:
            if tb_writer:
                tb_writer.add_images(
                    "render/gt_image", gt_image.unsqueeze(0), global_step=iteration
                )
            img_dict["render/gt_image"] = wandb_module.Image(gt_image.cpu())
    wandb_module.log({**log_dict, **img_dict}, step=iteration)
