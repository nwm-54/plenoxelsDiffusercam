import os
import shutil
import tempfile
import types
import unittest
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

import train_sim_multiviews as tsm
from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from lpipsPyTorch import lpips
from scene import GaussianModel, Scene
from train_sim_multiviews import TrainingConfig, WandbImageConfig, training
from utils.image_utils import psnr
from utils.loss_utils import ssim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Config = Tuple[str, Sequence[str], Optional[int]]
CONFIGS: Sequence[Config] = (
    ("base", tuple(), None),
    ("stereo", ("--use_stereo",), None),
    ("iphone", ("--use_iphone",), None),
    ("multiplexing", ("--use_multiplexing",), 20),
    ("lightfield", ("--use_multiplexing",), 12),
)

N_TRAIN_IMAGES = 1
TRAIN_ITERS = 1_000
DEFAULT_DATASET_CANDIDATES = (
    os.environ.get("GS_TEST_LEGO_PATH"),
    "/home/wl757/multiplexed-pixels/plenoxels/blender_data/lego_gen12",
    "/home/wl757/multiplexed-pixels/plenoxels/blender_data/lego",
)


def _resolve_lego_path() -> Path:
    for candidate in DEFAULT_DATASET_CANDIDATES:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not locate the lego dataset. Set GS_TEST_LEGO_PATH to the dataset root."
    )


def _build_params(
    args_list: Iterable[str],
) -> Tuple[ModelParams, OptimizationParams, PipelineParams]:
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parsed_args = parser.parse_args(list(args_list))
    model_params = lp.extract(parsed_args)
    optim_params = op.extract(parsed_args)
    pipe_params = pp.extract(parsed_args)
    return model_params, optim_params, pipe_params


def _wandb_stub():
    stub = types.SimpleNamespace(
        init=lambda *a, **k: None,
        login=lambda *a, **k: None,
        log=lambda *a, **k: None,
        Image=lambda *a, **k: None,
    )
    return stub


def _train_and_load(
    model_params: ModelParams,
    opt_params: OptimizationParams,
    pipe_params: PipelineParams,
    cache_dir: Path,
    dls: Optional[int],
) -> Scene:
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_params.model_path = str(cache_dir)
    opt_params.iterations = TRAIN_ITERS

    # Override wandb with no-ops during training to avoid external dependencies
    original_wandb = tsm.wandb
    tsm.wandb = _wandb_stub()
    try:
        training_config = TrainingConfig(
            dataset=model_params,
            opt=opt_params,
            pipe=pipe_params,
            testing_iterations=[],
            saving_iterations=[TRAIN_ITERS],
            debug_from=TRAIN_ITERS + 1,
            resolution=1,
            dls=dls if dls is not None else 20,
            size_threshold=150,
            extent_multiplier=1.0,
            wandb_images=WandbImageConfig(interval=0, max_images=0, enable_eval_images=False),
            include_test_cameras=True,
        )
        training(training_config)
    finally:
        tsm.wandb = original_wandb

    original_use_blender = model_params.use_blender
    model_params.use_blender = False
    scene = Scene(
        model_params,
        GaussianModel(model_params.sh_degree),
        include_test_cameras=True,
    )
    model_params.use_blender = original_use_blender

    ply_path = (
        Path(model_params.model_path)
        / "point_cloud"
        / f"iteration_{TRAIN_ITERS}"
        / "point_cloud.ply"
    )
    if not ply_path.exists():
        raise FileNotFoundError(f"Expected trained point cloud at {ply_path}")

    scene.gaussians.load_ply(str(ply_path))

    if model_params.use_multiplexing and dls is not None:
        _maybe_prepare_multiplexing(scene, model_params, dls)

    return scene


def _evaluate(
    scene: Scene, pipe: PipelineParams, background: torch.Tensor
) -> Dict[str, float]:
    full_test_cameras = scene.getFullTestCameras()
    if not full_test_cameras:
        raise AssertionError(
            "Scene does not provide any full test cameras for evaluation"
        )

    l1_acc = torch.zeros((), dtype=torch.float64, device=DEVICE)
    psnr_acc = torch.zeros((), dtype=torch.float64, device=DEVICE)
    ssim_acc = torch.zeros((), dtype=torch.float64, device=DEVICE)
    lpips_acc = torch.zeros((), dtype=torch.float64, device=DEVICE)

    for cam in full_test_cameras:
        with torch.no_grad():
            render_pkg = render(cam, scene.gaussians, pipe, background)
            pred = render_pkg["render"].to(DEVICE)
        gt = cam.original_image.to(DEVICE)

        l1_acc += F.l1_loss(pred, gt).double()
        psnr_acc += psnr(pred, gt).mean().double()
        ssim_acc += ssim(pred, gt)
        with torch.no_grad():
            lp_val = lpips(pred.unsqueeze(0), gt.unsqueeze(0), net_type="vgg")
        lpips_acc += lp_val.mean().double()

    denom = torch.tensor(len(full_test_cameras), dtype=torch.float64, device=DEVICE)
    return {
        "l1": (l1_acc / denom).item(),
        "psnr": (psnr_acc / denom).item(),
        "ssim": (ssim_acc / denom).item(),
        "lpips": (lpips_acc / denom).item(),
    }


def _maybe_prepare_multiplexing(
    scene: Scene, model_params: ModelParams, dls: Optional[int]
) -> None:
    if not model_params.use_multiplexing or dls is None:
        return
    train_views = scene.getTrainCameras()
    if not train_views:
        return
    first_group = next(iter(train_views.values()))
    if not first_group:
        return
    sample_cam = first_group[0]
    _, height, width = sample_cam.original_image.shape
    scene.init_multiplexing(dls, int(height), int(width))


class LegoConfigEvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.lego_path = _resolve_lego_path()
        except FileNotFoundError as exc:
            raise unittest.SkipTest(str(exc))

    def test_configs_report_metrics(self) -> None:
        results: Dict[str, Dict[str, float]] = {}

        for name, flags, dls in CONFIGS:
            with self.subTest(name=name):
                output_dir = Path(tempfile.mkdtemp(prefix=f"lego_{name}_"))
                args = [
                    "--source_path",
                    str(self.lego_path),
                    "--model_path",
                    str(output_dir),
                    "--n_train_images",
                    str(N_TRAIN_IMAGES),
                    "--iterations",
                    str(TRAIN_ITERS),
                    "--use_blender",
                ]
                args.extend(flags)

                model_params, optim_params, pipe_params = _build_params(args)

                scene = _train_and_load(
                    model_params,
                    optim_params,
                    pipe_params,
                    output_dir,
                    dls,
                )

                background = torch.tensor(
                    [1.0, 1.0, 1.0]
                    if model_params.white_background
                    else [0.0, 0.0, 0.0],
                    dtype=torch.float32,
                    device=DEVICE,
                )

                metrics = _evaluate(scene, pipe_params, background)
                results[name] = metrics

                shutil.rmtree(output_dir, ignore_errors=True)

                for metric_name in ("psnr", "ssim", "lpips"):
                    self.assertIn(
                        metric_name, metrics, f"Missing {metric_name} for {name}"
                    )

        print("\nLego evaluation metrics (n_train_images=3, iterations=1000):")
        for name, metrics in results.items():
            print(
                f"  {name:<12} PSNR {metrics['psnr']:.4f}  SSIM {metrics['ssim']:.4f}  LPIPS {metrics['lpips']:.4f}"
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
