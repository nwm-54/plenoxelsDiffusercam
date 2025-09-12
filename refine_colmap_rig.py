from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import imageio.v3 as iio
import numpy as np
from plenopticam.cfg import PlenopticamConfig
from skimage.util import img_as_float, img_as_ubyte


# —————————————————— Utilities ————————————————————————————
def run(cmd: List[str], cwd: Optional[Path] = None) -> None:
    print("$", " ".join(cmd))
    proc = subprocess.run(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if proc.returncode != 0:
        print(proc.stdout)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    if proc.stdout:
        print(proc.stdout)


def ensure_empty(dirpath: Path):
    if dirpath.exists():
        shutil.rmtree(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)


def load_viewpoints(lfr_path: Path):
    cfg = PlenopticamConfig()
    cfg.params["lfp_path"] = str(lfr_path.resolve())
    vp_dir = max(
        list(Path(cfg.exp_path).glob("**/viewpoints_*px")),
        key=lambda d: len(list(d.glob("*.png"))),
    )

    # Infer grid size (V, U) from filenames like 'YY_XX.png'
    max_j = max_i = -1
    tiles = []
    for f in vp_dir.glob("*.png"):
        m = re.fullmatch(r"(\d{2})_(\d{2})\.png", f.name)
        if m:
            j, i = int(m.group(1)), int(m.group(2))
            max_j, max_i = max(max_j, j), max(max_i, i)
            tiles.append((j, i, f))
    V, U = max_j + 1, max_i + 1

    # Read a sample to get (H, W, C)
    sample_path = vp_dir / "00_00.png"
    if not sample_path.exists():
        sample_path = tiles[0][2]
    sample = iio.imread(sample_path.as_posix())
    if sample.ndim == 2:
        sample = sample[..., None]
    H, W = sample.shape[:2]
    C = sample.shape[2] if sample.ndim == 3 else 1

    # Allocate and fill
    imgs = np.empty((V, U, H, W, C), dtype=sample.dtype)
    for j in range(V):
        for i in range(U):
            fp = vp_dir / f"{j:02d}_{i:02d}.png"
            im = iio.imread(fp.as_posix())
            if im.ndim == 2:
                im = im[..., None]
            imgs[j, i] = im
    return imgs


# —————————————————— COLMAP ————————————————————————————
@dataclass
class RigCamera:
    row: int
    col: int
    name: str
    prefix: Path
    is_ref: bool
    tx: float
    ty: float
    tz: float
    qw: float = 1.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0


def disparity_to_baseline(
    sx_px_per_step: float,
    sy_px_per_step: float,
    focal_px: float,
    scale: float = 0.05,
) -> Tuple[float, float]:
    bx = (sx_px_per_step / focal_px) * scale
    by = (sy_px_per_step / focal_px) * scale
    return float(bx), float(by)


def save_as_rig(
    images: np.ndarray,
    images_root: Path,
    inner: Optional[int] = None,
) -> List[RigCamera]:
    V, U, H, W, C = images.shape
    if inner is not None:
        assert inner <= V and inner <= U, "Inner must smaller than the grid size"
        images = images[inner : V - inner, inner : U - inner]
        V, U = V - 2 * inner, U - 2 * inner
    print(images.shape, V, U)
    vc, uc = V // 2, U // 2
    ensure_empty(images_root)

    cams: List[RigCamera] = []
    for v in range(V):
        for u in range(U):
            cam_folder = images_root / f"{v + inner:02d}_{u + inner:02d}"
            cam_folder.mkdir(parents=True, exist_ok=True)

            img = img_as_float(images[v, u])
            out_path = cam_folder / "frame.png"
            iio.imwrite(out_path.as_posix(), img_as_ubyte(img))

            tx = float(u - uc)
            ty = float(v - vc)
            prefix_rel = (cam_folder.relative_to(images_root)).as_posix() + "/"
            cams.append(
                RigCamera(
                    row=v,
                    col=u,
                    name=f"{v + inner:02d}_{u + inner:02d}",
                    prefix=Path(prefix_rel),
                    is_ref=(v == vc and u == uc),
                    tx=tx,
                    ty=ty,
                    tz=0.0,
                )
            )
    return cams


def save_with_inner(
    images: np.ndarray, out_root: Path, inner: int
) -> List[Tuple[int, int, Path]]:
    V, U, _, _, _ = images.shape

    inner = 0 if inner in (None, 0) else int(inner)
    assert inner >= 0, "inner must be >= 0"
    assert 2 * inner < V and 2 * inner < U, "inner too large for grid size"

    v0, v1 = inner, V - inner
    u0, u1 = inner, U - inner
    cropped = images[v0:v1, u0:u1]
    Vc, Uc = cropped.shape[:2]

    inner_tag = f"{inner:02d}"
    out_dir = out_root / f"inner_{inner_tag}" / "images"
    ensure_empty(out_dir)

    for jj in range(Vc):
        for ii in range(Uc):
            r = jj + v0
            c = ii + u0
            fname = f"{r:02d}_{c:02d}.png"
            img = cropped[jj, ii]
            fp = out_dir / fname
            iio.imwrite(fp.as_posix(), img_as_ubyte(img))


def write_rig_config(
    cams: List[RigCamera],
    rig_config_path: Path,
    spacing: Tuple[float, float] = (1.0, 1.0),
):
    sx, sy = spacing
    cams_sorted = sorted(cams, key=lambda c: (not c.is_ref, c.row, c.col))
    cameras_json = []
    for cam in cams_sorted:
        entry = {
            "image_prefix": cam.prefix.as_posix(),
        }
        if cam.is_ref:
            entry["ref_sensor"] = True
        # else:
        #     entry["cam_from_rig_rotation"] = [cam.qw, cam.qx, cam.qy, cam.qz]
        #     entry["cam_from_rig_translation"] = [cam.tx * sx, cam.ty * sy, cam.tz]
        cameras_json.append(entry)
    rig_json = [{"cameras": cameras_json}]
    rig_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rig_config_path, "w") as f:
        json.dump(rig_json, f, indent=4)


def run_colmap(
    out_dir: Path, ground_truth: Path, fix_rig: bool, focal_px: float = None
):
    images = out_dir / "images"
    database = out_dir / "database.db"
    sparse = out_dir / "sparse"
    ensure_empty(sparse)
    if database.exists():
        database.unlink()

    # sample_img = next(images.rglob(f"*{'.png'}"))
    # sample = iio.imread(sample_img.as_posix())
    # H, W = sample.shape[:2]
    # cx, cy = W / 2.0, H / 2.0

    extract_cmd = [
        "colmap",
        "feature_extractor",
        "--database_path",
        str(database),
        "--image_path",
        str(images),
        "--ImageReader.single_camera_per_folder",
        "1",
        "--SiftExtraction.estimate_affine_shape",
        "1",
        "--SiftExtraction.domain_size_pooling",
        "1",
        "--ImageReader.camera_model",
        "PINHOLE",
        # "--ImageReader.camera_params",
        # f"{focal_px},{focal_px},{cx},{cy}",
    ]
    run(extract_cmd)

    rig_cfg = out_dir / "rig_config.json"
    run(
        [
            "colmap",
            "rig_configurator",
            "--database_path",
            str(database),
            "--rig_config_path",
            str(rig_cfg),
            "--input_path",
            str(ground_truth),
        ]
    )

    match_cmd = [
        "colmap",
        "exhaustive_matcher",
        "--database_path",
        str(database),
        "--SiftMatching.guided_matching",
        "1",
    ]
    run(match_cmd)

    mapper_cmd = [
        "colmap",
        "mapper",
        "--database_path",
        str(database),
        "--image_path",
        str(images),
        "--output_path",
        str(sparse),
        "--Mapper.ba_refine_sensor_from_rig",
        "0" if fix_rig else "1",
        "--Mapper.tri_ignore_two_view_tracks",
        "0",
        "--Mapper.init_min_tri_angle",
        "0.01",
        "--Mapper.local_ba_min_tri_angle",
        "0.01",
        "--Mapper.filter_min_tri_angle",
        "0.01",
        "--Mapper.tri_min_angle",
        "0.01",
        "--Mapper.max_focal_length_ratio",
        "100",
        "--Mapper.init_min_num_inliers",
        "20",
    ]
    run(mapper_cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Refine a COLMAP reconstruction from a VGGT reconstruction"
    )
    parser.add_argument(
        "--lfr", type=Path, required=True, help="Path to Lytro Illum .lfr"
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Output directory for COLMAP results",
    )
    parser.add_argument(
        "--ground_truth",
        type=Path,
        required=True,
        help="Path to the ground truth COLMAP reconstruction",
    )
    parser.add_argument(
        "--inner",
        type=int,
        default=1,
        help="Remover outer N images (N must be odd)",
    )
    args = parser.parse_args()

    # ensure_empty(args.out_dir)
    images_path = args.out_dir / "images"
    images_path.mkdir(parents=True, exist_ok=True)
    images = load_viewpoints(args.lfr)
    # pixel_size_m = 0.000001399999950081109936235848
    # pixel_size_mm = pixel_size_m * 1000
    # focal_mm = 11.4
    # focal_mm = 37.6
    # focal_px = focal_mm / pixel_size_mm
    # print(pitch)
    # print(focal_px)

    inner = None if args.inner in (None, 0) else args.inner
    # save_with_inner(images, args.out_dir, inner=inner)
    cams = save_as_rig(images, images_path, inner=inner)

    rig_config = args.out_dir / "rig_config.json"
    # write_rig_config(cams, rig_config, spacing=(bx, by))
    write_rig_config(cams, rig_config)

    run_colmap(args.out_dir, args.ground_truth, fix_rig=True)


if __name__ == "__main__":
    # img = iio.immeta(
    #     "/home/wl757/multiplexed-pixels/plenopticam/data/gradient_rose_close.lfr"
    # )
    # print(img)
    main()
