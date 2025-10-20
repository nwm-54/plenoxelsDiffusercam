from __future__ import annotations

import copy
import json
import os
import subprocess
import tempfile
import time
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData, PlyElement

from arguments import PipelineParams
from gaussian_renderer import render
from scene.cameras import Camera
from scene.gaussian_model import BasicPointCloud, GaussianModel
from utils.general_utils import get_dataset_name
from utils.graphics_utils import fov2focal, getWorld2View2

if TYPE_CHECKING:
    from arguments import ModelParams
    from scene.scene_utils import CameraInfo, SceneInfo

PLYS_ROOT = Path("/home/wl757/multiplexed-pixels/plenoxels/plys")


def get_pretrained_splat_path(args: ModelParams) -> Optional[str]:
    if args.pretrained_ply and os.path.exists(args.pretrained_ply):
        return args.pretrained_ply

    candidate = PLYS_ROOT / f"{get_dataset_name(args.source_path)}.ply"
    if candidate.exists():
        return str(candidate)

    return None


def load_pretrained_splat(path: str, sh_degree: int = 3) -> GaussianModel:
    if path is None:
        raise ValueError("Expected a valid pretrained ply path, received None")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pretrained ply file not found at {path}")
    gs = GaussianModel(sh_degree)
    gs.load_ply(path)
    print(f"Loaded pretrained ply: {path}")
    return gs


def render_splat(
    args: ModelParams, gs: GaussianModel, scene_info: SceneInfo
) -> SceneInfo:
    dummy_args = ArgumentParser()
    pp = PipelineParams(dummy_args).extract(dummy_args.parse_args([]))

    bg = torch.tensor([0.0, 0.0, 0.0], device=args.data_device, dtype=torch.float32)
    out_dir = os.path.join(args.model_path, "input_views")
    os.makedirs(out_dir, exist_ok=True)

    new_train_cameras = copy.deepcopy(scene_info.train_cameras)
    for view_index, cam_info_list in new_train_cameras.items():
        updated_cam_info_list = []
        for cam_info in cam_info_list:
            cam_info: CameraInfo
            tmp_camera = Camera(
                colmap_id=cam_info.uid,
                R=cam_info.R,
                T=cam_info.T,
                FoVx=cam_info.FovX,
                FoVy=cam_info.FovY,
                image=torch.zeros(
                    (3, cam_info.height, cam_info.width),
                    device=args.data_device,
                    dtype=torch.float32,
                ),
                gt_alpha_mask=None,
                mask=cam_info.mask,
                image_name=cam_info.image_name,
                uid=0,
                data_device=args.data_device,
            )

            new_image = render_splat_from_camera(tmp_camera, gs, pp, bg)

            png_name = f"{cam_info.image_name}.png"
            png_path = os.path.join(out_dir, png_name)
            new_image.save(png_path, format="PNG")

            updated_cam_info = cam_info._replace(image=new_image, image_path=png_path)
            updated_cam_info_list.append(updated_cam_info)
        new_train_cameras[view_index] = updated_cam_info_list

    return scene_info._replace(train_cameras=new_train_cameras)


def _camera_info_to_opengl_transform(cam_info: "CameraInfo") -> np.ndarray:
    w2c = getWorld2View2(cam_info.R, cam_info.T).astype(np.float64)
    c2w = np.linalg.inv(w2c)
    c2w[:3, 1:3] *= -1.0
    return c2w


def _camera_info_to_frame_entry(
    cam_info: "CameraInfo", file_stub: str, split: str
) -> Dict[str, object]:
    width, height = cam_info.image.size
    transform = _camera_info_to_opengl_transform(cam_info)
    fl_x = float(fov2focal(cam_info.FovX, width))
    fl_y = float(fov2focal(cam_info.FovY, height))

    entry: Dict[str, object] = {
        "file_path": file_stub,
        "transform_matrix": transform.tolist(),
        "fl_x": fl_x,
        "fl_y": fl_y,
        "cx": width / 2.0,
        "cy": height / 2.0,
        "w": width,
        "h": height,
        "split": split,
    }

    return entry


def build_transforms_data(
    scene_info: "SceneInfo", include_test: bool = False
) -> Tuple[Optional[Dict[str, object]], Dict[Tuple[int, int], Dict[str, object]]]:
    frames: List[Dict[str, object]] = []
    train_lookup: Dict[Tuple[int, int], Dict[str, object]] = {}

    first_train_entry: Optional[Dict[str, object]] = None
    first_train_cam: Optional["CameraInfo"] = None

    camera_name_map: Dict[int, str] = {}
    frame_counters: Dict[str, int] = defaultdict(int)

    def _camera_name_for(cam_uid: int) -> str:
        if cam_uid not in camera_name_map:
            camera_name_map[cam_uid] = f"camera_{len(camera_name_map) + 1:02d}"
        return camera_name_map[cam_uid]

    def _file_stub(cam_uid: int) -> Tuple[str, int]:
        cam_name = _camera_name_for(cam_uid)
        frame_idx = frame_counters[cam_name]
        frame_counters[cam_name] += 1
        return f"./{cam_name}/r_{frame_idx:03d}", frame_idx

    for view_idx, cam_list in sorted(scene_info.train_cameras.items()):
        for cam_idx, cam_info in enumerate(cam_list):
            file_stub, _ = _file_stub(cam_info.uid)
            entry = _camera_info_to_frame_entry(cam_info, file_stub, split="train")
            frames.append(entry)

            if first_train_entry is None:
                first_train_entry = entry
                first_train_cam = cam_info

            train_lookup[(view_idx, cam_idx)] = {
                "file_stub": file_stub,
                "image_name": cam_info.image_name
                or f"view_{view_idx:03d}_cam_{cam_idx:02d}",
                "cam_info": cam_info,
            }

    if include_test and scene_info.test_cameras:
        for cam_info in scene_info.test_cameras:
            file_stub, _ = _file_stub(cam_info.uid)
            frames.append(
                _camera_info_to_frame_entry(cam_info, file_stub, split="test")
            )

    if include_test and scene_info.full_test_cameras:
        for cam_info in scene_info.full_test_cameras:
            file_stub, _ = _file_stub(cam_info.uid)
            frames.append(
                _camera_info_to_frame_entry(cam_info, file_stub, split="full")
            )

    if not frames or first_train_entry is None or first_train_cam is None:
        return None, train_lookup

    transforms = {
        "camera_angle_x": float(first_train_cam.FovX),
        "camera_angle_y": float(first_train_cam.FovY),
        "fl_x": first_train_entry["fl_x"],
        "fl_y": first_train_entry["fl_y"],
        "cx": first_train_entry["w"] / 2.0,
        "cy": first_train_entry["h"] / 2.0,
        "w": first_train_entry["w"],
        "h": first_train_entry["h"],
        "frames": frames,
    }

    return transforms, train_lookup


def write_camera_json(
    scene_info: "SceneInfo",
    output_path: os.PathLike | str,
    include_test: bool = False,
    object_center: Optional[np.ndarray] = None,
) -> Tuple[Optional[Dict[str, object]], Dict[Tuple[int, int], Dict[str, object]]]:
    transforms_data, frame_lookup = build_transforms_data(
        scene_info, include_test=include_test
    )
    if transforms_data is None:
        return None, frame_lookup

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if getattr(scene_info, "nerf_normalization", None):
        normalization = scene_info.nerf_normalization
        if normalization:
            transforms_data = copy.deepcopy(transforms_data)
            transforms_data["normalization"] = {
                "translate": normalization.get("translate", []).tolist()
                if hasattr(normalization.get("translate"), "tolist")
                else normalization.get("translate"),
                "radius": float(normalization.get("radius"))
                if normalization.get("radius") is not None
                else None,
            }
    if object_center is not None:
        transforms_data = copy.deepcopy(transforms_data)
        transforms_data["object_center"] = (
            object_center.tolist()
            if hasattr(object_center, "tolist")
            else list(object_center)
        )
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(transforms_data, file, indent=2)

    return transforms_data, frame_lookup


def render_with_blender(
    args: "ModelParams",
    scene_info: "SceneInfo",
    dataset_name: str,
    object_center: Optional[np.ndarray] = None,
) -> "SceneInfo":
    if not scene_info.train_cameras:
        return scene_info

    transforms_path = Path(args.model_path) / "transforms.json"
    transforms_data, frame_lookup = write_camera_json(
        scene_info,
        transforms_path,
        include_test=False,
        object_center=object_center,
    )
    if transforms_data is None:
        return scene_info

    project_root = Path(__file__).resolve().parents[1]
    render_script = project_root / "blender" / "render_blender.py"
    blend_file = project_root / "blender" / f"{dataset_name}.blend"

    if not blend_file.exists():
        raise FileNotFoundError(
            f"Expected Blender scene for '{dataset_name}' at {blend_file}"
        )

    input_views_dir = Path(args.model_path) / "input_views"
    input_views_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="blender_render_") as tmp_root:
        tmp_root_path = Path(tmp_root)
        total_train_frames = max(len(frame_lookup), 1)
        cmd = [
            "conda",
            "run",
            "-n",
            "blender",
            "blender",
            "-b",
            str(blend_file),
            "-P",
            str(render_script),
            "--",
            "--transforms-json",
            str(transforms_path),
            "--results",
            str(tmp_root_path),
            "--views",
            str(max(total_train_frames, 1)),
        ]

        cmd.append("--disable-animation")

        render_start = time.perf_counter()

        process = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
        )

        render_elapsed = time.perf_counter() - render_start

        if process.returncode != 0:
            combined_output = process.stderr.strip() or process.stdout.strip()
            raise RuntimeError(
                "Blender rendering failed"
                + (f": {combined_output}" if combined_output else "")
            )

        new_train_cameras: Dict[int, List["CameraInfo"]] = {}
        rendered_count = 0
        for view_idx, cam_list in scene_info.train_cameras.items():
            updated_list: List["CameraInfo"] = []
            for cam_idx, cam_info in enumerate(cam_list):
                meta = frame_lookup.get((view_idx, cam_idx))
                if meta is None:
                    updated_list.append(cam_info)
                    continue

                file_stub = meta["file_stub"]
                image_name = meta["image_name"] or cam_info.image_name or f"view_{view_idx:03d}_cam_{cam_idx:02d}"

                rel_path = file_stub[2:] if file_stub.startswith("./") else file_stub
                rendered_path = tmp_root_path / Path(rel_path)
                rendered_file = rendered_path.with_suffix(".png")

                if not rendered_file.exists():
                    print(
                        f"Warning: Blender output missing for '{rel_path}' at {rendered_file}; keeping original image."
                    )
                    updated_list.append(cam_info)
                    continue

                with Image.open(rendered_file) as pil_image:
                    image_rgba = pil_image.convert("RGBA")

                black_bg = Image.new("RGBA", image_rgba.size, (0, 0, 0, 255))
                composited = Image.alpha_composite(black_bg, image_rgba)
                image_rgb = composited.convert("RGB")

                final_path = input_views_dir / f"{image_name}.png"
                image_rgb.save(final_path, format="PNG")

                rendered_count += 1
                updated_list.append(
                    cam_info._replace(image=image_rgb, image_path=str(final_path))
                )

            new_train_cameras[view_idx] = updated_list

    print(
        f"Blender render completed: {rendered_count} train frames in {render_elapsed:.1f}s; outputs stored in {input_views_dir}"
    )

    return scene_info._replace(train_cameras=new_train_cameras)

def render_splat_from_camera(
    camera: Camera, gs: GaussianModel, pp: PipelineParams, bg: torch.Tensor
) -> Image:
    with torch.no_grad():
        rendering = render(camera, gs, pp, bg)["render"]
    new_image_data = (rendering.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255)
    new_image = Image.fromarray(new_image_data.astype(np.uint8), "RGB")
    return new_image


def camera_forward(camera: Camera) -> np.ndarray:
    z_cam = np.array([0, 0, 1])
    forward = camera.R @ z_cam
    return forward / np.linalg.norm(forward)


def camera_center(camera: Camera) -> np.ndarray:
    R = (
        camera.R.detach().cpu().numpy()
        if isinstance(camera.R, torch.Tensor)
        else camera.R
    )
    T = (
        camera.T.detach().cpu().numpy()
        if isinstance(camera.T, torch.Tensor)
        else camera.T
    )
    return -(R @ T)


def find_max_min_dispersion_subset(
    points: np.ndarray, k: int, initial_point_index: Optional[int]
) -> np.ndarray:
    """Finds a near-optimal subset of k points that maximizes the minimum distance."""
    if k < 1:
        return np.array([])
    n_points = len(points)
    if k >= n_points:
        return np.arange(n_points)

    # Get a good initial solution using Farthest Point Sampling
    selected_indices = np.zeros(k, dtype=int)
    rng = np.random.default_rng(seed=42)
    selected_indices[0] = (
        initial_point_index
        if initial_point_index is not None
        else rng.integers(n_points)
    )

    dists = np.linalg.norm(points - points[selected_indices[0]], axis=1)
    for i in range(1, k):
        farthest_idx = np.argmax(dists)
        selected_indices[i] = farthest_idx
        new_dists = np.linalg.norm(points - points[farthest_idx], axis=1)
        dists = np.minimum(dists, new_dists)

    # Iteratively improve the solution with a local search (exchange heuristic)
    for _ in range(10):
        current_subset = set(selected_indices)

        sub_dist_matrix = np.linalg.norm(
            points[selected_indices, None] - points[None, selected_indices], axis=-1
        )
        np.fill_diagonal(sub_dist_matrix, np.inf)
        min_dist = sub_dist_matrix.min()

        made_swap = False
        for i in range(k):
            for p_out_idx in range(n_points):
                if p_out_idx in current_subset:
                    continue

                temp_indices = np.copy(selected_indices)
                temp_indices[i] = p_out_idx
                new_sub_dist_matrix = np.linalg.norm(
                    points[temp_indices, None] - points[None, temp_indices], axis=-1
                )
                np.fill_diagonal(new_sub_dist_matrix, np.inf)
                new_min_dist = new_sub_dist_matrix.min()

                if new_min_dist > min_dist:
                    selected_indices = temp_indices
                    min_dist = new_min_dist
                    made_swap = True
                    break
            if made_swap:
                break
        if not made_swap:
            break

    return selected_indices


def fetchPly(path) -> BasicPointCloud:
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
    normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)
