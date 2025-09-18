from __future__ import annotations

import copy
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData, PlyElement

from arguments import PipelineParams
from gaussian_renderer import render
from scene.cameras import Camera
from scene.gaussian_model import BasicPointCloud, GaussianModel
from utils.general_utils import get_dataset_name

if TYPE_CHECKING:
    from arguments import ModelParams
    from scene.scene_utils import CameraInfo, SceneInfo

PLYS_ROOT = Path("/home/wl757/multiplexed-pixels/plenoxels/plys")


def resolve_pretrained_ply_path(args: ModelParams) -> Optional[str]:
    if args.pretrained_ply and os.path.exists(args.pretrained_ply):
        return args.pretrained_ply

    candidate = PLYS_ROOT / f"{get_dataset_name(args.source_path)}.ply"
    if candidate.exists():
        return str(candidate)

    return None


def load_pretrained_ply(path: str, sh_degree: int = 3) -> GaussianModel:
    if path is None:
        raise ValueError("Expected a valid pretrained ply path, received None")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pretrained ply file not found at {path}")
    gs = GaussianModel(sh_degree)
    gs.load_ply(path)
    print(f"Loaded pretrained ply: {path}")
    return gs


def render_ply(
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

            new_image = render_ply_from_camera(tmp_camera, gs, pp, bg)

            png_name = f"{cam_info.image_name}.png"
            png_path = os.path.join(out_dir, png_name)
            new_image.save(png_path, format="PNG")

            updated_cam_info = cam_info._replace(image=new_image, image_path=png_path)
            updated_cam_info_list.append(updated_cam_info)
        new_train_cameras[view_index] = updated_cam_info_list

    return scene_info._replace(train_cameras=new_train_cameras)


def render_ply_from_camera(
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
