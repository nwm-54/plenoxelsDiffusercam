from __future__ import annotations

import copy
import os
import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from arguments import PipelineParams
from gaussian_renderer import render
from PIL import Image
from scene.cameras import Camera
from utils.general_utils import get_dataset_name
from scene.gaussian_model import GaussianModel

if TYPE_CHECKING:
    from arguments import ModelParams
    from scene.dataset_readers_multiviews import CameraInfo, SceneInfo

PLYS_ROOT = Path("/home/wl757/multiplexed-pixels/plenoxels/plys")

def load_pretrained_ply(args: ModelParams) -> Optional[GaussianModel]:
    ply_path: Optional[str] = None
    if args.pretrained_ply and os.path.exists(args.pretrained_ply):
        ply_path = args.pretrained_ply
    else:
        ply_path = PLYS_ROOT / f"{get_dataset_name(args)}.ply"
    
    if not os.path.exists(ply_path):
        warnings.warn(f"Pretrained ply file not found at {ply_path}.")
        return None
    gs = GaussianModel(sh_degree=args.sh_degree)
    gs.load_ply(ply_path)
    print(f"Loaded pretrained ply: {ply_path}")
    return gs

def render_ply(args: ModelParams, gs: GaussianModel, scene_info: SceneInfo) -> SceneInfo:
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
                colmap_id = cam_info.uid,
                R = cam_info.R,
                T = cam_info.T,
                FoVx = cam_info.FovX,
                FoVy = cam_info.FovY,
                image = torch.zeros((3, cam_info.height, cam_info.width), device=args.data_device, dtype=torch.float32),
                gt_alpha_mask=None,
                mask=cam_info.mask,
                image_name=cam_info.image_name,
                uid=0,
                data_device=args.data_device
            )

            with torch.no_grad():
                rendering = render(tmp_camera, gs, pp, bg)['render']
            
            new_image_data = (rendering.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255)
            new_image = Image.fromarray(new_image_data.astype(np.uint8), "RGB")

            png_name = f"{cam_info.image_name}.png"
            png_path = os.path.join(out_dir, png_name)
            new_image.save(png_path, format="PNG")

            updated_cam_info = cam_info._replace(image=new_image, image_path=png_path)
            updated_cam_info_list.append(updated_cam_info)
        new_train_cameras[view_index] = updated_cam_info_list
    
    return scene_info._replace(train_cameras=new_train_cameras)

def camera_forward(camera: Camera) -> np.ndarray:
        z_cam = np.array([0, 0, 1])
        forward = camera.R @ z_cam
        return forward / np.linalg.norm(forward)