from __future__ import annotations

from typing import TYPE_CHECKING

import os
import numpy as np

from scene.cameras import Camera
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

if TYPE_CHECKING:
    from scene.scene_utils import CameraInfo

WARNED = False


def _compute_resolution(orig_w, orig_h, args_resolution, resolution_scale):
    if args_resolution in [1, 2, 4, 8]:
        scale = resolution_scale * args_resolution
        return (round(orig_w / scale), round(orig_h / scale))

    if args_resolution == -1:
        if orig_w > 1600:
            global WARNED
            if not WARNED:
                print(
                    "[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                    "If this is not desired, please explicitly specify '--resolution/-r' as 1"
                )
                WARNED = True
            global_down = orig_w / 1600
        else:
            global_down = 1
    else:
        global_down = orig_w / args_resolution

    scale = global_down * resolution_scale
    return (int(orig_w / scale), int(orig_h / scale))


def loadCam(args, id, cam_info: CameraInfo, resolution_scale):
    orig_w, orig_h = cam_info.image.size
    resolution = _compute_resolution(orig_w, orig_h, args.resolution, resolution_scale)
    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(
        colmap_id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        image=gt_image,
        gt_alpha_mask=loaded_mask,
        mask=cam_info.mask,
        #   image=gt_image, gt_alpha_mask=loaded_mask,mask=PILtoTorch(cam_info.mask.mask, resolution).cuda() if cam_info.mask is not None else None,
        image_name=cam_info.image_name,
        uid=id,
        group_id=getattr(cam_info, "groupid", cam_info.uid),
        data_device=args.data_device,
    )


def cameraList_from_camInfos(cam_infos: CameraInfo, resolution_scale, args):
    camera_list = []
    # print(len(cam_infos))

    for id, c in enumerate(cam_infos):
        # print("from loadCam caller", c.image)
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list


def _default_file_stub(camera: Camera) -> str:
    base_name = camera.image_name or f"camera_{camera.uid}"
    safe_name = base_name.replace(os.sep, "_")
    return f"./{safe_name}/r_0"


def camera_to_JSON(camera: Camera, file_path: str | None = None) -> dict:
    """Return a NeRF-compatible frame dictionary for ``camera``."""

    Rt = np.zeros((4, 4), dtype=np.float64)
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    c2w = np.linalg.inv(Rt)
    c2w[:3, 1:3] *= -1.0

    width = float(camera.image_width)
    height = float(camera.image_height)

    frame_entry = {
        "file_path": file_path or _default_file_stub(camera),
        "transform_matrix": c2w.tolist(),
        "fl_x": float(fov2focal(camera.FoVx, camera.image_width)),
        "fl_y": float(fov2focal(camera.FoVy, camera.image_height)),
        "cx": width / 2.0,
        "cy": height / 2.0,
        "w": camera.image_width,
        "h": camera.image_height,
    }

    return frame_entry
