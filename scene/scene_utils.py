from __future__ import annotations

import os
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from PIL import Image

from scene.gaussian_model import BasicPointCloud
from utils.graphics_utils import focal2fov, fov2focal, getWorld2View2
from utils.render_utils import storePly
from utils.sh_utils import SH2RGB

WORLD_TO_M: float = 0.09904


class CameraInfo(NamedTuple):
    uid: int
    groupid: int
    R: np.ndarray
    T: np.ndarray
    FovY: float
    FovX: float
    image: Image.Image
    image_path: str
    image_name: str
    width: int
    height: int
    mask_name: str
    mask: Optional[np.ndarray]


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: Dict[int, List[CameraInfo]]
    test_cameras: List[CameraInfo]
    full_test_cameras: List[CameraInfo]
    nerf_normalization: Dict[str, np.ndarray]
    ply_path: str


def configure_world_to_m(scale_m_per_world: float) -> None:
    """This is set from `Scene.world_to_m` once the Scene is constructed."""
    global WORLD_TO_M
    WORLD_TO_M = float(scale_m_per_world)


def world_to_m(world_value: float) -> float:
    return float(world_value) * WORLD_TO_M


def m_to_world(meters: float) -> float:
    return float(meters) / WORLD_TO_M


def mm_to_world(mm: float) -> float:
    """Convert millimeters to world units using the configured scale."""
    return m_to_world(float(mm) / 1000.0)


def world_to_mm(world_value: float) -> float:
    """Convert world units to millimeters using the configured scale."""
    return world_to_m(float(world_value)) * 1000.0


class PinholeMask(NamedTuple):
    bbox: list
    mask: np.ndarray
    path: str


def _offset_camera(cam: CameraInfo, offset_xyz: np.ndarray) -> CameraInfo:
    """Applies a translation offset to the camera position."""
    c2w = np.linalg.inv(getWorld2View2(cam.R, cam.T))
    new_c2w = c2w.copy()
    world_offset = new_c2w[:3, :3] @ np.asarray(offset_xyz, dtype=np.float32)
    new_c2w[:3, 3] += world_offset
    new_w2c = np.linalg.inv(new_c2w)
    new_R = np.transpose(new_w2c[:3, :3])
    new_T = new_w2c[:3, 3]
    return cam._replace(R=new_R, T=new_T)


def _scale_camera_fov(cam: CameraInfo, scale: float) -> CameraInfo:
    fx = fov2focal(cam.FovX, cam.width)
    fy = fov2focal(cam.FovY, cam.height)
    new_fx = fx * float(scale)
    new_fy = fy * float(scale)
    new_FovX = focal2fov(new_fx, cam.width)
    new_FovY = focal2fov(new_fy, cam.height)
    return cam._replace(FovX=new_FovX, FovY=new_FovY)


def _make_shifted_scaled_cam(
    cam: CameraInfo, offset_xyz: np.ndarray, scale: float, uid: int, image_name: str
) -> CameraInfo:
    cam = _offset_camera(cam, offset_xyz)
    if abs(scale - 1.0) > 1e-8:
        cam = _scale_camera_fov(cam, scale)
    return cam._replace(uid=uid, image_name=image_name)


def getNerfppNorm(cam_info: List[CameraInfo]):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []
    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1
    translate = -center

    return {"translate": translate, "radius": radius}


def generate_random_pcd(
    path: str, num_pts: int = 100_000
) -> Tuple[BasicPointCloud, str]:
    print(f"Generating random spherical point cloud with {num_pts} points")
    ply_path = os.path.join(path, "points3d.ply")

    # Cuboid
    # We create random points inside the bounds of the synthetic Blender scenes
    xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3

    # Spherical
    # Generate random radii, uniformly distributed within [0, 1)
    # r = np.random.random(num_pts) ** (1/3) * 1.3 # Adjust distribution to account for volume
    r = np.random.random(num_pts) ** (1 / 3)
    theta = np.random.uniform(0, 2 * np.pi, num_pts)  # Azimuthal angle [0, 2π)
    phi = np.random.uniform(0, np.pi, num_pts)  # Polar angle [0, π]
    # Convert spherical coordinates to Cartesian coordinates
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    xyz = np.stack((x, y, z), axis=-1)
    shs = np.random.random((num_pts, 3)) / 255.0  # random colors
    pcd = BasicPointCloud(
        points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
    )
    storePly(ply_path, xyz, SH2RGB(shs) * 255)

    return pcd, ply_path


def read_camera(
    image_filepath: str,
    image_name: str,
    transform_matrix: np.array,
    white_background: bool,
    uid_for_image: int,
    fov_x: float,
) -> CameraInfo:
    c2w = np.array(
        transform_matrix
    )  # NeRF 'transform_matrix' is a camera-to-world transform
    c2w[
        :3, 1:3
    ] *= -1  # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
    w2c = np.linalg.inv(c2w)  # get the world-to-camera transform and set R, T
    R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
    T = w2c[:3, 3]

    image = np.array(Image.open(image_filepath).convert("RGBA")) / 255.0
    bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
    image = image[:, :, :3] * image[:, :, 3:4] + bg * (1 - image[:, :, 3:4])
    image = Image.fromarray(np.array(image * 255.0, dtype=np.byte), "RGB")
    fov_y = focal2fov(fov2focal(fov_x, image.size[0]), image.size[1])

    return CameraInfo(
        uid=uid_for_image,
        groupid=uid_for_image,
        R=R,
        T=T,
        FovY=fov_y,
        FovX=fov_x,
        image=image,
        mask=None,
        mask_name="",
        image_path=image_filepath,
        image_name=image_name,
        width=image.size[0],
        height=image.size[1],
    )
