from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from PIL import Image

from scene.gaussian_model import BasicPointCloud
from utils.graphics_utils import focal2fov, fov2focal, getWorld2View2
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
    ply_path: Optional[str]


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


def camera_center_world(cam: CameraInfo) -> np.ndarray:
    """Return the camera center expressed in world coordinates."""
    return -cam.R @ cam.T


def _camera_to_world_matrix(cam: CameraInfo) -> np.ndarray:
    return np.linalg.inv(getWorld2View2(cam.R, cam.T))


def _camera_from_world_matrix(c2w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    new_w2c = np.linalg.inv(c2w)
    new_R = new_w2c[:3, :3].T
    new_T = new_w2c[:3, 3]
    return new_R, new_T


def camera_with_center(cam: CameraInfo, center_world: np.ndarray) -> CameraInfo:
    """Return a camera whose center is replaced with the provided world position."""
    c2w = _camera_to_world_matrix(cam)
    new_c2w = c2w.copy()
    new_c2w[:3, 3] = np.asarray(center_world, dtype=np.float32)
    new_R, new_T = _camera_from_world_matrix(new_c2w)
    return cam._replace(R=new_R, T=new_T)


def _offset_camera(cam: CameraInfo, offset_xyz: np.ndarray) -> CameraInfo:
    c2w = _camera_to_world_matrix(cam)
    world_offset = c2w[:3, :3] @ np.asarray(offset_xyz, dtype=np.float32)
    new_c2w = c2w.copy()
    new_c2w[:3, 3] += world_offset
    new_R, new_T = _camera_from_world_matrix(new_c2w)
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


def solve_offset_for_angle(
    base_vec: np.ndarray,
    axis: np.ndarray,
    angle_deg: float,
    *,
    orth_axis: Optional[np.ndarray] = None,
    orth_ratio: float = 0.0,
) -> float:
    """Return the symmetric offset that achieves the requested coverage angle.

    When ``orth_axis``/``orth_ratio`` are provided, both extreme cameras share an
    additional shift of ``orth_ratio * offset`` along ``orth_axis``.
    """

    angle = float(angle_deg)
    if angle <= 0.0:
        return 0.0
    if angle >= 180.0:
        return float("inf")

    base_vec = np.asarray(base_vec, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)

    base_norm = np.linalg.norm(base_vec)
    axis_norm = np.linalg.norm(axis)
    if base_norm < 1e-12 or axis_norm < 1e-12:
        return 0.0

    axis_unit = axis / axis_norm
    ratio = float(orth_ratio) if orth_axis is not None else 0.0

    if orth_axis is None or abs(ratio) < 1e-8:
        cos_theta = np.cos(np.radians(angle))

        if abs(cos_theta - 1.0) < 1e-12:
            return 0.0
        if abs(cos_theta + 1.0) < 1e-12:
            return float("inf")

        q = float(np.dot(base_vec, axis_unit))
        S = base_norm * base_norm
        c2 = cos_theta * cos_theta

        a = c2 - 1.0
        b = 2.0 * (S * (c2 + 1.0) - 2.0 * c2 * q * q)
        c = (c2 - 1.0) * S * S

        if abs(a) < 1e-12:
            if abs(b) < 1e-12:
                return 0.0
            x = -c / b
            return float(np.sqrt(max(0.0, x))) if x >= 0.0 else 0.0

        discriminant = b * b - 4.0 * a * c
        if discriminant < -1e-10:
            raise ValueError("No real solution for requested angle.")
        discriminant = max(discriminant, 0.0)
        sqrt_disc = np.sqrt(discriminant)
        x1 = (-b + sqrt_disc) / (2.0 * a)
        x2 = (-b - sqrt_disc) / (2.0 * a)

        valid_x = [x for x in (x1, x2) if x >= 0.0]
        if not valid_x:
            raise ValueError("No positive offset solution for requested angle.")
        offset_sq = max(min(valid_x), 0.0)
        return float(np.sqrt(offset_sq))

    orth = np.asarray(orth_axis, dtype=np.float64)
    orth_norm = np.linalg.norm(orth)
    if orth_norm < 1e-12:
        return solve_offset_for_angle(base_vec, axis_unit, angle)

    orth_unit = orth / orth_norm
    orth_unit = orth_unit - np.dot(orth_unit, axis_unit) * axis_unit
    orth_norm_adj = np.linalg.norm(orth_unit)
    if orth_norm_adj < 1e-8:
        return solve_offset_for_angle(base_vec, axis_unit, angle)
    orth_unit /= orth_norm_adj

    b_vec = base_vec
    q = float(np.dot(b_vec, axis_unit))
    p = float(np.dot(b_vec, orth_unit))
    a_norm_sq = float(np.dot(b_vec, b_vec))

    cos_theta = np.cos(np.radians(angle))
    cos_sq = float(cos_theta * cos_theta)
    rho = ratio

    a0 = a_norm_sq
    b_coef = -2.0 * p * rho
    gamma = rho * rho + 1.0
    gamma_prime = rho * rho - 1.0

    q0 = a0 * a0
    q1 = 2.0 * a0 * b_coef
    q2 = b_coef * b_coef + 2.0 * a0 * gamma
    q3 = 2.0 * b_coef * gamma
    q4 = gamma * gamma

    q2 -= 4.0 * q * q

    poly_main = np.array([q0, q1, q2, q3, q4], dtype=np.float64) * cos_sq

    qp0 = a0 * a0
    qp1 = 2.0 * a0 * b_coef
    qp2 = b_coef * b_coef + 2.0 * a0 * gamma_prime
    qp3 = 2.0 * b_coef * gamma_prime
    qp4 = gamma_prime * gamma_prime

    poly_shifted = np.array([qp0, qp1, qp2, qp3, qp4], dtype=np.float64)

    poly = poly_main - poly_shifted
    coeffs = np.trim_zeros(poly[::-1], trim="f")
    if coeffs.size == 0:
        raise ValueError("Degenerate polynomial for coupled offset.")

    roots = np.roots(coeffs)
    tol = 1e-6
    real_roots = [root.real for root in roots if abs(root.imag) < tol and root.real >= 0.0]
    if not real_roots:
        raise ValueError("No positive real solution for coupled offset.")
    offset = min(real_roots)
    return float(offset)


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
) -> Tuple[BasicPointCloud, Optional[str]]:
    print(f"Generating random spherical point cloud with {num_pts} points")
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

    return pcd, None


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
