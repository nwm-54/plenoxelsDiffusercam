from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go

from utils.graphics_utils import getWorld2View2

if TYPE_CHECKING:
    from scene.scene_utils import CameraInfo, SceneInfo


def get_camera_frustum_vertices(
    c2w: np.ndarray,
    fov_x: float,
    fov_y: Optional[float] = None,
    scale: float = 0.3,
    target_point: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return the 5 vertices of a camera frustum pyramid oriented toward the scene."""

    cam_center = c2w[:3, 3]
    rotation = c2w[:3, :3]
    right = rotation[:, 0]
    up = rotation[:, 1]
    forward = -rotation[:, 2]

    if target_point is not None:
        to_target = np.asarray(target_point, dtype=np.float32) - cam_center
        norm = np.linalg.norm(to_target)
        if norm > 1e-6:
            to_target /= norm
            if np.dot(forward, to_target) < 0:
                forward = -forward
                up = -up

    half_w = scale * np.tan(fov_x / 2)
    fy = fov_y if fov_y is not None else fov_x
    half_h = scale * np.tan(fy / 2)

    base_center = cam_center + forward * scale
    corners = np.stack(
        [
            base_center + (-right * half_w) + (up * half_h),
            base_center + (right * half_w) + (up * half_h),
            base_center + (right * half_w) + (-up * half_h),
            base_center + (-right * half_w) + (-up * half_h),
        ],
        axis=0,
    )
    vertices = np.vstack([cam_center, corners]).astype(np.float32)
    return vertices


def create_frustum_mesh_data(
    frustum_vertices_list: Sequence[np.ndarray],
) -> go.Mesh3d:
    """Flatten multiple frusta into a single Plotly Mesh3d payload."""

    all_x, all_y, all_z, all_i, all_j, all_k = [], [], [], [], [], []
    offset = 0
    for vertices in frustum_vertices_list:
        if vertices is None or len(vertices) != 5:
            continue
        all_x.extend(vertices[:, 0])
        all_y.extend(vertices[:, 1])
        all_z.extend(vertices[:, 2])
        faces = ((0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1), (1, 2, 4), (2, 3, 4))
        for face in faces:
            all_i.append(offset + face[0])
            all_j.append(offset + face[1])
            all_k.append(offset + face[2])
        offset += 5

    return go.Mesh3d(x=all_x, y=all_y, z=all_z, i=all_i, j=all_j, k=all_k)


def _camera_centers_and_frustums(
    cameras: Sequence["CameraInfo"],
    scale: float,
    target_point: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    centers: List[np.ndarray] = []
    frustums: List[np.ndarray] = []
    for cam in cameras:
        w2c = getWorld2View2(cam.R, cam.T)
        c2w = np.linalg.inv(w2c)
        centers.append(c2w[:3, 3])
        frustums.append(
            get_camera_frustum_vertices(
                c2w,
                cam.FovX,
                cam.FovY,
                scale=scale,
                target_point=target_point,
            )
        )
    return (np.asarray(centers) if centers else np.zeros((0, 3))), frustums


def _sample_points(points: np.ndarray, max_points: int = 20_000) -> np.ndarray:
    if len(points) <= max_points:
        return points
    stride = max(1, len(points) // max_points)
    return points[::stride]


def _add_camera_group(
    fig: go.Figure,
    centers: np.ndarray,
    frustums: Sequence[np.ndarray],
    color: str,
    opacity: float,
    name: str,
    center_size: float = 2.0,
    symbol: str = "circle",
) -> None:
    if len(centers) == 0:
        return
    mesh = create_frustum_mesh_data(frustums)
    fig.add_trace(
        go.Mesh3d(
            x=mesh.x,
            y=mesh.y,
            z=mesh.z,
            i=mesh.i,
            j=mesh.j,
            k=mesh.k,
            color=color,
            opacity=opacity,
            name=f"{name} Frustums",
            showscale=False,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=centers[:, 0],
            y=centers[:, 1],
            z=centers[:, 2],
            mode="markers",
            name=f"{name} Cameras",
            marker=dict(size=center_size, color=color, symbol=symbol),
        )
    )


def _is_gaussian_model(candidate: object) -> bool:
    return hasattr(candidate, "get_xyz") and callable(getattr(candidate, "get_xyz"))


def _is_basic_point_cloud(candidate: object) -> bool:
    return hasattr(candidate, "points") and hasattr(candidate, "colors")


def save_camera_visualization(
    scene_info: "SceneInfo",
    output_dir: str,
    filename: str,
    title: Optional[str] = None,
    highlighted_view_indices: Optional[Sequence[int]] = None,
    camera_scale: float = 0.25,
    point_cloud: Optional[object] = None,
) -> Optional[str]:
    """Create a Plotly visualization for the current cameras and point cloud."""

    os.makedirs(output_dir, exist_ok=True)
    fig = go.Figure()

    pc_data: Optional[object] = point_cloud
    if pc_data is None:
        pc_data = scene_info.point_cloud

    pc_center: Optional[np.ndarray] = None
    if pc_data is not None and _is_gaussian_model(pc_data):
        pts = pc_data.get_xyz.detach().cpu().numpy()  # type: ignore[attr-defined]
        pc_center = np.mean(pts, axis=0)
        sampled_pts = _sample_points(pts)
        fig.add_trace(
            go.Scatter3d(
                x=sampled_pts[:, 0],
                y=sampled_pts[:, 1],
                z=sampled_pts[:, 2],
                mode="markers",
                name="Model",
                marker=dict(size=1.5, color="#b0b0b0", opacity=0.6),
            )
        )
    elif pc_data is not None and _is_basic_point_cloud(pc_data):
        pts_src = getattr(pc_data, "points")  # type: ignore[attr-defined]
        if pts_src is not None and len(pts_src) > 0:
            pts = np.asarray(pts_src)
            pc_center = np.mean(pts, axis=0)
            cols = np.asarray(getattr(pc_data, "colors"))  # type: ignore[attr-defined]
            sampled_pts = _sample_points(pts)
            if len(cols) == len(pts):
                stride = max(1, len(pts) // max(1, len(sampled_pts)))
                sampled_cols = cols[::stride][: len(sampled_pts)]
                marker_colors = [
                    f"rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})"
                    for r, g, b in sampled_cols
                ]
            else:
                marker_colors = "#888888"
            fig.add_trace(
                go.Scatter3d(
                    x=sampled_pts[:, 0],
                    y=sampled_pts[:, 1],
                    z=sampled_pts[:, 2],
                    mode="markers",
                    name="Model",
                    marker=dict(size=1.5, color=marker_colors, opacity=0.6),
                )
            )
    elif scene_info.nerf_normalization and "translate" in scene_info.nerf_normalization:
        translate = np.asarray(
            scene_info.nerf_normalization["translate"], dtype=np.float32
        )
        if translate.size == 3:
            pc_center = -translate

    train_cameras = [
        cam
        for cam_list in (scene_info.train_cameras or {}).values()
        for cam in cam_list
    ]
    train_centers, train_frustums = _camera_centers_and_frustums(
        train_cameras, scale=camera_scale, target_point=pc_center
    )

    test_cameras = scene_info.test_cameras or []
    test_centers, test_frustums = _camera_centers_and_frustums(
        test_cameras, scale=camera_scale, target_point=pc_center
    )

    full_test = scene_info.full_test_cameras or []
    filtered_full_test = [
        cam
        for cam in full_test
        if cam.uid not in {c.uid for c in train_cameras}
        and cam.uid not in {c.uid for c in test_cameras}
    ]
    full_centers, full_frustums = _camera_centers_and_frustums(
        filtered_full_test, scale=camera_scale, target_point=pc_center
    )

    _add_camera_group(fig, train_centers, train_frustums, "#e74c3c", 0.35, "Train")
    _add_camera_group(fig, test_centers, test_frustums, "#2ecc71", 0.45, "Test")
    _add_camera_group(fig, full_centers, full_frustums, "#95a5a6", 0.15, "Other")

    if highlighted_view_indices:
        highlighted = []
        for view_idx in highlighted_view_indices:
            highlighted.extend(scene_info.train_cameras.get(view_idx, []))
        highlight_centers, _ = _camera_centers_and_frustums(
            highlighted,
            scale=camera_scale * 1.05,
            target_point=pc_center,
        )
        if len(highlight_centers) > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=highlight_centers[:, 0],
                    y=highlight_centers[:, 1],
                    z=highlight_centers[:, 2],
                    mode="markers",
                    name="Highlighted",
                    marker=dict(size=6, color="#f1c40f", symbol="diamond"),
                )
            )

    axis_style = dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        showbackground=False,
        visible=False,
    )
    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis=axis_style,
            yaxis=axis_style,
            zaxis=axis_style,
        ),
        title=title,
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    output_path = os.path.join(output_dir, filename)
    fig.write_html(output_path)
    return output_path
