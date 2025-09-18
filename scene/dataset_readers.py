import copy
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import plotly.graph_objects as go
from PIL import Image

from arguments import ModelParams
from arguments.multiviews_indices import MULTIVIEW_INDICES
from scene import multiplexing
from scene.colmap_loader import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
)
from scene.gaussian_model import BasicPointCloud, GaussianModel
from scene.scene_utils import (
    CameraInfo,
    PinholeMask,
    SceneInfo,
    _make_shifted_scaled_cam,
    _offset_camera,
    generate_random_pcd,
    getNerfppNorm,
    mm_to_world,
    read_camera,
    world_to_m,
)
from utils.general_utils import get_dataset_name
from utils.graphics_utils import focal2fov, fov2focal, getWorld2View2
from utils.render_utils import (
    fetchPly,
    find_max_min_dispersion_subset,
    load_pretrained_ply,
    storePly,
)
from utils.sh_utils import SH2RGB

FIRST_VIEW: Dict[str, List[int]] = MULTIVIEW_INDICES[1]
PIXEL_SIZE_MM = 0.00244


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
    forward = rotation[:, 2]

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
    cameras: Sequence[CameraInfo],
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


def save_camera_visualization(
    scene_info: SceneInfo,
    output_dir: str,
    filename: str,
    title: Optional[str] = None,
    highlighted_view_indices: Optional[Sequence[int]] = None,
    camera_scale: float = 0.25,
    ply_override_path: Optional[str] = None,
) -> Optional[str]:
    """Create a Plotly visualization for the current cameras and point cloud."""

    os.makedirs(output_dir, exist_ok=True)
    fig = go.Figure()

    point_cloud: Optional[object] = None
    if ply_override_path:
        point_cloud = load_pretrained_ply(ply_override_path)

    if point_cloud is None:
        point_cloud = scene_info.point_cloud
        if point_cloud is None and scene_info.ply_path:
            point_cloud = fetchPly(scene_info.ply_path)

    pc_center: Optional[np.ndarray] = None
    if isinstance(point_cloud, GaussianModel):
        pts = point_cloud.get_xyz.detach().cpu().numpy()
        pc_center = np.mean(pts, axis=0)
        sampled_pts = _sample_points(pts)
        fig.add_trace(
            go.Scatter3d(
                x=sampled_pts[:, 0],
                y=sampled_pts[:, 1],
                z=sampled_pts[:, 2],
                mode="markers",
                name="Input PLY",
                marker=dict(size=1.5, color="#b0b0b0", opacity=0.6),
            )
        )
    elif isinstance(point_cloud, BasicPointCloud) and len(point_cloud.points) > 0:
        pts = np.asarray(point_cloud.points)
        pc_center = np.mean(pts, axis=0)
        cols = np.asarray(point_cloud.colors)
        sampled_pts = _sample_points(pts)
        if len(cols) == len(pts):
            stride = max(1, len(pts) // len(sampled_pts))
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
                name="Input PLY",
                marker=dict(size=1.5, color=marker_colors, opacity=0.6),
            )
        )
    elif scene_info.nerf_normalization and "translate" in scene_info.nerf_normalization:
        translate = np.asarray(scene_info.nerf_normalization["translate"], dtype=np.float32)
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

    def _add_camera_group(
        centers: np.ndarray,
        frustums: Sequence[np.ndarray],
        color: str,
        opacity: float,
        name: str,
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
                name=f"{name} Centers",
                marker=dict(size=4, color=color, symbol=symbol),
            )
        )

    _add_camera_group(train_centers, train_frustums, "#e74c3c", 0.35, "Train")
    _add_camera_group(test_centers, test_frustums, "#2ecc71", 0.45, "Test")
    _add_camera_group(full_centers, full_frustums, "#95a5a6", 0.15, "Other")

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


def print_camera_metrics(scene_info: SceneInfo, obj_center: np.ndarray):
    pixel_size_mm = PIXEL_SIZE_MM

    printed = 0
    for view_idx, cam_list in scene_info.train_cameras.items():
        if not cam_list:
            continue
        cam = cam_list[0]

        fx_px = fov2focal(cam.FovX, cam.width)
        fy_px = fov2focal(cam.FovY, cam.height)
        fx_mm = fx_px * pixel_size_mm
        fy_mm = fy_px * pixel_size_mm

        cam_center_world = -cam.R @ cam.T
        if obj_center is None:
            dist_world = np.linalg.norm(cam_center_world)
        else:
            dist_world = np.linalg.norm(cam_center_world - obj_center)
        dist_m = world_to_m(dist_world)

        print(
            f"View {view_idx} ({cam.image_name}): fx={fx_mm:.3f} mm (px={fx_px:.1f}), "
            f"fy={fy_mm:.3f} mm (px={fy_px:.1f}); distance={dist_m:.3f} m"
        )

        printed += 1
        if printed >= 10:  # show up to 10 views
            break

    groups: Dict[int, List[Tuple[CameraInfo, np.ndarray]]] = defaultdict(list)
    for cam_list in scene_info.train_cameras.values():
        for cam in cam_list:
            cam_center_world = -cam.R @ cam.T
            groups[cam.groupid].append((cam, cam_center_world))

    if not groups:
        return

    group_angles_deg: Dict[int, float] = {}
    for gid, entries in sorted(groups.items(), key=lambda kv: kv[0]):
        if len(entries) < 2:
            continue

        ref_cam, _ = entries[0]
        R_ref = ref_cam.R
        group_center_world = np.mean([c for (_, c) in entries], axis=0)

        local_entries = []
        for cam_i, center_i in entries:
            delta_world = center_i - group_center_world
            delta_local = R_ref.T @ delta_world  # express in camera-local frame
            local_entries.append((cam_i, center_i, float(delta_local[0])))

        left_cam, left_center, left_x = min(local_entries, key=lambda e: e[2])
        right_cam, right_center, right_x = max(local_entries, key=lambda e: e[2])

        if np.isclose(left_x, right_x):
            continue

        v_left = left_center - obj_center
        v_right = right_center - obj_center

        n_left = np.linalg.norm(v_left)
        n_right = np.linalg.norm(v_right)
        if n_left < 1e-12 or n_right < 1e-12:
            print(f"Group {gid}: object center coincides with an extreme; angle N/A")
            continue

        cos_theta = float(
            np.clip(np.dot(v_left, v_right) / (n_left * n_right), -1.0, 1.0)
        )
        angle_deg = np.degrees(np.arccos(cos_theta))
        group_angles_deg[gid] = float(angle_deg)

        # Distance (baseline) between the two extreme cameras
        baseline_world = float(np.linalg.norm(right_center - left_center))
        baseline_m = world_to_m(baseline_world)

        print(
            f"Group {gid}: left={left_cam.image_name} "
            f"(local x={left_x:.6f}), right={right_cam.image_name} "
            f"(local x={right_x:.6f}) -> angle={float(angle_deg):.2f} deg, baseline={baseline_m:.3f} m"
        )

    avg_group_angle = (
        float(np.mean(list(group_angles_deg.values()))) if group_angles_deg else None
    )
    if avg_group_angle is not None:
        print(f"Average group angle: {avg_group_angle:.2f} deg")

    return {
        "group_angles_deg": group_angles_deg,
        "avg_group_angle_deg": avg_group_angle,
    }


def read_cameras_from_transforms(
    path: str,
    train_transforms_file: str,
    test_transforms_file: str,
    white_background: bool,
    extension: str,
) -> Tuple[Dict[int, List[CameraInfo]], List[CameraInfo]]:
    train_cameras_info, test_cameras_info = defaultdict(list), []

    with open(os.path.join(path, train_transforms_file)) as json_file:
        contents = json.load(json_file)
        fov_x = contents["camera_angle_x"]
        frames = contents["frames"]

        for frame in frames:
            image_name: str = frame["file_path"].split("/")[-1]
            parts = image_name.split("_")
            index = int(parts[1])
            image_filepath = os.path.join(path, frame["file_path"] + extension)
            train_cameras_info[index].append(
                read_camera(
                    image_filepath,
                    image_name,
                    frame["transform_matrix"],
                    white_background,
                    index,
                    fov_x,
                )
            )

    with open(os.path.join(path, test_transforms_file)) as json_file:
        contents = json.load(json_file)
        fov_x = contents["camera_angle_x"]
        frames = contents["frames"]

        for frame in frames:
            image_name: str = frame["file_path"].split("/")[-1]
            parts = image_name.split("_")
            index = int(parts[1])
            image_filepath = os.path.join(path, frame["file_path"] + extension)
            camera_info = read_camera(
                image_filepath,
                image_name,
                frame["transform_matrix"],
                white_background,
                index,
                fov_x,
            )
            test_cameras_info.append(camera_info)

    return train_cameras_info, test_cameras_info


def readNerfSyntheticInfo(
    path: str,
    white_background: bool,
    eval: bool,
    extension: str = ".png",
    n_train_images: int = 1,
    use_orbital_trajectory: bool = False,
    visualization_dir: Optional[str] = None,
    visualization_filename: Optional[str] = None,
    input_ply_path: Optional[str] = None,
) -> SceneInfo:
    print(f"Reading Nerf synthetic scene from {path}")
    train_transforms_file = (
        "transforms_train.json"
        if not use_orbital_trajectory
        else "orbital_trajectory.json"
    )
    test_transforms_file = "transforms_test.json"

    train_cameras_info, full_test_cameras_info = read_cameras_from_transforms(
        path, train_transforms_file, test_transforms_file, white_background, extension
    )

    all_train_cameras_list = sorted(
        [cam for cam_list in train_cameras_info.values() for cam in cam_list],
        key=lambda c: c.uid,
    )
    dataset_name = get_dataset_name(path)
    first_view_uid = FIRST_VIEW.get(dataset_name, [all_train_cameras_list[0].uid])[0]

    selected_view_indices: List[int] = []
    if n_train_images >= len(all_train_cameras_list):
        selected_view_indices = [cam.uid for cam in all_train_cameras_list]
    elif n_train_images > 1:
        print(f"Selecting {n_train_images} training views using max-min dispersion.")
        cam_positions = np.array(
            [
                np.linalg.inv(getWorld2View2(cam.R, cam.T))[:3, 3]
                for cam in all_train_cameras_list
            ]
        )
        try:
            first_view_list_index = [cam.uid for cam in all_train_cameras_list].index(
                first_view_uid
            )
        except ValueError:
            first_view_list_index = None
        selected_indices_in_list = find_max_min_dispersion_subset(
            cam_positions, n_train_images, first_view_list_index
        )
        selected_cameras = [all_train_cameras_list[i] for i in selected_indices_in_list]
        selected_view_indices = [cam.uid for cam in selected_cameras]
    else:
        selected_view_indices = [first_view_uid]

    print("Main training views indices:", selected_view_indices)
    train_cameras_dict = {
        idx: cams
        for idx, cams in train_cameras_info.items()
        if idx in selected_view_indices
    }

    adjacent_views = []
    for view in selected_view_indices:
        adjacent_views.extend(
            multiplexing.get_adjacent_views(
                train_cameras_dict[view][0], full_test_cameras_info
            )
        )
    adjacent_views = list(set(adjacent_views))

    test_cameras_info = [
        cam for cam in full_test_cameras_info if cam.uid in adjacent_views
    ]
    nerf_normalization = getNerfppNorm(all_train_cameras_list)

    pcd, ply_path = generate_random_pcd(path, num_pts=100_000)
    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cameras_dict,
        test_cameras=test_cameras_info,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        full_test_cameras=full_test_cameras_info,
    )

    if visualization_dir:
        default_filename = (
            visualization_filename or f"{dataset_name}_cameras_initial.html"
        )
        save_camera_visualization(
            scene_info,
            visualization_dir,
            filename=default_filename,
            title=f"{dataset_name} Cameras",
            highlighted_view_indices=selected_view_indices,
            ply_override_path=input_ply_path,
        )
    return scene_info


def apply_offset(
    args: ModelParams, gs: GaussianModel, scene_info: SceneInfo
) -> SceneInfo:
    print(f"Applying camera offset (mm): {args.camera_offset}")

    new_train_cameras = copy.deepcopy(scene_info.train_cameras)
    gs_center = gs.get_xyz.mean(dim=0).detach().cpu().numpy()

    for view_index, cam_info_list in scene_info.train_cameras.items():
        updated_cam_info_list = []
        for cam_info in cam_info_list:
            world_center = -cam_info.R @ cam_info.T
            initial_distance = np.linalg.norm(world_center - gs_center)

            # Offset along camera's local +Z using the reusable helper
            offset_world = mm_to_world(float(args.camera_offset))
            new_cam = _offset_camera(cam_info, np.array([0.0, 0.0, offset_world]))
            new_world_center = -new_cam.R @ new_cam.T

            new_distance = np.linalg.norm(new_world_center - gs_center)
            focal_scaling_factor = new_distance / initial_distance

            initial_focal_x = fov2focal(cam_info.FovX, cam_info.width)
            initial_focal_y = fov2focal(cam_info.FovY, cam_info.height)

            new_focal_x = initial_focal_x * focal_scaling_factor
            new_focal_y = initial_focal_y * focal_scaling_factor

            newFovX = focal2fov(new_focal_x, cam_info.width)
            newFovY = focal2fov(new_focal_y, cam_info.height)

            updated_cam_info = new_cam._replace(FovX=newFovX, FovY=newFovY)
            updated_cam_info_list.append(updated_cam_info)
        new_train_cameras[view_index] = updated_cam_info_list

    return scene_info._replace(train_cameras=new_train_cameras)


def create_multiplexed_views(
    scene_info: SceneInfo, n_multiplexed_images: int = 16
) -> SceneInfo:
    new_train_cameras: Dict[int, List[CameraInfo]] = defaultdict(list)
    x_linspace = np.linspace(
        start=-0.5, stop=0.5, num=int(np.sqrt(n_multiplexed_images))
    )

    for view_idx, cam_info_list in scene_info.train_cameras.items():
        for cam_info in cam_info_list:
            c2w = np.linalg.inv(getWorld2View2(cam_info.R, cam_info.T))

            sub_image_idx: int = 0
            for x in x_linspace:
                for y in x_linspace:
                    new_c2w = c2w.copy()
                    camera_offset = np.array([x, y, 0])
                    world_offset = new_c2w[:3, :3] @ camera_offset
                    new_c2w[:3, 3] += world_offset

                    new_w2c = np.linalg.inv(new_c2w)
                    new_R = np.transpose(new_w2c[:3, :3])
                    new_T = new_w2c[:3, 3]

                    uid = sub_image_idx
                    image_name = f"r_{view_idx}_{uid}"

                    new_cam_info = CameraInfo(
                        uid=uid,
                        groupid=view_idx,
                        R=new_R,
                        T=new_T,
                        FovX=cam_info.FovX,
                        FovY=cam_info.FovY,
                        image=cam_info.image,
                        image_path=cam_info.image_path,
                        image_name=image_name,
                        width=cam_info.width,
                        height=cam_info.height,
                        mask_name=cam_info.mask_name,
                        mask=cam_info.mask,
                    )
                    new_train_cameras[view_idx].append(new_cam_info)
                    sub_image_idx += 1
    return scene_info._replace(train_cameras=new_train_cameras)


def create_stereo_views(scene_info: SceneInfo, baseline: float = 0.3) -> SceneInfo:
    new_train_cameras: Dict[int, List[CameraInfo]] = defaultdict(list)
    key_offset = max(list(scene_info.train_cameras.keys())) + 1
    print(world_to_m(baseline), "m baseline")
    for view_idx, cam_info_list in scene_info.train_cameras.items():
        for cam_info in cam_info_list:
            left_cam = _make_shifted_scaled_cam(
                cam_info,
                offset_xyz=[-baseline / 2, 0, 0],
                scale=1.0,
                uid=view_idx,
                image_name=f"{cam_info.image_name}_left",
            )
            left_cam = left_cam._replace(groupid=view_idx)
            new_train_cameras[view_idx] = [left_cam]

            right_view_idx = view_idx + key_offset
            right_cam = _make_shifted_scaled_cam(
                cam_info,
                offset_xyz=[baseline / 2, 0, 0],
                scale=1.0,
                uid=right_view_idx,
                image_name=f"{cam_info.image_name}_right",
            )
            right_cam = right_cam._replace(groupid=view_idx)
            new_train_cameras[right_view_idx] = [right_cam]

    return scene_info._replace(train_cameras=new_train_cameras)


def create_iphone_views(
    scene_info: SceneInfo,
    same_focal_lengths: bool = False,
    baseline_x: float = 9.5,
    baseline_y: float = 9.5,
) -> SceneInfo:
    """Create iPhone-like triple-camera views with metric baselines and focal lengths.

    Notes:
    - `baseline_x`, `baseline_y` are interpreted as millimeters and converted to world units.
    - If `same_focal_lengths` is False and `PIXEL_SIZE_MM > 0`, focal length is enforced
      in absolute millimeters (13mm ultrawide, 24mm wide, 77mm tele) by scaling intrinsics.
      Otherwise, falls back to ratios (13/24 and 77/24) relative to the source view.
    """
    # Convert millimeter baselines to world units
    bx_world, by_world = mm_to_world(baseline_x), mm_to_world(baseline_y)

    new_train_cameras: Dict[int, List[CameraInfo]] = defaultdict(list)
    key_offset = max(list(scene_info.train_cameras.keys())) + 1
    second_offset = key_offset * 2
    for view_idx, cam_info_list in scene_info.train_cameras.items():
        for cam_info in cam_info_list:
            if same_focal_lengths:
                uw_scale = wide_scale = tele_scale = 1.0
            else:
                fx_px = fov2focal(cam_info.FovX, cam_info.width)
                fx_uw_px = 13.0 / PIXEL_SIZE_MM
                fx_wide_px = 24.0 / PIXEL_SIZE_MM
                fx_tele_px = 77.0 / PIXEL_SIZE_MM

                uw_scale = fx_uw_px / fx_px
                wide_scale = fx_wide_px / fx_px
                tele_scale = fx_tele_px / fx_px

            uw_idx = view_idx
            ultrawide = _make_shifted_scaled_cam(
                cam_info,
                offset_xyz=[-bx_world, -by_world, 0],
                scale=uw_scale,
                uid=uw_idx,
                image_name=f"{cam_info.image_name}_uw",
            )
            ultrawide = ultrawide._replace(groupid=view_idx)
            new_train_cameras[view_idx] = [ultrawide]

            wide_idx = view_idx + key_offset
            wide = _make_shifted_scaled_cam(
                cam_info,
                offset_xyz=[bx_world, -by_world, 0],
                scale=wide_scale,
                uid=wide_idx,
                image_name=f"{cam_info.image_name}_wide",
            )
            wide = wide._replace(groupid=view_idx)
            new_train_cameras[wide_idx] = [wide]

            tele_idx = view_idx + second_offset
            tele = _make_shifted_scaled_cam(
                cam_info,
                offset_xyz=[0, by_world, 0],
                scale=tele_scale,
                uid=tele_idx,
                image_name=f"{cam_info.image_name}_tele",
            )
            tele = tele._replace(groupid=view_idx)
            new_train_cameras[tele_idx] = [tele]

    return scene_info._replace(train_cameras=new_train_cameras)


######### Read COLMAP Format #########


def get_bounding_box(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY)
    scale_factor = 1 / 8
    binary_image = cv2.resize(binary_image, (0, 0), fx=scale_factor, fy=scale_factor)
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise ValueError("No contours found in the image.")
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    return (x, y, w, h), largest_contour


def draw_bounding_box(image_path, bbox):
    image = cv2.imread(image_path)
    scale_factor = 1 / 8
    image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
    _, image = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY)

    x, y, w, h = bbox
    # Draw the bounding box on the image
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image


# TODO: fix the below function and retrain with and without multiplexing
def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    test_scene = []
    # print("before ", list(cam_extrinsics.keys()))
    # cam_extrinsics_keys = random.choices(list(cam_extrinsics.keys()), k=3)
    # The line `cam_extrinsics_keys = [4,10,26,33]` is creating a list of specific keys that will be
    # used to filter out certain camera extrinsics during the processing of reading Colmap cameras.
    # Only the camera extrinsics with keys matching the values in the list `[4, 10, 26, 33]` will be
    # considered, while others will be skipped. This allows for selective processing of camera
    # extrinsics based on the specified keys.
    # cam_extrinsics_keys = [4,10,26,33]
    # 5 views:45,41, 29,16,12
    # 9 views:45,43,41,31,29,27,16,14,12
    # random.sample(range(0,42,1), k=35)
    selected_img_name = [f"IMG_00{i}.JPG" for i in [45, 43, 41, 31, 29, 27, 16, 14, 12]]
    # must_have_test = "IMG_0029.JPG"
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write("\r")
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        is_test_scene = False
        if extr.name not in selected_img_name:
            is_test_scene = True

        focal_length_x = 4820.643636961659
        focal_length_y = 3594.0708161747893
        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        image_mask_combo = {
            "IMG_0011.JPG": "IMG_0098.JPG",
            "IMG_0012.JPG": "IMG_0097.JPG",
            "IMG_0013.JPG": "IMG_0096.JPG",
            "IMG_0014.JPG": "IMG_0095.JPG",
            "IMG_0015.JPG": "IMG_0094.JPG",
            "IMG_0016.JPG": "IMG_0093.JPG",
            "IMG_0017.JPG": "IMG_0092.JPG",
            # "IMG_0018.JPG":"IMG_0091.JPG",
            "IMG_0019.JPG": "IMG_0091.JPG",
            "IMG_0020.JPG": "IMG_0090.JPG",
            "IMG_0021.JPG": "IMG_0089.JPG",
            "IMG_0022.JPG": "IMG_0088.JPG",
            "IMG_0023.JPG": "IMG_0086.JPG",
            "IMG_0024.JPG": "IMG_0085.JPG",
            "IMG_0025.JPG": "IMG_0084.JPG",
            "IMG_0026.JPG": "IMG_0083.JPG",
            "IMG_0027.JPG": "IMG_0082.JPG",
            "IMG_0028.JPG": "IMG_0081.JPG",
            "IMG_0029.JPG": "IMG_0080.JPG",
            "IMG_0030.JPG": "IMG_0079.JPG",
            "IMG_0031.JPG": "IMG_0078.JPG",
            "IMG_0032.JPG": "IMG_0077.JPG",
            "IMG_0033.JPG": "IMG_0076.JPG",
            "IMG_0034.JPG": "IMG_0075.JPG",
            "IMG_0035.JPG": "IMG_0074.JPG",
            "IMG_0036.JPG": "IMG_0073.JPG",
            "IMG_0037.JPG": "IMG_0072.JPG",
            "IMG_0038.JPG": "IMG_0071.JPG",
            "IMG_0039.JPG": "IMG_0070.JPG",
            "IMG_0040.JPG": "IMG_0055.JPG",
            "IMG_0041.JPG": "IMG_0056.JPG",
            "IMG_0042.JPG": "IMG_0057.JPG",
            "IMG_0043.JPG": "IMG_0058.JPG",
            "IMG_0044.JPG": "IMG_0059.JPG",
            "IMG_0045.JPG": "IMG_0060.JPG",
            "IMG_0046.JPG": "IMG_0061.JPG",
            "IMG_0047.JPG": "IMG_0062.JPG",
            "IMG_0048.JPG": "IMG_0063.JPG",
            "IMG_0049.JPG": "IMG_0064.JPG",
            "IMG_0050.JPG": "IMG_0065.JPG",
            "IMG_0051.JPG": "IMG_0066.JPG",
            "IMG_0052.JPG": "IMG_0067.JPG",
            "IMG_0053.JPG": "IMG_0069.JPG",
        }
        mask_name = ""
        if image_mask_combo.get(extr.name, None):
            # mask_path = f"/home/vitran/gs6/2024_04_06/masks/{image_mask_combo[extr.name]}"
            mask_name = image_mask_combo[extr.name]
            # mask = Image.open(mask_path).convert('L')
            # threshold = 20
            # mask = mask.point(lambda p: 1 if p > threshold else 0)

            image_path = (
                f"/home/vitran/gs6/2024_04_06/masks/{image_mask_combo[extr.name]}"
            )
            bbox, contour = get_bounding_box(image_path)
            image_with_bbox = draw_bounding_box(image_path, bbox)
            image_rgb = cv2.cvtColor(image_with_bbox, cv2.COLOR_BGR2RGB)
            mask = PinholeMask(
                bbox=bbox, mask=image_rgb, path=image_path.split("/")[-1]
            )
        else:
            mask = None

        cam_info = CameraInfo(
            uid=uid,
            groupid=uid,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=image,
            mask=mask,
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height,
            mask_name=mask_name,
        )
        if is_test_scene:
            test_scene.append(cam_info)
        else:
            cam_infos.append(cam_info)
    sys.stdout.write("\n")
    return cam_infos, test_scene


def readColmapSceneInfo(
    path,
    images,
    eval,
    use_multiplexing: bool = False,
    n_multiplexed_images: int = 16,
    llffhold: int = 8,
    visualization_dir: Optional[str] = None,
    visualization_filename: Optional[str] = None,
    input_ply_path: Optional[str] = None,
):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except Exception:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = (
        "/home/vitran/gs6/2024_04_06/input"  # "input" if images == None else images
    )
    # print("read colmapSceneInfo ", cam_extrinsics.keys())
    cam_infos_unsorted, test_cam_infos = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=os.path.join(path, reading_dir),
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
    #     test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    # bin_path = os.path.join(path, "sparse/0/points3D.bin")
    # txt_path = os.path.join(path, "sparse/0/points3D.txt")
    # random init for real data
    num_pts = 10000
    print(f"Generating random point cloud ({num_pts})...")

    # We create random points inside the bounds of the synthetic Blender scenes
    # xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3 #original
    # custom mean and std
    mean_xyz = [0.44064777, -0.25568339, 14.77189822]
    std_xyz = [1.66879624, 2.07055282, 2.40053613]
    xyz = []
    for n in range(3):
        pts = np.random.random(num_pts)
        # shifted_pts = pts * std_xyz[n] + (mean_xyz[n] - std_xyz[n] / 2)
        shifted_pts = pts * std_xyz[n] + mean_xyz[n]
        xyz.append(shifted_pts)

        actual_mean = np.mean(shifted_pts)
        actual_stddev = np.std(shifted_pts)
        print(f"Desired Mean: {mean_xyz[n]}, Actual Mean: {actual_mean}")
        print(f"Desired Stddev: {std_xyz[n]}, Actual Stddev: {actual_stddev}")
    xyz = np.stack(xyz, axis=1)
    shs = np.random.random((num_pts, 3)) / 255.0
    pcd = BasicPointCloud(
        points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
    )

    storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except Exception:
        pcd = None
    # uncomment to get colmap init
    # if not os.path.exists(ply_path):
    #     print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
    #     try:
    #         xyz, rgb, _ = read_points3D_binary(bin_path)
    #     except:
    #         xyz, rgb, _ = read_points3D_text(txt_path)
    #     storePly(ply_path, xyz, rgb)
    # try:
    #     pcd = fetchPly(ply_path)
    # except:
    #     pcd = None

    # tv_cam_infos = readCamerasFromTransforms("/home/vitran/plenoxels/blender_data/lego", "transforms_train.json", False, ".png")
    # tv_cam_infos = readCamerasFromTransforms("/home/vitran/plenoxels/blender_data/lego", "transforms_train.json", False, ".png")

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        full_test_cameras=[],
    )

    if visualization_dir:
        dataset_name = get_dataset_name(path)
        default_filename = (
            visualization_filename or f"{dataset_name}_cameras_initial.html"
        )
        save_camera_visualization(
            scene_info,
            visualization_dir,
            filename=default_filename,
            title=f"{dataset_name} Cameras",
            ply_override_path=input_ply_path,
        )
    return scene_info
