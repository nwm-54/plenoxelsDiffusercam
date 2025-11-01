import copy
import json
import os
import sys
from collections import defaultdict
from itertools import product
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from arguments import ModelParams
from arguments.camera_presets import MULTIVIEW_INDICES
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
    camera_center_world,
    camera_with_center,
    generate_random_pcd,
    getNerfppNorm,
    mm_to_world,
    read_camera,
    solve_offset_for_angle,
    world_to_m,
)
from utils.general_utils import get_dataset_name
from utils.graphics_utils import focal2fov, fov2focal, getWorld2View2
from utils.render_utils import fetchPly, find_max_min_dispersion_subset
from utils.sh_utils import SH2RGB

FIRST_VIEW: Dict[str, List[int]] = MULTIVIEW_INDICES[1]
PIXEL_SIZE_MM = 0.00244

IMAGE_MASK_COMBO = {
    "IMG_0011.JPG": "IMG_0098.JPG",
    "IMG_0012.JPG": "IMG_0097.JPG",
    "IMG_0013.JPG": "IMG_0096.JPG",
    "IMG_0014.JPG": "IMG_0095.JPG",
    "IMG_0015.JPG": "IMG_0094.JPG",
    "IMG_0016.JPG": "IMG_0093.JPG",
    "IMG_0017.JPG": "IMG_0092.JPG",
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


def _camera_basis_vectors(
    cam: CameraInfo,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the camera centre and orthonormal basis vectors (x, y, z)."""

    base_center = camera_center_world(cam)
    R = np.asarray(cam.R, dtype=np.float64)
    x_axis = R[:, 0]
    y_axis = R[:, 1]
    z_axis = R[:, 2]

    x_unit = x_axis / max(np.linalg.norm(x_axis), 1e-12)
    y_unit = y_axis / max(np.linalg.norm(y_axis), 1e-12)
    z_unit = z_axis / max(np.linalg.norm(z_axis), 1e-12)

    return base_center, x_unit, y_unit, z_unit


def _camera_with_new_center(
    cam: CameraInfo, center_world: np.ndarray, uid: int, groupid: int, image_name: str
) -> CameraInfo:
    """Create a copy of ``cam`` with an updated centre and identifiers."""

    return camera_with_center(cam, center_world)._replace(
        uid=uid,
        groupid=groupid,
        image_name=image_name,
    )


def _compute_baseline(
    cam: CameraInfo,
    obj_center: np.ndarray,
    angle_deg: float,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base_center, x_unit, y_unit, _ = _camera_basis_vectors(cam)
    base_vec = base_center - obj_center
    offset = solve_offset_for_angle(base_vec, x_unit, angle_deg)
    left_center = base_center - offset * x_unit
    right_center = base_center + offset * x_unit
    return offset, base_center, x_unit, y_unit, left_center, right_center


def print_camera_metrics(scene_info: SceneInfo, obj_center: Optional[np.ndarray]):
    pixel_size_mm = PIXEL_SIZE_MM

    headline_views = [
        (view_idx, cam_list[0])
        for view_idx, cam_list in scene_info.train_cameras.items()
        if cam_list
    ][:10]
    for view_idx, cam in headline_views:
        fx_px = fov2focal(cam.FovX, cam.width)
        fy_px = fov2focal(cam.FovY, cam.height)
        fx_mm = fx_px * pixel_size_mm
        fy_mm = fy_px * pixel_size_mm

        center_world = camera_center_world(cam)
        dist_world = (
            np.linalg.norm(center_world)
            if obj_center is None
            else np.linalg.norm(center_world - obj_center)
        )
        print(
            f"View {view_idx} ({cam.image_name}): fx={fx_mm:.3f} mm (px={fx_px:.1f}), "
            f"fy={fy_mm:.3f} mm (px={fy_px:.1f}); distance={world_to_m(dist_world):.3f} m"
        )

    if obj_center is None:
        return None

    groups: Dict[int, List[Tuple[CameraInfo, np.ndarray]]] = defaultdict(list)
    for cam_list in scene_info.train_cameras.values():
        for cam in cam_list:
            groups[cam.groupid].append((cam, camera_center_world(cam)))

    if not groups:
        return None

    group_angles: Dict[int, float] = {}
    group_distances: Dict[int, float] = {}
    group_metrics: Dict[int, Dict[str, float]] = {}

    for gid, entries in sorted(groups.items()):
        if len(entries) < 2:
            continue

        ref_cam, ref_center = entries[0]
        ref_x_axis = ref_cam.R[:, 0]
        ref_x_axis = ref_x_axis / max(np.linalg.norm(ref_x_axis), 1e-12)

        projected = [
            (
                float(np.dot(center - ref_center, ref_x_axis)),
                cam,
                center,
            )
            for cam, center in entries
        ]

        left_offset, left_cam, left_center = min(projected, key=lambda item: item[0])
        right_offset, right_cam, right_center = max(projected, key=lambda item: item[0])

        if np.isclose(left_offset, right_offset):
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
        angle_deg = float(np.degrees(np.arccos(cos_theta)))
        mean_distance_m = float(
            world_to_m(
                np.mean([np.linalg.norm(center - obj_center) for _, center in entries])
            )
        )

        group_angles[gid] = angle_deg
        group_distances[gid] = mean_distance_m
        group_metrics[gid] = {"angle_deg": angle_deg, "distance_m": mean_distance_m}

        baseline_m = world_to_m(float(np.linalg.norm(right_center - left_center)))
        print(
            f"Group {gid}: left={left_cam.image_name} (local x={left_offset:.6f}), "
            f"right={right_cam.image_name} (local x={right_offset:.6f}) -> "
            f"angle={angle_deg:.2f} deg, avg_dist={mean_distance_m:.3f} m, "
            f"baseline={baseline_m:.3f} m"
        )

    avg_angle = float(np.mean(list(group_angles.values()))) if group_angles else None
    if avg_angle is not None:
        print(f"Average group angle: {avg_angle:.2f} deg")

    return {
        "group_angles_deg": group_angles,
        "group_distances_m": group_distances,
        "group_metrics": group_metrics,
        "avg_group_angle_deg": avg_angle,
    }


def _load_cameras_from_json(
    json_path: str,
    path: str,
    white_background: bool,
    extension: str,
) -> Tuple[float, List[Tuple[str, str, int, str]]]:
    with open(json_path) as json_file:
        contents = json.load(json_file)
        fov_x = contents["camera_angle_x"]
        frames = contents["frames"]

        camera_data = []
        for frame in frames:
            image_name = frame["file_path"].split("/")[-1]
            index = int(image_name.split("_")[1])
            image_filepath = os.path.join(path, frame["file_path"] + extension)
            camera_data.append((image_filepath, image_name, index, frame["transform_matrix"]))

        return fov_x, camera_data


def read_cameras_from_transforms(
    path: str,
    train_transforms_file: str,
    test_transforms_file: str,
    white_background: bool,
    extension: str,
) -> Tuple[Dict[int, List[CameraInfo]], List[CameraInfo]]:
    train_cameras_info = defaultdict(list)
    test_cameras_info = []

    fov_x, train_data = _load_cameras_from_json(
        os.path.join(path, train_transforms_file), path, white_background, extension
    )
    for image_filepath, image_name, index, transform_matrix in train_data:
        train_cameras_info[index].append(
            read_camera(
                image_filepath,
                image_name,
                transform_matrix,
                white_background,
                index,
                fov_x,
            )._replace(groupid=index)
        )

    fov_x, test_data = _load_cameras_from_json(
        os.path.join(path, test_transforms_file), path, white_background, extension
    )
    for image_filepath, image_name, index, transform_matrix in test_data:
        test_cameras_info.append(
            read_camera(
                image_filepath,
                image_name,
                transform_matrix,
                white_background,
                index,
                fov_x,
            )._replace(groupid=index)
        )

    return train_cameras_info, test_cameras_info


def readNerfSyntheticInfo(
    path: str,
    white_background: bool,
    eval: bool,
    extension: str = ".png",
    n_train_images: int = 1,
    use_orbital_trajectory: bool = False,
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
        ).tolist()
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
            world_center = camera_center_world(cam_info)
            initial_distance = np.linalg.norm(world_center - gs_center)

            # Offset along camera's local +Z
            offset_world = mm_to_world(float(args.camera_offset))
            new_cam = _offset_camera(cam_info, np.array([0.0, 0.0, offset_world]))
            new_world_center = camera_center_world(new_cam)

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
    scene_info: SceneInfo,
    obj_center: Optional[np.ndarray],
    angle_deg: float,
    n_multiplexed_images: int = 16,
) -> SceneInfo:
    if obj_center is None:
        return scene_info

    def _generate_multiplexed_group(
        cam_info: CameraInfo,
        groupid: int,
        name_prefix: str,
    ) -> List[CameraInfo]:
        """Return ``n_multiplexed_images`` cameras offset around ``cam_info``."""
        base_center, x_unit, y_unit, _ = _camera_basis_vectors(cam_info)
        stereo_offset = solve_offset_for_angle(base_center - obj_center, x_unit, angle_deg)

        grid_size = max(int(np.sqrt(n_multiplexed_images)), 1)
        grid_positions = list(product(np.linspace(-1.0, 1.0, grid_size), repeat=2))

        generated: List[CameraInfo] = []
        for uid, (x_pos, y_pos) in enumerate(grid_positions[:n_multiplexed_images]):
            if np.isclose(x_pos, -1.0):
                new_center = base_center - stereo_offset * x_unit
            elif np.isclose(x_pos, 1.0):
                new_center = base_center + stereo_offset * x_unit
            else:
                x_offset_world = x_unit * (stereo_offset * x_pos)
                y_offset_world = y_unit * (stereo_offset * y_pos)
                new_center = base_center + x_offset_world + y_offset_world

            new_cam = _camera_with_new_center(
                cam_info,
                new_center,
                uid=uid,
                groupid=groupid,
                image_name=f"{name_prefix}_{uid}",
            )
            generated.append(new_cam)
        return generated

    new_train_cameras: Dict[int, List[CameraInfo]] = defaultdict(list)

    for view_idx, cam_info_list in scene_info.train_cameras.items():
        for cam_info in cam_info_list:
            new_train_cameras[view_idx].extend(
                _generate_multiplexed_group(
                    cam_info,
                    view_idx,
                    name_prefix=f"r_{view_idx}",
                )
            )

    # Leave test and full-test camera lists untouched so evaluation occurs on single views.
    return scene_info._replace(train_cameras=new_train_cameras)


def create_stereo_views(
    scene_info: SceneInfo,
    obj_center: Optional[np.ndarray],
    angle_deg: float,
) -> SceneInfo:
    if obj_center is None or angle_deg <= 0.0:
        return scene_info

    new_train_cameras: Dict[int, List[CameraInfo]] = defaultdict(list)
    key_offset = max(scene_info.train_cameras.keys()) + 1

    for view_idx, cam_info_list in scene_info.train_cameras.items():
        for cam_info in cam_info_list:
            (_, _, _, _, left_center, right_center) = _compute_baseline(
                cam_info, obj_center, angle_deg
            )

            left_cam = camera_with_center(
                cam_info, left_center
            )._replace(uid=view_idx, groupid=view_idx, image_name=f"{cam_info.image_name}_left")
            new_train_cameras[view_idx] = [left_cam]

            right_cam = camera_with_center(
                cam_info, right_center
            )._replace(uid=view_idx + key_offset, groupid=view_idx, image_name=f"{cam_info.image_name}_right")
            new_train_cameras[view_idx + key_offset] = [right_cam]

    return scene_info._replace(train_cameras=new_train_cameras)


def _compute_iphone_focal_scales(
    cam_info: CameraInfo, same_focal_lengths: bool, use_angle_positioning: bool
) -> Tuple[float, float, float]:
    if same_focal_lengths:
        return 1.0, 1.0, 1.0

    if use_angle_positioning:
        return 13.0 / 24.0, 1.0, 77.0 / 24.0

    fx_px = fov2focal(cam_info.FovX, cam_info.width)
    return (
        (13.0 / PIXEL_SIZE_MM) / fx_px,
        (24.0 / PIXEL_SIZE_MM) / fx_px,
        (77.0 / PIXEL_SIZE_MM) / fx_px,
    )


def _compute_iphone_offsets(
    cam_info: CameraInfo,
    obj_center: Optional[np.ndarray],
    angle_deg: Optional[float],
    baseline_x: float,
    baseline_y: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    bx_world = mm_to_world(baseline_x)
    by_world = mm_to_world(baseline_y)
    baseline_ratio = abs(baseline_y / baseline_x) if abs(baseline_x) > 1e-6 else 1.0

    if angle_deg is not None and obj_center is not None and angle_deg > 0.0:
        base_center, x_unit, y_unit, _ = _camera_basis_vectors(cam_info)
        base_vec = base_center - obj_center
        offset_x = solve_offset_for_angle(
            base_vec, x_unit, angle_deg, orth_axis=y_unit, orth_ratio=baseline_ratio
        )
        offset_y = baseline_ratio * offset_x

        if not np.isfinite(offset_x) or offset_x <= 0.0:
            offset_x, offset_y = bx_world, by_world

        return (
            np.array([-offset_x, -offset_y, 0.0]),
            np.array([offset_x, -offset_y, 0.0]),
            np.array([0.0, offset_y, 0.0]),
        )

    return (
        np.array([-bx_world, -by_world, 0.0]),
        np.array([bx_world, -by_world, 0.0]),
        np.array([0.0, by_world, 0.0]),
    )


def create_iphone_views(
    scene_info: SceneInfo,
    obj_center: Optional[np.ndarray] = None,
    angle_deg: Optional[float] = None,
    same_focal_lengths: bool = False,
    baseline_x: float = 9.5,
    baseline_y: float = 9.5,
) -> SceneInfo:
    use_angle_positioning = angle_deg is not None and obj_center is not None and angle_deg > 0.0

    new_train_cameras: Dict[int, List[CameraInfo]] = defaultdict(list)
    key_offset = max(scene_info.train_cameras.keys()) + 1
    second_offset = key_offset * 2

    for view_idx, cam_info_list in scene_info.train_cameras.items():
        for cam_info in cam_info_list:
            uw_scale, wide_scale, tele_scale = _compute_iphone_focal_scales(
                cam_info, same_focal_lengths, use_angle_positioning
            )
            uw_offset, wide_offset, tele_offset = _compute_iphone_offsets(
                cam_info, obj_center, angle_deg, baseline_x, baseline_y
            )

            cameras = [
                (view_idx, uw_offset, uw_scale, "uw"),
                (view_idx + key_offset, wide_offset, wide_scale, "wide"),
                (view_idx + second_offset, tele_offset, tele_scale, "tele"),
            ]

            for uid, offset, scale, suffix in cameras:
                cam = _make_shifted_scaled_cam(
                    cam_info, offset, scale, uid, f"{cam_info.image_name}_{suffix}"
                )
                new_train_cameras[uid] = [cam._replace(groupid=view_idx)]

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

        mask_name = ""
        if IMAGE_MASK_COMBO.get(extr.name, None):
            mask_name = IMAGE_MASK_COMBO[extr.name]

            image_path = f"/home/vitran/gs6/2024_04_06/masks/{IMAGE_MASK_COMBO[extr.name]}"
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
    pcd: Optional[BasicPointCloud] = None
    if os.path.exists(ply_path):
        try:
            pcd = fetchPly(ply_path)
        except Exception:
            pcd = None

    if pcd is None:
        num_pts = 10000
        print(f"Generating random point cloud ({num_pts})...")

        mean_xyz = [0.44064777, -0.25568339, 14.77189822]
        std_xyz = [1.66879624, 2.07055282, 2.40053613]
        xyz = []
        for n in range(3):
            pts = np.random.random(num_pts)
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
        ply_path = None

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

    return scene_info
