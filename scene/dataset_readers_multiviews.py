import copy
import json
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, NamedTuple, Optional, Tuple

import cv2
import numpy as np
import torch
from arguments import ModelParams
from gaussian_renderer import render
from PIL import Image
from plyfile import PlyData, PlyElement
from scene import multiplexing
from scene.cameras import Camera
from scene.colmap_loader import (qvec2rotmat, read_extrinsics_binary,
                                 read_extrinsics_text, read_intrinsics_binary,
                                 read_intrinsics_text, read_points3D_binary,
                                 read_points3D_text)
from scene.gaussian_model import BasicPointCloud, GaussianModel
from utils.graphics_utils import focal2fov, fov2focal, getWorld2View2
from utils.render_utils import camera_forward
from utils.sh_utils import SH2RGB
from arguments.multiviews_indices import MULTIVIEW_INDICES
from utils.general_utils import get_dataset_name

FIRST_VIEW: Dict[str, List[int]] = MULTIVIEW_INDICES[1]
class PinholeMask(NamedTuple):
    bbox: list
    mask: np.ndarray
    path: str
    
class CameraInfo(NamedTuple):
    uid: int
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

def find_max_min_dispersion_subset(points: np.ndarray, k: int) -> np.ndarray:
    """Finds a near-optimal subset of k points that maximizes the minimum distance."""
    if k < 1:
        return np.array([])
    n_points = len(points)
    if k >= n_points:
        return np.arange(n_points)

    # Get a good initial solution using Farthest Point Sampling
    selected_indices = np.zeros(k, dtype=int)
    rng = np.random.default_rng(seed=42)
    selected_indices[0] = rng.integers(n_points)
    dists = np.linalg.norm(points - points[selected_indices[0]], axis=1)
    for i in range(1, k):
        farthest_idx = np.argmax(dists)
        selected_indices[i] = farthest_idx
        new_dists = np.linalg.norm(points - points[farthest_idx], axis=1)
        dists = np.minimum(dists, new_dists)
    
    # Iteratively improve the solution with a local search (exchange heuristic)
    for _ in range(10):
        current_subset = set(selected_indices)
        
        sub_dist_matrix = np.linalg.norm(points[selected_indices, None] - points[None, selected_indices], axis=-1)
        np.fill_diagonal(sub_dist_matrix, np.inf)
        min_dist = sub_dist_matrix.min()

        made_swap = False
        for i in range(k):
            for p_out_idx in range(n_points):
                if p_out_idx in current_subset:
                    continue

                temp_indices = np.copy(selected_indices)
                temp_indices[i] = p_out_idx
                new_sub_dist_matrix = np.linalg.norm(points[temp_indices, None] - points[None, temp_indices], axis=-1)
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

def fetchPly(path) -> BasicPointCloud:
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def get_bounding_box(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY)
    scale_factor = 1/8
    binary_image = cv2.resize(binary_image, (0, 0), fx=scale_factor, fy=scale_factor)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No contours found in the image.")
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    return (x, y, w, h), largest_contour

def draw_bounding_box(image_path, bbox):
    image = cv2.imread(image_path)
    scale_factor = 1/8
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
    cam_extrinsics_keys = random.choices(list(cam_extrinsics.keys()), k=3)
    # The line `cam_extrinsics_keys = [4,10,26,33]` is creating a list of specific keys that will be
    # used to filter out certain camera extrinsics during the processing of reading Colmap cameras.
    # Only the camera extrinsics with keys matching the values in the list `[4, 10, 26, 33]` will be
    # considered, while others will be skipped. This allows for selective processing of camera
    # extrinsics based on the specified keys.
    # cam_extrinsics_keys = [4,10,26,33]
    # 5 views:45,41, 29,16,12
    # 9 views:45,43,41,31,29,27,16,14,12
    # random.sample(range(0,42,1), k=35)
    selected_img_name = [f"IMG_00{i}.JPG" for i in [45,43,41,31,29,27,16,14,12]] 
    # must_have_test = "IMG_0029.JPG"
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
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
        
        image_mask_combo = {"IMG_0011.JPG":"IMG_0098.JPG", 
                            "IMG_0012.JPG":"IMG_0097.JPG", 
                            "IMG_0013.JPG":"IMG_0096.JPG", 
                            "IMG_0014.JPG":"IMG_0095.JPG", 
                            "IMG_0015.JPG":"IMG_0094.JPG", 
                            "IMG_0016.JPG":"IMG_0093.JPG", 
                            "IMG_0017.JPG":"IMG_0092.JPG", 
                            # "IMG_0018.JPG":"IMG_0091.JPG", 
                            "IMG_0019.JPG":"IMG_0091.JPG", 
                            "IMG_0020.JPG":"IMG_0090.JPG", 
                            "IMG_0021.JPG":"IMG_0089.JPG", 
                            "IMG_0022.JPG":"IMG_0088.JPG", 
                            "IMG_0023.JPG":"IMG_0086.JPG", 
                            "IMG_0024.JPG":"IMG_0085.JPG", 
                            "IMG_0025.JPG":"IMG_0084.JPG", 
                            "IMG_0026.JPG":"IMG_0083.JPG", 
                            "IMG_0027.JPG":"IMG_0082.JPG", 
                            "IMG_0028.JPG":"IMG_0081.JPG", 
                            "IMG_0029.JPG":"IMG_0080.JPG", 
                            "IMG_0030.JPG":"IMG_0079.JPG", 
                            "IMG_0031.JPG":"IMG_0078.JPG", 
                            "IMG_0032.JPG":"IMG_0077.JPG", 
                            "IMG_0033.JPG":"IMG_0076.JPG", 
                            "IMG_0034.JPG":"IMG_0075.JPG", 
                            "IMG_0035.JPG":"IMG_0074.JPG", 
                            "IMG_0036.JPG":"IMG_0073.JPG", 
                            "IMG_0037.JPG":"IMG_0072.JPG", 
                            "IMG_0038.JPG":"IMG_0071.JPG", 
                            "IMG_0039.JPG":"IMG_0070.JPG", 
                            "IMG_0040.JPG":"IMG_0055.JPG", 
                            "IMG_0041.JPG":"IMG_0056.JPG", 
                            "IMG_0042.JPG":"IMG_0057.JPG", 
                            "IMG_0043.JPG":"IMG_0058.JPG", 
                            "IMG_0044.JPG":"IMG_0059.JPG",
                            "IMG_0045.JPG":"IMG_0060.JPG",
                            "IMG_0046.JPG":"IMG_0061.JPG",
                            "IMG_0047.JPG":"IMG_0062.JPG",
                            "IMG_0048.JPG":"IMG_0063.JPG", 
                            "IMG_0049.JPG":"IMG_0064.JPG", 
                            "IMG_0050.JPG":"IMG_0065.JPG", 
                            "IMG_0051.JPG":"IMG_0066.JPG", 
                            "IMG_0052.JPG":"IMG_0067.JPG", 
                            "IMG_0053.JPG":"IMG_0069.JPG",}
        mask_name = ""
        if image_mask_combo.get(extr.name, None):
            # mask_path = f"/home/vitran/gs6/2024_04_06/masks/{image_mask_combo[extr.name]}"
            mask_name = image_mask_combo[extr.name]
            # mask = Image.open(mask_path).convert('L')
            # threshold = 20  
            # mask = mask.point(lambda p: 1 if p > threshold else 0)
            
            image_path = mask_path = f"/home/vitran/gs6/2024_04_06/masks/{image_mask_combo[extr.name]}"
            bbox, contour = get_bounding_box(image_path)
            image_with_bbox = draw_bounding_box(image_path, bbox)
            image_rgb = cv2.cvtColor(image_with_bbox, cv2.COLOR_BGR2RGB)
            mask = PinholeMask(bbox=bbox, mask=image_rgb, path=image_path.split("/")[-1])
        else:
            mask = None

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, mask=mask,
                              image_path=image_path, image_name=image_name, width=width, height=height, mask_name=mask_name)
        if is_test_scene:
            test_scene.append(cam_info)
        else:
            cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos, test_scene

def readColmapSceneInfo(path, images, eval, use_multiplexing=False, n_multiplexed_images=16, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "/home/vitran/gs6/2024_04_06/input"#"input" if images == None else images
    # print("read colmapSceneInfo ", cam_extrinsics.keys())
    cam_infos_unsorted, test_cam_infos = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
    #     test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    #random init for real data
    num_pts = 10000
    print(f"Generating random point cloud ({num_pts})...")
    
    # We create random points inside the bounds of the synthetic Blender scenes
    # xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3 #original
    # custom mean and std
    mean_xyz = [ 0.44064777, -0.25568339, 14.77189822]
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
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

    storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    #uncomment to get colmap init
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

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path, full_test_cameras=[])
    return scene_info

def read_camera(image_filepath: str, image_name: str, transform_matrix: np.array, 
                white_background: bool, uid_for_image: int, fov_x: float) -> CameraInfo:
    c2w = np.array(transform_matrix) # NeRF 'transform_matrix' is a camera-to-world transform
    c2w[:3, 1:3] *= -1 # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
    w2c = np.linalg.inv(c2w) # get the world-to-camera transform and set R, T
    R = np.transpose(w2c[:3, :3]) # R is stored transposed due to 'glm' in CUDA code
    T = w2c[:3, 3]

    image = np.array(Image.open(image_filepath).convert("RGBA")) / 255.0
    bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
    image = image[:,:,:3] * image[:, :, 3:4] + bg * (1 - image[:, :, 3:4])
    image = Image.fromarray(np.array(image * 255.0, dtype=np.byte), "RGB")
    fov_y = focal2fov(fov2focal(fov_x, image.size[0]), image.size[1])

    return CameraInfo(uid=uid_for_image, R=R, T=T, FovY=fov_y, FovX=fov_x,
                      image=image, mask=None, mask_name="", image_path=image_filepath,
                      image_name=image_name, width=image.size[0], height=image.size[1])

def read_cameras_from_transforms(path: str, train_transforms_file: str, test_transforms_file: str,
                                 white_background: bool, extension: str
                                 ) -> Tuple[Dict[int, List[CameraInfo]], List[CameraInfo]]:
    train_cameras_info, test_cameras_info = defaultdict(list), []

    with open(os.path.join(path, train_transforms_file)) as json_file:
        contents = json.load(json_file)
        fov_x = contents["camera_angle_x"]
        frames = contents["frames"]

        for frame in frames:
            image_name: str = frame['file_path'].split('/')[-1]
            parts = image_name.split('_')
            index = int(parts[1])
            image_filepath = os.path.join(path, frame["file_path"] + extension)
            train_cameras_info[index].append(read_camera(image_filepath, image_name, frame["transform_matrix"],
                                                         white_background, index, fov_x))

    with open(os.path.join(path, test_transforms_file)) as json_file:
        contents = json.load(json_file)
        fov_x = contents["camera_angle_x"]
        frames = contents["frames"]

        for frame in frames:
            image_name: str = frame['file_path'].split('/')[-1]
            parts = image_name.split('_')
            index = int(parts[1])
            image_filepath = os.path.join(path, frame["file_path"] + extension)
            camera_info = read_camera(image_filepath, image_name, frame["transform_matrix"],
                                      white_background, index, fov_x)
            test_cameras_info.append(camera_info)

    return train_cameras_info, test_cameras_info

def readNerfSyntheticInfo(path: str, white_background: bool, eval: bool, 
                          extension: str = ".png", 
                          n_train_images: int = 1, 
                          use_orbital_trajectory: bool = False) -> SceneInfo:
    print(f"Reading Nerf synthetic scene from {path}")
    train_transforms_file = "transforms_train.json" if not use_orbital_trajectory else "orbital_trajectory.json"
    test_transforms_file = "transforms_test.json"

    train_cameras_info, full_test_cameras_info = read_cameras_from_transforms(
        path, train_transforms_file, test_transforms_file, white_background, extension)
    
    all_train_cameras_list = sorted([cam for cam_list in train_cameras_info.values() for cam in cam_list], key=lambda c: c.uid)
    selected_view_indices: List[int] = []
    if n_train_images == 1:
        dataset_name = get_dataset_name(path)
        selected_view_indices = FIRST_VIEW.get(dataset_name, [all_train_cameras_list[0].uid])
    elif n_train_images > 1 and n_train_images < len(all_train_cameras_list):
        print(f"Selecting {n_train_images} training views using max-min dispersion.")
        cam_positions = np.array([np.linalg.inv(getWorld2View2(cam.R, cam.T))[:3, 3] for cam in all_train_cameras_list])
        selected_indices_in_list = find_max_min_dispersion_subset(cam_positions, n_train_images)
        selected_cameras = [all_train_cameras_list[i] for i in selected_indices_in_list]
        selected_view_indices = [cam.uid for cam in selected_cameras]
    else:
        selected_view_indices = [cam.uid for cam in all_train_cameras_list]

    print("Main training views indices:", selected_view_indices)
    train_cameras_dict = {idx: cams for idx, cams in train_cameras_info.items() if idx in selected_view_indices}
    
    adjacent_views = []
    for view in selected_view_indices:
        adjacent_views.extend(multiplexing.get_adjacent_views(train_cameras_dict[view][0], full_test_cameras_info))
    adjacent_views = list(set(adjacent_views))

    test_cameras_info = [cam for cam in full_test_cameras_info if cam.uid in adjacent_views]
    nerf_normalization = getNerfppNorm(all_train_cameras_list)

    pcd, ply_path = generate_random_pcd(path, num_pts=100_000)
    return SceneInfo(point_cloud=pcd,
                     train_cameras=train_cameras_dict,
                     test_cameras=test_cameras_info,
                     nerf_normalization=nerf_normalization,
                     ply_path=ply_path, 
                     full_test_cameras=full_test_cameras_info)

def generate_random_pcd(path: str, num_pts: int = 100_000) -> Tuple[BasicPointCloud, str]:
    print(f"Generating random spherical point cloud with {num_pts} points")
    ply_path = os.path.join(path, "points3d.ply")
    
    # Cuboid
    # We create random points inside the bounds of the synthetic Blender scenes
    # xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
    
    # Spherical
    # Generate random radii, uniformly distributed within [0, 1)
    # r = np.random.random(num_pts) ** (1/3) * 1.3 # Adjust distribution to account for volume
    r = np.random.random(num_pts) ** (1/3)
    theta = np.random.uniform(0, 2 * np.pi, num_pts)  # Azimuthal angle [0, 2π)
    phi = np.random.uniform(0, np.pi, num_pts)        # Polar angle [0, π]
    # Convert spherical coordinates to Cartesian coordinates
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    xyz = np.stack((x, y, z), axis=-1)
    shs = np.random.random((num_pts, 3)) / 255.0 # random colors
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
    storePly(ply_path, xyz, SH2RGB(shs) * 255)
    
    return pcd, ply_path

def apply_offset(args: ModelParams, gs: GaussianModel, scene_info: SceneInfo) -> SceneInfo:
    print(f"Applying camera offset: {args.camera_offset}")

    new_train_cameras = copy.deepcopy(scene_info.train_cameras)
    gs_center = gs.get_xyz.mean(dim=0).detach().cpu().numpy()

    for view_index, cam_info_list in scene_info.train_cameras.items():
        updated_cam_info_list = []
        for cam_info in cam_info_list:
            world_center = -cam_info.R @ cam_info.T
            initial_distance = np.linalg.norm(world_center - gs_center)

            delta = camera_forward(cam_info) * float(args.camera_offset)
            new_world_center = world_center + delta
            new_T = -cam_info.R.T @ new_world_center
            
            new_distance = np.linalg.norm(new_world_center - gs_center)
            focal_scaling_factor = new_distance / initial_distance
            
            initial_focal_x = fov2focal(cam_info.FovX, cam_info.width)
            initial_focal_y = fov2focal(cam_info.FovY, cam_info.height)
            
            new_focal_x = initial_focal_x * focal_scaling_factor
            new_focal_y = initial_focal_y * focal_scaling_factor
            
            newFovX = focal2fov(new_focal_x, cam_info.width)
            newFovY = focal2fov(new_focal_y, cam_info.height)
            
            updated_cam_info = cam_info._replace(
                T=new_T,
                FovX=newFovX,
                FovY=newFovY
            )
            updated_cam_info_list.append(updated_cam_info)
        new_train_cameras[view_index] = updated_cam_info_list        

    return scene_info._replace(train_cameras=new_train_cameras)

def create_multiplexed_views(scene_info: SceneInfo, 
                             n_multiplexed_images: int = 16) -> SceneInfo:
    new_train_cameras: Dict[int, List[CameraInfo]] = defaultdict(list)
    x_linspace = np.linspace(start=-0.5, stop=0.5, num=int(np.sqrt(n_multiplexed_images)))

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
                        mask=cam_info.mask
                    )
                    new_train_cameras[view_idx].append(new_cam_info)
                    sub_image_idx += 1
    return scene_info._replace(train_cameras=new_train_cameras)