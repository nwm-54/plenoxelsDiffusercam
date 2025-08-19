#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import json
import os
import sys
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from utils.graphics_utils import focal2fov, fov2focal, getWorld2View2
from utils.sh_utils import SH2RGB

from scene import multiplexing
from scene.colmap_loader import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
)
from scene.gaussian_model import BasicPointCloud


class PinholeMask(NamedTuple):
    bbox: list
    mask: np.array
    path: str


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    mask_name: str
    mask: np.array


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    full_test_cameras: list


def getNerfppNorm(cam_info):
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


def get_bounding_box(image_path):
    # Read the grayscale image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Image file {image_path} not found.")

    # Apply a binary threshold to get a binary image
    _, binary_image = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY)

    scale_factor = 1 / 8
    binary_image = cv2.resize(binary_image, (0, 0), fx=scale_factor, fy=scale_factor)

    # Find contours in the binary image
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise ValueError("No contours found in the image.")

    # Get the largest contour (assuming the circle is the largest object in the image)
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding box for the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    return (x, y, w, h), largest_contour


def draw_bounding_box(image_path, bbox):
    image = cv2.imread(image_path)
    scale_factor = 1 / 8
    image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)

    if image is None:
        raise FileNotFoundError(f"Image file {image_path} not found.")
    _, image = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY)

    x, y, w, h = bbox
    # Draw the bounding box on the image
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image


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
    # selected_img_name = [f"IMG_00{i}.JPG" for i in range(11,53)]
    # must_have_test = "IMG_0029.JPG"
    print("print selected ", selected_img_name)
    for idx, key in enumerate(cam_extrinsics):
        # if key not in cam_extrinsics_keys:
        #     continue
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

        # image_mask_combo = {"IMG_0012.JPG":"IMG_0097.JPG",
        #                     "IMG_0014.JPG":"IMG_0095.JPG",
        #                     "IMG_0016.JPG":"IMG_0093.JPG",
        #                     "IMG_0020.JPG":"IMG_0090.JPG",
        #                     "IMG_0022.JPG":"IMG_0088.JPG",
        #                     "IMG_0027.JPG":"IMG_0082.JPG",
        #                     "IMG_0029.JPG":"IMG_0080.JPG",
        #                     "IMG_0030.JPG":"IMG_0079.JPG",
        #                     "IMG_0031.JPG":"IMG_0078.JPG",
        #                     "IMG_0033.JPG":"IMG_0076.JPG",
        #                     "IMG_0037.JPG":"IMG_0072.JPG",
        #                     "IMG_0041.JPG":"IMG_0056.JPG",
        #                     "IMG_0043.JPG":"IMG_0058.JPG",
        #                     "IMG_0044.JPG":"IMG_0059.JPG",
        #                     "IMG_0045.JPG":"IMG_0060.JPG",
        #                     "IMG_0046.JPG":"IMG_0061.JPG",
        #                     "IMG_0047.JPG":"IMG_0062.JPG",}
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


def fetchPly(path):
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


def readColmapSceneInfo(path, images, eval, llffhold=8):
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
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []
    cam_infos_test = []
    if "lego" in path:
        adjacent_views = multiplexing.get_adjacent_views(59, path)
    elif "hotdog" in path:
        adjacent_views = multiplexing.get_adjacent_views(0, path)
    else:
        adjacent_views = multiplexing.get_adjacent_views(2, path)
    # print("adjacent views", adjacent_views)

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(
                w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)
            # image = image.resize((200, 200))
            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (
                1 - norm_data[:, :, 3:4]
            )
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            # fix the below part for non simulation
            if ("test" not in transformsfile and idx in multiplexing.SUBIMAGES) or (
                "test" in transformsfile and idx in adjacent_views
            ):
                cam_infos.append(
                    CameraInfo(
                        uid=idx,
                        R=R,
                        T=T,
                        FovY=FovY,
                        FovX=FovX,
                        image=image,
                        mask=None,
                        mask_name="",
                        image_path=image_path,
                        image_name=image_name,
                        width=image.size[0],
                        height=image.size[1],
                    )
                )
            if "test" in transformsfile:  # all test view
                cam_infos_test.append(
                    CameraInfo(
                        uid=idx,
                        R=R,
                        T=T,
                        FovY=FovY,
                        FovX=FovX,
                        image=image,
                        mask=None,
                        mask_name="",
                        image_path=image_path,
                        image_name=image_name,
                        width=image.size[0],
                        height=image.size[1],
                    )
                )

    return cam_infos, cam_infos_test


def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    # train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    # lego - chair -others
    # hotdog
    train_cam_infos, _ = readCamerasFromTransforms(
        path, "transforms_train_gaussian_splatting.json", white_background, extension
    )

    test_cam_infos, full_test_cam_infos = readCamerasFromTransforms(
        path, "transforms_test.json", white_background, extension
    )

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if True:
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random spherical point cloud with {num_pts} points")

        # Cuboid
        # We create random points inside the bounds of the synthetic Blender scenes
        # xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3

        # Spherical
        # Generate random radii, uniformly distributed within [0, 1)
        r = (
            np.random.random(num_pts) ** (1 / 3) * 1.3
        )  # Adjust distribution to account for volume + magic number 1.3
        # r = np.random.random(num_pts) ** (1/3)

        # Generate spherical coordinates
        theta = np.random.uniform(0, 2 * np.pi, num_pts)  # Azimuthal angle [0, 2π)
        phi = np.random.uniform(0, np.pi, num_pts)  # Polar angle [0, π]
        # Convert spherical coordinates to Cartesian coordinates
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        xyz = np.stack((x, y, z), axis=-1)

        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(
            points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
        )

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except Exception:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        full_test_cameras=full_test_cam_infos,
    )
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
}
