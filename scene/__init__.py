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
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import imageio.v3 as iio
import torch

from arguments import ModelParams
from scene import multiplexing
from scene.cameras import Camera
from scene.dataset_readers import (
    apply_offset,
    create_iphone_views,
    create_multiplexed_views,
    create_stereo_views,
    print_camera_metrics,
    readColmapSceneInfo,
    readNerfSyntheticInfo,
)
from scene.gaussian_model import GaussianModel
from scene.scene_utils import CameraInfo, configure_world_to_m
from utils.camera_utils import camera_to_JSON, cameraList_from_camInfos
from utils.general_utils import get_dataset_name
from utils.render_utils import (
    get_pretrained_splat_path,
    load_pretrained_splat,
    render_splat,
)
from utils.visualization_utils import save_camera_visualization

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Scene:
    gaussians: GaussianModel
    world_to_m: float

    def __init__(
        self,
        args: ModelParams,
        gaussians: GaussianModel,
        load_iteration: Optional[int] = None,
        resolution_scales: List[float] = [1.0],
    ):
        """
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.gaussians = gaussians
        self.world_to_m = args.world_to_m
        # Propagate world<->meters scale to dataset helpers
        configure_world_to_m(self.world_to_m)

        dataset_name = get_dataset_name(args.source_path)
        self.dataset_name = dataset_name
        self.avg_angle: Optional[float] = None
        self.group_metrics: Dict[int, Dict[str, float]] = {}
        self.object_center = None

        self.train_cameras: Dict[float, Dict[int, List]] = {}
        self.test_cameras: Dict[float, List] = {}
        self.full_test_cameras: Dict[float, List] = {}

        blender_transform_files = [
            "transforms_train.json",
            "transforms_test.json",
        ]
        is_blender_type = all(
            os.path.exists(os.path.join(args.source_path, f))
            for f in blender_transform_files
        )

        pretrained_ply_path = get_pretrained_splat_path(args)

        gs: Optional[GaussianModel] = None

        if os.path.exists(os.path.join(args.source_path, "sparse")):  # COLMAP
            print("Found sparse folder, assuming COLMAP dataset")
            self.n_multiplexed_images = 9  # TODO: set the number of multiplexed images

            scene_info = readColmapSceneInfo(
                args.source_path,
                args.images,
                args.eval,
                use_multiplexing=args.use_multiplexing,
                n_multiplexed_images=self.n_multiplexed_images,
                visualization_dir=args.model_path,
                input_ply_path=pretrained_ply_path,
            )
        elif is_blender_type:  # Blender
            self.n_multiplexed_images = 16
            print("Found transforms_*.json file, assuming Blender dataset")

            scene_info = readNerfSyntheticInfo(
                args.source_path,
                args.white_background,
                args.eval,
                n_train_images=args.n_train_images,
                use_orbital_trajectory=args.use_orbital_trajectory,
            )

            # re-rendering new views based on pretrained ply
            if pretrained_ply_path:
                gs = load_pretrained_splat(
                    pretrained_ply_path, sh_degree=args.sh_degree
                )
            if gs is not None:
                scene_info = apply_offset(args, gs, scene_info)
                obj_center = (
                    gs.get_xyz.mean(dim=0).detach().cpu().numpy()
                )
                self.object_center = obj_center
                if args.use_multiplexing:
                    scene_info = create_multiplexed_views(
                        scene_info,
                        obj_center,
                        args.angle_deg,
                        self.n_multiplexed_images,
                    )
                elif args.use_stereo:
                    scene_info = create_stereo_views(
                        scene_info, obj_center, args.angle_deg
                    )
                elif args.use_iphone:  # NEW
                    scene_info = create_iphone_views(
                        scene_info,
                        obj_center,
                        args.angle_deg,
                        args.iphone_same_focal_length
                    )
                scene_info = render_splat(args, gs, scene_info)
                metrics = print_camera_metrics(scene_info, self.object_center)
                if metrics:
                    self.avg_angle = metrics.get("avg_group_angle_deg")
                    self.group_metrics = metrics.get("group_metrics", {})
        else:
            raise ValueError(
                f"Could not infer scene type from source path: {args.source_path}"
            )

        point_cloud_for_vis: Optional[object] = (
            gs if gs is not None else scene_info.point_cloud
        )
        save_camera_visualization(
            scene_info,
            args.model_path,
            filename=f"{dataset_name}_cameras.html",
            title="",
            highlighted_view_indices=list(scene_info.train_cameras.keys()),
            point_cloud=point_cloud_for_vis,
        )

        with open(scene_info.ply_path, "rb") as src_file, open(os.path.join(self.model_path, "input.ply"), "wb") as dest_file:  # fmt: skip
            dest_file.write(src_file.read())

        json_cams: List[Dict] = []
        camlist_for_json: List[CameraInfo] = []
        if scene_info.train_cameras:
            for cam_list in scene_info.train_cameras.values():
                camlist_for_json.extend(cam_list)
        # if scene_info.test_cameras: camlist_for_json.extend(scene_info.test_cameras)
        # if scene_info.full_test_cameras: camlist_for_json.extend(scene_info.full_test_cameras)
        for id, cam in enumerate(camlist_for_json):
            json_cams.append(camera_to_JSON(id, cam))
        with open(os.path.join(self.model_path, "cameras.json"), "w") as file:
            json.dump(json_cams, file, indent=4)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            # Train cameras; Dict[main_view_idx, List[Camera]]
            self.train_cameras[resolution_scale] = {}
            if scene_info.train_cameras:
                for view_idx, cam_infos_for_view in scene_info.train_cameras.items():
                    self.train_cameras[resolution_scale][view_idx] = (
                        cameraList_from_camInfos(
                            cam_infos_for_view, resolution_scale, args
                        )
                    )
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.test_cameras, resolution_scale, args
            )
            self.full_test_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.full_test_cameras, resolution_scale, args
            )

        self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
        self.multiplexed_gt: Optional[Dict[int, torch.Tensor]] = None

    def save(self, iteration: int, path: str = "point_cloud.ply"):
        point_cloud_path = os.path.join(
            self.model_path, "point_cloud/iteration_{}".format(iteration)
        )
        print("Saving point cloud at ", point_cloud_path)
        self.gaussians.save_ply(os.path.join(point_cloud_path, path))

    def getTrainCameras(self, scale=1.0) -> Dict[int, List[Camera]]:
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0) -> List[Camera]:
        return self.test_cameras[scale]

    def getFullTestCameras(self, scale=1.0) -> List[Camera]:
        return self.full_test_cameras[scale]

    def init_multiplexing(self, dls: int, H: int, W: int):
        comap_yx, dim_lens_lf_yx = multiplexing.get_comap(
            self.n_multiplexed_images, dls, H, W
        )
        self.dim_lens_lf_yx = dim_lens_lf_yx
        self.comap_yx = torch.from_numpy(comap_yx).to(device)
        self.max_overlap = multiplexing.get_max_overlap(
            self.comap_yx, self.n_multiplexed_images, H, W
        )
        self.multiplexed_gt = self._load_ground_truth(H, W)

    def _load_ground_truth(self, H: int, W: int, scale=1.0) -> Dict[int, torch.Tensor]:
        gt: Dict[int, torch.Tensor] = {}
        for view_idx, cam_list in self.getTrainCameras(scale).items():
            if not isinstance(view_idx, int):
                continue

            cam_list = sorted(cam_list, key=lambda c: c.uid)
            imgs = [cam.original_image.to(cam.data_device) for cam in cam_list]
            gt[view_idx] = multiplexing.generate(
                imgs,
                self.comap_yx,
                self.dim_lens_lf_yx,
                self.n_multiplexed_images,
                H,
                W,
                self.max_overlap,
            )
            iio.imwrite(
                os.path.join(self.model_path, f"gt_view_{view_idx}.png"),
                (gt[view_idx].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"),
            )
        return gt

    def _write_camera_json(self):
        json_cams: List[Dict] = []
        id_counter = 0
        for _, viewdict in self.train_cameras.items():
            for view_idx, cam_list in viewdict.items():
                for cam in cam_list:
                    json_cams.append(camera_to_JSON(id_counter, cam))
                    id_counter += 1
        for _, cam_list in self.test_cameras.items():
            for cam in cam_list:
                json_cams.append(camera_to_JSON(id_counter, cam))
                id_counter += 1
        for _, cam_list in self.full_test_cameras.items():
            for cam in cam_list:
                json_cams.append(camera_to_JSON(id_counter, cam))
                id_counter += 1

        with open(os.path.join(self.model_path, "cameras.json"), "w") as file:
            json.dump(json_cams, file, indent=4)
