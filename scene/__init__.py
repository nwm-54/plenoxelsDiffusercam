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

import os
from typing import Dict, List, Optional

import imageio.v3 as iio
import numpy as np
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
from scene.scene_utils import configure_world_to_m
from utils.camera_utils import cameraList_from_camInfos
from utils.general_utils import get_dataset_name
from utils.render_utils import (
    get_pretrained_splat_path,
    load_pretrained_splat,
    render_splat,
    render_with_blender,
    write_camera_json,
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
        include_test_cameras: bool = False,
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
                obj_center = gs.get_xyz.mean(dim=0).detach().cpu().numpy()
                self.object_center = obj_center
        else:
            raise ValueError(
                f"Could not infer scene type from source path: {args.source_path}"
            )

        if gs is not None:
            if args.use_multiplexing:
                scene_info = create_multiplexed_views(
                    scene_info,
                    self.object_center,
                    args.angle_deg,
                    self.n_multiplexed_images,
                )
            elif args.use_stereo:
                scene_info = create_stereo_views(
                    scene_info, self.object_center, args.angle_deg
                )
            elif args.use_iphone:
                scene_info = create_iphone_views(
                    scene_info,
                    self.object_center,
                    args.angle_deg,
                    args.iphone_same_focal_length,
                )

        if args.use_blender:
            if is_blender_type:
                scene_info = render_with_blender(
                    args, scene_info, dataset_name, object_center=self.object_center
                )
            else:
                print(
                    "Warning: use_blender requested but dataset is not Blender-type; "
                    "skipping Blender rendering."
                )
                if gs is not None:
                    scene_info = render_splat(args, gs, scene_info)
        elif gs is not None:
            scene_info = render_splat(args, gs, scene_info)

        self.scene_info = scene_info

        metrics = print_camera_metrics(scene_info, self.object_center)
        if metrics:
            self.avg_angle = metrics.get("avg_group_angle_deg")
            self.group_metrics = metrics.get("group_metrics", {})

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

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            # Train cameras; Dict[main_view_idx, List[Camera]]
            self.train_cameras[resolution_scale] = {}
            if scene_info.train_cameras:
                for view_idx, cam_infos_for_view in scene_info.train_cameras.items():
                    normalized_cam_infos = [
                        cam_info
                        if getattr(cam_info, "groupid", None) is not None
                        else cam_info._replace(groupid=view_idx)
                        for cam_info in cam_infos_for_view
                    ]
                    self.train_cameras[resolution_scale][view_idx] = (
                        cameraList_from_camInfos(
                            normalized_cam_infos, resolution_scale, args
                        )
                    )
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.test_cameras, resolution_scale, args
            )
            self.full_test_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.full_test_cameras, resolution_scale, args
            )

        self._write_camera_json(include_test_cameras)

        self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
        if args.pretrained_ply:
            try:
                self.gaussians.load_ply(args.pretrained_ply)
                print(f"Initialized Gaussians from pretrained PLY: {args.pretrained_ply}")
            except Exception as exc:
                print("Warning: failed to load pretrained PLY '{}': {}".format(args.pretrained_ply, exc))
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
        self.comap_yx = torch.from_numpy(comap_yx.astype(np.float32)).to(device)
        self.max_overlap = multiplexing.get_max_overlap(
            self.comap_yx, self.n_multiplexed_images, H, W
        )
        self.multiplexed_gt = self._load_ground_truth(H, W)
        # Keep test evaluation single-view; no multiplexed composites for these splits.
        self.multiplexed_test_gt = {}
        self.multiplexed_full_test_gt = {}

    def _load_ground_truth(
        self,
        H: int,
        W: int,
        scale: float = 1.0,
        camera_groups: Optional[Dict[int, List[Camera]]] = None,
    ) -> Dict[int, torch.Tensor]:
        gt: Dict[int, torch.Tensor] = {}
        groups = (
            camera_groups if camera_groups is not None else self.getTrainCameras(scale)
        )
        for view_idx, cam_list in groups.items():
            if not isinstance(view_idx, int):
                continue

            cam_list = sorted(cam_list, key=lambda c: c.uid)
            imgs = [
                cam.original_image.to("cpu", dtype=torch.float32) for cam in cam_list
            ]
            gt[view_idx] = multiplexing.generate(
                imgs,
                self.comap_yx,
                self.dim_lens_lf_yx,
                self.n_multiplexed_images,
                H,
                W,
                self.max_overlap,
            ).to(dtype=torch.float32)
            iio.imwrite(
                os.path.join(self.model_path, f"gt_view_{view_idx}.png"),
                (gt[view_idx].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"),
            )
        return gt

    def _write_camera_json(self, include_test: bool = False) -> None:
        """Export camera poses in NeRF transforms.json format."""

        if not getattr(self, "scene_info", None):
            return

        output_path = os.path.join(self.model_path, "transforms.json")
        transforms_data, _ = write_camera_json(
            self.scene_info,
            output_path,
            include_test=include_test,
            object_center=self.object_center,
        )
        if transforms_data is None:
            print("Warning: No train cameras available to write transforms.json")
