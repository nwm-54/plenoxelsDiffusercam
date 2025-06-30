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
from typing import Dict, List, Optional

import numpy as np
import torch
from arguments import ModelParams
from scene import multiplexing
from scene.cameras import Camera
from scene.dataset_readers_multiviews import (CameraInfo, SceneInfo, fetchPly,
                                              readColmapSceneInfo,
                                              readNerfSyntheticInfo, apply_offset)
from scene.gaussian_model import GaussianModel
from utils.camera_utils import camera_to_JSON, cameraList_from_camInfos
from utils.graphics_utils import getWorld2View2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Scene:
    gaussians : GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, 
                 resolution_scales=[1.0], views_index=[]):
        """
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.gaussians = gaussians

        self.train_cameras: Dict[float, Dict[int, List]] = {}
        self.test_cameras: Dict[float, List] = {}
        self.full_test_cameras: Dict[float, List] = {}

        blender_transform_files = [
            "transforms_train.json",
            "transforms_test.json",
            "transforms_train_gaussian_splatting.json",
            "transforms_train_gaussian_splatting_multiviews.json",
        ]
        is_blender_type = any(os.path.exists(os.path.join(args.source_path, f)) for f in blender_transform_files)

        if os.path.exists(os.path.join(args.source_path, "sparse")): # COLMAP
            print("Found sparse folder, assuming COLMAP dataset")
            self.n_multiplexed_images = 9 # TODO: set the number of multiplexed images
            scene_info = readColmapSceneInfo(args.source_path, args.images, args.eval, use_multiplexing=args.use_multiplexing,
                                             n_multiplexed_images=self.n_multiplexed_images) 
        elif is_blender_type: # Blender
            self.n_multiplexed_images = 16
            print("Found transforms_*.json file, assuming Blender dataset")
            scene_info = readNerfSyntheticInfo(args.source_path, args.white_background, args.eval, 
                                               views_index=views_index, use_multiplexing=args.use_multiplexing,
                                               n_multiplexed_images=self.n_multiplexed_images)
        else:
            raise ValueError(f"Could not infer scene type from source path: {args.source_path}")

        with open(scene_info.ply_path, 'rb') as src_file, \
            open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
            dest_file.write(src_file.read())

        apply_offset(args, scene_info)

        json_cams: List[CameraInfo] = []
        camlist_for_json = []
        if scene_info.train_cameras:
            for cam in scene_info.train_cameras.values():
                camlist_for_json.extend(cam)
        if scene_info.test_cameras: camlist_for_json.extend(scene_info.test_cameras)
        if scene_info.full_test_cameras: camlist_for_json.extend(scene_info.full_test_cameras)        
        for id, cam in enumerate(camlist_for_json):
            json_cams.append(camera_to_JSON(id, cam))
        with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
            json.dump(json_cams, file)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            # Train cameras; Dict[main_view_idx, List[Camera]]
            self.train_cameras[resolution_scale] = {}
            if scene_info.train_cameras:
                for view_idx, cam_infos_for_view in scene_info.train_cameras.items():
                    self.train_cameras[resolution_scale][view_idx] \
                        = cameraList_from_camInfos(cam_infos_for_view, resolution_scale, args)
            self.test_cameras[resolution_scale] \
                = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            self.full_test_cameras[resolution_scale] \
                = cameraList_from_camInfos(scene_info.full_test_cameras, resolution_scale, args)

        self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
        self.multiplexed_gt: Optional[Dict[int, torch.Tensor]] = None

    def save(self, iteration, path: str = "point_cloud.ply"):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        print("saving point cloud at ", point_cloud_path)
        self.gaussians.save_ply(os.path.join(point_cloud_path, path))

    def getTrainCameras(self, scale=1.0) -> Dict[int, List[Camera]]:
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0) -> List[Camera]:
        return self.test_cameras[scale]
    
    def getFullTestCameras(self, scale=1.0) -> List[Camera]:
        return self.full_test_cameras[scale]
    
    def init_multiplexing(self, dls: int, H: int, W: int): 
        comap_yx, dim_lens_lf_yx = multiplexing.get_comap(self.n_multiplexed_images, dls, H, W)
        self.dim_lens_lf_yx = dim_lens_lf_yx
        self.comap_yx = torch.from_numpy(comap_yx).to(device)
        self.max_overlap = multiplexing.get_max_overlap(self.comap_yx, self.n_multiplexed_images, H, W)
        self.multiplexed_gt = self._load_ground_truth(H, W)
    
    def _load_ground_truth(self, H: int, W: int, scale=1.0) -> Dict[int, torch.Tensor]:
        gt: Dict[int, torch.Tensor] = {}
        for view_idx, cam_list in self.getTrainCameras(scale).items():
            if not isinstance(view_idx, int):
                continue

            cam_list = sorted(cam_list, key=lambda c: c.uid)
            imgs = [cam.original_image.to(cam.data_device) for cam in cam_list]
            gt[view_idx] = multiplexing.generate(imgs, self.comap_yx, self.dim_lens_lf_yx, 
                                                 self.n_multiplexed_images, H, W, self.max_overlap)
        return gt

    def _write_camera_json(self):
        json_cams: List[CameraInfo] = []
        id_counter = 0
        for scale, viewdict in self.train_cameras.items():
            for view_idx, cam_list in viewdict.items():
                for cam in cam_list:
                    json_cams.append(camera_to_JSON(id_counter, cam))
                    id_counter += 1
        for scale, cam_list in self.test_cameras.items():
            for cam in cam_list:
                json_cams.append(camera_to_JSON(id_counter, cam))
                id_counter += 1
        for scale, cam_list in self.full_test_cameras.items():
            for cam in cam_list:
                json_cams.append(camera_to_JSON(id_counter, cam))
                id_counter += 1

        with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
            json.dump(json_cams, file)