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
import random
from typing import Dict, List

from arguments import ModelParams
from scene.cameras import Camera
from scene.dataset_readers_multiviews import (CameraInfo, readColmapSceneInfo,
                                              readNerfSyntheticInfo)
from scene.gaussian_model import GaussianModel
from utils.camera_utils import camera_to_JSON, cameraList_from_camInfos


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

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        print("saving point cloud at ", point_cloud_path)
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0) -> Dict[int, List[Camera]]:
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0) -> List[Camera]:
        return self.test_cameras[scale]
    
    def getFullTestCameras(self, scale=1.0) -> List[Camera]:
        return self.full_test_cameras[scale]