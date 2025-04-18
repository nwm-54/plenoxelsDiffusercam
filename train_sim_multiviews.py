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

import os
import random
import numpy as np
import scipy
import scipy.ndimage
import torch
from random import randint
import wandb
import yaml
from utils.loss_utils import l1_loss, ssim, l2_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
import scene as scn
import torch.nn.functional as F

from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import json
import collections
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
    
import torch
import torch.nn.functional as F

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, source_path, resolution, dls, views_index):
    first_iter = 0
    tb_writer, model_path = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, multiviews=views_index)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        
    print("tv weight ", opt.tv_weight)
    ##SIMILATED data
    H = W = 800//resolution
    num_lens = 16
    d_lens_sensor = dls
    MAX_PER_PIXEL =  5 if dls<=16 else 10
    comap_yx, _ = scn.multiplexing.get_comap(num_lens, d_lens_sensor, H, W)
    maps_pixel_to_rays, real_ray_mask, _ = scn.multiplexing.get_rays_per_pixel(H,W, comap_yx,MAX_PER_PIXEL, num_lens)
    multiplexed_mask, pad_mapping, border_minmax = scn.multiplexing.model_output_mask(comap_yx, num_lens, maps_pixel_to_rays, real_ray_mask, H,W, MAX_PER_PIXEL)
    # The code snippet is using the `multiplexing.generate` function from the `scn` module to generate
    # a multiplexed image. It takes the `comap_yx` input, the string "59", a file path
    # "/home/vitran/plenoxels/blender_data/lego_gen12/train_multilens_16_black", the number of lenses
    # `num_lens`, and the height `H` and width `W` of the image as parameters. The function returns
    # two values, with the first one being assigned to `rgb_gt`.
    #lego
    rgb_gts = {}
    for view in views_index:
        if "lego" in source_path:
            rgb_gt, _ = scn.multiplexing.generate(comap_yx, str(view), "/home/wl757/multiplexed-pixels/plenoxels/blender_data/lego_gen12/train_multilens_16_black", num_lens, H, W)
        else:
            rgb_gt, _ = scn.multiplexing.generate(comap_yx, str(view), f"{source_path}/render_5_views", num_lens, H, W)
        rgb_gt = torch.from_numpy(rgb_gt)[:,:,:3].permute(2,0,1).float().cuda()
        rgb_gts[view] = rgb_gt
    # rgb_gt, _ = scn.multiplexing.generate(comap_yx, "2", "/home/vitran/plenoxels/blender_data/chair/render_5_views", num_lens, H, W)
    # rgb_gt, _ = scn.multiplexing.generate(comap_yx, "2", f"{source_path}/render_5_views", num_lens, H, W)
    # rgb_gt /= 4.
    # rgb_gt, _ = scn.multiplexing.generate(comap_yx, "0", "/home/vitran/plenoxels/blender_data/hotdog/render_5_views", num_lens, H, W)
    
    ##### SIMULATION
    ##
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        pair = collections.defaultdict(dict)
        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = []
        model_output = []
        gt_list = []
        negative_images = []
        scene_train_cameras = scene.getTrainCameras() # dict [59] to cameras
       
        tv_train_loss = 0.
        total_l1 = 0.
        for scene_index,train_cameras in scene_train_cameras.items():
            image_list = []
            for j,single_viewpoint in enumerate(train_cameras):
                # print('line 137', j)
                render_pkg = render(single_viewpoint, gaussians, pipe, bg, scaling_modifier=1.)
                image_raw, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                tv_train_loss += tv_2d(image_raw)
                image_raw = scn.multiplexing.generate_single_training(j, border_minmax, comap_yx, image_raw, multiplexed_mask, pad_mapping, maps_pixel_to_rays, H, W  )
                # image_raw = scn.multiplexing.generate_single_training_pinhole_with_mask(image_raw, single_viewpoint)
                # print(j, single_viewpoint.image_name, image_raw.size())
                image_list.append(image_raw)  #or image for multiplexing   
                # tv_train_loss += tv_2d(image_raw)
            # print(len(image_list))
            image_tensor = torch.stack(image_list, dim=0) #for sim
            _output_image = torch.sum(image_tensor, 0) #/MAX_PIXEL
            output_image = _output_image #/torch.max(_output_image)
            gt_image = rgb_gts[scene_index] #/torch.max(_gt_image) FOR real data # rgb_gt #for simulation only #_gt_image #/torch.max(_gt_image) 

            total_l1 += l1_loss(output_image, gt_image) #use image_tensor, gt_tensor for non-multiplex
            
        #TV viewpoint
        tv_viewpoints = random.choices(scene.getFullTestCameras(), k=10) #scene.getFullTestCameras() #
        tv_loss = 0.
        tv_image = None
        for tv_viewpoint in tv_viewpoints:
            render_pkg = render(tv_viewpoint, gaussians, pipe, bg)
            tv_image, tv_viewspace_point_tensor, tv_visibility_filter, tv_radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            tv_loss += tv_2d(tv_image)


        # Loss
        Ll1 = total_l1 / len(scene_train_cameras.keys())
        Ltv = tv_loss/len(tv_viewpoints)*opt.tv_unseen_weight + tv_train_loss/(len(scene_train_cameras)*16)*opt.tv_weight
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(output_image, gt_image)) + tv_loss/len(tv_viewpoints)*opt.tv_weight + Ltv#+ negative_image*1e-4  + l2_loss(img1, img2)*(10*6)
        # Ll1 = l1_loss(image_tensor, gt_tensor) 
        # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image_tensor, gt_tensor)) + tv_loss#+ negative_image*1e-4  + l2_loss(img1, img2)*(10*6)
        
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), pair, tv_image, tv_loss, model_path, gt_image) #gt_image for multiplexing #random.choice(gt_list) for non-multiplexing
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def tv_2d(image):
    # print((torch.pow(image[:,1:,:] - image[:,:-1,:], 2).mean()))
    # print((torch.square(image[:,1:,:] - image[:,:-1,:]).mean() + torch.square(image[:,:,1:] - image[:,:,:-1]).mean()).values)
    return torch.square(image[:,1:,:] - image[:,:-1,:]).mean() + torch.square(image[:,:,1:] - image[:,:,:-1]).mean()
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer, args.model_path

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, pair, tv_image, tv_loss, model_path, true_gt_image=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        # tb_writer.add_scalar('train_loss_patches/tv_loss', (tv_loss*1e-2).item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
    f = open(f"{model_path}/output.txt", "a")
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'non adj test', 'cameras' : scene.getFullTestCameras()}, 
                            #   {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx] for idx in range(0,len(scene.getTrainCameras()))]})
                              {'name': 'train', 'cameras' : [v[3] for k,v in scene.getTrainCameras().items()]})
                            #   {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        if tv_image is not None:
            tb_writer.add_images("tv_image", tv_image[None], global_step=iteration)
        rendered_trained_images = []
        rendered_gt_images = []
        multiplex_image_list = []
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                # print(config['name'], len(config['cameras']))
                for idx, viewpoint in enumerate(config['cameras']):
                    # if config['name']=="test" :
                    #     print(viewpoint.image_name)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    image_small = image #None
                    # if config['name']=="train":
                    #     image_small = scn.multiplexing.generate_single_training_pinhole_with_mask(image[None].squeeze(0), viewpoint)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        #real time only
                        
                        # if image_small is not None:
                        #     tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image_small.unsqueeze(0), global_step=iteration)
                        # simulation or non multiplex only
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        # if iteration == testing_iterations[0]:
                        tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    
                    if config["name"]=="train":
                        rendered_trained_images.append(image[None])
                        rendered_gt_images.append(gt_image[None])
                        
                        multiplex_image_list.append(image_small)
                        
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])   
                
       
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                f.write("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

                    if config["name"]=="train" and len(multiplex_image_list) > 0:
                        multiplex_image = torch.stack(multiplex_image_list, 0)
                        multiplex_image = torch.sum(multiplex_image, 0).unsqueeze(0)
                        multiplex_image /= torch.max(multiplex_image)
                        
                        tb_writer.add_images("trained_multiplex", multiplex_image, global_step=iteration)

        if len(rendered_trained_images)>0:
            rendered_trained_image = torch.stack(rendered_trained_images, 0).squeeze(0)
            rendered_trained_image = torch.sum(rendered_trained_image, 0)
            rendered_trained_image /= torch.max(rendered_trained_image)
            rendered_gt_image = torch.stack(rendered_gt_images, 0).squeeze(0)
            rendered_gt_image = torch.sum(rendered_gt_image, 0)
            rendered_gt_image /= torch.max(rendered_gt_image)
            
            if true_gt_image is not None:
                # true_gt_image_np = np.transpose(true_gt_image.cpu().numpy(), (1,2,0))
                # print(true_gt_image.unsqueeze(0).shape)
                tb_writer.add_images("gt_multiplex", true_gt_image.unsqueeze(0), global_step=iteration)
            else:
                tb_writer.add_images("gt_multiplex", rendered_gt_image, global_step=iteration)
                tb_writer.add_images("trained_multiplex", rendered_trained_image, global_step=iteration)
            
        if tb_writer:
            # tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
    f.close()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6004)
    parser.add_argument('--debug_from', type=int, default=3000)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[10,1_000, 2_000, 3_000,4_000,7000, 7001,7050,8000,10_000, 12_000, 14_000, 18_000,20_000, 27_000, 29_000,30_001,30_010,30_200,30_400,30_600,31_000,32_000,33_000,34_000,35_000,36_000,40_000, 50_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10,7_000, 30_000, 1_000,2_000,3_000,4_000,7000, 7001,7050,8000,10_000, 12_000, 14_000, 18_000,20_000, 27_000, 29_000,30_001,30_010,30_200,30_400,30_600,31_000,32_000,33_000,34_000,35_000,36_000,40_000, 50_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[ 4000, 6000, 8000,10000, 20000,30000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--dls", type=int, default = 20)
    parser.add_argument('--views_index', nargs='+', type=int, default=[59])

    # parser.add_argument("--tv_weight", type=float, default = 1)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)
    print('view index ', args.views_index)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.source_path, args.resolution, args.dls, args.views_index)

    # All done
    print("\nTraining complete.")
