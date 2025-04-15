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
import sys
import uuid
from argparse import ArgumentParser, Namespace

import torch
import torch.nn.functional as F
from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import network_gui, render
from scene import GaussianModel, Scene, multiplexing
from tqdm import tqdm
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
    
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
print(f"Using device: {device}")
print(f"GPU Name: {torch.cuda.get_device_name(device)}")

import wandb

wandb.login()

def training(dataset: ModelParams, 
             opt: OptimizationParams, 
             pipe: PipelineParams, 
             testing_iterations, 
             saving_iterations, 
             checkpoint_iterations, 
             debug_from, 
             source_path: str, 
             resolution: int, 
             dls: float):
    tb_writer, model_path = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
        
    print("TV weight ", opt.tv_weight)
    print("TV unseen weight ", opt.tv_unseen_weight)
    print("Train Cameras: ", len(scene.getTrainCameras()))
    print("Test Cameras: ", len(scene.getTestCameras()))
    print("Full Test Cameras: ", len(scene.getFullTestCameras()))

    ##SIMILATED data
    H = W = 800 // resolution
    num_lens = 16
    comap_yx, dim_lens_lf_yx = multiplexing.get_comap(num_lens, dls, H, W)
    alpha_map = torch.from_numpy(multiplexing.generate_alpha_map(comap_yx, num_lens, H, W)).float().to(device)
    comap_yx = torch.from_numpy(comap_yx).to(device)
    if 'lego' in source_path:
        input_images = multiplexing.read_images(num_lens, "/home/wl757/multiplexed-pixels/plenoxels/blender_data/lego_gen12/train_multilens_16_black", "59")
    elif 'hotdog' in source_path:
        input_images = multiplexing.read_images(num_lens, "/home/wl757/multiplexed-pixels/plenoxels/blender_data/hotdog/render_5_views", "0")
    else:
        input_images = multiplexing.read_images(num_lens, f"{source_path}/render_5_views", "2")
    gt_image = multiplexing.generate(input_images, comap_yx, dim_lens_lf_yx, num_lens, H, alpha_map)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(0, opt.iterations), desc="Training progress")
    for iteration in range(1, opt.iterations + 1):        
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        image_list = []
        scene_train_cameras = scene.getTrainCameras()
        tv_train_loss = 0.
        tv_loss = 0.
        for j, single_viewpoint in enumerate(scene_train_cameras):
            render_pkg = render(single_viewpoint, gaussians, pipe, bg, scaling_modifier=1.)
            image_raw, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            tv_train_loss += tv_2d(image_raw)
            image_list.append(image_raw)        
            
        #TV viewpoint
        tv_viewpoints = random.choices(scene.getFullTestCameras(), k=1) #scene.getFullTestCameras() #
        # tv_viewpoints = random.choices(scene.getTestCameras(), k=1) #scene.getFullTestCameras() #
        tv_image = None
        for tv_viewpoint in tv_viewpoints:
            tv_image = render(tv_viewpoint, gaussians, pipe, bg)["render"]
            tv_loss += tv_2d(tv_image)
        
        output_image = multiplexing.generate(image_list, comap_yx, dim_lens_lf_yx, num_lens, H, alpha_map)
       
        # Loss
        Ll1 = l1_loss(output_image, gt_image)  #use image_tensor, gt_tensor for non-multiplex
        Ltv = tv_loss / len(tv_viewpoints) * opt.tv_unseen_weight + tv_train_loss / len(scene_train_cameras) * opt.tv_weight
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(output_image, gt_image)) + Ltv
        
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
            training_report(tb_writer, 
                            iteration, 
                            Ll1,
                            Ltv, 
                            loss, 
                            iter_start.elapsed_time(iter_end), 
                            testing_iterations, 
                            scene, 
                            render, 
                            (pipe, background), 
                            tv_image, 
                            model_path, 
                            output_image,
                            gt_image)
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

def training_report(tb_writer, 
                    iteration, 
                    Ll1, 
                    Ltv, 
                    loss, 
                    elapsed, 
                    testing_iterations, 
                    scene : Scene, 
                    renderFunc, 
                    renderArgs, 
                    tv_image, 
                    model_path, 
                    multiplexed_image,
                    true_gt_image=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/tv_loss', Ltv.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
    f = open(f"{model_path}/output.txt", "a")
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'non adj test', 'cameras' : scene.getFullTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx] for idx in range(0,len(scene.getTrainCameras()))]})
        if tv_image is not None:
            tb_writer.add_images("tv_image", tv_image[None], global_step=iteration)
        rendered_trained_images = []
        rendered_gt_images = []
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                # print(config['name'], len(config['cameras']))
                for idx, viewpoint in enumerate(config['cameras']):
                    # if config['name']=="test" :
                    #     print(viewpoint.image_name)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)

                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        # simulation or non multiplex only
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        # if iteration == testing_iterations[0]:
                        tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    
                    if config["name"]=="train":
                        rendered_trained_images.append(image[None])
                        rendered_gt_images.append(gt_image[None])
                                                
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])   

                if config['name'] == 'test':
                    wandb.log(
                    {
                        "epoch": iteration,
                        "psnr_test": psnr_test,
                        "l1_test": l1_test,
                    }
                )
                    
                if config['name'] == 'non adj test':
                    wandb.log(
                    {
                        "epoch": iteration,
                        "psnr_test_full": psnr_test,
                        "l1_test_full": l1_test,
                    }
                )
                
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                f.write("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

                    if config["name"]=="train" and len(rendered_trained_images) > 0:
                        # multiplexed_image = multiplexing.generate(rendered_trained_images, comap_yx, dim_lens_lf_yx, num_lens, H)                        
                        tb_writer.add_images("trained_multiplex", multiplexed_image.unsqueeze(0), global_step=iteration)

        tb_writer.add_images("gt_multiplex", true_gt_image.unsqueeze(0), global_step=iteration)
            
        if tb_writer:
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
    parser.add_argument('--port', type=int, default=6001)
    parser.add_argument('--debug_from', type=int, default=3000)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=list(range(10, 30_000 + 1, 500)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=list(range(2000, 30_000 + 1, 1000)))
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[1_500, 2_000, 2_500, 3000, 4000, 6000, 8000, 10000, 20000, 30000])
    parser.add_argument("--dls", type=int, default = 20)
    parser.add_argument("--device", type=int, default = 0)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    # Initialize system state (RNG)
    safe_state(args.quiet)

    wandb.init()

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(dataset=lp.extract(args), 
             opt=op.extract(args), 
             pipe=pp.extract(args), 
             testing_iterations=args.test_iterations, 
             saving_iterations=args.save_iterations, 
             checkpoint_iterations=args.checkpoint_iterations, 
             debug_from=args.debug_from, 
             source_path=args.source_path, 
             resolution=args.resolution, 
             dls=args.dls)

    # All done
    print("\nTraining complete.")
