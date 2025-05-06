#
# For simulation images - multi-view
#

import os
import random
import sys
import uuid
from argparse import ArgumentParser, Namespace
from random import randint
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from lpipsPyTorch import lpips
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
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)
print(f"Using device: {device}")
print(f"GPU Name: {torch.cuda.get_device_name(device)}")
wandb.login()

multiview_5_idx = {
    'lego': [50, 59, 60, 70, 90],
    'hotdog': [0, 11, 23, 27, 37],
    'chair': [2, 25, 38, 79, 90],
    'drums': [2, 25, 38, 79, 90],
    'ficus': [2, 25, 38, 79, 90],
    'materials': [2, 25, 38, 79, 90],
    'mic': [2, 25, 38, 79, 90],
    'ship': [2, 25, 38, 79, 90],
}

multiview_3_idx = {
    'lego': [50, 59, 60],
    'hotdog': [0, 11, 23],
    'chair': [2, 25, 38],
    'drums': [2, 25, 38],
    'ficus': [2, 25, 38],
    'materials': [2, 25, 38],
    'mic': [2, 25, 38],
    'ship': [2, 25, 38],
}


def training(dataset: ModelParams, 
             opt: OptimizationParams, 
             pipe: PipelineParams, 
             testing_iterations, 
             saving_iterations, 
             checkpoint_iterations, 
             checkpoint, 
             debug_from, 
             source_path: str, 
             resolution: int, 
             dls: float, 
             num_views: Literal[3, 5],
             size_threshold_arg: int,
             extent_multiplier: float):
    tb_writer, model_path = prepare_output_and_logger(dataset)
    dataset_name = os.path.basename(dataset.source_path).replace("lego_gen12", "lego")
    views_index = multiview_5_idx[dataset_name] if num_views == 5 else multiview_3_idx[dataset_name]
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, multiviews=views_index)
    gaussians.training_setup(opt)
        
    print("Tv weight ", opt.tv_weight)
    print("TV unseen weight ", opt.tv_unseen_weight)
    print("Train cameras:", sum(len(c) for c in scene.getTrainCameras().values()))
    print("Test cameras:", len(scene.getTestCameras()))
    print("Full test cameras:", len(scene.getFullTestCameras()))
    print("Views index:", views_index)

    H = W = 800 // resolution
    num_lens = 16
    comap_yx, dim_lens_lf_yx = multiplexing.get_comap(num_lens, dls, H, W)
    comap_yx = torch.from_numpy(comap_yx).to(device)
    max_overlap = multiplexing.get_max_overlap(comap_yx, num_lens, H, W)
    rgb_gts = {}
    for view in views_index:
        if 'lego' in source_path:
            input_images = multiplexing.read_images(num_lens, "/home/wl757/multiplexed-pixels/plenoxels/blender_data/lego_gen12/train_multilens_16_black", str(view))
        elif 'hotdog' in source_path:
            input_images = multiplexing.read_images(num_lens, "/home/wl757/multiplexed-pixels/plenoxels/blender_data/hotdog/render_5_views", str(view))
        else:
            input_images = multiplexing.read_images(num_lens, f"{source_path}/render_5_views", str(view))
        rgb_gts[view] = multiplexing.generate(input_images, comap_yx, dim_lens_lf_yx, num_lens, H, W, max_overlap)
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(0, opt.iterations), desc="Training progress")
    for iteration in range(1, opt.iterations + 1):        
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = []
        scene_train_cameras = scene.getTrainCameras()
       
        total_loss = 0.
        for scene_index, train_cameras in scene_train_cameras.items():
            tv_train_loss = 0.
            image_list = []
            for j, single_viewpoint in enumerate(train_cameras):
                render_pkg = render(single_viewpoint, gaussians, pipe, bg, scaling_modifier=1.)
                image_raw, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                tv_train_loss += tv_2d(image_raw)
                image_list.append(image_raw)

            output_image = multiplexing.generate(image_list, comap_yx, dim_lens_lf_yx, num_lens, H, W, max_overlap)
            gt_image = rgb_gts[scene_index] #/torch.max(_gt_image) FOR real data # rgb_gt #for simulation only #_gt_image #/torch.max(_gt_image) 

            L_l1 = (1.0 - opt.lambda_dssim) * l1_loss(output_image, gt_image)
            _ssim = opt.lambda_dssim * (1.0 - ssim(output_image, gt_image))
            train_tv = opt.tv_weight * tv_train_loss / len(train_cameras)
            total_loss += L_l1 + _ssim + train_tv
            
        # TV viewpoint
        tv_viewpoints = random.choices(scene.getFullTestCameras(), k=1) # scene.getFullTestCameras()
        tv_loss = 0.
        tv_image = None
        for tv_viewpoint in tv_viewpoints:
            tv_image = render(tv_viewpoint, gaussians, pipe, bg)["render"]
            tv_loss += tv_2d(tv_image)

        # Loss
        unseen_tv = tv_loss / len(tv_viewpoints) * opt.tv_unseen_weight
        loss = total_loss / len(scene_train_cameras.keys()) + unseen_tv
        
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
                            loss, 
                            iter_start.elapsed_time(iter_end), 
                            testing_iterations, 
                            scene, 
                            (pipe, background), 
                            unseen_tv, 
                            model_path,
                            (comap_yx, dim_lens_lf_yx, num_lens, H, W, max_overlap))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = size_threshold_arg if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(max_grad=opt.densify_grad_threshold, 
                                                min_opacity=0.005, 
                                                extent=scene.cameras_extent * extent_multiplier, 
                                                max_screen_size=size_threshold)
                
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
                    loss, 
                    elapsed, 
                    testing_iterations, 
                    scene : Scene, 
                    renderArgs, 
                    unseen_tv_loss, 
                    model_path,
                    multiplexing_args):
    log_dict = {
        'train_loss/total_loss': loss.item(),
        'unseen_tv_loss': unseen_tv_loss.item(),
        'iter_time': elapsed,
    }

    if tb_writer:
        for key, value in log_dict.items():
            tb_writer.add_scalar(key, value, iteration)

    wandb.log(log_dict, step=iteration)
    with open(f"{model_path}/output.txt", "a") as f:
        if iteration in testing_iterations:
            img_dict = {}

            validation_configs = (
                {'name': 'adjacent test camera', 'cameras' : scene.getTestCameras()}, 
                {'name': 'full test camera',     'cameras' : scene.getFullTestCameras()}, 
                # TODO: maybe get all train cameras instead of the 6th one at each viewpoint
                {'name': 'train camera',         'cameras' : [cam for cam_list in scene.getTrainCameras().values() for cam in cam_list]})

            for config in validation_configs:
                cams = config['cameras']
                if not cams: continue

                l1_test, psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0, 0.0
                for idx, viewpoint in enumerate(cams):
                    out = render(viewpoint, scene.gaussians, *renderArgs)["render"]
                    gt = viewpoint.original_image.to(device)

                    if idx < 10:
                        if tb_writer:
                            tb_writer.add_images(f"{config['name']}_{viewpoint.image_name}/render",      out[None], global_step=iteration)
                            tb_writer.add_images(f"{config['name']}_{viewpoint.image_name}/ground_truth", gt[None], global_step=iteration)
                        img_dict[f"{config['name']}_{viewpoint.image_name}/render"] = wandb.Image(out.cpu())
                    
                    l1_test    += l1_loss(out, gt).mean().double()
                    psnr_test  += psnr(out, gt).mean().double()
                    ssim_test  += ssim(out, gt)
                    lpips_test += lpips(out.unsqueeze(0), gt.unsqueeze(0), net_type='vgg').mean().double()
                
                l1_test    /= len(cams)
                psnr_test  /= len(cams)
                ssim_test  /= len(cams)
                lpips_test /= len(cams)

                msg = f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test:.4f} PSNR {psnr_test:.4f} SSIM {ssim_test:.4f} LPIPS {lpips_test:.4f}"
                print(msg); f.write(msg)

                if tb_writer:
                    tb_writer.add_scalar(f"{config['name']}/l1_loss", l1_test, iteration)
                    tb_writer.add_scalar(f"{config['name']}/psnr",    psnr_test, iteration)
                    tb_writer.add_scalar(f"{config['name']}/ssim",    ssim_test, iteration)
                    tb_writer.add_scalar(f"{config['name']}/lpips",   lpips_test, iteration)
                log_dict[f"{config['name']}/l1_loss"] = l1_test.item()
                log_dict[f"{config['name']}/psnr"]    = psnr_test.item()
                log_dict[f"{config['name']}/ssim"]    = ssim_test.item()
                log_dict[f"{config['name']}/lpips"]   = lpips_test.item()

            train_cameras = list(scene.getTrainCameras().values())[0]
            image_list = [render(single_viewpoint, scene.gaussians, *renderArgs)["render"] for single_viewpoint in train_cameras]
            multiplexed_image = multiplexing.generate(image_list, *multiplexing_args)
            if tb_writer:
                tb_writer.add_images("trained_multiplex", multiplexed_image.unsqueeze(0), global_step=iteration)
            img_dict["trained_multiplex"] = wandb.Image(multiplexed_image.cpu())

            total_pts = scene.gaussians.get_xyz.shape[0]
            if tb_writer:
                tb_writer.add_scalar('total_points', total_pts, iteration)
            log_dict['total_points'] = total_pts
            wandb.log({**log_dict, **img_dict}, step=iteration)

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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[10] + list(range(500, 30_000 + 1, 250)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=list(range(2000, 30_000 + 1, 1000)))
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[4000, 6000, 8000, 10000, 20000, 30000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--dls", type=int, default = 20)
    parser.add_argument('--num_views', nargs='+', type=int, choices=[3, 5], default=5)
    parser.add_argument("--size_threshold", type=int, default = 200)
    parser.add_argument("--extent_multiplier", type=float, default = 4.)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)
    dataset_name = os.path.basename(dataset.source_path).replace("lego_gen12", "lego")
    run_name = f"{dataset_name}_multiview{args.num_views}_resolution{800 // dataset.resolution}_dls{args.dls}_tv{opt.tv_weight}_unseen{opt.tv_unseen_weight}"
    if not dataset.model_path:
        dataset.model_path = "/share/monakhova/shamus_data/multiplexed_pixels/output7/" + run_name
    wandb.init(name=run_name)

    
    training(dataset=dataset,
             opt=opt, 
             pipe=pipe,
             testing_iterations=args.test_iterations, 
             saving_iterations=args.save_iterations, 
             checkpoint_iterations=args.checkpoint_iterations, 
             checkpoint=args.start_checkpoint, 
             debug_from=args.debug_from, 
             source_path=args.source_path, 
             resolution=args.resolution, 
             dls=args.dls, 
             num_views=args.num_views,
             size_threshold_arg=args.size_threshold,
             extent_multiplier=args.extent_multiplier)

    # All done
    print("\nTraining complete.")
