import os
import json
from argparse import ArgumentParser
from re import split
import numpy as np
from tqdm import tqdm
import imageio
from PIL import Image
import jax
import random
import math
import imageio.v2 as imageio
np.random.seed(0)
# from jax.config import config 
# config.update("jax_debug_nans", True)
# from jax.experimental.host_callback import id_print
# from torch.utils.tensorboard import SummaryWriter
from functools import partial
from skimage.transform import resize
#test TV

CONST_LR_RGB = 150 # MODIFY ORG 150
CONST_LR_SIGMA = 51.5 #MODIFY ORG 51.5

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Total >tmp')
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >tmp2')
    memory_total = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    memory_used = [int(x.split()[2]) for x in open('tmp2', 'r').readlines()]
    memory_available = [x-y for x, y in zip(memory_total, memory_used)]
    return np.argmax(memory_available)

gpu = get_freer_gpu()
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
print(f'gpu is {gpu}')

# Import jax only after setting the visible gpu
import jax
import jax.numpy as jnp
import plenoxel
# from jax.ops import index, index_update, index_add
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)
if __name__ != "__main__":
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.001'


flags = ArgumentParser()


flags.add_argument(
    "--data_dir", '-d',
    type=str,
    default='./nerf/data/nerf_synthetic/',
    help="Dataset directory e.g. nerf_synthetic/"
)
flags.add_argument(
    "--expname",
    type=str,
    default="experiment",
    help="Experiment name."
)
flags.add_argument(
    "--scene",
    type=str,
    default='lego',
    help="Name of the synthetic scene."
)
flags.add_argument(
    "--log_dir",
    type=str,
    default='jax_logs/',
    help="Directory to save outputs."
)
flags.add_argument(
    "--resolution",
    type=int,
    default=256,
    help="Grid size."
)
flags.add_argument(
    "--ini_rgb",
    type=float,
    default=0.0,
    help="Initial harmonics value in grid."
)
flags.add_argument(
    "--ini_sigma",
    type=float,
    default=0.1,
    help="Initial sigma value in grid."
)
flags.add_argument(
    "--radius",
    type=float,
    default=1.3,
    help="Grid radius. 1.3 works well on most scenes, but ship requires 1.5"
)
flags.add_argument(
    "--harmonic_degree",
    type=int,
    default=2,
    help="Degree of spherical harmonics. Supports 0, 1, 2, 3, 4."
)
flags.add_argument(
    '--num_epochs',
    type=int,
    default=1,
    help='Epochs to train for.'
)
flags.add_argument(
    '--render_interval',
    type=int,
    default=40,
    help='Render images during test/val step every x images.'
)
flags.add_argument(
    '--val_interval',
    type=int,
    default=1,
    help='Run test/val step every x epochs.'
)
flags.add_argument(
    '--lr_rgb',
    type=float,
    default=None,
    help='SGD step size for rgb. Default chooses automatically based on resolution.'
    )
flags.add_argument(
    '--lr_sigma',
    type=float,
    default=None,
    help='SGD step size for sigma. Default chooses automatically based on resolution.'
    )
flags.add_argument(
    '--physical_batch_size',
    type=int,
    default=4000,
    help='Number of rays per batch, to avoid OOM.'
    )
flags.add_argument(
    '--logical_batch_size',
    type=int,
    default=4000,
    help='Number of rays per optimization batch. Must be a multiple of physical_batch_size.'
    )
flags.add_argument(
    '--jitter',
    type=float,
    default=0.0,
    help='Take samples that are jittered within each voxel, where values are computed with trilinear interpolation. Parameter controls the std dev of the jitter, as a fraction of voxel_len.'
)
flags.add_argument(
    '--uniform',
    type=float,
    default=0.5,
    help='Initialize sample locations to be uniformly spaced at this interval (as a fraction of voxel_len), rather than at voxel intersections (default if uniform=0).'
)
flags.add_argument(
    '--occupancy_penalty',
    type=float,
    default=0.0,
    help='Penalty in the loss term for occupancy; encourages a sparse grid.'
)
flags.add_argument(
    '--reload_epoch',
    type=int,
    default=None,
    help='Epoch at which to resume training from a saved model.'
)
flags.add_argument(
    '--save_interval',
    type=int,
    default=1,
    help='Save the grid checkpoints after every x epochs.'
)
flags.add_argument(
    '--prune_epochs',
    type=int,
    nargs='+',
    default=[],
    help='List of epoch numbers when pruning should be done.'
)
flags.add_argument(
    '--prune_method',
    type=str,
    default='weight',
    help='Weight or sigma: prune based on contribution to training rays, or opacity.'
)
flags.add_argument(
    '--prune_threshold',
    type=float,
    default=0.001,
    help='Threshold for pruning voxels (either by weight or by sigma).'
)
flags.add_argument(
    '--split_epochs',
    type=int,
    nargs='+',
    default=[],
    help='List of epoch numbers when splitting should be done.'
)
flags.add_argument(
    '--interpolation',
    type=str,
    default='trilinear',
    help='Type of interpolation to use. Options are constant, trilinear, or tricubic.'
)
flags.add_argument(
    '--nv',
    action='store_true',
    help='Use the Neural Volumes rendering formula instead of the Max (NeRF) rendering formula.'
)
flags.add_argument(
    '--pairs_file',
    type=str,
    default='pairs_conv.npy',
    help='group of images index to multiplexed.'
)
flags.add_argument(
    '--tv_rgb',
    type=float,
    default=1e-3,
    help=''
)
flags.add_argument(
    '--tv_sigma',
    type=float,
    default=1e-4,
    help=''
)
flags.add_argument(
    '--tv_norm',
    type=int,
    default=1,
    help=''
)
flags.add_argument(
    '--n_views',
    type=int,
    default=1,
    help='number of views to build the grid'
)
flags.add_argument(
    '--n_mtp',
    type=int,
    default=1,
    help='multiplexing'
)
flags.add_argument(
    '--eval_only',
    action='store_true',
    help='evaluation only, no training'
)
flags.add_argument(
    '--downsampling_with_mtp',
    action='store_true',
    help='factor to decrease resolution'
)
flags.add_argument(
    '--d_lenses',
    type=int,
    default=2,
    help='2 for multiplex and 4 for non_multiplexing'
)
flags.add_argument(
    '--train_json',
    type=str,
    default='multilens_16',
    help='name of transform json file'
)
flags.add_argument(
    '--max_per_pixel',
    type=int,
    default=1,
    help='max multiplexing per pixel'
)
flags.add_argument(
    '--num_lens',
    type=int,
    default=1,
    help="number of minilens"
)
flags.add_argument(
    '--dim_lens',
    type=int,
    default=800,
    help="number of pixel that 1 minilen map to"
)
flags.add_argument(
    '--black_bkgd',
    action='store_true',
    help="enable black background"
)
FLAGS = flags.parse_args()
data_dir = FLAGS.data_dir + FLAGS.scene
radius = FLAGS.radius
# writer = SummaryWriter(log_dir=f'/home/vitran/plenoxels/runs/{FLAGS.log_dir}{FLAGS.expname}/tv{FLAGS.tv_rgb}_{FLAGS.tv_sigma}/tv_norm{FLAGS.tv_norm}')


def get_data(root, stage):
    all_c2w = []
    all_gt = []

    data_path = root #os.path.join(root, stage)
    data_json = os.path.join(root, 'transforms_' + stage + '.json')
    print('LOAD DATA', data_path)
    j = json.load(open(data_json, 'r'))

    for frame in tqdm(j['frames']):
#         print(frame['file_path']) 
        fpath = os.path.join(data_path, frame['file_path'] + '.png')
#         fpath = os.path.join(data_path, os.path.basename(frame['file_path']) + '.png')
        c2w = frame['transform_matrix']
        im_gt = imageio.imread(fpath).astype(np.float32) / 255.0
        im_gt = im_gt[..., :3] * im_gt[..., 3:] + (1.0 - im_gt[..., 3:])
        all_c2w.append(c2w)
        all_gt.append(im_gt)
    focal = 0.5 * all_gt[0].shape[1] / np.tan(0.5 * j['camera_angle_x'])
    all_gt = np.asarray(all_gt)
    all_c2w = np.asarray(all_c2w)
    return focal, all_c2w, all_gt

    


if __name__ == "__main__":
    focal, train_c2w, train_gt = get_data(data_dir, f"{FLAGS.train_json}")
    test_focal, test_c2w, test_gt = get_data(data_dir, "test")
    print(f"Multiplex {FLAGS.pairs_file[10]} img_size {train_gt.shape[1]} {train_gt.shape[2]}")
#     test_focal, test_c2w, test_gt = get_data(data_dir, f"test{FLAGS.n_views}")
    assert focal == test_focal
    H, W = train_gt[0].shape[:2]
    n_train_imgs = len(train_c2w)
    n_test_imgs = len(test_c2w)


log_dir = FLAGS.log_dir + FLAGS.expname
os.makedirs(log_dir, exist_ok=True)
os.makedirs(os.path.join(log_dir, 'output_grid'), exist_ok=True) #MODIFY making output_grid
MAX_PER_PIXEL = FLAGS.max_per_pixel #10
NUM_LENS = FLAGS.num_lens #16
white_bkgd = not FLAGS.black_bkgd
print('white background', white_bkgd)

with open(os.path.join(log_dir,'command.txt'), 'w') as f: #MODIFY saving command line
    json.dump(FLAGS.__dict__, f, indent=2)

log_file = open(os.path.join(log_dir, 'log.txt'), 'a')
log_file.write(str(FLAGS))

automatic_lr = False
if FLAGS.lr_rgb is None or FLAGS.lr_sigma is None:
    automatic_lr = True
    FLAGS.lr_rgb = CONST_LR_RGB * (FLAGS.resolution ** 1.75) 
    FLAGS.lr_sigma = CONST_LR_SIGMA * (FLAGS.resolution ** 2.37) 
    print('lr_rgb ', FLAGS.lr_rgb)
    print('lr_sigma', FLAGS.lr_sigma)


if FLAGS.reload_epoch is not None:
    reload_dir = os.path.join(log_dir, f'epoch_{FLAGS.reload_epoch}')
    print(f'Reloading the grid from {reload_dir}')
    data_dict = plenoxel.load_grid(dirname=reload_dir, sh_dim = (FLAGS.harmonic_degree + 1)**2)
else:
    print(f'Initializing the grid')
    data_dict = plenoxel.initialize_grid(resolution=FLAGS.resolution, ini_rgb=FLAGS.ini_rgb, ini_sigma=FLAGS.ini_sigma, harmonic_degree=FLAGS.harmonic_degree)


# low-pass filter the ground truth image so the effective resolution matches twice that of the grid
def lowpass(gt, resolution):
    if gt.ndim > 3:
        print(f'lowpass called on image with more than 3 dimensions; did you mean to use multi_lowpass?')
    H = gt.shape[0]
    W = gt.shape[1]
    im = Image.fromarray((np.squeeze(np.asarray(gt))*255).astype(np.uint8))
    im = im.resize(size=(resolution*2, resolution*2))
    im = im.resize(size=(H, W))
    return np.asarray(im) / 255.0


# low-pass filter a stack of images where the first dimension indexes over the images
def multi_lowpass(gt, resolution):
#     print('inside multi lowpass ', gt.shape, resolution)
    if gt.ndim <= 3:
        print(f'multi_lowpass called on image with 3 or fewer dimensions; did you mean to use lowpass instead?')
    H = gt.shape[-3]
    W = gt.shape[-2]
    clean_gt = np.copy(gt)
    for i in range(len(gt)):
        im = Image.fromarray(np.squeeze(gt[i,...] * 255).astype(np.uint8))
        im = im.resize(size=(resolution*2, resolution*2))
        im = im.resize(size=(H, W))
        im = np.asarray(im) / 255.0
        clean_gt[i,...] = im
    return clean_gt

def get_loss(data_dict, c2w, gt, H, W, focal, resolution, radius, harmonic_degree, jitter, uniform, key, sh_dim, occupancy_penalty, interpolation, nv):
    rays = plenoxel.get_rays(H, W, focal, c2w)
    rgb1, disp1, acc1, weights1, voxel_ids1 = plenoxel.render_rays(data_dict, rays[:,0,:,:], resolution, key, radius, harmonic_degree, jitter, uniform, interpolation, nv)#MODIFY individual rgb of each image
    rgb2, disp2, acc2, weights2, voxel_ids2 = plenoxel.render_rays(data_dict, rays[:,1,:,:], resolution, key, radius, harmonic_degree, jitter, uniform, interpolation, nv) #MODIFY individual rgb of each image
#     rgb, disp, acc, weights, voxel_ids = plenoxel.render_rays(data_dict, rays, resolution, key, radius, harmonic_degree, jitter, uniform, interpolation, nv)
    mse = jnp.mean((rgb1+rgb2 - lowpass(gt, resolution))**2)
    indices, data = data_dict
    mse /= float(rays[0].shape[1]) #MODIFY divide by k images
    loss = mse + occupancy_penalty * jnp.mean(jax.nn.relu(data[-1])) 
    return loss

@partial(jax.jit, static_argnums=(1))
def tv_helper(data_dict, resolution, keys):
    voxel_ids1 = jax.random.randint(keys, [int((resolution**3)*0.01)] ,0,resolution**3)
    neighbors = jax.vmap(lambda idx: plenoxel.get_neighbors(idx, resolution))(voxel_ids1)
    neighbors = jax.vmap(lambda idx: plenoxel.vectorize(idx, resolution))(neighbors)

    neighbors = jnp.transpose(neighbors,[0,2,1])
    center_ids, neighbor_ids = neighbors[:,0,:], neighbors.take(jnp.array([1,3,5]), axis=1) 
    neighbor_data = plenoxel.grid_lookup(neighbor_ids[...,0],neighbor_ids[...,1], neighbor_ids[...,2],data_dict) 
    center_data = plenoxel.grid_lookup(center_ids[...,0],center_ids[...,1], center_ids[...,2],data_dict) 
    center_data3 = [jnp.stack([center_data[i]] * neighbor_ids.shape[1], axis=1) for i in range(len(center_data))] 
    if FLAGS.tv_norm==1:
        tv_rgb = [jnp.mean(jnp.abs(n.flatten() - c.flatten() )) for n, c in zip(neighbor_data[:-1], center_data3[:-1])] #l1 norm
    elif FLAGS.tv_norm==2:
        tv_rgb = [jnp.sqrt(jnp.sum( (n.flatten() - c.flatten())**2 )+1e-7) for n, c in zip(neighbor_data, center_data3)] #l2 norm        
    tv_rgb = sum(tv_rgb)/len(tv_rgb)
    return tv_rgb

def get_comap():
    # sensor parameters:
    dim_yx = [800, 800] #[800, 800]
    dx = 0.020  # pixel size in mm

    # microlens array
    if math.isqrt(NUM_LENS)**2 == NUM_LENS:
        num_lenses_yx = [int(math.sqrt(NUM_LENS)),int(math.sqrt(NUM_LENS))] #[1,1]#[4,4] #[6, 8]
    else:
        num_lenses_yx = [2,1]
    print('micorlens array', num_lenses_yx)
        
    d_lens_sensor_lf = 10  # distance between lens array and sensor when no multiplexing (lightfield), in mm
    dim_lens_lf_yx = [FLAGS.dim_lens,FLAGS.dim_lens]#[80,80] #[100, 100]  # number of pixels corresponding to a microlens at the lightfield situation
    d_lenses = FLAGS.d_lenses  # distance between the centers of adjacent microlenses, in mm
    
    d_lens_sensor = 10  # this is the value to change for more or less multiplexing

    lenses_loc_yx = np.meshgrid((np.arange(num_lenses_yx[0]) - (num_lenses_yx[0]-1)/2) * d_lenses,
                                (np.arange(num_lenses_yx[1]) - (num_lenses_yx[1]-1)/2) * d_lenses, indexing='ij')
    lenses_loc_yx = np.array(lenses_loc_yx).reshape(2, np.prod(num_lenses_yx)).transpose()

    dim_lens_yx = [dim_lens_lf_yx[0] / d_lens_sensor_lf * d_lens_sensor, dim_lens_lf_yx[1] / d_lens_sensor_lf * d_lens_sensor]
    dim_lens_yx = [dim_lens_yx[0] - dim_lens_yx[0]%2, dim_lens_yx[1] - dim_lens_yx[1]%2]  # assuming dim_lens_yx is even
    lens_sensor_ind_yx = np.array(np.meshgrid(np.arange(dim_lens_yx[0]), np.arange(dim_lens_yx[1]), indexing='ij')).transpose((1, 2, 0))

    sensor_pixel_loc_y = (np.arange(dim_yx[0]) - dim_yx[0]/2) * dx
    sensor_pixel_loc_x = (np.arange(dim_yx[1]) - dim_yx[1]/2) * dx

    comap_yx = -np.ones((len(lenses_loc_yx), dim_yx[0], dim_yx[1], 2))  

    for i, lens_loc_yx in enumerate(lenses_loc_yx):
        center_index_yx = [np.argmin(np.abs(lens_loc_yx[0] - sensor_pixel_loc_y)), np.argmin(np.abs(lens_loc_yx[1] - sensor_pixel_loc_x))]
        start_index_sensor_yx = [np.maximum(0, center_index_yx[0] - dim_lens_yx[0]//2).astype(int),
                                 np.maximum(0, center_index_yx[1] - dim_lens_yx[1]//2).astype(int)]
        end_index_sensor_yx = [np.minimum(dim_yx[0], center_index_yx[0] + dim_lens_yx[0]//2).astype(int),
                               np.minimum(dim_yx[1], center_index_yx[1] + dim_lens_yx[1]//2).astype(int)]  

        start_index_lens_yx = [int(dim_lens_yx[0]//2 - center_index_yx[0] + start_index_sensor_yx[0]), 
                               int(dim_lens_yx[1]//2 - center_index_yx[1] + start_index_sensor_yx[1])]
        end_index_lens_yx = [int(dim_lens_yx[0]//2 - center_index_yx[0] + end_index_sensor_yx[0]), 
                             int(dim_lens_yx[1]//2 - center_index_yx[1] + end_index_sensor_yx[1])]

        comap_yx[i, start_index_sensor_yx[0]:end_index_sensor_yx[0], start_index_sensor_yx[1]:end_index_sensor_yx[1], :] = lens_sensor_ind_yx[start_index_lens_yx[0]:end_index_lens_yx[0], start_index_lens_yx[1]:end_index_lens_yx[1],:]
    return comap_yx


def get_rays_per_pixel(H,W, comap_yx):
    per_pixel = np.zeros((800,800,MAX_PER_PIXEL,3)).astype(np.uint32)
    mask = np.zeros((800,800,MAX_PER_PIXEL)).astype(float)
    cnt = np.zeros((800,800)).astype(np.uint8)
    for a in range(800):
        for b in range(800):
            for l in range(NUM_LENS):
                x=comap_yx[l, b, a, 1]
                y=comap_yx[l, b, a, 0]

                if x!=-1 and y!=-1:
                    per_pixel[a,b, cnt[a,b]] = np.array([x,y, l])
                    mask[a,b, cnt[a,b]] = 1.
                    cnt[a,b]+=1
    # assert per_pixel[649,0,0,0] == 799, per_pixel[649,0,0]
    return per_pixel, mask  
#MODIFY to cal loss of k images
def get_loss_rays(data_dict, rays, gt, resolution, radius, harmonic_degree, jitter, uniform, key, sh_dim, occupancy_penalty, interpolation, nv):
    #MODIFY rays is [batch_ro, batch_rd], each has size [N,k,x]
    rgb_total = gt_total = jnp.zeros(gt[:,0,:].shape)
    
    for k in range(rays[0].shape[1]):
        rgb, disp, acc, weights, voxel_ids = plenoxel.render_rays(data_dict, [rays[0][:,k,:], rays[1][:,k,:]], resolution, key, radius, harmonic_degree, jitter, uniform, interpolation, nv)
        rgb_total += rgb
        gt_total += gt[:,k,:]
    mse = jnp.mean((rgb_total - gt_total)**2) #MODIFY mse on sum of image pair
    mse /= float(rays[0].shape[1]) #MODIFY divide by k images
    indices, data = data_dict
    loss = mse + occupancy_penalty * jnp.mean(jax.nn.relu(data[-1]))
    return loss


def get_rays_np(H, W, focal, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32) + 0.5, np.arange(H, dtype=np.float32) + 0.5, indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def render_pose_rays(data_dict, c2w, H, W, focal, resolution, radius, harmonic_degree, jitter, uniform, key, sh_dim, batch_size, interpolation, nv):
    rays_o, rays_d = get_rays_np(H, W, focal, c2w)
    rays_o = np.reshape(rays_o, [-1,3])
    rays_d = np.reshape(rays_d, [-1,3])
    rgbs = []
    disps = []
    for i in range(int(np.ceil(H*W/batch_size))):
        start = i*batch_size
        stop = min(H*W, (i+1)*batch_size)
        if jitter > 0:
            rgbi, dispi, acci, weightsi, voxel_idsi = jax.lax.stop_gradient(plenoxel.render_rays(data_dict, (rays_o[start:stop], rays_d[start:stop]), resolution, key[start:stop], radius, harmonic_degree, jitter, uniform, interpolation, nv))
        else:
            rgbi, dispi, acci, weightsi, voxel_idsi = jax.lax.stop_gradient(plenoxel.render_rays(data_dict, (rays_o[start:stop], rays_d[start:stop]), resolution, None, radius, harmonic_degree, jitter, uniform, interpolation, nv))
        rgbs.append(rgbi)
        disps.append(dispi)
    rgb = jnp.reshape(jnp.concatenate(rgbs, axis=0), (H, W, 3))
    disp = jnp.reshape(jnp.concatenate(disps, axis=0), (H, W))
    return rgb, disp, None, None


def run_test_step(i, data_dict, test_c2w, test_gt, H, W, focal, FLAGS, key, name_appendage=''):
    print('Evaluating')
    sh_dim = (FLAGS.harmonic_degree + 1)**2
    tpsnr = 0.0
#     pick = random.choices(list(range(200)), k=5)

    for j, (c2w, gt) in tqdm(enumerate(zip(test_c2w, test_gt))):    
        rgb, disp, _, _ = render_pose_rays(data_dict, c2w, H, W, focal, FLAGS.resolution, radius, FLAGS.harmonic_degree, FLAGS.jitter, FLAGS.uniform, key, sh_dim, FLAGS.physical_batch_size, FLAGS.interpolation, FLAGS.nv)
        mse = jnp.mean((rgb - gt)**2)
        psnr = -10.0 * np.log(mse) / np.log(10.0)
        tpsnr += psnr

#         MODIFY TO SAVE ALL TEST IMAGES ON GRID
        if (FLAGS.render_interval > 0 and j % FLAGS.render_interval == 0) or (j in [10,41,62,120,136,172]):
#         if j in pick:
            disp3 = jnp.concatenate((disp[...,jnp.newaxis], disp[...,jnp.newaxis], disp[...,jnp.newaxis]), axis=2)
            vis = jnp.concatenate((gt, rgb, disp3), axis=1)
            vis = np.asarray((vis * 255)).astype(np.uint8)
            imageio.imwrite(f"{log_dir}/output_grid/{j:04}_{i:04}{name_appendage}.png", vis)
            del rgb, disp
    tpsnr /= n_test_imgs
    log_file.writelines(f'epoch {i}')
    log_file.writelines(f'psnr {tpsnr}')
#     writer.add_scalar('psnr', tpsnr, i)
    return tpsnr


def update_grid(old_grid, lr, grid_grad):
    return old_grid[i] + -1 * lr * grid_grad
    


def update_grids(old_grid, lrs, grid_grad):
    for i in range(len(old_grid)):
        old_grid[i] += -1 * lrs[i] * grid_grad[i]
    return old_grid


def lens_to_offset(l):
    return l//4, l%4

if FLAGS.physical_batch_size is not None:
    print(f'precomputing all the training rays')
    # Precompute all the training rays and shuffle them
    rays = []
    # print(train_c2w.shape)
    for view in range(train_c2w.shape[0]):
        rays_list = []
        for p in range(train_c2w.shape[1]): # train_c2w[view]: #,:,:3,:4]:
            p = train_c2w[view,p,:3,:4]
            rays_list.append(get_rays_np(H, W, focal, p))
        rays_list = np.stack(np.array(rays_list),0)
        rays.append(rays_list)
    rays = np.stack(rays,0) # rays.shape [N, NUM_LENS, ro+rd, H, W, 3]
    print('rays shape', rays.shape)
    # rays = np.stack([get_rays_np(H, W, focal, p) for p in train_c2w[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
    vis = train_gt[1,:,:,:]
    imageio.imwrite(f"{log_dir}/output_grid/sample_training_image_test_{1}.png", vis)
    
    temp = multi_lowpass(train_gt[:,None,None], FLAGS.resolution)
    temp = np.broadcast_to(temp, (10, 16,1,800,800,3))
    print(temp.shape)
    rays_rgb = np.concatenate([rays, temp.astype(float)], 2) # [N,num_lens, ro+rd+rgb, H, W,   3]
    rays_rgb = np.transpose(rays_rgb, [0,3,4,1,2,5]) # [N, H, W,MAX_PER_PIXEL ro+rd+rgb, 3]
    
    vis = rays_rgb[9,:,:,10,2,:]
    imageio.imwrite(f"{log_dir}/output_grid/sample_training_image_test_{1}.png", vis)

    print('shape rays_rgb', rays_rgb.shape)
    
    comap_yx = get_comap()
    maps_pixel_to_rays, mask = get_rays_per_pixel(H,W,comap_yx) #shape (H,W,MAX_PER_PIXEL,3), (H,W,MAX_PER_PIXEL)
    mask = jnp.expand_dims(mask, 0)
    mask = jnp.broadcast_to(mask, (10, mask.shape[1], mask.shape[2], mask.shape[3])) #mask.shape [10,800,800,MAX_PER_PIXEL]
    
    new_rays_rgb = np.zeros((800,800,10,MAX_PER_PIXEL, 3,3))
    for i_index in range(H):
        for j_index in range(W):
            per_pixel_lst = []
            
            for cnt_rays in range(maps_pixel_to_rays.shape[2]): # over MAX_PER_PIXEL
                x_index = maps_pixel_to_rays[i_index, j_index,cnt_rays,0]#*4 #height
                y_index = maps_pixel_to_rays[i_index, j_index,cnt_rays,1]#*4 #width
                l_index = maps_pixel_to_rays[i_index, j_index,cnt_rays,2] #lens
                rgb[i_index, j_index] += sub_lens[l_index, x_index, y_index,:]
                
                y_offset, x_offset = lens_to_offset(l_index)

                per_pixel_lst.append(rays_rgb[:,x_index+x_offset*200, y_index+y_offset*200,l_index,:,:]) 
                
                # per_pixel_lst.append(rays_rgb[:,x_index, y_index,l_index,:,:]) #each dim [N, ro+rd+rgb, 3]
                # new_rays_rgb[:,i_index, j_index,cnt_rays,:,:] = rays_rgb[:,x_index, y_index,l_index,:,:]
                # print(i_index, j_index,x_index+x_offset*200, y_index+y_offset*200,l_index)
            new_rays_rgb[i_index, j_index] = np.stack(per_pixel_lst, axis=1) #[N, MAX_PER_PIXEL, ro+rd+rgb, 3]
    print('new ray shape', new_rays_rgb.shape)
    rays_rgb = jnp.transpose(new_rays_rgb, [2,0,1,3,4,5])# from [H,W,N,16,ro+rd+rgb,3] to [N,H,W,16, ro+rd+rgb, 3]
    # for o in range(10):
    #     vis = np.asarray(rays_rgb[o,:,:,5,2,:] * 255).astype(np.uint8)
    #     imageio.imwrite(f"{log_dir}/output_grid/sample_training_image_{o}.png", vis)
    

#     rays_rgb = np.stack([np.take(rays_rgb, pairs[t], axis=0) for t in range(pairs.shape[0])]) #MODIFY pairing imgs
#     print('after conv pairing rays_rgb shape ', rays_rgb.shape)
    rays_rgb = np.reshape(rays_rgb,[-1, MAX_PER_PIXEL, 3, 3] ) #MODIFY  [(N-1)*H*W, k, ro+rd+rgb, 3] 
    rays_rgb = rays_rgb.take(np.random.permutation(rays_rgb.shape[0]), axis=0)

    print('shape of rays_rgb ', rays_rgb.shape)


print(f'generating random keys')
rootkeys = jax.random.PRNGKey(0)
split_keys_partial = jax.vmap(jax.random.split, in_axes=0, out_axes=0)
split_keys = jax.vmap(split_keys_partial, in_axes=1, out_axes=1)
if FLAGS.physical_batch_size is None:
    keys = jax.vmap(jax.vmap(jax.random.PRNGKey, in_axes=0, out_axes=0), in_axes=1, out_axes=1)(jnp.reshape(jnp.arange(800*800), (800,800)))
else: 
    keys = jax.vmap(jax.random.PRNGKey, in_axes=0, out_axes=0)(jnp.arange(FLAGS.physical_batch_size))
render_keys = jax.vmap(jax.random.PRNGKey, in_axes=0, out_axes=0)(jnp.arange(800*800))
if FLAGS.jitter == 0:
    render_keys = None
    keys = None

cnt=0
def main():
    global rays_rgb, keys, render_keys, data_dict, FLAGS, radius, train_c2w, train_gt, test_c2w, test_gt, automatic_lr , rootkeys, cnt
    start_epoch = 0
    sh_dim = (FLAGS.harmonic_degree + 1)**2
    if FLAGS.reload_epoch is not None:
        start_epoch = FLAGS.reload_epoch + 1
    if np.isin(FLAGS.reload_epoch, FLAGS.prune_epochs):
        data_dict = plenoxel.prune_grid(data_dict, method=FLAGS.prune_method, threshold=FLAGS.prune_threshold, train_c2w=train_c2w, H=H, W=W, focal=focal, batch_size=FLAGS.physical_batch_size, resolution=FLAGS.resolution, key=render_keys, radius=FLAGS.radius, harmonic_degree=FLAGS.harmonic_degree, jitter=FLAGS.jitter, uniform=FLAGS.uniform, interpolation=FLAGS.interpolation)
    if np.isin(FLAGS.reload_epoch, FLAGS.split_epochs):
        data_dict = plenoxel.split_grid(data_dict)
        FLAGS.resolution = FLAGS.resolution * 2
        if automatic_lr:
            FLAGS.lr_rgb = CONST_LR_RGB * (FLAGS.resolution ** 1.75)
            FLAGS.lr_sigma = CONST_LR_SIGMA * (FLAGS.resolution ** 2.37)
            
    if FLAGS.eval_only:
        print('Eval only')
        validation_psnr = run_test_step(start_epoch + 1, data_dict, test_c2w, test_gt, H, W, focal, FLAGS, render_keys)
        print(f'at epoch {start_epoch}, test psnr is {validation_psnr}')
    else:   
        for i in range(start_epoch, FLAGS.num_epochs):
            # Shuffle data before each epoch
            if FLAGS.physical_batch_size is None:
                pass
                # temp = list(zip(train_c2w, train_gt))
                # np.random.shuffle(temp)
                # train_c2w, train_gt = zip(*temp)
            else:
                assert FLAGS.logical_batch_size % FLAGS.physical_batch_size == 0
                # Shuffle rays over all training images
                rays_rgb = rays_rgb.take(np.random.permutation(rays_rgb.shape[0]), axis=0)

            print('epoch', i)
            indices, data = data_dict
            if FLAGS.physical_batch_size is None:
                occupancy_penalty = FLAGS.occupancy_penalty / len(train_c2w)
                for j, (c2w, gt) in tqdm(enumerate(zip(train_c2w, train_gt)), total=len(train_c2w)):
                    if FLAGS.jitter > 0:
                        splitkeys = split_keys(keys)
                        keys = splitkeys[...,0,:]
                        subkeys = splitkeys[...,1,:]
                    else:
                        subkeys = None
                    mse, data_grad = jax.value_and_grad(lambda grid: get_loss((indices, grid), c2w, gt, H, W, focal, FLAGS.resolution, radius, FLAGS.harmonic_degree, FLAGS.jitter, FLAGS.uniform, subkeys, sh_dim, occupancy_penalty, FLAGS.interpolation, FLAGS.nv))(data) 
            else:
                occupancy_penalty = FLAGS.occupancy_penalty / (len(rays_rgb) // FLAGS.logical_batch_size)
                for k in tqdm(range(len(rays_rgb) // FLAGS.logical_batch_size)):
                    logical_grad = None
                    for j in range(FLAGS.logical_batch_size // FLAGS.physical_batch_size):
                        if FLAGS.jitter > 0:
                            splitkeys = split_keys_partial(keys)
                            keys = splitkeys[...,0,:]
                            subkeys = splitkeys[...,1,:]
                        else:
                            subkeys = None
                        effective_j = k*(FLAGS.logical_batch_size // FLAGS.physical_batch_size) + j
                        batch = rays_rgb[effective_j*FLAGS.physical_batch_size:(effective_j+1)*FLAGS.physical_batch_size] # [B, 2+1, 3*?] -> change to [B,2,2+1,3*]
                        batch_rays, target_s = (batch[:,:,0,:], batch[:,:,1,:]), batch[:,:,2,:] #MODIFY from ro+rd, rgb_gt

                        mse, mse_grad = jax.value_and_grad(lambda grid: get_loss_rays((indices, grid), batch_rays, target_s, FLAGS.resolution, radius, FLAGS.harmonic_degree, FLAGS.jitter, FLAGS.uniform, subkeys, sh_dim, occupancy_penalty, FLAGS.interpolation, FLAGS.nv))(data) 

                        splitting = jax.random.split(rootkeys)
                        rootkeys = splitting[0,...]
                        subkey = splitting[1,...]
                        tv, tv_grad = jax.value_and_grad(lambda grid: tv_helper((indices, grid),FLAGS.resolution, subkey))(data)
                        tv_grad = [jnp.where(tv_grad[l] > -1e10,tv_grad[l],jnp.ones_like(tv_grad[l])*-1) for l in range(len(tv_grad))]
                        data_grad = [a + FLAGS.tv_rgb*b for a, b in zip(mse_grad, tv_grad)]                    
    #                     writer.add_scalars('loss_', {'mse':float(mse), 'tv':float(tv)*FLAGS.tv_rgb}, cnt)
    #                     writer.add_scalars('grad_', {'mse':np.linalg.norm(data_grad[0]), 'tv':np.linalg.norm(tv_grad[0])*FLAGS.tv_rgb}, cnt)
                        cnt+=1

                        if FLAGS.logical_batch_size > FLAGS.physical_batch_size:
                            if logical_grad is None:
                                logical_grad = data_grad
                            else:
                                logical_grad = [a + b for a, b in zip(logical_grad, data_grad)]
                            del data_grad
                        del mse, batch, batch_rays, target_s, subkeys, effective_j, mse_grad, tv_grad, tv
                    lrs = [FLAGS.lr_rgb / (FLAGS.logical_batch_size // FLAGS.physical_batch_size)]*sh_dim + [FLAGS.lr_sigma / (FLAGS.logical_batch_size // FLAGS.physical_batch_size)]
                    if FLAGS.logical_batch_size > FLAGS.physical_batch_size:
                        data = update_grids(data, lrs, logical_grad)
                        del logical_grad
                    else:
                        data = update_grids(data, lrs, data_grad)
                        del data_grad, logical_grad
            data_dict = (indices, data)
            del indices, data
            if np.isin(i, FLAGS.prune_epochs):
                data_dict = plenoxel.prune_grid(data_dict, method=FLAGS.prune_method, threshold=FLAGS.prune_threshold, train_c2w=train_c2w, H=H, W=W, focal=focal, batch_size=FLAGS.physical_batch_size, resolution=FLAGS.resolution, key=render_keys, radius=FLAGS.radius, harmonic_degree=FLAGS.harmonic_degree, jitter=FLAGS.jitter, uniform=FLAGS.uniform, interpolation=FLAGS.interpolation)
            if np.isin(i, FLAGS.split_epochs):
                data_dict = plenoxel.split_grid(data_dict)
                FLAGS.lr_rgb = FLAGS.lr_rgb * 3
                FLAGS.lr_sigma = FLAGS.lr_sigma * 3
                FLAGS.resolution = FLAGS.resolution * 2
                if automatic_lr:
                    FLAGS.lr_rgb = CONST_LR_RGB * (FLAGS.resolution ** 1.75)
                    FLAGS.lr_sigma = CONST_LR_SIGMA * (FLAGS.resolution ** 2.37)
                if FLAGS.physical_batch_size is not None:
                    # Recompute all the training rays at the new resolution and shuffle them
                    rays = np.stack([get_rays_np(H, W, focal, p) for p in train_c2w[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
                    rays_rgb = np.concatenate([rays, multi_lowpass(train_gt[:,None], FLAGS.resolution).astype(np.float32)], 1) # [N, ro+rd+rgb, H, W,   3]
                    rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
                    rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
                    rays_rgb = rays_rgb.take(np.random.permutation(rays_rgb.shape[0]), axis=0)

            if i % FLAGS.save_interval == FLAGS.save_interval - 1 or i == FLAGS.num_epochs - 1:
                print(f'Saving checkpoint at epoch {i}')
                plenoxel.save_grid(data_dict, os.path.join(log_dir, f'epoch_{i}'))

            if i % FLAGS.val_interval == FLAGS.val_interval - 1 or i == FLAGS.num_epochs - 1:
                validation_psnr = run_test_step(i + 1, data_dict, test_c2w, test_gt, H, W, focal, FLAGS, render_keys)
                print(f'at epoch {i}, test psnr is {validation_psnr}')
#                 if i == FLAGS.num_epochs - 1:
#                     com = {}
#                     with open("/home/vitran/plenoxels/jax_logs_lego9/combined_runs.json", "r") as read_content:
#                         com = json.load(read_content)
#                         mtp = int(FLAGS.pairs_file[10])
#                         if com.get(f'{FLAGS.n_views}views_downsampling', -1)==-1:
#                             com[f'{FLAGS.n_views}views_downsampling'] = {}
#                         com[f'{FLAGS.n_views}views_downsampling'][mtp] = {}
#                         com[f'{FLAGS.n_views}views_downsampling'][mtp]['tpsnr'] = validation_psnr
#                     with open("/home/vitran/plenoxels/jax_logs_lego9/combined_runs.json", 'w') as outfile:
#                         json.dump(com, outfile)

        
    log_file.close()
if __name__ == "__main__":
    main()