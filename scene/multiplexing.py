import os
import json
from argparse import ArgumentParser
from re import split
import numpy as np
import torch
# from tqdm import tqdm
import imageio as imageio
from PIL import Image
# import jax
import random
import math
np.random.seed(0)
# from functools import partial
from skimage.transform import resize
import csv
import torch.nn.functional as F

SUBIMAGES = [x for x in range(16)]#[1,2,5,6,9,10, 13,14]

# import torchvision.transforms as T

# import jax
# import jax.numpy as jnp
# import plenoxel
# from jax.lib import xla_bridge

# function taken from multiplex_demo_cleanup.jpynb
def get_comap(num_lens, d_lens_sensor, H, W):
#     global dim_lens_lf_yx
    dim_yx = [H,W] 
    dx = 0.020  # pixel size in mm
    # num_lenses_yx = [6,7]
    if math.sqrt(num_lens)**2 == num_lens:
        # print(math.sqrt(num_lens), num_lens)
        num_lenses_yx = [int(math.sqrt(num_lens)),int(math.sqrt(num_lens))] 
    else:
        print('Number of sublens should be a square number')
        assert False

    d_lens_sensor_lf = 10  # distance between lens array and sensor when no multiplexing (lightfield), in mm
    dim_lens_lf_yx = [dim_yx[0]//num_lenses_yx[0], dim_yx[0]//num_lenses_yx[0]] # number of pixels corresponding to a microlens at the lightfield situation
    d_lenses = dim_lens_lf_yx[0]*dx # # distance between the centers of adjacent microlenses, in mm

    # d_lens_sensor = FLAGS.d_lens_sensor  # this is the value to change for more or less multiplexing

    # print(f'd_lenses {d_lenses}, dim_lens_lf_yx {dim_lens_lf_yx}, d_lens_sensor {d_lens_sensor}')

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
    return comap_yx, dim_lens_lf_yx
          
#Function that maps which sublens rays falls on which sensor pixel
def get_rays_per_pixel(H,W, comap_yx, max_per_pixel, num_lens):
    MAX_PER_PIXEL_INIT = 20
    # per_pixel = torch.zeros((H,W,max_per_pixel,3), dtype=torch.int)
    # mask = torch.zeros((H,W,max_per_pixel), dtype=torch.int)
    # cnt_mpp = torch.zeros((H,W), dtype=torch.int)
    
    per_pixel = np.zeros((W,H,max_per_pixel,3)).astype(int)
    mask = np.zeros((W,H,max_per_pixel)).astype(int)
    cnt_mpp = np.zeros((W, H)).astype(int)
    
    # per_pixel = (comap_yx != -1).nonzero(as_tuple=True) #number_of_matches x tensor_dimension
    # print(H, W)
    for l in range(num_lens):
        for a in range(W):
            for b in range(H):
                x=comap_yx[l, b, a, 1]
                y=comap_yx[l, b, a, 0]
                if x!=-1 and y!=-1:
                    per_pixel[a,b, cnt_mpp[a,b]] = np.array([x,y, l])
                    mask[a,b, cnt_mpp[a,b]] = 1.
                    cnt_mpp[a,b]+=1
    return per_pixel , mask, cnt_mpp

def make_non_multiplex_mask(cnt_pixels, H,W,FLAGS, N_VIEWS,MAX_PER_PIXEL):
    mask = np.ones((N_VIEWS,H,W,MAX_PER_PIXEL,3))
    for i in range(H):
        for j in range(W):
            if cnt_pixels[i,j]>1:
                mask[:,i,j,:,:] = 0
    return mask

def max_overlapping_pixels(H,W,comap_yx, FLAGS):
    maps_pixel_to_rays, mask, cnt_mpp = get_rays_per_pixel(H,W, comap_yx, FLAGS)
    rgb = np.zeros((H,W,3))
    sub_pixels = np.zeros((H,W,MAX_PER_PIXEL,3))
    cnt_pixels = np.zeros((H,W)).astype(np.int)
    for i_index in range(H):
        for j_index in range(W):
            for cnt_rays in range(maps_pixel_to_rays.shape[2]): # over MAX_PER_PIXEL
                x_index = maps_pixel_to_rays[i_index, j_index,cnt_rays,0] #height
                y_index = maps_pixel_to_rays[i_index, j_index,cnt_rays,1] #width
                l_index = maps_pixel_to_rays[i_index, j_index,cnt_rays,2] #lens

                cnt = cnt_pixels[i_index, j_index]
                sub_pixels[i_index, j_index, cnt] = np.array([x_index, y_index, l_index])
                cnt_pixels[i_index, j_index] += 1
    return sub_pixels, cnt_pixels

def generate(comap_yx,base, model_path, num_lens, H, W):
#     rendered_views_path = '/home/vitran/plenoxels/jax_logs10/original2/multilens16_5img_5679_and59'
    # rendered_views_path = model_path +'/train_multilens_16_black'#+ "/multiplexed_input"
    rendered_views_path = model_path #+'/train_grid_att2'#+ "/multiplexed_input"
    
    maps_pixel_to_rays, mask, cnt_mpp = get_rays_per_pixel(H,W, comap_yx,20, num_lens)
    MAX_PER_PIXEL = np.max(cnt_mpp)
    maps_pixel_to_rays, mask, cnt_mpp = get_rays_per_pixel(H,W, comap_yx,MAX_PER_PIXEL, num_lens)
    u = int(np.max(comap_yx[0,:,:,:], axis=(0,1,2)) +1)
    sub_lens = np.zeros((num_lens,u,u,3))
    

    for j in range(num_lens):
        sub_lens_path = f"r_{base}_{j}.png"
        im_gt = imageio.imread(f'{rendered_views_path}/{sub_lens_path}').astype(np.float32) / 255.0 
        
        a = int(np.max(comap_yx[0,:,:,:], axis=(0,1,2)))+1
        im_gt = resize(im_gt, (a,a), anti_aliasing=True)
        sub_lens[j,:,:,:] = im_gt[:,:,:3]
        
        
    rgb = np.zeros((H,W,3)).astype(float)
    cnt_subpixels = np.zeros((H,W), dtype=int)
    
    for i_index in range(H):
        for j_index in range(W):
            for cnt_rays in range(maps_pixel_to_rays.shape[2]): # over MAX_PER_PIXEL
                x_index = maps_pixel_to_rays[i_index, j_index,cnt_rays,0] #height
                y_index = maps_pixel_to_rays[i_index, j_index,cnt_rays,1] #width
                l_index = maps_pixel_to_rays[i_index, j_index,cnt_rays,2] #lens
                if mask[i_index, j_index,cnt_rays]==1 and l_index in SUBIMAGES:
                    rgb[i_index, j_index] += sub_lens[l_index, x_index, y_index,:]
                    cnt_subpixels[i_index, j_index] +=1
            # rgb[i_index,j_index] = rgb[i_index,j_index] / cnt_subpixels[i_index, j_index] +1e-9#MAX_PER_PIXEL 
    print(f"MAX VALUE {np.max(rgb)}")
    max_pixel = np.max(rgb)
    rgb2 = rgb #/np.max(rgb)
    # vis = np.asarray(rgb2 * 255).astype(np.uint8)
    rgb = np.concatenate((rgb2, np.ones((H,W,1))), axis=2)
    im = Image.fromarray(np.uint8(rgb*255))
    return rgb, max_pixel

def model_output_mask(comap, num_lens, maps_pixel_to_rays, real_ray_mask, H,W,MAX_PER_PIXEL):
    mask = torch.zeros((num_lens,3,H,W), device='cuda', dtype=torch.float)
    pad_mapping = [[[] for _ in range(4)] for _ in range(num_lens)]
    border_minmax = [[[799, 0] for _ in range(2)] for _ in range(num_lens)]#[range_X],[range_Y]
    # max_u -=1
    #comap has shape (16, 800, 800,2)
    for k in range(num_lens):
        for ii in range(H):
            for jj in range(W):
                if comap[k,ii,jj,0]==-1:
                    continue
                aa = comap[k,ii,jj,0] #x
                bb = comap[k,ii,jj,1] #y
            
                min_x = min(border_minmax[k][0][0], bb)
                max_x = max(border_minmax[k][0][1], bb)
                min_y = min(border_minmax[k][1][0], aa)
                max_y = max(border_minmax[k][1][1], aa)
                border_minmax[k]=[[min_x, max_x],[min_y, max_y]]
        # print('border_minmax[k]',k, border_minmax[k])

    for i_index in range(H):
        for j_index in range(W):
            for cnt_rays in range(MAX_PER_PIXEL): # over MAX_PER_PIXEL
                # i and j in range 0-800
                # x and y in range 0-200
                x_index = maps_pixel_to_rays[i_index, j_index,cnt_rays,0] #height
                y_index = maps_pixel_to_rays[i_index, j_index,cnt_rays,1] #width
                l_index = maps_pixel_to_rays[i_index, j_index,cnt_rays,2] #lens
                if real_ray_mask[i_index, j_index,cnt_rays]==1:
                    # if l_index==0:
                    #     print(x_index, y_index, l_index)
                    mask[l_index, :, i_index, j_index] = 1.
                    
                    min_x, max_x = border_minmax[l_index][0]
                    min_y, max_y = border_minmax[l_index][1]
                    corner = -1
                    #. A----B
                    #  C----D
                    if x_index==min_x and y_index==min_y:   
                        corner = 0
                    elif x_index==max_x and y_index==min_y:
                        corner = 1
                    elif x_index==min_x and y_index==max_y:
                        corner = 2
                    elif x_index==max_x and y_index==max_y:
                        corner = 3
                    if corner!=-1:
                        pad_mapping[l_index][corner] = [i_index,j_index]
                        # print('update pad_mapping', pad_mapping[l_index][corner] , ' corner ', corner, 'l_index', l_index)
    mask.requires_grad = True
    return mask,pad_mapping, border_minmax
def generate_training(border_minmax,comap_yx, model_output, multiplexed_mask, pad_mapping, maps_pixel_to_rays  ):
    #model_output: list of 16 model output
    # im_gt = model_output[0]
    H = 800  # im_gt.shape[0]
    W = 800 # im_gt.shape[1]
    num_lens = len(model_output)
    
    # maps_pixel_to_rays, _, _ = get_rays_per_pixel(H,W, comap_yx,20, num_lens)
    # MAX_PER_PIXEL =  4 #np.max(cnt_mpp)
    # maps_pixel_to_rays, real_ray_mask, _ = get_rays_per_pixel(H,W, comap_yx,MAX_PER_PIXEL, num_lens)
    # multiplexed_mask,pad_mapping = model_output_mask(num_lens, maps_pixel_to_rays, real_ray_mask, H,W, MAX_PER_PIXEL)

    u = int(np.max(comap_yx[0,:,:,:], axis=(0,1,2)) +1)
    
    
    for j in range(num_lens):
        im_gt = model_output[j].unsqueeze(0).unsqueeze(0)
        im_gt = F.interpolate(im_gt, size=(3,u,u))  #The resize operation on tensor.
        im_gt = im_gt.squeeze(0).squeeze(0) #each is 3,200,200
        
        top_left = pad_mapping[j][0] # [pad_mapping[j][0][0],pad_mapping[j][1][0]] #[min_x, min_y]
        bottom_right = pad_mapping[j][3] #[pad_mapping[j][0][1],pad_mapping[j][1][1]]  #[max_x, max_y]
        
        # add to remove partial top and bottom row
        # print(top_left, bottom_right)
        if top_left == [0,0] and bottom_right ==[u,u]:
            print("full")
            min_y, max_y = border_minmax[j][0]
            min_x, max_x = border_minmax[j][1]

            im_gt = im_gt[:,int(min_y):int(max_y)+1,int(min_x):int(max_x)+1] 

            p2d =  (top_left[1], W-bottom_right[1]-1, top_left[0], H-bottom_right[0]-1) # pad last dim by (1, 1) and 2nd to last by (2, 2)
        else: 
            im_gt = im_gt[:0, :0]
            p2d = (u,u,u,u)
        model_output[j] = F.pad(im_gt, p2d, "constant", 0)
    model_output = torch.stack(model_output,0) #[num_lens, 3, H,W]
    
    rgb = model_output*multiplexed_mask
    rgb = torch.sum(rgb,0)
    return rgb


def generate_single_training(index, border_minmax,comap_yx, model_output_single, multiplexed_mask, pad_mapping, maps_pixel_to_rays, H, W  ):
    #model_output: list of 16 model output
    # H = 800  # im_gt.shape[0]
    # W = 800 # im_gt.shape[1]

    u = int(np.max(comap_yx[0,:,:,:], axis=(0,1,2)) +1)
    
    j=index
    # for j in range(num_lens):
    im_gt = model_output_single.unsqueeze(0).unsqueeze(0)
    im_gt = F.interpolate(im_gt, size=(3,u,u))  #The resize operation on tensor.
    im_gt = im_gt.squeeze(0).squeeze(0) #each is 3,200,200
    
    top_left = pad_mapping[j][0] # [pad_mapping[j][0][0],pad_mapping[j][1][0]] #[min_x, min_y]
    bottom_right = pad_mapping[j][3] #[pad_mapping[j][0][1],pad_mapping[j][1][1]]  #[max_x, max_y]
    
    min_y, max_y = border_minmax[j][0]
    min_x, max_x = border_minmax[j][1]

    im_gt = im_gt[:,int(min_y):int(max_y)+1,int(min_x):int(max_x)+1] 

    p2d =  (top_left[1], W-bottom_right[1]-1, top_left[0], H-bottom_right[0]-1) # pad last dim by (1, 1) and 2nd to last by (2, 2)

    if j not in SUBIMAGES:
        m = torch.zeros(im_gt.size(), device='cuda')
        im_gt = m*im_gt
    model_output_single = F.pad(im_gt, p2d, "constant", 0)
    return model_output_single

def generate_negative_image(model_output_single, single_viewpoint):
    flip_mask = torch.where(single_viewpoint.mask.mask != 0, 0, 1)
    im_gt = model_output_single * flip_mask
    return im_gt

def generate_single_training_pinhole_with_mask(model_output_single, single_viewpoint):
    # NptoTorch(cam_info.mask.mask) if cam_info.mask is not None else None
    x,y, h, w = single_viewpoint.mask.bbox
    im_gt = model_output_single.unsqueeze(0).unsqueeze(0)
    # print(f"from multiplexing {im_gt.shape} h={h} w={w}")
    # print("line 313 size", im_gt.size())
    im_gt = F.interpolate(im_gt, size=(3,323,323))  #The resize operation on tensor.
    im_gt = im_gt.squeeze(0).squeeze(0) #each is 3,200,200
    # print("line 316 size", im_gt.size())
    
    H=520
    W=780
    _, cur_h, cur_w = im_gt.shape
    p2d = (x,W-x-cur_w, y,H-y-cur_h)
    # print(f"name {mask_name}, p2d {p2d}")

    # print(index,p2d)
    model_output_single = F.pad(im_gt, p2d, "constant", 0)
    # print('here', model_output_single.size())
    mask_torch = torch.from_numpy(single_viewpoint.mask.mask).cuda()
    mask_torch = torch.permute(mask_torch, (2,0,1))
    # print('mask', mask_torch.size())
    # return im_gt 
    return model_output_single * mask_torch
    
def generate_single_training_pinhole(index, model_output_single, json_padding):
    json_data = json_padding[str(index)]
    ux,uy = json_data['size']
    
    im_gt = model_output_single.unsqueeze(0).unsqueeze(0)
    im_gt = F.interpolate(im_gt, size=(3,ux,uy))  #The resize operation on tensor.
    im_gt = im_gt.squeeze(0).squeeze(0) #each is 3,200,200
    
    startx = json_data['startx'] 
    endx = json_data['endx'] 
    starty = json_data['starty'] 
    endy = json_data['endy'] 
    im_gt = im_gt[:, startx:endx, starty:endy]

    list_of_lists = json_data['padding']
    p2d = [item for sublist in list_of_lists for item in sublist]
    p2d = tuple(p2d)
    p2d_swapped = (p2d[-2], p2d[-1], p2d[0], p2d[1])

    # print(index,p2d)
    model_output_single = F.pad(im_gt, p2d_swapped, "constant", 0)
    # print('here', model_output_single.size())
    return model_output_single


def generate_single_training_noop(index, border_minmax,comap_yx, model_output_single, multiplexed_mask, pad_mapping, maps_pixel_to_rays  ):
    #model_output: list of 16 model output
    return model_output_single


def get_adjacent_views(index, path):
    img90 = None
    # print("get adj from path", path)
    # with open('/home/vitran/plenoxels/blender_data/lego/transforms_train.json') as f:
    with open(f'{path}/transforms_train.json') as f:
    # with open('/home/vitran/plenoxels/blender_data/hotdog/transforms_train.json') as f:
        org_json = json.load(f)
        img90 = np.array(org_json['frames'][index[0]]['transform_matrix']).astype(float)
    allDiff = []
    # with open('/home/vitran/plenoxels/blender_data/lego/transforms_test.json') as f:
    with open(f'{path}/transforms_test.json') as f:
    # with open('/home/vitran/plenoxels/blender_data/hotdog/transforms_test.json') as f:
        org_json = json.load(f)
        for i,frames in enumerate(org_json['frames']):
            img = np.array(frames['transform_matrix']).astype(float)
            diff = np.mean(np.square(img90[:,-1]-img[:,-1]))
            allDiff.append([i, diff])
    allDiff.sort(key=lambda x:x[1])
    allDiff_index = [d[0] for d in allDiff]
    return allDiff_index[:6]

