import json
import numpy as np
import torch
import imageio as imageio
import math
import torch.nn.functional as F

SUBIMAGES = list(range(16))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from multiplexing_updated.py
def get_comap(num_lens, d_lens_sensor, H, W):
    # Verify input and calculate the grid dimensions
    if math.sqrt(num_lens)**2 == num_lens:
        num_lenses_yx = [int(math.sqrt(num_lens)), int(math.sqrt(num_lens))]
    else:
        print('Number of sublens should be a square number')
        assert False
    
    # Calculate microlens dimensions in pixels based on d_lens_sensor
    base_microlens_size = min(H // num_lenses_yx[0], W // num_lenses_yx[1]) // 12
    microlens_height = int(base_microlens_size * d_lens_sensor)
    microlens_height = microlens_height - (microlens_height % 2)  # Make dimensions even for convenience
    microlens_width = microlens_height  # Keep microlenses square
    comap_yx = -np.ones((num_lens, H, W, 2))
    
    # Calculate positions for microlenses to distribute from edge to edge
    if num_lenses_yx[0] > 1:
        y_positions = np.linspace(
            microlens_height // 2,  # First lens centered at top edge + half lens height
            H - microlens_height // 2,  # Last lens centered at bottom edge - half lens height
            num_lenses_yx[0]
        )
    else: y_positions = np.array([H // 2]) # If only one row, place it in the center
    if num_lenses_yx[1] > 1:
        x_positions = np.linspace(
            microlens_width // 2,  # First lens centered at left edge + half lens width
            W - microlens_width // 2,  # Last lens centered at right edge - half lens width
            num_lenses_yx[1]
        )
    else: x_positions = np.array([W // 2]) # If only one column, place it in the center
    
    for i in range(num_lens):
        row, col = i // num_lenses_yx[1], i % num_lenses_yx[1]        
        center_y, center_x = int(y_positions[row]), int(x_positions[col])
        start_y = int(max(0, center_y - microlens_height // 2))
        end_y = int(min(H, center_y + microlens_height // 2))
        start_x = int(max(0, center_x - microlens_width // 2))
        end_x = int(min(W, center_x + microlens_width // 2))
        
        for y in range(start_y, end_y):
            for x in range(start_x, end_x):
                local_y, local_x = y - start_y, x - start_x
                comap_yx[i, y, x, 0] = local_y
                comap_yx[i, y, x, 1] = local_x
    
    # Return the original dimension as second return value
    dim_lens_lf_yx = [microlens_height, microlens_width]
    return comap_yx, dim_lens_lf_yx

def read_images(num_lens, model_path, base):
    images = []
    for j in range(num_lens):
        sub_lens_path = f"r_{base}_{j}.png"
        im_gt = imageio.imread(f'{model_path}/{sub_lens_path}').astype(np.float32) / 255.0
        im_tensor = torch.from_numpy(im_gt[:, :, :3]).permute(2, 0, 1).to(device)
        images.append(im_tensor)  # Keep only RGB channels

    return images

def get_max_overlap(comap_yx, num_lens, H, W):
    overlap_count = torch.zeros(H, W, dtype=torch.int32, device=device)
    for i in range(num_lens):
        valid_mask = (comap_yx[i][:,:,1] != -1)
        overlap_count += valid_mask
    return overlap_count.max()

def generate_alpha_map(comap_yx, num_lens, H, W):
    overlap_count = np.zeros((H, W), dtype=np.int32)
    
    for i in range(num_lens):
        valid_mask = (comap_yx[i, :, :, 0] != -1)
        overlap_count += valid_mask
    
    alpha_map = np.zeros((H, W))
    non_zero_mask = (overlap_count > 0)
    alpha_map[non_zero_mask] = 1.0 / overlap_count[non_zero_mask]
    return alpha_map

def generate(images, comap_yx, dim_lens_lf_yx, num_lens, H, W, max_overlap):
    grid_size = int(math.sqrt(num_lens))
    idx = torch.arange(grid_size, device=device)
    grid_i, grid_j = torch.meshgrid(idx, idx, indexing='ij')
    mapping = ((grid_size - 1 - grid_i) + (grid_size - 1 - grid_j) * grid_size).reshape(-1)
    
    images_tensor = torch.stack(images, dim=0).to(device)
    selected_images = images_tensor[mapping]
    resized_images = F.interpolate(
        selected_images, 
        size=(dim_lens_lf_yx[0], dim_lens_lf_yx[1]), 
        mode='bilinear', 
        align_corners=False
    )

    output_image = torch.zeros(3, H, W, device=device, dtype=torch.float32)
    for i in range(num_lens):
        y_coords = comap_yx[i, :, :, 0]
        x_coords = comap_yx[i, :, :, 1]
        
        valid_mask = (y_coords != -1) & (x_coords != -1) & \
                     (y_coords >= 0) & (y_coords < dim_lens_lf_yx[0]) & \
                     (x_coords >= 0) & (x_coords < dim_lens_lf_yx[1])
        
        # Only process this microlens if there are any valid mapping positions.
        if valid_mask.any():
            # Get 2D indices within the sub-image where valid_mask is True.
            y_indices, x_indices = torch.where(valid_mask)
            y_src = y_coords[valid_mask].long()
            x_src = x_coords[valid_mask].long()
            output_image[:, y_indices, x_indices] += resized_images[i, :, y_src, x_src]

    output_image = torch.div(output_image, max_overlap)
    return output_image

# def generate(images, comap_yx, dim_lens_lf_yx, num_lens, sensor_size, alpha_map):
#     grid_size = int(math.sqrt(num_lens))
#     idx = torch.arange(grid_size, device=device)
#     grid_i, grid_j = torch.meshgrid(idx, idx, indexing='ij')
#     mapping = ((grid_size - 1 - grid_i) + (grid_size - 1 - grid_j) * grid_size).reshape(-1)
    
#     images_tensor = torch.stack(images, dim=0).to(device)
#     selected_images = images_tensor[mapping]
#     resized_images = F.interpolate(
#         selected_images, 
#         size=(dim_lens_lf_yx[0], dim_lens_lf_yx[1]), 
#         mode='bilinear', 
#         align_corners=False
#     )
    
#     output_image = torch.zeros(3, sensor_size, sensor_size, device=device, dtype=torch.float32)
    
#     for i in range(num_lens):
#         y_coords = comap_yx[i, :, :, 0]
#         x_coords = comap_yx[i, :, :, 1]
        
#         valid_mask = (y_coords != -1) & (x_coords != -1) & \
#                      (y_coords >= 0) & (y_coords < dim_lens_lf_yx[0]) & \
#                      (x_coords >= 0) & (x_coords < dim_lens_lf_yx[1])
        
#         # Only process this microlens if there are any valid mapping positions.
#         if valid_mask.any():
#             # Get 2D indices within the sub-image where valid_mask is True.
#             y_indices, x_indices = torch.where(valid_mask)
#             y_src = y_coords[valid_mask].long()
#             x_src = x_coords[valid_mask].long()
#             output_image[:, y_indices, x_indices] += resized_images[i, :, y_src, x_src] * alpha_map[y_indices, x_indices].unsqueeze(0)
    
#     # Clamp the final output to ensure pixel values are in the valid range [0, 1].
#     output_image = torch.clamp(output_image, 0, 1)
    
#     return output_image

def get_rays_per_pixel(H, W, comap_yx, max_per_pixel, num_lens):    
    per_pixel = np.zeros((W, H, max_per_pixel, 3)).astype(int)
    mask = np.zeros((W, H, max_per_pixel)).astype(int)
    cnt_mpp = np.zeros((W, H)).astype(int)
    
    for l in range(num_lens):
        # Use reversed lens index (num_lens - 1 - l) instead of l
        reversed_l = num_lens - 1 - l
        # reversed_l = l
        
        for a in range(W):
            for b in range(H):
                x = comap_yx[l, b, a, 1]
                y = comap_yx[l, b, a, 0]
                if x != -1 and y != -1:
                    per_pixel[a, b, cnt_mpp[a, b]] = np.array([x, y, reversed_l])
                    mask[a, b, cnt_mpp[a, b]] = 1.
                    cnt_mpp[a, b] += 1
                    
    return per_pixel, mask, cnt_mpp

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


def generate_single_training(index, border_minmax, comap_yx, model_output_single, pad_mapping, H, W):
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

    if j not in list(range(16)):
        m = torch.zeros(im_gt.size(), device='cuda')
        im_gt = m*im_gt
    model_output_single = F.pad(im_gt, p2d, "constant", 0)
    return model_output_single


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

