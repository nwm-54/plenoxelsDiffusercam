import math
from typing import List, Tuple

import imageio as imageio
import numpy as np
import torch
import torch.nn.functional as F
from utils.graphics_utils import getWorld2View2
from scene.dataset_readers_multiviews import CameraInfo

SUBIMAGES = list(range(16))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# from multiplexing_updated.py
def get_comap(
    num_lens: int, d_lens_sensor: int, H: int, W: int
) -> Tuple[np.ndarray, List[int]]:
    # Verify input and calculate the grid dimensions
    if math.sqrt(num_lens) ** 2 == num_lens:
        num_lenses_yx = [int(math.sqrt(num_lens)), int(math.sqrt(num_lens))]
    else:
        print("Number of sublens should be a square number")
        assert False

    # Calculate microlens dimensions in pixels based on d_lens_sensor
    base_microlens_size = min(H // num_lenses_yx[0], W // num_lenses_yx[1]) // 12
    microlens_height = int(base_microlens_size * d_lens_sensor)
    microlens_height = microlens_height - (
        microlens_height % 2
    )  # Make dimensions even for convenience
    microlens_width = microlens_height  # Keep microlenses square
    comap_yx = -np.ones((num_lens, H, W, 2))

    # Calculate positions for microlenses to distribute from edge to edge
    if num_lenses_yx[0] > 1:
        y_positions = np.linspace(
            microlens_height // 2,  # First lens centered at top edge + half lens height
            H
            - microlens_height
            // 2,  # Last lens centered at bottom edge - half lens height
            num_lenses_yx[0],
        )
    else:
        y_positions = np.array([H // 2])  # If only one row, place it in the center
    if num_lenses_yx[1] > 1:
        x_positions = np.linspace(
            microlens_width // 2,  # First lens centered at left edge + half lens width
            W
            - microlens_width
            // 2,  # Last lens centered at right edge - half lens width
            num_lenses_yx[1],
        )
    else:
        x_positions = np.array([W // 2])  # If only one column, place it in the center

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


def read_images(num_lens, img_dir, base):
    images = []
    for j in range(num_lens):
        sub_lens_path = f"r_{base}_{j}.png"
        im_gt = imageio.imread(f"{img_dir}/{sub_lens_path}").astype(np.float32) / 255.0
        im_tensor = torch.from_numpy(im_gt[:, :, :3]).permute(2, 0, 1).to(device)
        images.append(im_tensor)  # Keep only RGB channels

    return images


def get_max_overlap(comap_yx, num_lens, H, W):
    overlap_count = torch.zeros(H, W, dtype=torch.int32, device=device)
    for i in range(num_lens):
        valid_mask = comap_yx[i][:, :, 1] != -1
        overlap_count = overlap_count + valid_mask
    return overlap_count.max().item()


def generate_alpha_map(comap_yx, num_lens, H, W):
    overlap_count = np.zeros((H, W), dtype=np.int32)

    for i in range(num_lens):
        valid_mask = comap_yx[i, :, :, 0] != -1
        overlap_count = overlap_count + valid_mask

    alpha_map = np.zeros((H, W))
    non_zero_mask = overlap_count > 0
    alpha_map[non_zero_mask] = 1.0 / overlap_count[non_zero_mask]
    return alpha_map


def generate(images, comap_yx, dim_lens_lf_yx, num_lens, H, W, max_overlap):
    grid_size = int(math.sqrt(num_lens))
    idx = torch.arange(grid_size, device=device)
    grid_i, grid_j = torch.meshgrid(idx, idx, indexing="ij")
    mapping = ((grid_size - 1 - grid_i) + (grid_size - 1 - grid_j) * grid_size).reshape(
        -1
    )

    images_tensor = torch.stack(images, dim=0).to(device)
    selected_images = images_tensor[mapping]
    resized_images = F.interpolate(
        selected_images,
        size=(dim_lens_lf_yx[0], dim_lens_lf_yx[1]),
        mode="bilinear",
        align_corners=False,
    )

    output_image = torch.zeros(3, H, W, device=device, dtype=torch.float32)
    for i in range(num_lens):
        y_coords = comap_yx[i, :, :, 0]
        x_coords = comap_yx[i, :, :, 1]

        valid_mask = (
            (y_coords != -1)
            & (x_coords != -1)
            & (y_coords >= 0)
            & (y_coords < dim_lens_lf_yx[0])
            & (x_coords >= 0)
            & (x_coords < dim_lens_lf_yx[1])
        )

        # Only process this microlens if there are any valid mapping positions.
        if valid_mask.any():
            # Get 2D indices within the sub-image where valid_mask is True.
            y_indices, x_indices = torch.where(valid_mask)
            y_src = y_coords[valid_mask].long()
            x_src = x_coords[valid_mask].long()
            output_image[:, y_indices, x_indices] += resized_images[i, :, y_src, x_src]

    output_image = torch.div(output_image, max_overlap)
    return output_image


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
                    mask[a, b, cnt_mpp[a, b]] = 1.0
                    cnt_mpp[a, b] += 1

    return per_pixel, mask, cnt_mpp


def get_adjacent_views(
    train_cam: CameraInfo, all_test_cams: List[CameraInfo]
) -> List[int]:
    train_w2c = getWorld2View2(train_cam.R, train_cam.T)
    train_c2w = np.linalg.inv(train_w2c)
    train_pos = train_c2w[:3, 3]
    all_diffs = []
    for test_cam in all_test_cams:
        test_w2c = getWorld2View2(test_cam.R, test_cam.T)
        test_c2w = np.linalg.inv(test_w2c)
        test_pos = test_c2w[:3, 3]

        dist_sq = np.sum(np.square(train_pos - test_pos))
        all_diffs.append((test_cam.uid, dist_sq))

    all_diffs.sort(key=lambda x: x[1])
    closest_uids = [uid for uid, dist in all_diffs]
    return closest_uids[:6]  # Return the closest 6 views
