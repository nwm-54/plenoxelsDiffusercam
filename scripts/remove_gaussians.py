import sys
sys.path.append("/home/wl757/multiplexed-pixels/gs7")

import argparse
import os

import numpy as np
import torch
from scene.gaussian_model import GaussianModel
from utils.sh_utils import SH2RGB


def remove_gaussians(ply_path: str, output_path: str, brightness_threshold_low: float = 0.1,
                   brightness_threshold_high: float = 0.9, distance_threshold: float = 3.0, 
                   sh_degree: int = 3) -> None:
   gaussians = GaussianModel(sh_degree)
   
   print(f"Loading PLY file from: {ply_path}")
   gaussians.load_ply(ply_path)
   
   initial_count = gaussians._xyz.shape[0]
   print(f"Initial number of gaussians: {initial_count}")
   
   features_dc = gaussians._features_dc.squeeze(1)
   rgb_colors = SH2RGB(features_dc)
   mean_brightness = rgb_colors.mean(dim=1)
   black_mask = mean_brightness <= brightness_threshold_low
   white_mask = mean_brightness >= brightness_threshold_high
   black_count = black_mask.sum().item()
   white_count = white_mask.sum().item()
   print(f"Found {black_count} black gaussians ({black_count/initial_count*100:.2f}%)")
   print(f"Found {white_count} white gaussians ({white_count/initial_count*100:.2f}%)")

   positions = gaussians._xyz
   center_of_mass = positions.mean(dim=0)
   distances = torch.norm(positions - center_of_mass, dim=1)
   median_distance = torch.median(distances)
   mad = torch.median(torch.abs(distances - median_distance))
   outlier_mask_mad = distances > (median_distance + distance_threshold * mad * 1.4826)
   outlier_mask = outlier_mask_mad
   outlier_count = outlier_mask.sum().item()

   print(f"  Center of mass: [{center_of_mass[0]:.3f}, {center_of_mass[1]:.3f}, {center_of_mass[2]:.3f}]")
   print(f"  Median distance: {median_distance:.3f}")
   print(f"  MAD: {mad:.3f}")
   print(f"  Distance threshold: {median_distance + distance_threshold * mad * 1.4826:.3f}")
   print(f"Found {outlier_count} spatial outliers ({outlier_count/initial_count*100:.2f}%)")

   remove_mask = black_mask | white_mask | outlier_mask
   keep_mask = ~remove_mask
   black_and_outlier_count = (black_mask & outlier_mask).sum().item()
   white_and_outlier_count = (white_mask & outlier_mask).sum().item()
   total_remove = remove_mask.sum().item()
   
   print(f"\nSummary:")
   print(f"  Black gaussians: {black_count}")
   print(f"  White gaussians: {white_count}")
   print(f"  Spatial outliers: {outlier_count}")
   print(f"  Both black and outlier: {black_and_outlier_count}")
   print(f"  Both white and outlier: {white_and_outlier_count}")
   print(f"  Total to remove: {total_remove} ({total_remove/initial_count*100:.2f}%)")
   
   if total_remove == 0:
       print("No gaussians to remove. Copying file as-is.")
       import shutil
       shutil.copy2(ply_path, output_path)
       return
   
   gaussians._xyz = gaussians._xyz[keep_mask]
   gaussians._features_dc = gaussians._features_dc[keep_mask]
   gaussians._features_rest = gaussians._features_rest[keep_mask]
   gaussians._opacity = gaussians._opacity[keep_mask]
   gaussians._scaling = gaussians._scaling[keep_mask]
   gaussians._rotation = gaussians._rotation[keep_mask]
   
   print(f"Saving cleaned PLY file to: {output_path}")
   gaussians.save_ply(output_path)
   
   final_count = gaussians._xyz.shape[0]
   print(f"Final number of gaussians: {final_count}")

def main() -> None:
   parser = argparse.ArgumentParser(description="Remove black, white, and outlier gaussians from a PLY file")
   parser.add_argument("input_ply", type=str, help="Path to input PLY file")
   parser.add_argument("output_ply", type=str, help="Path to output PLY file")
   parser.add_argument("--brightness_threshold_low", type=float, default=0.07, 
                       help="RGB threshold below which gaussians are considered black (default: 0.07)")
   parser.add_argument("--brightness_threshold_high", type=float, default=0.93,
                       help="RGB threshold above which gaussians are considered white (default: 0.93)")
   parser.add_argument("--distance_threshold", type=float, default=0.5,
                       help="Number of MADs from median distance to consider as outlier (default: 0.5)")
   parser.add_argument("--sh_degree", type=int, default=3,
                       help="Spherical harmonics degree (default: 3)")
   parser.add_argument("--cuda", action="store_true",
                       help="Use CUDA if available")
   
   args = parser.parse_args()
   
   if not os.path.exists(args.input_ply):
       print(f"Error: Input file '{args.input_ply}' does not exist")
       return
   
   output_dir = os.path.dirname(args.output_ply)
   if output_dir and not os.path.exists(output_dir):
       os.makedirs(output_dir, exist_ok=True)
   
   if args.cuda and torch.cuda.is_available():
       torch.cuda.set_device(0)
   
   remove_gaussians(
       args.input_ply,
       args.output_ply,
       brightness_threshold_low=args.brightness_threshold_low,
       brightness_threshold_high=args.brightness_threshold_high,
       distance_threshold=args.distance_threshold,
       sh_degree=args.sh_degree
   )

if __name__ == "__main__":
   main()