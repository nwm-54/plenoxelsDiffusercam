import copy
import json
import sys
from typing import Any, Dict, List

import numpy as np

sys.path.append("../")

from utils.graphics_utils import getWorld2View2

def generate_orbital_trajectory(
    input_path: str,
    output_path: str,
    view_name: str,
    num_frames: int,
    speed: float,
) -> None:
    """
    Generates a smooth orbital camera trajectory from a specified view in a JSON file.
    """
    with open(input_path, 'r') as f:
        data: Dict[str, Any] = json.load(f)

    start_frame: Dict[str, Any] | None = None
    for frame in data["frames"]:
        if frame["file_path"].endswith(view_name):
            start_frame = frame
            break

    if not start_frame: raise Exception(f"View '{view_name}' not found in '{input_path}'.")

    c2w: np.ndarray = np.array(start_frame["transform_matrix"])
    c2w[:3, 1:3] *= -1
    w2c = np.linalg.inv(c2w)
        
    print(f"Successfully generated a {num_frames}-frame orbital trajectory.")
    print(f"Output saved to '{output_path}'.")

if __name__ == "__main__":
    lego_cog = [0.0323316864669323, -0.056738946586847305, 0.19368238747119904]
    generate_orbital_trajectory(
        input_path="/home/wl757/multiplexed-pixels/plenoxels/blender_data/lego_gen12/transforms_train.json",
        output_path="orbital_trajectory.json",
        view_name="r_50",
        num_frames=90,
        speed=0.4
    )