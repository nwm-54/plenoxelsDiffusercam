import copy
import json
import sys
from typing import Any, Dict, List

import numpy as np

sys.path.append("../")


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
    with open(input_path, "r") as f:
        data: Dict[str, Any] = json.load(f)

    start_frame: Dict[str, Any] | None = None
    for frame in data["frames"]:
        if frame["file_path"].endswith(view_name):
            start_frame = frame
            break

    if not start_frame:
        raise Exception(f"View '{view_name}' not found in '{input_path}'.")

    c2w_matrices = [np.array(frame["transform_matrix"]) for frame in data["frames"]]
    cam_positions = np.array([c2w[:3, 3] for c2w in c2w_matrices])  # noqa: F841
    center = lego_cog

    start_c2w = np.array(start_frame["transform_matrix"])
    start_pos = start_c2w[:3, 3]
    radius = np.linalg.norm(start_pos - center)
    height = start_pos[2]
    up = np.array([0, 0, 1])

    new_frames: List[Dict[str, Any]] = []
    for i in range(num_frames):
        angle = i * (2 * np.pi / num_frames) * speed

        # Rotates the starting camera position around the center.
        new_pos = np.array(
            [
                center[0] + radius * np.cos(angle),
                center[2] + radius * np.sin(angle),
                height,
            ]
        )

        # Creates a view matrix that looks at the center.
        forward = center - new_pos
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        new_up = np.cross(right, forward)

        new_c2w = np.eye(4)
        new_c2w[:3, 0] = right
        new_c2w[:3, 1] = new_up
        new_c2w[:3, 2] = -forward
        new_c2w[:3, 3] = new_pos

        new_frame = copy.deepcopy(start_frame)
        new_frame["transform_matrix"] = new_c2w.tolist()
        new_frame["file_path"] = f"./train/r_{i}"
        new_frames.append(new_frame)

    output_data = {
        "camera_angle_x": data["camera_angle_x"],
        "frames": new_frames,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"Successfully generated a {num_frames}-frame orbital trajectory.")
    print(f"Output saved to '{output_path}'.")


if __name__ == "__main__":
    lego_cog = [0.0323316864669323, -0.056738946586847305, 0.19368238747119904]
    generate_orbital_trajectory(
        input_path="/home/wl757/multiplexed-pixels/plenoxels/blender_data/lego_gen12/transforms_train.json",
        output_path="/home/wl757/multiplexed-pixels/plenoxels/blender_data/lego_gen12/orbital_trajectory.json",
        view_name="r_50",
        num_frames=90,
        speed=0.3,
    )
