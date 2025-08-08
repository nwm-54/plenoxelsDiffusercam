import argparse
import json
import os
from typing import List, Dict, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scene.dataset_readers_multiviews import find_max_min_dispersion_subset

def get_camera_frustum(c2w: np.ndarray, fov_x: float, scale: float = 0.3) -> np.ndarray:
    """Calculates the 5 vertices of a camera frustum pyramid in world space."""
    half_w = scale * np.tan(fov_x / 2)
    half_h = half_w

    local_vertices = np.array([
        [0, 0, 0],
        [-half_w, half_h, -scale],
        [half_w, half_h, -scale],
        [half_w, -half_h, -scale],
        [-half_w, -half_h, -scale]
    ])
    local_vertices_hom = np.hstack([local_vertices, np.ones((5, 1))])
    world_vertices_hom = (c2w @ local_vertices_hom.T).T
    return world_vertices_hom[:, :3]

def visualize_cameras(source_path: str, n_train_images: int = -1) -> None:
    json_path = os.path.join(source_path, "transforms_train.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found at {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)

    frames = data['frames']
    fov_x = data['camera_angle_x']

    cam_positions = np.array([np.array(frame['transform_matrix'])[:3, 3] for frame in frames])
    selected_indices = set()
    if n_train_images > 0:
        selected_indices = set(find_max_min_dispersion_subset("lego", cam_positions, n_train_images))
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    frustrums_selected, frustrums_other = [], []
    for i, frame in enumerate(frames):
        c2w = np.array(frame['transform_matrix'])
        vertices = get_camera_frustum(c2w, fov_x)
        faces = [
            [vertices[0], vertices[1], vertices[2]],
            [vertices[0], vertices[2], vertices[3]],
            [vertices[0], vertices[3], vertices[4]],
            [vertices[0], vertices[4], vertices[1]],
            [vertices[1], vertices[2], vertices[3], vertices[4]],
        ]

        if i in selected_indices:
            frustrums_selected.extend(faces)
        else:
            frustrums_other.extend(faces)
        
    if frustrums_other:
        ax.add_collection3d(Poly3DCollection(frustrums_other, facecolors='black', linewidths=0.5, edgecolors='k', alpha=0.3))
    if frustrums_selected:
        ax.add_collection3d(Poly3DCollection(frustrums_selected, facecolors='red', linewidths=0.5, edgecolors='k', alpha=0.6))
    
    ax.set_box_aspect([1, 1, 1])
    ax.auto_scale_xyz(cam_positions[:, 0], cam_positions[:, 1], cam_positions[:, 2])

    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.axis('off')

    plt.tight_layout()
    output_path = 'camera_visualization.png'
    plt.savefig(output_path, dpi=300, transparent=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize camera frustums.")
    parser.add_argument('--source_path', type=str, required=True, help='Path to the source directory containing transforms_train.json')
    parser.add_argument('--n_train_images', type=int, default=-1)
    args = parser.parse_args()

    visualize_cameras(args.source_path, args.n_train_images)