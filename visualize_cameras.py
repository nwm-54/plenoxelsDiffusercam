import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from scene.dataset_readers_multiviews import find_max_min_dispersion_subset

def get_camera_frustum_vertices(c2w: np.ndarray, fov_x: float, scale: float = 0.3) -> np.ndarray:
    """Calculates the 5 vertices of a camera frustum pyramid in world space."""
    half_w = scale * np.tan(fov_x / 2)
    local_vertices = np.array([[0,0,0], [-half_w,half_w,-scale], [half_w,half_w,-scale], [half_w,-half_w,-scale], [-half_w,-half_w,-scale]])
    local_vertices_hom = np.hstack([local_vertices, np.ones((5, 1))])
    return (c2w @ local_vertices_hom.T).T[:, :3]

def create_frustum_mesh_data(frustum_vertices_list: List[np.ndarray]) -> Tuple[go.Mesh3d, ...]:
    """Converts a list of frustum vertices into Plotly Mesh3D data format."""
    all_x, all_y, all_z, all_i, all_j, all_k = [], [], [], [], [], []
    offset = 0
    for vertices in frustum_vertices_list:
        all_x.extend(vertices[:, 0]); all_y.extend(vertices[:, 1]); all_z.extend(vertices[:, 2])
        faces = [[0,1,2], [0,2,3], [0,3,4], [0,4,1], [1,2,4], [2,3,4]]
        for face in faces:
            all_i.append(offset + face[0]); all_j.append(offset + face[1]); all_k.append(offset + face[2])
        offset += 5
    return go.Mesh3d(x=all_x, y=all_y, z=all_z, i=all_i, j=all_j, k=all_k)

def _create_static_plot(fig: go.Figure, n: int, c2w_matrices: List, cam_positions: np.ndarray, fov_x: float):
    """Configures the Figure for a single, static visualization."""
    print(f"✨ Selecting {n} maximally dispersed cameras...")
    selected_indices = set(find_max_min_dispersion_subset(cam_positions, n))

    frustums_selected, frustums_other = [], []
    for i, c2w in enumerate(c2w_matrices):
        vertices = get_camera_frustum_vertices(c2w, fov_x)
        (frustums_selected if i in selected_indices else frustums_other).append(vertices)

    if frustums_other:
        mesh = create_frustum_mesh_data(frustums_other)
        fig.add_trace(go.Mesh3d(x=mesh.x, y=mesh.y, z=mesh.z, i=mesh.i, j=mesh.j, k=mesh.k, color='black', opacity=0.2, name='Other Cameras'))
    if frustums_selected:
        mesh = create_frustum_mesh_data(frustums_selected)
        fig.add_trace(go.Mesh3d(x=mesh.x, y=mesh.y, z=mesh.z, i=mesh.i, j=mesh.j, k=mesh.k, color='red', opacity=0.5, name='Selected Cameras'))

    fig.update_layout(title=f'Camera Selections for n = {n}')

def _create_animated_plot(fig: go.Figure, n_range: range, c2w_matrices: List, cam_positions: np.ndarray, fov_x: float):
    """Configures the Figure for an animated visualization."""
    # Create the base frame
    _create_static_plot(fig, n_range.start, c2w_matrices, cam_positions, fov_x)

    # Create animation frames
    animation_frames = []
    for n in n_range:
        print(f"✨ Calculating animation frame for n = {n}...")
        selected_indices = set(find_max_min_dispersion_subset(cam_positions, n))
        frustums_selected, frustums_other = [], []
        for i, c2w in enumerate(c2w_matrices):
            vertices = get_camera_frustum_vertices(c2w, fov_x)
            (frustums_selected if i in selected_indices else frustums_other).append(vertices)
        
        mesh_other = create_frustum_mesh_data(frustums_other)
        mesh_selected = create_frustum_mesh_data(frustums_selected)
        animation_frames.append(go.Frame(name=str(n), data=[mesh_other, mesh_selected]))
    
    fig.frames = animation_frames
    
    # Create and configure animation controls
    slider_steps = [{'method': 'animate', 'label': str(n), 'args': [[str(n)], {'mode': 'immediate', 'frame': {'redraw': True}}]} for n in n_range]
    
    fig.update_layout(
        updatemenus=[{'type': 'buttons', 'showactive': False, 'direction': 'left', 'x': 0.1, 'y': 0, 'xanchor': 'right', 'yanchor': 'top', 'pad': {'r': 10, 't': 87},
                      'buttons': [{'label': '▶️ Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 200, 'redraw': True}, 'transition': {'duration': 0}, 'fromcurrent': True}]},
                                  {'label': '⏸️ Pause', 'method': 'animate', 'args': [[None], {'mode': 'immediate'}]}]}],
        sliders=[{'active': 0, 'steps': slider_steps, 'currentvalue': {'prefix': 'Number of Views: '}, 'pad': {'t': 50}}]
    )

def visualize_cameras(json_path: str, n_train_images: int = -1) -> None:
    print(f"📸 Loading cameras from {json_path}...")
    with open(json_path, 'r') as f: data = json.load(f)

    frames, fov_x = data['frames'], data['camera_angle_x']
    c2w_matrices = [np.array(frame['transform_matrix']) for frame in frames]
    cam_positions = np.array([c2w[:3, 3] for c2w in c2w_matrices])

    fig = go.Figure()
    if len(n_train_images) == 1:
        n = n_train_images[0]
        _create_static_plot(fig, n, c2w_matrices, cam_positions, fov_x)
        output_filename = f"static_visualization_n{n}.html"
    else:
        n_range = range(n_train_images[0], n_train_images[1], n_train_images[2] if len(n_train_images) == 3 else 1)
        _create_animated_plot(fig, n_range, c2w_matrices, cam_positions, fov_x)
        output_filename = f"camera_animation_{n_range.start}_{n_range.stop-1}.html"

    fig.update_layout(
        scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, aspectmode='data'),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )

    print(f"💾 Saving interactive visualization to {output_filename}")
    fig.write_html(output_filename)
    # fig.write_image("interactive_visualization.png", width=800, height=600)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize camera frustums.")
    parser.add_argument('--source_path', type=str, required=True, help='Path to transforms_train.json')
    parser.add_argument('--n_train_images', type=int, nargs='+', default=[-1])
    args = parser.parse_args()

    if not (1 <= len(args.n_train_images) <= 3):
        raise ValueError("The --n_train_images argument must have 1, 2, or 3 integer values.")
    if len(args.n_train_images) > 1 and args.n_train_images[0] < 2:
        raise ValueError("For an animation, the starting number of views must be at least 2.")

    visualize_cameras(args.source_path, args.n_train_images)