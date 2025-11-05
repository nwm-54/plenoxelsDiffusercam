import argparse
import json
import math
from dataclasses import dataclass
from itertools import product
from typing import List, Optional, Sequence, Set, Tuple

import numpy as np
import plotly.graph_objects as go

from scene.scene_utils import configure_world_to_m, mm_to_world, solve_offset_for_angle
from utils.dispersion_utils import find_max_min_dispersion_subset
from utils.visualization_utils import (
    create_frustum_mesh_data,
    get_camera_frustum_vertices,
)


@dataclass
class CameraOptions:
    mode: str = "default"
    angle_deg: float = 10.0
    n_multiplexed_images: int = 16
    object_center: Optional[np.ndarray] = None
    iphone_same_focal_length: bool = False
    iphone_baseline_x_mm: float = 9.5
    iphone_baseline_y_mm: float = 9.5
    iphone_ultrawide_mm: float = 13.0
    iphone_wide_mm: float = 24.0
    iphone_tele_mm: float = 77.0
    dls: int = 20  # retained for CLI parity with train_sim_multiviews


def _create_static_plot(
    fig: go.Figure,
    n: int,
    c2w_matrices: List,
    cam_positions: np.ndarray,
    fov_x: float,
    initial_index: Optional[int],
    options: CameraOptions,
):
    """Configures the Figure for a single, static visualization."""
    print(f"✨ Selecting {n} maximally dispersed cameras...")
    selected_indices = set(
        find_max_min_dispersion_subset(cam_positions, n, initial_index)
    )

    traces = _build_camera_traces(
        selected_indices, c2w_matrices, fov_x, options
    )
    for trace in traces:
        fig.add_trace(trace)

    fig.update_layout(title=f"Camera Selections for n = {n}")


def _create_animated_plot(
    fig: go.Figure,
    n_values: List[int],
    c2w_matrices: List,
    cam_positions: np.ndarray,
    fov_x: float,
    initial_index: Optional[int],
    options: CameraOptions,
):
    """Configures the Figure for an animated visualization."""
    # Create the base frame
    _create_static_plot(
        fig,
        n_values[0],
        c2w_matrices,
        cam_positions,
        fov_x,
        initial_index,
        options,
    )

    # Create animation frames
    animation_frames = []
    for n in n_values:
        print(f"✨ Calculating animation frame for n = {n}...")
        selected_indices = set(
            find_max_min_dispersion_subset(cam_positions, n, initial_index)
        )
        traces = _build_camera_traces(
            selected_indices, c2w_matrices, fov_x, options
        )
        animation_frames.append(go.Frame(name=str(n), data=traces))

    fig.frames = animation_frames

    # Create and configure animation controls
    slider_steps = []
    for n in n_values:
        slider_steps.append(
            {
                "method": "animate",
                "label": str(n),
                "args": [[str(n)], {"mode": "immediate", "frame": {"redraw": True}}],
            }
        )

    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "direction": "left",
                "x": 0.1,
                "y": 0,
                "xanchor": "right",
                "yanchor": "top",
                "pad": {"r": 10, "t": 87},
                "buttons": [
                    {
                        "label": "▶️ Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 200, "redraw": True},
                                "transition": {"duration": 0},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "⏸️ Pause",
                        "method": "animate",
                        "args": [[None], {"mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "steps": slider_steps,
                "currentvalue": {"prefix": "Number of Views: "},
                "pad": {"t": 50},
            }
        ],
    )


def _resolve_n_values(
    requested: Optional[List[int]], total_cameras: int
) -> Tuple[bool, List[int]]:
    """Returns whether an animation is requested and the ordered list of n values."""
    if total_cameras < 1:
        raise ValueError("Expected at least one camera in the transforms file.")

    if not requested or requested == [-1]:
        return False, [total_cameras]

    if len(requested) == 1:
        n = requested[0]
        if n < 1:
            raise ValueError("Number of views must be at least 1.")
        return False, [min(n, total_cameras)]

    if not (2 <= len(requested) <= 3):
        raise ValueError("Animation expects 2 or 3 integer values.")

    start, stop = requested[0], requested[1]
    step = requested[2] if len(requested) == 3 else 1

    if start < 1:
        raise ValueError("Starting number of views must be at least 1.")
    if step < 1:
        raise ValueError("Step for number of views must be at least 1.")
    if stop <= start:
        raise ValueError("Stop must be greater than start for an animation.")

    if stop == -1:
        stop = total_cameras + 1

    n_values = [
        n for n in range(start, stop, step) if 1 <= n <= total_cameras
    ]
    if not n_values:
        raise ValueError(
            "Requested animation range does not intersect with available cameras."
        )
    return True, n_values


def _camera_positions(c2w_matrices: Sequence[np.ndarray]) -> np.ndarray:
    return np.asarray([c2w[:3, 3] for c2w in c2w_matrices])


def _build_camera_traces(
    selected_indices: Set[int],
    c2w_matrices: List,
    fov_x: float,
    options: CameraOptions,
) -> List[object]:
    """Create mesh and scatter traces for the selected camera indices."""
    if not selected_indices:
        return []

    frustums: List[np.ndarray] = []
    for idx in sorted(selected_indices):
        c2w = c2w_matrices[idx]
        frustums.extend(
            _frustums_for_camera(c2w, fov_x, options)
        )
    if not frustums:
        return []

    mesh = create_frustum_mesh_data(frustums)
    mesh_trace = go.Mesh3d(
        x=mesh.x,
        y=mesh.y,
        z=mesh.z,
        i=mesh.i,
        j=mesh.j,
        k=mesh.k,
        color="#d0d0d0",
        opacity=0.4,
        name="Camera Frustums",
        showscale=False,
    )
    return [mesh_trace]


def _frustums_for_camera(
    c2w: np.ndarray,
    fov_x: float,
    options: CameraOptions,
) -> List[np.ndarray]:
    mode = options.mode
    if mode == "iphone":
        return _iphone_frustums(c2w, fov_x, options)
    if mode == "stereo":
        return _stereo_frustums(c2w, fov_x, options)
    if mode == "multiplexing":
        return _multiplexed_frustums(c2w, fov_x, options)
    return [get_camera_frustum_vertices(c2w, fov_x)]


def _camera_basis_from_c2w(c2w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rotation = c2w[:3, :3]
    center = c2w[:3, 3]
    x_axis = rotation[:, 0]
    y_axis = rotation[:, 1]
    z_axis = -rotation[:, 2]
    x_unit = x_axis / max(np.linalg.norm(x_axis), 1e-12)
    y_unit = y_axis / max(np.linalg.norm(y_axis), 1e-12)
    z_unit = z_axis / max(np.linalg.norm(z_axis), 1e-12)
    return center, x_unit, y_unit, z_unit


def _iphone_frustums(
    c2w: np.ndarray,
    base_fov_x: float,
    options: CameraOptions,
) -> List[np.ndarray]:
    same_focal = options.iphone_same_focal_length
    baseline_x_mm = options.iphone_baseline_x_mm
    baseline_y_mm = options.iphone_baseline_y_mm
    ul_mm = options.iphone_ultrawide_mm
    wide_mm = options.iphone_wide_mm
    tele_mm = options.iphone_tele_mm

    offsets_mm = np.array(
        [
            [-baseline_x_mm, -baseline_y_mm, 0.0],
            [baseline_x_mm, -baseline_y_mm, 0.0],
            [0.0, baseline_y_mm, 0.0],
        ],
        dtype=np.float64,
    )

    if same_focal:
        fov_values = [base_fov_x, base_fov_x, base_fov_x]
    else:
        fov_values = [
            _adjust_fov_for_focal(base_fov_x, wide_mm, ul_mm),
            base_fov_x,
            _adjust_fov_for_focal(base_fov_x, wide_mm, tele_mm),
        ]

    center, x_unit, y_unit, z_unit = _camera_basis_from_c2w(c2w)

    frustums: List[np.ndarray] = []
    for offset_mm, fov in zip(offsets_mm, fov_values):
        offset_local = _mm_vector_to_world(offset_mm)
        offset_world = (
            x_unit * offset_local[0]
            + y_unit * offset_local[1]
            + z_unit * offset_local[2]
        )
        new_c2w = _c2w_with_center(c2w, center + offset_world)
        frustums.append(get_camera_frustum_vertices(new_c2w, fov))
    return frustums


def _stereo_frustums(
    c2w: np.ndarray,
    fov_x: float,
    options: CameraOptions,
) -> List[np.ndarray]:
    if options.object_center is None or options.angle_deg <= 0.0:
        return [get_camera_frustum_vertices(c2w, fov_x)]

    center, x_unit, _, _ = _camera_basis_from_c2w(c2w)
    base_vec = center - options.object_center
    offset = _solve_stereo_offset(base_vec, x_unit, options.angle_deg)
    if offset <= 0.0:
        return [get_camera_frustum_vertices(c2w, fov_x)]

    left_center = center - offset * x_unit
    right_center = center + offset * x_unit

    return [
        get_camera_frustum_vertices(_c2w_with_center(c2w, left_center), fov_x),
        get_camera_frustum_vertices(_c2w_with_center(c2w, right_center), fov_x),
    ]


def _multiplexed_frustums(
    c2w: np.ndarray,
    fov_x: float,
    options: CameraOptions,
) -> List[np.ndarray]:
    if options.object_center is None or options.angle_deg <= 0.0:
        return [get_camera_frustum_vertices(c2w, fov_x)]

    center, x_unit, y_unit, _ = _camera_basis_from_c2w(c2w)
    base_vec = center - options.object_center
    stereo_offset = _solve_stereo_offset(base_vec, x_unit, options.angle_deg)
    if stereo_offset <= 0.0:
        stereo_offset = 0.0

    grid_size = max(int(math.sqrt(options.n_multiplexed_images)), 1)
    lattice = list(product(np.linspace(-1.0, 1.0, grid_size), repeat=2))

    frustums: List[np.ndarray] = []
    for idx, (gx, gy) in enumerate(lattice):
        if idx >= options.n_multiplexed_images:
            break
        offset_center = center + stereo_offset * gx * x_unit + stereo_offset * gy * y_unit
        frustums.append(
            get_camera_frustum_vertices(_c2w_with_center(c2w, offset_center), fov_x)
        )
    return frustums if frustums else [get_camera_frustum_vertices(c2w, fov_x)]


def _solve_stereo_offset(base_vec: np.ndarray, axis: np.ndarray, angle_deg: float) -> float:
    if angle_deg <= 0.0:
        return 0.0
    offset = solve_offset_for_angle(base_vec, axis, angle_deg)
    if not np.isfinite(offset) or offset <= 0.0:
        distance = np.linalg.norm(base_vec)
        offset = distance * math.tan(math.radians(angle_deg) / 2.0)
    if not np.isfinite(offset) or offset < 0.0:
        return 0.0
    return float(offset)


def _mm_vector_to_world(offset_mm: np.ndarray) -> np.ndarray:
    return np.array([mm_to_world(float(value)) for value in offset_mm], dtype=np.float64)


def _c2w_with_center(c2w: np.ndarray, center: np.ndarray) -> np.ndarray:
    updated = c2w.copy()
    updated[:3, 3] = center
    return updated


def _adjust_fov_for_focal(base_fov: float, base_focal_mm: float, target_focal_mm: float) -> float:
    if target_focal_mm <= 0.0 or base_focal_mm <= 0.0:
        return base_fov
    ratio = base_focal_mm / target_focal_mm
    tan_half = np.tan(base_fov / 2.0) * ratio
    tan_half = np.clip(tan_half, 1e-6, 1e6)
    new_fov = 2.0 * np.arctan(tan_half)
    return float(np.clip(new_fov, 1e-4, np.pi - 1e-4))


def visualize_cameras(
    json_path: str,
    n_train_images: Optional[List[int]] = None,
    options: Optional[CameraOptions] = None,
    world_to_m: float = 99.04 / 1000,
) -> None:
    print(f"📸 Loading cameras from {json_path}...")
    with open(json_path, "r") as f:
        data = json.load(f)

    configure_world_to_m(world_to_m)
    camera_options = options or CameraOptions()

    if isinstance(data, dict) and "frames" in data:
        frames, fov_x = data["frames"], data["camera_angle_x"]
        c2w_matrices = [np.array(frame["transform_matrix"]) for frame in frames]
    elif isinstance(data, list):
        c2w_matrices = []
        for frame in data:
            R = np.array(frame["rotation"]).T
            T = np.array(frame["position"])
            w2c = np.eye(4)
            w2c[:3, :3] = R
            w2c[:3, 3] = T
            c2w = np.linalg.inv(w2c)
            c2w[:3, 1:3] *= -1
            c2w_matrices.append(c2w)

        W = data[0].get("width")
        fx = data[0].get("fx")
        fov_x = 2 * np.arctan(W / (2 * fx))

    cam_positions = _camera_positions(c2w_matrices)
    total_cameras = len(c2w_matrices)
    if (
        camera_options.object_center is None
        and camera_options.mode in {"stereo", "multiplexing"}
        and total_cameras > 0
    ):
        camera_options.object_center = np.mean(cam_positions, axis=0)

    is_animation, n_values = _resolve_n_values(n_train_images, total_cameras)
    initial_index = (
        min(59, total_cameras - 1) if total_cameras > 0 else None
    )

    fig = go.Figure()
    if not is_animation:
        n = n_values[0]
        _create_static_plot(
            fig,
            n,
            c2w_matrices,
            cam_positions,
            fov_x,
            initial_index,
            camera_options,
        )
        output_filename = f"static_visualization_n{n}.html"
    else:
        _create_animated_plot(
            fig,
            n_values,
            c2w_matrices,
            cam_positions,
            fov_x,
            initial_index,
            camera_options,
        )
        output_filename = (
            f"camera_animation_{n_values[0]}_{n_values[-1]}.html"
        )

    fig.update_layout(
        scene=dict(
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False,
            aspectmode="data",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    print(f"💾 Saving interactive visualization to {output_filename}")
    fig.write_html(output_filename)
    # fig.write_image("interactive_visualization.png", width=800, height=600)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize camera frustums.")
    parser.add_argument(
        "--source_path", type=str, required=True, help="Path to transforms_train.json"
    )
    parser.add_argument(
        "--n_train_images",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Number of training views to visualise. "
            "Provide a single value for a static plot or start/stop[/step] "
            "for an animation. Omit to use all available views."
        ),
    )
    parser.add_argument(
        "--use_multiplexing",
        action="store_true",
        help="Render multiplexed sub-views for each selected camera (matching train_sim_multiviews).",
    )
    parser.add_argument(
        "--use_stereo",
        action="store_true",
        help="Render stereo camera pairs for each selected camera.",
    )
    parser.add_argument(
        "--use_iphone",
        action="store_true",
        help="Render iPhone triple-lens camera clusters for each selected camera.",
    )
    parser.add_argument(
        "--angle_deg",
        type=float,
        default=10.0,
        help="Baseline angle in degrees used for stereo/multiplexing offsets (same as train_sim_multiviews).",
    )
    parser.add_argument(
        "--n_multiplexed_images",
        type=int,
        default=16,
        help="Number of multiplexed sub-views to visualise when --use_multiplexing is set.",
    )
    parser.add_argument(
        "--dls",
        type=int,
        default=20,
        help="Microlens-to-sensor distance used for multiplexing runs (for parity with training flags).",
    )
    parser.add_argument(
        "--iphone_same_focal_length",
        action="store_true",
        help="When using the iPhone model, keep all lenses at the base focal length.",
    )
    parser.add_argument(
        "--iphone_baseline_x_mm",
        type=float,
        default=9.5,
        help="Horizontal baseline (mm) between iPhone lenses.",
    )
    parser.add_argument(
        "--iphone_baseline_y_mm",
        type=float,
        default=9.5,
        help="Vertical baseline (mm) between iPhone lenses.",
    )
    parser.add_argument(
        "--iphone_ultrawide_mm",
        type=float,
        default=13.0,
        help="Focal length (mm) for the ultra-wide iPhone lens.",
    )
    parser.add_argument(
        "--iphone_wide_mm",
        type=float,
        default=24.0,
        help="Focal length (mm) for the main wide iPhone lens.",
    )
    parser.add_argument(
        "--iphone_tele_mm",
        type=float,
        default=77.0,
        help="Focal length (mm) for the telephoto iPhone lens.",
    )
    parser.add_argument(
        "--world_to_m",
        type=float,
        default=99.04 / 1000.0,
        help="Meters per world unit (used to convert millimeter offsets, matches train_sim_multiviews default).",
    )
    parser.add_argument(
        "--object_center",
        type=float,
        nargs=3,
        default=None,
        help="Optional object center override (x y z). Defaults to the mean camera position.",
    )
    args = parser.parse_args()

    enabled_modes = [
        ("multiplexing", args.use_multiplexing),
        ("stereo", args.use_stereo),
        ("iphone", args.use_iphone),
    ]
    active = [name for name, flag in enabled_modes if flag]
    if len(active) > 1:
        raise ValueError("Please enable at most one of --use_multiplexing, --use_stereo, or --use_iphone.")

    mode = active[0] if active else "default"

    object_center = (
        np.asarray(args.object_center, dtype=np.float64)
        if args.object_center is not None
        else None
    )

    options = CameraOptions(
        mode=mode,
        angle_deg=float(args.angle_deg),
        n_multiplexed_images=int(args.n_multiplexed_images),
        object_center=object_center,
        iphone_same_focal_length=bool(args.iphone_same_focal_length),
        iphone_baseline_x_mm=float(args.iphone_baseline_x_mm),
        iphone_baseline_y_mm=float(args.iphone_baseline_y_mm),
        iphone_ultrawide_mm=float(args.iphone_ultrawide_mm),
        iphone_wide_mm=float(args.iphone_wide_mm),
        iphone_tele_mm=float(args.iphone_tele_mm),
        dls=int(args.dls),
    )

    visualize_cameras(
        args.source_path,
        args.n_train_images,
        options,
        world_to_m=float(args.world_to_m),
    )
