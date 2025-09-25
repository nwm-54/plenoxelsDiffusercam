"""
Refactored Blender rendering script with transforms.json support:
- Can load existing transforms.json files for rendering
- Generates transforms.json structure first, then renders based on that
- Modular camera trajectories (turntable, random full, random upper hemisphere, random static)
- Compatible with plenoxels/blender_data dataset format
- Optional animation of the LEGO scene arm via the Controlpanel_Arm object
- GPU/OPTIX/CUDA/HIP/ONEAPI auto‑selection for Cycles

Usage examples:

  # Generate and render from new camera poses
  blender -b lego.blend -P render_blender.py -- --views 50 --results results

  # Load existing transforms.json and render from those poses
  blender -b lego.blend -P render_blender.py -- --transforms-json path/to/transforms.json --results results

  # Generate transforms.json only (no rendering)
  blender -b lego.blend -P render_blender.py -- --views 120 --generate-only --results results

  # Turntable trajectory with custom parameters
  blender -b lego.blend -P render_blender.py -- \
      --trajectory turntable --turntable-pitch-deg 15 --views 120

  # Multi-camera: generate 3 cameras (camera_00, camera_01, camera_02)
  blender -b lego.blend -P render_blender.py -- --num-cams 3 --views 60

Notes:
- All custom CLI flags must appear after a literal "--" so Blender passes them to Python.
- When using --transforms-json, camera poses are loaded from the file.
- Use --generate-only to create transforms.json without rendering images.
- Use --num-cams to specify number of cameras (default: 1, creates camera_00, camera_01, camera_02, etc.)
- Compatible with NeRF/plenoxels transforms.json format.
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import bpy
import numpy as np
from mathutils import Matrix


@dataclass
class RenderConfig:
    views: int
    resolution: int
    results: str
    img_format: str
    color_depth: int
    trajectory_name: str
    seed: Optional[int] = None
    num_cams: int = 1
    start: int = 0
    end: Optional[int] = None
    turntable_pitch_deg: float = 0.0
    loc_rmin: float = 3.5
    loc_rmax: float = 5.0
    loc_upper: bool = True
    arm_mode: str = "off"
    arm_z: float = 10.0
    arm_z_start: float = 10.0
    arm_z_end: float = 4.0
    debug: bool = False
    transforms_json: Optional[str] = None
    generate_only: bool = False

    def create_arm_animator(self) -> "ArmAnimator":
        return ArmAnimator(
            mode=self.arm_mode,
            z=self.arm_z,
            z_start=self.arm_z_start,
            z_end=self.arm_z_end,
        )


def parse_args(argv: List[str]) -> RenderConfig:
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Render multi-camera Blender dataset")
    parser.add_argument("--views", type=int, default=500)
    parser.add_argument("--resolution", type=int, default=800)
    parser.add_argument("--results", type=str, default="results")
    parser.add_argument("--format", type=str, default="PNG")
    parser.add_argument("--color-depth", dest="color_depth", type=int, default=8)
    parser.add_argument(
        "--trajectory",
        type=str,
        choices=list(TRAJ_REGISTRY.keys()),
        default="random_upper",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-cams", dest="num_cams", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument(
        "--turntable-pitch-deg", dest="turntable_pitch_deg", type=float, default=0.0
    )
    parser.add_argument("--loc-rmin", dest="loc_rmin", type=float, default=3.5)
    parser.add_argument("--loc-rmax", dest="loc_rmax", type=float, default=5.0)
    parser.add_argument("--loc-upper", dest="loc_upper", action="store_true")
    parser.add_argument("--arm-mode", dest="arm_mode", type=str, default="off")
    parser.add_argument("--arm-z", dest="arm_z", type=float, default=10.0)
    parser.add_argument("--arm-z-start", dest="arm_z_start", type=float, default=10.0)
    parser.add_argument("--arm-z-end", dest="arm_z_end", type=float, default=4.0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--transforms-json",
        dest="transforms_json",
        type=str,
        default=None,
        help="Load camera poses from existing transforms.json file",
    )
    parser.add_argument(
        "--generate-only",
        dest="generate_only",
        action="store_true",
        help="Generate transforms.json only, skip rendering",
    )

    parsed = parser.parse_args(argv)

    return RenderConfig(
        views=parsed.views,
        resolution=parsed.resolution,
        results=parsed.results,
        img_format=parsed.format,
        color_depth=parsed.color_depth,
        trajectory_name=parsed.trajectory,
        seed=parsed.seed,
        num_cams=parsed.num_cams,
        start=parsed.start,
        end=parsed.end if parsed.end is not None else parsed.views,
        turntable_pitch_deg=parsed.turntable_pitch_deg,
        loc_rmin=parsed.loc_rmin,
        loc_rmax=parsed.loc_rmax,
        loc_upper=parsed.loc_upper,
        arm_mode=parsed.arm_mode,
        arm_z=parsed.arm_z,
        arm_z_start=parsed.arm_z_start,
        arm_z_end=parsed.arm_z_end,
        debug=parsed.debug,
        transforms_json=parsed.transforms_json,
        generate_only=parsed.generate_only,
    )


def listify_matrix(matrix) -> List[List[float]]:
    return [list(row) for row in matrix]


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def extract_camera_name(file_path: str) -> str:
    """Extract camera name from file path like ./camera_name/r_X"""
    return file_path.split("/")[-2] if "/" in file_path else "Camera"


def resolve_output_root(path_str: str) -> str:
    path_str = path_str or "results"
    if path_str.startswith("//"):
        return bpy.path.abspath(path_str)
    expanded = os.path.expanduser(path_str)
    return (
        os.path.abspath(expanded)
        if os.path.isabs(expanded)
        else os.path.abspath(expanded)
    )


def ensure_camera_exists(name: str, template: bpy.types.Object) -> bpy.types.Object:
    cam = bpy.data.objects.get(name)
    if cam is not None:
        return cam
    if template is None:
        raise RuntimeError(
            f"Cannot auto-create camera '{name}' because template camera is missing."
        )
    new_cam = template.copy()
    new_cam.data = template.data.copy()
    new_cam.name = name
    bpy.context.scene.collection.objects.link(new_cam)
    return new_cam


def enable_cycles_and_gpus() -> None:
    bpy.context.scene.render.engine = "CYCLES"
    scn = bpy.context.scene
    scn.render.use_persistent_data = True
    scn.cycles.use_adaptive_sampling = True
    scn.cycles.use_denoising = True
    scn.cycles.denoiser = "OPTIX"

    try:
        prefs = bpy.context.preferences
        cycles_prefs = prefs.addons["cycles"].preferences
        cycles_prefs.compute_device_type = "OPTIX"
        cycles_prefs.refresh_devices()
        for device in cycles_prefs.devices:
            device.use = device.type == "OPTIX"
        bpy.context.scene.cycles.device = "GPU"
    except Exception as e:
        print("GPU enable skipped:", e)


def configure_image_output(img_format: str, color_depth: int):
    scn = bpy.context.scene
    scn.render.image_settings.file_format = str(img_format)
    scn.render.image_settings.color_depth = str(color_depth)
    try:
        if scn.render.image_settings.file_format not in {"JPEG", "BMP"}:
            scn.render.image_settings.color_mode = "RGBA"
        else:
            scn.render.image_settings.color_mode = "RGB"
    except Exception:
        pass
    scn.render.dither_intensity = 0.0
    scn.render.film_transparent = True


@dataclass
class CameraRig:
    camera: bpy.types.Object
    pivot: bpy.types.Object

    @staticmethod
    def create(camera: bpy.types.Object) -> "CameraRig":
        """Create a unique pivot for this camera and set up a Track To."""
        pivot_name = f"Pivot_{camera.name}"

        pivot = bpy.data.objects.get(pivot_name)
        if pivot is None:
            pivot = bpy.data.objects.new(pivot_name, None)
            pivot.location = (0.0, 0.0, 0.0)
            bpy.context.scene.collection.objects.link(pivot)

        camera.parent = pivot
        for c in list(camera.constraints):
            if c.type == "TRACK_TO":
                camera.constraints.remove(c)
        con = camera.constraints.new(type="TRACK_TO")
        con.target = pivot
        con.track_axis = "TRACK_NEGATIVE_Z"
        con.up_axis = "UP_Y"

        return CameraRig(camera=camera, pivot=pivot)

    def set_pivot_euler(self, euler_xyz: Tuple[float, float, float]):
        self.pivot.rotation_euler = euler_xyz

    def set_camera_location(self, loc_xyz: Tuple[float, float, float]):
        self.camera.location = loc_xyz


class Trajectory:
    def pose(self, i: int, total: int) -> Tuple[float, float, float]:
        raise NotImplementedError


class TurntableTrajectory(Trajectory):
    def __init__(self, pitch_deg: float = 0.0):
        self.pitch = math.radians(pitch_deg)

    def pose(self, i: int, total: int) -> Tuple[float, float, float]:
        yaw = (2.0 * math.pi) * (i / max(1, total))
        return (self.pitch, 0.0, yaw)


class RandomFullEulerTrajectory(Trajectory):
    def pose(self, i: int, total: int) -> Tuple[float, float, float]:
        return tuple(np.random.uniform(0.0, 2.0 * math.pi, size=3))


class RandomUpperHemisphereTrajectory(Trajectory):
    def pose(self, i: int, total: int) -> Tuple[float, float, float]:
        rot = np.random.uniform(0.0, 1.0, size=3) * (1.0, 0.0, 2.0 * math.pi)
        rot[0] = abs(math.acos(1.0 - 2.0 * rot[0]) - math.pi / 2.0)
        return float(rot[0]), float(rot[1]), float(rot[2])


class RandomStaticTrajectory(Trajectory):
    def __init__(self, r_min: float = 3.5, r_max: float = 5.0, upper: bool = True):
        self.r_min = r_min
        self.r_max = r_max
        self.upper = upper
        self._pos = self._sample_position()

    def _sample_position(self) -> Tuple[float, float, float]:
        v = np.random.normal(size=3)
        if self.upper:
            v[2] = abs(v[2])
        v = v / (np.linalg.norm(v) + 1e-9)
        r = np.random.uniform(self.r_min, self.r_max)
        p = (v * r).tolist()
        return (float(p[0]), float(p[1]), float(p[2]))

    def apply(self, rig: CameraRig, i: int, total: int):
        rig.set_camera_location(self._pos)
        rig.set_pivot_euler((0.0, 0.0, 0.0))


TRAJ_REGISTRY = {
    "turntable": TurntableTrajectory,
    "random_full": RandomFullEulerTrajectory,
    "random_upper": RandomUpperHemisphereTrajectory,
    "random_static": RandomStaticTrajectory,
}


class ArmAnimator:
    def __init__(
        self,
        mode: str = "off",
        z: float = 10.0,
        z_start: float = 10.0,
        z_end: float = 4.0,
    ):
        self.mode = mode
        self.z = z
        self.z_start = z_start
        self.z_end = z_end
        self.obj = bpy.data.objects.get("Controlpanel_Arm")
        if self.obj is None and mode != "off":
            print("ArmAnimator: object Controlpanel_Arm not found; animation disabled.")
            self.mode = "off"

    def apply(self, i: int, total: int):
        if self.mode == "off" or self.obj is None:
            return
        if self.mode == "fixed":
            self.obj.location.z = float(self.z)
        elif self.mode == "range":
            t = 0.0 if total <= 1 else i / (total - 1)
            z_val = (1.0 - t) * self.z_start + t * self.z_end
            self.obj.location.z = float(z_val)
        else:
            print(f"ArmAnimator: unknown mode {self.mode}; skipping.")


def load_transforms_json(filepath: str) -> dict:
    """Load transforms.json file and return the data structure."""
    with open(filepath, "r") as f:
        return json.load(f)


def save_transforms_json(data: dict, filepath: str) -> None:
    """Save transforms data to JSON file."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {filepath}")


def generate_transforms(
    cfg: RenderConfig,
    cameras: List[bpy.types.Object],
    rigs: Dict[str, CameraRig],
    traj,
    arm_anim: ArmAnimator,
    fixed_positions: Dict[str, Tuple[float, float, float]] = None,
) -> dict:
    assert cameras, "No cameras provided for transforms generation"
    camera_angle_x = float(cameras[0].data.angle_x)

    out_data = {
        "camera_angle_x": camera_angle_x,
        "frames": [],
    }

    start = max(0, cfg.start)
    end = cfg.end if cfg.end is not None else cfg.views

    for i in range(start, min(end, cfg.views)):
        arm_anim.apply(i, cfg.views)

        for cam in cameras:
            rig = rigs[cam.name]

            if cfg.trajectory_name == "random_static":
                rig.set_camera_location(fixed_positions[cam.name])
                rig.set_pivot_euler((0.0, 0.0, 0.0))
            elif hasattr(traj, "apply"):
                traj.apply(rig, i, cfg.views)
            else:
                rig.set_pivot_euler(traj.pose(i, cfg.views))

            # Make this camera active to update matrix_world
            bpy.context.scene.camera = cam
            bpy.context.view_layer.update()

            rel_path = f"./{cam.name}/r_{i}"

            frame_entry = {
                "file_path": rel_path,
                "transform_matrix": listify_matrix(cam.matrix_world),
            }
            out_data["frames"].append(frame_entry)

    return out_data


def render(
    cfg: RenderConfig, transforms_data: dict, cameras: List[bpy.types.Object], root: str
) -> None:
    frames = transforms_data.get("frames", [])
    if not frames:
        print("No frames found in transforms data")
        return

    rendered_frames = 0
    start_time = time.time()

    # Create arm animator for rendering phase
    arm_anim = cfg.create_arm_animator()

    camera_frames = {}
    for frame in frames:
        cam_name = extract_camera_name(frame["file_path"])

        if cam_name not in camera_frames:
            camera_frames[cam_name] = []
        camera_frames[cam_name].append(frame)

    camera_map = {cam.name: cam for cam in cameras}

    for cam_name, cam_frames in camera_frames.items():
        if cam_name not in camera_map:
            print(
                f"Warning: Camera '{cam_name}' from transforms.json not found in scene"
            )
            continue

        cam = camera_map[cam_name]
        out_dir = os.path.join(root, cam_name)
        ensure_dir(out_dir)

        for frame in cam_frames:
            transform_matrix = frame["transform_matrix"]
            cam.matrix_world = Matrix(transform_matrix)

            # Extract frame index and apply arm animation
            file_path = frame["file_path"]
            frame_idx = int(file_path.split("_")[-1])

            # Apply arm animation for this frame
            arm_anim.apply(frame_idx, cfg.views)

            bpy.context.scene.camera = cam
            bpy.context.view_layer.update()

            scn = bpy.context.scene
            scn.render.filepath = os.path.join(out_dir, f"r_{frame_idx}")

            if not cfg.debug:
                bpy.ops.render.render(write_still=True)

            rendered_frames += 1

    if rendered_frames > 0:
        elapsed = time.time() - start_time
        print(f"Rendered {rendered_frames} frames in {elapsed:.1f} seconds.")


def run(cfg: RenderConfig):
    if cfg.seed is not None:
        np.random.seed(cfg.seed)

    enable_cycles_and_gpus()
    configure_image_output(cfg.img_format, cfg.color_depth)

    arm_anim = cfg.create_arm_animator()

    scn = bpy.context.scene
    scn.render.resolution_x = cfg.resolution
    scn.render.resolution_y = cfg.resolution
    scn.render.resolution_percentage = 100

    root = resolve_output_root(cfg.results)
    ensure_dir(root)

    transforms_data = None

    if cfg.transforms_json:
        print(f"Loading transforms from {cfg.transforms_json}")
        transforms_data = load_transforms_json(cfg.transforms_json)

        camera_names = {
            extract_camera_name(frame["file_path"])
            for frame in transforms_data.get("frames", [])
        }

        cfg.num_cams = len(camera_names) if camera_names else 1

    else:
        print(f"Generating new camera poses using trajectory: {cfg.trajectory_name}")

        traj_kwargs = {
            "turntable": {"pitch_deg": cfg.turntable_pitch_deg},
            "random_static": {
                "r_min": cfg.loc_rmin,
                "r_max": cfg.loc_rmax,
                "upper": cfg.loc_upper,
            },
        }
        Trajectory = TRAJ_REGISTRY[cfg.trajectory_name]
        traj = Trajectory(**traj_kwargs.get(cfg.trajectory_name, {}))

    def find_template_camera():
        template_cam = bpy.data.objects.get("Camera")
        if template_cam is None:
            for obj in bpy.data.objects:
                if obj.type == "CAMERA":
                    return obj
        return template_cam

    template_cam = find_template_camera()
    if template_cam is None:
        raise RuntimeError(
            "No template camera found. Please ensure the scene contains at least one camera."
        )

    if cfg.transforms_json:
        camera_names = {
            extract_camera_name(frame["file_path"])
            for frame in transforms_data.get("frames", [])
        }
        cameras = [
            bpy.data.objects.get(name) or ensure_camera_exists(name, template_cam)
            for name in sorted(camera_names)
        ]
    else:
        cameras = [
            ensure_camera_exists(f"camera_{i:02d}", template_cam)
            for i in range(cfg.num_cams)
        ]

    if not cameras:
        raise RuntimeError("No valid cameras found.")

    if not cfg.transforms_json:
        rigs: Dict[str, CameraRig] = {}
        for cam in cameras:
            rig = CameraRig.create(cam)
            rig.set_camera_location((0.0, 4.0, 0.5))
            rigs[cam.name] = rig

        fixed_positions: Dict[str, Tuple[float, float, float]] = {}
        if cfg.trajectory_name == "random_static":

            def sample_pos() -> Tuple[float, float, float]:
                v = np.random.normal(size=3)
                if cfg.loc_upper:
                    v[2] = abs(v[2])
                v = v / (np.linalg.norm(v) + 1e-9)
                r = np.random.uniform(cfg.loc_rmin, cfg.loc_rmax)
                p = (v * r).tolist()
                return (float(p[0]), float(p[1]), float(p[2]))

            for cam in cameras:
                fixed_positions[cam.name] = sample_pos()

        transforms_data = generate_transforms(
            cfg, cameras, rigs, traj, arm_anim, fixed_positions
        )

        json_path = os.path.join(root, "transforms.json")
        save_transforms_json(transforms_data, json_path)

        if cfg.generate_only:
            print("Generated transforms.json only (no rendering performed)")
            return

    if not cfg.generate_only:
        print("Rendering images from transforms data...")
        render(cfg, transforms_data, cameras, root)


if __name__ == "__main__":
    cfg = parse_args(sys.argv)
    run(cfg)
