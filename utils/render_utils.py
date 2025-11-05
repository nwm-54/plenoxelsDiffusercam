from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData, PlyElement

from arguments import PipelineParams
from gaussian_renderer import render
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from utils.general_utils import get_dataset_name
from utils.graphics_utils import fov2focal, getWorld2View2

if TYPE_CHECKING:
    from arguments import ModelParams
    from scene.scene_utils import CameraInfo, SceneInfo

PLYS_ROOT = Path("/home/wl757/multiplexed-pixels/plenoxels/plys")
BLENDER_CACHE_ROOT = Path(
    "/share/monakhova/shamus_data/multiplexed_pixels/blender_cache"
)


def get_pretrained_splat_path(args: ModelParams) -> Optional[str]:
    if args.pretrained_ply and os.path.exists(args.pretrained_ply):
        return args.pretrained_ply

    candidate = PLYS_ROOT / f"{get_dataset_name(args.source_path)}.ply"
    if candidate.exists():
        return str(candidate)

    return None


def load_pretrained_splat(path: str, sh_degree: int = 3) -> GaussianModel:
    if path is None:
        raise ValueError("Expected a valid pretrained ply path, received None")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pretrained ply file not found at {path}")
    gs = GaussianModel(sh_degree)
    gs.load_ply(path)
    print(f"Loaded pretrained ply: {path}")
    return gs


def render_splat(
    args: ModelParams, gs: GaussianModel, scene_info: SceneInfo
) -> SceneInfo:
    dummy_args = ArgumentParser()
    pp = PipelineParams(dummy_args).extract(dummy_args.parse_args([]))

    bg = torch.tensor([0.0, 0.0, 0.0], device=args.data_device, dtype=torch.float32)
    out_dir = os.path.join(args.model_path, "input_views")
    os.makedirs(out_dir, exist_ok=True)

    def _render_cam_list(cam_infos: List[CameraInfo]) -> List[CameraInfo]:
        updated: List[CameraInfo] = []
        for cam_info in cam_infos:
            tmp_camera = Camera(
                colmap_id=cam_info.uid,
                R=cam_info.R,
                T=cam_info.T,
                FoVx=cam_info.FovX,
                FoVy=cam_info.FovY,
                image=torch.zeros(
                    (3, cam_info.height, cam_info.width),
                    device=args.data_device,
                    dtype=torch.float32,
                ),
                gt_alpha_mask=None,
                mask=cam_info.mask,
                image_name=cam_info.image_name,
                uid=0,
                group_id=getattr(cam_info, "groupid", 0),
                data_device=args.data_device,
            )

            new_image = render_splat_from_camera(tmp_camera, gs, pp, bg)

            png_name = f"{cam_info.image_name}.png"
            png_path = os.path.join(out_dir, png_name)
            new_image.save(png_path, format="PNG")

            updated.append(cam_info._replace(image=new_image, image_path=png_path))
        return updated

    new_train_cameras = copy.deepcopy(scene_info.train_cameras)
    for view_index, cam_info_list in new_train_cameras.items():
        new_train_cameras[view_index] = _render_cam_list(cam_info_list)

    new_test_cameras = _render_cam_list(list(scene_info.test_cameras))
    new_full_test_cameras = _render_cam_list(list(scene_info.full_test_cameras))

    return scene_info._replace(
        train_cameras=new_train_cameras,
        test_cameras=new_test_cameras,
        full_test_cameras=new_full_test_cameras,
    )


def _camera_info_to_opengl_transform(cam_info: "CameraInfo") -> np.ndarray:
    w2c = getWorld2View2(cam_info.R, cam_info.T).astype(np.float64)
    c2w = np.linalg.inv(w2c)
    c2w[:3, 1:3] *= -1.0
    return c2w


def _camera_info_to_frame_entry(
    cam_info: "CameraInfo", file_stub: str, split: str
) -> Dict[str, object]:
    width, height = cam_info.image.size
    transform = _camera_info_to_opengl_transform(cam_info)
    fl_x = float(fov2focal(cam_info.FovX, width))
    fl_y = float(fov2focal(cam_info.FovY, height))

    entry: Dict[str, object] = {
        "file_path": file_stub,
        "transform_matrix": transform.tolist(),
        "fl_x": fl_x,
        "fl_y": fl_y,
        "cx": width / 2.0,
        "cy": height / 2.0,
        "w": width,
        "h": height,
        "split": split,
    }

    return entry


def build_transforms_data(
    scene_info: "SceneInfo", include_test: bool = False
) -> Tuple[Optional[Dict[str, object]], Dict[Tuple[int, int], Dict[str, object]]]:
    frames: List[Dict[str, object]] = []
    train_lookup: Dict[Tuple[int, int], Dict[str, object]] = {}

    first_train_entry: Optional[Dict[str, object]] = None
    first_train_cam: Optional["CameraInfo"] = None

    camera_name_map: Dict[int, str] = {}
    frame_counters: Dict[str, int] = defaultdict(int)

    def _camera_name_for(cam_uid: int) -> str:
        if cam_uid not in camera_name_map:
            camera_name_map[cam_uid] = f"camera_{len(camera_name_map) + 1:02d}"
        return camera_name_map[cam_uid]

    def _file_stub(cam_uid: int) -> Tuple[str, int]:
        cam_name = _camera_name_for(cam_uid)
        frame_idx = frame_counters[cam_name]
        frame_counters[cam_name] += 1
        return f"./{cam_name}/r_{frame_idx:03d}", frame_idx

    for view_idx, cam_list in sorted(scene_info.train_cameras.items()):
        for cam_idx, cam_info in enumerate(cam_list):
            file_stub, _ = _file_stub(cam_info.uid)
            entry = _camera_info_to_frame_entry(cam_info, file_stub, split="train")
            frames.append(entry)

            if first_train_entry is None:
                first_train_entry = entry
                first_train_cam = cam_info

            train_lookup[(view_idx, cam_idx)] = {
                "file_stub": file_stub,
                "image_name": cam_info.image_name
                or f"view_{view_idx:03d}_cam_{cam_idx:02d}",
                "cam_info": cam_info,
            }

    if include_test and scene_info.test_cameras:
        for cam_info in scene_info.test_cameras:
            file_stub, _ = _file_stub(cam_info.uid)
            frames.append(
                _camera_info_to_frame_entry(cam_info, file_stub, split="test")
            )

    if include_test and scene_info.full_test_cameras:
        for cam_info in scene_info.full_test_cameras:
            file_stub, _ = _file_stub(cam_info.uid)
            frames.append(
                _camera_info_to_frame_entry(cam_info, file_stub, split="full")
            )

    if not frames or first_train_entry is None or first_train_cam is None:
        return None, train_lookup

    transforms = {
        "camera_angle_x": float(first_train_cam.FovX),
        "camera_angle_y": float(first_train_cam.FovY),
        "fl_x": first_train_entry["fl_x"],
        "fl_y": first_train_entry["fl_y"],
        "cx": first_train_entry["w"] / 2.0,
        "cy": first_train_entry["h"] / 2.0,
        "w": first_train_entry["w"],
        "h": first_train_entry["h"],
        "frames": frames,
    }

    return transforms, train_lookup


def write_camera_json(
    scene_info: "SceneInfo",
    output_path: os.PathLike | str,
    include_test: bool = False,
    object_center: Optional[np.ndarray] = None,
) -> Tuple[Optional[Dict[str, object]], Dict[Tuple[int, int], Dict[str, object]]]:
    transforms_data, frame_lookup = build_transforms_data(
        scene_info, include_test=include_test
    )
    if transforms_data is None:
        return None, frame_lookup

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if getattr(scene_info, "nerf_normalization", None):
        normalization = scene_info.nerf_normalization
        if normalization:
            transforms_data = copy.deepcopy(transforms_data)
            transforms_data["normalization"] = {
                "translate": normalization.get("translate", []).tolist()
                if hasattr(normalization.get("translate"), "tolist")
                else normalization.get("translate"),
                "radius": float(normalization.get("radius"))
                if normalization.get("radius") is not None
                else None,
            }
    if object_center is not None:
        transforms_data = copy.deepcopy(transforms_data)
        transforms_data["object_center"] = (
            object_center.tolist()
            if hasattr(object_center, "tolist")
            else list(object_center)
        )
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(transforms_data, file, indent=2)

    return transforms_data, frame_lookup


def _build_blender_command(
    blend_file: Path,
    render_script: Path,
    transforms_path: Path,
    tmp_root_path: Path,
    total_frames: int,
) -> List[str]:
    return [
        "conda",
        "run",
        "-n",
        "blender",
        "blender",
        "-b",
        str(blend_file),
        "-P",
        str(render_script),
        "--",
        "--transforms-json",
        str(transforms_path),
        "--results",
        str(tmp_root_path),
        "--views",
        str(max(total_frames, 1)),
        "--disable-animation",
    ]


def _camera_fingerprint(cam_info: "CameraInfo") -> str:
    h = hashlib.sha1()
    h.update(np.asarray(cam_info.R, dtype=np.float32).tobytes())
    h.update(np.asarray(cam_info.T, dtype=np.float32).tobytes())
    h.update(np.asarray([cam_info.FovX, cam_info.FovY], dtype=np.float32).tobytes())
    h.update(np.asarray([cam_info.width, cam_info.height], dtype=np.float32).tobytes())
    return h.hexdigest()


def _load_blender_rgba(rendered_file: Path) -> Image:
    with Image.open(rendered_file) as pil_image:
        return pil_image.convert("RGBA")


def _process_blender_outputs(
    scene_info: "SceneInfo",
    frame_lookup: Dict[Tuple[int, int], Dict[str, object]],
    tmp_root_path: Path,
    input_views_dir: Path,
) -> Tuple[Dict[int, List["CameraInfo"]], int]:
    new_train_cameras: Dict[int, List["CameraInfo"]] = {}
    rendered_count = 0

    for view_idx, cam_list in scene_info.train_cameras.items():
        updated_list: List["CameraInfo"] = []
        for cam_idx, cam_info in enumerate(cam_list):
            meta = frame_lookup.get((view_idx, cam_idx))
            if meta is None:
                updated_list.append(cam_info)
                continue

            file_stub = meta["file_stub"]
            image_name = (
                meta["image_name"]
                or cam_info.image_name
                or f"view_{view_idx:03d}_cam_{cam_idx:02d}"
            )

            rel_path = file_stub[2:] if file_stub.startswith("./") else file_stub
            rendered_file = (tmp_root_path / Path(rel_path)).with_suffix(".png")

            if not rendered_file.exists():
                raise RuntimeError(
                    f"Blender did not produce expected output '{rel_path}' at {rendered_file}."
                )

            image_rgba = _load_blender_rgba(rendered_file)
            final_path = input_views_dir / f"{image_name}.png"
            image_rgba.save(final_path, format="PNG")

            image_rgb = image_rgba.convert("RGB")
            rendered_count += 1
            updated_list.append(
                cam_info._replace(image=image_rgb, image_path=str(final_path))
            )

        new_train_cameras[view_idx] = updated_list

    return new_train_cameras, rendered_count


def render_with_blender(
    args: "ModelParams",
    scene_info: "SceneInfo",
    dataset_name: str,
    object_center: Optional[np.ndarray] = None,
) -> "SceneInfo":
    if not scene_info.train_cameras:
        return scene_info

    transforms_path = Path(args.model_path) / "transforms.json"
    transforms_data, frame_lookup = write_camera_json(
        scene_info, transforms_path, include_test=False, object_center=object_center
    )
    if transforms_data is None:
        return scene_info

    project_root = Path(__file__).resolve().parents[1]
    render_script = project_root / "blender" / "render_blender.py"
    blend_file = project_root / "blender" / f"{dataset_name}.blend"

    if not blend_file.exists():
        raise FileNotFoundError(
            f"Expected Blender scene for '{dataset_name}' at {blend_file}"
        )

    input_views_dir = Path(args.model_path) / "input_views"
    input_views_dir.mkdir(parents=True, exist_ok=True)

    cache_dataset_dir = BLENDER_CACHE_ROOT / dataset_name
    cache_dataset_dir.mkdir(parents=True, exist_ok=True)

    fingerprint_info: Dict[Tuple[int, int], Tuple[Path, str, "CameraInfo"]] = {}
    all_cached = True
    for view_idx, cam_list in scene_info.train_cameras.items():
        for cam_idx, cam_info in enumerate(cam_list):
            fp = _camera_fingerprint(cam_info)
            cache_png = cache_dataset_dir / f"{fp}.png"
            meta = frame_lookup.get((view_idx, cam_idx))
            image_name = (
                meta["image_name"]
                if meta and meta.get("image_name")
                else cam_info.image_name or f"view_{view_idx:03d}_cam_{cam_idx:02d}"
            )
            fingerprint_info[(view_idx, cam_idx)] = (cache_png, image_name, cam_info)
            if not cache_png.exists():
                all_cached = False

    if all_cached and fingerprint_info:
        print(f"Loaded Blender renderings from cache: {cache_dataset_dir}")
        new_train_cameras: Dict[int, List["CameraInfo"]] = defaultdict(list)
        for view_idx, cam_idx in sorted(fingerprint_info.keys()):
            cache_png, image_name, original_cam = fingerprint_info[(view_idx, cam_idx)]
            with Image.open(cache_png) as pil_image:
                image_rgba = pil_image.convert("RGBA")
            final_path = input_views_dir / f"{image_name}.png"
            shutil.copy2(cache_png, final_path)
            new_train_cameras[view_idx].append(
                original_cam._replace(
                    image=image_rgba.convert("RGB"), image_path=str(final_path)
                )
            )
        return scene_info._replace(train_cameras=dict(new_train_cameras))

    with tempfile.TemporaryDirectory(prefix="blender_render_") as tmp_root:
        tmp_root_path = Path(tmp_root)
        cmd = _build_blender_command(
            blend_file, render_script, transforms_path, tmp_root_path, len(frame_lookup)
        )

        total_views = len(fingerprint_info)
        print(
            f"Invoking Blender to render {total_views} training sub-views; this may take a few minutes..."
        )
        render_start = time.perf_counter()

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        stdout_lines: List[str] = []
        stderr_lines: List[str] = []

        def _consume_stream(stream: Optional[object], sink: List[str]) -> None:
            if stream is None:
                return
            try:
                with stream:
                    for line in iter(stream.readline, ""):
                        sink.append(line)
            except Exception:
                # Stream consumption failures should not mask Blender errors.
                pass

        stdout_thread = threading.Thread(
            target=_consume_stream, args=(process.stdout, stdout_lines), daemon=True
        )
        stderr_thread = threading.Thread(
            target=_consume_stream, args=(process.stderr, stderr_lines), daemon=True
        )
        stdout_thread.start()
        stderr_thread.start()
        try:
            while True:
                retcode = process.poll()
                elapsed = time.perf_counter() - render_start
                png_count = sum(1 for _ in tmp_root_path.glob("**/*.png"))
                print(
                    f"Blender render running... {png_count}/{total_views} images rendered, {elapsed:.1f}s elapsed",
                    end="\r",
                    flush=True,
                )
                if retcode is not None:
                    break
                time.sleep(2.0)

        finally:
            try:
                process.wait()
            except Exception:
                pass
            stdout_thread.join()
            stderr_thread.join()
            elapsed = time.perf_counter() - render_start

        stdout = "".join(stdout_lines)
        stderr = "".join(stderr_lines)
        final_png_count = sum(1 for _ in tmp_root_path.glob("**/*.png"))
        print(
            f"Blender render completed in {elapsed:.1f}s with {final_png_count}/{total_views} images{' ' * 20}"
        )

        render_elapsed = time.perf_counter() - render_start

        if process.returncode != 0:
            combined_output = stderr.strip() or stdout.strip()
            raise RuntimeError(
                "Blender rendering failed"
                + (f": {combined_output}" if combined_output else "")
            )

        new_train_cameras, rendered_count = _process_blender_outputs(
            scene_info, frame_lookup, tmp_root_path, input_views_dir
        )

        if rendered_count > 0:
            for (view_idx, cam_idx), (
                cache_png,
                image_name,
                _,
            ) in fingerprint_info.items():
                cam_list = new_train_cameras.get(view_idx)
                if not cam_list or cam_idx >= len(cam_list):
                    continue
                cache_png.parent.mkdir(parents=True, exist_ok=True)
                src_path = input_views_dir / f"{image_name}.png"
                if src_path.exists():
                    shutil.copy2(src_path, cache_png)
        else:
            raise RuntimeError(
                "Blender returned without producing any training images."
            )

    print(
        f"Blender render completed: {rendered_count} train frames in {render_elapsed:.1f}s; outputs stored in {input_views_dir}"
    )
    return scene_info._replace(train_cameras=new_train_cameras)


def render_splat_from_camera(
    camera: Camera, gs: GaussianModel, pp: PipelineParams, bg: torch.Tensor
) -> Image:
    with torch.no_grad():
        rendering = render(camera, gs, pp, bg)["render"]
    new_image_data = (rendering.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255)
    new_image = Image.fromarray(new_image_data.astype(np.uint8), "RGB")
    return new_image


def _to_numpy(tensor_or_array):
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.detach().cpu().numpy()
    return tensor_or_array


def camera_forward(camera: Camera) -> np.ndarray:
    z_cam = np.array([0, 0, 1])
    R = _to_numpy(camera.R)
    forward = R @ z_cam
    return forward / np.linalg.norm(forward)


def camera_center(camera: Camera) -> np.ndarray:
    R = _to_numpy(camera.R)
    T = _to_numpy(camera.T)
    return -(R @ T)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)
