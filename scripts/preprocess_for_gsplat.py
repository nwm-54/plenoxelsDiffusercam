#!/usr/bin/env python3
"""Preprocess dog captures for gsplat (static) or 4DGaussians (dynamic) pipelines."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Sequence


VIDEO_EXTS = {
    ".mov",
    ".mp4",
    ".m4v",
    ".avi",
    ".mkv",
    ".mpg",
    ".mpeg",
    ".webm",
}


@dataclass
class Task:
    kind: str  # "lfr" or "video"
    source: Path
    label: str


def slugify(name: str) -> str:
    name = name.replace(os.sep, "-")
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    sanitized = sanitized.strip("_")
    return sanitized or "scene"


def run_command(cmd: Sequence[str], cwd: Path | None = None) -> None:
    printable = " ".join(shlex.quote(str(part)) for part in cmd)
    print(f"$ {printable}")
    proc = subprocess.run(cmd, cwd=cwd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {printable}")


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_output_root(repo_root: Path) -> Path:
    return (repo_root / ".." / "gsplat" / "data").resolve()


def default_fourdgs_root(repo_root: Path) -> Path:
    return (repo_root / ".." / "4DGaussians").resolve()


def default_vggt_script() -> Path:
    return Path.home() / "repos" / "vggt" / "demo_colmap.py"


def default_lf_script(repo_root: Path) -> Path:
    return repo_root / "lf_to_colmap.py"


def default_video_to_4dgs_script(repo_root: Path) -> Path:
    return default_fourdgs_root(repo_root) / "video_to_4dgs.py"


def default_lf_calib_path() -> Path:
    return (
        Path.home()
        / "multiplexed-pixels"
        / "plenopticam"
        / "cornell-lightfield"
        / "caldata-B5143806010.tar"
    )


def infer_calibration(repo_root: Path) -> Path | None:
    # Use the specific calibration file
    specific_calib = Path.home() / "multiplexed-pixels" / "plenopticam" / "cornell-lightfield" / "caldata-B5143806010.tar"
    if specific_calib.exists():
        return specific_calib

    # Fallback to searching
    candidates: List[Path] = [
        repo_root.parent / "plenopticam" / "data",
        repo_root.parent / "plenopticam" / "cornell-lightfield",
    ]
    for base in candidates:
        if not base.exists():
            continue
        for tar in sorted(base.glob("caldata-*.tar")):
            return tar
    return None


@dataclass
class InputSpec:
    path: Path
    base: Path
    kind: str  # 'dynamic', 'multi_video', or 'file'


def determine_input_spec(input_path: Path) -> InputSpec:
    candidate = input_path.expanduser().resolve()

    if not candidate.exists():
        raise FileNotFoundError(f"Input path does not exist: {candidate}")

    if candidate.is_file():
        return InputSpec(candidate, candidate.parent, "file")

    # Directory handling
    video_files = sorted(
        p for p in candidate.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )
    lfr_files = sorted(
        p for p in candidate.iterdir() if p.is_file() and p.suffix.lower() == ".lfr"
    )

    supported = video_files + lfr_files
    if not supported:
        raise RuntimeError(f"No video or .lfr files found in {candidate}")

    if len(supported) == 1:
        single = supported[0]
        return InputSpec(single, single.parent, "file")

    if video_files and not lfr_files:
        # batch of videos
        return InputSpec(candidate, candidate, "multi_video")

    # Multiple candidates. If all videos originate in same directory, allow batching.
    names = ", ".join(p.name for p in supported[:5])
    raise RuntimeError(
        f"Multiple .lfr files found in {candidate}. "
        "Specify one explicitly with --input. "
        f"Found: {names}{'...' if len(supported) > 5 else ''}"
    )


def build_task_for_file(base_root: Path, file_path: Path) -> Task:
    try:
        rel_posix = PurePosixPath(file_path.relative_to(base_root).as_posix())
    except ValueError:
        rel_posix = PurePosixPath(file_path.name)
    suffix = file_path.suffix.lower()
    if suffix == ".lfr":
        label_base = f"{rel_posix.with_suffix('').as_posix()}_lfr"
        return Task("lfr", file_path, slugify(label_base))
    if suffix in VIDEO_EXTS:
        label_base = rel_posix.with_suffix("").as_posix()
        return Task("video", file_path, slugify(label_base))
    raise RuntimeError(f"Unsupported input type: {file_path}")


def ensure_ready_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def process_lfr(task: Task, out_dir: Path, lf_script: Path, calib_tar: Path, inner: int | None, downscale: float | None) -> None:
    cmd = [
        sys.executable,
        str(lf_script),
        "--lfr",
        str(task.source),
        "--calib_tar",
        str(calib_tar),
        "--out_dir",
        str(out_dir),
    ]
    if inner is not None:
        cmd.extend(["--inner", str(inner)])
    if downscale is not None:
        cmd.extend(["--downscale", str(downscale)])
    run_command(cmd)


def extract_frames(video_path: Path, images_dir: Path, ffmpeg_bin: str, fps: float | None, overwrite: bool = False) -> int:
    """Extract frames from video. Returns the number of frames extracted."""
    # Check if frames already exist
    if images_dir.exists() and not overwrite:
        existing_frames = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))
        if existing_frames:
            print(f"  Using existing {len(existing_frames)} frames in {images_dir}")
            return len(existing_frames)

    ensure_ready_dir(images_dir, overwrite=True)
    output_pattern = images_dir / "frame_%05d.png"
    cmd: List[str] = [ffmpeg_bin, "-y", "-i", str(video_path)]
    if fps is not None:
        cmd.extend(["-vf", f"fps={fps}"])
    else:
        cmd.extend(["-fps_mode", "passthrough"])
    cmd.append(str(output_pattern))
    run_command(cmd)

    extracted_frames = list(images_dir.glob("*.png"))
    if not extracted_frames:
        raise RuntimeError(f"No frames extracted from {video_path}")

    print(f"  Extracted {len(extracted_frames)} frames")
    return len(extracted_frames)


def normalize_frame_counts(image_dirs: List[Path]) -> None:
    """Ensure all image directories have the same number of frames by removing extras."""
    if len(image_dirs) <= 1:
        return

    frame_counts: Dict[Path, List[Path]] = {}
    for images_dir in image_dirs:
        frames = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))
        if frames:
            frame_counts[images_dir] = frames

    if len(frame_counts) <= 1:
        return

    min_count = min(len(frames) for frames in frame_counts.values())
    max_count = max(len(frames) for frames in frame_counts.values())
    if min_count == max_count:
        print(f"All videos share {min_count} frame(s); no normalization needed.")
        return

    print(
        f"Frame count mismatch detected (min={min_count}, max={max_count}); "
        f"trimming to {min_count} frame(s)."
    )
    for images_dir, frames in frame_counts.items():
        if len(frames) <= min_count:
            continue
        extras = frames[min_count:]
        print(f"  Removing {len(extras)} excess frame(s) from {images_dir}")
        for frame_path in extras:
            frame_path.unlink()


def run_vggt(scene_dir: Path, conda_exe: str, conda_env: str, vggt_script: Path, extra_args: Sequence[str]) -> None:
    images_dir = scene_dir / "images"
    frame_count = len(list(images_dir.glob("*.png"))) + len(list(images_dir.glob("*.jpg")))

    cmd: List[str] = [
        conda_exe,
        "run",
        "-n",
        conda_env,
        "python",
        str(vggt_script),
        "--scene_dir",
        str(scene_dir),
        "--use_ba",
    ]
    if frame_count > 0:
        cmd.extend(["--query_frame_num", str(frame_count)])
    cmd.extend(extra_args)
    run_command(cmd)


def process_video_static(
    task: Task,
    out_root: Path,
    conda_exe: str,
    conda_env: str,
    vggt_script: Path,
    ffmpeg_bin: str,
    fps: float | None,
    stage: str,
    stage_cache: str | None,
    overwrite: bool,
    base_label: str | None = None,
    extra_args: Optional[List[str]] = None,
) -> None:
    label = base_label or task.label
    out_dir = out_root / label
    sparse_dir = out_dir / "sparse"
    if sparse_dir.exists():
        print(f"Skipping {label}: outputs already exist at {sparse_dir}")
        return

    print(f"Processing video capture: {task.source} -> {out_dir}")
    ensure_ready_dir(out_dir, overwrite=overwrite)

    images_dir = out_dir / "images"
    extract_frames(task.source, images_dir, ffmpeg_bin, fps, overwrite=overwrite)

    vggt_args: List[str] = []
    if stage:
        vggt_args.extend(["--stage", stage])
    if stage_cache:
        vggt_args.extend(["--stage_cache", stage_cache])
    if extra_args:
        vggt_args.extend(extra_args)

    run_vggt(out_dir, conda_exe, conda_env, vggt_script, vggt_args)


def process_multiple_videos_static(
    video_files: List[Path],
    out_root: Path,
    base_root: Path,
    conda_exe: str,
    conda_env: str,
    vggt_script: Path,
    ffmpeg_bin: str,
    fps: float | None,
    stage: str,
    stage_cache: str | None,
    overwrite: bool,
) -> None:
    """Process multiple videos with frame extraction, normalization, and VGGT."""
    print(f"Found {len(video_files)} videos, processing as a batch...")

    # Phase 1: Extract frames from all videos
    print("\n=== Phase 1: Extracting frames ===")
    sorted_videos = sorted(video_files)

    tmp_root = out_root / "_tmp_frames"
    if overwrite and tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    image_dirs: List[Path] = []
    tasks: List[Task] = []
    for video_file in sorted_videos:
        task = build_task_for_file(base_root, video_file)
        images_dir = tmp_root / task.label
        extract_frames(video_file, images_dir, ffmpeg_bin, fps, overwrite=overwrite)
        image_dirs.append(images_dir)
        tasks.append(task)

    if not image_dirs:
        print("No frames extracted; aborting.")
        return

    print("\n=== Phase 2: Normalizing frame counts ===")
    normalize_frame_counts(image_dirs)

    print("\n=== Phase 3: Generating combined dataset ===")
    combined_scene = out_root / "combined"
    combined_images = combined_scene / "images"
    if not overwrite and (combined_scene / "sparse").exists():
        print("Combined sparse exists and overwrite disabled; skipping recompute.")
    else:
        ensure_ready_dir(combined_images, overwrite=overwrite)
        copied = 0
        for task, images_dir in zip(tasks, image_dirs):
            prefix = task.label
            for img in sorted(images_dir.glob("*.png")):
                shutil.copy2(img, combined_images / f"{prefix}__{img.name}")
                copied += 1
        print(f"Combined dataset prepared with {copied} frame(s).")
        run_vggt_on_images(
            combined_scene,
            conda_exe,
            conda_env,
            vggt_script,
            stage,
            stage_cache,
            overwrite,
        )

    print("\n=== Phase 4: Generating main-camera dataset ===")

    def camera_rank(path: Path) -> tuple[int, str]:
        stem = path.stem.lower()
        if "right" in stem:
            return (0, path.name)
        if "left" in stem:
            return (1, path.name)
        if "wide" in stem:
            return (2, path.name)
        return (3, path.name)

    main_idx = min(range(len(tasks)), key=lambda idx: camera_rank(sorted_videos[idx]))
    main_task = tasks[main_idx]
    main_images_src = image_dirs[main_idx]
    main_scene = out_root / "main_camera"
    main_images = main_scene / "images"
    if not overwrite and (main_scene / "sparse").exists():
        print("Main camera sparse exists and overwrite disabled; skipping recompute.")
    else:
        ensure_ready_dir(main_images, overwrite=overwrite)
        copied_main = 0
        for img in sorted(main_images_src.glob("*.png")):
            shutil.copy2(img, main_images / img.name)
            copied_main += 1
        print(
            f"Main camera selected: {video_files[main_idx].name} with {copied_main} frame(s)."
        )
        run_vggt_on_images(
            main_scene,
            conda_exe,
            conda_env,
            vggt_script,
            stage,
            stage_cache,
            overwrite,
        )

    if tmp_root.exists():
        shutil.rmtree(tmp_root)


def run_vggt_on_images(
    scene_dir: Path,
    conda_exe: str,
    conda_env: str,
    vggt_script: Path,
    stage: str,
    stage_cache: str | None,
    overwrite: bool,
) -> None:
    images_dir = scene_dir / "images"
    if not images_dir.exists() or not any(images_dir.glob("*.png")):
        raise RuntimeError(f"LFR images not found at {images_dir}")

    sparse_dir = scene_dir / "sparse"
    if sparse_dir.exists():
        if overwrite:
            shutil.rmtree(sparse_dir)
        else:
            print(f"Skipping VGGT for {scene_dir}: sparse outputs already exist.")
            return

    vggt_args: List[str] = []
    if stage:
        vggt_args.extend(["--stage", stage])
    if stage_cache:
        vggt_args.extend(["--stage_cache", stage_cache])

    print(f"Running VGGT (BA) on light field images at {scene_dir}")
    run_vggt(scene_dir, conda_exe, conda_env, vggt_script, vggt_args)

def process_dynamic_directory(
    scene_dir: Path,
    output_root: Path,
    base_root: Path,
    scene_name: str,
    conda_exe: str,
    conda_env: str,
    video_to_4dgs: Path,
    vggt_script: Path,
    overwrite: bool,
) -> None:
    if not video_to_4dgs.exists():
        raise FileNotFoundError(f"video_to_4dgs.py not found at {video_to_4dgs}")

    videos = sorted(
        p for p in scene_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )
    if not videos:
        raise RuntimeError(f"No video files found in {scene_dir}")

    try:
        rel = scene_dir.relative_to(base_root)
        label = slugify(rel.as_posix())
    except ValueError:
        label = slugify(scene_dir.name)

    dataset_root = output_root / scene_name if scene_name else output_root
    out_dir = dataset_root / label

    if out_dir.exists():
        if not overwrite:
            print(f"Skipping {scene_dir}: outputs already exist at {out_dir}")
            return
        shutil.rmtree(out_dir)

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Preparing dynamic capture {scene_dir} -> {out_dir}")

    cmd: List[str] = [
        conda_exe,
        "run",
        "-n",
        conda_env,
        "python",
        str(video_to_4dgs),
        "--videos",
        str(scene_dir),
        "--output",
        str(out_dir),
        "--dataset_name",
        label,
        "--vggt_script",
        str(vggt_script),
        "--vggt_conda_env",
        "",
    ]
    run_command(cmd, cwd=video_to_4dgs.parent)


def parse_args() -> argparse.Namespace:
    repo_root = default_repo_root()
    parser = argparse.ArgumentParser(description="Preprocess dog dataset for gsplat.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the capture folder or single file to preprocess.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional root for processed scenes. Defaults to the input directory.",
    )
    parser.add_argument(
        "--scene-name",
        type=str,
        default=None,
        help="Optional scene folder name under the output root. Defaults to the parent folder name.",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Use the 4DGaussians dynamic pipeline instead of static processing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recreate outputs even if they already exist.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Frame sampling rate for video extraction. Omit to keep every native frame.",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        type=str,
        default="ffmpeg",
        help="ffmpeg executable used for frame extraction.",
    )
    parser.add_argument(
        "--conda-exe",
        type=str,
        default=os.environ.get("CONDA_EXE", "conda"),
        help="Path to the conda executable for launching VGGT.",
    )
    parser.add_argument(
        "--conda-env",
        type=str,
        default="transformers",
        help="Conda environment used for VGGT when processing static captures.",
    )
    parser.add_argument(
        "--vggt-script",
        type=Path,
        default=default_vggt_script(),
        help="Path to VGGT demo_colmap.py script.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="both",
        choices=["both", "vggt", "ba"],
        help="VGGT pipeline stage passed to demo_colmap.py.",
    )
    parser.add_argument(
        "--stage-cache",
        type=str,
        default=None,
        help="Optional cache path used when splitting VGGT stages.",
    )
    parser.add_argument(
        "--fourdgs-root",
        type=Path,
        default=default_fourdgs_root(repo_root),
        help="Path to the 4DGaussians repository (used when --include-dynamic is set).",
    )
    parser.add_argument(
        "--fourdgs-conda-env",
        type=str,
        default="4dgaussians",
        help="Conda environment to use when preparing dynamic captures for 4DGaussians.",
    )
    parser.add_argument(
        "--video-to-4dgs",
        type=Path,
        default=default_video_to_4dgs_script(repo_root),
        help="Path to the video_to_4dgs.py helper inside the 4DGaussians repository.",
    )
    parser.add_argument(
        "--lf-to-colmap",
        type=Path,
        default=default_lf_script(repo_root),
        help="Path to lf_to_colmap.py script.",
    )
    parser.add_argument(
        "--lf-calib",
        type=Path,
        default=default_lf_calib_path(),
        help="Path to Light Field calibration tarball (caldata-*.tar).",
    )
    parser.add_argument(
        "--lf-inner",
        type=int,
        default=2,
        help="Inner grid crop parameter forwarded to lf_to_colmap.py.",
    )
    parser.add_argument(
        "--lf-downscale",
        type=float,
        default=None,
        help="Optional downscale factor forwarded to lf_to_colmap.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve() if args.output_root else None
    args.vggt_script = args.vggt_script.expanduser().resolve()
    args.lf_to_colmap = args.lf_to_colmap.expanduser().resolve()
    args.fourdgs_root = args.fourdgs_root.expanduser().resolve()
    args.video_to_4dgs = args.video_to_4dgs.expanduser().resolve()
    if args.lf_calib is not None:
        args.lf_calib = args.lf_calib.expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if not args.vggt_script.exists():
        raise FileNotFoundError(f"VGGT script not found: {args.vggt_script}")
    if not args.lf_to_colmap.exists():
        raise FileNotFoundError(f"lf_to_colmap.py not found: {args.lf_to_colmap}")
    if args.dynamic and not args.fourdgs_root.exists():
        raise FileNotFoundError(f"4DGaussians repository not found: {args.fourdgs_root}")

    if shutil.which(args.ffmpeg_bin) is None:
        raise RuntimeError(f"ffmpeg binary '{args.ffmpeg_bin}' not found in PATH.")
    if args.fps is not None and args.fps <= 0:
        raise ValueError("FPS must be positive.")

    input_spec = determine_input_spec(input_path)
    if output_root is None:
        base_output_dir = input_spec.path if input_spec.path.is_dir() else input_spec.path.parent
    else:
        base_output_dir = output_root
    base_output_dir = base_output_dir.expanduser().resolve()
    scene_name = args.scene_name
    print(f"Selected input: {input_spec.path}")

    if args.dynamic:
        dynamic_spec = input_spec
        if dynamic_spec.kind == "file":
            raise RuntimeError("Dynamic pipeline expects a directory containing video files.")
        if dynamic_spec.kind == "multi_video":
            dynamic_path = dynamic_spec.path
        else:
            dynamic_path = input_path
        dynamic_root = (args.fourdgs_root / "data" / "multipleview").resolve()
        dynamic_root.mkdir(parents=True, exist_ok=True)
        process_dynamic_directory(
            dynamic_path,
            dynamic_root,
            dynamic_spec.base,
            scene_name,
            args.conda_exe,
            args.fourdgs_conda_env,
            args.video_to_4dgs,
            args.vggt_script,
            args.overwrite,
        )
        return

    static_env = args.conda_env or "transformers"

    def resolve_scene_root(default_name: str) -> Path:
        if scene_name:
            return base_output_dir / scene_name
        if base_output_dir == input_spec.path and input_spec.path.is_dir():
            return base_output_dir
        return base_output_dir / default_name

    if input_spec.kind == "multi_video":
        video_files = sorted(
            p for p in input_spec.path.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS
        )
        if not video_files:
            raise RuntimeError(f"No video files found in {input_spec.path}")
        scene_root = resolve_scene_root(input_spec.path.name)
        process_multiple_videos_static(
            video_files,
            scene_root,
            input_spec.base,
            args.conda_exe,
            static_env,
            args.vggt_script,
            args.ffmpeg_bin,
            args.fps,
            args.stage,
            args.stage_cache,
            args.overwrite,
        )
        return

    if input_spec.kind != "file":
        raise RuntimeError(f"Unsupported input type: {input_spec.kind}")

    source_path = input_spec.path
    task = build_task_for_file(input_spec.base, source_path)
    calib_tar = args.lf_calib
    if task.kind == "lfr":
        if calib_tar is None or not calib_tar.exists():
            calib_tar = infer_calibration(default_repo_root())
            if calib_tar:
                print(f"Using inferred calibration tar: {calib_tar}")
        if calib_tar is None:
            raise RuntimeError("Calibration tar is required for light field preprocessing.")
        calib_tar = calib_tar.expanduser().resolve()
        if not calib_tar.exists():
            raise FileNotFoundError(f"Calibration tar not found: {calib_tar}")

    if task.kind == "lfr":
        scene_root = resolve_scene_root(task.label)
        out_dir = scene_root
        inner_value = args.lf_inner if args.lf_inner is not None else 2
        inner_tag = f"inner_{inner_value:02d}"
        inner_scene = out_dir / inner_tag
        images_dir = inner_scene / "images"

        print(f"Processing light field capture: {task.source} -> {out_dir}")
        ensure_ready_dir(out_dir, overwrite=args.overwrite)
        existing_images = images_dir.exists() and any(images_dir.glob("*.png"))

        if existing_images and not args.overwrite:
            print(f"Reusing existing decoded images at {images_dir}")
        else:
            process_lfr(
                task,
                out_dir,
                args.lf_to_colmap,
                calib_tar,
                inner_value,
                args.lf_downscale,
            )
            existing_images = images_dir.exists() and any(images_dir.glob("*.png"))

        if not existing_images:
            raise RuntimeError(f"Failed to prepare light field images at {images_dir}")

        normalize_frame_counts([images_dir])
        run_vggt_on_images(
            inner_scene,
            args.conda_exe,
            static_env,
            args.vggt_script,
            args.stage,
            args.stage_cache,
            args.overwrite,
        )
        return

    out_dir_root = resolve_scene_root(task.label)
    process_video_static(
        task,
        out_dir_root,
        args.conda_exe,
        static_env,
        args.vggt_script,
        args.ffmpeg_bin,
        args.fps,
        args.stage,
        args.stage_cache,
        args.overwrite,
    )


if __name__ == "__main__":
    main()
