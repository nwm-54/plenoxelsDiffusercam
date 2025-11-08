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
from dataclasses import dataclass, field
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

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".ppm", ".pgm"}


@dataclass
class Task:
    kind: str  # "lfr" or "video"
    source: Path
    label: str


# Shared-mode data types
@dataclass
class ImageRecord:
    source_path: Path
    relative_path: Path
    combined_name: str
    subset: str


@dataclass
class SubsetConfig:
    name: str
    source_dir: Path
    prefix: str
    kind: str  # "train" or "eval"
    include_list: Optional[set[str]] = None
    match: Optional[str] = None
    records: list[ImageRecord] = field(default_factory=list)

    def display_name(self) -> str:
        return f"{self.kind}:{self.name}"

    @property
    def match_token(self) -> str:
        return self.prefix


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


def default_gsplat_examples_root() -> Path:
    # <repo>/gs7/scripts -> parents[2] == workspace root containing gsplat
    return Path(__file__).resolve().parents[2] / "gsplat" / "examples"


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


def ensure_dir(path: Path, overwrite: bool = False) -> None:
    if path.exists() and overwrite:
        if path.is_file():
            path.unlink()
        else:
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

    extracted_frames = [p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
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
    frame_count = sum(1 for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)

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
        if overwrite:
            shutil.rmtree(sparse_dir)
        else:
            print(f"Skipping {label}: outputs already exist at {sparse_dir}")
            return

    print(f"Processing video capture: {task.source} -> {out_dir}")
    # If no sparse reconstruction exists yet, force re-extraction of frames
    # to avoid stale images. Otherwise honor the overwrite flag.
    images_dir = out_dir / "images"
    reextract = not sparse_dir.exists()
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_ready_dir(images_dir, overwrite=(overwrite or reextract))

    extract_frames(
        task.source,
        images_dir,
        ffmpeg_bin,
        fps,
        overwrite=(overwrite or reextract),
    )

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
    need_rebuild = not (out_root / "sparse").exists()
    if (overwrite or need_rebuild) and tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    image_dirs: List[Path] = []
    tasks: List[Task] = []
    for video_file in sorted_videos:
        task = build_task_for_file(base_root, video_file)
        images_dir = tmp_root / task.label
        extract_frames(
            video_file,
            images_dir,
            ffmpeg_bin,
            fps,
            overwrite=(overwrite or need_rebuild),
        )
        image_dirs.append(images_dir)
        tasks.append(task)

    if not image_dirs:
        print("No frames extracted; aborting.")
        return

    print("\n=== Phase 2: Normalizing frame counts ===")
    normalize_frame_counts(image_dirs)

    print("\n=== Phase 3: Generating combined dataset ===")
    combined_scene = out_root
    combined_images = combined_scene / "images"
    if not (overwrite or need_rebuild) and (combined_scene / "sparse").exists():
        print("Combined sparse exists and overwrite disabled; skipping recompute.")
    else:
        ensure_ready_dir(combined_images, overwrite=(overwrite or need_rebuild))
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
    if not images_dir.exists() or not any(p.suffix.lower() in IMAGE_EXTS for p in images_dir.iterdir()):
        raise RuntimeError(f"Images not found at {images_dir}")

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


# ----------------------------- Shared-mode helpers -----------------------------

def _default_output_dir(train_dir: Path) -> Path:
    base = train_dir.name.rstrip("/")
    return (train_dir.parent / f"{base}_shared").resolve()


def _has_any_images(images_root: Path) -> bool:
    return any(p.suffix.lower() in IMAGE_EXTS for p in images_root.rglob("*"))


def _symlink_images(src_images: Path, dst_root: Path) -> Path:
    dst_images = dst_root / "images"
    if dst_images.exists() or dst_images.is_symlink():
        if dst_images.is_dir() or dst_images.is_symlink():
            try:
                dst_images.unlink()
            except IsADirectoryError:
                shutil.rmtree(dst_images)
        else:
            dst_images.unlink()
    dst_images.parent.mkdir(parents=True, exist_ok=True)
    rel = os.path.relpath(src_images, dst_images.parent)
    dst_images.symlink_to(rel, target_is_directory=True)
    return dst_images


def materialize_images_if_missing(
    subset: SubsetConfig,
    *,
    ffmpeg_bin: str,
    fps: Optional[float],
    lf_to_colmap: Path,
    lf_calib: Optional[Path],
    lf_inner: Optional[int],
    lf_downscale: Optional[float],
    overwrite: bool,
) -> Path:
    src = subset.source_dir
    # 1) Direct images/
    direct_images = src / "images"
    if direct_images.exists() and _has_any_images(direct_images):
        return direct_images

    # 2) inner_XX/images layout from LFR decode
    inner_candidates = sorted((p for p in src.glob("inner_*/images") if p.is_dir()))
    for cand in inner_candidates:
        if _has_any_images(cand):
            return _symlink_images(cand, src)

    # 3) Infer inputs and materialize
    input_spec = determine_input_spec(src)

    def _extract_video_frames(video_path: Path, label: str) -> None:
        out_dir = src / "images" / label
        extract_frames(video_path, out_dir, ffmpeg_bin, fps, overwrite=overwrite)

    if input_spec.kind == "multi_video":
        if shutil.which(ffmpeg_bin) is None:
            raise RuntimeError(f"ffmpeg binary '{ffmpeg_bin}' not found in PATH.")
        videos = sorted(
            p for p in input_spec.path.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS
        )
        if not videos:
            raise RuntimeError(f"No video files found in {input_spec.path}")
        for vid in videos:
            task = build_task_for_file(input_spec.base, vid)
            _extract_video_frames(vid, task.label)
        images_dir = src / "images"
        if not _has_any_images(images_dir):
            raise RuntimeError(f"Failed to extract frames into {images_dir}")
        return images_dir

    if input_spec.kind == "file":
        task = build_task_for_file(input_spec.base, input_spec.path)
        if task.kind == "video":
            if shutil.which(ffmpeg_bin) is None:
                raise RuntimeError(f"ffmpeg binary '{ffmpeg_bin}' not found in PATH.")
            images_dir = src / "images"
            extract_frames(task.source, images_dir, ffmpeg_bin, fps, overwrite=overwrite)
            if not _has_any_images(images_dir):
                raise RuntimeError(f"Failed to extract frames into {images_dir}")
            return images_dir

        if task.kind == "lfr":
            calib_tar = lf_calib
            if calib_tar is None or not calib_tar.exists():
                calib_tar = infer_calibration(default_repo_root())
                if calib_tar:
                    print(f"Using inferred calibration tar: {calib_tar}")
            if calib_tar is None or not calib_tar.exists():
                raise FileNotFoundError(
                    "Calibration tar is required for light field preprocessing and was not found."
                )
            inner_value = lf_inner if lf_inner is not None else 2
            process_lfr(
                task,
                out_dir=src,
                lf_script=lf_to_colmap,
                calib_tar=calib_tar,
                inner=inner_value,
                downscale=lf_downscale,
            )
            decoded = src / f"inner_{inner_value:02d}" / "images"
            if not decoded.exists() or not _has_any_images(decoded):
                raise RuntimeError(f"Failed to prepare light-field images under {decoded}")
            return _symlink_images(decoded, src)

    raise RuntimeError(
        f"{subset.display_name()} has no usable images, videos, or .lfr files under {src}"
    )


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:  # symlink
        rel = os.path.relpath(src, dst.parent)
        dst.symlink_to(rel)


def gather_images(subset: SubsetConfig, args: argparse.Namespace) -> None:
    images_dir = materialize_images_if_missing(
        subset,
        ffmpeg_bin=args.ffmpeg_bin,
        fps=args.fps,
        lf_to_colmap=args.lf_to_colmap.expanduser().resolve(),
        lf_calib=args.lf_calib.expanduser().resolve() if args.lf_calib else None,
        lf_inner=args.lf_inner,
        lf_downscale=args.lf_downscale,
        overwrite=True,
    )

    all_images = sorted(
        p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS
    )
    if not all_images:
        raise RuntimeError(f"No images found under {images_dir}")

    include = subset.include_list
    if include is not None:
        missing = [name for name in include if not (images_dir / name).exists()]
        if missing:
            missing_preview = ", ".join(missing[:5])
            raise RuntimeError(
                f"{subset.display_name()} include list references missing files: {missing_preview}"
                f"{'...' if len(missing) > 5 else ''}"
            )

    filtered: list[Path] = []
    for img_path in all_images:
        rel = img_path.relative_to(images_dir)
        rel_posix = rel.as_posix()
        if include is not None and rel_posix not in include:
            continue
        if subset.match and subset.match.lower() not in rel_posix.lower():
            continue
        filtered.append(img_path)

    if not filtered:
        raise RuntimeError(f"{subset.display_name()} produced zero images after filtering.")

    subset.records = []
    for img_path in filtered:
        rel = img_path.relative_to(images_dir)
        flattened = rel.as_posix().replace("/", "__")
        combined_name = f"{subset.prefix}__{flattened}"
        subset.records.append(
            ImageRecord(
                source_path=img_path,
                relative_path=rel,
                combined_name=combined_name,
                subset=subset.name,
            )
        )
    print(
        f"  {subset.display_name()} -> {len(subset.records)} image(s) (prefix='{subset.prefix}')"
    )


def prepare_combined_images(
    output_dir: Path,
    subsets: list[SubsetConfig],
    copy_mode: str,
    overwrite: bool,
) -> dict[str, ImageRecord]:
    combined_dir = output_dir / "images"
    ensure_dir(combined_dir, overwrite=overwrite)

    mapping: dict[str, ImageRecord] = {}
    for subset in subsets:
        for rec in subset.records:
            dest = combined_dir / rec.combined_name
            link_or_copy(rec.source_path, dest, mode=copy_mode)
            mapping[rec.combined_name] = rec
    return mapping


def run_vggt_reconstruction(
    dataset_dir: Path,
    *,
    conda_exe: str,
    conda_env: str,
    vggt_script: Path,
    stage: str,
    stage_cache: Optional[str],
    overwrite: bool,
) -> None:
    sparse_dir = dataset_dir / "sparse"
    if sparse_dir.exists() and not overwrite:
        print(f"Found existing sparse at {sparse_dir}; reusing.")
        return
    run_vggt_on_images(
        dataset_dir,
        conda_exe,
        conda_env,
        vggt_script,
        stage,
        stage_cache,
        overwrite,
    )


def _write_lines(path: Path, lines: list[str]) -> None:
    with path.open("w") as f:
        for line in lines:
            f.write(line if line.endswith("\n") else f"{line}\n")


def _update_count_comment(lines: list[str], prefix: str, value: int) -> list[str]:
    updated: list[str] = []
    replaced = False
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith(prefix.lower()):
            updated.append(f"{prefix}: {value}\n")
            replaced = True
        else:
            updated.append(line)
    if not replaced:
        updated.insert(0, f"{prefix}: {value}\n")
    return updated


def _convert_sparse_to_text(sparse_dir: Path, txt_dir: Path) -> None:
    ensure_dir(txt_dir, overwrite=True)
    cmd = [
        "colmap",
        "model_converter",
        f"--input_path={sparse_dir}",
        f"--output_path={txt_dir}",
        "--output_type=TXT",
    ]
    run_command(cmd)


def _parse_image_records(lines: list[str]) -> list[tuple[int, int, str, str, str]]:
    records: list[tuple[int, int, str, str, str]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        data_line = line
        obs_line = lines[i + 1] if i + 1 < len(lines) else ""
        parts = stripped.split()
        if len(parts) < 10:
            raise RuntimeError(f"Unexpected images.txt format: '{line.strip()}'")
        image_id = int(parts[0])
        camera_id = int(parts[8])
        name = " ".join(parts[9:])
        records.append((image_id, camera_id, name, data_line, obs_line))
        i += 2
    return records


def _filter_images_txt(
    lines: list[str],
    keep_names: set[str],
) -> tuple[list[str], set[int], set[int]]:
    headers: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            headers.append(line)
        else:
            break

    records = _parse_image_records(lines)
    kept_records = [r for r in records if r[2] in keep_names]
    if not kept_records:
        raise RuntimeError("Subset filtering removed all images.")

    keep_image_ids = {r[0] for r in kept_records}
    keep_camera_ids = {r[1] for r in kept_records}

    body: list[str] = []
    for rec in kept_records:
        body.append(rec[3])
        obs_line = rec[4]
        if obs_line and not obs_line.strip().startswith("#"):
            body.append(obs_line)
        else:
            body.append("\n")

    headers = _update_count_comment(headers, "# Number of images", len(kept_records))
    return headers + body, keep_image_ids, keep_camera_ids


def _filter_cameras_txt(lines: list[str], keep_camera_ids: set[int]) -> tuple[list[str], int]:
    headers: list[str] = []
    body: list[str] = []
    count = 0
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            headers.append(line)
            continue
        camera_id = int(stripped.split()[0])
        if camera_id in keep_camera_ids:
            body.append(line)
            count += 1
    headers = _update_count_comment(headers, "# Number of cameras", count)
    return headers + body, count


def _filter_points_txt(lines: list[str], keep_image_ids: set[int]) -> list[str]:
    headers: list[str] = []
    filtered: list[str] = []
    count = 0
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            headers.append(line)
            continue
        parts = stripped.split()
        if len(parts) < 8:
            continue
        base = parts[:8]
        track = parts[8:]
        new_track: list[str] = []
        for img_id_str, feat_str in zip(track[0::2], track[1::2]):
            try:
                img_id = int(img_id_str)
            except ValueError:
                continue
            if img_id in keep_image_ids:
                new_track.extend([img_id_str, feat_str])
        if not new_track:
            continue
        filtered.append(" ".join(base + new_track) + "\n")
        count += 1
    headers = _update_count_comment(headers, "# Number of points", count)
    return headers + filtered


def _filter_sparse_subset(
    combined_sparse: Path,
    combined_txt: Path,
    subset: SubsetConfig,
    output_sparse: Path,
) -> None:
    ensure_dir(output_sparse, overwrite=True)
    txt_subset = combined_txt / f"_subset_{subset.name}"
    ensure_dir(txt_subset, overwrite=True)

    with (combined_txt / "images.txt").open("r") as f:
        image_lines = f.readlines()
    with (combined_txt / "cameras.txt").open("r") as f:
        camera_lines = f.readlines()
    with (combined_txt / "points3D.txt").open("r") as f:
        points_lines = f.readlines()

    keep_names = {rec.combined_name for rec in subset.records}
    filtered_images, keep_image_ids, keep_camera_ids = _filter_images_txt(image_lines, keep_names)
    filtered_cameras, _ = _filter_cameras_txt(camera_lines, keep_camera_ids)
    filtered_points = _filter_points_txt(points_lines, keep_image_ids)

    _write_lines(txt_subset / "images.txt", filtered_images)
    _write_lines(txt_subset / "cameras.txt", filtered_cameras)
    _write_lines(txt_subset / "points3D.txt", filtered_points)

    cmd = [
        "colmap",
        "model_converter",
        f"--input_path={txt_subset}",
        f"--output_path={output_sparse}",
        "--output_type=BIN",
    ]
    run_command(cmd)

    # Propagate auxiliary files when present (best-effort)
    for extra_name in ("rigs.bin", "frames.bin", "points.ply"):
        extra_path = combined_sparse / extra_name
        if extra_path.exists():
            shutil.copy2(extra_path, output_sparse / extra_name)


def materialize_subset_dirs(
    output_dir: Path,
    subsets: list[SubsetConfig],
    copy_mode: str,
    combined_sparse: Path,
    combined_txt: Path,
    *,
    top_level_two_subfolders: bool = False,
) -> dict[str, Path]:
    if not top_level_two_subfolders:
        subset_root = output_dir / "subsets"
        ensure_dir(subset_root, overwrite=False)

    subset_dirs: dict[str, Path] = {}
    for subset in subsets:
        if top_level_two_subfolders:
            sub_dir = output_dir / ("train" if subset.kind == "train" else "test")
        else:
            sub_dir = subset_root / subset.name

        ensure_dir(sub_dir / "images", overwrite=True)
        for rec in subset.records:
            src = output_dir / "images" / rec.combined_name
            dst = sub_dir / "images" / rec.combined_name
            # Always materialize images in subset dirs (symlink→symlink, copy otherwise)
            link_or_copy(src, dst, mode="symlink" if copy_mode == "symlink" else "copy")

        _filter_sparse_subset(
            combined_sparse=combined_sparse,
            combined_txt=combined_txt,
            subset=subset,
            output_sparse=sub_dir / "sparse",
        )
        subset_dirs[subset.name] = sub_dir
    return subset_dirs


def write_split_lists(output_dir: Path, subsets: list[SubsetConfig]) -> None:
    splits_dir = output_dir / "splits"
    ensure_dir(splits_dir, overwrite=False)
    for subset in subsets:
        list_path = splits_dir / f"{subset.name}.txt"
        with list_path.open("w") as f:
            for rec in subset.records:
                f.write(f"{rec.combined_name}\n")


def write_metadata(
    output_dir: Path,
    subsets: list[SubsetConfig],
    combined_mapping: dict[str, ImageRecord],
    train_test_every: int,
    eval_test_every: int,
) -> None:
    metadata_dir = output_dir / "metadata"
    ensure_dir(metadata_dir, overwrite=False)
    summary = {
        "combined_images": len(combined_mapping),
        "train_test_every": train_test_every,
        "eval_test_every": eval_test_every,
        "subsets": {
            subset.name: {
                "kind": subset.kind,
                "prefix": subset.prefix,
                "match_token": subset.match_token,
                "source_dir": str(subset.source_dir),
                "image_count": len(subset.records),
                "list_path": str(output_dir / "splits" / f"{subset.name}.txt"),
            }
            for subset in subsets
        },
    }
    with (metadata_dir / "summary.json").open("w") as f:
        import json

        json.dump(summary, f, indent=2)

    mapping_records = [
        {
            "subset": rec.subset,
            "combined_name": name,
            "source_relative": rec.relative_path.as_posix(),
        }
        for name, rec in combined_mapping.items()
    ]
    with (metadata_dir / "mapping.json").open("w") as f:
        import json

        json.dump(mapping_records, f, indent=2)


def compute_covisible_masks(
    *,
    subsets: list[SubsetConfig],
    subset_dirs: dict[str, Path],
    train_subset: SubsetConfig,
    output_base: Path,
    examples_root: Path,
    device: str,
    chunk: int,
    micro_chunk: Optional[int],
    train_test_every: int,
    eval_test_every: int,
    factor: int,
    conda_exe: str,
    covisible_conda_env: str,
) -> None:
    script_path = examples_root / "preprocess_covisible_colmap.py"
    if not script_path.exists():
        raise FileNotFoundError(f"preprocess_covisible_colmap.py not found at {script_path}")

    ensure_dir(output_base, overwrite=False)

    # Train covisible (val vs train split within train subset)
    train_dir = subset_dirs[train_subset.name]
    train_out = output_base / train_subset.name
    ensure_dir(train_out, overwrite=False)
    cmd = [
        conda_exe,
        "run",
        "-n",
        covisible_conda_env,
        "python",
        str(script_path),
        "--base_dir",
        str(train_dir),
        "--support_dir",
        str(train_dir),
        "--factor",
        str(factor),
        "--test_every",
        str(train_test_every),
        "--base_split",
        "val",
        "--support_split",
        "train",
        "--support_test_every",
        str(train_test_every),
        "--chunk",
        str(chunk),
        "--device",
        device,
        "--output_dir",
        str(train_out),
    ]
    if micro_chunk is not None:
        cmd.extend(["--micro_chunk", str(micro_chunk)])
    run_command(cmd)

    # External eval subsets
    for subset in subsets:
        if subset.kind != "eval":
            continue
        base_dir = subset_dirs[subset.name]
        out_dir = output_base / subset.name
        ensure_dir(out_dir, overwrite=False)
        cmd = [
            conda_exe,
            "run",
            "-n",
            covisible_conda_env,
            "python",
            str(script_path),
            "--base_dir",
            str(base_dir),
            "--support_dir",
            str(train_dir),
            "--factor",
            str(factor),
            "--test_every",
            str(eval_test_every),
            "--base_split",
            "val",
            "--support_split",
            "train",
            "--support_test_every",
            str(train_test_every),
            "--chunk",
            str(chunk),
            "--device",
            device,
            "--output_dir",
            str(out_dir),
        ]
        if micro_chunk is not None:
            cmd.extend(["--micro_chunk", str(micro_chunk)])
        run_command(cmd)

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
    # Shared-mode (multi-subset) options — enabled when --eval-dir is provided.
    parser.add_argument(
        "--eval-dir",
        action="append",
        default=[],
        help="Hold-out dataset directory (format: name=PATH). Repeatable.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for combined dataset in shared mode (defaults to <train>_shared).",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("symlink", "copy", "hardlink"),
        default="symlink",
        help="How to materialize images inside the combined dataset.",
    )
    parser.add_argument("--train-prefix", type=str, default="train")
    parser.add_argument("--eval-prefix", type=str, default=None)
    parser.add_argument("--train-match", type=str, default=None)
    parser.add_argument("--train-include-list", type=Path, default=None)
    parser.add_argument("--skip-reconstruction", action="store_true")
    parser.add_argument("--train-test-every", type=int, default=8)
    parser.add_argument("--eval-test-every", type=int, default=1)
    parser.add_argument("--covisible", action="store_true")
    parser.add_argument("--covisible-output", type=Path, default=None)
    parser.add_argument("--covisible-device", type=str, default="cuda")
    parser.add_argument("--covisible-chunk", type=int, default=32)
    parser.add_argument("--covisible-micro-chunk", type=int, default=None)
    parser.add_argument("--covisible-factor", type=int, default=1)
    parser.add_argument(
        "--covisible-conda-env",
        type=str,
        default="gaussian_splatting",
        help="Conda environment used to run dycheck/RAFT covisible step.",
    )
    parser.add_argument(
        "--examples-root",
        type=Path,
        default=default_gsplat_examples_root(),
        help="Path to gsplat/examples directory (for covisible script).",
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

    # Shared mode: trigger when at least one --eval-dir is provided
    if args.eval_dir:
        if args.dynamic:
            raise RuntimeError("--dynamic cannot be used together with --eval-dir shared mode.")
        # Train subset is the --input path
        train_dir = input_path
        if not train_dir.exists():
            raise FileNotFoundError(f"Train directory not found: {train_dir}")

        # Resolve output directory for the combined dataset
        output_dir = (
            args.output_dir.expanduser().resolve() if args.output_dir else _default_output_dir(train_dir)
        )
        # Do NOT clobber existing reconstructions; create if missing.
        ensure_dir(output_dir, overwrite=False)

        # Normalize lf paths if provided
        if args.lf_to_colmap is not None:
            args.lf_to_colmap = args.lf_to_colmap.expanduser().resolve()
        if args.lf_calib is not None:
            args.lf_calib = args.lf_calib.expanduser().resolve()

        # Train subset config
        def _parse_include_list(path: Optional[Path]) -> Optional[set[str]]:
            if path is None:
                return None
            with path.open("r") as f:
                entries = {line.strip() for line in f if line.strip()}
            return entries

        def _parse_eval_dir(raw: str) -> tuple[str, Path]:
            if "=" in raw:
                name, path_str = raw.split("=", 1)
                return name.strip(), Path(path_str).expanduser().resolve()
            p = Path(raw).expanduser().resolve()
            return p.name, p

        train_subset = SubsetConfig(
            name="train",
            source_dir=train_dir,
            prefix=args.train_prefix,
            kind="train",
            include_list=_parse_include_list(args.train_include_list),
            match=args.train_match,
        )
        gather_images(train_subset, args)

        # Eval subsets
        eval_subsets: list[SubsetConfig] = []
        for raw in args.eval_dir:
            _name, path = _parse_eval_dir(raw)
            # Force canonical eval subset name/prefix to 'test'
            subset = SubsetConfig(
                name="test",
                source_dir=path,
                prefix=args.eval_prefix or "test",
                kind="eval",
            )
            gather_images(subset, args)
            eval_subsets.append(subset)

        all_subsets: list[SubsetConfig] = [train_subset] + eval_subsets

        # If there's no shared sparse yet, rebuild combined images.
        shared_sparse = output_dir / "sparse"
        mapping = prepare_combined_images(
            output_dir=output_dir,
            subsets=all_subsets,
            copy_mode=args.copy_mode,
            overwrite=(args.overwrite or not shared_sparse.exists()),
        )

        if not args.skip_reconstruction:
            run_vggt_reconstruction(
                output_dir,
                conda_exe=args.conda_exe,
                conda_env=args.conda_env,
                vggt_script=args.vggt_script,
                stage=args.stage,
                stage_cache=args.stage_cache,
                # If sparse/ exists, reuse it by default (no BA re-run).
                overwrite=False,
            )
        else:
            print("Skipping reconstruction (--skip-reconstruction).")

        sparse_dir = output_dir / "sparse"
        if not sparse_dir.exists():
            raise RuntimeError(f"Shared sparse/ not found at {sparse_dir}")

        import tempfile

        with tempfile.TemporaryDirectory(prefix="colmap_txt_") as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            combined_txt = tmpdir / "combined_txt"
            _convert_sparse_to_text(sparse_dir, combined_txt)

            subset_dirs = materialize_subset_dirs(
                output_dir=output_dir,
                subsets=all_subsets,
                copy_mode=args.copy_mode,
                combined_sparse=sparse_dir,
                combined_txt=combined_txt,
                top_level_two_subfolders=(len(eval_subsets) == 1),
            )

        write_split_lists(output_dir, all_subsets)
        write_metadata(
            output_dir,
            all_subsets,
            mapping,
            train_test_every=args.train_test_every,
            eval_test_every=args.eval_test_every,
        )

        # In shared mode, compute covisible masks by default.
        covi_base = (
            args.covisible_output.expanduser().resolve()
            if args.covisible_output
            else output_dir / "covisible"
        )
        compute_covisible_masks(
            subsets=all_subsets,
            subset_dirs=subset_dirs,
            train_subset=train_subset,
            output_base=covi_base,
            examples_root=args.examples_root.expanduser().resolve(),
            device=args.covisible_device,
            chunk=args.covisible_chunk,
            micro_chunk=args.covisible_micro_chunk,
            train_test_every=args.train_test_every,
            eval_test_every=args.eval_test_every,
            factor=args.covisible_factor,
            conda_exe=args.conda_exe,
            covisible_conda_env=args.covisible_conda_env,
        )
        print(f"Covisible masks written under {covi_base}")
        
        print("\nDone. Combined dataset available at:", output_dir)
        return

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
        lfr_need_rebuild = not (inner_scene / "sparse").exists()
        ensure_ready_dir(out_dir, overwrite=(args.overwrite or lfr_need_rebuild))
        existing_images = images_dir.exists() and any(images_dir.glob("*.png"))

        if existing_images and not (args.overwrite or lfr_need_rebuild):
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
