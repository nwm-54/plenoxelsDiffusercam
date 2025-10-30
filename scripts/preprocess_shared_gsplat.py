#!/usr/bin/env python3
"""Build a shared COLMAP/VGGT reconstruction from training and eval subsets.

This script creates a combined dataset that re-runs VGGT (or COLMAP) on the
union of a training capture and one or more evaluation captures so that every
camera shares the same reference frame. After reconstruction it splits the
results back into per-subset dataset folders, generates metadata describing the
split, and optionally precomputes covisible masks aligned to the shared frame.

Typical usage:

    python scripts/preprocess_shared_gsplat.py \
        --train-dir ../gs7/input_data/dog/iphone4 \
        --eval-dir eval=../gs7/input_data/dog/eval/test \
        --output-dir ../gs7/input_data/dog/iphone4_shared

The resulting layout is:
    <output-dir>/
        images/                      # Combined images (symlinks by default)
        sparse/                      # Shared reconstruction from VGGT
        subsets/train/{images,sparse}
        subsets/<eval_name>/{images,sparse}
        splits/<subset>.txt          # Combined image names per subset
        metadata/*.json              # Summary + mapping back to sources
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    if __package__:
        from . import preprocess_for_gsplat as base_prep
    else:  # Script execution via python path/to/script.py
        sys.path.append(str(Path(__file__).resolve().parent))
        import preprocess_for_gsplat as base_prep  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Failed to import preprocess_for_gsplat helpers. "
        "Ensure this script lives alongside preprocess_for_gsplat.py."
    ) from exc


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".ppm", ".pgm"}


@dataclass
class SubsetConfig:
    name: str
    source_dir: Path
    prefix: str
    kind: str  # "train" or "eval"
    include_list: Optional[Set[str]] = None
    match: Optional[str] = None
    records: List["ImageRecord"] = field(default_factory=list)

    def display_name(self) -> str:
        return f"{self.kind}:{self.name}"

    @property
    def match_token(self) -> str:
        """Token suitable for match_string filtering."""
        return self.prefix


@dataclass
class ImageRecord:
    source_path: Path
    relative_path: Path
    combined_name: str
    subset: str


def _default_output_dir(train_dir: Path) -> Path:
    base = train_dir.name.rstrip("/")
    return (train_dir.parent / f"{base}_shared").resolve()


def _default_gsplat_examples_root() -> Path:
    # <repo>/gs7/scripts -> parents[2] == workspace root containing gsplat
    return Path(__file__).resolve().parents[2] / "gsplat" / "examples"


def run(cmd: Sequence[str], cwd: Optional[Path] = None) -> None:
    printable = " ".join(map(str, cmd))
    print(f"$ {printable}")
    subprocess.run(cmd, check=True, cwd=cwd)


def ensure_dir(path: Path, overwrite: bool = False) -> None:
    if path.exists() and overwrite:
        if path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def gather_images(
    subset: SubsetConfig,
    copy_mode: str,
) -> None:
    images_dir = subset.source_dir / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images/ in {subset.source_dir}")

    all_images = sorted(
        p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS
    )
    if not all_images:
        raise RuntimeError(f"No images found under {images_dir}")

    include: Optional[Set[str]] = subset.include_list
    if include is not None:
        missing = [name for name in include if not (images_dir / name).exists()]
        if missing:
            missing_preview = ", ".join(missing[:5])
            raise RuntimeError(
                f"{subset.display_name()} include list references missing files: "
                f"{missing_preview}{'...' if len(missing) > 5 else ''}"
            )

    filtered: List[Path] = []
    for img_path in all_images:
        rel = img_path.relative_to(images_dir)
        rel_posix = rel.as_posix()
        if include is not None and rel_posix not in include:
            continue
        if subset.match and subset.match.lower() not in rel_posix.lower():
            continue
        filtered.append(img_path)

    if not filtered:
        raise RuntimeError(
            f"{subset.display_name()} produced zero images after filtering."
        )

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
        f"  {subset.display_name()} -> {len(subset.records)} image(s) "
        f"({copy_mode}, prefix='{subset.prefix}')"
    )


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:  # default to symlink
        rel = os.path.relpath(src, dst.parent)
        dst.symlink_to(rel)


def prepare_combined_images(
    output_dir: Path,
    subsets: Sequence[SubsetConfig],
    copy_mode: str,
    overwrite: bool,
) -> Dict[str, ImageRecord]:
    combined_dir = output_dir / "images"
    ensure_dir(combined_dir, overwrite=overwrite)

    mapping: Dict[str, ImageRecord] = {}
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
    base_prep.run_vggt_on_images(
        dataset_dir,
        conda_exe=conda_exe,
        conda_env=conda_env,
        vggt_script=vggt_script,
        stage=stage,
        stage_cache=stage_cache,
        overwrite=overwrite,
    )


def _convert_sparse_to_text(sparse_dir: Path, txt_dir: Path) -> None:
    ensure_dir(txt_dir, overwrite=True)
    run(
        [
            "colmap",
            "model_converter",
            f"--input_path={sparse_dir}",
            f"--output_path={txt_dir}",
            "--output_type=TXT",
        ]
    )


def _write_lines(path: Path, lines: Iterable[str]) -> None:
    with path.open("w") as f:
        for line in lines:
            f.write(line if line.endswith("\n") else f"{line}\n")


def _update_count_comment(lines: List[str], prefix: str, value: int) -> List[str]:
    updated: List[str] = []
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


def _filter_cameras_txt(
    lines: List[str],
    keep_camera_ids: Set[int],
) -> Tuple[List[str], int]:
    headers: List[str] = []
    body: List[str] = []
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


def _parse_image_records(lines: List[str]) -> List[Tuple[int, int, str, str, str]]:
    """Return tuples of (image_id, camera_id, name, data_line, obs_line)."""
    records: List[Tuple[int, int, str, str, str]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        data_line = line
        obs_line = ""
        if i + 1 < len(lines):
            obs_line = lines[i + 1]
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
    lines: List[str],
    keep_names: Set[str],
) -> Tuple[List[str], Set[int], Set[int]]:
    headers: List[str] = []
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

    body: List[str] = []
    for rec in kept_records:
        body.append(rec[3])
        obs_line = rec[4]
        if obs_line and not obs_line.strip().startswith("#"):
            body.append(obs_line)
        else:
            body.append("\n")

    headers = _update_count_comment(headers, "# Number of images", len(kept_records))
    return headers + body, keep_image_ids, keep_camera_ids


def _filter_points_txt(
    lines: List[str],
    keep_image_ids: Set[int],
) -> List[str]:
    headers: List[str] = []
    filtered: List[str] = []
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
        new_track: List[str] = []
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
    filtered_images, keep_image_ids, keep_camera_ids = _filter_images_txt(
        image_lines, keep_names
    )
    filtered_cameras, _ = _filter_cameras_txt(camera_lines, keep_camera_ids)
    filtered_points = _filter_points_txt(points_lines, keep_image_ids)

    _write_lines(txt_subset / "images.txt", filtered_images)
    _write_lines(txt_subset / "cameras.txt", filtered_cameras)
    _write_lines(txt_subset / "points3D.txt", filtered_points)

    run(
        [
            "colmap",
            "model_converter",
            f"--input_path={txt_subset}",
            f"--output_path={output_sparse}",
            "--output_type=BIN",
        ]
    )

    # Propagate auxiliary files when present (best-effort).
    for extra_name in ("rigs.bin", "frames.bin", "points.ply"):
        extra_path = combined_sparse / extra_name
        if extra_path.exists():
            shutil.copy2(extra_path, output_sparse / extra_name)


def write_metadata(
    output_dir: Path,
    subsets: Sequence[SubsetConfig],
    combined_mapping: Dict[str, ImageRecord],
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
        json.dump(mapping_records, f, indent=2)


def write_split_lists(output_dir: Path, subsets: Sequence[SubsetConfig]) -> None:
    splits_dir = output_dir / "splits"
    ensure_dir(splits_dir, overwrite=False)
    for subset in subsets:
        list_path = splits_dir / f"{subset.name}.txt"
        with list_path.open("w") as f:
            for rec in subset.records:
                f.write(f"{rec.combined_name}\n")


def materialize_subset_dirs(
    output_dir: Path,
    subsets: Sequence[SubsetConfig],
    copy_mode: str,
    combined_sparse: Path,
    combined_txt: Path,
) -> Dict[str, Path]:
    subset_root = output_dir / "subsets"
    ensure_dir(subset_root, overwrite=False)

    subset_dirs: Dict[str, Path] = {}
    for subset in subsets:
        sub_dir = subset_root / subset.name
        ensure_dir(sub_dir / "images", overwrite=True)
        for rec in subset.records:
            src = output_dir / "images" / rec.combined_name
            dst = sub_dir / "images" / rec.combined_name
            link_or_copy(src, dst, mode="symlink" if copy_mode == "symlink" else "copy")

        _filter_sparse_subset(
            combined_sparse=combined_sparse,
            combined_txt=combined_txt,
            subset=subset,
            output_sparse=sub_dir / "sparse",
        )
        subset_dirs[subset.name] = sub_dir
    return subset_dirs


def compute_covisible_masks(
    *,
    subsets: Sequence[SubsetConfig],
    subset_dirs: Dict[str, Path],
    train_subset: SubsetConfig,
    output_base: Path,
    examples_root: Path,
    device: str,
    chunk: int,
    micro_chunk: Optional[int],
    train_test_every: int,
    eval_test_every: int,
    factor: int,
) -> None:
    script_path = examples_root / "preprocess_covisible_colmap.py"
    if not script_path.exists():
        raise FileNotFoundError(
            f"preprocess_covisible_colmap.py not found at {script_path}"
        )

    ensure_dir(output_base, overwrite=False)

    # Train covisible (val vs train split within train subset)
    train_dir = subset_dirs[train_subset.name]
    train_out = output_base / train_subset.name
    ensure_dir(train_out, overwrite=False)
    cmd = [
        sys.executable,
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
    run(cmd)

    # External eval subsets
    for subset in subsets:
        if subset is train_subset:
            continue
        eval_dir = subset_dirs[subset.name]
        eval_out = output_base / subset.name
        ensure_dir(eval_out, overwrite=False)
        cmd = [
            sys.executable,
            str(script_path),
            "--base_dir",
            str(eval_dir),
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
            str(eval_out),
        ]
        if micro_chunk is not None:
            cmd.extend(["--micro_chunk", str(micro_chunk)])
        run(cmd)


def _parse_include_list(path: Optional[Path]) -> Optional[Set[str]]:
    if path is None:
        return None
    with path.open("r") as f:
        entries = {line.strip() for line in f if line.strip()}
    return entries


def _parse_eval_dir(raw: str) -> Tuple[str, Path]:
    if "=" in raw:
        name, path_str = raw.split("=", 1)
        return name.strip(), Path(path_str).expanduser().resolve()
    path = Path(raw).expanduser().resolve()
    return path.name, path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build combined VGGT/COLMAP datasets with shared poses."
    )
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument(
        "--eval-dir",
        action="append",
        default=[],
        help="Hold-out dataset directory (format: name=path). Repeatable.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for combined dataset (defaults to <train-dir>_shared).",
    )
    parser.add_argument(
        "--train-prefix",
        type=str,
        default="train",
        help="Prefix applied to training images in the combined set.",
    )
    parser.add_argument(
        "--eval-prefix",
        type=str,
        default=None,
        help="Optional prefix for eval images (defaults to subset name).",
    )
    parser.add_argument(
        "--train-match",
        type=str,
        default=None,
        help="Optional case-insensitive substring to filter training images.",
    )
    parser.add_argument(
        "--train-include-list",
        type=Path,
        default=None,
        help="Path to newline-delimited list of training images to keep "
        "(relative to train images/).",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("symlink", "copy", "hardlink"),
        default="symlink",
        help="How to materialize images inside the combined dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recreate output directory even if it already exists.",
    )
    parser.add_argument(
        "--skip-reconstruction",
        action="store_true",
        help="Assume sparse/ already exists and skip VGGT.",
    )
    parser.add_argument(
        "--conda-exe",
        type=str,
        default=os.environ.get("CONDA_EXE", "conda"),
    )
    parser.add_argument(
        "--conda-env",
        type=str,
        default="transformers",
        help="Environment to run VGGT demo_colmap.py.",
    )
    parser.add_argument(
        "--vggt-script",
        type=Path,
        default=base_prep.default_vggt_script(),
    )
    parser.add_argument("--stage", type=str, default="")
    parser.add_argument("--stage-cache", type=str, default=None)

    parser.add_argument(
        "--train-test-every",
        type=int,
        default=8,
        help="Cadence used by training split (matches --test_every).",
    )
    parser.add_argument(
        "--eval-test-every",
        type=int,
        default=1,
        help="Cadence used for eval subsets when computing covisible masks.",
    )
    parser.add_argument(
        "--covisible-output",
        type=Path,
        default=None,
        help="Optional directory to store covisible masks "
        "(defaults to <output-dir>/covisible).",
    )
    parser.add_argument(
        "--covisible",
        action="store_true",
        help="Enable covisible mask precomputation after reconstruction.",
    )
    parser.add_argument("--covisible-device", type=str, default="cuda")
    parser.add_argument("--covisible-chunk", type=int, default=32)
    parser.add_argument("--covisible-micro-chunk", type=int, default=None)
    parser.add_argument("--covisible-factor", type=int, default=1)
    parser.add_argument(
        "--examples-root",
        type=Path,
        default=_default_gsplat_examples_root(),
        help="Path to gsplat/examples directory (for covisible script).",
    )

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    train_dir = args.train_dir.expanduser().resolve()
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    eval_specs = args.eval_dir or []
    if not eval_specs:
        raise RuntimeError("At least one --eval-dir is required.")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else _default_output_dir(train_dir)
    )
    if output_dir.exists() and not args.overwrite:
        raise RuntimeError(
            f"Output directory {output_dir} already exists. Use --overwrite to rebuild."
        )
    ensure_dir(output_dir, overwrite=args.overwrite)

    train_subset = SubsetConfig(
        name="train",
        source_dir=train_dir,
        prefix=args.train_prefix,
        kind="train",
        include_list=_parse_include_list(args.train_include_list),
        match=args.train_match,
    )
    gather_images(train_subset, args.copy_mode)

    eval_subsets: List[SubsetConfig] = []
    for raw in eval_specs:
        name, path = _parse_eval_dir(raw)
        subset = SubsetConfig(
            name=name,
            source_dir=path,
            prefix=args.eval_prefix or name,
            kind="eval",
        )
        gather_images(subset, args.copy_mode)
        eval_subsets.append(subset)

    all_subsets: List[SubsetConfig] = [train_subset] + eval_subsets

    mapping = prepare_combined_images(
        output_dir=output_dir,
        subsets=all_subsets,
        copy_mode=args.copy_mode,
        overwrite=True,
    )

    if not args.skip_reconstruction:
        run_vggt_reconstruction(
            output_dir,
            conda_exe=args.conda_exe,
            conda_env=args.conda_env,
            vggt_script=args.vggt_script.expanduser().resolve(),
            stage=args.stage,
            stage_cache=args.stage_cache,
            overwrite=True,
        )
    else:
        print("Skipping reconstruction (--skip-reconstruction).")

    sparse_dir = output_dir / "sparse"
    if not sparse_dir.exists():
        raise RuntimeError(f"Shared sparse/ not found at {sparse_dir}")

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
        )

    write_split_lists(output_dir, all_subsets)
    write_metadata(
        output_dir,
        all_subsets,
        mapping,
        train_test_every=args.train_test_every,
        eval_test_every=args.eval_test_every,
    )

    if args.covisible:
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
        )
        print(f"Covisible masks written under {covi_base}")

    print("\nDone. Combined dataset available at:", output_dir)


if __name__ == "__main__":  # pragma: no cover
    main()
