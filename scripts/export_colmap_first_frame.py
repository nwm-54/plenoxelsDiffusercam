#!/usr/bin/env python3
"""
Utility to export the first frame of every camera in an ActorsHQ sequence to
COLMAP's text format (cameras/images/points3D).

Example usage:
python export_colmap_first_frame.py \
    --sequence_root /share/monakhova/actorshq/Actor01/Sequence1/4x \
    --output_dir /tmp/Actor01_Sequence1_colmap \
    --frame 0
"""

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from actorshq.dataset.camera_data import read_calibration_csv


def find_frame_path(images_root: Path, camera_name: str, frame: int) -> Path:
    """Locate the RGB frame on disk, checking common file extensions."""
    stem = f"{camera_name}_rgb{frame:06d}"
    camera_dir = images_root / camera_name

    for ext in (".png", ".jpg", ".jpeg"):
        candidate = camera_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"RGB frame not found for {camera_name} frame {frame} in {camera_dir}")


def main():
    parser = argparse.ArgumentParser(description="Export first-frame data to COLMAP text format.")
    parser.add_argument(
        "--sequence_root",
        required=True,
        type=Path,
        help="Path to the scale folder inside a sequence (e.g. .../Sequence1/4x).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Destination directory for COLMAP files.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index to export (default: 0, i.e. the first frame).",
    )
    parser.add_argument(
        "--copy_images",
        action="store_true",
        help="Copy image files into the COLMAP images/ folder (default: create relative symlinks).",
    )
    args = parser.parse_args()

    sequence_root = args.sequence_root.resolve()
    calibration_csv = sequence_root / "calibration.csv"
    images_root = sequence_root / "rgbs"

    if not calibration_csv.exists():
        raise FileNotFoundError(f"Calibration file not found: {calibration_csv}")
    if not images_root.is_dir():
        raise FileNotFoundError(f"RGB directory not found: {images_root}")

    output_dir = args.output_dir.resolve()
    images_output_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_output_dir.mkdir(exist_ok=True)

    cameras = read_calibration_csv(calibration_csv)

    # Prepare COLMAP cameras.txt
    cameras_txt_lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        "# Number of cameras = {}".format(len(cameras)),
        "",
    ]

    # Prepare COLMAP images.txt
    images_txt_lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
        "# Number of images = {}".format(len(cameras)),
        "",
    ]

    # CSV export for reference (optional)
    poses_csv_path = output_dir / "poses.csv"
    poses_csv_file = poses_csv_path.open("w", newline="")
    poses_writer = csv.writer(poses_csv_file)
    poses_writer.writerow(
        [
            "camera_id",
            "camera_name",
            "qw",
            "qx",
            "qy",
            "qz",
            "tx",
            "ty",
            "tz",
            "fx",
            "fy",
            "cx",
            "cy",
            "image_name",
        ]
    )

    for idx, camera in enumerate(cameras, start=1):
        fx_pixel = camera.width * camera.focal_length[0]
        fy_pixel = camera.height * camera.focal_length[1]
        cx_pixel = camera.width * camera.principal_point[0]
        cy_pixel = camera.height * camera.principal_point[1]

        # COLMAP PINHOLE camera entry
        cameras_txt_lines.append(
            f"{idx} PINHOLE {camera.width} {camera.height} "
            f"{fx_pixel:.8f} {fy_pixel:.8f} {cx_pixel:.8f} {cy_pixel:.8f}"
        )

        # Compute world-to-camera rotation/translation
        rotation_c2w = Rotation.from_rotvec(camera.rotation_axisangle).as_matrix()
        rotation_w2c = rotation_c2w.T
        translation_c2w = camera.translation
        translation_w2c = -rotation_w2c @ translation_c2w

        quaternion = Rotation.from_matrix(rotation_w2c).as_quat()  # [x, y, z, w]
        qw = quaternion[3]
        qx, qy, qz = quaternion[0:3]

        # Locate RGB frame
        source_path = find_frame_path(images_root, camera.name, args.frame)
        image_name = source_path.name
        destination_path = images_output_dir / image_name

        if not destination_path.exists():
            if args.copy_images:
                shutil.copy2(source_path, destination_path)
            else:
                # Use relative symlink for storage efficiency
                try:
                    destination_path.symlink_to(source_path)
                except OSError:
                    # Fall back to copying if symlinks are not permitted
                    shutil.copy2(source_path, destination_path)

        images_txt_lines.append(
            f"{idx} {qw:.12f} {qx:.12f} {qy:.12f} {qz:.12f} "
            f"{translation_w2c[0]:.6f} {translation_w2c[1]:.6f} {translation_w2c[2]:.6f} "
            f"{idx} images/{image_name}"
        )
        images_txt_lines.append("")  # No 2D-3D matches available

        poses_writer.writerow(
            [
                idx,
                camera.name,
                f"{qw:.12f}",
                f"{qx:.12f}",
                f"{qy:.12f}",
                f"{qz:.12f}",
                f"{translation_w2c[0]:.6f}",
                f"{translation_w2c[1]:.6f}",
                f"{translation_w2c[2]:.6f}",
                f"{fx_pixel:.8f}",
                f"{fy_pixel:.8f}",
                f"{cx_pixel:.8f}",
                f"{cy_pixel:.8f}",
                image_name,
            ]
        )

    poses_csv_file.close()

    (output_dir / "cameras.txt").write_text("\n".join(cameras_txt_lines) + "\n", encoding="utf-8")
    (output_dir / "images.txt").write_text("\n".join(images_txt_lines) + "\n", encoding="utf-8")
    (output_dir / "points3D.txt").write_text(
        "# 3D point list is empty for this export.\n", encoding="utf-8"
    )

    print(f"Exported {len(cameras)} cameras to {output_dir}")


if __name__ == "__main__":
    main()
