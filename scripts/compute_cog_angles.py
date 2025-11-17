#!/usr/bin/env python3
"""
Compute the average angle (in degrees) between each camera's viewing direction
and the vector from the camera center to the scene center-of-gravity (COG).

This script traverses a dataset prepared by the preprocessing pipeline (with
VGGT/Colmap outputs under <scene>/<mode>/sparse). For each scene and for each
capture mode (iphone or stereo), it:

- Loads the VGGT point cloud (points.ply) to estimate the scene COG.
- Reads COLMAP extrinsics from images.bin.
- Stereo: uses static/shared stereo images (stereo__stereo_left.png and
  stereo__stereo_right.png) and computes the angle at the COG between
  (COG->LEFT) and (COG->RIGHT).
- iPhone: uses static/shared iPhone images (iphone__ultrawide.png and
  iphone__wide.png) and computes the angle at the COG between
  (COG->ULTRAWIDE) and (COG->WIDE).
- Averages per-pair angles per scene, then averages those scene means across
  all scenes to produce a single number per mode.

Usage examples:
  - python scripts/compute_cog_angles.py --dataset-root dataset
  - python scripts/compute_cog_angles.py --dataset-root dataset --mode iphone
  - python scripts/compute_cog_angles.py --dataset-root dataset --mode stereo

Notes:
  - This relies on the VGGT/Colmap reconstruction available under
    <scene>/static/shared/sparse/, specifically images.bin and points.ply.
  - Using points.ply keeps memory use modest compared to parsing points3D.bin.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import struct

# Global configuration (set in main)
FG_PERCENT: float = 40.0
METRIC: str = "angle"  # "angle" or "pixels"


@dataclass
class SceneAngles:
    scene: str
    mode: str  # "iphone" or "stereo"
    per_pair_deg: List[float]

    @property
    def mean_deg(self) -> Optional[float]:
        if not self.per_pair_deg:
            return None
        return sum(self.per_pair_deg) / len(self.per_pair_deg)

    @property
    def median_deg(self) -> Optional[float]:
        if not self.per_pair_deg:
            return None
        vals = sorted(self.per_pair_deg)
        mid = len(vals) // 2
        if len(vals) % 2 == 1:
            return vals[mid]
        return 0.5 * (vals[mid - 1] + vals[mid])


def _qvec2rotmat_py(q: Tuple[float, float, float, float]) -> List[List[float]]:
    q0, q1, q2, q3 = q
    return [
        [1 - 2 * q2 * q2 - 2 * q3 * q3, 2 * q1 * q2 - 2 * q0 * q3, 2 * q1 * q3 + 2 * q0 * q2],
        [2 * q1 * q2 + 2 * q0 * q3, 1 - 2 * q1 * q1 - 2 * q3 * q3, 2 * q2 * q3 - 2 * q0 * q1],
        [2 * q1 * q3 - 2 * q0 * q2, 2 * q2 * q3 + 2 * q0 * q1, 1 - 2 * q1 * q1 - 2 * q2 * q2],
    ]


def _dot(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a: Tuple[float, float, float]) -> float:
    return math.sqrt(_dot(a, a))


def _mul_mat_vec(Mt: List[List[float]], v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (
        Mt[0][0] * v[0] + Mt[0][1] * v[1] + Mt[0][2] * v[2],
        Mt[1][0] * v[0] + Mt[1][1] * v[1] + Mt[1][2] * v[2],
        Mt[2][0] * v[0] + Mt[2][1] * v[1] + Mt[2][2] * v[2],
    )


def _camera_center_and_forward(
    qvec: Tuple[float, float, float, float], tvec: Tuple[float, float, float]
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Return (center_world, forward_dir_world) from a COLMAP image record.

    COLMAP stores world-to-camera as X_cam = R * X_world + t.
    The camera center in world coords is C = -R^T * t.
    The camera-to-world rotation is Rcw = R^T.
    We define the forward (optical) axis in world coords as -Rcw[:, 2].
    """
    R = _qvec2rotmat_py(qvec)  # world -> camera
    Rcw = [
        [R[0][0], R[1][0], R[2][0]],
        [R[0][1], R[1][1], R[2][1]],
        [R[0][2], R[1][2], R[2][2]],
    ]
    t = (tvec[0], tvec[1], tvec[2])
    Rt = _mul_mat_vec(Rcw, t)
    center = (-Rt[0], -Rt[1], -Rt[2])
    forward = (Rcw[0][2], Rcw[1][2], Rcw[2][2])
    fn = _norm(forward)
    if fn > 1e-12:
        forward = (forward[0] / fn, forward[1] / fn, forward[2] / fn)
    return center, forward


def _safe_acosd(x: float) -> float:
    if x < -1.0:
        x = -1.0
    elif x > 1.0:
        x = 1.0
    return math.degrees(math.acos(x))


def _compute_scene_angles(scene_dir: Path, mode: str) -> Optional[SceneAngles]:
    # Always use static/shared reconstructions
    sparse_dir = scene_dir / "static" / "shared" / "sparse"
    points_path = sparse_dir / "points.ply"
    images_bin = sparse_dir / "images.bin"
    if not points_path.exists() or not images_bin.exists():
        return None

    # Load point cloud and compute foreground COG
    try:
        cog = _compute_ply_cog_foreground(points_path, keep_percent=FG_PERCENT)
    except Exception:
        return None

    # Read COLMAP extrinsics with names and camera IDs
    try:
        entries = _read_extrinsics_with_names(images_bin)
    except Exception:
        return None

    name_to_center: dict[str, Tuple[float, float, float]] = {}
    name_to_axes: dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = {}
    name_to_camid: dict[str, int] = {}
    for name, cam_id, qvec, tvec in entries:
        center, fwd = _camera_center_and_forward(qvec, tvec)
        # Derive x_unit from rotation as before
        R = _qvec2rotmat_py(qvec)
        x_unit = (R[0][0], R[1][0], R[2][0])  # Rcw[:,0]
        nx = _norm(x_unit)
        if nx > 1e-12:
            x_unit = (x_unit[0] / nx, x_unit[1] / nx, x_unit[2] / nx)
        name_to_center[name] = center
        name_to_axes[name] = (x_unit, fwd)
        name_to_camid[name] = cam_id

    per_pair_values: List[float] = []
    # Read intrinsics if pixel metric is requested
    cams: Dict[int, Tuple[int, int, int, List[float]]] = {}
    if METRIC == "pixels":
        try:
            cams = _read_cameras_binary_light(sparse_dir / "cameras.bin")
        except Exception:
            cams = {}

    if mode == "stereo":
        # Static explicit filenames
        ln = "stereo__stereo_left.png"
        rn = "stereo__stereo_right.png"
        lc = name_to_center.get(ln)
        rc = name_to_center.get(rn)
        if lc is not None and rc is not None:
            if METRIC == "angle":
                vL = (lc[0] - cog[0], lc[1] - cog[1], lc[2] - cog[2])
                vR = (rc[0] - cog[0], rc[1] - cog[1], rc[2] - cog[2])
                nL = _norm(vL)
                nR = _norm(vR)
                if nL >= 1e-12 and nR >= 1e-12:
                    cos_th = _dot(vL, vR) / (nL * nR)
                    per_pair_values.append(_safe_acosd(cos_th))
            else:
                # pixel distance using left as reference
                ln = "stereo__stereo_left.png"
                x_unit, z_unit = name_to_axes.get(ln, ((1.0,0.0,0.0),(0.0,0.0,1.0)))
                dx = _dot((rc[0]-lc[0], rc[1]-lc[1], rc[2]-lc[2]), x_unit)
                Z = _dot((cog[0]-lc[0], cog[1]-lc[1], cog[2]-lc[2]), z_unit)
                cam = cams.get(name_to_camid.get(ln, -1))
                fx = cam[3][0] if cam and cam[3] else None
                if fx is not None and Z > 1e-9:
                    per_pair_values.append(abs(fx * dx / Z))

    elif mode == "iphone":
        # Static explicit filenames
        un = "iphone__ultrawide.png"
        wn = "iphone__wide.png"
        uc = name_to_center.get(un)
        wc = name_to_center.get(wn)
        if uc is not None and wc is not None:
            if METRIC == "angle":
                vU = (uc[0] - cog[0], uc[1] - cog[1], uc[2] - cog[2])
                vW = (wc[0] - cog[0], wc[1] - cog[1], wc[2] - cog[2])
                nU = _norm(vU)
                nW = _norm(vW)
                if nU >= 1e-12 and nW >= 1e-12:
                    cos_th = _dot(vU, vW) / (nU * nW)
                    per_pair_values.append(_safe_acosd(cos_th))
            else:
                # pixel distance using ultrawide as reference
                x_unit, z_unit = name_to_axes.get(un, ((1.0,0.0,0.0),(0.0,0.0,1.0)))
                dx = _dot((wc[0]-uc[0], wc[1]-uc[1], wc[2]-uc[2]), x_unit)
                Z = _dot((cog[0]-uc[0], cog[1]-uc[1], cog[2]-uc[2]), z_unit)
                cam = cams.get(name_to_camid.get(un, -1))
                fx = cam[3][0] if cam and cam[3] else None
                if fx is not None and Z > 1e-9:
                    per_pair_values.append(abs(fx * dx / Z))

    elif mode == "iphone_static":
        # Expect names: iphone__ultrawide.png and iphone__wide.png
        uw_name = "iphone__ultrawide.png"
        w_name = "iphone__wide.png"
        uc = name_to_center.get(uw_name)
        wc = name_to_center.get(w_name)
        if uc is not None and wc is not None:
            vU = (uc[0] - cog[0], uc[1] - cog[1], uc[2] - cog[2])
            vW = (wc[0] - cog[0], wc[1] - cog[1], wc[2] - cog[2])
            nU = _norm(vU)
            nW = _norm(vW)
            if nU >= 1e-12 and nW >= 1e-12:
                cos_th = _dot(vU, vW) / (nU * nW)
                per_pair_angles.append(_safe_acosd(cos_th))

    elif mode == "lightfield":
        # Pair leftmost and rightmost sub-views in each available LF row
        pat = re.compile(r"^lf__([0-9]{2})_([0-9]{2})\.png$", re.IGNORECASE)
        rows: dict[int, List[Tuple[int, str]]] = {}
        for name in name_to_center.keys():
            m = pat.match(name)
            if not m:
                continue
            v = int(m.group(1))
            u = int(m.group(2))
            rows.setdefault(v, []).append((u, name))
        for v, items in rows.items():
            if not items:
                continue
            items.sort(key=lambda x: x[0])
            left_u, left_name = items[0]
            right_u, right_name = items[-1]
            if left_name == right_name:
                continue
            lc = name_to_center.get(left_name)
            rc = name_to_center.get(right_name)
            if lc is None or rc is None:
                continue
            if METRIC == "angle":
                vL = (lc[0] - cog[0], lc[1] - cog[1], lc[2] - cog[2])
                vR = (rc[0] - cog[0], rc[1] - cog[1], rc[2] - cog[2])
                nL = _norm(vL)
                nR = _norm(vR)
                if nL < 1e-12 or nR < 1e-12:
                    continue
                cos_th = _dot(vL, vR) / (nL * nR)
                per_pair_values.append(_safe_acosd(cos_th))
            else:
                # pixels using leftmost as reference
                x_unit, z_unit = name_to_axes.get(left_name, ((1.0,0.0,0.0),(0.0,0.0,1.0)))
                dx = _dot((rc[0]-lc[0], rc[1]-lc[1], rc[2]-lc[2]), x_unit)
                Z = _dot((cog[0]-lc[0], cog[1]-lc[1], cog[2]-lc[2]), z_unit)
                cam = cams.get(name_to_camid.get(left_name, -1))
                fx = cam[3][0] if cam and cam[3] else None
                if fx is not None and Z > 1e-9:
                    per_pair_values.append(abs(fx * dx / Z))

    if not per_pair_values:
        return None

    return SceneAngles(scene=scene_dir.name, mode=mode, per_pair_deg=per_pair_values)


def _iter_scenes(dataset_root: Path) -> Iterable[Path]:
    for sub in sorted(dataset_root.iterdir()):
        if not sub.is_dir():
            continue
        if sub.name.lower() == "dynamic":
            continue
        yield sub


def compute_averages(dataset_root: Path, modes: List[str]) -> Dict[str, Tuple[float, int]]:
    """Compute summary per mode.

    - If METRIC == 'angle': returns median across scenes (deg).
    - If METRIC == 'pixels': returns mean across scenes (px).
    """
    results: Dict[str, List[float]] = {m: [] for m in modes}

    for scene_dir in _iter_scenes(dataset_root):
        for mode in modes:
            scene_angles = _compute_scene_angles(scene_dir, mode)
            if scene_angles and scene_angles.mean_deg is not None:
                results[mode].append(scene_angles.mean_deg)

    aggregated: Dict[str, Tuple[float, int]] = {}
    for mode in modes:
        vals = results.get(mode, [])
        if not vals:
            aggregated[mode] = (float("nan"), 0)
            continue
        if METRIC == "pixels":
            mean_val = sum(vals) / len(vals)
            aggregated[mode] = (mean_val, len(vals))
        else:
            vals_sorted = sorted(vals)
            mid = len(vals_sorted) // 2
            if len(vals_sorted) % 2 == 1:
                med = vals_sorted[mid]
            else:
                med = 0.5 * (vals_sorted[mid - 1] + vals_sorted[mid])
            aggregated[mode] = (med, len(vals_sorted))
    return aggregated


def _read_extrinsics_binary_light(path: Path) -> List[Tuple[Tuple[float, float, float, float], Tuple[float, float, float]]]:
    """Read only (qvec, tvec) from COLMAP images.bin efficiently.

    Structure per image (see COLMAP read_write_model.py / Inria loader):
      - image_id: int32
      - qvec[4]: 4 x float64
      - tvec[3]: 3 x float64
      - camera_id: int32
      - name: null-terminated string (bytes)
      - num_points2D: uint64
      - then num_points2D records of (x: float64, y: float64, point3D_id: int64)

    We seek past the 2D records without allocating them.
    """
    qts: List[Tuple[Tuple[float, float, float, float], Tuple[float, float, float]]] = []
    with open(path, "rb") as fid:
        # num_reg_images: uint64
        num_reg_images = int.from_bytes(fid.read(8), byteorder="little", signed=False)
        for _ in range(num_reg_images):
            # image header: id (int32), qvec (4xfloat64), tvec (3xfloat64), camera_id (int32)
            # We read via numpy for convenience and correctness.
            _ = int.from_bytes(fid.read(4), byteorder="little", signed=True)  # image_id
            qvec = struct.unpack("<dddd", fid.read(8 * 4))
            tvec = struct.unpack("<ddd", fid.read(8 * 3))
            _ = fid.read(4)  # camera_id (int32)

            # image name (null-terminated)
            name_bytes = bytearray()
            while True:
                ch = fid.read(1)
                if ch == b"\x00" or ch == b"":
                    break
                name_bytes.extend(ch)

            # num_points2D: uint64
            n2d = int.from_bytes(fid.read(8), byteorder="little", signed=False)
            # Seek past 2D points: each record is (float64, float64, int64) = 24 bytes
            fid.seek(24 * n2d, os.SEEK_CUR)

            qts.append((qvec, tvec))

    return qts


def _read_extrinsics_with_names(path: Path) -> List[Tuple[str, int, Tuple[float, float, float, float], Tuple[float, float, float]]]:
    """Return (name, camera_id, qvec, tvec) tuples from COLMAP images.bin."""
    out: List[Tuple[str, int, Tuple[float, float, float, float], Tuple[float, float, float]]] = []
    with open(path, "rb") as fid:
        num_reg_images = int.from_bytes(fid.read(8), byteorder="little", signed=False)
        for _ in range(num_reg_images):
            _ = int.from_bytes(fid.read(4), byteorder="little", signed=True)  # image_id
            qvec = struct.unpack("<dddd", fid.read(8 * 4))
            tvec = struct.unpack("<ddd", fid.read(8 * 3))
            cam_id = int.from_bytes(fid.read(4), byteorder="little", signed=True)  # camera_id
            # read null-terminated name
            name_bytes = bytearray()
            while True:
                ch = fid.read(1)
                if ch == b"\x00" or ch == b"":
                    break
                name_bytes.extend(ch)
            name = name_bytes.decode("utf-8", errors="ignore")
            # skip observations
            n2d = int.from_bytes(fid.read(8), byteorder="little", signed=False)
            fid.seek(24 * n2d, os.SEEK_CUR)
            out.append((name, cam_id, qvec, tvec))
    return out


def _parse_ply_header_for_xyz(f) -> Tuple[int, List[Tuple[str, str]], List[int], int, int, int, int]:
    fmt = None
    vertex_count = None
    vertex_props: List[Tuple[str, str]] = []
    # Read header
    while True:
        line = f.readline()
        if not line:
            raise RuntimeError("Unexpected EOF while reading PLY header")
        line_s = line.decode("ascii", errors="ignore").strip()
        if line_s.startswith("format "):
            fmt = line_s.split()[1]
        elif line_s.startswith("element vertex"):
            vertex_count = int(line_s.split()[2])
        elif line_s.startswith("property ") and vertex_count is not None:
            parts = line_s.split()
            if len(parts) >= 3:
                vertex_props.append((parts[1], parts[2]))
        elif line_s == "end_header":
            break
    if fmt != "binary_little_endian":
        raise RuntimeError(f"Unsupported PLY format: {fmt}")
    if vertex_count is None:
        raise RuntimeError("PLY header missing vertex count")
    # Compute offsets and stride
    type_sizes = {
        "char": 1,
        "uchar": 1,
        "int8": 1,
        "uint8": 1,
        "short": 2,
        "ushort": 2,
        "int": 4,
        "uint": 4,
        "float": 4,
        "float32": 4,
        "double": 8,
        "float64": 8,
    }
    offsets: List[int] = []
    stride = 0
    for t, _ in vertex_props:
        offsets.append(stride)
        if t not in type_sizes:
            raise RuntimeError(f"Unsupported PLY property type: {t}")
        stride += type_sizes[t]
    names = [n for _, n in vertex_props]
    try:
        x_idx = names.index("x")
        y_idx = names.index("y")
        z_idx = names.index("z")
    except ValueError as e:
        raise RuntimeError("PLY is missing x/y/z properties") from e
    return vertex_count, vertex_props, offsets, stride, x_idx, y_idx, z_idx


def _read_float_from_record(rec: bytes, offset: int, t: str) -> float:
    if t in ("float", "float32"):
        return struct.unpack_from("<f", rec, offset)[0]
    if t in ("double", "float64"):
        return struct.unpack_from("<d", rec, offset)[0]
    if t in ("char", "int8"):
        return float(struct.unpack_from("<b", rec, offset)[0])
    if t in ("uchar", "uint8"):
        return float(struct.unpack_from("<B", rec, offset)[0])
    if t in ("short", "ushort"):
        fmt = "<h" if t == "short" else "<H"
        return float(struct.unpack_from(fmt, rec, offset)[0])
    if t in ("int", "uint"):
        fmt = "<i" if t == "int" else "<I"
        return float(struct.unpack_from(fmt, rec, offset)[0])
    raise RuntimeError(f"Unsupported PLY property type in record: {t}")


def _compute_ply_cog_foreground(ply_path: Path, keep_percent: float = 40.0) -> Tuple[float, float, float]:
    """Compute a foreground COG by keeping the nearest K% of points to the initial mean.

    Implementation: three lightweight passes over the binary PLY.
    1) initial mean
    2) sample distances to estimate the Kth percentile threshold
    3) mean of points with distance <= threshold
    """
    keep_percent = max(0.1, min(100.0, keep_percent))

    # Parse header once for metadata
    with open(ply_path, "rb") as f:
        n, props, offsets, stride, xi, yi, zi = _parse_ply_header_for_xyz(f)
        header_end_pos = f.tell()

    # Pass 1: initial mean
    sx = sy = sz = 0.0
    with open(ply_path, "rb") as f:
        f.seek(header_end_pos)
        for _ in range(n):
            rec = f.read(stride)
            if not rec or len(rec) < stride:
                break
            x = _read_float_from_record(rec, offsets[xi], props[xi][0])
            y = _read_float_from_record(rec, offsets[yi], props[yi][0])
            z = _read_float_from_record(rec, offsets[zi], props[zi][0])
            sx += x
            sy += y
            sz += z
    if n == 0:
        raise RuntimeError("PLY contains zero vertices")
    mx, my, mz = sx / n, sy / n, sz / n

    # Pass 2: sample distances for percentile
    sample_max = 50000
    step = max(1, n // sample_max)
    dists_sample: List[float] = []
    with open(ply_path, "rb") as f:
        f.seek(header_end_pos)
        i = 0
        for _ in range(n):
            rec = f.read(stride)
            if not rec or len(rec) < stride:
                break
            if i % step == 0:
                x = _read_float_from_record(rec, offsets[xi], props[xi][0])
                y = _read_float_from_record(rec, offsets[yi], props[yi][0])
                z = _read_float_from_record(rec, offsets[zi], props[zi][0])
                dx = x - mx
                dy = y - my
                dz = z - mz
                d = (dx * dx + dy * dy + dz * dz) ** 0.5
                dists_sample.append(d)
            i += 1
    dists_sample.sort()
    k_idx = max(0, min(len(dists_sample) - 1, int(len(dists_sample) * (keep_percent / 100.0))))
    threshold = dists_sample[k_idx] if dists_sample else float("inf")

    # Pass 3: mean of near points
    sx = sy = sz = 0.0
    m = 0
    with open(ply_path, "rb") as f:
        f.seek(header_end_pos)
        for _ in range(n):
            rec = f.read(stride)
            if not rec or len(rec) < stride:
                break
            x = _read_float_from_record(rec, offsets[xi], props[xi][0])
            y = _read_float_from_record(rec, offsets[yi], props[yi][0])
            z = _read_float_from_record(rec, offsets[zi], props[zi][0])
            dx = x - mx
            dy = y - my
            dz = z - mz
            d = (dx * dx + dy * dy + dz * dz) ** 0.5
            if d <= threshold:
                sx += x
                sy += y
                sz += z
                m += 1
    if m == 0:
        return (mx, my, mz)
    return (sx / m, sy / m, sz / m)


def _read_cameras_binary_light(path: Path) -> Dict[int, Tuple[int, int, int, List[float]]]:
    """Lightweight reader for COLMAP cameras.bin.

    Returns mapping: camera_id -> (model_id, width, height, params_list).
    """
    model_num_params = {
        0: 3,   # SIMPLE_PINHOLE
        1: 4,   # PINHOLE
        2: 4,   # SIMPLE_RADIAL
        3: 5,   # RADIAL
        4: 8,   # OPENCV
        5: 8,   # OPENCV_FISHEYE
        6: 12,  # FULL_OPENCV
        7: 5,   # FOV
        8: 4,   # SIMPLE_RADIAL_FISHEYE
        9: 5,   # RADIAL_FISHEYE
        10: 12, # THIN_PRISM_FISHEYE
    }
    cams: Dict[int, Tuple[int, int, int, List[float]]] = {}
    with open(path, "rb") as fid:
        num_cams = int.from_bytes(fid.read(8), byteorder="little", signed=False)
        for _ in range(num_cams):
            cam_id = int.from_bytes(fid.read(4), byteorder="little", signed=True)
            model_id = int.from_bytes(fid.read(4), byteorder="little", signed=True)
            width = int.from_bytes(fid.read(8), byteorder="little", signed=False)
            height = int.from_bytes(fid.read(8), byteorder="little", signed=False)
            nparams = model_num_params.get(model_id, 4)
            params = list(struct.unpack("<" + "d" * nparams, fid.read(8 * nparams)))
            cams[cam_id] = (model_id, width, height, params)
    return cams


def _compute_ply_cog(ply_path: Path) -> Tuple[float, float, float]:
    """Compute centroid of a binary_little_endian PLY by streaming vertices.

    Only x/y/z are used. Other vertex properties are skipped via seek.
    """
    with open(ply_path, "rb") as f:
        # Parse header
        fmt = None
        vertex_count = None
        vertex_props: List[Tuple[str, str]] = []
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError("Unexpected EOF while reading PLY header")
            line = line.decode("ascii", errors="ignore").strip()
            if line.startswith("format "):
                fmt = line.split()[1]
            elif line.startswith("element vertex"):
                vertex_count = int(line.split()[2])
            elif line.startswith("property ") and vertex_count is not None:
                parts = line.split()
                if len(parts) >= 3:
                    vertex_props.append((parts[1], parts[2]))
            elif line == "end_header":
                break
        if fmt != "binary_little_endian":
            raise RuntimeError(f"Unsupported PLY format: {fmt}")
        if vertex_count is None:
            raise RuntimeError("PLY header missing vertex count")

        type_sizes = {
            "char": 1,
            "uchar": 1,
            "int8": 1,
            "uint8": 1,
            "short": 2,
            "ushort": 2,
            "int": 4,
            "uint": 4,
            "float": 4,
            "float32": 4,
            "double": 8,
            "float64": 8,
        }
        offsets = []
        stride = 0
        for t, _ in vertex_props:
            offsets.append(stride)
            if t not in type_sizes:
                raise RuntimeError(f"Unsupported PLY property type: {t}")
            stride += type_sizes[t]
        # Find indices for x, y, z
        names = [n for _, n in vertex_props]
        try:
            x_idx = names.index("x")
            y_idx = names.index("y")
            z_idx = names.index("z")
        except ValueError as e:
            raise RuntimeError("PLY is missing x/y/z properties") from e

        # Stream vertices
        sx = sy = sz = 0.0
        n = 0
        for _ in range(vertex_count):
            # Move to start of record
            # Read x
            f.seek(offsets[x_idx], os.SEEK_CUR)
            tx = vertex_props[x_idx][0]
            if tx in ("float", "float32"):
                x = struct.unpack("<f", f.read(4))[0]
                pos_after_x = offsets[x_idx] + 4
            elif tx in ("double", "float64"):
                x = struct.unpack("<d", f.read(8))[0]
                pos_after_x = offsets[x_idx] + 8
            else:
                raise RuntimeError("Unsupported x type in PLY")
            # Read y
            rel = offsets[y_idx] - pos_after_x
            if rel:
                f.seek(rel, os.SEEK_CUR)
            ty = vertex_props[y_idx][0]
            y = struct.unpack("<f", f.read(4))[0] if ty in ("float", "float32") else struct.unpack("<d", f.read(8))[0]
            pos_after_y = offsets[y_idx] + (4 if ty in ("float", "float32") else 8)
            # Read z
            rel = offsets[z_idx] - pos_after_y
            if rel:
                f.seek(rel, os.SEEK_CUR)
            tz = vertex_props[z_idx][0]
            z = struct.unpack("<f", f.read(4))[0] if tz in ("float", "float32") else struct.unpack("<d", f.read(8))[0]
            pos_after_z = offsets[z_idx] + (4 if tz in ("float", "float32") else 8)
            # Skip to end of record
            tail = stride - pos_after_z
            if tail:
                f.seek(tail, os.SEEK_CUR)

            sx += x
            sy += y
            sz += z
            n += 1

        if n == 0:
            raise RuntimeError("PLY contains zero vertices")
        return (sx / n, sy / n, sz / n)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute average inter-camera angle at the scene COG using static/shared "
            "VGGT/Colmap outputs under <scene>/static/shared/sparse."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset"),
        help="Root containing scene folders (default: dataset)",
    )
    parser.add_argument(
        "--mode",
        choices=["iphone", "stereo", "lightfield", "both"],
        default="both",
        help=(
            "Which capture mode(s) to evaluate. All modes use static/shared images."
        ),
    )
    parser.add_argument(
        "--foreground-percent",
        type=float,
        default=40.0,
        help=(
            "Percent of nearest points (by distance to initial mean) used for the "
            "foreground COG (default: 40)."
        ),
    )
    parser.add_argument(
        "--metric",
        choices=["angle", "pixels"],
        default="angle",
        help=(
            "Report 'angle' (degrees, median across scenes) or 'pixels' "
            "(mean disparity in px across scenes)."
        ),
    )
    parser.add_argument(
        "--per-scene",
        action="store_true",
        help="Print per-scene median angles and number of contributing pairs.",
    )
    args = parser.parse_args()
    global FG_PERCENT, METRIC
    FG_PERCENT = float(args.foreground_percent)
    METRIC = args.metric

    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.exists():
        raise SystemExit(f"Dataset root not found: {dataset_root}")

    modes = ["iphone", "stereo"] if args.mode == "both" else [args.mode]
    aggregated = compute_averages(dataset_root, modes)

    for mode in modes:
        value, nscenes = aggregated[mode]
        if nscenes == 0 or (value != value):  # NaN check
            print(f"{mode}: N/A (no scenes)")
        else:
            if METRIC == "angle":
                print(f"{mode}: {value:.3f} deg (median across {nscenes} scene(s))")
            else:
                print(f"{mode}: {value:.3f} px (mean across {nscenes} scene(s))")

        if args.per_scene:
            rows: List[Tuple[str, float, int]] = []
            for scene_dir in _iter_scenes(dataset_root):
                res = _compute_scene_angles(scene_dir, mode)
                if res and res.per_pair_deg:
                    stat = (res.mean_deg if METRIC == "pixels" else res.median_deg)
                    rows.append((scene_dir.name, float(stat) if stat is not None else float("nan"), len(res.per_pair_deg)))
            if rows:
                if METRIC == "pixels":
                    print(f"Per-scene means for {mode}:")
                else:
                    print(f"Per-scene medians for {mode}:")
                for name, val, k in sorted(rows, key=lambda x: x[0]):
                    if val == val:
                        unit = "px" if METRIC == "pixels" else "deg"
                        print(f"  {name}: {val:.3f} {unit} ({k} pair{'s' if k!=1 else ''})")
                    else:
                        print(f"  {name}: N/A")


if __name__ == "__main__":
    main()
