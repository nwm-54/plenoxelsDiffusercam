from __future__ import annotations

import numpy as np
from plyfile import PlyData

from utils.graphics_utils import BasicPointCloud


def fetchPly(path: str) -> BasicPointCloud:
    """Load a point cloud from a PLY file into a BasicPointCloud."""
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    colors = (
        np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
    )
    if {"nx", "ny", "nz"}.issubset(vertices.dtype.names):
        normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
    else:
        normals = np.zeros_like(positions)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)
