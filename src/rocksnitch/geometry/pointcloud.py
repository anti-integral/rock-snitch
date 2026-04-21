"""Disparity -> 3D point cloud helpers."""
from __future__ import annotations

import numpy as np

from rocksnitch.contracts import (
    DisparityMap,
    FloatArray,
    PointCloud,
    RectifiedPair,
)


def disparity_to_pointcloud(
    pair: RectifiedPair,
    disparity: DisparityMap,
    *,
    frame: str = "left_cam",
    max_depth_m: float = 200.0,
) -> PointCloud:
    """Back-project a disparity map to 3D.

    If ``frame == "rover_site"``, transforms points using ``pair.left_to_world``.
    """
    fx = float(pair.K[0, 0])
    fy = float(pair.K[1, 1])
    cx = float(pair.K[0, 2])
    cy = float(pair.K[1, 2])
    H, W = disparity.disparity.shape

    us = np.arange(W, dtype=np.float32)
    vs = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)

    d = disparity.disparity
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = fx * pair.baseline_m / d
    valid = disparity.mask & np.isfinite(Z) & (Z > 0) & (Z < max_depth_m)
    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy
    xyz_cam = np.stack([X, Y, Z], axis=-1).astype(np.float32)

    if frame == "rover_site":
        xyz_hom = np.concatenate(
            [xyz_cam, np.ones((H, W, 1), dtype=np.float32)], axis=-1
        )
        flat = xyz_hom.reshape(-1, 4)
        world = (pair.left_to_world @ flat.T).T
        xyz_cam = world[:, :3].reshape(H, W, 3).astype(np.float32)

    return PointCloud(
        xyz=xyz_cam,
        valid=valid.astype(bool),
        frame="rover_site" if frame == "rover_site" else "left_cam",
    )


def mask_points(pc: PointCloud, mask: np.ndarray) -> FloatArray:
    """Return a (K,3) float32 array of points inside ``mask`` and ``pc.valid``."""
    combined = pc.valid & mask.astype(bool)
    return pc.xyz[combined].astype(np.float32)


__all__ = ["disparity_to_pointcloud", "mask_points"]
