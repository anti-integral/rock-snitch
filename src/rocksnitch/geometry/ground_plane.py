"""Robust ground plane estimation (RANSAC + local-patch variant)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rocksnitch.contracts import BoolArray, FloatArray, GroundPlane


@dataclass
class RansacConfig:
    max_iters: int = 2000
    distance_thresh_m: float = 0.05
    rng_seed: int = 0
    min_inliers: int = 50


def fit_plane_svd(points: FloatArray) -> tuple[FloatArray, float, float]:
    """Least-squares plane fit via SVD. Returns (normal, d, rmse)."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal = normal / np.linalg.norm(normal)
    d = float(-np.dot(normal, centroid))
    residuals = points @ normal + d
    rmse = float(np.sqrt(np.mean(residuals ** 2))) if len(residuals) else 0.0
    return normal.astype(np.float64), d, rmse


def fit_ransac(
    points: FloatArray, config: RansacConfig | None = None
) -> GroundPlane:
    """RANSAC plane fit. Assumes points are in rover-site frame with Z pointing up."""
    cfg = config or RansacConfig()
    rng = np.random.default_rng(cfg.rng_seed)
    N = len(points)
    if N < 3:
        raise ValueError("Need at least 3 points to fit a plane.")
    best_inliers: BoolArray = np.zeros(N, dtype=bool)
    best_n = np.array([0.0, 0.0, 1.0])
    best_d = 0.0
    for _ in range(cfg.max_iters):
        idx = rng.choice(N, size=3, replace=False)
        sample = points[idx]
        v1 = sample[1] - sample[0]
        v2 = sample[2] - sample[0]
        normal = np.cross(v1, v2)
        nn = np.linalg.norm(normal)
        if nn < 1e-6:
            continue
        normal = normal / nn
        if normal[2] < 0:
            normal = -normal
        d = -np.dot(normal, sample[0])
        dist = np.abs(points @ normal + d)
        inliers = dist < cfg.distance_thresh_m
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_n = normal
            best_d = d
    if best_inliers.sum() < cfg.min_inliers:
        # Fall back to SVD over all points
        normal, d, rmse = fit_plane_svd(points)
        if normal[2] < 0:
            normal = -normal
            d = -d
        residuals = points @ normal + d
        inliers = np.abs(residuals) < cfg.distance_thresh_m
        return GroundPlane(normal=normal, d=float(d), inlier_mask=inliers, rmse=rmse)
    # Re-fit on inliers for precision
    normal, d, rmse = fit_plane_svd(points[best_inliers])
    if normal[2] < 0:
        normal = -normal
        d = -d
    return GroundPlane(
        normal=normal.astype(np.float64),
        d=float(d),
        inlier_mask=best_inliers,
        rmse=float(rmse),
    )


__all__ = ["RansacConfig", "fit_ransac", "fit_plane_svd"]
