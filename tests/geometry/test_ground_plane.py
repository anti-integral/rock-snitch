from __future__ import annotations

import numpy as np

from rocksnitch.geometry.ground_plane import RansacConfig, fit_plane_svd, fit_ransac


def test_fit_plane_svd_xy_plane() -> None:
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((500, 3))
    pts[:, 2] = 0.0
    n, d, rmse = fit_plane_svd(pts)
    assert abs(abs(n[2]) - 1.0) < 1e-6
    assert abs(d) < 1e-6
    assert rmse < 1e-6


def test_fit_ransac_rejects_outliers() -> None:
    rng = np.random.default_rng(42)
    inliers = rng.standard_normal((400, 3))
    inliers[:, 2] = 0.0 + 0.01 * rng.standard_normal(400)
    outliers = rng.standard_normal((50, 3))
    outliers[:, 2] += 5.0
    pts = np.vstack([inliers, outliers]).astype(np.float32)
    plane = fit_ransac(pts, RansacConfig(max_iters=500, distance_thresh_m=0.05))
    assert plane.inlier_mask.sum() >= 350
    assert plane.rmse < 0.05
