from __future__ import annotations

import numpy as np

from rocksnitch.contracts import GroundPlane
from rocksnitch.geometry.height import mask_height_stats, mask_max_height, signed_height_above_plane


def test_signed_height_zero_on_plane() -> None:
    plane = GroundPlane(normal=np.array([0, 0, 1.0]), d=0.0, inlier_mask=np.array([]), rmse=0.0)
    pts = np.array([[0, 0, 0], [1, 2, 0]], dtype=np.float32)
    h = signed_height_above_plane(pts, plane)
    assert np.allclose(h, 0)


def test_max_height_finds_peak() -> None:
    plane = GroundPlane(normal=np.array([0, 0, 1.0]), d=0.0, inlier_mask=np.array([]), rmse=0.0)
    pts = np.array([[0, 0, 0.1], [1, 2, 0.5]], dtype=np.float32)
    assert mask_max_height(pts, plane) == 0.5


def test_stats_shape() -> None:
    plane = GroundPlane(normal=np.array([0, 0, 1.0]), d=0.0, inlier_mask=np.array([]), rmse=0.0)
    pts = np.array([[0, 0, 0.1], [1, 2, 0.2], [3, 4, 0.15]], dtype=np.float32)
    s = mask_height_stats(pts, plane)
    assert s["count"] == 3
    assert s["max"] > s["mean"]
