from __future__ import annotations

import numpy as np
import pytest

from rocksnitch.contracts import CameraModel, StereoPair
from rocksnitch.geometry.rectify import (
    compute_rectification,
    disparity_to_depth,
    rectify_pair,
)


def test_compute_rectification_baseline(
    synthetic_left_cam: CameraModel, synthetic_right_cam: CameraModel
) -> None:
    out = compute_rectification(synthetic_left_cam, synthetic_right_cam)
    assert out.baseline_m == pytest.approx(0.424, rel=1e-6)
    assert out.K_common.shape == (3, 3)
    # Principal point roughly at image centre
    cx = out.K_common[0, 2]
    assert 600 < cx < 660


def test_compute_rectification_zero_baseline_raises() -> None:
    from tests.conftest import _make_camera  # type: ignore[attr-defined]

    cam = _make_camera(C=np.zeros(3))
    with pytest.raises(ValueError):
        compute_rectification(cam, cam)


def test_rectify_pair_returns_same_size(
    synthetic_stereo_pair: StereoPair,
) -> None:
    H, W = synthetic_stereo_pair.left.camera_model.image_size
    left = np.random.default_rng(0).integers(0, 255, size=(H, W, 3), dtype=np.uint8)
    right = np.random.default_rng(1).integers(0, 255, size=(H, W, 3), dtype=np.uint8)
    rp = rectify_pair(synthetic_stereo_pair, left, right)
    assert rp.left.shape == (H, W, 3)
    assert rp.right.shape == (H, W, 3)


def test_disparity_to_depth_matches_f_b_over_d() -> None:
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    disp = np.array([[10.0, 5.0], [1.0, np.nan]], dtype=np.float32)
    depth = disparity_to_depth(disp, K, baseline_m=0.4)
    assert depth[0, 0] == pytest.approx(500 * 0.4 / 10.0)
    assert depth[0, 1] == pytest.approx(500 * 0.4 / 5.0)
    assert np.isnan(depth[1, 1])
