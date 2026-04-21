from __future__ import annotations

import numpy as np

from rocksnitch.contracts import RectifiedPair
from rocksnitch.geometry.disparity import SGBMMatcher


def _shifted_pair(shift: int = 10) -> RectifiedPair:
    H, W = 128, 256
    rng = np.random.default_rng(0)
    left = rng.integers(0, 255, size=(H, W), dtype=np.uint8)
    left = np.stack([left] * 3, axis=-1)
    right = np.roll(left, -shift, axis=1)
    K = np.array([[500.0, 0.0, W / 2.0], [0.0, 500.0, H / 2.0], [0.0, 0.0, 1.0]])
    return RectifiedPair(
        left=left,
        right=right,
        K=K,
        baseline_m=0.4,
        left_to_world=np.eye(4),
    )


def test_sgbm_recovers_constant_disparity() -> None:
    pair = _shifted_pair(shift=16)
    matcher = SGBMMatcher()
    d = matcher.compute(pair)
    assert d.disparity.shape == pair.left.shape[:2]
    median = np.nanmedian(d.disparity)
    # SGBM on synthetic noise finds disparity within a couple of pixels
    assert abs(median - 16) <= 3
