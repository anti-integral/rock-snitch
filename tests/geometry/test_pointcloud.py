from __future__ import annotations

import numpy as np

from rocksnitch.contracts import DisparityMap, RectifiedPair
from rocksnitch.geometry.pointcloud import disparity_to_pointcloud, mask_points


def _pair(H: int = 32, W: int = 64) -> RectifiedPair:
    K = np.array([[100.0, 0.0, W / 2], [0.0, 100.0, H / 2], [0.0, 0.0, 1.0]])
    return RectifiedPair(
        left=np.zeros((H, W, 3), dtype=np.uint8),
        right=np.zeros((H, W, 3), dtype=np.uint8),
        K=K,
        baseline_m=0.4,
        left_to_world=np.eye(4),
    )


def test_disparity_to_pointcloud_depth_consistent() -> None:
    pair = _pair()
    H, W = 32, 64
    disp = np.full((H, W), 10.0, dtype=np.float32)
    dmap = DisparityMap(disparity=disp, confidence=np.ones_like(disp), mask=np.ones_like(disp, dtype=bool))
    pc = disparity_to_pointcloud(pair, dmap)
    assert pc.valid.all()
    z = pc.xyz[..., 2]
    assert np.allclose(z, 100.0 * 0.4 / 10.0)


def test_mask_points_filters() -> None:
    pair = _pair()
    H, W = 32, 64
    disp = np.full((H, W), 10.0, dtype=np.float32)
    dmap = DisparityMap(disparity=disp, confidence=np.ones_like(disp), mask=np.ones_like(disp, dtype=bool))
    pc = disparity_to_pointcloud(pair, dmap)
    mask = np.zeros((H, W), dtype=bool)
    mask[10:15, 20:25] = True
    pts = mask_points(pc, mask)
    assert pts.shape == (5 * 5, 3)
