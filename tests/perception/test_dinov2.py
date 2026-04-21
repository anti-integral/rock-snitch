from __future__ import annotations

import numpy as np

from rocksnitch.contracts import Mask2D
from rocksnitch.perception.dinov2 import (
    MockFeatureExtractor,
    pool_mask_features,
    pool_masklist_features,
)


def test_mock_features_shape() -> None:
    ex = MockFeatureExtractor(dim=16, grid=8)
    feats = ex.extract(np.zeros((64, 64, 3), dtype=np.uint8))
    assert feats.shape == (8, 8, 16)


def test_pool_mask_features_basic() -> None:
    feats = np.ones((8, 8, 4), dtype=np.float32)
    mask = np.zeros((16, 16), dtype=bool)
    mask[:8, :8] = True
    pooled = pool_mask_features(feats, mask)
    assert pooled.shape == (4,)
    assert np.allclose(pooled, 1.0)


def test_pool_mask_features_empty_returns_zeros() -> None:
    feats = np.ones((4, 4, 3), dtype=np.float32)
    pooled = pool_mask_features(feats, np.zeros((4, 4), dtype=bool))
    assert np.allclose(pooled, 0)


def test_pool_masklist_features() -> None:
    feats = np.ones((4, 4, 5), dtype=np.float32)
    mask1 = np.zeros((8, 8), dtype=bool)
    mask1[0:4, 0:4] = True
    mask2 = np.zeros((8, 8), dtype=bool)
    mask2[4:, 4:] = True
    masks = [
        Mask2D(mask=mask1, bbox_xywh=(0, 0, 4, 4), score=1.0),
        Mask2D(mask=mask2, bbox_xywh=(4, 4, 4, 4), score=1.0),
    ]
    out = pool_masklist_features(feats, masks, image_size=(8, 8))
    assert out.shape == (2, 5)
