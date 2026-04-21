from __future__ import annotations

import numpy as np

from rocksnitch.training.mono_head import MonoHeadConfig, build_head


def test_head_callable_returns_float() -> None:
    head = build_head(MonoHeadConfig(feature_dim=16))
    y = head(np.zeros(16, dtype=np.float32))
    assert isinstance(y, float)
    assert y >= 0.0


def test_head_forward_batch() -> None:
    head = build_head(MonoHeadConfig(feature_dim=8))
    features = np.random.default_rng(0).standard_normal((4, 8)).astype(np.float32)
    y = head.forward_batch(features)
    assert y.shape == (4,)
