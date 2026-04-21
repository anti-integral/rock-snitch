from __future__ import annotations

import numpy as np

from rocksnitch.perception.mono_depth import MockDepthEstimator


def test_mock_depth_shape_and_range() -> None:
    est = MockDepthEstimator()
    image = np.zeros((32, 64, 3), dtype=np.uint8)
    out = est.predict(image)
    assert out.depth.shape == (32, 64)
    assert out.uncertainty is not None
    assert np.all(out.depth > 0)
