from __future__ import annotations

import numpy as np

from rocksnitch.perception.mono_depth import MockDepthEstimator
from rocksnitch.perception.sam2 import MockSegmenter
from rocksnitch.pipeline.far_field import FarFieldConfig, run_far_field


def test_far_field_produces_detections() -> None:
    img = np.zeros((64, 128, 3), dtype=np.uint8)
    img[20:40, 40:80] = 200
    K = np.array([[100.0, 0.0, 64.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]])
    det = run_far_field(
        img,
        segmenter=MockSegmenter(min_area=20),
        depth_estimator=MockDepthEstimator(base=30.0),
        K=K,
        config=FarFieldConfig(min_height_m=0.0, min_range_m=0.0, max_range_m=1000.0),
    )
    assert any(d.source == "mono" for d in det)
