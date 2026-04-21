from __future__ import annotations

import cv2
import numpy as np

from rocksnitch.contracts import StereoPair
from rocksnitch.geometry.disparity import SGBMMatcher
from rocksnitch.perception.sam2 import MockSegmenter
from rocksnitch.pipeline.near_field import NearFieldConfig, run_near_field


def _synth_scene(pair: StereoPair) -> tuple[np.ndarray, np.ndarray]:
    H, W = pair.left.camera_model.image_size
    left = np.zeros((H, W, 3), dtype=np.uint8)
    for _ in range(200):
        x, y = np.random.default_rng().integers(0, W - 8), np.random.default_rng().integers(0, H - 8)
        left[y : y + 6, x : x + 6] = np.random.default_rng().integers(60, 255)
    shift = 8
    right = np.roll(left, -shift, axis=1)
    return left, right


def test_near_field_pipeline_runs(synthetic_stereo_pair: StereoPair) -> None:
    left, right = _synth_scene(synthetic_stereo_pair)
    detections, artefacts = run_near_field(
        synthetic_stereo_pair,
        left,
        right,
        stereo=SGBMMatcher(),
        segmenter=MockSegmenter(min_area=20),
        config=NearFieldConfig(min_height_m=0.0, max_range_m=100.0),
    )
    assert artefacts.rectified.left.shape == left.shape
    assert artefacts.disparity.disparity.shape == left.shape[:2]
    for d in detections:
        assert d.source == "stereo"
