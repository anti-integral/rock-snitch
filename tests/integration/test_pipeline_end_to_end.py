"""Full pipeline smoke on synthetic data using mocks (no GPU, no models)."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from rocksnitch.contracts import StereoPair
from rocksnitch.geometry.disparity import SGBMMatcher
from rocksnitch.perception.dinov2 import MockFeatureExtractor
from rocksnitch.perception.mono_depth import MockDepthEstimator
from rocksnitch.perception.sam2 import MockSegmenter
from rocksnitch.pipeline.run import run_pipeline, write_detections_json


def _noise_scene(pair: StereoPair) -> tuple[np.ndarray, np.ndarray]:
    H, W = pair.left.camera_model.image_size
    rng = np.random.default_rng(0)
    left = rng.integers(0, 255, size=(H, W, 3), dtype=np.uint8)
    cv2.rectangle(left, (100, 200), (200, 280), (255, 255, 255), -1)
    cv2.circle(left, (600, 400), 30, (230, 230, 230), -1)
    right = np.roll(left, -10, axis=1)
    return left, right


def test_full_pipeline(synthetic_stereo_pair: StereoPair, tmp_path):
    left, right = _noise_scene(synthetic_stereo_pair)
    result = run_pipeline(
        synthetic_stereo_pair,
        left,
        right,
        stereo=SGBMMatcher(),
        segmenter=MockSegmenter(min_area=30),
        depth_estimator=MockDepthEstimator(base=10.0, scale=0.1),
        features=MockFeatureExtractor(dim=32, grid=16),
    )
    out = tmp_path / "detections.json"
    write_detections_json(result, out)
    assert out.exists()
    assert isinstance(result.detections, list)


@pytest.mark.integration
def test_cli_help_smoke(capsys):
    import subprocess

    r = subprocess.run(["rock-snitch", "--help"], capture_output=True, text=True)
    assert r.returncode == 0
    assert "Mars Navcam" in r.stdout
