from __future__ import annotations

from pathlib import Path

import numpy as np

from rocksnitch.contracts import RockDetection
from rocksnitch.eval.viz import overlay_detections, write_overlay


def _det() -> RockDetection:
    return RockDetection(
        uv_bbox=(10, 10, 40, 30),
        mask_rle=None,
        centroid_uv=(30, 25),
        range_m=12.5,
        height_m=0.15,
        width_m=0.2,
        confidence=0.8,
        source="stereo",
    )


def test_overlay_returns_same_shape() -> None:
    img = np.zeros((80, 120, 3), dtype=np.uint8)
    out = overlay_detections(img, [_det()])
    assert out.shape == img.shape


def test_write_overlay(tmp_path: Path) -> None:
    img = np.zeros((80, 120, 3), dtype=np.uint8)
    p = tmp_path / "o.png"
    write_overlay(img, [_det()], p)
    assert p.exists() and p.stat().st_size > 0
