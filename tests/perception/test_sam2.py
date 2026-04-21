from __future__ import annotations

import numpy as np

from rocksnitch.perception.sam2 import MockSegmenter


def test_mock_segmenter_finds_bright_blobs() -> None:
    img = np.zeros((80, 80, 3), dtype=np.uint8)
    img[10:20, 10:20] = 255
    img[40:55, 40:55] = 255
    seg = MockSegmenter(min_area=50)
    masks = seg.segment(img)
    assert masks.image_size == (80, 80)
    assert 1 <= len(masks.masks) <= 2
    for m in masks.masks:
        assert m.mask.shape == (80, 80)
        assert m.bbox_xywh[2] > 0
        assert m.bbox_xywh[3] > 0
        assert m.score > 0
