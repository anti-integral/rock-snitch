from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

from rocksnitch.contracts import StereoPair
from rocksnitch.io.dataset import StereoPairDataset


def _write_checker(path: Path, size: tuple[int, int] = (240, 320)) -> None:
    H, W = size
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[::32, :, :] = 255
    img[:, ::32, :] = 255
    cv2.imwrite(str(path), img)


def test_stereo_pair_dataset_iteration(synthetic_stereo_pair: StereoPair) -> None:
    _write_checker(synthetic_stereo_pair.left.image_path)
    _write_checker(synthetic_stereo_pair.right.image_path)
    ds = StereoPairDataset([synthetic_stereo_pair])
    assert len(ds) == 1
    sample = ds[0]
    assert sample.left_image.shape[2] == 3
    assert sample.right_image.shape[2] == 3
    assert sample.sol == synthetic_stereo_pair.left.sol
