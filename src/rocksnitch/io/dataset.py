"""Torch-friendly Dataset over stereo pairs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np

from rocksnitch.contracts import ImageMeta, StereoPair, UInt8Array
from rocksnitch.io.metadata import load_meta
from rocksnitch.io.pairing import load_index


def _load_rgb(path: Path) -> UInt8Array:
    """Read an image and return (H, W, 3) uint8 RGB."""
    try:
        import cv2

        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except ImportError:  # pragma: no cover
        from PIL import Image

        return np.asarray(Image.open(path).convert("RGB"))


@dataclass(frozen=True)
class StereoSample:
    pair: StereoPair
    left_image: UInt8Array
    right_image: UInt8Array

    @property
    def sol(self) -> int:
        return self.pair.left.sol


class StereoPairDataset:
    """Flat iterable / indexable dataset over stereo pairs.

    ``source`` may be either a list of :class:`StereoPair` or a parquet index
    path (loaded via :func:`load_index`).
    """

    def __init__(
        self,
        source: Sequence[StereoPair] | Path | str,
        transform: Callable[[StereoSample], StereoSample] | None = None,
    ) -> None:
        if isinstance(source, (str, Path)):
            rows = load_index(Path(source))
            # Reconstruct StereoPair by re-parsing JSONs for each row.
            self._pairs = [self._row_to_pair(r) for r in rows]
        else:
            self._pairs = list(source)
        self._transform = transform

    @staticmethod
    def _row_to_pair(row: dict) -> StereoPair:
        left = load_meta(Path(row["left_json_path"]), image_path=Path(row["left_image_path"]))
        right = load_meta(Path(row["right_json_path"]), image_path=Path(row["right_image_path"]))
        return StereoPair(
            left=left,
            right=right,
            sclk_delta_s=row.get("sclk_delta_s", 0.0),
            mast_az_delta_deg=row.get("mast_az_delta_deg", 0.0),
            mast_el_delta_deg=row.get("mast_el_delta_deg", 0.0),
        )

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> StereoSample:
        pair = self._pairs[idx]
        left_img = _load_rgb(pair.left.image_path)
        right_img = _load_rgb(pair.right.image_path)
        sample = StereoSample(pair=pair, left_image=left_img, right_image=right_img)
        if self._transform is not None:
            sample = self._transform(sample)
        return sample

    def __iter__(self) -> Iterator[StereoSample]:
        for i in range(len(self)):
            yield self[i]

    def pairs(self) -> list[StereoPair]:
        return list(self._pairs)


class MonoDataset:
    """Single-image iterable for the mono branch."""

    def __init__(self, metas: Sequence[ImageMeta]) -> None:
        self._metas = list(metas)

    def __len__(self) -> int:
        return len(self._metas)

    def __getitem__(self, idx: int) -> tuple[ImageMeta, UInt8Array]:
        m = self._metas[idx]
        return m, _load_rgb(m.image_path)


__all__ = ["StereoPairDataset", "MonoDataset", "StereoSample"]
