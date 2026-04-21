"""SAM2 everything-mode wrapper.

Lazy-loads :mod:`sam2` only when an actual SAM2 backend is constructed so that
CPU-only installs can import the module for tests and type checks.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rocksnitch.contracts import (
    BoolArray,
    Mask2D,
    MaskList,
    SegmenterProto,
    UInt8Array,
)


@dataclass
class SAM2Config:
    checkpoint: Path | str
    config_name: str = "sam2_hiera_l"
    device: str = "cuda"
    points_per_side: int = 32
    pred_iou_thresh: float = 0.65
    stability_score_thresh: float = 0.88
    min_mask_region_area: int = 20
    max_mask_region_area: int = 20_000
    aspect_min: float = 0.2
    aspect_max: float = 5.0


class SAM2Segmenter(SegmenterProto):
    """Wrapper around :class:`sam2.automatic_mask_generator.SAM2AutomaticMaskGenerator`."""

    def __init__(self, config: SAM2Config) -> None:
        self.config = config
        self._generator: Any = None

    def _load(self) -> None:
        try:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # type: ignore
            from sam2.build_sam import build_sam2  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError("sam2 not installed. `pip install -e '.[gpu]'`.") from e
        sam = build_sam2(self.config.config_name, str(self.config.checkpoint), device=self.config.device)
        self._generator = SAM2AutomaticMaskGenerator(
            model=sam,
            points_per_side=self.config.points_per_side,
            pred_iou_thresh=self.config.pred_iou_thresh,
            stability_score_thresh=self.config.stability_score_thresh,
            min_mask_region_area=self.config.min_mask_region_area,
        )

    def segment(self, image: UInt8Array) -> MaskList:
        if self._generator is None:
            self._load()
        assert self._generator is not None
        raw = self._generator.generate(image)
        H, W = image.shape[:2]
        keep: list[Mask2D] = []
        for r in raw:
            mask: BoolArray = r["segmentation"].astype(bool)
            area = int(mask.sum())
            if area < self.config.min_mask_region_area:
                continue
            if area > self.config.max_mask_region_area:
                continue
            ys, xs = np.where(mask)
            if xs.size == 0:
                continue
            x0, y0 = int(xs.min()), int(ys.min())
            bw, bh = int(xs.max() - x0 + 1), int(ys.max() - y0 + 1)
            aspect = bw / max(bh, 1)
            if not (self.config.aspect_min <= aspect <= self.config.aspect_max):
                continue
            keep.append(
                Mask2D(
                    mask=mask,
                    bbox_xywh=(x0, y0, bw, bh),
                    score=float(r.get("predicted_iou", r.get("stability_score", 0.0))),
                    source="sam2",
                )
            )
        return MaskList(image_size=(H, W), masks=keep)


class MockSegmenter(SegmenterProto):
    """Deterministic mock segmenter for tests.

    Finds bright-on-dark connected components via OpenCV thresholding; the
    output is a real :class:`MaskList`, so unit tests that go through the
    pipeline can assert on non-trivial masks without loading a 900 MB network.
    """

    def __init__(self, min_area: int = 20, max_area: int = 20_000) -> None:
        self.min_area = min_area
        self.max_area = max_area

    def segment(self, image: UInt8Array) -> MaskList:
        H, W = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        _, binary = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        masks: list[Mask2D] = []
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            if area < self.min_area or area > self.max_area:
                continue
            comp_mask = (labels == i)
            masks.append(
                Mask2D(
                    mask=comp_mask,
                    bbox_xywh=(int(x), int(y), int(w), int(h)),
                    score=float(min(1.0, area / 500.0)),
                    source="mock",
                )
            )
        return MaskList(image_size=(H, W), masks=masks)


__all__ = ["SAM2Config", "SAM2Segmenter", "MockSegmenter"]
