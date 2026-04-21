"""Far-field mono-depth + SAM2 branch."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rocksnitch.contracts import (
    BoolArray,
    DepthEstimatorProto,
    DepthMap,
    FeatureExtractorProto,
    MaskList,
    RockDetection,
    SegmenterProto,
    UInt8Array,
)


@dataclass
class FarFieldConfig:
    min_height_m: float = 0.10
    min_mask_pixels: int = 10
    min_range_m: float = 10.0
    max_range_m: float = 120.0


def _bbox_from_mask(mask: BoolArray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return 0, 0, 0, 0
    x0, y0 = int(xs.min()), int(ys.min())
    return x0, y0, int(xs.max() - x0 + 1), int(ys.max() - y0 + 1)


def _project_height(
    mask: BoolArray,
    depth: DepthMap,
    K: np.ndarray,
) -> tuple[float, float, float]:
    """Estimate (range_m, height_m, width_m) for a mask using pinhole geometry."""
    ys, xs = np.where(mask)
    if xs.size == 0:
        return 0.0, 0.0, 0.0
    d_vals = depth.depth[mask]
    valid = np.isfinite(d_vals) & (d_vals > 0)
    if not valid.any():
        return 0.0, 0.0, 0.0
    d_valid = d_vals[valid]
    xs_v = xs[valid]
    ys_v = ys[valid]
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    X = (xs_v - cx) * d_valid / fx
    Y = (ys_v - cy) * d_valid / fy
    Z = d_valid
    range_m = float(np.sqrt((X ** 2 + Y ** 2 + Z ** 2).mean()))
    height_m = float(Y.max() - Y.min())
    width_m = float(X.max() - X.min())
    return range_m, height_m, width_m


def run_far_field(
    image: UInt8Array,
    *,
    segmenter: SegmenterProto,
    depth_estimator: DepthEstimatorProto,
    features: FeatureExtractorProto | None = None,
    K: np.ndarray | None = None,
    height_head: Any = None,
    config: FarFieldConfig | None = None,
) -> list[RockDetection]:
    """Run the mono branch. ``height_head``, if provided, is called with the
    per-mask feature vector to produce a refined height (metres).
    """
    cfg = config or FarFieldConfig()
    masks: MaskList = segmenter.segment(image)
    depth = depth_estimator.predict(image, K=K if K is not None else None)

    # If features/head provided, pool features for all masks up-front.
    pooled = None
    if features is not None and len(masks.masks) > 0:
        feats = features.extract(image)
        from rocksnitch.perception.dinov2 import pool_masklist_features

        pooled = pool_masklist_features(feats, masks.masks, image_size=image.shape[:2])

    detections: list[RockDetection] = []
    K_arr = K if K is not None else np.eye(3)
    for i, m in enumerate(masks.masks):
        if int(m.mask.sum()) < cfg.min_mask_pixels:
            continue
        range_m, height_m, width_m = _project_height(m.mask, depth, K_arr)
        if range_m < cfg.min_range_m or range_m > cfg.max_range_m:
            continue
        if height_head is not None and pooled is not None:
            try:
                pred = float(height_head(pooled[i]))
                height_m = max(height_m, pred)
            except Exception:
                pass
        if height_m < cfg.min_height_m:
            continue
        bbox = _bbox_from_mask(m.mask)
        ys, xs = np.where(m.mask)
        cu, cv = float(xs.mean()), float(ys.mean())
        confidence = float(min(1.0, m.score))
        detections.append(
            RockDetection(
                uv_bbox=bbox,
                mask_rle=None,
                centroid_uv=(cu, cv),
                range_m=range_m,
                height_m=height_m,
                width_m=width_m,
                confidence=confidence,
                source="mono",
            )
        )
    return detections


__all__ = ["FarFieldConfig", "run_far_field"]
