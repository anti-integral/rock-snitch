"""Fuse near-field (stereo) + far-field (mono) detections by range."""
from __future__ import annotations

from dataclasses import dataclass

from rocksnitch.contracts import RockDetection, STEREO_TRUST_RANGE_M


@dataclass
class FusionConfig:
    stereo_trust_range_m: float = STEREO_TRUST_RANGE_M
    iou_merge_thresh: float = 0.3


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / max(union, 1)


def fuse_detections(
    near: list[RockDetection],
    far: list[RockDetection],
    config: FusionConfig | None = None,
) -> list[RockDetection]:
    """Combine the two lists, preferring stereo when range <= trust threshold."""
    cfg = config or FusionConfig()
    out: list[RockDetection] = []
    for n in near:
        if n.range_m <= cfg.stereo_trust_range_m:
            out.append(n)
    kept_stereo_bboxes = [d.uv_bbox for d in out]
    for f in far:
        if f.range_m <= cfg.stereo_trust_range_m:
            if any(_iou(f.uv_bbox, b) > cfg.iou_merge_thresh for b in kept_stereo_bboxes):
                continue
        out.append(f)
    # Re-tag as fused when a mono detection overlaps a stereo one
    merged: list[RockDetection] = []
    for det in out:
        other = [d for d in out if d is not det and _iou(det.uv_bbox, d.uv_bbox) > cfg.iou_merge_thresh]
        if other:
            merged.append(
                RockDetection(
                    uv_bbox=det.uv_bbox,
                    mask_rle=det.mask_rle,
                    centroid_uv=det.centroid_uv,
                    range_m=det.range_m,
                    height_m=max(det.height_m, *(d.height_m for d in other)),
                    width_m=max(det.width_m, *(d.width_m for d in other)),
                    confidence=max(det.confidence, *(d.confidence for d in other)),
                    source="fused",
                    xyz_rover=det.xyz_rover,
                )
            )
        else:
            merged.append(det)
    return merged


__all__ = ["FusionConfig", "fuse_detections"]
