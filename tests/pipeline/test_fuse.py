from __future__ import annotations

from rocksnitch.contracts import RockDetection
from rocksnitch.pipeline.fuse import FusionConfig, fuse_detections


def _d(x: int = 0, y: int = 0, w: int = 20, h: int = 20, source: str = "stereo", range_m: float = 10.0, height_m: float = 0.15) -> RockDetection:
    return RockDetection(
        uv_bbox=(x, y, w, h),
        mask_rle=None,
        centroid_uv=(x + w / 2, y + h / 2),
        range_m=range_m,
        height_m=height_m,
        width_m=0.2,
        confidence=0.8,
        source=source,  # type: ignore[arg-type]
    )


def test_fusion_prefers_stereo_near_field() -> None:
    near = [_d(x=0, range_m=5.0)]
    far = [_d(x=0, range_m=5.0, source="mono")]
    out = fuse_detections(near, far, config=FusionConfig(stereo_trust_range_m=20.0, iou_merge_thresh=0.3))
    sources = {d.source for d in out}
    assert sources <= {"stereo", "fused"}


def test_fusion_keeps_far_field_mono() -> None:
    far = [_d(x=100, range_m=50.0, source="mono")]
    out = fuse_detections([], far)
    assert len(out) == 1
    assert out[0].source == "mono"
