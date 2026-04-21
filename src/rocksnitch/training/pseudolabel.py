"""Generate stereo-derived pseudolabels for mono-head training."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from rocksnitch.contracts import (
    FeatureExtractorProto,
    FloatArray,
    RockDetection,
    SegmenterProto,
    StereoMatcherProto,
    StereoPair,
)
from rocksnitch.io.dataset import StereoPairDataset
from rocksnitch.pipeline.near_field import NearFieldConfig, run_near_field


@dataclass
class PseudolabelConfig:
    min_stereo_confidence: float = 0.3
    max_range_m: float = 15.0
    min_height_m: float = 0.08
    max_ground_rmse_m: float = 0.05


@dataclass
class PseudolabelRecord:
    left_imageid: str
    right_imageid: str
    sol: int
    uv_bbox: list[int]
    centroid_uv: list[float]
    range_m: float
    height_m: float
    width_m: float
    confidence: float
    feature: list[float] | None = None


def _det_to_record(
    pair: StereoPair,
    det: RockDetection,
    feature: FloatArray | None,
) -> PseudolabelRecord:
    return PseudolabelRecord(
        left_imageid=pair.left.imageid,
        right_imageid=pair.right.imageid,
        sol=pair.left.sol,
        uv_bbox=list(det.uv_bbox),
        centroid_uv=list(det.centroid_uv),
        range_m=float(det.range_m),
        height_m=float(det.height_m),
        width_m=float(det.width_m),
        confidence=float(det.confidence),
        feature=feature.tolist() if feature is not None else None,
    )


def generate_pseudolabels(
    dataset: StereoPairDataset,
    *,
    stereo: StereoMatcherProto,
    segmenter: SegmenterProto,
    features: FeatureExtractorProto | None = None,
    config: PseudolabelConfig | None = None,
) -> Iterable[PseudolabelRecord]:
    """Yield pseudolabel records over a dataset.

    Filtering: rocks must be within ``max_range_m``, above ``min_height_m``, and
    the underlying ground plane RMSE <= ``max_ground_rmse_m``.
    """
    cfg = config or PseudolabelConfig()
    near_cfg = NearFieldConfig(
        min_height_m=cfg.min_height_m,
        max_range_m=cfg.max_range_m,
        min_stereo_confidence=cfg.min_stereo_confidence,
    )
    for sample in dataset:
        detections, art = run_near_field(
            sample.pair,
            sample.left_image,
            sample.right_image,
            stereo=stereo,
            segmenter=segmenter,
            config=near_cfg,
        )
        if art.ground_plane.rmse > cfg.max_ground_rmse_m:
            continue
        feats_grid = features.extract(sample.left_image) if features is not None else None
        for det in detections:
            if det.confidence < cfg.min_stereo_confidence:
                continue
            feat = None
            if feats_grid is not None:
                from rocksnitch.perception.dinov2 import pool_mask_features
                mask = np.zeros(sample.left_image.shape[:2], dtype=bool)
                x, y, w, h = det.uv_bbox
                mask[y : y + h, x : x + w] = True
                feat = pool_mask_features(feats_grid, mask)
            yield _det_to_record(sample.pair, det, feat)


def write_pseudolabels(records: Iterable[PseudolabelRecord], out_path: Path) -> int:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec)) + "\n")
            count += 1
    return count


def read_pseudolabels(path: Path) -> list[PseudolabelRecord]:
    out: list[PseudolabelRecord] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        out.append(PseudolabelRecord(**d))
    return out


__all__ = [
    "PseudolabelConfig",
    "PseudolabelRecord",
    "generate_pseudolabels",
    "write_pseudolabels",
    "read_pseudolabels",
]
