"""End-to-end pipeline orchestration."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rocksnitch.contracts import (
    DepthEstimatorProto,
    FeatureExtractorProto,
    RockDetection,
    SegmenterProto,
    StereoMatcherProto,
    StereoPair,
    UInt8Array,
)
from rocksnitch.io.cahvore import intrinsics_matrix, linearize_to_cahv
from rocksnitch.pipeline.far_field import FarFieldConfig, run_far_field
from rocksnitch.pipeline.fuse import FusionConfig, fuse_detections
from rocksnitch.pipeline.near_field import NearFieldConfig, run_near_field


@dataclass
class PipelineConfig:
    near: NearFieldConfig
    far: FarFieldConfig
    fusion: FusionConfig


@dataclass
class PipelineResult:
    detections: list[RockDetection]
    near_artefacts: Any | None
    far_detections: list[RockDetection]

    def to_jsonable(self) -> dict:
        return {
            "detections": [asdict(d) for d in self.detections],
            "far_detections": [asdict(d) for d in self.far_detections],
        }


def run_pipeline(
    pair: StereoPair,
    left_image: UInt8Array,
    right_image: UInt8Array,
    *,
    stereo: StereoMatcherProto,
    segmenter: SegmenterProto,
    depth_estimator: DepthEstimatorProto,
    features: FeatureExtractorProto | None = None,
    height_head: Any = None,
    config: PipelineConfig | None = None,
) -> PipelineResult:
    cfg = config or PipelineConfig(
        near=NearFieldConfig(), far=FarFieldConfig(), fusion=FusionConfig()
    )
    near, art = run_near_field(
        pair, left_image, right_image, stereo=stereo, segmenter=segmenter, config=cfg.near
    )
    K_linear = intrinsics_matrix(linearize_to_cahv(pair.left.camera_model))
    far = run_far_field(
        left_image,
        segmenter=segmenter,
        depth_estimator=depth_estimator,
        features=features,
        K=K_linear,
        height_head=height_head,
        config=cfg.far,
    )
    fused = fuse_detections(near, far, config=cfg.fusion)
    return PipelineResult(detections=fused, near_artefacts=art, far_detections=far)


def write_detections_json(result: PipelineResult, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_jsonable()
    # Stringify numpy tuples, etc
    def _default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        return str(o)

    out_path.write_text(json.dumps(payload, indent=2, default=_default))


__all__ = [
    "PipelineConfig",
    "PipelineResult",
    "run_pipeline",
    "write_detections_json",
]
