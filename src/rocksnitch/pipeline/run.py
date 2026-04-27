"""End-to-end pipeline orchestration with optional branches."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

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
from rocksnitch.logging_utils import get_logger
from rocksnitch.pipeline.far_field import FarFieldConfig, run_far_field
from rocksnitch.pipeline.fuse import FusionConfig, fuse_detections
from rocksnitch.pipeline.near_field import NearFieldConfig, run_near_field


@dataclass
class PipelineConfig:
    """Configuration for one detection run.

    All fields with defaults — pass only the ones you want to change.
    """

    # Branch toggles
    enable_stereo: bool = True
    enable_mono: bool = True

    # CAHVORE distortion correction. The delivered Mars2020 raw products
    # have sub-pixel distortion at the image centre and ~5–15 px at the
    # corners (panoramic strips can be far worse). Linearization is a real
    # remap step; off by default for speed since most rock detections live
    # near the image centre. Turn on for corner-precision applications.
    enable_linearize: bool = False

    # Sub-stage configs (preserved for backward compat)
    near: NearFieldConfig = field(default_factory=NearFieldConfig)
    far: FarFieldConfig = field(default_factory=FarFieldConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)


@dataclass
class PipelineResult:
    detections: list[RockDetection]
    near_artefacts: Any | None
    far_detections: list[RockDetection]
    branches_run: list[str] = field(default_factory=list)

    def to_jsonable(self) -> dict:
        return {
            "detections": [asdict(d) for d in self.detections],
            "far_detections": [asdict(d) for d in self.far_detections],
            "branches_run": self.branches_run,
        }


def run_pipeline(
    pair: Optional[StereoPair],
    left_image: UInt8Array,
    right_image: Optional[UInt8Array] = None,
    *,
    stereo: Optional[StereoMatcherProto] = None,
    segmenter: SegmenterProto,
    depth_estimator: Optional[DepthEstimatorProto] = None,
    features: Optional[FeatureExtractorProto] = None,
    height_head: Any = None,
    config: Optional[PipelineConfig] = None,
) -> PipelineResult:
    """Run any subset of branches, gated by config + available inputs.

    Stereo branch runs iff ``config.enable_stereo`` AND ``pair`` AND
    ``right_image`` AND ``stereo`` are all provided. Mono branch runs iff
    ``config.enable_mono`` AND ``depth_estimator`` is provided.
    """
    cfg = config or PipelineConfig()
    log = get_logger(__name__)
    branches: list[str] = []

    near: list[RockDetection] = []
    art = None
    can_stereo = (
        cfg.enable_stereo
        and pair is not None
        and right_image is not None
        and stereo is not None
    )
    if can_stereo:
        log.info("running stereo branch (range <= %.0f m)", cfg.near.max_range_m)
        near, art = run_near_field(
            pair, left_image, right_image, stereo=stereo, segmenter=segmenter, config=cfg.near
        )
        branches.append("stereo")
    elif cfg.enable_stereo:
        log.info("stereo enabled but pair/matcher missing - skipping near-field")

    far: list[RockDetection] = []
    if cfg.enable_mono and depth_estimator is not None:
        if pair is not None:
            cam = pair.left.camera_model
            K = intrinsics_matrix(linearize_to_cahv(cam) if cfg.enable_linearize else cam)
        else:
            K = None
        log.info("running mono branch (depth backend = %s)", type(depth_estimator).__name__)
        far = run_far_field(
            left_image,
            segmenter=segmenter,
            depth_estimator=depth_estimator,
            features=features,
            K=K,
            height_head=height_head,
            config=cfg.far,
        )
        branches.append("mono")
    elif cfg.enable_mono:
        log.info("mono enabled but depth_estimator missing - skipping far-field")

    fused = fuse_detections(near, far, config=cfg.fusion) if branches else []
    log.info(
        "pipeline done: %d detections (stereo=%d mono=%d branches=%s)",
        len(fused),
        len(near),
        len(far),
        ",".join(branches) or "none",
    )
    return PipelineResult(
        detections=fused,
        near_artefacts=art,
        far_detections=far,
        branches_run=branches,
    )


def write_detections_json(result: PipelineResult, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_jsonable()

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
