"""Pipeline profile presets.

A *profile* bundles a set of feature toggles and backend choices into one
named configuration. Profiles cover the common operational stances:

  - ``full``       — every branch on, real GPU models. The default.
  - ``stereo-only`` — only the geometric near-field branch. No GPU needed
                      for inference (SAM2 backend is the only GPU model;
                      see ``--segmenter mock`` to drop GPU entirely).
  - ``mono-only``  — far-field branch only. Useful for single-image input
                      (no stereo partner) or for benchmarking the learned
                      head in isolation.
  - ``minimal``    — everything mocked, fastest possible. Smoke-test only.
  - ``mock``       — alias of ``minimal`` for clarity in CI scripts.

Use as a CLI shortcut: ``rock-snitch detect --profile stereo-only ...``.

Individual ``--enable-X`` / ``--disable-X`` flags override profile settings.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rocksnitch.pipeline.run import PipelineConfig

SegmenterChoice = Literal["sam2", "mock"]
DepthChoice = Literal["unidepth", "metric3d", "mock", "none"]
FeatureChoice = Literal["dinov2", "mock", "none"]
StereoChoice = Literal["sgbm", "raft", "none"]


@dataclass
class ProfileSpec:
    """Bundle of CLI-configurable choices that defines a profile."""

    name: str
    description: str
    pipeline: PipelineConfig
    segmenter: SegmenterChoice = "sam2"
    depth: DepthChoice = "unidepth"
    features: FeatureChoice = "dinov2"
    stereo: StereoChoice = "sgbm"
    use_grounding_dino: bool = False

    def explain(self) -> str:
        """One-line summary suitable for CLI logs."""
        parts = [
            f"profile={self.name}",
            f"stereo={self.stereo if self.pipeline.enable_stereo else 'off'}",
            f"mono={'on' if self.pipeline.enable_mono else 'off'}",
            f"segmenter={self.segmenter}",
            f"depth={self.depth}",
            f"features={self.features}",
            f"linearize={'on' if self.pipeline.enable_linearize else 'off'}",
        ]
        if self.use_grounding_dino:
            parts.append("grounded=on")
        return " ".join(parts)


def _full() -> ProfileSpec:
    return ProfileSpec(
        name="full",
        description="Every branch on. Stereo + mono fusion, DINOv2 features, "
        "UniDepth depth, CAHVORE linearization. Best detection quality. Needs GPU.",
        pipeline=PipelineConfig(
            enable_stereo=True,
            enable_mono=True,
            enable_linearize=True,
        ),
        segmenter="sam2",
        depth="unidepth",
        features="dinov2",
        stereo="sgbm",
    )


def _stereo_only() -> ProfileSpec:
    return ProfileSpec(
        name="stereo-only",
        description="Geometric near-field only. SGBM disparity + ground plane "
        "+ per-mask height. No mono depth, no learned head. Fast.",
        pipeline=PipelineConfig(
            enable_stereo=True,
            enable_mono=False,
            enable_linearize=False,
        ),
        segmenter="sam2",
        depth="none",
        features="none",
        stereo="sgbm",
    )


def _mono_only() -> ProfileSpec:
    return ProfileSpec(
        name="mono-only",
        description="Single-image far-field branch. UniDepth + SAM2, no stereo. "
        "Use for monocular inputs or pure mono-head ablations.",
        pipeline=PipelineConfig(
            enable_stereo=False,
            enable_mono=True,
            enable_linearize=False,
        ),
        segmenter="sam2",
        depth="unidepth",
        features="dinov2",
        stereo="none",
    )


def _minimal() -> ProfileSpec:
    return ProfileSpec(
        name="minimal",
        description="All-mock backends, no GPU, no weights. Smoke testing only.",
        pipeline=PipelineConfig(
            enable_stereo=True,
            enable_mono=True,
            enable_linearize=False,
        ),
        segmenter="mock",
        depth="mock",
        features="mock",
        stereo="sgbm",
    )


_BUILDERS = {
    "full": _full,
    "stereo-only": _stereo_only,
    "mono-only": _mono_only,
    "minimal": _minimal,
    "mock": _minimal,  # alias
}


def list_profiles() -> list[str]:
    return list(_BUILDERS.keys())


def get_profile(name: str) -> ProfileSpec:
    if name not in _BUILDERS:
        raise KeyError(f"Unknown profile: {name!r}. Choose from {list_profiles()}")
    return _BUILDERS[name]()


__all__ = [
    "DepthChoice",
    "FeatureChoice",
    "ProfileSpec",
    "SegmenterChoice",
    "StereoChoice",
    "get_profile",
    "list_profiles",
]
