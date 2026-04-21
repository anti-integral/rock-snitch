"""DINOv2 patch-feature extractor."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rocksnitch.contracts import FeatureExtractorProto, FloatArray, Mask2D, UInt8Array


@dataclass
class DINOv2Config:
    hf_repo: str = "facebook/dinov2-large"
    device: str = "cuda"
    image_size: int = 518  # DINOv2-L native
    patch_size: int = 14


class DINOv2FeatureExtractor(FeatureExtractorProto):
    """Return per-patch features of shape (Hp, Wp, D)."""

    def __init__(self, config: DINOv2Config | None = None) -> None:
        self.config = config or DINOv2Config()
        self._model: Any = None
        self._processor: Any = None

    def _load(self) -> None:
        try:
            import torch  # noqa: F401
            from transformers import AutoImageProcessor, AutoModel  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError("transformers not installed. pip install -e '.[gpu]'") from e
        self._processor = AutoImageProcessor.from_pretrained(self.config.hf_repo)
        self._model = AutoModel.from_pretrained(self.config.hf_repo).to(self.config.device).eval()

    def extract(self, image: UInt8Array) -> FloatArray:
        if self._model is None:
            self._load()
        import torch

        processed = self._processor(images=image, return_tensors="pt").to(self.config.device)
        with torch.no_grad():
            out = self._model(**processed)
        tokens = out.last_hidden_state  # (1, N+1, D)
        patches = tokens[:, 1:, :].squeeze(0).cpu().numpy()
        N, D = patches.shape
        side = int(np.sqrt(N))
        if side * side != N:
            raise RuntimeError(f"DINOv2 returned non-square token grid: {N}")
        return patches.reshape(side, side, D).astype(np.float32)


def pool_mask_features(
    features: FloatArray, mask: np.ndarray
) -> FloatArray:
    """Mean-pool DINOv2 patch features inside a coarse mask (Hp, Wp)."""
    if features.shape[:2] != mask.shape:
        # Resize the mask via nearest neighbour to patch grid
        import cv2

        mask_resized = cv2.resize(
            mask.astype(np.uint8),
            (features.shape[1], features.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    else:
        mask_resized = mask.astype(bool)
    if not mask_resized.any():
        return np.zeros(features.shape[-1], dtype=np.float32)
    return features[mask_resized].mean(axis=0).astype(np.float32)


def pool_masklist_features(
    features: FloatArray, masks: list[Mask2D], image_size: tuple[int, int]
) -> FloatArray:
    """Return (K, D) float32 stack of per-mask mean features."""
    import cv2

    Hp, Wp, D = features.shape
    H, W = image_size
    out = np.zeros((len(masks), D), dtype=np.float32)
    for i, m in enumerate(masks):
        small = cv2.resize(
            m.mask.astype(np.uint8), (Wp, Hp), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        if small.any():
            out[i] = features[small].mean(axis=0)
    return out


class MockFeatureExtractor(FeatureExtractorProto):
    """Deterministic mock that returns a small random feature grid."""

    def __init__(self, dim: int = 32, grid: int = 16) -> None:
        self.dim = dim
        self.grid = grid

    def extract(self, image: UInt8Array) -> FloatArray:
        rng = np.random.default_rng(0)
        return rng.standard_normal((self.grid, self.grid, self.dim)).astype(np.float32)


__all__ = [
    "DINOv2Config",
    "DINOv2FeatureExtractor",
    "MockFeatureExtractor",
    "pool_mask_features",
    "pool_masklist_features",
]
