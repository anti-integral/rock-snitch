"""Monocular metric-depth backends (UniDepthV2, Metric3D v2)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rocksnitch.contracts import DepthEstimatorProto, DepthMap, FloatArray, UInt8Array


@dataclass
class UniDepthConfig:
    hf_repo: str = "lpiccinelli/unidepth-v2-vitl14"
    device: str = "cuda"


class UniDepthV2(DepthEstimatorProto):
    """UniDepthV2 wrapper taking optional rectified K (3x3)."""

    def __init__(self, config: UniDepthConfig | None = None) -> None:
        self.config = config or UniDepthConfig()
        self._model: Any = None

    def _load(self) -> None:
        try:
            import torch  # noqa: F401
            from unidepth.models import UniDepthV2 as _UD  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError("unidepth not installed. pip install -e '.[gpu]'") from e
        self._model = _UD.from_pretrained(self.config.hf_repo).to(self.config.device).eval()

    def predict(self, image: UInt8Array, K: FloatArray | None = None) -> DepthMap:
        if self._model is None:
            self._load()
        import torch

        img_t = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(self.config.device)
        K_t = (
            torch.from_numpy(np.asarray(K, dtype=np.float32)).unsqueeze(0).to(self.config.device)
            if K is not None
            else None
        )
        with torch.no_grad():
            pred = self._model.infer(img_t, K_t)
        depth = pred["depth"].squeeze().cpu().numpy().astype(np.float32)
        unc = None
        if "confidence" in pred:
            unc = pred["confidence"].squeeze().cpu().numpy().astype(np.float32)
        return DepthMap(depth=depth, uncertainty=unc, K=np.asarray(K, dtype=np.float64) if K is not None else None)


class MockDepthEstimator(DepthEstimatorProto):
    """Deterministic mock that returns a synthetic distance-from-image-centre ramp."""

    def __init__(self, base: float = 5.0, scale: float = 0.05) -> None:
        self.base = base
        self.scale = scale

    def predict(self, image: UInt8Array, K: FloatArray | None = None) -> DepthMap:
        H, W = image.shape[:2]
        yy, xx = np.mgrid[0:H, 0:W]
        depth = (self.base + self.scale * np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2)).astype(np.float32)
        return DepthMap(depth=depth, uncertainty=np.ones_like(depth) * 0.1, K=np.asarray(K) if K is not None else None)


__all__ = ["UniDepthConfig", "UniDepthV2", "MockDepthEstimator"]
