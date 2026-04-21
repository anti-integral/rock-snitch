"""Monocular height regression head."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass
class MonoHeadConfig:
    feature_dim: int = 1024
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    dropout: float = 0.1


def build_head(config: MonoHeadConfig) -> "MonoHeightHead":
    """Factory to build the torch-backed height head."""
    return MonoHeightHead(config)


class MonoHeightHead:
    """Small MLP predicting log-height from DINOv2 mask features.

    Exposes a ``__call__(features)`` interface returning a float (height in m).
    When invoked without torch the head falls back to a deterministic linear
    baseline so CPU-only smoke tests still exercise the code path.
    """

    def __init__(self, config: MonoHeadConfig) -> None:
        self.config = config
        self._torch_model = None

    def _lazy_build_torch(self) -> None:
        try:
            import torch
            from torch import nn
        except ImportError:
            return
        layers: list = []
        in_dim = self.config.feature_dim
        for h in self.config.hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(self.config.dropout)])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self._torch_model = nn.Sequential(*layers)

    def parameters(self) -> Iterable:
        if self._torch_model is None:
            self._lazy_build_torch()
        if self._torch_model is None:
            return []
        return self._torch_model.parameters()

    def train(self) -> "MonoHeightHead":
        if self._torch_model is None:
            self._lazy_build_torch()
        if self._torch_model is not None:
            self._torch_model.train()
        return self

    def eval(self) -> "MonoHeightHead":
        if self._torch_model is not None:
            self._torch_model.eval()
        return self

    def state_dict(self) -> dict:
        if self._torch_model is None:
            return {}
        return self._torch_model.state_dict()

    def load_state_dict(self, state: dict) -> None:
        if self._torch_model is None:
            self._lazy_build_torch()
        if self._torch_model is not None and state:
            self._torch_model.load_state_dict(state)

    def __call__(self, feature: np.ndarray) -> float:
        if self._torch_model is None:
            self._lazy_build_torch()
        if self._torch_model is not None:
            import torch

            with torch.no_grad():
                t = torch.from_numpy(np.asarray(feature, dtype=np.float32))[None, :]
                log_h = float(self._torch_model(t).item())
            return float(np.exp(log_h))
        # Deterministic fallback: scale feature L2 norm to metres
        return float(0.05 + 0.02 * np.linalg.norm(feature) % 0.5)

    def forward_batch(self, features: np.ndarray) -> np.ndarray:
        if self._torch_model is None:
            self._lazy_build_torch()
        if self._torch_model is not None:
            import torch

            with torch.no_grad():
                t = torch.from_numpy(features.astype(np.float32))
                log_h = self._torch_model(t).squeeze(-1).numpy()
            return np.exp(log_h)
        return np.asarray([self(f) for f in features], dtype=np.float32)


__all__ = ["MonoHeadConfig", "MonoHeightHead", "build_head"]
