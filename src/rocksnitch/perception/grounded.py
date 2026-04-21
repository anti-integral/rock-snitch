"""GroundingDINO text-prompted box detector + SAM2 prompt bridge."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rocksnitch.contracts import UInt8Array


@dataclass
class GroundingDINOConfig:
    hf_repo: str = "IDEA-Research/grounding-dino-base"
    device: str = "cuda"
    text_prompt: str = "rock. boulder. stone."
    box_threshold: float = 0.3
    text_threshold: float = 0.25


class GroundingDINO:
    """Return a list of bounding boxes for the prompt."""

    def __init__(self, config: GroundingDINOConfig | None = None) -> None:
        self.config = config or GroundingDINOConfig()
        self._model: Any = None
        self._processor: Any = None

    def _load(self) -> None:
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError("transformers not installed.") from e
        self._processor = AutoProcessor.from_pretrained(self.config.hf_repo)
        self._model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(self.config.hf_repo)
            .to(self.config.device)
            .eval()
        )

    def detect(self, image: UInt8Array) -> list[tuple[int, int, int, int, float]]:
        """Return list of (x, y, w, h, score) boxes."""
        if self._model is None:
            self._load()
        import torch

        inputs = self._processor(
            images=image, text=self.config.text_prompt, return_tensors="pt"
        ).to(self.config.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        H, W = image.shape[:2]
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.config.box_threshold,
            text_threshold=self.config.text_threshold,
            target_sizes=[(H, W)],
        )[0]
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        out: list[tuple[int, int, int, int, float]] = []
        for (x1, y1, x2, y2), s in zip(boxes, scores):
            out.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1), float(s)))
        return out


__all__ = ["GroundingDINOConfig", "GroundingDINO"]
