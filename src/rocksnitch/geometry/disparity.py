"""Stereo disparity estimation.

Two backends:
  * :class:`SGBMMatcher` - OpenCV SGBM, CPU-only baseline.
  * :class:`RaftStereoMatcher` - GPU (requires RAFT-Stereo weights + PyTorch).

Both implement :class:`StereoMatcherProto`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from rocksnitch.contracts import (
    BoolArray,
    DisparityMap,
    FloatArray,
    RectifiedPair,
    StereoMatcherProto,
)


@dataclass
class SGBMConfig:
    min_disparity: int = 0
    num_disparities: int = 192
    block_size: int = 5
    p1_mult: int = 8
    p2_mult: int = 32
    disp12_max_diff: int = 1
    uniqueness_ratio: int = 10
    speckle_window_size: int = 100
    speckle_range: int = 2
    mode: int = cv2.STEREO_SGBM_MODE_SGBM_3WAY


class SGBMMatcher(StereoMatcherProto):
    """CPU SGBM matcher."""

    def __init__(self, config: SGBMConfig | None = None) -> None:
        self.config = config or SGBMConfig()
        # lazily built per first call because channels are known then
        self._matcher: cv2.StereoSGBM | None = None

    def _build(self, channels: int) -> cv2.StereoSGBM:
        c = self.config
        blk = c.block_size
        return cv2.StereoSGBM_create(
            minDisparity=c.min_disparity,
            numDisparities=c.num_disparities,
            blockSize=blk,
            P1=c.p1_mult * channels * blk * blk,
            P2=c.p2_mult * channels * blk * blk,
            disp12MaxDiff=c.disp12_max_diff,
            uniquenessRatio=c.uniqueness_ratio,
            speckleWindowSize=c.speckle_window_size,
            speckleRange=c.speckle_range,
            mode=c.mode,
        )

    def compute(self, pair: RectifiedPair) -> DisparityMap:
        """Return a DisparityMap in pixels (reference = left)."""
        left = pair.left
        right = pair.right
        if left.ndim == 3:
            left_gray = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
            right_gray = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
            channels = 1
        else:
            left_gray = left
            right_gray = right
            channels = 1
        if self._matcher is None:
            self._matcher = self._build(channels)
        raw = self._matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
        mask: BoolArray = raw > self.config.min_disparity
        # left-right consistency
        if self._matcher is not None:
            right_matcher = cv2.ximgproc.createRightMatcher(self._matcher) if hasattr(
                cv2, "ximgproc"
            ) else None
            if right_matcher is not None:
                raw_r = right_matcher.compute(right_gray, left_gray).astype(np.float32) / 16.0
                # project right disparity back to left
                h, w = raw.shape
                xx = np.arange(w)
                back = np.full_like(raw, np.nan)
                for y in range(h):
                    d = raw[y]
                    x_in_right = xx - d
                    valid = (
                        np.isfinite(d)
                        & (d > 0)
                        & (x_in_right >= 0)
                        & (x_in_right < w)
                    )
                    idx = x_in_right[valid].astype(np.int32)
                    back[y, valid] = -raw_r[y, idx]
                diff = np.abs(raw - back)
                consistent = np.isfinite(diff) & (diff < 1.5)
                mask = mask & consistent
        disparity = np.where(mask, raw, np.nan).astype(np.float32)
        confidence = _confidence_from_disparity(raw, mask)
        return DisparityMap(disparity=disparity, confidence=confidence, mask=mask)


def _confidence_from_disparity(raw: FloatArray, mask: BoolArray) -> FloatArray:
    """Heuristic: confidence = 1 - local gradient of disparity, clipped."""
    if not mask.any():
        return np.zeros_like(raw, dtype=np.float32)
    filled = np.where(np.isfinite(raw), raw, 0.0).astype(np.float32)
    gx = cv2.Sobel(filled, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(filled, cv2.CV_32F, 0, 1, ksize=3)
    g = np.sqrt(gx * gx + gy * gy)
    c = np.clip(1.0 - g / (g.max() + 1e-6), 0.0, 1.0).astype(np.float32)
    c = np.where(mask, c, 0.0)
    return c


class RaftStereoMatcher(StereoMatcherProto):
    """RAFT-Stereo wrapper.

    Loads weights on first call; requires :mod:`torch` and a CUDA device.
    Construction does not import torch so CPU-only installs can reference this
    class without a runtime error.
    """

    def __init__(self, checkpoint: Path | str, device: str = "cuda") -> None:
        self.checkpoint = Path(checkpoint)
        self.device = device
        self._model = None

    def _load(self) -> None:
        import torch  # local import so CPU clients don't pay for it
        try:
            from raft_stereo.core.raft_stereo import RAFTStereo  # type: ignore
            from raft_stereo.core.utils.utils import InputPadder  # type: ignore
        except ImportError as e:
            raise ImportError(
                "RAFT-Stereo not installed. See README.md for install hints."
            ) from e
        ckpt = torch.load(self.checkpoint, map_location="cpu")
        # The RAFT-Stereo checkpoint is a state_dict; wrap in a dummy namespace
        # that RAFTStereo() can load from.
        args = _raft_default_args()
        model = RAFTStereo(args).to(self.device).eval()
        state = ckpt if isinstance(ckpt, dict) else ckpt.get("state_dict", {})
        new_state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(new_state, strict=False)
        self._model = model
        self._InputPadder = InputPadder  # type: ignore[attr-defined]

    def compute(self, pair: RectifiedPair) -> DisparityMap:
        import torch

        if self._model is None:
            self._load()
        assert self._model is not None

        def _to_tensor(arr):
            t = torch.from_numpy(arr).float()
            if t.dim() == 3:
                t = t.permute(2, 0, 1)
            else:
                t = t.unsqueeze(0)
            return t.unsqueeze(0).to(self.device)

        left = _to_tensor(pair.left)
        right = _to_tensor(pair.right)
        padder = self._InputPadder(left.shape, divis_by=32)
        left, right = padder.pad(left, right)
        with torch.no_grad():
            _, flow_up = self._model(left, right, iters=32, test_mode=True)
        disp = -padder.unpad(flow_up)[0, 0].cpu().numpy().astype(np.float32)
        mask = disp > 0
        confidence = _confidence_from_disparity(disp, mask)
        return DisparityMap(
            disparity=np.where(mask, disp, np.nan).astype(np.float32),
            confidence=confidence,
            mask=mask,
        )


def _raft_default_args():
    """Build the small Namespace RAFTStereo() expects."""
    from types import SimpleNamespace

    return SimpleNamespace(
        hidden_dims=[128, 128, 128],
        corr_implementation="reg",
        shared_backbone=False,
        corr_levels=4,
        corr_radius=4,
        n_downsample=2,
        context_norm="batch",
        slow_fast_gru=False,
        n_gru_layers=3,
        mixed_precision=True,
    )


__all__ = [
    "SGBMConfig",
    "SGBMMatcher",
    "RaftStereoMatcher",
]
