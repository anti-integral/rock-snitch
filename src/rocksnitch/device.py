"""Device auto-detection for CUDA / Apple MPS / CPU.

Single source of truth used by the CLI, the Gradio UI, and the backend
constructors. Returns a string compatible with ``torch.device(...)``.
"""
from __future__ import annotations

from typing import Literal


Device = Literal["cuda", "mps", "cpu"]


def detect_device(prefer: str | None = None) -> Device:
    """Return the best available device.

    Order of preference:
      1. an explicit ``prefer`` argument if it is actually available
      2. CUDA if visible
      3. Apple Silicon MPS if visible
      4. CPU as a last resort

    Never raises. If torch is not installed, returns ``"cpu"``.
    """
    if prefer is not None:
        prefer = prefer.lower().strip()
        if prefer in ("cuda", "mps", "cpu") and _is_available(prefer):
            return prefer  # type: ignore[return-value]
    if _is_available("cuda"):
        return "cuda"
    if _is_available("mps"):
        return "mps"
    return "cpu"


def _is_available(device: str) -> bool:
    if device == "cpu":
        return True
    try:
        import torch
    except ImportError:
        return False
    if device == "cuda":
        try:
            return bool(torch.cuda.is_available())
        except Exception:
            return False
    if device == "mps":
        try:
            return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        except Exception:
            return False
    return False


def device_label(device: Device) -> str:
    """Human-readable label for the UI."""
    if device == "cuda":
        return "CUDA (NVIDIA GPU)"
    if device == "mps":
        return "MPS (Apple Silicon GPU)"
    return "CPU"


__all__ = ["Device", "detect_device", "device_label"]
