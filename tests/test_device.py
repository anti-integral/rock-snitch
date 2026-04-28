"""Unit tests for device auto-detection."""
from __future__ import annotations

from rocksnitch.device import detect_device, device_label


def test_detect_device_returns_valid_choice() -> None:
    d = detect_device()
    assert d in {"cuda", "mps", "cpu"}


def test_detect_device_prefers_cpu_when_explicitly_requested() -> None:
    # CPU is always available; preferring it must always honour.
    assert detect_device(prefer="cpu") == "cpu"


def test_detect_device_falls_back_when_prefer_unavailable() -> None:
    # Asking for a nonsense device must fall back to auto-detect.
    d = detect_device(prefer="not-a-device")
    assert d in {"cuda", "mps", "cpu"}


def test_device_label_is_human_readable() -> None:
    assert "GPU" in device_label("cuda") or "NVIDIA" in device_label("cuda")
    assert "CPU" in device_label("cpu")
    assert "Apple" in device_label("mps") or "MPS" in device_label("mps")
