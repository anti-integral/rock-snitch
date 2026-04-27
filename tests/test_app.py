"""Tests for the Gradio web UI in :mod:`rocksnitch.app`."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from rocksnitch.contracts import StereoPair

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("gradio") is None,
    reason="gradio not installed (install with `pip install -e '.[ui]'`)",
)


@pytest.fixture(autouse=True)
def _reset_backend_cache():
    """Ensure each test gets a fresh lazy-backend cache."""
    from rocksnitch import app

    app._lazy_backends.clear()
    yield
    app._lazy_backends.clear()


def _write_synthetic_pair_to_disk(pair: StereoPair, tmp_path: Path) -> tuple[Path, Path]:
    """Write a small textured RGB pair + JSON sidecar to disk for the UI."""
    H, W = pair.left.camera_model.image_size
    rng = np.random.default_rng(0)
    left_rgb = rng.integers(40, 215, size=(H, W, 3), dtype=np.uint8)

    left_png = tmp_path / "L.png"
    cv2.imwrite(str(left_png), cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR))

    sidecar = {
        "imageid": pair.left.imageid,
        "sol": pair.left.sol,
        "site": pair.left.site,
        "drive": pair.left.drive,
        "attitude": "(1,0,0,0)",
        "extended": {
            "mastAz": str(pair.left.mast_az_deg),
            "mastEl": str(pair.left.mast_el_deg),
            "sclk": str(pair.left.sclk),
            "scaleFactor": "1",
            "xyz": "(0,0,0)",
            "subframeRect": f"(1,1,{W},{H})",
            "dimension": f"({W},{H})",
        },
        "camera": {
            "instrument": "NAVCAM_LEFT",
            "camera_model_type": "CAHV",
            "camera_model_component_list": "",
        },
    }
    json_path = tmp_path / "L.json"
    json_path.write_text(json.dumps(sidecar))
    return left_png, json_path


def test_build_app_returns_blocks() -> None:
    import gradio as gr

    from rocksnitch.app import build_app

    demo = build_app()
    assert isinstance(demo, gr.Blocks)


def test_run_detection_with_missing_inputs_returns_friendly_message() -> None:
    from rocksnitch.app import run_detection

    overlay, rows, disparity, json_out, log_text = run_detection(
        left_image_path=None,
        json_file=None,
        mock_models=True,
        min_height_cm=10.0,
        max_range_m=60.0,
        stereo_trust_range_m=20.0,
    )
    assert overlay is None
    assert rows == []
    assert disparity is None
    assert json_out is None
    assert "Upload" in log_text or "upload" in log_text


def test_run_detection_far_field_with_synthetic_pair(
    synthetic_stereo_pair: StereoPair, tmp_path: Path
) -> None:
    """End-to-end smoke test using the synthetic fixture + mock backends.

    We force the "no stereo partner" branch by patching :func:`_find_stereo_partner`
    and short-circuit metadata parsing so the test does not depend on the real
    data tree.
    """
    from rocksnitch import app
    from rocksnitch.app import run_detection

    left_png, json_path = _write_synthetic_pair_to_disk(synthetic_stereo_pair, tmp_path)

    original_finder = app._find_stereo_partner
    original_meta = app.meta_from_json
    app._find_stereo_partner = lambda *_a, **_k: None  # type: ignore[assignment]
    app.meta_from_json = lambda *_a, **_k: synthetic_stereo_pair.left  # type: ignore[assignment]

    try:
        overlay, rows, _disparity, json_out, log_text = run_detection(
            left_image_path=str(left_png),
            json_file=str(json_path),
            mock_models=True,
            min_height_cm=10.0,
            max_range_m=60.0,
            stereo_trust_range_m=20.0,
        )
    finally:
        app._find_stereo_partner = original_finder  # type: ignore[assignment]
        app.meta_from_json = original_meta  # type: ignore[assignment]

    assert overlay is not None, log_text
    assert overlay.dtype == np.uint8
    assert overlay.ndim == 3 and overlay.shape[2] == 3
    assert isinstance(rows, list)
    assert json_out is not None
    payload = json.loads(Path(json_out).read_text())
    assert "detections" in payload
    assert isinstance(payload["detections"], list)


def test_run_detection_with_stereo_partner(
    synthetic_stereo_pair: StereoPair, tmp_path: Path
) -> None:
    """Drive the stereo branch by patching :func:`_find_stereo_partner`."""
    from rocksnitch import app
    from rocksnitch.app import run_detection

    left_png, json_path = _write_synthetic_pair_to_disk(synthetic_stereo_pair, tmp_path)

    H, W = synthetic_stereo_pair.right.camera_model.image_size
    right_img = np.full((H, W, 3), 80, dtype=np.uint8)
    cv2.imwrite(
        str(synthetic_stereo_pair.right.image_path),
        cv2.cvtColor(right_img, cv2.COLOR_RGB2BGR),
    )

    original_finder = app._find_stereo_partner
    app._find_stereo_partner = lambda *_a, **_k: synthetic_stereo_pair  # type: ignore[assignment]
    try:
        overlay, rows, _disparity, json_out, log_text = run_detection(
            left_image_path=str(left_png),
            json_file=str(json_path),
            mock_models=True,
            min_height_cm=10.0,
            max_range_m=60.0,
            stereo_trust_range_m=20.0,
        )
    finally:
        app._find_stereo_partner = original_finder  # type: ignore[assignment]

    assert overlay is not None, log_text
    assert isinstance(rows, list)
    assert json_out is not None
