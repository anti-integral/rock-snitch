"""Gradio web UI for the rock-snitch-v2 Mars Navcam rock detector.

Single-file Blocks app: upload a Navcam left PNG + raw-images JSON sidecar
(or pick an example), configure detection thresholds, click "Detect rocks",
and see overlay, detections table, disparity preview, and downloadable JSON.

Backends are loaded lazily on the first detection click. Mock backends are
selected by default so the UI works on CPU-only machines without weights.

Run via ``rock-snitch ui`` or ``python -m rocksnitch.app``.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from rocksnitch.contracts import RockDetection, StereoPair
from rocksnitch.eval.viz import overlay_detections
from rocksnitch.io.metadata import load_meta, meta_from_json
from rocksnitch.io.pairing import find_pairs
from rocksnitch.logging_utils import get_logger

log = get_logger("rocksnitch.app")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
INDEX_PATH = DATA_ROOT / ".state" / "stereo_index.parquet"
RELEVANT_DIR = DATA_ROOT / "relevant"
METADATA_DIR = DATA_ROOT / "metadata"
IMAGES_DIR = DATA_ROOT / "images"

TITLE = "Rock-Snitch v2"
TAGLINE = (
    "Two-regime rock detection on Mars Perseverance Navcam imagery: "
    "stereo for the near-field, monocular depth + SAM2 for the far-field, "
    "fused into a single hazard list of rocks >=10 cm."
)
README_URL = "https://github.com/anti-integral/rock-snitch-v2#readme"

DEFAULT_MIN_HEIGHT_CM = 10.0
DEFAULT_MAX_RANGE_M = 60.0
DEFAULT_STEREO_TRUST_RANGE_M = 20.0

TABLE_HEADERS = [
    "id",
    "source",
    "bbox(x,y,w,h)",
    "range_m",
    "height_cm",
    "confidence",
]

# --------------------------------------------------------------------------- #
# Lazy backend cache
# --------------------------------------------------------------------------- #

# Module-level cache so the first click pays the construction cost once.
# Keyed by (mock_models, use_gpu) so flipping the toggle doesn't poison state.
_lazy_backends: dict[tuple[bool, bool], dict[str, Any]] = {}


def _get_backends(*, mock_models: bool, use_gpu: bool = False) -> dict[str, Any]:
    """Return cached perception backends, building them on first use."""
    key = (bool(mock_models), bool(use_gpu))
    if key not in _lazy_backends:
        from rocksnitch.cli import _build_backends
        from rocksnitch.profiles import get_profile

        log.info("Building backends (mock_models=%s, use_gpu=%s)", mock_models, use_gpu)
        profile = get_profile("minimal" if mock_models else "full")
        stereo, segmenter, depth, features = _build_backends(
            config={}, profile=profile, use_gpu=use_gpu
        )
        _lazy_backends[key] = {
            "stereo": stereo,
            "segmenter": segmenter,
            "depth": depth,
            "features": features,
        }
    return _lazy_backends[key]


# --------------------------------------------------------------------------- #
# Stereo-pair resolution from a single uploaded left frame
# --------------------------------------------------------------------------- #


def _read_rgb(path: Path) -> np.ndarray:
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image at {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _find_stereo_partner(left_meta_payload: dict, left_json_path: Path) -> StereoPair | None:
    """Try to locate the right-eye partner in the local data tree.

    Returns a StereoPair if found, otherwise None.
    """
    sol = int(left_meta_payload.get("sol", -1))
    if sol < 0:
        return None
    sol_dir = METADATA_DIR / f"{sol:05d}"
    if not sol_dir.exists():
        return None

    metas = []
    for jf in sorted(sol_dir.glob("*.json")):
        try:
            img_candidate = IMAGES_DIR / sol_dir.name / f"{jf.stem}.png"
            metas.append(load_meta(jf, image_path=img_candidate))
        except Exception:  # best-effort scan
            continue
    pairs = find_pairs(metas)
    left_imageid = left_meta_payload.get("imageid")
    for pair in pairs:
        if pair.left.imageid == left_imageid and pair.right.image_path.exists():
            return pair
    for pair in pairs:
        if pair.left.json_path == left_json_path and pair.right.image_path.exists():
            return pair
    return None


# --------------------------------------------------------------------------- #
# Detection orchestration
# --------------------------------------------------------------------------- #


def _detections_to_rows(detections: list[RockDetection]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for i, det in enumerate(detections):
        x, y, w, h = det.uv_bbox
        rows.append(
            [
                i,
                det.source,
                f"({x},{y},{w},{h})",
                round(float(det.range_m), 2),
                round(float(det.height_m) * 100.0, 1),
                round(float(det.confidence), 3),
            ]
        )
    return rows


def _disparity_to_png(disparity: np.ndarray) -> np.ndarray | None:
    """Render disparity as an RGB uint8 image suitable for gr.Image."""
    import cv2

    d = disparity.copy()
    mask = np.isfinite(d)
    if not mask.any():
        return None
    d[~mask] = 0.0
    rng = max(float(d.max() - d.min()), 1e-6)
    d = (d - d.min()) / rng
    vis = (d * 255).astype(np.uint8)
    color_bgr = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)


def _result_json_payload(detections: list[RockDetection]) -> dict:
    return {"detections": [asdict(d) for d in detections]}


def _write_json_temp(payload: dict, stem: str) -> str:
    tmp_dir = Path(tempfile.gettempdir()) / "rocksnitch-ui"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_path = tmp_dir / f"{stem}.detections.json"

    def _default(o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        return str(o)

    out_path.write_text(json.dumps(payload, indent=2, default=_default))
    return str(out_path)


def run_detection(
    left_image_path: str | None,
    json_file: str | None,
    mock_models: bool,
    min_height_cm: float,
    max_range_m: float,
    stereo_trust_range_m: float,
) -> tuple[
    np.ndarray | None,
    list[list[Any]],
    np.ndarray | None,
    str | None,
    str,
]:
    """Top-level Gradio click handler.

    Returns: (overlay_rgb, table_rows, disparity_rgb, json_filepath, log_text)
    """
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        return _run_detection_inner(
            left_image_path,
            json_file,
            mock_models=mock_models,
            min_height_cm=float(min_height_cm),
            max_range_m=float(max_range_m),
            stereo_trust_range_m=float(stereo_trust_range_m),
            log_buf=buf,
        )
    except Exception as exc:  # surface any pipeline failure to the UI
        log.exception("run_detection failed")
        return (None, [], None, None, f"{buf.getvalue()}\nERROR: {type(exc).__name__}: {exc}")
    finally:
        root.removeHandler(handler)


def _run_detection_inner(
    left_image_path: str | None,
    json_file: str | None,
    *,
    mock_models: bool,
    min_height_cm: float,
    max_range_m: float,
    stereo_trust_range_m: float,
    log_buf: io.StringIO,
) -> tuple[
    np.ndarray | None,
    list[list[Any]],
    np.ndarray | None,
    str | None,
    str,
]:
    if not left_image_path:
        return (None, [], None, None, "Upload a Navcam left PNG first.")

    left_path = Path(left_image_path)
    left_rgb = _read_rgb(left_path)

    json_payload: dict | None = None
    json_path: Path | None = None
    if json_file:
        json_path = Path(json_file)
        if json_path.exists():
            json_payload = json.loads(json_path.read_text())
    if json_payload is None:
        candidate = left_path.with_suffix(".json")
        if candidate.exists():
            json_path = candidate
            json_payload = json.loads(candidate.read_text())
        else:
            relevant_candidate = RELEVANT_DIR / f"{left_path.stem}.json"
            if relevant_candidate.exists():
                json_path = relevant_candidate
                json_payload = json.loads(relevant_candidate.read_text())
    if json_payload is None or json_path is None:
        return (
            None,
            [],
            None,
            None,
            "No JSON metadata supplied and none found alongside the image. "
            "Upload the matching raw-images JSON.",
        )

    pair = _find_stereo_partner(json_payload, json_path)

    backends = _get_backends(mock_models=mock_models, use_gpu=False)

    from rocksnitch.pipeline.far_field import FarFieldConfig
    from rocksnitch.pipeline.fuse import FusionConfig
    from rocksnitch.pipeline.near_field import NearFieldConfig
    from rocksnitch.pipeline.run import PipelineConfig

    cfg = PipelineConfig(
        near=NearFieldConfig(
            min_height_m=min_height_cm / 100.0,
            max_range_m=max_range_m,
        ),
        far=FarFieldConfig(
            min_height_m=min_height_cm / 100.0,
            max_range_m=max_range_m,
        ),
        fusion=FusionConfig(stereo_trust_range_m=stereo_trust_range_m),
    )

    detections: list[RockDetection]
    disparity_rgb: np.ndarray | None = None

    if pair is not None:
        log.info("Stereo partner located: %s", pair.right.imageid)
        right_rgb = _read_rgb(pair.right.image_path)
        from rocksnitch.pipeline.run import run_pipeline

        result = run_pipeline(
            pair,
            left_rgb,
            right_rgb,
            stereo=backends["stereo"],
            segmenter=backends["segmenter"],
            depth_estimator=backends["depth"],
            features=backends["features"],
            config=cfg,
        )
        detections = list(result.detections)
        if result.near_artefacts is not None:
            disparity_rgb = _disparity_to_png(result.near_artefacts.disparity.disparity)
    else:
        log.info("No stereo partner — running far-field branch only.")
        from rocksnitch.io.cahvore import intrinsics_matrix, linearize_to_cahv
        from rocksnitch.pipeline.far_field import run_far_field

        left_meta = meta_from_json(json_payload, image_path=left_path, json_path=json_path)
        K = intrinsics_matrix(linearize_to_cahv(left_meta.camera_model))
        detections = run_far_field(
            left_rgb,
            segmenter=backends["segmenter"],
            depth_estimator=backends["depth"],
            features=backends["features"],
            K=K,
            config=cfg.far,
        )

    overlay = overlay_detections(left_rgb, detections)
    rows = _detections_to_rows(detections)
    payload = _result_json_payload(detections)
    json_out = _write_json_temp(payload, stem=left_path.stem)

    log.info("Detections: %d (overlay shape=%s)", len(detections), overlay.shape)
    return overlay, rows, disparity_rgb, json_out, log_buf.getvalue()


# --------------------------------------------------------------------------- #
# Examples
# --------------------------------------------------------------------------- #


def _example_rows(limit: int = 5) -> list[list[str]]:
    """Find a few <png, json> pairs we can use as one-click demos."""
    rows: list[list[str]] = []
    if RELEVANT_DIR.exists():
        for png in sorted(RELEVANT_DIR.glob("*.png"))[:limit]:
            js = png.with_suffix(".json")
            if js.exists():
                rows.append([str(png), str(js)])
    if rows:
        return rows
    for sol_dir_name in ("00100", "00050"):
        meta_sol_dir = METADATA_DIR / sol_dir_name
        img_sol_dir = IMAGES_DIR / sol_dir_name
        if not (meta_sol_dir.exists() and img_sol_dir.exists()):
            continue
        for jf in sorted(meta_sol_dir.glob("NLF_*.json"))[:limit]:
            png = img_sol_dir / f"{jf.stem}.png"
            if png.exists():
                rows.append([str(png), str(jf)])
        if rows:
            break
    return rows


# --------------------------------------------------------------------------- #
# Blocks construction
# --------------------------------------------------------------------------- #


def build_app() -> Any:
    """Build and return the Gradio Blocks app. No server is launched."""
    import gradio as gr

    examples = _example_rows()

    with gr.Blocks(title=TITLE, theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            f"# {TITLE}\n"
            f"{TAGLINE}\n\n"
            f"[README]({README_URL}) — green = stereo, orange = mono, "
            "yellow = fused"
        )

        with gr.Row():
            with gr.Column(scale=1):
                left_image = gr.Image(
                    type="filepath",
                    label="Navcam left frame (.png)",
                    height=320,
                )
                json_file = gr.File(
                    file_types=[".json"],
                    label="Raw-images JSON sidecar (.json)",
                )
                if examples:
                    gr.Examples(
                        examples=examples,
                        inputs=[left_image, json_file],
                        label="Example pairs (click to load)",
                        examples_per_page=8,
                    )

                with gr.Accordion("Detection thresholds", open=True):
                    min_height_cm = gr.Slider(
                        minimum=1.0,
                        maximum=50.0,
                        value=DEFAULT_MIN_HEIGHT_CM,
                        step=1.0,
                        label="min height [cm]",
                    )
                    max_range_m = gr.Slider(
                        minimum=5.0,
                        maximum=120.0,
                        value=DEFAULT_MAX_RANGE_M,
                        step=1.0,
                        label="max range [m]",
                    )
                    stereo_trust_range_m = gr.Slider(
                        minimum=2.0,
                        maximum=40.0,
                        value=DEFAULT_STEREO_TRUST_RANGE_M,
                        step=1.0,
                        label="stereo-trust range [m]",
                    )
                    mock_toggle = gr.Checkbox(
                        value=True,
                        label="Use mock models (CPU, no weights)",
                    )

                run_btn = gr.Button("Detect rocks", variant="primary")

            with gr.Column(scale=2):
                overlay_out = gr.Image(
                    label="Detections overlay",
                    height=420,
                    show_download_button=True,
                )
                with gr.Row():
                    table_out = gr.Dataframe(
                        headers=TABLE_HEADERS,
                        label="Detections",
                        wrap=True,
                        interactive=False,
                    )
                with gr.Row():
                    disparity_out = gr.Image(
                        label="Stereo disparity (near-field)",
                        height=240,
                    )
                    json_out = gr.File(label="detections.json")
                log_out = gr.Textbox(
                    label="Pipeline log",
                    lines=8,
                    max_lines=20,
                    interactive=False,
                )

        run_btn.click(
            fn=run_detection,
            inputs=[
                left_image,
                json_file,
                mock_toggle,
                min_height_cm,
                max_range_m,
                stereo_trust_range_m,
            ],
            outputs=[overlay_out, table_out, disparity_out, json_out, log_out],
        )

    return demo


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rock-snitch ui",
        description="Launch the rock-snitch Gradio web UI.",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    demo = build_app()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=bool(args.share),
        show_error=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
