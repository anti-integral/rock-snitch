"""Visualise detections on an image."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from rocksnitch.contracts import RockDetection, UInt8Array


_COLOR_BY_SOURCE = {
    "stereo": (0, 255, 0),
    "mono": (255, 128, 0),
    "fused": (255, 255, 0),
}


def overlay_detections(
    image: UInt8Array,
    detections: list[RockDetection],
    *,
    show_height: bool = True,
    show_range: bool = True,
) -> UInt8Array:
    """Return RGB image with bbox + height/range annotations for each detection."""
    out = image.copy()
    for det in detections:
        x, y, w, h = det.uv_bbox
        color = _COLOR_BY_SOURCE.get(det.source, (255, 255, 255))
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        label = f"{det.source}"
        if show_height:
            label += f" h={det.height_m * 100:.0f}cm"
        if show_range:
            label += f" r={det.range_m:.1f}m"
        cv2.putText(
            out,
            label,
            (x, max(0, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            lineType=cv2.LINE_AA,
        )
    return out


def write_overlay(
    image: UInt8Array, detections: list[RockDetection], out_path: Path
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay = overlay_detections(image, detections)
    bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR) if overlay.ndim == 3 else overlay
    cv2.imwrite(str(out_path), bgr)


def write_disparity_preview(disparity: np.ndarray, out_path: Path) -> None:
    d = disparity.copy()
    mask = np.isfinite(d)
    if not mask.any():
        return
    d[~mask] = 0.0
    d = (d - d.min()) / max(d.max() - d.min(), 1e-6)
    vis = (d * 255).astype(np.uint8)
    vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis_color)


__all__ = ["overlay_detections", "write_overlay", "write_disparity_preview"]
