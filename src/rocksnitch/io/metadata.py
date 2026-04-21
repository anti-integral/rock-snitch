"""Parse Mars 2020 raw-images JSON sidecars into :class:`ImageMeta`."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from rocksnitch.contracts import ImageMeta, Instrument, Subframe
from rocksnitch.io.cahvore import parse_component_list
from rocksnitch.io.filename import parse as parse_filename


_TUPLE_RE = re.compile(r"\(\s*([^)]+?)\s*\)")


def _parse_tuple(s: str) -> list[float]:
    m = _TUPLE_RE.fullmatch(s.strip())
    if m is None:
        raise ValueError(f"Expected tuple-in-parens, got {s!r}")
    return [float(x.strip()) for x in m.group(1).split(",")]


def _parse_subframe(s: str) -> Subframe:
    parts = _parse_tuple(s)
    if len(parts) != 4:
        raise ValueError(f"Subframe needs 4 ints, got {parts}")
    return Subframe(int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))


def _parse_dim(s: str) -> tuple[int, int]:
    parts = _parse_tuple(s)
    if len(parts) != 2:
        raise ValueError(f"Dimension needs 2 ints, got {parts}")
    return int(parts[0]), int(parts[1])


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s or s.upper() == "UNK":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def meta_from_json(
    payload: dict[str, Any],
    image_path: Path,
    json_path: Path,
) -> ImageMeta:
    """Construct an :class:`ImageMeta` from a raw-images JSON dict."""
    imageid = payload["imageid"]
    sol = int(payload["sol"])
    site = int(payload["site"])
    drive = str(payload["drive"])

    ext = payload.get("extended", {}) or {}
    sclk = _optional_float(ext.get("sclk")) or 0.0
    mast_az = _optional_float(ext.get("mastAz"))
    mast_el = _optional_float(ext.get("mastEl"))
    xyz = _parse_tuple(ext.get("xyz", "(0,0,0)"))
    subframe = _parse_subframe(ext.get("subframeRect", "(1,1,1280,960)"))
    dimension = _parse_dim(ext.get("dimension", "(1280,960)"))
    scale_factor = int(ext.get("scaleFactor", "1"))

    attitude = _parse_tuple(payload.get("attitude", "(1,0,0,0)"))
    if len(attitude) != 4:
        raise ValueError(f"attitude must be quaternion with 4 floats, got {attitude}")

    camera_info = payload.get("camera", {}) or {}
    instrument: Instrument = camera_info.get("instrument", "NAVCAM_LEFT")  # type: ignore[assignment]
    model_blob = camera_info.get("camera_model_component_list", "")
    # Image size note: Mars2020 dimension is (W, H) order in the JSON
    image_size = (dimension[1], dimension[0])  # -> (H, W)
    if model_blob:
        camera_model = parse_component_list(model_blob, image_size=image_size)
    else:
        raise ValueError(f"{imageid}: missing camera_model_component_list")

    return ImageMeta(
        imageid=imageid,
        sol=sol,
        site=site,
        drive=drive,
        instrument=instrument,
        sclk=sclk,
        mast_az_deg=mast_az,
        mast_el_deg=mast_el,
        attitude_quat_wxyz=np.asarray(attitude, dtype=np.float64),
        rover_xyz=np.asarray(xyz, dtype=np.float64),
        subframe=subframe,
        dimension=dimension,
        scale_factor=scale_factor,
        camera_model=camera_model,
        image_path=image_path,
        json_path=json_path,
        filename=parse_filename(imageid),
        caption=payload.get("caption", ""),
        date_taken_utc=payload.get("date_taken_utc"),
    )


def load_meta(json_path: Path, image_path: Path | None = None) -> ImageMeta:
    """Load and parse a raw-images JSON from disk."""
    payload = json.loads(Path(json_path).read_text())
    if image_path is None:
        stem = Path(json_path).stem
        image_path = Path(json_path).with_name(f"{stem}.png")
    return meta_from_json(payload, image_path=Path(image_path), json_path=Path(json_path))


def iter_meta(data_root: Path) -> list[ImageMeta]:
    """Walk ``data_root / metadata`` and yield every parseable ImageMeta."""
    meta_dir = Path(data_root) / "metadata"
    img_dir = Path(data_root) / "images"
    out: list[ImageMeta] = []
    if not meta_dir.exists():
        return out
    for sol_dir in sorted(meta_dir.iterdir()):
        if not sol_dir.is_dir():
            continue
        for jf in sorted(sol_dir.glob("*.json")):
            try:
                img_candidate = img_dir / sol_dir.name / f"{jf.stem}.png"
                m = load_meta(jf, image_path=img_candidate)
            except Exception:
                continue
            out.append(m)
    return out


__all__ = ["meta_from_json", "load_meta", "iter_meta"]
