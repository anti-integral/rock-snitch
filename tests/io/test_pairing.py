from __future__ import annotations

from pathlib import Path

import numpy as np

from rocksnitch.contracts import ImageMeta, Subframe
from rocksnitch.io.pairing import find_pairs, save_index, load_index


def _mk(
    meta_template: ImageMeta,
    *,
    eye: str,
    sclk: float,
    az: float,
    el: float,
) -> ImageMeta:
    instr = "NAVCAM_LEFT" if eye == "L" else "NAVCAM_RIGHT"
    return ImageMeta(
        imageid=f"fake_{eye}_{sclk}",
        sol=meta_template.sol,
        site=meta_template.site,
        drive=meta_template.drive,
        instrument=instr,  # type: ignore[arg-type]
        sclk=sclk,
        mast_az_deg=az,
        mast_el_deg=el,
        attitude_quat_wxyz=meta_template.attitude_quat_wxyz,
        rover_xyz=meta_template.rover_xyz,
        subframe=meta_template.subframe,
        dimension=meta_template.dimension,
        scale_factor=meta_template.scale_factor,
        camera_model=meta_template.camera_model,
        image_path=meta_template.image_path,
        json_path=meta_template.json_path,
        filename=meta_template.filename,
    )


def test_find_pairs_matches_close_sclk(synthetic_image_meta: ImageMeta) -> None:
    left = _mk(synthetic_image_meta, eye="L", sclk=1000.0, az=78.0, el=1.0)
    right = _mk(synthetic_image_meta, eye="R", sclk=1000.5, az=78.0, el=1.0)
    far = _mk(synthetic_image_meta, eye="R", sclk=2000.0, az=78.0, el=1.0)
    pairs = find_pairs([left, right, far])
    assert len(pairs) == 1
    assert pairs[0].left is left
    assert pairs[0].right is right


def test_find_pairs_rejects_different_pointing(synthetic_image_meta: ImageMeta) -> None:
    left = _mk(synthetic_image_meta, eye="L", sclk=1000.0, az=78.0, el=1.0)
    right = _mk(synthetic_image_meta, eye="R", sclk=1000.1, az=90.0, el=1.0)
    pairs = find_pairs([left, right])
    assert pairs == []


def test_save_and_load_index(synthetic_image_meta: ImageMeta, tmp_path: Path) -> None:
    left = _mk(synthetic_image_meta, eye="L", sclk=1000.0, az=78.0, el=1.0)
    right = _mk(synthetic_image_meta, eye="R", sclk=1000.1, az=78.0, el=1.0)
    pairs = find_pairs([left, right])
    out = tmp_path / "idx.parquet"
    save_index(pairs, out)
    assert out.exists()
    rows = load_index(out)
    assert len(rows) == 1
    assert rows[0]["left_imageid"].startswith("fake_L")
