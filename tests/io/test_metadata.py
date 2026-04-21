from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rocksnitch.io.metadata import meta_from_json, load_meta


def test_meta_from_sample(sample_real_json: dict, tmp_path: Path) -> None:
    img = tmp_path / "x.png"
    img.write_bytes(b"")
    jp = tmp_path / "x.json"
    jp.write_text(json.dumps(sample_real_json))
    m = meta_from_json(sample_real_json, image_path=img, json_path=jp)
    assert m.sol == 100
    assert m.site == 4
    assert m.instrument == "NAVCAM_LEFT"
    assert m.subframe.w == 5120
    assert m.dimension == (1280, 240)
    assert m.camera_model.image_size == (240, 1280)
    # CAHVORE Cx/Cy/Cz match the JSON
    assert np.allclose(m.camera_model.C, [1.03927, 0.645857, -1.9785])
    assert m.mast_az_deg == pytest.approx(78.7995)
    assert m.rover_xyz.shape == (3,)


def test_meta_raises_without_camera_model(sample_real_json: dict, tmp_path: Path) -> None:
    bad = dict(sample_real_json)
    bad["camera"] = dict(bad["camera"])
    bad["camera"]["camera_model_component_list"] = ""
    with pytest.raises(ValueError):
        meta_from_json(bad, image_path=tmp_path / "x.png", json_path=tmp_path / "x.json")


def test_load_meta_roundtrip(sample_real_json: dict, tmp_path: Path) -> None:
    jp = tmp_path / "x.json"
    jp.write_text(json.dumps(sample_real_json))
    (tmp_path / "x.png").write_bytes(b"")
    m = load_meta(jp)
    assert m.imageid.startswith("NLF_0100")
