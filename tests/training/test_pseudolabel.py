from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from rocksnitch.training.pseudolabel import PseudolabelRecord, write_pseudolabels, read_pseudolabels


def test_write_read_roundtrip(tmp_path: Path) -> None:
    recs = [
        PseudolabelRecord(
            left_imageid="A",
            right_imageid="B",
            sol=100,
            uv_bbox=[0, 0, 10, 10],
            centroid_uv=[5.0, 5.0],
            range_m=12.5,
            height_m=0.15,
            width_m=0.2,
            confidence=0.8,
            feature=[0.1, 0.2, 0.3],
        )
    ]
    p = tmp_path / "labels.jsonl"
    write_pseudolabels(recs, p)
    back = read_pseudolabels(p)
    assert back == recs
