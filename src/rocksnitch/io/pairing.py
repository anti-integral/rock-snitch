"""Index Mars2020 Navcam stereo pairs.

A stereo pair is two :class:`ImageMeta` with:
  * opposite eyes (NAVCAM_LEFT <-> NAVCAM_RIGHT)
  * same sol
  * SCLK within ``max_sclk_delta_s`` (default 5 s)
  * matching mast pointing within ``max_pointing_delta_deg`` (default 0.5 deg)
  * identical subframe rectangle
"""
from __future__ import annotations

import bisect
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from rocksnitch.contracts import ImageMeta, StereoPair
from rocksnitch.io.metadata import iter_meta


def find_pairs(
    metas: Iterable[ImageMeta],
    max_sclk_delta_s: float = 5.0,
    max_pointing_delta_deg: float = 0.5,
    require_same_subframe: bool = True,
) -> list[StereoPair]:
    """Return all confirmed stereo pairs from a flat list of ImageMeta."""
    by_sol: dict[int, list[ImageMeta]] = {}
    for m in metas:
        by_sol.setdefault(m.sol, []).append(m)

    pairs: list[StereoPair] = []
    for sol_items in by_sol.values():
        lefts = [m for m in sol_items if m.instrument == "NAVCAM_LEFT"]
        rights = sorted(
            (m for m in sol_items if m.instrument == "NAVCAM_RIGHT"),
            key=lambda m: m.sclk,
        )
        rsclk = [m.sclk for m in rights]
        for left in lefts:
            if left.mast_az_deg is None or left.mast_el_deg is None:
                continue
            idx = bisect.bisect_left(rsclk, left.sclk)
            candidates = rights[max(0, idx - 1) : idx + 2]
            best: StereoPair | None = None
            best_dt = float("inf")
            for r in candidates:
                if r.mast_az_deg is None or r.mast_el_deg is None:
                    continue
                dt = abs(r.sclk - left.sclk)
                if dt > max_sclk_delta_s:
                    continue
                daz = abs(r.mast_az_deg - left.mast_az_deg)
                dal = abs(r.mast_el_deg - left.mast_el_deg)
                if daz > max_pointing_delta_deg or dal > max_pointing_delta_deg:
                    continue
                if require_same_subframe and left.subframe.as_tuple() != r.subframe.as_tuple():
                    continue
                if dt < best_dt:
                    best_dt = dt
                    best = StereoPair(
                        left=left,
                        right=r,
                        sclk_delta_s=r.sclk - left.sclk,
                        mast_az_delta_deg=r.mast_az_deg - left.mast_az_deg,
                        mast_el_delta_deg=r.mast_el_deg - left.mast_el_deg,
                    )
            if best is not None:
                pairs.append(best)
    return pairs


def index_dataset(
    data_root: Path,
    max_sclk_delta_s: float = 5.0,
    max_pointing_delta_deg: float = 0.5,
) -> list[StereoPair]:
    """Walk ``data_root`` and build the stereo-pair list."""
    metas = iter_meta(Path(data_root))
    return find_pairs(
        metas,
        max_sclk_delta_s=max_sclk_delta_s,
        max_pointing_delta_deg=max_pointing_delta_deg,
    )


# ---------------------------------------------------------------------------
# Parquet persistence
# ---------------------------------------------------------------------------


_COLS = [
    "sol", "site", "drive",
    "left_imageid", "right_imageid",
    "left_image_path", "right_image_path",
    "left_json_path", "right_json_path",
    "sclk_left", "sclk_right",
    "mast_az_left", "mast_el_left",
    "mast_az_right", "mast_el_right",
    "sclk_delta_s",
    "mast_az_delta_deg", "mast_el_delta_deg",
]


def pairs_to_table(pairs: list[StereoPair]) -> pa.Table:
    """Materialize pairs as a pyarrow Table suitable for parquet."""
    rows = [
        {
            "sol": p.left.sol,
            "site": p.left.site,
            "drive": p.left.drive,
            "left_imageid": p.left.imageid,
            "right_imageid": p.right.imageid,
            "left_image_path": str(p.left.image_path),
            "right_image_path": str(p.right.image_path),
            "left_json_path": str(p.left.json_path),
            "right_json_path": str(p.right.json_path),
            "sclk_left": p.left.sclk,
            "sclk_right": p.right.sclk,
            "mast_az_left": p.left.mast_az_deg or 0.0,
            "mast_el_left": p.left.mast_el_deg or 0.0,
            "mast_az_right": p.right.mast_az_deg or 0.0,
            "mast_el_right": p.right.mast_el_deg or 0.0,
            "sclk_delta_s": p.sclk_delta_s,
            "mast_az_delta_deg": p.mast_az_delta_deg,
            "mast_el_delta_deg": p.mast_el_delta_deg,
        }
        for p in pairs
    ]
    return pa.Table.from_pylist(rows, schema=_schema())


def _schema() -> pa.Schema:
    return pa.schema([
        ("sol", pa.int32()),
        ("site", pa.int32()),
        ("drive", pa.string()),
        ("left_imageid", pa.string()),
        ("right_imageid", pa.string()),
        ("left_image_path", pa.string()),
        ("right_image_path", pa.string()),
        ("left_json_path", pa.string()),
        ("right_json_path", pa.string()),
        ("sclk_left", pa.float64()),
        ("sclk_right", pa.float64()),
        ("mast_az_left", pa.float64()),
        ("mast_el_left", pa.float64()),
        ("mast_az_right", pa.float64()),
        ("mast_el_right", pa.float64()),
        ("sclk_delta_s", pa.float64()),
        ("mast_az_delta_deg", pa.float64()),
        ("mast_el_delta_deg", pa.float64()),
    ])


def save_index(pairs: list[StereoPair], out_path: Path) -> None:
    """Persist pairs to a parquet file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pairs_to_table(pairs), out_path)


def load_index(path: Path) -> list[dict]:
    """Load an existing stereo-pair parquet into plain dicts (no metadata re-parse)."""
    table = pq.read_table(Path(path))
    return table.to_pylist()


__all__ = [
    "find_pairs",
    "index_dataset",
    "pairs_to_table",
    "save_index",
    "load_index",
]

# keep asdict imported for downstream stable sorting utilities
_ = asdict  # pragma: no cover
