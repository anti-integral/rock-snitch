"""Mars 2020 Perseverance Navcam filename decoder.

Format (variable length, underscores inside the venue block):
    NL[FEBGRM] _ NNNN _ SSSSSSSSSS _ MMM XXX _ <venue> _ FF _ <ds>

where
  - NL[FEBGRM] : instrument + eye + product kind (3 chars)
  - NNNN       : sol (4 digits)
  - SSSSSSSSSS : SCLK seconds (10 digits)
  - MMM        : SCLK milliseconds (3 digits)
  - XXX        : product-type code, 3 chars (directly concatenated after MMM, no underscore)
  - venue      : 1+ underscore-separated tokens
  - FF         : 2-char camera flags
  - ds         : 3-5 char downsample/version code (alphanumeric)

Parsing is anchored on fixed-width prefix segments and then peels the last two
tokens off the tail to robustly handle venues that contain additional
underscores (as seen in real Sol 2 Navcam filenames).
"""
from __future__ import annotations

import re
from typing import cast

from rocksnitch.contracts import Eye, FilenameParts, ProductKind


_VALID_KINDS: set[str] = {"F", "E", "B", "G", "R", "M"}
_SUFFIX_RE = re.compile(r"\.(png|jpg|jpeg|json)$", re.IGNORECASE)
_HEAD_RE = re.compile(
    r"^N(?P<eye>[LR])(?P<kind>[A-Z])_"
    r"(?P<sol>\d{4})_"
    r"(?P<sclk>\d{10})_"
    r"(?P<msec>\d{3})"
    r"(?P<prod>[A-Z]{3})_"
)


def _strip_suffix(name: str) -> str:
    return _SUFFIX_RE.sub("", name.strip())


def parse(name: str) -> FilenameParts:
    """Decode a Mars 2020 Navcam filename or image ID."""
    stripped = _strip_suffix(name)
    m = _HEAD_RE.match(stripped)
    if m is None:
        raise ValueError(f"Unrecognized Navcam filename: {name!r}")
    tail = stripped[m.end():]
    tokens = tail.split("_")
    if len(tokens) < 3:
        raise ValueError(f"Tail of {name!r} lacks venue/flags/ds ({tokens})")
    ds = tokens[-1]
    flags = tokens[-2]
    venue = "_".join(tokens[:-2])
    if not re.fullmatch(r"[A-Z0-9]{2}", flags):
        raise ValueError(f"Bad flags field in {name!r}: {flags!r}")
    if not re.fullmatch(r"[A-Z0-9]{3,8}", ds):
        raise ValueError(f"Bad downsample code in {name!r}: {ds!r}")
    if not venue:
        raise ValueError(f"Empty venue in {name!r}")

    kind_raw = m.group("kind")
    kind = cast(ProductKind, kind_raw if kind_raw in _VALID_KINDS else "UNKNOWN")

    return FilenameParts(
        instrument="N",
        eye=cast(Eye, m.group("eye")),
        kind=kind,
        sol=int(m.group("sol")),
        sclk_sec=int(m.group("sclk")),
        sclk_msec=int(m.group("msec")),
        product_code=m.group("prod"),
        venue=venue,
        camera_flags=flags,
        downsample_code=ds,
        raw=stripped,
    )


def is_navcam(name: str) -> bool:
    """True if ``name`` matches the Navcam ID pattern."""
    try:
        parse(name)
    except ValueError:
        return False
    return True


def is_stereo_compatible(a: FilenameParts, b: FilenameParts) -> bool:
    """Cheap pre-filter: same sol + product kind + venue, different eyes."""
    return (
        a.eye != b.eye
        and a.sol == b.sol
        and a.kind == b.kind
        and a.venue == b.venue
        and a.product_code == b.product_code
    )


def sclk_float(parts: FilenameParts) -> float:
    """Return SCLK as seconds (with millisecond fraction)."""
    return parts.sclk_sec + parts.sclk_msec / 1000.0


__all__ = ["parse", "is_navcam", "is_stereo_compatible", "sclk_float"]
