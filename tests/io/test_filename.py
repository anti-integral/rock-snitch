from __future__ import annotations

import pytest

from rocksnitch.io.filename import is_navcam, is_stereo_compatible, parse, sclk_float


def test_parse_nlf_sol100() -> None:
    parts = parse("NLF_0100_0675828717_276ECM_N0040218NCAM00503_01_295J")
    assert parts.instrument == "N"
    assert parts.eye == "L"
    assert parts.kind == "F"
    assert parts.sol == 100
    assert parts.sclk_sec == 675828717
    assert parts.sclk_msec == 276
    assert parts.product_code == "ECM"
    assert parts.venue == "N0040218NCAM00503"
    assert parts.camera_flags == "01"
    assert parts.downsample_code == "295J"


def test_parse_with_png_suffix() -> None:
    parts = parse("NRF_0018_0668555532_151ECM_N0030578NCAM00188_01_290J.png")
    assert parts.eye == "R"
    assert parts.kind == "F"
    assert parts.sol == 18


def test_parse_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        parse("not-a-valid-name")


def test_is_navcam() -> None:
    assert is_navcam("NLF_0100_0675828717_276ECM_N0040218NCAM00503_01_295J")
    assert not is_navcam("random_file.png")


def test_stereo_compat_same_sol_venue_opposite_eye() -> None:
    left = parse("NLF_0100_0675828717_276ECM_N0040218NCAM00503_01_295J")
    right = parse("NRF_0100_0675828717_276ECM_N0040218NCAM00503_01_295J")
    assert is_stereo_compatible(left, right)


def test_stereo_incompat_different_sol() -> None:
    left = parse("NLF_0100_0675828717_276ECM_N0040218NCAM00503_01_295J")
    right = parse("NRF_0101_0675828717_276ECM_N0040218NCAM00503_01_295J")
    assert not is_stereo_compatible(left, right)


def test_sclk_float_has_ms_fraction() -> None:
    parts = parse("NLF_0100_0675828717_276ECM_N0040218NCAM00503_01_295J")
    assert sclk_float(parts) == pytest.approx(675828717.276, rel=1e-12)


@pytest.mark.parametrize(
    "raw,expected_kind",
    [
        ("NLF_0002_0667129618_444EBY_N0010052AUT_04096_00_0LLJ", "F"),
        ("NLE_0002_0667130166_842ECM_N0010052AUT_04096_00_0LLJ01", "E"),
        ("NLG_0001_0667035548_700ECM_N0010052AUT_04096_00_2I3J01", "G"),
    ],
)
def test_various_kinds(raw: str, expected_kind: str) -> None:
    parts = parse(raw)
    assert parts.kind == expected_kind
