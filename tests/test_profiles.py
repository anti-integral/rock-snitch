"""Unit tests for the profile system."""
from __future__ import annotations

import pytest

from rocksnitch.profiles import get_profile, list_profiles


def test_list_profiles_includes_all() -> None:
    names = list_profiles()
    for expected in ("full", "stereo-only", "mono-only", "minimal", "mock"):
        assert expected in names, f"missing {expected!r}"


def test_get_full_profile_enables_everything() -> None:
    p = get_profile("full")
    assert p.pipeline.enable_stereo
    assert p.pipeline.enable_mono
    assert p.pipeline.enable_linearize
    assert p.segmenter == "sam2"
    assert p.depth == "unidepth"
    assert p.features == "dinov2"


def test_get_stereo_only_profile_disables_mono() -> None:
    p = get_profile("stereo-only")
    assert p.pipeline.enable_stereo
    assert not p.pipeline.enable_mono
    assert p.depth == "none"
    assert p.features == "none"


def test_get_mono_only_profile_disables_stereo() -> None:
    p = get_profile("mono-only")
    assert not p.pipeline.enable_stereo
    assert p.pipeline.enable_mono
    assert p.stereo == "none"


def test_get_minimal_profile_uses_mocks() -> None:
    p = get_profile("minimal")
    assert p.segmenter == "mock"
    assert p.depth == "mock"
    assert p.features == "mock"


def test_mock_is_alias_of_minimal() -> None:
    a = get_profile("minimal")
    b = get_profile("mock")
    assert a.segmenter == b.segmenter
    assert a.depth == b.depth


def test_unknown_profile_raises() -> None:
    with pytest.raises(KeyError, match="Unknown profile"):
        get_profile("does-not-exist")


def test_explain_returns_concise_summary() -> None:
    p = get_profile("full")
    s = p.explain()
    assert "profile=full" in s
    assert "stereo=" in s
    assert "mono=" in s


def test_pipeline_config_defaults_are_safe() -> None:
    """Default PipelineConfig should not crash with missing optional inputs."""
    from rocksnitch.pipeline.run import PipelineConfig

    cfg = PipelineConfig()
    assert cfg.enable_stereo is True
    assert cfg.enable_mono is True
    assert cfg.enable_linearize is False  # opt-in
