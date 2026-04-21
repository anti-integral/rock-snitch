from __future__ import annotations

import numpy as np
import pytest

from rocksnitch.contracts import CameraModel
from rocksnitch.io.cahvore import (
    focal_lengths,
    intrinsics_matrix,
    linearize_to_cahv,
    parse_component_list,
    principal_point,
    project_cahv,
    unproject_cahv,
)


def test_parse_component_list_roundtrip() -> None:
    blob = (
        "(1.0,2.0,3.0);(0.0,1.0,0.0);(100.0,0.0,50.0);(0.0,100.0,30.0);"
        "(0.0,1.0,0.0);(0.1,0.2,0.3);(0.01,0.02,0.03);1.0;0.0;0.0"
    )
    cam = parse_component_list(blob, image_size=(60, 100))
    assert np.allclose(cam.C, [1.0, 2.0, 3.0])
    assert cam.linearity == 1.0
    assert cam.image_size == (60, 100)


def test_principal_point_and_focal(synthetic_left_cam: CameraModel) -> None:
    cx, cy = principal_point(synthetic_left_cam)
    fx, fy = focal_lengths(synthetic_left_cam)
    assert cx == pytest.approx(640.0)
    assert cy == pytest.approx(480.0)
    assert fx == pytest.approx(576.0, rel=1e-6)
    assert fy == pytest.approx(576.0, rel=1e-6)


def test_project_unproject_roundtrip(synthetic_left_cam: CameraModel) -> None:
    pts = np.array([[0.0, 10.0, 0.0], [1.0, 20.0, -1.0], [-0.5, 5.0, 0.5]])
    uv = project_cahv(synthetic_left_cam, pts)
    rays = unproject_cahv(synthetic_left_cam, uv)
    for ray, p in zip(rays, pts - synthetic_left_cam.C):
        p_hat = p / np.linalg.norm(p)
        assert np.allclose(ray, p_hat, atol=1e-6)


def test_linearize_clears_distortion(synthetic_left_cam: CameraModel) -> None:
    lin = linearize_to_cahv(synthetic_left_cam)
    assert np.allclose(lin.R, 0)
    assert np.allclose(lin.E, 0)
    assert lin.linearity == 1.0


def test_intrinsics_matrix(synthetic_left_cam: CameraModel) -> None:
    K = intrinsics_matrix(synthetic_left_cam)
    assert K.shape == (3, 3)
    assert K[2, 2] == 1.0
    assert K[0, 0] == pytest.approx(576.0, rel=1e-6)


def test_project_point_at_center_yields_principal_point(
    synthetic_left_cam: CameraModel,
) -> None:
    forward = synthetic_left_cam.C + synthetic_left_cam.A * 10.0
    uv = project_cahv(synthetic_left_cam, forward)
    cx, cy = principal_point(synthetic_left_cam)
    assert np.allclose(uv[0], [cx, cy], atol=1e-6)
