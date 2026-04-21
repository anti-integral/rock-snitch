"""Shared pytest fixtures."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from rocksnitch.contracts import (
    CameraModel,
    FilenameParts,
    ImageMeta,
    StereoPair,
    Subframe,
)


def _has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    gpu_available = _has_cuda()
    have_models = os.environ.get("ROCKSNITCH_MODELS_DIR") or (Path("models").exists() and any(Path("models").iterdir()))
    skip_gpu = pytest.mark.skip(reason="CUDA not available")
    skip_models = pytest.mark.skip(reason="Model weights not downloaded")
    for item in items:
        if "gpu" in item.keywords and not gpu_available:
            item.add_marker(skip_gpu)
        if "models" in item.keywords and not have_models:
            item.add_marker(skip_models)


# ---------------------------------------------------------------------------
# Synthetic CAHVORE / stereo fixtures
# ---------------------------------------------------------------------------


def _make_camera(C: np.ndarray, image_size: tuple[int, int] = (960, 1280)) -> CameraModel:
    """Build a minimal near-linear CAHVORE with pinhole H, V and zero distortion."""
    H_img, W_img = image_size
    fx = 576.0  # ~Navcam 4x-binned
    fy = 576.0
    cx = W_img / 2.0
    cy = H_img / 2.0

    A = np.array([0.0, 1.0, 0.0], dtype=np.float64)  # +Y = forward
    # x_image axis in world: +X
    # y_image axis in world: -Z (image y grows downward)
    Hx = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    Vy = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    H = fx * Hx + cx * A
    V = fy * Vy + cy * A
    return CameraModel(
        C=np.asarray(C, dtype=np.float64),
        A=A.copy(),
        H=H,
        V=V,
        O=A.copy(),  # zero decentering
        R=np.zeros(3, dtype=np.float64),
        E=np.zeros(3, dtype=np.float64),
        linearity=1.0,
        mtype=0.0,
        mparam=0.0,
        image_size=image_size,
    )


@pytest.fixture
def synthetic_left_cam() -> CameraModel:
    return _make_camera(C=np.array([-0.212, 0.0, 0.0]))


@pytest.fixture
def synthetic_right_cam() -> CameraModel:
    return _make_camera(C=np.array([+0.212, 0.0, 0.0]))  # 42.4 cm baseline


@pytest.fixture
def synthetic_filename_parts() -> FilenameParts:
    return FilenameParts(
        instrument="N",
        eye="L",
        kind="F",
        sol=100,
        sclk_sec=675828717,
        sclk_msec=276,
        product_code="ECM",
        venue="N0040218NCAM00503",
        camera_flags="01",
        downsample_code="295J",
        raw="NLF_0100_0675828717_276ECM_N0040218NCAM00503_01_295J",
    )


@pytest.fixture
def synthetic_image_meta(
    synthetic_left_cam: CameraModel,
    synthetic_filename_parts: FilenameParts,
    tmp_path: Path,
) -> ImageMeta:
    img = np.zeros((synthetic_left_cam.image_size[0], synthetic_left_cam.image_size[1], 3), dtype=np.uint8)
    p = tmp_path / "fake.png"
    try:
        import cv2

        cv2.imwrite(str(p), img)
    except Exception:
        p.write_bytes(b"\x89PNG\r\n\x1a\n")
    jp = tmp_path / "fake.json"
    jp.write_text(json.dumps({"imageid": "fake"}))
    return ImageMeta(
        imageid="fake",
        sol=100,
        site=4,
        drive="218",
        instrument="NAVCAM_LEFT",
        sclk=675828717.276,
        mast_az_deg=78.8,
        mast_el_deg=1.06,
        attitude_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        rover_xyz=np.array([-27.567, -7.029, 0.093]),
        subframe=Subframe(1, 1, 1280, 960),
        dimension=(1280, 960),
        scale_factor=1,
        camera_model=synthetic_left_cam,
        image_path=p,
        json_path=jp,
        filename=synthetic_filename_parts,
    )


@pytest.fixture
def synthetic_stereo_pair(
    synthetic_left_cam: CameraModel,
    synthetic_right_cam: CameraModel,
    synthetic_filename_parts: FilenameParts,
    tmp_path: Path,
) -> StereoPair:
    def _mk(eye: str, cam: CameraModel) -> ImageMeta:
        p = tmp_path / f"{eye}.png"
        jp = tmp_path / f"{eye}.json"
        try:
            import cv2

            cv2.imwrite(
                str(p), np.zeros((cam.image_size[0], cam.image_size[1], 3), dtype=np.uint8)
            )
        except Exception:
            p.write_bytes(b"\x89PNG\r\n\x1a\n")
        jp.write_text("{}")
        return ImageMeta(
            imageid=f"fake_{eye}",
            sol=100,
            site=4,
            drive="218",
            instrument="NAVCAM_LEFT" if eye == "L" else "NAVCAM_RIGHT",
            sclk=675828717.276,
            mast_az_deg=78.8,
            mast_el_deg=1.06,
            attitude_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            rover_xyz=np.array([0.0, 0.0, 0.0]),
            subframe=Subframe(1, 1, cam.image_size[1], cam.image_size[0]),
            dimension=(cam.image_size[1], cam.image_size[0]),
            scale_factor=1,
            camera_model=cam,
            image_path=p,
            json_path=jp,
            filename=synthetic_filename_parts,
        )

    return StereoPair(
        left=_mk("L", synthetic_left_cam),
        right=_mk("R", synthetic_right_cam),
        sclk_delta_s=0.0,
        mast_az_delta_deg=0.0,
        mast_el_delta_deg=0.0,
    )


@pytest.fixture
def sample_real_json() -> dict:
    """A realistic Mars2020 raw-images JSON payload (based on sol-100 sample)."""
    return {
        "extended": {
            "mastAz": "78.7995",
            "mastEl": "1.06137",
            "sclk": "675828717.752",
            "scaleFactor": "4",
            "xyz": "(-27.567,-7.02897,0.0931684)",
            "subframeRect": "(1,1441,5120,960)",
            "dimension": "(1280,240)",
        },
        "sol": 100,
        "attitude": "(0.353249,0.00650504,-0.00830915,-0.93547)",
        "imageid": "NLF_0100_0675828717_276ECM_N0040218NCAM00503_01_295J",
        "camera": {
            "filter_name": "UNK",
            "camera_vector": "(0.19354973918874185,0.9809280486566764,-0.017851101326669873)",
            "camera_model_component_list": (
                "(1.03927,0.645857,-1.9785);(0.182183,0.983214,-0.010707);"
                "(-609.121,772.41,-9.07759);(22.3091,131.353,738.105);"
                "(0.180782,0.983474,-0.0105907);(2e-06,0.049535,-0.015973);"
                "(-0.003612,0.013016,-0.023961);2.0;0.0"
            ),
            "camera_position": "(1.03927,0.645857,-1.9785)",
            "instrument": "NAVCAM_LEFT",
            "camera_model_type": "CAHVORE",
        },
        "caption": "Mars Perseverance Sol 100 Navcam",
        "sample_type": "Full",
        "date_taken_mars": "Sol-00100M15:26:09.623",
        "credit": "NASA/JPL-Caltech",
        "date_taken_utc": "2021-06-01T14:14:37.984",
        "json_link": "https://mars.nasa.gov/rss/api/?id=...",
        "link": "https://mars.nasa.gov/mars2020/multimedia/raw-images/NLF_0100_0675828717_276ECM_N0040218NCAM00503_01_295J",
        "drive": "218",
        "title": "Mars Perseverance Sol 100: Left Navigation Camera (Navcam)",
        "site": 4,
        "date_received": "2021-06-01T18:53:30Z",
    }
