"""Shared dataclasses, enums, and Protocols used across rock-snitch modules.

All modules import from here to stay decoupled. Numpy arrays carry shape/dtype
contracts in their field docstrings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.floating]
IntArray = npt.NDArray[np.integer]
BoolArray = npt.NDArray[np.bool_]
UInt8Array = npt.NDArray[np.uint8]


# ---------------------------------------------------------------------------
# Camera-side data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraModel:
    """CAHVORE camera model.

    All vectors are 3-d ndarrays (float64) in the rover local-level frame, metres.
    R, E are 3-vectors of dimensionless distortion coefficients.
    `linearity` and `mtype`/`mparam` are the CAHVORE extension scalars.
    `image_size` is (H, W) in pixels of the image this model corresponds to.
    """

    C: FloatArray
    A: FloatArray
    H: FloatArray
    V: FloatArray
    O: FloatArray
    R: FloatArray
    E: FloatArray
    linearity: float
    mtype: float
    mparam: float
    image_size: tuple[int, int]

    def __post_init__(self) -> None:
        for name in ("C", "A", "H", "V", "O", "R", "E"):
            v = getattr(self, name)
            if v.shape != (3,):
                raise ValueError(f"CAHVORE.{name} must be shape (3,), got {v.shape}")

    def focal_length_px(self) -> float:
        """Return |H x A|, approx focal length in pixels (CAHV linear core)."""
        return float(np.linalg.norm(np.cross(self.H, self.A)))


@dataclass(frozen=True)
class PinholeIntrinsics:
    """Simple pinhole intrinsics K = [[fx,0,cx],[0,fy,cy],[0,0,1]]."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def K(self) -> FloatArray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


# ---------------------------------------------------------------------------
# Image metadata parsed from the Mars2020 JSON sidecar
# ---------------------------------------------------------------------------


Instrument = Literal["NAVCAM_LEFT", "NAVCAM_RIGHT"]
Eye = Literal["L", "R"]
ProductKind = Literal["F", "E", "B", "G", "R", "M", "UNKNOWN"]


@dataclass(frozen=True)
class FilenameParts:
    """Decoded Mars2020 Navcam filename fields.

    Example: NLF_0100_0675828717_276ECM_N0040218NCAM00503_01_295J
      - instrument='N' (navcam), eye='L', kind='F' (full color)
      - sol=100, sclk_sec=675828717, sclk_msec=276
      - product_code='ECM', venue='N0040218NCAM00503'
      - camera_flags='01', downsample_code='295J'
    """

    instrument: Literal["N"]
    eye: Eye
    kind: ProductKind
    sol: int
    sclk_sec: int
    sclk_msec: int
    product_code: str
    venue: str
    camera_flags: str
    downsample_code: str
    raw: str


@dataclass(frozen=True)
class Subframe:
    """Sensor-coordinate subframe (x, y, w, h) in native pixels."""

    x: int
    y: int
    w: int
    h: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


@dataclass(frozen=True)
class ImageMeta:
    """Flattened view of a Mars2020 raw-images JSON sidecar + resolved paths."""

    imageid: str
    sol: int
    site: int
    drive: str
    instrument: Instrument
    sclk: float
    mast_az_deg: float | None
    mast_el_deg: float | None
    attitude_quat_wxyz: FloatArray
    rover_xyz: FloatArray
    subframe: Subframe
    dimension: tuple[int, int]
    scale_factor: int
    camera_model: CameraModel
    image_path: Path
    json_path: Path
    filename: FilenameParts
    caption: str = ""
    date_taken_utc: str | None = None


@dataclass(frozen=True)
class StereoPair:
    """A confirmed left/right stereo pair (same SCLK and mast pointing)."""

    left: ImageMeta
    right: ImageMeta
    sclk_delta_s: float
    mast_az_delta_deg: float
    mast_el_delta_deg: float


# ---------------------------------------------------------------------------
# Geometry outputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RectifiedPair:
    """Rectified stereo pair in a common CAHV frame.

    left, right: (H, W, C) uint8 images after remap.
    K: 3x3 rectified intrinsics shared by both eyes.
    baseline_m: stereo baseline in metres.
    left_to_world: 4x4 homogeneous transform from rectified-left to rover site frame.
    """

    left: UInt8Array
    right: UInt8Array
    K: FloatArray
    baseline_m: float
    left_to_world: FloatArray


@dataclass(frozen=True)
class DisparityMap:
    """Per-pixel disparity (left-eye reference).

    disparity: (H, W) float32 pixels; NaN where invalid.
    confidence: (H, W) float32 in [0, 1].
    mask: (H, W) bool - True where disparity is usable.
    """

    disparity: FloatArray
    confidence: FloatArray
    mask: BoolArray


@dataclass(frozen=True)
class PointCloud:
    """Per-pixel 3D points.

    xyz: (H, W, 3) float32 metres in `frame`.
    valid: (H, W) bool.
    frame: reference frame name.
    """

    xyz: FloatArray
    valid: BoolArray
    frame: Literal["left_cam", "rover_site"]


@dataclass(frozen=True)
class GroundPlane:
    """Plane n . x + d = 0 fit to ground points in rover site frame."""

    normal: FloatArray  # (3,) unit vector
    d: float
    inlier_mask: BoolArray  # (N,) over fit input points
    rmse: float


# ---------------------------------------------------------------------------
# Perception outputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Mask2D:
    """Binary mask in image coordinates plus bbox and score."""

    mask: BoolArray  # (H, W)
    bbox_xywh: tuple[int, int, int, int]
    score: float
    source: str = ""  # "sam2", "grounded", etc.


@dataclass(frozen=True)
class MaskList:
    """Collection of 2D masks for one image."""

    image_size: tuple[int, int]  # (H, W)
    masks: list[Mask2D] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.masks)


@dataclass(frozen=True)
class DepthMap:
    """Per-pixel metric depth (along camera boresight).

    depth: (H, W) float32 metres. `uncertainty`: optional (H, W) stddev.
    """

    depth: FloatArray
    uncertainty: FloatArray | None = None
    K: FloatArray | None = None


# ---------------------------------------------------------------------------
# Detection output
# ---------------------------------------------------------------------------


DetectionSource = Literal["stereo", "mono", "fused"]


@dataclass(frozen=True)
class RockDetection:
    """A single detected rock."""

    uv_bbox: tuple[int, int, int, int]  # (x, y, w, h)
    mask_rle: str | None  # RLE-encoded binary mask (optional)
    centroid_uv: tuple[float, float]
    range_m: float
    height_m: float
    width_m: float
    confidence: float
    source: DetectionSource
    xyz_rover: tuple[float, float, float] | None = None  # centroid in rover site frame


# ---------------------------------------------------------------------------
# Protocols for pluggable backends
# ---------------------------------------------------------------------------


@runtime_checkable
class StereoMatcherProto(Protocol):
    def compute(self, pair: RectifiedPair) -> DisparityMap: ...


@runtime_checkable
class SegmenterProto(Protocol):
    def segment(self, image: UInt8Array) -> MaskList: ...


@runtime_checkable
class FeatureExtractorProto(Protocol):
    def extract(self, image: UInt8Array) -> FloatArray: ...  # (Hp, Wp, D)


@runtime_checkable
class DepthEstimatorProto(Protocol):
    def predict(
        self, image: UInt8Array, K: FloatArray | None = None
    ) -> DepthMap: ...


# ---------------------------------------------------------------------------
# Constants (Mars 2020 Navcam physical parameters)
# ---------------------------------------------------------------------------


NAVCAM_BASELINE_M = 0.424  # Maki et al. 2020
NAVCAM_NATIVE_IFOV_RAD = 0.33e-3
NAVCAM_NATIVE_WIDTH = 5120
NAVCAM_NATIVE_HEIGHT = 3840
NAVCAM_HFOV_DEG = 96.0
NAVCAM_VFOV_DEG = 73.0

ROCK_HEIGHT_THRESHOLD_M = 0.10  # target: detect rocks >=10 cm tall

# Stereo trust horizon: beyond this we fall back to monocular depth.
STEREO_TRUST_RANGE_M = 20.0

__all__ = [
    "FloatArray",
    "IntArray",
    "BoolArray",
    "UInt8Array",
    "CameraModel",
    "PinholeIntrinsics",
    "Instrument",
    "Eye",
    "ProductKind",
    "FilenameParts",
    "Subframe",
    "ImageMeta",
    "StereoPair",
    "RectifiedPair",
    "DisparityMap",
    "PointCloud",
    "GroundPlane",
    "Mask2D",
    "MaskList",
    "DepthMap",
    "DetectionSource",
    "RockDetection",
    "StereoMatcherProto",
    "SegmenterProto",
    "FeatureExtractorProto",
    "DepthEstimatorProto",
    "NAVCAM_BASELINE_M",
    "NAVCAM_NATIVE_IFOV_RAD",
    "NAVCAM_NATIVE_WIDTH",
    "NAVCAM_NATIVE_HEIGHT",
    "NAVCAM_HFOV_DEG",
    "NAVCAM_VFOV_DEG",
    "ROCK_HEIGHT_THRESHOLD_M",
    "STEREO_TRUST_RANGE_M",
]
