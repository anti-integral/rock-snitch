"""Stereo rectification for a CAHVORE pair.

Approach:
  1. Linearize each image to its CAHV pinhole model (``linearize_to_cahv``).
  2. Build a common rectified CAHV whose axis bisects the two linearized axes
     and whose image rows are aligned with the baseline direction.
  3. Compute per-eye homographies (3x3) and remap the images.

Output is a :class:`RectifiedPair` in the rover-site frame (the left eye's
position is kept as origin; the transform to world is provided).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from rocksnitch.contracts import (
    CameraModel,
    FloatArray,
    RectifiedPair,
    StereoPair,
    UInt8Array,
)
from rocksnitch.io.cahvore import (
    focal_lengths,
    intrinsics_matrix,
    linearize_to_cahv,
    principal_point,
)


@dataclass(frozen=True)
class RectifyOutputs:
    """Intermediate rectification artefacts, useful for tests."""

    K_common: FloatArray
    R_left: FloatArray
    R_right: FloatArray
    baseline_m: float
    left_to_world: FloatArray


def _cam_to_world_rot(cam: CameraModel) -> FloatArray:
    """Return a 3x3 rotation from cam frame (x=right, y=down, z=forward) to world."""
    cx, cy = principal_point(cam)
    fx, fy = focal_lengths(cam)
    Hp = (cam.H - cx * cam.A) / fx
    Vp = (cam.V - cy * cam.A) / fy
    return np.column_stack([Hp, Vp, cam.A])


def compute_rectification(
    left: CameraModel, right: CameraModel
) -> RectifyOutputs:
    """Compute rectified intrinsics and per-eye rotations.

    Result: both eyes share intrinsics K and look along a new axis A_rect.
    Baseline is measured in world units along the new x-axis.
    """
    left_lin = linearize_to_cahv(left)
    right_lin = linearize_to_cahv(right)

    # New x-axis: normalized baseline (right cam position - left cam position).
    baseline_vec = right_lin.C - left_lin.C
    baseline_m = float(np.linalg.norm(baseline_vec))
    if baseline_m < 1e-6:
        raise ValueError("Zero stereo baseline; cameras coincide.")
    x_axis = baseline_vec / baseline_m

    # Forward: average of A's (should be nearly parallel for a matched pair).
    fwd = left_lin.A + right_lin.A
    fwd /= np.linalg.norm(fwd)
    # Orthogonalize: project out x_axis component
    fwd = fwd - x_axis * np.dot(fwd, x_axis)
    fwd /= np.linalg.norm(fwd)

    # y-axis: right-handed frame -> x cross z (forward) = y
    # We define image frame: x=right, y=down, z=forward
    y_axis = np.cross(fwd, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    R_new_world = np.column_stack([x_axis, y_axis, fwd])  # 3x3, cam_new -> world

    R_left_cam = _cam_to_world_rot(left_lin)
    R_right_cam = _cam_to_world_rot(right_lin)

    # Per-eye rotation: old-cam -> new-cam
    R_left = R_new_world.T @ R_left_cam
    R_right = R_new_world.T @ R_right_cam

    # Shared intrinsics: average fx, fy of both lin cameras
    fx_l, fy_l = focal_lengths(left_lin)
    fx_r, fy_r = focal_lengths(right_lin)
    fx = 0.5 * (fx_l + fx_r)
    fy = 0.5 * (fy_l + fy_r)
    H_img, W_img = left_lin.image_size
    cx = W_img / 2.0
    cy = H_img / 2.0
    K_common = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    left_to_world = np.eye(4, dtype=np.float64)
    left_to_world[:3, :3] = R_new_world
    left_to_world[:3, 3] = left_lin.C

    return RectifyOutputs(
        K_common=K_common,
        R_left=R_left,
        R_right=R_right,
        baseline_m=baseline_m,
        left_to_world=left_to_world,
    )


def _homography_from_rotation(
    K_src: FloatArray, K_dst: FloatArray, R: FloatArray
) -> FloatArray:
    """H = K_dst R K_src^-1."""
    return K_dst @ R @ np.linalg.inv(K_src)


def rectify_pair(
    pair: StereoPair,
    left_image: UInt8Array,
    right_image: UInt8Array,
) -> RectifiedPair:
    """Remap a stereo image pair into a shared rectified CAHV frame."""
    out = compute_rectification(pair.left.camera_model, pair.right.camera_model)

    K_left = intrinsics_matrix(linearize_to_cahv(pair.left.camera_model))
    K_right = intrinsics_matrix(linearize_to_cahv(pair.right.camera_model))

    H_left = _homography_from_rotation(K_left, out.K_common, out.R_left)
    H_right = _homography_from_rotation(K_right, out.K_common, out.R_right)

    H_img, W_img = pair.left.camera_model.image_size
    left_rect = cv2.warpPerspective(left_image, H_left, (W_img, H_img))
    right_rect = cv2.warpPerspective(right_image, H_right, (W_img, H_img))

    return RectifiedPair(
        left=left_rect,
        right=right_rect,
        K=out.K_common,
        baseline_m=out.baseline_m,
        left_to_world=out.left_to_world,
    )


def disparity_to_depth(
    disparity: FloatArray, K: FloatArray, baseline_m: float
) -> FloatArray:
    """Convert rectified disparity (pixels) to metric depth Z = f*b / d.

    Returns float32 depth; invalid disparities become NaN.
    """
    d = np.asarray(disparity, dtype=np.float32)
    fx = float(K[0, 0])
    out = np.full_like(d, np.nan, dtype=np.float32)
    mask = np.isfinite(d) & (d > 0.0)
    out[mask] = fx * baseline_m / d[mask]
    return out


__all__ = [
    "RectifyOutputs",
    "compute_rectification",
    "rectify_pair",
    "disparity_to_depth",
]
