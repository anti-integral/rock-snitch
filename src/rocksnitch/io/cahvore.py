"""CAHVORE camera-model parsing and math.

Port of the subset of Gennery (2001) / JPL `CAHVORE` needed for:
  * parsing ``camera_model_component_list`` strings from Mars2020 sidecar JSONs
  * projecting a 3D world point -> pixel
  * linearizing a distorted image -> equivalent pinhole CAHV model
  * extracting focal length, principal point, and optical centre

The full CAHVORE has 20 parameters. Component list order observed in Mars2020
sidecars is:

    (Cx,Cy,Cz); (Ax,Ay,Az); (Hx,Hy,Hz); (Vx,Vy,Vz); (Ox,Oy,Oz);
    (R0,R1,R2); (E0,E1,E2); LIN; MTYPE; MPARAM

We preserve E, O, R, linearity parameters and expose :func:`project`,
:func:`linearize_to_cahv`, :func:`principal_point`, and :func:`focal_lengths`.
"""
from __future__ import annotations

import re

import numpy as np

from rocksnitch.contracts import CameraModel, FloatArray


_VEC_RE = re.compile(r"\(\s*([^)]+?)\s*\)")


def _split_components(blob: str) -> list[str]:
    """Split on ';' but tolerate whitespace + trailing bits."""
    return [p.strip() for p in blob.split(";") if p.strip()]


def _parse_vec3(text: str) -> FloatArray:
    m = _VEC_RE.fullmatch(text)
    if m is None:
        raise ValueError(f"Expected a 3-tuple like (x,y,z), got {text!r}")
    parts = [float(x.strip()) for x in m.group(1).split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 components, got {parts}")
    return np.asarray(parts, dtype=np.float64)


def _parse_scalar(text: str) -> float:
    return float(text.strip())


def parse_component_list(blob: str, image_size: tuple[int, int]) -> CameraModel:
    """Parse a Mars2020 ``camera_model_component_list`` string."""
    comps = _split_components(blob)
    if len(comps) < 10:
        # Some sidecars ship without E/extension scalars; pad with zeros.
        while len(comps) < 10:
            comps.append("0.0" if len(comps) >= 7 else "(0,0,0)")

    C, A, H, V, O, R, E = (_parse_vec3(c) for c in comps[:7])
    linearity = _parse_scalar(comps[7]) if len(comps) > 7 else 1.0
    mtype = _parse_scalar(comps[8]) if len(comps) > 8 else 0.0
    mparam = _parse_scalar(comps[9]) if len(comps) > 9 else 0.0

    return CameraModel(
        C=C, A=A, H=H, V=V, O=O, R=R, E=E,
        linearity=linearity, mtype=mtype, mparam=mparam,
        image_size=image_size,
    )


# ---------------------------------------------------------------------------
# Basic CAHV projection (post-linearization)
# ---------------------------------------------------------------------------


def principal_point(cam: CameraModel) -> tuple[float, float]:
    """Return (cx, cy) for the CAHV linear core."""
    Hc = float(np.dot(cam.H, cam.A))
    Vc = float(np.dot(cam.V, cam.A))
    return Hc, Vc


def focal_lengths(cam: CameraModel) -> tuple[float, float]:
    """Return (fx, fy) = (|H' - cx*A|, |V' - cy*A|)."""
    cx, cy = principal_point(cam)
    fx = float(np.linalg.norm(cam.H - cx * cam.A))
    fy = float(np.linalg.norm(cam.V - cy * cam.A))
    return fx, fy


def project_cahv(cam: CameraModel, xyz: FloatArray) -> FloatArray:
    """Project (N, 3) world points through the CAHV linear core.

    Returns (N, 2) float64 pixel coords (u, v). No distortion applied.
    """
    xyz = np.atleast_2d(np.asarray(xyz, dtype=np.float64))
    if xyz.shape[-1] != 3:
        raise ValueError(f"project_cahv: expected (...,3), got {xyz.shape}")
    d = xyz - cam.C
    denom = d @ cam.A
    denom = np.where(np.abs(denom) < 1e-12, np.sign(denom) * 1e-12 + 1e-12, denom)
    u = (d @ cam.H) / denom
    v = (d @ cam.V) / denom
    return np.stack([u, v], axis=-1)


def unproject_cahv(cam: CameraModel, uv: FloatArray) -> FloatArray:
    """Unproject pixel coords through CAHV to unit rays in world frame.

    Returns (N, 3) float64 rays.
    """
    uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
    u = uv[..., 0]
    v = uv[..., 1]
    cx, cy = principal_point(cam)
    fx, fy = focal_lengths(cam)
    Hp = (cam.H - cx * cam.A) / fx
    Vp = (cam.V - cy * cam.A) / fy
    ray = (u[..., None] - cx) * Hp / fx + (v[..., None] - cy) * Vp / fy + cam.A
    ray = ray / np.linalg.norm(ray, axis=-1, keepdims=True)
    return ray


# ---------------------------------------------------------------------------
# Distortion
# ---------------------------------------------------------------------------


def _cahvore_distort(cam: CameraModel, x: FloatArray) -> FloatArray:
    """Apply CAHVOR radial distortion to a set of unit rays in world frame.

    Simplified model: the ray makes angle theta with axis O; radial distortion
    polynomial r(theta) = theta * (1 + R0 + R1*theta^2 + R2*theta^4).
    CAHVORE "mtype/mparam" linearity extension is applied as a pre-bend of theta
    when linearity != 1 (perspective .. fish-eye interpolation).
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    cos_theta = np.clip(x @ cam.O, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    # CAHVORE linearity: theta' = mu*tan(theta) / tan(mu*theta) * theta, mu = linearity.
    # If linearity == 1, this reduces to identity.
    mu = float(cam.linearity) if cam.linearity > 0 else 1.0
    if abs(mu - 1.0) > 1e-9:
        safe = np.where(theta > 1e-9, theta, 1.0)
        theta_p = np.where(
            theta > 1e-9,
            np.arctan(mu * np.tan(safe)) / mu,
            theta,
        )
    else:
        theta_p = theta
    r = cam.R
    distort = theta_p * (1.0 + r[0] + r[1] * theta_p ** 2 + r[2] * theta_p ** 4)
    sin_theta = np.sin(theta)
    safe_sin = np.where(sin_theta > 1e-9, sin_theta, 1.0)
    # Perpendicular component in the O-plane
    perp = x - cos_theta[..., None] * cam.O
    perp_norm = np.linalg.norm(perp, axis=-1, keepdims=True)
    perp_unit = np.where(perp_norm > 1e-9, perp / np.where(perp_norm > 1e-12, perp_norm, 1.0), 0.0)
    # Reconstruct distorted ray: cos(distort)*O + sin(distort)*perp_unit
    cos_d = np.cos(distort)
    sin_d = np.sin(distort)
    out = cos_d[..., None] * cam.O + sin_d[..., None] * perp_unit
    out = out / np.linalg.norm(out, axis=-1, keepdims=True)
    # Avoid unused-variable lint for safe_sin (kept for documentation clarity)
    _ = safe_sin
    return out


def project(cam: CameraModel, xyz: FloatArray) -> FloatArray:
    """Full CAHVORE projection (N,3) -> (N,2)."""
    xyz = np.atleast_2d(np.asarray(xyz, dtype=np.float64))
    d = xyz - cam.C
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    d_distorted = _cahvore_distort(cam, d)
    # Step back into "world units"
    denom = d_distorted @ cam.A
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    u = (d_distorted @ cam.H) / denom
    v = (d_distorted @ cam.V) / denom
    return np.stack([u, v], axis=-1)


# ---------------------------------------------------------------------------
# Linearization
# ---------------------------------------------------------------------------


def linearize_to_cahv(cam: CameraModel) -> CameraModel:
    """Return a copy of the camera with zero distortion (R=0, E=0, linearity=1).

    The CAHV linear core (C, A, H, V) is kept verbatim because most pipelines
    first linearize the image then project with the linear K.
    """
    return CameraModel(
        C=cam.C.copy(),
        A=cam.A.copy(),
        H=cam.H.copy(),
        V=cam.V.copy(),
        O=cam.A.copy(),
        R=np.zeros(3, dtype=np.float64),
        E=np.zeros(3, dtype=np.float64),
        linearity=1.0,
        mtype=0.0,
        mparam=0.0,
        image_size=cam.image_size,
    )


def extrinsics_matrix(cam: CameraModel) -> FloatArray:
    """Return a 4x4 homogeneous cam->world transform.

    Camera frame: +x = right (H'), +y = down (V'), +z = forward (A).
    World-origin position = cam.C.
    """
    cx, cy = principal_point(cam)
    fx, fy = focal_lengths(cam)
    Hp = (cam.H - cx * cam.A) / fx
    Vp = (cam.V - cy * cam.A) / fy
    R = np.column_stack([Hp, Vp, cam.A])  # 3x3
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = cam.C
    return T


def intrinsics_matrix(cam: CameraModel) -> FloatArray:
    """Return 3x3 pinhole K for the linearized CAHV core."""
    cx, cy = principal_point(cam)
    fx, fy = focal_lengths(cam)
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


__all__ = [
    "parse_component_list",
    "principal_point",
    "focal_lengths",
    "project",
    "project_cahv",
    "unproject_cahv",
    "linearize_to_cahv",
    "extrinsics_matrix",
    "intrinsics_matrix",
]
