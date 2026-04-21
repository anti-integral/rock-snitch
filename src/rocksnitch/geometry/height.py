"""Height above ground utilities."""
from __future__ import annotations

import numpy as np

from rocksnitch.contracts import FloatArray, GroundPlane


def signed_height_above_plane(
    points: FloatArray, plane: GroundPlane
) -> FloatArray:
    """Return signed height (metres) of each point above the plane."""
    return points @ plane.normal + plane.d


def mask_max_height(
    points: FloatArray, plane: GroundPlane
) -> float:
    """Peak signed height across a set of points."""
    if len(points) == 0:
        return 0.0
    return float(np.max(signed_height_above_plane(points, plane)))


def mask_height_stats(
    points: FloatArray, plane: GroundPlane
) -> dict[str, float]:
    """Summary stats of heights above plane."""
    if len(points) == 0:
        return {"max": 0.0, "mean": 0.0, "p95": 0.0, "count": 0}
    h = signed_height_above_plane(points, plane)
    return {
        "max": float(np.max(h)),
        "mean": float(np.mean(h)),
        "p95": float(np.percentile(h, 95.0)),
        "count": int(len(h)),
    }


__all__ = ["signed_height_above_plane", "mask_max_height", "mask_height_stats"]
