from __future__ import annotations

from rocksnitch.contracts import RockDetection
from rocksnitch.eval.metrics import (
    mean_height_error,
    precision_recall,
    range_binned_pr,
)


def _d(x: int, h_m: float = 0.2, r_m: float = 15.0) -> RockDetection:
    return RockDetection(
        uv_bbox=(x, 0, 20, 20),
        mask_rle=None,
        centroid_uv=(x + 10, 10),
        range_m=r_m,
        height_m=h_m,
        width_m=0.1,
        confidence=0.9,
        source="stereo",
    )


def test_perfect_match_gives_p1_r1() -> None:
    pred = [_d(0), _d(50)]
    gt = [_d(0), _d(50)]
    p = precision_recall(pred, gt)
    assert p.precision == 1.0
    assert p.recall == 1.0


def test_half_missing() -> None:
    p = precision_recall([_d(0)], [_d(0), _d(50)])
    assert p.recall == 0.5


def test_range_binned_pr_shape() -> None:
    out = range_binned_pr([_d(0, r_m=5)], [_d(0, r_m=5), _d(50, r_m=25)])
    assert (0, 10) in out
    assert (20, 30) in out


def test_mean_height_error() -> None:
    pred = [_d(0, h_m=0.18)]
    gt = [_d(0, h_m=0.20)]
    err = mean_height_error(pred, gt)
    assert abs(err - 0.02) < 1e-6
