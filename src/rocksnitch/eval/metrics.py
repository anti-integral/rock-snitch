"""Detection metrics."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from rocksnitch.contracts import RockDetection


@dataclass
class PRPoint:
    threshold: float
    precision: float
    recall: float
    tp: int
    fp: int
    fn: int


def _iou(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / max(union, 1)


def match_detections(
    pred: list[RockDetection],
    gt: list[RockDetection],
    iou_thresh: float = 0.3,
) -> tuple[list[int], list[int]]:
    """Greedy bbox matching. Returns (pred_to_gt, gt_matched). -1 == unmatched."""
    pred_to_gt = [-1] * len(pred)
    gt_matched = [0] * len(gt)
    order = sorted(range(len(pred)), key=lambda i: -pred[i].confidence)
    for pi in order:
        best = -1
        best_iou = iou_thresh
        for gi, g in enumerate(gt):
            if gt_matched[gi]:
                continue
            v = _iou(pred[pi].uv_bbox, g.uv_bbox)
            if v > best_iou:
                best_iou = v
                best = gi
        if best >= 0:
            pred_to_gt[pi] = best
            gt_matched[best] = 1
    return pred_to_gt, gt_matched


def precision_recall(
    pred: list[RockDetection],
    gt: list[RockDetection],
    height_thresh_m: float = 0.10,
    iou_thresh: float = 0.3,
) -> PRPoint:
    pred_ok = [p for p in pred if p.height_m >= height_thresh_m]
    gt_ok = [g for g in gt if g.height_m >= height_thresh_m]
    pred_to_gt, gt_matched = match_detections(pred_ok, gt_ok, iou_thresh=iou_thresh)
    tp = sum(1 for x in pred_to_gt if x >= 0)
    fp = len(pred_ok) - tp
    fn = sum(1 for m in gt_matched if m == 0)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return PRPoint(threshold=height_thresh_m, precision=prec, recall=rec, tp=tp, fp=fp, fn=fn)


def range_binned_pr(
    pred: list[RockDetection],
    gt: list[RockDetection],
    *,
    bins_m: list[tuple[float, float]] | None = None,
    height_thresh_m: float = 0.10,
    iou_thresh: float = 0.3,
) -> dict[tuple[float, float], PRPoint]:
    bins = bins_m or [(0, 10), (10, 20), (20, 30), (30, 50), (50, 100)]
    out: dict[tuple[float, float], PRPoint] = {}
    for lo, hi in bins:
        p = [d for d in pred if lo <= d.range_m < hi]
        g = [d for d in gt if lo <= d.range_m < hi]
        out[(lo, hi)] = precision_recall(p, g, height_thresh_m=height_thresh_m, iou_thresh=iou_thresh)
    return out


def mean_height_error(
    pred: list[RockDetection],
    gt: list[RockDetection],
    iou_thresh: float = 0.3,
) -> float:
    pred_to_gt, _ = match_detections(pred, gt, iou_thresh=iou_thresh)
    errs = []
    for pi, gi in enumerate(pred_to_gt):
        if gi >= 0:
            errs.append(abs(pred[pi].height_m - gt[gi].height_m))
    if not errs:
        return 0.0
    return float(np.mean(errs))


__all__ = ["PRPoint", "match_detections", "precision_recall", "range_binned_pr", "mean_height_error"]
