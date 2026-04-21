"""Stereo-first near-field rock detection branch."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from rocksnitch.contracts import (
    BoolArray,
    DepthEstimatorProto,
    DisparityMap,
    FeatureExtractorProto,
    GroundPlane,
    MaskList,
    PointCloud,
    RectifiedPair,
    RockDetection,
    SegmenterProto,
    StereoMatcherProto,
    StereoPair,
    UInt8Array,
)
from rocksnitch.geometry.ground_plane import RansacConfig, fit_ransac
from rocksnitch.geometry.height import mask_height_stats
from rocksnitch.geometry.pointcloud import disparity_to_pointcloud, mask_points
from rocksnitch.geometry.rectify import rectify_pair


@dataclass
class NearFieldConfig:
    min_height_m: float = 0.10
    min_mask_pixels: int = 15
    max_range_m: float = 30.0
    ground_distance_thresh_m: float = 0.05
    ground_max_iters: int = 2000
    min_stereo_confidence: float = 0.3


@dataclass
class NearFieldArtefacts:
    rectified: RectifiedPair
    disparity: DisparityMap
    pointcloud: PointCloud
    ground_plane: GroundPlane
    masks: MaskList


def _bbox_from_mask(mask: BoolArray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return 0, 0, 0, 0
    x0, y0 = int(xs.min()), int(ys.min())
    return x0, y0, int(xs.max() - x0 + 1), int(ys.max() - y0 + 1)


def _centroid(mask: BoolArray) -> tuple[float, float]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return 0.0, 0.0
    return float(xs.mean()), float(ys.mean())


def run_near_field(
    pair: StereoPair,
    left_image: UInt8Array,
    right_image: UInt8Array,
    *,
    stereo: StereoMatcherProto,
    segmenter: SegmenterProto,
    config: NearFieldConfig | None = None,
) -> tuple[list[RockDetection], NearFieldArtefacts]:
    """Run the stereo branch end-to-end and return detections + artefacts."""
    cfg = config or NearFieldConfig()
    rectified = rectify_pair(pair, left_image, right_image)
    disparity = stereo.compute(rectified)
    pointcloud = disparity_to_pointcloud(rectified, disparity, frame="left_cam")

    ground_points = pointcloud.xyz[pointcloud.valid]
    if len(ground_points) < 100:
        detections: list[RockDetection] = []
        plane = GroundPlane(
            normal=np.array([0.0, 1.0, 0.0]), d=0.0, inlier_mask=np.zeros(0, dtype=bool), rmse=0.0
        )
        return detections, NearFieldArtefacts(
            rectified=rectified,
            disparity=disparity,
            pointcloud=pointcloud,
            ground_plane=plane,
            masks=MaskList(image_size=rectified.left.shape[:2]),
        )
    sample_idx = np.random.default_rng(0).choice(
        len(ground_points), size=min(20000, len(ground_points)), replace=False
    )
    plane = fit_ransac(
        ground_points[sample_idx],
        RansacConfig(
            max_iters=cfg.ground_max_iters,
            distance_thresh_m=cfg.ground_distance_thresh_m,
        ),
    )
    masks = segmenter.segment(rectified.left)

    detections = []
    for m in masks.masks:
        if int(m.mask.sum()) < cfg.min_mask_pixels:
            continue
        pts = mask_points(pointcloud, m.mask)
        if len(pts) < 3:
            continue
        range_m = float(np.linalg.norm(pts.mean(axis=0)))
        if range_m > cfg.max_range_m:
            continue
        stats = mask_height_stats(pts, plane)
        if stats["p95"] < cfg.min_height_m:
            continue
        bbox = _bbox_from_mask(m.mask)
        cu, cv = _centroid(m.mask)
        _norm = max(cfg.min_height_m, 1e-3); confidence = float(min(1.0, m.score * (stats["p95"] / _norm)))
        xyz_mean = tuple(float(v) for v in pts.mean(axis=0))
        width_m = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        detections.append(
            RockDetection(
                uv_bbox=bbox,
                mask_rle=None,
                centroid_uv=(cu, cv),
                range_m=range_m,
                height_m=float(stats["p95"]),
                width_m=width_m,
                confidence=confidence,
                source="stereo",
                xyz_rover=xyz_mean,  # type: ignore[arg-type]
            )
        )
    art = NearFieldArtefacts(
        rectified=rectified,
        disparity=disparity,
        pointcloud=pointcloud,
        ground_plane=plane,
        masks=masks,
    )
    return detections, art


__all__ = ["NearFieldConfig", "NearFieldArtefacts", "run_near_field"]
