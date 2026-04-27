"""Build all figures embedded in docs/DESIGN.md.

Deterministic: same inputs -> same PNG bytes (seeded RNG, fixed frames).
Run once after changes that affect the numbers cited in DESIGN.md.

    python scripts/build_design_figures.py [--out docs/media]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from matplotlib import patches as mpatches
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from rocksnitch.contracts import (
    NAVCAM_BASELINE_M,
    NAVCAM_NATIVE_IFOV_RAD,
    ROCK_HEIGHT_THRESHOLD_M,
    STEREO_TRUST_RANGE_M,
)
from rocksnitch.eval.viz import overlay_detections, write_disparity_preview
from rocksnitch.geometry.disparity import SGBMMatcher
from rocksnitch.io.cahvore import (
    focal_lengths,
    principal_point,
    project,
    project_cahv,
    unproject_cahv,
)
from rocksnitch.io.metadata import load_meta
from rocksnitch.io.pairing import find_pairs
from rocksnitch.perception.dinov2 import MockFeatureExtractor
from rocksnitch.perception.mono_depth import MockDepthEstimator
from rocksnitch.perception.sam2 import MockSegmenter
from rocksnitch.pipeline.run import run_pipeline

REPO = Path(__file__).resolve().parent.parent
RELEVANT = REPO / "data" / "relevant"

# Prefer sol 100 from the full local dataset (gitignored but on dev disks);
# fall back to sol 101 from the committed relevant/ subset (same site/drive,
# same Navcam-Left calibration, near-identical CAHVORE).
_SOL_100_FULL = REPO / "data" / "metadata" / "00100" / "NLF_0100_0675828717_276ECM_N0040218NCAM00503_01_295J.json"
_SOL_101_REL = RELEVANT / "NLF_0101_0675907694_659ECM_N0040218NCAM00503_01_295J.json"
SOL_100_LEFT = _SOL_100_FULL if _SOL_100_FULL.exists() else _SOL_101_REL
SOL_755_LEFT = (
    RELEVANT
    / "NLF_0755_0733967989_005ECM_N0372562NCAM03755_01_195J.json"
)
SOL_755_RIGHT = (
    RELEVANT
    / "NRF_0755_0733967989_005ECM_N0372562NCAM03755_01_195J.json"
)


def _load_image_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)




def _resolved_meta(json_path: Path):
    """Load_meta but correctly resolve image path for files under data/metadata/<sol>/."""
    parts = json_path.parts
    if "metadata" in parts:
        i = parts.index("metadata")
        img_parts = list(parts)
        img_parts[i] = "images"
        img_path = Path(*img_parts).with_suffix(".png")
    else:
        img_path = json_path.with_suffix(".png")
    return load_meta(json_path, image_path=img_path)

def fig_cahvore_vectors(out: Path) -> None:
    """3D quiver plot of C, A, H, V, O for the sol-100 left Navcam."""
    meta_v = _resolved_meta(SOL_100_LEFT)
    cam = meta_v.camera_model
    cx, cy = principal_point(cam)
    fx, fy = focal_lengths(cam)
    H_dir = (cam.H - cx * cam.A) / fx  # right
    V_dir = (cam.V - cy * cam.A) / fy  # down

    fig = plt.figure(figsize=(10, 7), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    origin = cam.C
    arrows = [
        (cam.A, "A  optical axis (boresight)", "tab:blue"),
        (H_dir, "H' horizontal image axis", "tab:orange"),
        (V_dir, "V' vertical image axis", "tab:green"),
        (cam.O, "O  distortion symmetry axis", "tab:red"),
    ]
    L = 1.5  # arrow length in metres
    for vec, label, color in arrows:
        ax.quiver(
            *origin,
            *(L * vec),
            color=color,
            linewidth=2.5,
            arrow_length_ratio=0.12,
            label=label,
        )
    ax.scatter(*origin, color="black", s=40, label=f"C  (camera centre) = ({origin[0]:.2f}, {origin[1]:.2f}, {origin[2]:.2f})")
    ax.scatter(0, 0, 0, color="grey", s=30, marker="x", label="rover origin (0,0,0)")

    span = 2.5
    ax.set_xlim(origin[0] - span, origin[0] + span)
    ax.set_ylim(origin[1] - span, origin[1] + span)
    ax.set_zlim(origin[2] - span, origin[2] + span)
    ax.set_xlabel("X (m, rover-local)")
    ax.set_ylabel("Y (m, rover-local)")
    ax.set_zlabel("Z (m, rover-local)")
    ax.set_title(
        "CAHVORE vectors — sol 100 left Navcam\n"
        "C = optical centre, A = boresight, H'/V' = image axes, O = distortion axis"
    )
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
    ax.view_init(elev=18, azim=-65)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def fig_cahvore_distortion(out: Path) -> None:
    """2-panel: image with grid + distortion magnitude heatmap."""
    meta = _resolved_meta(SOL_100_LEFT)
    cam = meta.camera_model
    img = _load_image_rgb(meta.image_path)
    H, W = img.shape[:2]

    # Sample a fine grid; 5 m forward
    nu, nv = 32, 8
    us = np.linspace(0, W - 1, nu)
    vs = np.linspace(0, H - 1, nv)
    uu, vv = np.meshgrid(us, vs)
    pix = np.stack([uu.ravel(), vv.ravel()], axis=-1)
    rays = unproject_cahv(cam, pix)
    pts = cam.C + 5.0 * rays
    uv_lin = project_cahv(cam, pts)
    uv_full = project(cam, pts)
    err = np.linalg.norm(uv_lin - uv_full, axis=-1).reshape(nv, nu)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), dpi=110, gridspec_kw={"height_ratios": [1, 1]})

    ax0 = axes[0]
    ax0.imshow(img)
    sample_us = np.linspace(0, W - 1, 16)
    sample_vs = np.linspace(0, H - 1, 5)
    for u in sample_us:
        ax0.axvline(u, color="cyan", linewidth=0.4, alpha=0.7)
    for v in sample_vs:
        ax0.axhline(v, color="cyan", linewidth=0.4, alpha=0.7)
    ax0.set_title(f"Sol 100 left Navcam ({W}x{H} delivered) with reference grid")
    ax0.set_xticks([])
    ax0.set_yticks([])

    ax1 = axes[1]
    extent = (0, W, H, 0)
    im = ax1.imshow(
        err,
        extent=extent,
        cmap="magma",
        norm=LogNorm(vmin=max(err.min(), 0.05), vmax=err.max()),
        aspect="auto",
    )
    fig.colorbar(im, ax=ax1, label="|project_cahv - project| (px)", shrink=0.85)
    ax1.set_title(
        "Pixel disagreement: linear-CAHV vs full CAHVORE projection\n"
        f"centre {err[err.shape[0]//2, err.shape[1]//2]:.2f} px,  mean {err.mean():.1f} px,  max {err.max():.0f} px"
    )
    ax1.set_xticks([])
    ax1.set_yticks([])
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def fig_angular_size_overlay(out: Path) -> None:
    """Sol 755 left frame with 10 cm scale bars at 10/30/50/100 m."""
    img = _load_image_rgb(_resolved_meta(SOL_755_LEFT).image_path)
    H, W = img.shape[:2]

    # Pixel size of a 10 cm rock at each range, given binned IFOV
    binning = 4
    ifov = NAVCAM_NATIVE_IFOV_RAD * binning  # rad/px
    ranges = [10.0, 30.0, 50.0, 100.0]
    rock_h = ROCK_HEIGHT_THRESHOLD_M

    fig, ax = plt.subplots(figsize=(11, 7), dpi=110)
    ax.imshow(img)

    # Place the bars in a row near the bottom of the frame, evenly spaced
    bar_y = H - 80
    spacing = W / (len(ranges) + 1)
    for i, R in enumerate(ranges, start=1):
        bar_px = rock_h / (R * ifov)  # height of a 10 cm rock in delivered pixels
        cx = i * spacing
        # Yellow vertical bar
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (cx - 6, bar_y - bar_px / 2),
                12,
                bar_px,
                boxstyle="round,pad=0",
                edgecolor="yellow",
                facecolor="yellow",
                alpha=0.95,
                linewidth=0,
            )
        )
        ax.annotate(
            f"R={R:.0f} m\n{bar_px:.1f} px",
            xy=(cx, bar_y),
            xytext=(cx, bar_y + 60),
            ha="center",
            fontsize=10,
            color="yellow",
            weight="bold",
            arrowprops=dict(arrowstyle="-", color="yellow", lw=0.7),
        )
    ax.set_title(
        "Angular size of a 10 cm rock at the binned (1.32 mrad/px) Navcam IFOV\n"
        "Yellow bars = actual pixel height a 10 cm rock would occupy at each range"
    )
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    out_jpg = out.with_suffix(".jpg")
    fig.savefig(out_jpg, dpi=110, bbox_inches="tight", pil_kwargs={"quality": 88})
    if out.suffix == ".png" and out.exists():
        out.unlink()
    plt.close(fig)


def fig_stereo_depth_error(out: Path) -> None:
    """log-log: dZ vs Z. Annotate trust thresholds."""
    f_px = 576.0
    b = NAVCAM_BASELINE_M
    dd = 0.5  # sub-pixel match precision
    Z = np.geomspace(2.0, 200.0, 400)
    dZ = (Z ** 2) * dd / (b * f_px)

    fig, ax = plt.subplots(figsize=(9, 6), dpi=110)
    ax.loglog(Z, dZ, color="tab:red", linewidth=2.2, label=r"$\Delta Z = Z^2 \Delta d / (b f)$")
    ax.axhline(
        ROCK_HEIGHT_THRESHOLD_M,
        color="tab:green",
        linestyle="--",
        linewidth=1.5,
        label="10 cm rock threshold",
    )
    ax.axvline(
        STEREO_TRUST_RANGE_M,
        color="tab:blue",
        linestyle="--",
        linewidth=1.5,
        label=f"stereo trust horizon ({STEREO_TRUST_RANGE_M:.0f} m)",
    )

    # Annotate the table values
    table_pts = [(10, 0.20), (20, 0.82), (30, 1.85), (50, 5.13)]
    for z, dz in table_pts:
        ax.annotate(
            f"{z:.0f} m -> dZ={dz:.2f} m",
            xy=(z, dz),
            xytext=(z * 1.05, dz * 1.7),
            fontsize=8,
            arrowprops=dict(arrowstyle="-", lw=0.5, color="grey"),
        )

    ax.fill_between(Z, dZ, 1e-3, where=(Z <= STEREO_TRUST_RANGE_M), color="tab:green", alpha=0.07)
    ax.fill_between(Z, dZ, 1e3, where=(Z > STEREO_TRUST_RANGE_M), color="tab:orange", alpha=0.07)

    ax.set_xlabel("Range $Z$ (m)")
    ax.set_ylabel(r"Depth uncertainty $\Delta Z$ (m)")
    ax.set_title(
        "Stereo depth error grows as $Z^2$\n"
        f"Navcam binned: $f \\approx 576$ px, $b = {NAVCAM_BASELINE_M}$ m, $\\Delta d = {dd}$ px"
    )
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.text(5, 0.5, "stereo regime", color="tab:green", fontsize=11, weight="bold", alpha=0.8)
    ax.text(60, 12, "monocular regime", color="tab:orange", fontsize=11, weight="bold", alpha=0.85)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def fig_two_regime(out: Path) -> None:
    """Range axis with stereo / mono / overlap zones."""
    fig, ax = plt.subplots(figsize=(11, 3.2), dpi=110)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.add_patch(mpatches.Rectangle((0, 0.4), STEREO_TRUST_RANGE_M, 0.2, color="tab:green", alpha=0.55))
    ax.add_patch(mpatches.Rectangle((STEREO_TRUST_RANGE_M, 0.4), 100 - STEREO_TRUST_RANGE_M, 0.2, color="tab:orange", alpha=0.55))

    for r in [0, 10, 20, 30, 50, 100]:
        ax.axvline(r, ymin=0.35, ymax=0.5, color="black", linewidth=1)
        ax.text(r, 0.32, f"{r} m", ha="center", fontsize=9)

    ax.text(STEREO_TRUST_RANGE_M / 2, 0.5, "stereo (geometric)\n+/- 1-30 cm", ha="center", va="center", fontsize=11, weight="bold", color="white")
    ax.text((STEREO_TRUST_RANGE_M + 100) / 2, 0.5, "monocular (learned)\n+/- 10-50 cm", ha="center", va="center", fontsize=11, weight="bold", color="white")
    ax.text(50, 0.85, "Two-regime detection: stereo teacher (<= 20 m) + mono student (> 20 m)", ha="center", fontsize=12, weight="bold")

    ax.annotate(
        "fusion gate\n(IoU dedup\nin overlap)",
        xy=(STEREO_TRUST_RANGE_M, 0.5),
        xytext=(STEREO_TRUST_RANGE_M, 0.05),
        ha="center",
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
    )

    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def fig_disparity_and_overlay(out_disp: Path, out_overlay: Path) -> None:
    """Run the pipeline once on sol 755 to produce real disparity + overlay PNGs."""
    left_meta = _resolved_meta(SOL_755_LEFT)
    right_meta = _resolved_meta(SOL_755_RIGHT)
    pairs = find_pairs([left_meta, right_meta])
    if not pairs:
        raise RuntimeError("Sol 755 pair not found - check data/relevant/")
    pair = pairs[0]
    left_img = _load_image_rgb(pair.left.image_path)
    right_img = _load_image_rgb(pair.right.image_path)

    result = run_pipeline(
        pair,
        left_img,
        right_img,
        stereo=SGBMMatcher(),
        segmenter=MockSegmenter(min_area=80),
        depth_estimator=MockDepthEstimator(base=10.0, scale=0.05),
        features=MockFeatureExtractor(dim=32, grid=16),
    )

    # Disparity preview
    write_disparity_preview(result.near_artefacts.disparity.disparity, out_disp)

    # Overlay
    overlay = overlay_detections(left_img, result.detections)
    out_jpg = out_overlay.with_suffix(".jpg")
    cv2.imwrite(
        str(out_jpg),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 88],
    )
    if out_overlay.suffix == ".png" and out_overlay.exists():
        out_overlay.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=REPO / "docs" / "media")
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    print(f"writing figures to {out}")

    fig_cahvore_vectors(out / "cahvore-vectors.png")
    print("  cahvore-vectors.png")
    fig_cahvore_distortion(out / "cahvore-distortion-heatmap.png")
    print("  cahvore-distortion-heatmap.png")
    fig_angular_size_overlay(out / "angular-size-overlay.png")
    print("  angular-size-overlay.jpg")
    fig_stereo_depth_error(out / "stereo-depth-error.png")
    print("  stereo-depth-error.png")
    fig_two_regime(out / "two-regime-diagram.png")
    print("  two-regime-diagram.png")
    fig_disparity_and_overlay(
        out / "disparity-sol755.png", out / "overlay-sol755.png"
    )
    print("  disparity-sol755.png  overlay-sol755.jpg")

    print("\nfile sizes:")
    for p in sorted(out.glob("*.png")):
        kb = p.stat().st_size / 1024
        flag = "" if kb < 500 else "  WARN: > 500 KB"
        print(f"  {p.name:40s}  {kb:7.1f} KB{flag}")


if __name__ == "__main__":
    sys.exit(main())
