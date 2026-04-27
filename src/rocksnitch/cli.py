"""Command-line interface for rock-snitch."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.table import Table

from rocksnitch import __version__
from rocksnitch.io.dataset import StereoPairDataset
from rocksnitch.io.metadata import iter_meta
from rocksnitch.io.pairing import find_pairs, save_index
from rocksnitch.logging_utils import get_logger

app = typer.Typer(help="Mars Navcam rock detection >=10 cm. See README for usage.", add_completion=False)
console = Console()
log = get_logger("rocksnitch.cli")


def _load_config(path: Optional[Path]) -> dict:
    if path is None:
        default = Path(__file__).resolve().parent.parent.parent / "configs" / "default.yaml"
        if default.exists():
            return yaml.safe_load(default.read_text()) or {}
        return {}
    return yaml.safe_load(Path(path).read_text()) or {}


def _version_cb(value: bool) -> None:
    if value:
        console.print(__version__)
        raise typer.Exit()


@app.callback()
def _root(
    version: bool = typer.Option(
        None, "--version", callback=_version_cb, is_eager=True, help="Print version and exit."
    ),
) -> None:
    _ = version


@app.command()
def fetch(
    data_root: Path = typer.Argument(Path("data"), help="Output root for images/, metadata/, .state/"),
    command: str = typer.Option("all", "--cmd", help="Phase to run: index | download | all"),
    index_workers: int = typer.Option(1, "--index-workers"),
    download_workers: int = typer.Option(8, "--download-workers"),
) -> None:
    """Download Mars 2020 Navcam L/R raw images and metadata from the public PDS feed."""
    from rocksnitch.fetch import fetch as _fetch

    if command not in {"index", "download", "all"}:
        raise typer.BadParameter("--cmd must be index, download, or all")
    _fetch(
        data_root=data_root,
        command=command,
        index_workers=index_workers,
        download_workers=download_workers,
    )


@app.command()
def index(
    data_root: Path = typer.Argument(Path("data"), help="Root containing images/ and metadata/"),
    out: Path = typer.Option(Path("data/.state/stereo_index.parquet"), "--out", help="Output parquet path"),
    max_sclk_delta_s: float = typer.Option(5.0, "--max-sclk-delta-s"),
    max_pointing_delta_deg: float = typer.Option(0.5, "--max-pointing-delta-deg"),
    limit_sols: Optional[int] = typer.Option(None, "--limit-sols", help="Cap scan to N sols"),
) -> None:
    """Build the stereo-pair index."""
    metas = iter_meta(data_root)
    if limit_sols is not None:
        sols = sorted({m.sol for m in metas})[:limit_sols]
        metas = [m for m in metas if m.sol in sols]
    log.info("Loaded %d ImageMeta records", len(metas))
    pairs = find_pairs(
        metas,
        max_sclk_delta_s=max_sclk_delta_s,
        max_pointing_delta_deg=max_pointing_delta_deg,
    )
    log.info("Found %d stereo pairs across %d sols", len(pairs), len({p.left.sol for p in pairs}))
    save_index(pairs, out)
    console.print(f"Wrote index to [bold]{out}[/bold]")


def _build_backends(config: dict, *, profile, use_gpu: bool):
    """Instantiate backends per ``profile`` choices.

    Returns (stereo, segmenter, depth, features). Any of these can be ``None``
    if the profile disables that branch.
    """
    from rocksnitch.profiles import ProfileSpec

    p: ProfileSpec = profile

    # Stereo matcher
    stereo = None
    if p.pipeline.enable_stereo and p.stereo != "none":
        if p.stereo == "raft" and use_gpu:
            from rocksnitch.geometry.disparity import RaftStereoMatcher

            ckpt = Path("models") / config.get("stereo", {}).get(
                "raft", {}
            ).get("checkpoint", "raftstereo-middlebury.pth")
            stereo = RaftStereoMatcher(ckpt, device="cuda")
        else:
            from rocksnitch.geometry.disparity import SGBMMatcher

            stereo = SGBMMatcher()

    # Segmenter
    if p.segmenter == "mock" or not use_gpu:
        from rocksnitch.perception.sam2 import MockSegmenter

        segmenter = MockSegmenter(min_area=20)
    else:
        from rocksnitch.perception.sam2 import SAM2Config, SAM2Segmenter

        sam_cfg = SAM2Config(
            checkpoint=Path("models")
            / config.get("segmenter", {})
            .get("sam2", {})
            .get("checkpoint", "sam2_hiera_large.pt"),
            config_name=config.get("segmenter", {})
            .get("sam2", {})
            .get("config", "sam2_hiera_l"),
            device="cuda",
        )
        segmenter = SAM2Segmenter(sam_cfg)

    # Depth estimator
    depth = None
    if p.pipeline.enable_mono and p.depth != "none":
        if p.depth == "mock" or not use_gpu:
            from rocksnitch.perception.mono_depth import MockDepthEstimator

            depth = MockDepthEstimator()
        else:
            from rocksnitch.perception.mono_depth import UniDepthConfig, UniDepthV2

            depth = UniDepthV2(UniDepthConfig(device="cuda"))

    # Feature extractor
    features = None
    if p.features != "none":
        if p.features == "mock" or not use_gpu:
            from rocksnitch.perception.dinov2 import MockFeatureExtractor

            features = MockFeatureExtractor(dim=32, grid=16)
        else:
            from rocksnitch.perception.dinov2 import (
                DINOv2Config,
                DINOv2FeatureExtractor,
            )

            features = DINOv2FeatureExtractor(DINOv2Config(device="cuda"))

    return stereo, segmenter, depth, features


@app.command()
def detect(
    image: Optional[Path] = typer.Option(None, "--image", help="Left image path (infers right)"),
    sol: Optional[int] = typer.Option(None, "--sol", help="Run detect on one sol using the stereo index"),
    limit: int = typer.Option(1, "--limit", help="Max pairs to run"),
    output: Path = typer.Option(Path("outputs"), "--output"),
    data_root: Path = typer.Option(Path("data"), "--data-root"),
    index_path: Path = typer.Option(Path("data/.state/stereo_index.parquet"), "--index"),
    config_path: Optional[Path] = typer.Option(None, "--config"),
    profile_name: str = typer.Option(
        "full",
        "--profile",
        help="Preset bundle: full | stereo-only | mono-only | minimal | mock",
    ),
    use_gpu: bool = typer.Option(True, "--use-gpu/--no-gpu"),
    no_overlay: bool = typer.Option(False, "--no-overlay"),
    # Branch toggles (override profile)
    enable_stereo: Optional[bool] = typer.Option(
        None, "--stereo/--no-stereo", help="Override profile's stereo branch toggle"
    ),
    enable_mono: Optional[bool] = typer.Option(
        None, "--mono/--no-mono", help="Override profile's mono branch toggle"
    ),
    enable_linearize: Optional[bool] = typer.Option(
        None,
        "--linearize/--no-linearize",
        help="Apply CAHVORE distortion correction. Off by default; on costs ~50ms/frame.",
    ),
    # Backend overrides
    segmenter_choice: Optional[str] = typer.Option(
        None, "--segmenter", help="sam2 | mock"
    ),
    depth_choice: Optional[str] = typer.Option(
        None, "--depth", help="unidepth | mock | none"
    ),
    features_choice: Optional[str] = typer.Option(
        None, "--features", help="dinov2 | mock | none"
    ),
    stereo_choice: Optional[str] = typer.Option(
        None, "--stereo-backend", help="sgbm | raft | none"
    ),
) -> None:
    """Run detection on a stereo pair (or a sol's worth).

    Examples:

        # quick smoke test, no GPU, no weights:
        rock-snitch detect --sol 100 --profile minimal

        # full pipeline on the 5090 (default profile):
        rock-snitch detect --sol 755

        # geometry-only, fast: no mono depth, no learned head:
        rock-snitch detect --sol 755 --profile stereo-only

        # mono branch only, single-image path:
        rock-snitch detect --image data/relevant/NLF_0395_..png --profile mono-only

        # full pipeline but skip CAHVORE linearization (faster, slightly less accurate):
        rock-snitch detect --sol 755 --no-linearize
    """
    from rocksnitch.eval.viz import write_disparity_preview, write_overlay
    from rocksnitch.pipeline.run import run_pipeline, write_detections_json
    from rocksnitch.profiles import get_profile, list_profiles

    if profile_name not in list_profiles():
        raise typer.BadParameter(
            f"--profile must be one of {list_profiles()}, got {profile_name!r}"
        )
    profile = get_profile(profile_name)

    # Apply per-flag overrides
    if enable_stereo is not None:
        profile.pipeline.enable_stereo = enable_stereo
    if enable_mono is not None:
        profile.pipeline.enable_mono = enable_mono
    if enable_linearize is not None:
        profile.pipeline.enable_linearize = enable_linearize
    if segmenter_choice is not None:
        profile.segmenter = segmenter_choice  # type: ignore[assignment]
    if depth_choice is not None:
        profile.depth = depth_choice  # type: ignore[assignment]
    if features_choice is not None:
        profile.features = features_choice  # type: ignore[assignment]
    if stereo_choice is not None:
        profile.stereo = stereo_choice  # type: ignore[assignment]

    config = _load_config(config_path)
    log.info(profile.explain())
    stereo, segmenter, depth, features = _build_backends(
        config, profile=profile, use_gpu=use_gpu
    )

    output.mkdir(parents=True, exist_ok=True)
    if image is not None and sol is None:
        raise typer.Exit("single-image detect not yet implemented; pass --sol")
    if sol is None:
        raise typer.BadParameter("Pass either --image or --sol")
    if not index_path.exists():
        raise typer.BadParameter(
            f"Index not found at {index_path}. Run `rock-snitch index` first."
        )
    dataset = StereoPairDataset(index_path)
    selected = [i for i, p in enumerate(dataset.pairs()) if p.left.sol == sol][:limit]
    if not selected:
        raise typer.Exit(f"No pairs for sol {sol}")
    for idx in selected:
        sample = dataset[idx]
        console.print(f"[bold]Sol {sol}[/bold] pair {sample.pair.left.imageid}")
        res = run_pipeline(
            sample.pair,
            sample.left_image,
            sample.right_image,
            stereo=stereo,
            segmenter=segmenter,
            depth_estimator=depth,
            features=features,
            config=profile.pipeline,
        )
        stem = sample.pair.left.imageid
        write_detections_json(res, output / f"{stem}.json")
        if not no_overlay:
            write_overlay(sample.left_image, res.detections, output / f"{stem}.overlay.png")
            if res.near_artefacts is not None:
                write_disparity_preview(
                    res.near_artefacts.disparity.disparity,
                    output / f"{stem}.disparity.png",
                )
        console.print(
            f"  detections={len(res.detections)}  branches={','.join(res.branches_run) or 'none'}  outputs in {output}"
        )


@app.command()
def pseudolabel(
    sols: str = typer.Option(..., "--sols", help="Inclusive sol range, e.g. 100-500"),
    out: Path = typer.Option(Path("outputs/pseudolabels/pseudolabels.jsonl"), "--out"),
    index_path: Path = typer.Option(Path("data/.state/stereo_index.parquet"), "--index"),
    config_path: Optional[Path] = typer.Option(None, "--config"),
    profile_name: str = typer.Option("stereo-only", "--profile"),
    use_gpu: bool = typer.Option(True, "--use-gpu/--no-gpu"),
    limit: int = typer.Option(0, "--limit", help="Cap number of pairs (0 = all)"),
) -> None:
    """Generate stereo-derived pseudolabels."""
    from rocksnitch.profiles import get_profile, list_profiles
    from rocksnitch.training.pseudolabel import (
        PseudolabelConfig,
        generate_pseudolabels,
        write_pseudolabels,
    )

    if profile_name not in list_profiles():
        raise typer.BadParameter(f"--profile must be one of {list_profiles()}")
    profile = get_profile(profile_name)
    # Pseudolabeling needs DINOv2 features for the head's training set
    if profile.features == "none":
        profile.features = "dinov2" if use_gpu else "mock"

    config = _load_config(config_path)
    stereo, segmenter, _depth, features = _build_backends(
        config, profile=profile, use_gpu=use_gpu
    )
    if not index_path.exists():
        raise typer.BadParameter(f"Index not found at {index_path}. Run `rock-snitch index` first.")

    lo, hi = (int(x) for x in sols.split("-"))
    dataset_all = StereoPairDataset(index_path)
    pairs = [p for p in dataset_all.pairs() if lo <= p.left.sol <= hi]
    if limit > 0:
        pairs = pairs[:limit]
    log.info("Generating pseudolabels for %d pairs (sols %d..%d)", len(pairs), lo, hi)

    sub_ds = StereoPairDataset(pairs)
    records = generate_pseudolabels(
        sub_ds,
        stereo=stereo,
        segmenter=segmenter,
        features=features,
        config=PseudolabelConfig(),
    )
    count = write_pseudolabels(records, out)
    console.print(f"Wrote {count} pseudolabels to {out}")


@app.command()
def train(
    labels: Path = typer.Option(Path("outputs/pseudolabels/pseudolabels.jsonl"), "--labels"),
    out: Path = typer.Option(Path("runs/mono_head_v1"), "--out"),
    max_epochs: int = typer.Option(40, "--max-epochs"),
    batch_size: int = typer.Option(128, "--batch-size"),
    lr: float = typer.Option(1e-3, "--lr"),
) -> None:
    """Train the monocular height head on pseudolabels."""
    from rocksnitch.training.train import TrainConfig, train_head

    cfg = TrainConfig(lr=lr, batch_size=batch_size, max_epochs=max_epochs)
    ckpt = train_head(labels, out, config=cfg)
    console.print(f"Saved checkpoint to {ckpt}")


@app.command("eval")
def eval_cmd(
    ckpt: Path = typer.Option(..., "--ckpt"),
    preds: Path = typer.Option(..., "--preds", help="Directory of *.json predictions"),
    labels: Path = typer.Option(..., "--labels", help="pseudolabels.jsonl acting as ground truth"),
    out: Path = typer.Option(Path("outputs/eval"), "--out"),
) -> None:
    """Evaluate predictions against pseudolabels."""
    from rocksnitch.contracts import RockDetection
    from rocksnitch.eval.metrics import mean_height_error, range_binned_pr
    from rocksnitch.training.pseudolabel import read_pseudolabels

    gt_recs = read_pseudolabels(labels)
    gt = [
        RockDetection(
            uv_bbox=tuple(r.uv_bbox),  # type: ignore[arg-type]
            mask_rle=None,
            centroid_uv=tuple(r.centroid_uv),  # type: ignore[arg-type]
            range_m=r.range_m,
            height_m=r.height_m,
            width_m=r.width_m,
            confidence=r.confidence,
            source="stereo",
        )
        for r in gt_recs
    ]
    pred: list[RockDetection] = []
    for f in Path(preds).glob("*.json"):
        payload = json.loads(f.read_text())
        for d in payload.get("detections", []):
            pred.append(
                RockDetection(
                    uv_bbox=tuple(d["uv_bbox"]),  # type: ignore[arg-type]
                    mask_rle=d.get("mask_rle"),
                    centroid_uv=tuple(d["centroid_uv"]),  # type: ignore[arg-type]
                    range_m=float(d["range_m"]),
                    height_m=float(d["height_m"]),
                    width_m=float(d["width_m"]),
                    confidence=float(d["confidence"]),
                    source=d["source"],
                )
            )
    report = range_binned_pr(pred, gt)
    mhe = mean_height_error(pred, gt)
    table = Table(title=f"range-binned P/R @ >=10 cm (MHE={mhe*100:.1f} cm)")
    table.add_column("range (m)")
    table.add_column("P")
    table.add_column("R")
    table.add_column("TP/FP/FN")
    for (lo, hi), r in report.items():
        table.add_row(f"{lo}-{hi}", f"{r.precision:.2f}", f"{r.recall:.2f}", f"{r.tp}/{r.fp}/{r.fn}")
    console.print(table)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(
        json.dumps(
            {
                "mean_height_error_m": mhe,
                "bins": [
                    {"lo": lo, "hi": hi, "precision": r.precision, "recall": r.recall, "tp": r.tp, "fp": r.fp, "fn": r.fn}
                    for (lo, hi), r in report.items()
                ],
            },
            indent=2,
        )
    )
    _ = ckpt  # reserved for future checkpoint-conditioned eval


@app.command()
def viz(
    image: Path = typer.Argument(...),
    detections: Path = typer.Argument(...),
    out: Path = typer.Option(Path("outputs/viz.png"), "--out"),
) -> None:
    """Overlay a detections JSON on an image."""
    import cv2

    from rocksnitch.contracts import RockDetection
    from rocksnitch.eval.viz import write_overlay

    bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if bgr is None:
        raise typer.Exit(f"Failed to read {image}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    payload = json.loads(detections.read_text())
    dets = [
        RockDetection(
            uv_bbox=tuple(d["uv_bbox"]),
            mask_rle=d.get("mask_rle"),
            centroid_uv=tuple(d["centroid_uv"]),
            range_m=float(d["range_m"]),
            height_m=float(d["height_m"]),
            width_m=float(d["width_m"]),
            confidence=float(d["confidence"]),
            source=d["source"],
        )
        for d in payload.get("detections", [])
    ]
    write_overlay(rgb, dets, out)
    console.print(f"Wrote overlay to {out}")


@app.command()
def ui(
    share: bool = typer.Option(False, "--share", help="Expose a public Gradio link"),
    port: int = typer.Option(7860, "--port"),
    host: str = typer.Option("127.0.0.1", "--host"),
) -> None:
    """Launch the Gradio web UI."""
    from rocksnitch.app import build_app

    build_app().launch(server_name=host, server_port=port, share=share)


@app.command()
def profiles() -> None:
    """List available pipeline profiles and their summaries."""
    from rocksnitch.profiles import get_profile, list_profiles

    table = Table(title="rock-snitch detection profiles")
    table.add_column("name")
    table.add_column("description")
    table.add_column("config")
    for name in list_profiles():
        p = get_profile(name)
        table.add_row(name, p.description, p.explain())
    console.print(table)


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
