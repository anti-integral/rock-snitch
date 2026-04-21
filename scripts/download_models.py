#!/usr/bin/env python3
"""Fetch SAM2, DINOv2, UniDepthV2, and RAFT-Stereo weights."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


MODELS_DIR = Path("models")


def _fetch(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        print(f"ok (cached): {dst}")
        return
    print(f"downloading: {url} -> {dst}")
    if not _has("curl"):
        raise RuntimeError("curl not available")
    subprocess.check_call(["curl", "-L", "-o", str(dst), url])


def _has(cmd: str) -> bool:
    from shutil import which

    return which(cmd) is not None


def _hf_snapshot(repo: str) -> None:
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError:
        print("huggingface_hub not available; install .[gpu] extras.", file=sys.stderr)
        sys.exit(1)
    token = os.environ.get("HF_TOKEN")
    snapshot_download(repo_id=repo, token=token, local_dir=str(MODELS_DIR / repo.replace("/", "__")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--cpu-only", action="store_true", help="Skip GPU-only weights")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # SAM2
    if "sam2" in cfg:
        _fetch(cfg["sam2"]["url"], MODELS_DIR / cfg["sam2"]["checkpoint"])

    if args.cpu_only:
        print("Skipping GPU-only HF downloads (--cpu-only).")
        return

    for key in ("dinov2", "unidepth_v2", "grounding_dino"):
        if key in cfg:
            _hf_snapshot(cfg[key]["hf_repo"])

    if "raft_stereo" in cfg:
        _fetch(cfg["raft_stereo"]["url"], MODELS_DIR / cfg["raft_stereo"]["filename"])


if __name__ == "__main__":  # pragma: no cover
    main()
