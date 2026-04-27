#!/usr/bin/env bash
# Full pipeline: stereo + mono + DINOv2 features + CAHVORE linearization.
# Real GPU models (SAM2, UniDepthV2, DINOv2). Best detection quality.
#
# Requirements:
#   - .venv with [gpu] extras installed (bash scripts/setup_env.sh --gpu)
#   - Model weights downloaded (python scripts/download_models.py)
#   - data/.state/stereo_index.parquet built (rock-snitch index data)
#
# Usage:
#   bash scripts/detect_full.sh <sol> [output_dir]
set -euo pipefail

SOL="${1:?usage: detect_full.sh <sol> [output_dir]}"
OUT="${2:-outputs/sol${SOL}_full}"

. .venv/bin/activate

rock-snitch detect \
    --sol "${SOL}" \
    --output "${OUT}" \
    --profile full
