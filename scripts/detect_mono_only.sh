#!/usr/bin/env bash
# Mono-only pipeline: SAM2 + UniDepthV2 + DINOv2 features.
# No stereo branch. Use when you only have a single image (no stereo
# partner) or want to benchmark the learned mono path in isolation.
#
# Usage:
#   bash scripts/detect_mono_only.sh <sol> [output_dir]
set -euo pipefail

SOL="${1:?usage: detect_mono_only.sh <sol> [output_dir]}"
OUT="${2:-outputs/sol${SOL}_mono}"

. .venv/bin/activate

rock-snitch detect \
    --sol "${SOL}" \
    --output "${OUT}" \
    --profile mono-only
