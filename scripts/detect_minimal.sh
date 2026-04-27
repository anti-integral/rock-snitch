#!/usr/bin/env bash
# Minimal pipeline: every backend mocked. No GPU. No model weights.
# Smoke-test only - the detections will not be physically meaningful,
# but the pipeline will run end-to-end and write all expected outputs.
#
# Usage:
#   bash scripts/detect_minimal.sh <sol> [output_dir]
set -euo pipefail

SOL="${1:?usage: detect_minimal.sh <sol> [output_dir]}"
OUT="${2:-outputs/sol${SOL}_minimal}"

. .venv/bin/activate

rock-snitch detect \
    --sol "${SOL}" \
    --output "${OUT}" \
    --profile minimal \
    --no-gpu
