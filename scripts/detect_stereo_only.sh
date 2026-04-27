#!/usr/bin/env bash
# Geometry-only pipeline: stereo (SGBM disparity + ground plane + height).
# No mono depth, no learned head. SAM2 still runs for masks.
#
# Useful when:
#   - You only care about close-range (<= 20 m) detections
#   - You want maximum trustworthiness from stereo geometry alone
#   - You haven't trained a mono-height head yet
#
# Usage:
#   bash scripts/detect_stereo_only.sh <sol> [output_dir]
set -euo pipefail

SOL="${1:?usage: detect_stereo_only.sh <sol> [output_dir]}"
OUT="${2:-outputs/sol${SOL}_stereo}"

. .venv/bin/activate

rock-snitch detect \
    --sol "${SOL}" \
    --output "${OUT}" \
    --profile stereo-only
