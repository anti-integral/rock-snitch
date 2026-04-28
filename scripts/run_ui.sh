#!/usr/bin/env bash
# Launch the rock-snitch Gradio web UI.
#
# Auto-detects the best available device (CUDA > Apple MPS > CPU) and
# starts with the "full" profile (all features on). Override either via
# environment variables or extra arguments — anything after the script
# name is forwarded directly to `rock-snitch ui`.
#
# Examples:
#   bash scripts/run_ui.sh                      # auto-device, full profile
#   bash scripts/run_ui.sh --device cpu         # force CPU
#   bash scripts/run_ui.sh --profile mono-only  # different starting profile
#   bash scripts/run_ui.sh --share              # public Gradio link
#
# Env vars:
#   ROCKSNITCH_PORT   port (default 7860)
#   ROCKSNITCH_HOST   host (default 127.0.0.1)
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d ".venv" ]; then
    echo "No .venv found. Run: bash scripts/setup_env.sh" >&2
    exit 1
fi
. .venv/bin/activate

PORT="${ROCKSNITCH_PORT:-7860}"
HOST="${ROCKSNITCH_HOST:-127.0.0.1}"

echo ">>> rock-snitch UI on http://${HOST}:${PORT}"
echo ">>> auto-detecting device (cuda > mps > cpu); pass --device to override"

exec rock-snitch ui --host "${HOST}" --port "${PORT}" "$@"
