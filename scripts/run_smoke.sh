#!/usr/bin/env bash
# End-to-end smoke test: run detect on a single stereo pair with mocked models.
set -euo pipefail

VENV=.venv
. "$VENV/bin/activate"

rock-snitch --version
rock-snitch index data/ --limit-sols 2 --out data/.state/stereo_index.parquet
rock-snitch detect --sol 100 --limit 1 --mock-models --output outputs/smoke
echo "Smoke test complete. See outputs/smoke/"
