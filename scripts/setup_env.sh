#!/usr/bin/env bash
# Create .venv and install deps. Pass --gpu for the 5090 setup.
set -euo pipefail

VENV=.venv
PY=${PYTHON:-python3}

if [ ! -d "$VENV" ]; then
    "$PY" -m venv "$VENV"
fi

. "$VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# CPU baseline
pip install -e '.[dev]'

if [[ "${1:-}" == "--gpu" ]]; then
    echo ">>> Installing PyTorch with CUDA 12.8 (Blackwell / sm_120 support) <<<"
    pip install --index-url https://download.pytorch.org/whl/cu128 \
        "torch>=2.6" "torchvision>=0.21"

    echo ">>> Installing GPU extras (transformers, accelerate, timm, tb) <<<"
    pip install -e '.[gpu]'

    echo ">>> Installing SAM 2 from source <<<"
    pip install "git+https://github.com/facebookresearch/sam2.git"

    echo ">>> Installing UniDepthV2 from source <<<"
    pip install "git+https://github.com/lpiccinelli-eth/UniDepth.git"

    echo ">>> (optional) RAFT-Stereo requires manual clone; see README <<<"
fi

echo "Environment ready. Activate with: . $VENV/bin/activate"
echo "Run tests: pytest -q"
