PY := python
VENV ?= .venv

.PHONY: help setup install test test-gpu lint format type smoke index pseudolabel train eval clean

help:
	@echo "Targets:"
	@echo "  setup        Create venv + install CPU deps"
	@echo "  install-gpu  Install GPU extras (torch cu128, transformers, etc.)"
	@echo "  test         Run pytest (CPU only)"
	@echo "  test-gpu     Run pytest including GPU-marked tests"
	@echo "  lint         Run ruff"
	@echo "  format       Run black"
	@echo "  type         Run mypy"
	@echo "  smoke        One-shot end-to-end smoke on 1 stereo pair"
	@echo "  index        Build stereo pair index"
	@echo "  pseudolabel  Generate pseudolabels (SOLS=100-200)"
	@echo "  train        Train monocular height head"
	@echo "  eval         Evaluate a checkpoint (CKPT=...)"
	@echo "  clean        Remove caches and outputs"

setup:
	bash scripts/setup_env.sh

install-gpu:
	$(VENV)/bin/pip install -e '.[gpu]'

test:
	$(VENV)/bin/pytest -q

test-gpu:
	$(VENV)/bin/pytest -q -m "not slow"

lint:
	$(VENV)/bin/ruff check .

format:
	$(VENV)/bin/black .
	$(VENV)/bin/ruff check --fix .

type:
	$(VENV)/bin/mypy src

smoke:
	bash scripts/run_smoke.sh

index:
	$(VENV)/bin/rock-snitch index data

SOLS ?= 100-200
pseudolabel:
	$(VENV)/bin/rock-snitch pseudolabel --sols $(SOLS) --out outputs/pseudolabels

train:
	$(VENV)/bin/rock-snitch train --labels outputs/pseudolabels --out runs/mono_head_v1

CKPT ?= runs/mono_head_v1/last.ckpt
eval:
	$(VENV)/bin/rock-snitch eval --ckpt $(CKPT) --sols 900-1000

clean:
	find . -type d -name __pycache__ -exec /bin/rm -r {} +
	find . -type d -name .pytest_cache -exec /bin/rm -r {} +
	find . -type d -name .mypy_cache -exec /bin/rm -r {} +
	find . -type d -name .ruff_cache -exec /bin/rm -r {} +
	find . -type d -name '*.egg-info' -exec /bin/rm -r {} +
	/bin/rm -r build dist 2>/dev/null || true
