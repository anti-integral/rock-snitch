# AGENTS.md

Repository conventions for this project.

## Project summary

rock-snitch-v2 detects Mars rocks >= 10 cm in Perseverance Navcam imagery. Two-regime
pipeline: stereo teacher (<= 20 m) plus monocular-depth student (> 20 m).

## Code conventions

- Python 3.11. Type-annotate every function signature.
- `ruff` + `black` must pass: `ruff check . && black --check .`.
- `pytest -q` must pass on CPU; GPU-only tests marked `@pytest.mark.gpu`.
- Units: SI (metres, radians) unless explicit.
- Pure functions where possible. Keep I/O out of core math.
- Shared dataclasses live in `src/rocksnitch/contracts.py`; import from there, do not re-define.
- Every public function has a docstring with shape/dtype/units for numpy arrays.
- Prefer frozen dataclasses for value objects.

## Dependency rule

Only add top-level deps via `pyproject.toml`. Allowed base stack: numpy, scipy, opencv-python,
torch, torchvision, transformers, einops, pyarrow, typer, rich, pydantic, pyyaml, tqdm,
scikit-image, scikit-learn, pillow. Model-specific deps (SAM2, UniDepth, DINOv2, RAFT-Stereo,
GroundingDINO) go behind an import-guard so CPU-only smoke tests pass without them.

## Test layout

- `tests/` mirrors `src/rocksnitch/` layout.
- Fixtures in `tests/conftest.py` and `tests/fixtures/`.
- Use `tmp_path` fixture for any FS work.
- Mark slow tests with `@pytest.mark.slow`; GPU tests with `@pytest.mark.gpu`;
  weight-dependent with `@pytest.mark.models`.

## Run layout

- Model weights: `./models/` (gitignored).
- Run outputs: `./outputs/`, `./runs/` (gitignored).
- Derived index: `data/.state/stereo_index.parquet` (gitignored).
