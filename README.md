# rock-snitch-v2

Detect rocks >=10 cm in the Perseverance rover's drive path from Mars 2020 Navcam imagery.

Two-regime pipeline:

- **Near-field (<= ~20 m)**: stereo rectification (CAHVORE -> CAHV) -> RAFT-Stereo / SGBM disparity -> point cloud -> RANSAC ground plane -> per-mask height.
- **Far-field (> ~20 m)**: SAM2 mask proposals + DINOv2 patch features + UniDepthV2 metric depth -> learned per-mask height regressor trained on stereo pseudolabels.

Fuses branches by range, outputs per-image JSON hit-list of rocks with `(u, v, range_m, height_m, confidence, source)`.

## Quick start (on the 5090 box over SSH)

```bash
git pull
bash scripts/setup_env.sh           # creates .venv, installs torch cu128 + deps
source .venv/bin/activate
python scripts/download_models.py   # fetches SAM2, DINOv2, UniDepthV2, RAFT-Stereo

rock-snitch index data/             # build stereo pair index
rock-snitch detect --sol 150        # run on one sol, write JSON + overlay PNG
rock-snitch pseudolabel --sols 100-500 --out outputs/pseudolabels
rock-snitch train --labels outputs/pseudolabels --out runs/mono_head_v1
rock-snitch eval --ckpt runs/mono_head_v1/last.ckpt --sols 900-1000
```

## Background

Mars 2020 Navcam: stereo 96 deg x 73 deg FOV, 42.4 cm baseline, 0.33 mrad/px native (~5120 x 3840).
Raw feed serves 1280 x 960 tiles that are typically 4x-binned -> effective 1.32 mrad/px.

At 4x-binned resolution a 10-cm rock subtends:

| Range | Pixel height |
|---|---|
| 10 m | ~7.6 |
| 20 m | ~3.8 |
| 30 m | ~2.5 |
| 50 m | ~1.5 |
| 100 m | ~0.76 |

Stereo depth error (dZ = Z^2 * dd / (b*f), b = 0.424 m, f ~ 576 px binned, dd = 0.5 px):

| Range | dZ |
|---|---|
| 10 m | 0.20 m |
| 20 m | 0.82 m |
| 30 m | 1.85 m |

-> Stereo-only height recovery is trustworthy only to ~10-15 m; beyond that we need learned depth.

## Layout

```
src/rocksnitch/
  contracts.py            shared dataclasses + Protocols
  io/                     filename + metadata + CAHVORE parser + pair indexer + Dataset
  geometry/               rectify, disparity, point-cloud, ground plane, height
  perception/             SAM2, DINOv2, GroundingDINO, UniDepthV2 wrappers
  pipeline/               near_field, far_field, fuse, run
  training/               pseudolabel gen + mono-height head + training loop
  eval/                   metrics + visualisation
  cli.py                  Typer CLI: rock-snitch {index,detect,pseudolabel,train,eval,viz}

tests/                    mirrors src/ layout; pytest -q (CPU), pytest -q -m gpu (GPU)
configs/                  default.yaml + models.yaml
scripts/                  setup_env.sh, download_models.py, run_smoke.sh
```

## Data sources

- Raw images + JSON sidecar: `download_navcam.py` (already provided).
- Format: Mars 2020 PDS `raw_images` feed, `NAVCAM_LEFT | NAVCAM_RIGHT`, CAHVORE calibration.
- ~1,501 sols present, ~26,790 verified stereo pairs.

Key references:

- Maki et al. 2020, The Mars 2020 Engineering Cameras and Microphone on the Perseverance Rover.
- Gennery 2001, CAHVOR(E) camera model.
- SAM 2 (facebookresearch/sam2), DINOv2 (facebookresearch/dinov2), UniDepthV2 (lpiccinelli-eth/UniDepth), RAFT-Stereo (princeton-vl/RAFT-Stereo), GroundingDINO (IDEA-Research/GroundingDINO).

## Development

```bash
pip install -e '.[dev]'
pytest -q
ruff check .
black --check .
```

GPU tests require CUDA + model weights; skip automatically when missing.

## License

Apache-2.0.
