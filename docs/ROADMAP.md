# rock-snitch-v2 — Roadmap

Prioritised follow-up work, drawn from a comparison against the parallel
implementation at `github.com/anti-integral/rock-snitch-v2`. Each item lists
**what**, **why**, **integration path**, **rough size**, and a **status**.

> Status legend:
> - `planned` — agreed but not started
> - `in-progress` — being worked on now
> - `done` — landed on `main`
> - `parked` — explicitly deferred

---

## Tier 1 — borrow these (high value, low risk)

### T1.1 — Drive-corridor scoping → `pipeline/corridor.py`

**Status**: `planned`

**What.** A forward fan in site frame (configurable: ±3 m lateral × 1–40 m
forward, plus a 1 m start offset to skip the deck) projected back to the
image. Each detection gets two new fields: `in_corridor: bool` and
`is_hazard: bool`. Detections outside the corridor are *kept but not flagged*
— their scientific value isn't lost.

**Why.** Single most mission-relevant feature in theirs that we don't have.
ENav doesn't care about a 15 cm rock 25 m off-axis; it cares about the ones
in the wheel path. This turns our "rock list" into a "hazard list" with the
right semantics.

**Integration.**
- New module `src/rocksnitch/pipeline/corridor.py` — ~200 LoC.
- New dataclass `CorridorMask` in `contracts.py` (image-space bool array +
  site-frame bounds).
- `pipeline/run.py:run_pipeline` calls `build_corridor()` after the ground
  plane is fit, then tags each `RockDetection` post-fusion.
- `RockDetection` gains `in_corridor: bool` and `is_hazard: bool` fields
  (default `False`, `False` for backward compatibility with existing JSON).
- `eval/viz.py:overlay_detections` colours hazards green, in-corridor
  non-hazards yellow, out-of-corridor blue.

**Risks.**
- Depends on a correct rover-heading extraction from the attitude quaternion.
  Unit-test against the sol 100 fixture before integrating.
- The image-space corridor mask costs one back-projection per pixel; cache
  per (camera, plane) pair to keep `detect` ≤ 2 s.

**Source for cribbing.**
- `theirs/src/rocksnitch/corridor.py` (366 LoC, half of which is fallback
  paths we don't need because we already have `GroundPlane`).

---

### T1.2 — Inverse-variance height fusion → upgrade `pipeline/fuse.py`

**Status**: `planned`

**What.** Replace `final_height = max(h_geom, h_learned)` with proper
Bayesian fusion:

```
1/sigma^2_fused = SUM_i 1/sigma^2_i
h_fused         = sigma^2_fused * SUM_i (h_i / sigma^2_i)
```

Plus an `agreement` score that down-weights confidence when sources disagree:

```
agreement   = 1 - range(h_i) / (|mean(h_i)| + 0.01)   (clamped to [0,1])
confidence  = min(1, agreement * mask_score
                       * (class_prob or 1)
                       * (1 - clamp(resolution_floor / max(h, 0.01), 0, 0.9)))
```

**Why.** Principled. The current `max` silently throws away information from
a low-but-confident estimate when a high-but-noisy one wins. With per-source
sigmas, a stereo measurement at 8 m (sigma ~ 2 cm) and a mono measurement at
the same point (sigma ~ 30 cm) get the right combined answer.

**Integration.**
- Add `sigma_m: float` to `HeightEstimate` (new dataclass) and require each
  branch to emit one:
  - **Stereo**: `sigma = Z^2 * delta_d / (b * f)` from the depth-error
    formula already in `DESIGN.md §4`. `delta_d` from sub-pixel matching
    confidence (~0.5 px in textured regions, ~2 px on sand).
  - **Learned head**: train a small uncertainty head alongside (Gaussian
    output: mean + log-variance), or default to a fixed `sigma = 0.20 *
    h_pred` if we don't want to retrain immediately.
  - **Shadow** (T1.5): `sigma_h = sqrt((tan(theta) * sigma_L)^2 +
    (L * sec^2(theta) * sigma_theta)^2)`.
- New module `src/rocksnitch/geometry/fuse_heights.py` (~80 LoC).
- `pipeline/fuse.py` calls it instead of taking `max`.

**Source.**
- `theirs/src/rocksnitch/fuse.py` (163 LoC) and
  `theirs/docs/03-math-appendix.md §7`.

---

### T1.3 — Hazard rule with uncertainty → 1-line addition under T1.2

**Status**: `planned`

**What.** `is_hazard <=> h_fused - sigma_fused >= 0.10 m` instead of
`h >= 0.10`.

**Why.** A 12 cm height with sigma=4 cm shouldn't be flagged when an 11.8 cm
height with sigma=0.5 cm should. ENav-grade reasoning. Cheap.

**Integration.** One conditional in `pipeline/fuse.py` after T1.2 lands.

---

### T1.4 — Sun position from SCLK → `perception/ephemeris.py`

**Status**: `planned`

**What.** `(sun_az, sun_el)` at acquisition time, two backends:

1. **Primary**: NAIF SPICE kernels via `spiceypy`. Sub-arcsecond accuracy.
   Requires kernel download (~50 MB).
2. **Fallback**: Allison & McEwen 2000 analytical solar-position formula.
   ~1° accuracy. Pure-Python, no dependency.

Output dataclass `SolarGeometry(sun_azimuth_site_deg, sun_elevation_deg,
sun_direction_ned: np.ndarray, source: Literal["spice","allison-mcewen"])`.

**Why.** Gates T1.5 (shadow height). Also useful as metadata enrichment:
shadow length and sun angle are independent signals worth recording even if
we don't use them for detection.

**Integration.**
- New module `src/rocksnitch/perception/ephemeris.py` (~250 LoC).
- Pure function: `solar_geometry(metadata: ImageMeta) -> SolarGeometry`. Zero
  coupling to the rest of the pipeline.
- New optional dep: `spiceypy>=6` under a new `[project.optional-dependencies]`
  section `ephemeris`.
- Kernel download script: `scripts/download_kernels.sh` fetching from the
  NAIF M2020 archive.

**Source.**
- `theirs/src/rocksnitch/ephemeris.py` (304 LoC). The fallback formula is
  ~80 LoC of pure Python — port that first; add SPICE later behind an
  import-guard.

---

### T1.5 — Shadow-based height → `geometry/shadow.py`

**Status**: `planned`

**What.** Per rock mask: find the attached shadow (dark connected region
down-sun of the base), measure shadow length on the ground plane, then
`h = L * tan(sun_el)`. Returns a third independent `HeightEstimate` that
flows into the fusion (T1.2).

**Why.** A *physically independent* signal that doesn't share noise sources
with stereo or mono-depth. Particularly strong at long range where the
shadow is large but the rock is sub-pixel — exactly our weakest regime.
Three-signal fusion is meaningfully more robust than two-signal.

**Integration.**
- New module `src/rocksnitch/geometry/shadow.py` (~150 LoC).
- New dataclass `ShadowMask(mask, tip_uv, base_uv, length_m, length_sigma_m)`
  in `contracts.py`.
- Two-step:
  1. Detect: SAM2 prompt at the down-sun edge of the rock mask + threshold
     on grayscale value (shadows on Mars are reliably the darkest connected
     region adjacent to a rock).
  2. Measure: back-project tip pixel onto ground plane, take Euclidean
     distance to the rock's base pixel.
- Hook into `pipeline/near_field.py` (and a parallel call in `far_field.py`)
  emitting one `HeightEstimate` per detection with `source="shadow"`.

**Risks.**
- Shadow segmentation breaks at high noon (sun_el > 80°). Gate by
  `sun_el < 75°`; emit a `nan` height with `sigma=inf` otherwise so the
  fusion ignores it.
- Self-shadowing on tall, narrow rocks. Apply a max-aspect filter before
  trusting the result.

**Source.**
- `theirs/src/rocksnitch/geometry.py:h_shadow` (~50 LoC of math).

---

## Tier 2 — borrow if time permits

### T2.1 — Gradio web UI → `src/rocksnitch/app.py`

**Status**: `in-progress` (Opus 4.7 sub-agent landing now)

**What.** Single-file Gradio Blocks app: drag-drop a left Navcam PNG + JSON
sidecar, click Detect, see overlay + detections table + disparity preview +
downloadable JSON. Lazy backend loading; works in mock mode without GPU.

**Why.** Demo-ability. For sharing with science / mission planners without
making them install the package or learn the CLI.

**Integration.**
- New module `src/rocksnitch/app.py` (~200 LoC).
- New CLI subcommand `rock-snitch ui --port --share`.
- New optional dep group `ui = ["gradio>=4.36", "pandas>=2"]`.

**Source.** `theirs/src/rocksnitch/app.py` (381 LoC, mostly UI plumbing we'd
rewrite anyway).

---

### T2.2 — Dense corridor heightmap output → `pipeline/terrain.py`

**Status**: `planned`

**What.** Per-pixel `.npz` of `{u, v, xyz_site, h_above_ground}` for all
corridor pixels (`stride` configurable, default 4). Useful for downstream
path-planning consumers.

**Why.** An additional deliverable type. Doesn't replace the JSON hit-list;
complements it. Especially useful for the science team's traversability
analyses that work over heightmaps rather than rock lists.

**Integration.**
- Depends on T1.1 (corridor) + the existing point cloud.
- New module `src/rocksnitch/pipeline/terrain.py` (~150 LoC).
- Add `--write-heightmap` flag to `rock-snitch detect`.

**Note.** Ours can be denser/more accurate than theirs because we have stereo
disparity, not just mono-depth.

**Source.** `theirs/src/rocksnitch/terrain.py` (266 LoC).

---

### T2.3 — CAHVORE-Gennery-2006 closed form → upgrade `io/cahvore.py`

**Status**: `planned`

**What.** Replace our 2001-style iterative distortion approximation with the
2006 closed-form path for `linearity = 2` (which is what every Mars2020
sidecar uses).

**Why.** More accurate at frame corners, where distortion is biggest.
Eliminates the "couple of pixels at the edges" caveat in `DESIGN.md §12`.

**Integration.**
- Drop-in replacement for `_cahvore_distort()` in `io/cahvore.py` (~80 LoC).
- Add a regression test: round-trip
  `project(unproject(uv)) == uv` to within 0.05 px across a 100×100 grid.

**Source.** `theirs/src/rocksnitch/cahvore.py` (338 LoC). Gennery 2006 paper.

---

## Tier 3 — nice to have

### T3.1 — Documentation split → `docs/`

**Status**: `planned`

**What.** Break `docs/DESIGN.md` (one 1,127-line monolith) into navigable
units:

```
docs/00-overview.md             - what + why (current §1)
docs/01-data.md                 - data shape (§2)
docs/02-geometry.md             - the angular-size + Z^2 maths (§3-§4)
docs/03-pipeline.md             - near-field + far-field (§6-§7)
docs/04-teacher-student.md      - the pseudolabel insight (§8-§10)
docs/05-fusion.md               - range gating + IV fusion + hazard rule (§11)
docs/06-cahvore.md              - the camera math (§12)
docs/07-testing.md              - mocks, protocols (§13)
docs/08-walkthrough.md          - end-to-end example (§14)
docs/09-roadmap-future.md       - this file effectively becomes 09
```

**Why.** Cross-referencing is easier when each topic is its own file; users
don't need to scroll past 700 lines to get to the one section they want.

---

### T3.2 — Demo notebooks → `notebooks/`

**Status**: `planned`

**What.** Three Jupyter notebooks:

```
notebooks/01-explore-data.ipynb       - poke at metadata, plot mast pointing histograms
notebooks/02-cahvore-sanity.ipynb     - linearize a real frame, project synthetic
                                        points, show distortion magnitude
notebooks/03-end-to-end-demo.ipynb    - run the pipeline on one stereo pair, show
                                        every artefact
```

**Why.** Onboarding. For new contributors and for the demo deck.

---

### T3.3 — Diff `frames.py` for attitude-quaternion correctness

**Status**: `planned`

**What.** Their git history has a commit `482f4b3 fix(v2): correct attitude
quaternion convention + gate hazards by corridor` — they hit a bug worth
checking against ours.

Our quaternion handling is in `io/cahvore.py:extrinsics_matrix` and `io/
metadata.py` (the attitude is parsed as `(w, x, y, z)`). Their fix likely
relates to `(w, x, y, z)` vs `(x, y, z, w)` convention or the sense of the
rotation (rover-into-site vs site-into-rover).

**Why.** A latent bug here would silently corrupt every site-frame point
cloud. Cheap to verify.

**Integration.**
- Read `theirs/src/rocksnitch/frames.py` (169 LoC).
- Compare against our `io/cahvore.py:extrinsics_matrix` and the JSON-attitude
  sense in `io/metadata.py`.
- Add a regression test that triangulates a known synthetic 3D point in
  rover-nav, transforms to site, and re-projects through both eyes.

---

## What we explicitly skip

- **Their flat module layout** (`src/rocksnitch/*.py`). Our layered layout is
  more navigable; flattening 30+ modules into one directory regresses
  readability.
- **Their `pipeline.py`**. Replacing our two-regime pipeline with their
  one-regime would discard the stereo branch entirely. No.
- **Their `propose.py` / `render.py`**. We already have `perception/sam2.py`
  + `eval/viz.py`; duplicates aren't useful.
- **Their `classify.py`**. They themselves call it vestigial.
- **Forking their entire repo and merging upstream into theirs**. Cleaner to
  keep ours independent and pull in specific modules at our own cadence.

---

## Suggested execution order

Tier 1 cluster (≈ 1 working week):

```
Day 1   T1.4 ephemeris.py             pure function, zero deps
Day 1   T1.2 IV fusion + sigmas       extend RockDetection
Day 2   T1.3 hazard rule              1-line under T1.2
Day 2   T1.1 corridor.py              needs T1.2 for hazard tagging
Day 3   T1.5 shadow.py                needs T1.4 + T1.2
Day 4   regression + DESIGN.md update document the new schema
Day 5   buffer / fix edges
```

After Tier 1 lands, T2.2 (terrain heightmap) and T2.3 (Gennery-2006)
are both ~1 day each and can be done in any order.

---

*Last updated: 2026-04-27.*
