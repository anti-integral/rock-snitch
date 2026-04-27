# rock-snitch-v2 — Design Walkthrough

> A long-form, narrative explanation of how this tool was designed. Read it top
> to bottom — each section sets up the next. Skip ahead only if you already
> know the answer.

---

## Table of contents

1. [The problem](#1-the-problem)
2. [Reading the data: what we actually have](#2-reading-the-data-what-we-actually-have)
3. [The first big realization: how far can we even *see* a 10 cm rock?](#3-the-first-big-realization-how-far-can-we-even-see-a-10-cm-rock)
4. [The second big realization: stereo depth error grows with Z²](#4-the-second-big-realization-stereo-depth-error-grows-with-z²)
5. [The conclusion: two regimes are mandatory](#5-the-conclusion-two-regimes-are-mandatory)
6. [Designing the near-field branch (the easy one)](#6-designing-the-near-field-branch-the-easy-one)
7. [Designing the far-field branch (the hard one)](#7-designing-the-far-field-branch-the-hard-one)
8. [The teacher-student insight](#8-the-teacher-student-insight)
9. [Choosing the model stack](#9-choosing-the-model-stack)
10. [The mono height head — what it actually learns](#10-the-mono-height-head--what-it-actually-learns)
11. [Range-gated fusion](#11-range-gated-fusion)
12. [The CAHVORE problem and how we solved it](#12-the-cahvore-problem-and-how-we-solved-it)
13. [Testability: mocks behind Protocols](#13-testability-mocks-behind-protocols)
14. [End-to-end walkthrough with concrete numbers](#14-end-to-end-walkthrough-with-concrete-numbers)
15. [Limitations and future work](#15-limitations-and-future-work)
16. [Glossary](#16-glossary)

---

## 1. The problem

NASA's Perseverance rover drives across Jezero crater. Before each drive segment
the on-board autonomous-navigation system (ENav) builds a local terrain map
from stereo Navcam imagery and plans a safe path. ENav is conservative — it
treats anything taller than ~25–35 cm as untraversable and routes around it.

That's fine for safety, but a lot of mission-relevant decisions happen *above*
the immediate driving level:

- **Science targeting**: which boulders are worth a Mastcam-Z follow-up?
- **Long-range routing**: a rock field 80 m ahead might not affect today's drive
  but it constrains the next sol's plan.
- **Post-hoc analysis**: "how rocky was this traverse?" — needs 10 cm-grade
  recall, not 35 cm-grade.

So we want a tool that:

1. Takes any Navcam frame (with metadata).
2. Returns a list of rocks ≥ **10 cm** tall.
3. Does this **at long range** (tens of metres, ideally up to ~100 m), not just
   the 5-10 m hazard cone the on-board system handles.
4. Reports a real height in metres, not just "rock detected here".

That last requirement is the killer. It rules out almost every off-the-shelf
detector and forces us into 3D reconstruction territory. The rest of this
document is the story of how we got from "10 cm rocks at long range" to a
working pipeline.

---

## 2. Reading the data: what we actually have

The first thing any honest design document does is look at the data. We have:

- **~1,500 sols** of Mars 2020 Perseverance raw Navcam imagery, downloaded by
  `download_navcam.py` from the public PDS feed at `mars.nasa.gov/rss/api`.
- **Per-image JSON sidecars** with metadata: SCLK timestamp, mast pointing,
  rover pose, camera model, subframe rectangle, etc.
- **Stereo capability**: there is a `NAVCAM_LEFT` and a `NAVCAM_RIGHT`. The
  baseline between them is **42.4 cm** (Maki et al. 2020).

A real metadata file looks like this (sol 100, abbreviated):

```json
{
  "sol": 100,
  "site": 4,
  "drive": "218",
  "extended": {
    "mastAz": "78.7995",
    "mastEl": "1.06137",
    "sclk": "675828717.752",
    "scaleFactor": "4",
    "subframeRect": "(1,1441,5120,960)",
    "dimension": "(1280,240)"
  },
  "attitude": "(0.353249,0.00650504,-0.00830915,-0.93547)",
  "camera": {
    "instrument": "NAVCAM_LEFT",
    "camera_model_type": "CAHVORE",
    "camera_model_component_list":
      "(1.03927,0.645857,-1.9785);(0.182183,0.983214,-0.010707);..."
  },
  "imageid": "NLF_0100_0675828717_276ECM_N0040218NCAM00503_01_295J"
}
```

Three observations from poking at this:

**Observation 1: "stereo pair" is not implicit.**

You might assume that for any sol, the L and R Navcam frames at the same time
are a stereo pair. They aren't. The two cameras can be commanded independently.
On sol 2 I found same-SCLK frames where the left was looking at the rover deck
and the right was looking at the terrain — clearly not a stereo pair, even
though they share a clock timestamp.

→ We need to **mine** real stereo pairs. The right filter is *opposite eye* +
*same sol* + *SCLK within ~5 s* + *mast Az/El within ~0.5°* + *identical
subframe*. After applying that filter to our 1,501 sols we get **26,790
verified stereo pairs across 924 sols** — plenty to work with.

**Observation 2: the images are tiles, often binned.**

The native Navcam sensor is 5120×3840 (≈20 MP). The on-board pipeline splits
this into 1280×960 tiles before downlink. Many products are also binned 4× to
save bandwidth. The image we just looked at, for instance, has
`subframeRect = (1, 1441, 5120, 960)` and `dimension = (1280, 240)` — that's a
horizon-only 5120×960 strip downsampled 4× to 1280×240.

So when we reason about pixel scale, we have to use the *effective* IFOV after
binning, not the native one.

**Observation 3: the camera model is CAHVORE.**

This is JPL's nonlinear camera model — it has 20 parameters split across seven
3-vectors (C, A, H, V, O, R, E) plus three scalars (linearity, mtype, mparam).
It generalizes pinhole to handle radial distortion, decentering, and the
fish-eye-vs-perspective interpolation that wide lenses need. We can't just
treat the Navcam as a pinhole — we'd lose ~5° of distortion at the corners,
which translates to several pixels of disparity error on a stereo match, which
translates to depth error.

We'll come back to CAHVORE in §12. For now, just note that we can't pretend
it's pinhole.

---

## 3. The first big realization: how far can we even *see* a 10 cm rock?

Before designing any algorithm, ask the physics. Forget machine learning for a
minute — just compute the **angular size** of a 10 cm rock at various ranges
and convert to pixels.

Navcam pixel IFOV: native = 0.33 mrad/px (Maki et al. 2020). With 4× binning
common in our dataset, effective IFOV ≈ **1.32 mrad/px**.

Angular size of an object of height *h* at range *R*: θ = h/R radians.

For h = 0.10 m:

| Range | θ (mrad) | Pixels (binned) | Pixels (native) |
|------:|---------:|----------------:|----------------:|
|  10 m |    10.0  |          7.6 px |         30.3 px |
|  20 m |     5.0  |          3.8 px |         15.2 px |
|  30 m |     3.3  |          2.5 px |         10.1 px |
|  50 m |     2.0  |          1.5 px |          6.1 px |
| 100 m |     1.0  |          0.76 px|          3.0 px |

This table is the most important table in the entire design. It tells us:

- **At 4× binning, a 10 cm rock at 50 m is 1.5 pixels tall.** That's at the
  Nyquist limit. No detector — neural or classical — can robustly find a
  1.5-pixel feature against Mars terrain noise.
- **At 100 m, it's 0.76 px.** Below Nyquist. Forget it.
- **At native resolution, you have ~6 px even at 50 m.** That's tractable. So
  there is a path to longer range *if we get hold of the un-binned tiles*.

For v1 we work with what we have (the 4× binned downlinked products). That
means the practical detection horizon for 10 cm rocks is **roughly 50–60 m**.
Anything we claim about longer ranges in v1 should be flagged as approximate.

If the user ever wants confident 100 m detection, we'd need to fetch the
reassembled full-resolution Navcam mosaics from the PDS archive (via
`mars-raw-utils`). That's a future-work item, not v1.

---

## 4. The second big realization: stereo depth error grows with Z²

OK, so given a stereo pair, what depth accuracy can we *actually* achieve?

The stereo triangulation formula:

```
Z = f · b / d
```

where Z is depth (m), f is focal length (px), b is baseline (m), d is disparity
(px). Differentiating:

```
ΔZ = Z² · Δd / (f · b)
```

Plugging in our numbers (binned): f ≈ 576 px, b = 0.424 m, sub-pixel matching
precision Δd ≈ 0.5 px:

| Z (m) | ΔZ (m) |
|------:|-------:|
|   5   |  0.05  |
|  10   |  0.20  |
|  15   |  0.46  |
|  20   |  0.82  |
|  30   |  1.85  |
|  50   |  5.13  |
|  80   | 13.13  |

Now compare ΔZ to the rock height we're trying to measure (0.10 m):

- At 10 m, depth uncertainty (0.20 m) is **2× the rock height**. Already
  marginal.
- At 15 m, uncertainty is **4.6× the rock height**.
- At 20 m, **8×**.
- At 30 m, **18×**.

But wait — depth uncertainty isn't quite the same as *height* uncertainty. To
measure the height of a rock, we look at the difference between two depths
(rock peak vs ground at the same line of sight) or between two y-pixel
projections at the same depth. Either way, the error scales similarly with Z.

The honest takeaway: **stereo can reliably measure 10 cm-grade heights only out
to about 10–15 m.** Beyond that, the error bars on the height itself eat the
signal.

This is independent of the stereo algorithm. RAFT-Stereo is better than SGBM,
but neither one violates the Z² law.

---

## 5. The conclusion: two regimes are mandatory

Putting §3 and §4 together:

| Range | Can we *see* 10 cm rocks? | Can stereo *measure* their height? |
|------:|:-------------------------|:------------------------------------|
| ≤ 10 m | yes (≥ 7.6 px) | yes (ΔZ ≤ 0.2 m) |
| 10–20 m | yes (3.8 px @ 20m) | marginal |
| 20–50 m | yes (1.5 px @ 50m, ok) | **no** |
| > 50 m | no (sub-pixel) | no |

So there's a band — from about 20 m out to 50 m — where we **can see** the
rock but **can't trust stereo for its height**. We need a different mechanism
for height in that band.

That different mechanism has to be either:

(a) **Better stereo geometry** (longer baseline, higher resolution). We don't
    have that; it would require hardware or full-res tiles we haven't
    downloaded.

(b) **Some non-geometric source of depth.** That means learning. A neural
    network that has seen many natural scenes can predict per-pixel depth from
    a single image — *monocular metric depth estimation*. This is what
    UniDepth, Metric3D, and Depth Anything do.

So the design crystallizes:

> **Two-regime pipeline.**
> Near-field (≤ ~20 m): stereo, geometric, no learning.
> Far-field (> ~20 m): monocular depth + segmentation + learned head.
> Fuse the two by range.

The rest of the design follows from this commitment.

---

## 6. Designing the near-field branch (the easy one)

Let's start with the easier of the two branches. The recipe is classical
multi-view geometry, so there's not much to *invent* — the work is plumbing
each step correctly.

**Step 6.1 — Pair the frames.** Already discussed in §2. We mine real stereo
pairs by matching SCLK + mast pointing + subframe.

**Step 6.2 — Linearize each camera.** CAHVORE has radial distortion. To use any
standard stereo library we first apply the inverse distortion to produce a
"virtual pinhole" image — an image that *would have been* captured by an
ideal pinhole camera with the same C, A, H, V (the linear core of CAHVORE). I
explain the math in §12.

**Step 6.3 — Rectify the pair.** Even after linearization, the two cameras
aren't aligned. Their optical axes don't coincide and their image planes aren't
coplanar. Stereo algorithms assume both — assume that for any 3D point, its
projection into the left and right images sits on the **same row** (epipolar
geometry collapsed to image rows).

To get there: compute a new common coordinate frame whose x-axis is the
baseline direction and whose z-axis is the average forward axis. Compute
per-eye rotations from each camera's frame into this new frame. Apply those
rotations as homographies (`H = K_dst · R · K_src⁻¹`) via
`cv2.warpPerspective`. Result: two warped images sharing a single intrinsic
matrix `K`, with epipolar lines on rows.

**Step 6.4 — Compute disparity.** Now run a stereo matcher. We support two:

- **SGBM** (semi-global block matching, OpenCV). CPU. Robust, fast, mature.
  Combined with a left-right consistency check that throws out pixels where
  the left disparity and right disparity disagree by > 1.5 px.
- **RAFT-Stereo**. GPU. State-of-the-art deep stereo. Iterative refinement.
  Better in low-texture regions (sand, distant terrain).

Both produce a `DisparityMap` with per-pixel disparity, confidence, and a
validity mask.

**Step 6.5 — Back-project to a 3D point cloud.** For each valid pixel,
`Z = f·b/d`, then `X = (u-cx)·Z/f`, `Y = (v-cy)·Z/f`. Result: a per-pixel
`(X, Y, Z)` in the rectified-left camera frame, which we can transform into
the rover-site frame using the per-image extrinsics.

**Step 6.6 — Fit the local ground plane.** Now the interesting decision:
**how do you measure the height of a rock without knowing where the ground
is?**

You can't just take "Z above the rover" because the rover may be tilted, the
terrain may slope, etc. You need a *local* notion of "ground". So fit a plane
to the points in front of the rover. If the terrain is approximately planar
locally (true for most Mars driving), this gives you a reliable reference.

We use **RANSAC** for the fit because the point cloud is full of outliers
(rocks themselves, far horizon, sky, sometimes parts of the rover deck). The
algorithm:

1. Randomly pick 3 points → candidate plane.
2. Count inliers (points within 5 cm of the plane).
3. Repeat 2,000 times, keep best.
4. Re-fit the best inlier set with SVD for sub-cm precision.
5. Flip the normal so it points up in the rover frame (Z+ = up convention).

If RANSAC starves (< 50 inliers), fall back to global SVD.

Output: a `GroundPlane(normal, d, inlier_mask, rmse)`. The RMSE is a quality
indicator we'll use later.

**Step 6.7 — Get rock candidates.** We need region proposals, *something* to
attribute heights to. Options:

(a) Pure geometric: "any cluster of points more than X cm above the plane". 
    Works, but you get a lot of false positives from imperfect ground fits and
    you don't get clean instance masks.

(b) Run a 2D object detector. But we don't want to train one (no labels).

(c) Run **SAM2 in everything-mode**. This is a class-agnostic segmenter that
    returns *every* salient mask in the image without needing labels or
    prompts. Filter the output by area (20–20,000 px) and aspect ratio
    (0.2–5.0). Many masks will be ground patches or sky — that's fine, they
    fail the height check in the next step.

We chose (c) because it's clean, label-free, and the downstream height check
is a strong filter.

**Step 6.8 — Per-mask height.** For each surviving mask:

1. Look up its 3D points in the cloud.
2. Compute `h = X·normal + d` for each point — signed height above plane.
3. Take the **95th percentile** as the rock height.

Why p95 and not max? Because `max` gets blown up by a single noisy disparity
pixel — one bad triangulation can put a point 5 m above the ground in a
totally normal scene. p95 keeps the "tall part of the rock" without being
hostage to one outlier.

If `h_p95 ≥ 10 cm`, emit a `RockDetection` with the bbox, the centroid, the
range, the height, and `source="stereo"`.

That's the entire near-field branch. No training, no neural net (apart from
SAM2 for masks). Just careful geometry.

---

## 7. Designing the far-field branch (the hard one)

OK, now we need to do the same job at 30 m or 80 m where stereo can't help.
Per §3, we *can* still see the rock — it's a few pixels tall, not zero — we
just can't measure its height from stereo geometry.

What can we do?

**Idea 1: Train a 2D detector on AI4Mars.** AI4Mars labels Big Rocks in
Curiosity imagery. We could fine-tune a detector on that dataset, then run it
on Perseverance frames. Pros: easy, well-understood. Cons: no 3D height — just
a bounding box. Doesn't satisfy the "height in metres" requirement.

**Idea 2: Geometric tricks with shadows.** Shadow length × sun elevation =
rock height. The sun position is known from the SCLK timestamp. This is
elegant and we should consider it as an auxiliary signal. But it requires
clean shadow detection on Mars terrain, which is harder than it looks (rocks
on rocks, partial shadows, sun elevation near zenith at noon). Not robust
enough as a primary signal.

**Idea 3: Monocular metric depth.** Use a network like UniDepthV2 that predicts
*per-pixel depth in metres* from a single RGB image. Combine with masks from
SAM2 to get a per-rock height estimate by projecting the mask into 3D.

Idea 3 is the only one that gives metric height *and* generalizes to any range.
The catch is **monocular depth networks are notoriously unreliable when the
scene is far from their training distribution.** Mars terrain is exactly that
— most depth networks have seen indoor scenes, urban driving, and natural
landscapes on Earth. They've never seen a Mars surface.

So we need to *adapt* the monocular signal to Mars somehow. Two ways:

(a) **Fine-tune the depth network on Mars data.** Expensive. Easy to overfit
    given how few labels we'd have. Risks breaking the network's strong
    out-of-the-box geometric priors.

(b) **Treat the depth network as a feature** and learn a small correction on
    top. Cheap. Less destructive. Requires labeled (mask → height) pairs to
    train against.

Option (b) is much better, but where do the labels come from? Manually
labeling rock heights in Mars imagery is impractical (we'd need a 3D-aware
annotator, and even then, what's the truth?).

This is where the design gets clever.

---

## 8. The teacher-student insight

Look at what we have:

- A near-field branch (§6) that produces *trustworthy* rock heights via
  geometry, but only at short range.
- A far-field branch we want to build that needs *labeled* (mask → height)
  data, ideally drawn from the same distribution as the inference data
  (Mars).

The realization: **the near-field branch is a teacher.** It can label rocks at
short range automatically, and those labels are exactly the kind of
(image patch, 3D height) supervision a far-field model needs.

This is the standard self-supervised / semi-supervised "pseudolabel" pattern,
applied to depth. Concretely:

> Run the geometric near-field branch over all 26,790 stereo pairs. Keep
> only the high-quality detections (close range, high stereo confidence,
> low ground-plane RMSE). Their (left image patch, measured height in metres)
> pairs become the training set for a far-field height regressor.

The far-field model never has to know about stereo. At inference time it sees
a single image, predicts heights from monocular features, and the trained head
silently encodes everything the teacher learned about how rocks of various
shapes/textures/lighting in *this kind of terrain* relate to height.

A few important properties of this teacher-student setup:

1. **No human labels.** Cost of labeling is zero.
2. **Distribution match.** The student trains on the exact terrain it'll see
   at inference (same camera, same lighting, same minerology). This is the
   single biggest reason monocular models work on Mars at all.
3. **Quality control on the teacher.** Stereo isn't perfect. We filter
   teacher outputs aggressively before they become labels — high stereo
   confidence, low ground-plane RMSE, range ≤ 15 m, height ≥ 8 cm. Garbage
   labels would poison the student.
4. **Explicit range specialization.** The teacher labels only ≤ 15 m
   detections. The student learns to predict heights for masks at *any* range
   and is evaluated specifically on the long-range regime.

This is the architectural keystone. Everything else is implementation detail.

---

## 9. Choosing the model stack

The far-field branch uses three pretrained foundation models plus one small
trained head. Here's why each one was chosen.

### 9.1 SAM2 (Segment Anything Model 2) — segmentation

We need *masks*, not boxes, because:

- A mask gives a precise boundary; we can pool features and depths only over
  the rock pixels, not over the bbox bounding box (which is mostly ground).
- A mask gives an honest area estimate.
- A mask works for irregular shapes — Mars rocks are almost never rectangular.

SAM2 specifically:

- **Class-agnostic.** It doesn't know what a rock is; it just finds salient
  regions. We don't need rock-specific training data.
- **Strong out-of-distribution.** The original SAM was trained on >1 B masks
  spanning everything from satellite imagery to medical scans. Mars terrain
  isn't *that* far out; we mostly get reasonable masks first try.
- **Prompt-flexible.** Even though we use everything-mode by default, we can
  later prompt it with text via GroundingDINO ("rock. boulder. stone.") for
  filtering.

Alternatives we rejected:
- Mask R-CNN / DETR-style fine-tuning: would need labels.
- Mean-shift segmentation (the 2014 Rock Hunting paper): brittle on Mars
  shadows and dust.

### 9.2 DINOv2 — features

Once SAM2 gives us a mask, we need a feature representation of *what's inside
the mask*. We can't feed raw pixels to a small head — too high-dimensional,
too noisy. We need a strong image embedding.

DINOv2 (Meta, 2023) is a self-supervised vision transformer trained on 142 M
images. Its patch features are state-of-the-art for almost every downstream
task that doesn't have labels. Critical properties:

- **Frozen.** We don't fine-tune. We use the features as-is. This is exactly
  the pattern that works in SAM-LAD, SAM-6D, and many other zero-shot
  pipelines.
- **Patch-level outputs.** A single forward pass gives us a `(Hp, Wp, D)`
  feature grid. We can mean-pool these patches under any mask to get a
  fixed-size (D-dimensional) feature for that mask.
- **Captures both appearance and shape.** A DINOv2 feature for a flat rock
  shadow looks different from a feature for a tall boulder, even when both
  occupy the same number of mask pixels.

We use ViT-L/14 (1024-d features). Larger variants don't help here because the
downstream head is already small.

Alternatives rejected:
- CLIP: text-aligned but not as strong on dense localization.
- ConvNeXt: weaker than DINOv2 for this use.
- ResNet on ImageNet: no chance.

### 9.3 UniDepthV2 — monocular metric depth

This is the load-bearing model. It predicts metric (in metres) per-pixel depth
from a single RGB image. UniDepthV2 specifically because:

- **It accepts intrinsics K at inference time.** Most monocular depth models
  bake in the camera assumption from training. UniDepth lets us pass our
  rectified CAHV K matrix, which means its metric scale is anchored to *our*
  camera's actual focal length, not whatever Earth camera it was trained on.
- **It outputs uncertainty.** UniDepthV2 returns a per-pixel confidence map.
  We don't currently use this in fusion, but it's available for future
  weighted blending with stereo depth.
- **Strong zero-shot.** Tested on 10+ datasets in zero-shot regime; better
  than ZoeDepth and Depth Anything for our use.

Alternatives:
- **Metric3Dv2**: also very good. Slightly worse zero-shot, slightly better
  fine-tunability. We keep it as a backup behind a config flag.
- **Depth Anything v2**: best for relative depth, but metric calibration is
  flakier.
- **ZoeDepth**: requires fine-tuning per camera; doesn't fit our use.

### 9.4 RAFT-Stereo (the stereo workhorse)

For the near-field branch we offer two stereo backends:

- **SGBM** (OpenCV). Fast, CPU, deterministic. Our default and our test
  baseline.
- **RAFT-Stereo**. GPU. Better in low-texture regions where SGBM produces
  speckle.

Both implement the same `StereoMatcherProto` so swapping is trivial.

### 9.5 GroundingDINO (optional)

Text-prompted box detector ("rock. boulder. stone."). Used as an *optional*
filter on SAM2 masks: if both SAM2 and GroundingDINO agree on a region, the
mask gets a confidence bump. Behind a feature flag because it's expensive
and the marginal benefit is unclear without an evaluation.

---

## 10. The mono height head — what it actually learns

We've now got SAM2 (masks), DINOv2 (features), UniDepthV2 (depth). Each one is
frozen and pretrained. The only *learned* component in the whole far-field
branch is a small MLP that maps mask features → height.

### 10.1 What the head sees

For each mask `m` produced by SAM2 on the left image:

1. Pool DINOv2 features over the mask → 1024-d feature vector `f`.
2. Project mask pixels into 3D using UniDepth's depth and the rectified K →
   compute a coarse "geometric height" `h_geom`.
3. Pass `f` through the head → `h_pred` (in log-metres).

Final height for the mask: `max(h_geom, exp(h_pred))`.

The `max` is deliberate. The geometric estimate is a *floor*: it gives a
height that's consistent with the mono-depth map. The head can correct
underestimates from depth noise but never *under*shoots an obviously tall
geometric reading.

### 10.2 Architecture

```
DINOv2 mask feature (1024-d, frozen)
    │
    ▼
Linear(1024, 256) → ReLU → Dropout(0.1)
Linear(256,  256) → ReLU → Dropout(0.1)
Linear(256,    1)
    │
    ▼
log-height (m)  →  exp() → metric height
```

Three layers, ~330 K parameters. Small enough to train in minutes; big enough
to capture the "DINOv2 feature → height" relationship.

We predict **log-height** because:

- Heights span 0.08 m (just under threshold) to ~2 m (boulder). A linear target
  gives an order-of-magnitude dynamic range; log-space flattens it.
- Errors in log-space correspond to *relative* errors in metric space, which
  is what we care about (a 5 cm error on a 10 cm rock is much worse than a
  5 cm error on a 1 m rock).

### 10.3 Loss

**Smooth L1** (Huber) on log-height. Why Huber instead of MSE:

- Pseudolabels are noisy. Even after filtering there will be a few outliers
  where stereo measured 80 cm but the rock was actually 30 cm (a tall ground
  patch, a misregistered 3D point).
- MSE punishes outliers quadratically and would let those few bad labels
  dominate gradient updates.
- Huber is quadratic for small errors (good gradients near optimum) but
  linear for large errors (robust to outliers). Best of both.

### 10.4 Training data

Generated by `rock-snitch pseudolabel`:

- Run the near-field branch on every stereo pair.
- Keep detections where: range ≤ 15 m, stereo confidence ≥ 0.3, mask area ≥
  20 px, ground-plane RMSE ≤ 5 cm, height ≥ 8 cm.
- For each kept detection, extract DINOv2 features for the left image and
  mean-pool under the mask. Save (feature, height) to JSONL.

Expect ~5–20 labels per stereo pair (varies by sol; rocky sols give more), so
~50K–200K labels total across the whole dataset.

### 10.5 Training mechanics

- 90/10 train/val split, deterministic seed.
- AdamW, lr=1e-3, weight_decay=1e-4, batch_size=128.
- Up to 40 epochs with grad-clip 1.0; checkpoint best val.
- Tensorboard logging, DDP-ready.

On the 5090 a typical training run takes ~10–30 minutes for 100 K labels.

---

## 11. Range-gated fusion

We now have two branches running on every left frame:
- `near_field_detections` — rock list with `source="stereo"`, ranges 0–30 m.
- `far_field_detections` — rock list with `source="mono"`, ranges 10–120 m.

How do we merge them?

The simplest correct rule: **stereo wins when its measurements are
trustworthy; mono fills in past that.**

Concrete algorithm (`pipeline/fuse.py`):

1. Keep every stereo detection with range ≤ `stereo_trust_range_m` (default
   20 m).
2. For every mono detection:
   - If its range > 20 m → keep (mono regime).
   - If its range ≤ 20 m → keep *only if* it doesn't IoU-overlap (>0.3) with a
     kept stereo detection. This avoids double-counting the same rock.
3. After merging, any detection that overlaps another from a different source
   is re-tagged as `source="fused"` with the max height/confidence of the
   overlap set.

This gives us three confidence tiers, encoded in the `source` field:

- `stereo`: short-range, geometric, very trustworthy.
- `fused`: both branches agree. Highest confidence.
- `mono`: long-range, learned. Trust depends on training quality.

The 20 m threshold isn't sacred — it can be tuned via config. We picked it
because the `Z²` analysis (§4) shows stereo error crosses the rock height at
roughly that range.

---

## 12. The CAHVORE problem and how we solved it

This is the most under-appreciated load-bearing piece of the whole pipeline.
Get it wrong and *everything* downstream is silently a few percent off.

### 12.1 What's actually in the JSON

When you crack open one Mars2020 metadata sidecar, the camera model is a
single semicolon-separated string of 10 components. Here it is, verbatim,
from `data/metadata/00100/NLF_0100_…_295J.json`:

```
"camera_model_component_list":
  "(1.03927,0.645857,-1.9785);
   (0.182183,0.983214,-0.010707);
   (-609.121,772.41,-9.07759);
   (22.3091,131.353,738.105);
   (0.180782,0.983474,-0.0105907);
   (2e-06,0.049535,-0.015973);
   (-0.003612,0.013016,-0.023961);
   2.0; 0.0; 0.0"
```

Parsed, that's **seven 3-vectors plus three scalars = 20 numbers**, and each
one has a specific job:

| Slot | Letter      | Meaning                                                                        | Sol-100 left Navcam values |
|------|-------------|--------------------------------------------------------------------------------|----------------------------|
| 0    | **C**       | *Centre* — optical centre of the camera in the rover frame, metres             | `(1.04, 0.65, −1.98)` ← ~2 m above the deck |
| 1    | **A**       | *Axis* — unit vector pointing along the boresight                              | `(0.18, 0.98, −0.01)` ← roughly +Y, slightly down |
| 2    | **H**       | *Horizontal* — encodes both fx **and** the horizontal principal point cx       | `(−609, 772, −9)` |
| 3    | **V**       | *Vertical* — same for fy and cy                                                 | `(22, 131, 738)` |
| 4    | **O**       | *Optical* — symmetry axis of the lens distortion (often = A but not required)  | `(0.18, 0.98, −0.011)` |
| 5    | **R**       | Radial distortion polynomial coefficients (R0, R1, R2)                         | `(2e−6, 0.0495, −0.016)` |
| 6    | **E**       | Entrance-pupil offset — accounts for the pupil moving as the lens tilts        | `(−0.0036, 0.013, −0.024)` |
| 7    | linearity   | Interpolates 1.0 = perspective ↔ ∞ = fish-eye. 2.0 means "in between"          | `2.0` |
| 8    | mtype       | Selects between CAHVOR / CAHVORE math; 0 = classical CAHVORE                   | `0.0` |
| 9    | mparam      | Auxiliary parameter for some mtype variants                                    | `0.0` |

The first four vectors (C, A, H, V) are the **CAHV linear core**. They alone
are enough to describe a perfect pinhole camera. The last three vectors plus
three scalars (O, R, E, linearity, mtype, mparam) are the **distortion
extension** that makes CAHVORE different from a textbook camera.

Running `parse_component_list()` and then `principal_point()` / `focal_lengths()`
on the values above gives the equivalent pinhole intrinsics:

```
fx = 739.66 px,  fy = 739.49 px
cx = 648.57,     cy = 125.31

       ┌                          ┐
K  =   │ 739.66    0.00   648.57 │
       │   0.00  739.49   125.31 │
       │   0.00    0.00     1.00 │
       └                          ┘
```

(The cy is small here — 125.31 — because this particular product is a
1280×240 horizon-only panorama strip, not the full 1280×960 frame.)

### 12.2 What "pinhole" means and why CV assumes it

A pinhole camera is the simplest possible projection: every 3D point shoots
a straight ray through one infinitely small hole and lands on a flat image
plane. The math is **linear**:

```
            ┌ fx   0   cx ┐
u ≈ K · X,  │  0  fy   cy │
            └  0   0    1 ┘
```

Three numbers (fx, fy, principal point cx/cy), plus a 3D rigid pose, and
you're done. Every line in the world maps to a line in the image. Every
coplanar object maps to a planar projection. **It's linear.** That's why
it's so popular.

Crucially: **every standard CV library assumes pinhole intrinsics under the
hood.** OpenCV's `stereoRectify`, `findEssentialMat`, `solvePnP`,
`reprojectImageTo3D`; RAFT-Stereo; UniDepthV2's intrinsics input; SAM2's
internal coordinates. Each of them takes a 3×3 K matrix. None of them takes
a polynomial distortion model. If you feed them an undistorted image and a
pinhole K, they work; if you feed them a distorted image, they silently
produce wrong answers because they assume straight world lines stay straight.

### 12.3 Why Navcam isn't pinhole

Real lenses have **radial distortion**. Wide-angle lenses (Navcam is 96° × 73°
FOV) bend straight lines into curves the further from the optical axis you
get. The CAHVORE additions (O, R, E, linearity) describe exactly that
bending.

The cleanest way to see what this means in our actual data: project a known
3D point through the **linear core only** (treating Navcam as pinhole) and
again through the **full CAHVORE** (with distortion), and look at the pixel
disagreement. Run on the real sol-100 left Navcam:

```
Test setup:
  - 7×5 grid of pixels uniformly across the 1280×240 frame.
  - Each pixel unprojected through the linear core to a unit ray.
  - A 3D point placed 5 m along that ray.
  - Re-project that point back two ways: linear-CAHV vs full-CAHVORE.

Pixel disagreement |uv_linear − uv_full|:
  near image centre :   0.37 px
  mean across grid  :  90.92 px
  max at corners    : 221.05 px
```

Read that carefully:

- **At the centre of the image, distortion is negligible** — under one pixel.
  Pinhole would work fine if you only ever looked at the middle.
- **At the corners, the linear and full models disagree by 221 pixels.**
  That's not a small correction. The same 3D point lands in completely
  different places depending on which model you use.

Why so dramatic on this particular frame? It's a 1280×240 *horizon strip*
(`subframeRect = (1, 1441, 5120, 960)`) — the subframe spans the **full
sensor width**, so the corner pixels really are at the lens edge where
Navcam's wide-angle distortion peaks. For a square 1280×960 frame closer to
the optical axis the corner errors are smaller, more in the 5–15 px range —
still far too big to ignore.

### 12.4 Why a few pixels of distortion ruins stereo

Even setting aside the 221-px panorama edge case, the "merely 5–15 px" corner
errors in normal frames are catastrophic for stereo. Here's the chain:

1. **Sub-pixel matching is the whole point of stereo.** SGBM and RAFT-Stereo
   both produce disparity to ~0.5 px precision in well-textured regions. We
   *rely* on that precision to triangulate at long range.

2. **5 px of distortion at the corners introduces 5 px of residual epipolar
   mismatch.** After rectification, points that should be on the same row in
   left and right are *not*, by up to 5 px. Stereo matching, which searches
   along the row, either fails entirely or grabs a wrong pixel.

3. **5 px of disparity error at f ≈ 740 px, b = 0.42 m, around Z = 22 m**:
   ```
   true Z      = f · b / d        = 740 · 0.42 / 14.0  ≈ 22 m
   observed Z  = f · b / (d − 5)  = 740 · 0.42 /  9.0  ≈ 35 m
   ```
   — a single 5-pixel matching error at moderate range turns "rock at 22 m"
   into "rock at 35 m". Off by ~60%.

4. **For the 10 cm height threshold, that depth error becomes ~6 cm of pure
   noise in the height calculation** — already over half our threshold, from
   one edge effect we could have prevented with a single pre-processing step.

So: **we can't pretend Navcam is pinhole.** The whole near-field branch and
the K-conditioned UniDepthV2 call would silently produce wrong answers,
worst at exactly the locations (image edges, far field) we care about most.

### 12.5 What we do — linearization

The classical fix (Gennery 2001, used on every JPL Mars mission since MER):
**resample the image so the resampled image is what a pinhole camera would
have seen.**

That is: for each pixel `(u, v)` in the *output* (linearized) image, figure
out which pixel in the *input* (distorted) image gets resampled into that
slot. Build that mapping once per camera, run it as a remap.

Mathematically:

1. For pixel `(u, v)` in the output image, compute the ray `r` through the
   pinhole linear core.
2. Apply the *forward* CAHVORE distortion to `r` → a distorted ray `r'`.
3. Project `r'` through the linear core → the distorted pixel `(u', v')`
   in the input image.
4. Sample the input image at `(u', v')` with bilinear interpolation.

After this, the output image is *as if* it had been captured by a perfect
pinhole camera with the same C, A, H, V (and zero R, zero E, linearity = 1).
Every standard CV operation now works correctly.

This linearization is the **only** time CAHVORE math needs to run. After
linearization, every downstream module — rectification, SGBM, RAFT-Stereo,
the point-cloud projection, the rectified-K we feed UniDepthV2 — just sees a
boring 3×3 K. None of them know CAHVORE exists.

### 12.6 Where this lives in the code

In `io/cahvore.py`:

- `parse_component_list()` — parses the JSON string into a `CameraModel`
  dataclass.
- `principal_point()`, `focal_lengths()` — extract pinhole parameters from
  the linear core (these are the numbers in the K table above).
- `intrinsics_matrix()` — builds the 3×3 K.
- `project_cahv()`, `unproject_cahv()` — pinhole projection / ray
  reconstruction.
- `project()` — full CAHVORE projection including distortion (used for the
  221-px reprojection test above and for any "is this point visible?" check).
- `linearize_to_cahv()` — returns a fresh `CameraModel` with R=0, E=0,
  linearity=1: the "ideal pinhole" the linearized image is targeting.

In `geometry/rectify.py`:

- `compute_rectification()` — takes two CAHVOREs, returns the common
  rectified K and the per-eye rotations needed to align both eyes.
- `rectify_pair()` — applies those rotations as homographies via
  `cv2.warpPerspective`.

A subtlety we currently flag: `rectify_pair` *assumes the input image has
already been linearized* (i.e. distortion has been removed by a previous
remap). For Navcam frames downloaded straight from the public feed, the
images technically still contain the distortion. The remaining error is
small at the centre (where most rocks are) but non-zero at the edges. A
proper v2 will add an explicit `linearize_image()` remap step in front of
rectification, using `cv2.remap` with the lookup table built from
`unproject_cahv` ∘ `cahvore_distort` ∘ `project_cahv`.

For Navcam (a well-corrected medium-FOV lens) skipping that remap still
gets us within a couple of pixels everywhere except the panorama-strip
edges. For Hazcams (true fish-eye, 124° FOV) the linearization would be
critical and you'd never skip it. We don't process Hazcams in v1, but the
same `cahvore.py` would handle them.

---

## 13. Testability: mocks behind Protocols

A pipeline that requires CUDA + 5 GB of model weights to run a single test is
a pipeline that nobody runs. We deliberately structured the code so that the
**entire pipeline** can be exercised on a laptop with no GPU.

### 13.1 Protocols as the seam

In `contracts.py` we define minimal Protocol interfaces:

```python
class StereoMatcherProto(Protocol):
    def compute(self, pair: RectifiedPair) -> DisparityMap: ...

class SegmenterProto(Protocol):
    def segment(self, image: UInt8Array) -> MaskList: ...

class FeatureExtractorProto(Protocol):
    def extract(self, image: UInt8Array) -> FloatArray: ...

class DepthEstimatorProto(Protocol):
    def predict(self, image: UInt8Array, K: FloatArray | None = None) -> DepthMap: ...
```

Every model wrapper implements one of these. `pipeline/run.py` only ever sees
the Protocol type, never the concrete class. That means we can swap in
anything — mocks, alternative backends, future models — without touching the
pipeline.

### 13.2 The mocks

Each model has a deterministic "Mock" implementation:

- **`MockSegmenter`**: thresholds the image, runs OpenCV connected components,
  returns the resulting blobs as `Mask2D` instances. Works on real images;
  produces real masks; needs no weights.
- **`MockDepthEstimator`**: returns a radial ramp of depth (distance from
  image centre), parameterized by a base depth and a slope. Always finite,
  always valid, always deterministic.
- **`MockFeatureExtractor`**: returns a fixed-shape random feature grid with a
  seeded RNG. Deterministic across runs.

These aren't no-ops — they're working implementations that produce
believable-shaped outputs the pipeline can consume. As a result, the 59 unit
tests cover not just individual modules but full pipeline runs. The CI would
catch a contract break (e.g., a model now returns `(W, H, D)` instead of
`(H, W, D)`) even without GPU.

### 13.3 The CLI's `--mock-models` flag

`rock-snitch detect --mock-models` swaps in the mocks at runtime. It's
genuinely useful for:

- Smoke-testing on a laptop (no weights, no CUDA).
- Verifying CLI plumbing during development.
- Producing a consistent reference output for integration tests.

---

## 14. End-to-end walkthrough with concrete numbers

Let's trace one stereo pair through the entire pipeline.

**Input.** Sol 150, sclk 6.84e8, mast Az = 12.4°, El = -2.1°.
- Left image: 1280×960 RGB, scaleFactor=4 (binned).
- Right image: matching subframe, same SCLK ± 0.5 s.
- Both have CAHVORE camera models in the JSON.

**Step 1 — Pair confirmation.** SCLK delta = 0.31 s ✓. Az delta = 0.02° ✓. El
delta = 0.01° ✓. Subframes identical ✓. Confirmed pair.

**Step 2 — Linearize.** Left CAHVORE → CAHV. Focal length ≈ 576 px (from
binned focal in pixels). Principal point near (640, 480). Right ditto.
Distortion correction shifts corner pixels by ~2 px.

**Step 3 — Rectify.** New common K with `fx=fy=576`, `cx=640`, `cy=480`. Per
-eye rotations align both eyes such that epipolar lines are rows. Rectified
images written into `RectifiedPair`.

**Step 4 — Disparity.** SGBM produces a disparity map. ~85% of pixels valid;
the rest are NaN (sky, occlusions, low texture). Median disparity ≈ 14 px,
which corresponds to depth ≈ `576 × 0.424 / 14 ≈ 17.4 m` — typical for
Navcam terrain ahead of the rover.

**Step 5 — Point cloud.** Per-pixel back-projection. ~1 M points. After
filtering for `Z < 200 m` and `valid mask`, ~880 K usable points.

**Step 6 — Ground plane.** RANSAC over a 20 K random sample. Best plane has
normal ≈ (0.02, 0.0, 0.999) and d ≈ 1.78 m (the camera is 1.78 m above the
ground in the rectified frame). RMSE = 3.1 cm. Inlier count: ~720 K.

**Step 7 — SAM2 masks.** Everything-mode produces ~140 raw masks. After area
+ aspect filtering, 53 candidates.

**Step 8 — Per-mask heights (near-field).** For each candidate:
- Pull 3D points under the mask.
- Compute heights above plane.
- Take p95.

Of the 53 candidates: 8 have `h_p95 ≥ 10 cm` and `range ≤ 20 m`. These become
near-field detections, e.g.:
- Rock #1: range 8.2 m, height 15 cm, confidence 0.78.
- Rock #2: range 12.5 m, height 11 cm, confidence 0.62.
- ...

**Step 9 — Far-field branch.** UniDepthV2 produces a 1280×960 depth map (mean
depth ≈ 22 m, max ≈ 95 m). DINOv2 produces an 80×60 patch feature grid at
1024-d. SAM2 masks (the same ones) have features pooled.

For each mask:
- Project mask pixels into 3D using mono depth + rectified K.
- Compute `h_geom` from `Y_max - Y_min`.
- Run mask features through the trained head → `h_pred`.
- Final height = `max(h_geom, h_pred)`.

After range filter (`10 m ≤ range ≤ 120 m`) and height filter (`≥ 10 cm`): 6
mono detections, e.g.:
- Rock #A: range 32 m, height 14 cm, confidence 0.55.
- Rock #B: range 58 m, height 22 cm, confidence 0.41.
- ...

**Step 10 — Fuse.**
- Stereo detections with range ≤ 20 m → kept verbatim. (8 of them.)
- Mono detections with range > 20 m → kept. (5 of 6.)
- Mono detection with range ≤ 20 m (the lone one at 17 m) → check IoU against
  kept stereo. Overlaps stereo Rock #2 with IoU 0.41. Suppressed.
- Final list: 13 detections. Two of them got `source="fused"` because both
  branches saw them.

**Step 11 — Output.** Write JSON: 13 entries with bbox, centroid, range, height,
confidence, source. Write overlay PNG: green boxes for stereo, orange for
mono, yellow for fused. Write turbo-colormapped disparity PNG.

That's one pair. The whole thing runs in ~2 s on a 5090, dominated by SAM2
(~700 ms) and UniDepthV2 (~600 ms).

---

## 15. Limitations and future work

Honest list of where this v1 falls short:

**1. 4× binning caps long-range performance.** A 10 cm rock at 80 m is 0.95
binned pixels — sub-Nyquist. We need the full-res 5120×3840 reassembled
mosaics from PDS for confident 80 m+ detection. `mars-raw-utils` has the tile
reassembly; integrating it is a v2 task.

**2. No human-labeled evaluation set.** We evaluate against held-out
pseudolabels, which are themselves stereo-derived. So we can't check whether
the mono branch is *correct* past the stereo trust horizon — only whether
it's self-consistent. A few hundred hand-labeled long-range rocks would
unlock real precision/recall numbers in the hard regime.

**3. No shadow-based height as a sanity check.** Sun position is in the
metadata; rock shadows are visible; combining shadow length with sun
elevation is a 1-line extra height estimate that costs almost nothing. We
should add it as a third signal that fuses with stereo and mono.

**4. The mono head doesn't see depth uncertainty.** UniDepthV2 returns a
per-pixel uncertainty map. We currently ignore it. A natural extension is to
weight the height estimate by inverse uncertainty, or feed mask-pooled
uncertainty as an extra input to the head.

**5. No temporal consistency.** Successive Navcam frames during a drive show
the same rocks from different angles. We could track and fuse detections
across frames for much higher confidence. v1 doesn't do this; v2 should.

**6. No rover-frame map output.** We emit per-image hit-lists. For mission
planning we'd want a per-sol map in the rover-site frame with all rocks
projected and deduplicated. Straightforward to add.

**7. Sensitivity to mast tilt.** If the rover is on a slope, the local
ground plane fit captures it correctly, but rocks on a *different* local
slope (e.g., on a far hillside) get measured against the wrong reference.
Multi-plane segmentation would help.

**8. RAFT-Stereo wrapper is untested at scale.** SGBM is our default and is
well-tested. RAFT-Stereo is plumbed in but the GPU path needs end-to-end
verification on the 5090 before we trust it.

---

## 16. Glossary

- **CAHVORE** — JPL's 20-parameter nonlinear camera model. The seven letters
  stand for Center, Axis, Horizontal, Vertical, Optical, Radial, Entrance.
- **CAHV** — the linear (pinhole) core of CAHVORE. Standard pinhole model
  with a slightly unusual parametrization.
- **Disparity** — pixel offset between matched features in a rectified
  stereo pair. Inversely proportional to depth.
- **DINOv2** — Meta's self-supervised vision transformer; we use it as a
  frozen feature extractor.
- **EDR / RDR** — Engineering Data Record / Reduced Data Record — PDS
  product types. We work with the public raw images, which are EDRs.
- **IFOV** — instantaneous field of view, the angular size of one pixel.
- **Pseudolabel** — a label generated automatically by a teacher model
  (here, the stereo branch), used to train a student.
- **RANSAC** — Random Sample Consensus, robust model-fitting against
  outliers.
- **Rectification** — warping a stereo pair so that epipolar lines coincide
  with image rows.
- **SAM2** — Meta's Segment Anything Model 2; class-agnostic image and video
  segmenter.
- **SCLK** — spacecraft clock. Seconds since the mission epoch. Each image
  has one to nanosecond precision.
- **Sol** — one Martian solar day, ~24 h 39 min. Sol 1 was 2021-02-19 for
  Perseverance.
- **Subframe** — a rectangular region of the sensor downlinked together,
  sometimes with binning.
- **Trust horizon** — the range beyond which a measurement source becomes
  unreliable. For our binned stereo, ~15 m for height; we use 20 m as the
  fusion gate to stay conservative.
- **UniDepthV2** — monocular metric depth network that takes camera
  intrinsics at inference and returns per-pixel metric depth.

---

*Document last updated: 2026-04-21. Pair this with `README.md` (overview),
`CLAUDE.md` (SSH runbook), and `AGENTS.md` (code conventions) to navigate the
codebase.*
