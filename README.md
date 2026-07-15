# Crack CV Pipeline + PINN Backend

Visual crack inspection, end to end: camera image → crack geometry (CV) →
Mode-I stress intensity factor K_I (physics-informed neural network) →
severity verdict against fracture toughness.

```
image ──► YOLO-seg (per-instance) ──► branch split + fragment linking
                                                 │
                       per-crack: arc length, angle, width, interactions
                                                 │  + mm_per_pixel calibration
                                                 ▼
                                        crack_report.json
                                                 │
   F(a/W) bank  ──────────►  K_I = F(a/W)·σ_eff·√(πa)  per crack, instant
   (PINN, trained                                │
    once offline)                                ▼
                              severity_report.json — ranked K_eff/K_IC
```

The PINN is a *solver*, not a general model: one training run = one solved
(a/W) configuration. By LEFM similarity that one solution covers every
crack with the same a/W at any size and load, so the expensive training
happens **once, offline** (`python -m bridge.build_bank`, ~18 min per
ratio) and every detected crack afterwards evaluates in microseconds.

## Layout

| Path | What it is |
|---|---|
| `pipeline_notebook_yolo_4.ipynb` | Full CV pipeline with all visualisations (research notebook) |
| `dataset.ipynb` | YOLO-seg training on the `crack-seg` dataset |
| `best-2.pt` | YOLO-seg crack weights — **not shipped; bring your own** (or pass `--weights`) |
| `bridge/` | **The glue** — headless CV, plugins, parameterised PINN |
| `bridge/pinn_solver.py` | The production PINN: parameterised Mode-I solver + F(a/W) bank evaluation |
| `bridge/build_bank.py` | Offline bank trainer (`python -m bridge.build_bank`) |
| `bridge/pinn_bank.json` | The trained F(a/W) bank shipped with the repo |
| `bridge/cv_features.py` | Headless multi-crack CV core (detection → geometry → screening) |
| `bridge/detectors.py` | Detector plugins: ultralytics-seg/-box, torchvision (SSD/RCNN), classical |
| `bridge/depth_sources.py` | Scale plugins: manual, reference, standoff, depth camera file, monocular (Depth Anything V2) |
| `bridge/member_extent.py` | Member width W via MobileSAM (`--member auto`), edge/center classification |
| `run_end_to_end.py` | One-command pipeline (single image → reports) |
| `setup_models.py` | One-time setup: installs deps, fetches MobileSAM + Depth Anything V2 |
| `capture_analyze.py` | Live capture → full pipeline → annotated 6-panel result figure |
| `live_demo.py` | Continuous live severity overlay (webcam / iPhone Continuity Camera) |
| `surface_depth_test.py` | Standalone surface + depth + SAM probe (no crack logic) |

The original notebook PINN (hardcoded geometry, plane stress + the
plane-strain and Bayesian-dropout variants) lives in its own repo:
[PiNNBased_SHM](https://github.com/AdityaKaleeswarGK/PiNNBased_SHM).
`bridge/pinn_solver.py` is that formulation refactored to take
geometry/material/load as arguments.

## Run it

**First time (fresh clone):**

```bash
# installs requirements.txt + downloads MobileSAM (~39 MB) and
# Depth Anything V2 (~100 MB); both cache and never re-download
python setup_models.py

# bring your own YOLO-seg crack weights — NOT included in the repo.
# put them in the repo root as best-2.pt, or pass --weights to any run script.
```

`mobile_sam.pt` and Depth Anything V2 download automatically; `best-2.pt`
(the crack detector) is your own trained model and must be supplied.

```bash
# one-time, offline: train the F(a/W) reference bank (~75 min for 4 ratios)
python -m bridge.build_bank --ratios 0.05 0.1 0.2 0.3 --epochs 6000

# per image: instant — no training
python run_end_to_end.py test_images/your_image.jpg \
    --mm-per-pixel 0.25 --sigma-mpa 2.0 --member-width-mm 1000

# live capture: space/c freezes a frame → 6-panel figure (segmentation,
# skeleton, severity + load dir, SAM surface + W line, depth, metrics table)
python capture_analyze.py --camera 0 --sigma-mpa 2.0 --focal-px 1400

# continuous live overlay from a webcam / iPhone Continuity Camera (~3 fps)
python live_demo.py --camera 0 --depth monocular --focal-px 1450 --sigma-mpa 2.0

# optional cross-check: full PINN solve on the worst crack (~18 min)
python run_end_to_end.py ... --validate-pinn
```

Without the bank, F(a/W) falls back to the Tada handbook polynomial — the
pipeline still runs; the bank substitutes the PINN-derived values.

Outputs land in `output/<timestamp>_bridge/`:
`crack_report.json` (per-crack arc length/angle/width, px and mm,
interaction flags) and `severity_report.json` (per-crack K_I, K_II,
K_eff/K_IC ranking, worst-crack verdict).

## The CV pipeline in detail (`bridge/cv_features.py`)

The CV side turns one BGR frame into a list of per-crack geometry records.
Every stage is headless (no plotting) so the same core drives the CLI, the
live demo, and the capture tool.

**1. Preprocessing.** The frame is converted to grayscale and contrast-
normalised with CLAHE (clip limit 2.0, 8×8 tiles), then fed to the
detector as a 3-channel image. Thin low-contrast cracks on concrete
survive this much better than raw frames.

**2. Detection (pluggable).** The default detector is YOLO-seg
(`best-2.pt`, conf 0.20, IoU 0.45, crack class 0). Crucially, **instance
identity is preserved**: per-instance masks are kept separate and never
OR-merged, so two crossing cracks detected as two instances stay two
cracks even where their pixels overlap. Box-only detectors (YOLO-det,
torchvision SSD/Faster-RCNN) plug into the same stage: their boxes get
masks completed by whole-frame Otsu thresholding cropped to each padded
box. A `classical` backend (pure Otsu, no model) exists as a baseline.

**3. Component extraction.** Each instance mask is split into 8-connected
components; components under 200 px² are dropped as noise. Each surviving
component carries its Euclidean distance transform (used later for width).

**4. Skeleton branch decomposition.** Each component is skeletonised
(scikit-image). Branchpoints (skeleton pixels with 3+ neighbours) are cut,
splitting the skeleton into simple segments; spurs shorter than 8 px are
discarded. This is what resolves junctions: an X-crossing inside one mask
becomes four branches, a Y-junction becomes three.

**5. Segment linking (union-find).** Segments re-join when their endpoints
are within 25 px, their PCA orientations agree within 25°, and the gap
direction matches both orientations. This does two jobs at once:
collinear branches cut at a junction re-link (so an X resolves into two
through-going cracks, a Y into a main crack + side branch), and
fragmented detections (mask gaps) re-connect. **Linked gaps count toward
crack length** — conservative, since K_I grows with length.

**6. Per-crack geometry.** For each assembled crack:
- **Arc length** — a true arc-length walk along the ordered skeleton path
  (diagonal steps count √2), plus any linked gap lengths. This is the `a`
  the PINN side uses.
- **Orientation** — PCA (SVD) over all skeleton pixels, reported in
  (-90°, 90°] image coordinates.
- **Width** — 2 × the distance-transform value sampled every 3rd skeleton
  pixel; mean/median/max are all reported.
- **Tip-to-tip span and endpoints** — the farthest pair among segment
  endpoints.

**7. Structural filters.** Two rejection gates, both logged with reasons
in the report rather than silently dropped: cracks shorter than 20 px
(junction stubs), and cracks whose **mean width is below the ACI 224R
0.3 mm cosmetic limit** (3 px fallback when uncalibrated) — those are
non-structural.

**8. Structural context.** Given a load point and direction
(`--load-point`, `--load-direction-deg`), each crack gets its **angular
deviation** from the ideal Mode-I plane (perpendicular to the tension
axis) — this is what resolves K into K_I/K_II downstream — and its
distance to the load point.

**9. Interaction screening.** Every crack pair with orientations within
30° is checked: if their minimum skeleton-to-skeleton distance is smaller
than the shorter crack's length, the pair is flagged — isolated-crack K_I
underestimates there (the fields amplify each other). Flagged, not fixed;
see Known gaps.

### Scale: pixels → mm (`bridge/depth_sources.py`)

Anything that yields mm/px plugs in; depth-based sources sample the
median depth over the detected crack region and convert via the pinhole
model (`mm_per_px = depth_mm / focal_px`).

```bash
--mm-per-pixel 0.25                                   # manual
--ref-mm 100 --ref-px 380                             # reference object
--standoff-mm 500 --focal-px 1400                     # known distance
--depth depth-file --depth-map img_depth.npy --focal-px 1400   # depth camera
--depth monocular --focal-px 1400                     # Depth Anything V2 metric
```

Monocular caveat: metric monocular depth carries 5–15% scale error in the
wild — treat it as an estimate, prefer a depth camera or reference when
the verdict matters.

### Member width W (`bridge/member_extent.py`, `--member auto`)

The crack is by definition ON the member, so its skeleton pixels are
free, always-correct prompts (offset perpendicular to the crack —
prompting on the crack itself segments the crack): MobileSAM returns the
member-surface mask, and a chord cast through it along each crack's
growth direction is that crack's W. The same mask classifies **edge vs
center** cracks (endpoint within 6 px of the mask boundary = edge; center
cracks use half length + the secant F, ~1.6× less severe). A chord
exiting through the image border instead of the mask boundary means W is
a lower bound (conservative; flagged, not hidden). Tolerance is friendly:
~10% W error ≈ only 2–4% K error at small a/W.

### Detector plugins

```bash
--detector ultralytics-seg --weights best-2.pt        # default (any YOLO-seg .pt)
--detector ultralytics-box --weights yolo_det.pt      # detection-only YOLO
--detector torchvision --arch ssd300_vgg16 --weights crack_ssd.pth
--detector classical                                  # no model, Otsu baseline
```

## The PINN in detail (`bridge/pinn_solver.py`)

### The problem it solves

2D linear elasticity (plane stress) for a **single edge crack under
tension (SENT)**: an edge crack of length `a` along y = 0 in a plate of
width `W`, modelled on the **upper symmetric half** (height `H_half`),
with far-field tension `σ` applied on the top edge. Units are mm and MPa
throughout, so K_I comes out in MPa·√mm (1 MPa·√m = 31.62 MPa·√mm).
Material defaults are concrete-like: E = 30 000 MPa, ν = 0.2. This is the
formulation validated against FEniCS FEM in the team's rundown.

### Architecture: smooth net + Williams enrichment

The displacement field is an ansatz with two parts:

```
u(x, y) = u_smooth(x, y)  +  A_KI · u_williams(r, θ)
```

- **`u_smooth`** — a plain MLP (2 → 128 → 128 → 128 → 128 → 2, tanh)
  taking normalised coordinates and predicting the smooth part of the
  displacement.
- **`u_williams`** — the analytic Williams near-tip Mode-I displacement
  field (√r singularity, plane-stress κ = (3−ν)/(1+ν)) centred on the
  crack tip, with **one learned scalar amplitude**. That amplitude *is*
  K_I: `K_I = 2μ · softplus(A_raw)`. Softplus keeps it positive, and it
  is initialised by inverse-softplus so training starts at the handbook
  K_I rather than from zero.

The network never has to learn the singularity — it only learns the
smooth correction, and the physics loss tunes the singular amplitude.

### Loss (what "physics-informed" means here)

Every epoch samples fresh points; all stress terms come from autograd
derivatives of the displacement ansatz:

| Term | Where | Enforces | Weight |
|---|---|---|---|
| PDE residual | ~4000 interior collocation points (crack-face strip excluded) | equilibrium ∂σ = 0 | 1 |
| Top edge | 600 pts at y = H_half | σ_yy = σ, σ_xy = 0 (the applied load) | 100 |
| Symmetry line | 600 pts, y = 0, x > a | v = 0, σ_xy = 0 | 100 |
| Crack faces | 600 pts, y = 0, x < a | traction-free (σ_yy = σ_xy = 0) | 100 |
| Left/right edges | 600 pts each | traction-free (σ_xx = σ_xy = 0) | 100 |
| Rigid-body pin | bottom-right corner | u = 0 (kills translation) | 500 |
| Tip suppression | 300 pts in a half-disc of radius 0.3a around the tip | `u_smooth` ≈ 0 near the tip, so the Williams term alone carries the singularity — this is what makes the learned amplitude a clean K_I | 75 |

Optimiser: Adam, lr 1e-3, StepLR (×0.3 every epochs/4). All
length-scale-dependent constants (suppression radius, J-ring radius,
crack-face strip) are expressed relative to `a` and `H_half`, so the
tuning survives a change of physical scale. Runs on CUDA, Apple MPS, or
CPU (auto-detected).

### Three independent K_I extractions

After training, K_I is read out three ways as a self-consistency check:

1. **Learned parameter** (`KI_param`) — `2μ · softplus(A_raw)` directly.
   This is the value the bank stores.
2. **J-integral** (`KI_J`) — path integral over a half-ring around the
   tip (radius ≈ 0.75a, kept inside the domain), K_I = √(J·E). Noisiest
   of the three (2–6% off).
3. **Williams displacement fit** (`KI_W`) — least-squares projection of
   the total predicted v-displacement on an arc near the tip onto the
   unit Williams field.

Each solve also reports the Tada handbook value and the relative error
against it.

### The F(a/W) bank: train once, evaluate forever

By LEFM similarity, one solve at a given a/W is the *same* dimensionless
problem for every crack with that ratio, at any physical size and load:

```
K_I = F(a/W) · σ · √(πa)
```

So `build_bank.py` runs one full PINN solve per ratio at a reference
scale (W = 100 mm, H/W = 1, σ = 100 MPa) and stores only the
dimensionless geometry factor `F = K_I / (σ√(πa))` — from all three
extractions, plus the Tada value and the deviation — in
`bridge/pinn_bank.json`. At inference, `geometry_factor()` linearly
interpolates F over the banked ratios (Tada fallback outside coverage),
and `evaluate_crack()` applies the scaling law per crack in microseconds.

Inclined cracks use resolved-stress relations with the CV-measured
angular deviation δ from the ideal Mode-I plane:

```
K_I  = F·σ·√(πa)·cos²δ        K_II = F·σ·√(πa)·cosδ·sinδ
K_eff = √(K_I² + K_II²)        (ranking metric, vs K_IC)
```

Center cracks (classified by `--member auto`) switch to two-tip form:
half the measured length and the Feddersen/Tada secant factor
`F = √(sec(π·a_half/W))`.

### Running the PINN

```bash
# build/rebuild the bank — the only slow step, run once offline
python -m bridge.build_bank --ratios 0.05 0.1 0.2 0.3 --epochs 6000
#   --ratios   a/W values to solve (each ≈18 min at 6000 epochs on Apple MPS)
#   --epochs   training epochs per ratio (6000 = validated config)
#   --device   cuda | mps | cpu (auto-detected if omitted)
#   --out      bank path (default bridge/pinn_bank.json)

# cross-check the scaling law: full training solve on the worst crack
python run_end_to_end.py img.jpg ... --validate-pinn

# or from Python — one solve, any geometry:
python -c "
from bridge.pinn_solver import solve_mode_I
import json
res = solve_mode_I(a_mm=20, W_mm=100, H_half_mm=50, sigma_MPa=100, epochs=6000)
print(json.dumps(res, indent=2))"
```

`solve_mode_I` guards its own validity: a/W must be in (0, 0.6) — the
range of the SENT geometry factor — and anything past the team's stated
LEFM limit a/W = 0.30 is flagged `lefm_valid: false` in the reports.

## The contract between the two halves

The CV side measures **pixels**; the PINN solves in **mm / MPa**. The bridge
needs three physical inputs that no camera can provide on its own:

1. **`mm_per_pixel`** — from a reference object (`--ref-mm/--ref-px`), a known
   standoff + focal length (`--standoff-mm/--focal-px`), a depth camera
   (`--depth depth-file`), or monocular Depth Anything V2
   (`--depth monocular`).
2. **`--sigma-mpa`** — far-field tensile stress at the crack. Structural
   input from loading analysis, not vision.
3. **`--member-width-mm`** — the member dimension W the SENT model is
   embedded in; or `--member auto` to measure it with MobileSAM.
   Defaults to `W = 5a` (the validated a/W = 0.2).

## Modeling assumptions (read before trusting numbers)

- Each crack maps to a **single edge crack under tension (SENT), upper
  symmetric half, plane stress** — the formulation validated against
  FEniCS FEM in the team's rundown (0.3% mesh-converged reference).
- Surface arc length is used as crack depth `a` — a photo cannot see how
  deep a crack cuts into the member. Conservative proxy; team decision.
- a/W ≤ 0.30 is the team's stated LEFM validity limit; evaluations beyond
  it carry a warning. The Tada polynomial itself is valid to a/W ≤ 0.6.
- Default K_IC = 31.6 MPa·√mm (1.0 MPa·√m, plain concrete). Override with
  `--kic` for your mix.

## Validation status (June 2026)

- F(a/W) bank vs Tada handbook: within 0.9–2.0% at a/W ∈ {0.05, 0.1, 0.2, 0.3}
  (6000 epochs each; param and Williams extractions agree within ~1%)
- Notebook config (a=20, W=100, σ=100): K_I within 0.94% of handbook,
  0.6% of the team's FEniCS FEM reference (1083.75 MPa·√mm)
- Scale invariance: independent solves at lab scale (a=20, W=100) and wall
  scale (a=400, W=2000, different σ) give the same F within 0.59% —
  empirical confirmation that the train-once bank applies at any size
- The J-integral extraction runs 2–6% off param/Williams throughout — the
  noisiest of the three; param is the value the bank stores

## Agreed design: acquiring W (member width)

W should come from measurement, not the crack. Three tiers, fallback order:

1. **No context** — assumption prior (current `W = 5a` default, pins
   a/W = 0.2). Honest upgrade planned: report K as a range over
   F(a/W→0)..F(0.3) instead of a silent point value.
2. **Context shot — IMPLEMENTED (`--member auto`)** — step back so the
   member's edges are in frame; MobileSAM measures W per crack (see the
   CV section above). Validated on a synthetic beam: W = 198 px vs truth
   200, correct edge/center/clipped classification. Requires perspective
   rectification for skewed views.
