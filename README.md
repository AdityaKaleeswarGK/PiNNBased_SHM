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
| `best-2.pt` | Trained YOLO segmentation weights |
| `PINN_backend/PINN_FINAL_WALLAHI.ipynb` | Validated Mode-I PINN (hardcoded geometry, plane stress) |
| `PINN_backend/pinn_strain.ipynb` | Plane-strain variant |
| `PINN_backend/PINN.ipynb` | Earlier Bayesian-dropout PINN experiment |
| `bridge/` | **The glue** — headless CV, plugins, parameterised PINN |
| `bridge/detectors.py` | Detector plugins: ultralytics-seg/-box, torchvision (SSD/RCNN), classical |
| `bridge/depth_sources.py` | Scale plugins: manual, reference, standoff, depth camera file, monocular (Depth Anything V2) |
| `run_end_to_end.py` | One-command pipeline |

## Run it

```bash
pip install -r requirements.txt

# one-time, offline: train the F(a/W) reference bank (~75 min for 4 ratios)
python -m bridge.build_bank --ratios 0.05 0.1 0.2 0.3 --epochs 6000

# per image: instant — no training
python run_end_to_end.py test_images/your_image.jpg \
    --mm-per-pixel 0.25 --sigma-mpa 2.0 --member-width-mm 1000

# optional cross-check: full PINN solve on the worst crack (~18 min)
python run_end_to_end.py ... --validate-pinn

# live demo from a webcam / iPhone Continuity Camera (~3 fps):
python live_demo.py --camera 0 --depth monocular --focal-px 1450 --sigma-mpa 2.0
```

Without the bank, F(a/W) falls back to the Tada handbook polynomial — the
pipeline still runs; the bank substitutes the PINN-derived values.

### Plugins

Detector (`--detector`): any mask or box model maps onto the same geometry
stage; box-only detectors get masks completed by whole-frame Otsu cropped
to each box.

```bash
--detector ultralytics-seg --weights best-2.pt        # default (any YOLO-seg .pt)
--detector ultralytics-box --weights yolo_det.pt      # detection-only YOLO
--detector torchvision --arch ssd300_vgg16 --weights crack_ssd.pth
--detector classical                                  # no model, Otsu baseline
```

Scale (`--depth`): anything that yields mm/px plugs in; depth-based
sources sample the median depth over the detected crack region.

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

Outputs land in `output/<timestamp>_bridge/`:
`crack_report.json` (per-crack arc length/angle/width, px and mm,
interaction flags) and `severity_report.json` (per-crack K_I, K_II,
K_eff/K_IC ranking, worst-crack verdict).

## The contract between the two halves

The CV side measures **pixels**; the PINN solves in **mm / MPa**. The bridge
needs three physical inputs that no camera can provide on its own:

1. **`mm_per_pixel`** — from a reference object (`--ref-mm/--ref-px`), a known
   standoff + focal length (`--standoff-mm/--focal-px`), or directly.
   The planned monocular depth model plugs into
   `bridge/calibration.py::from_monocular_depth` (stub with wiring notes —
   Depth Anything V2 metric → median depth over crack region → pinhole model).
2. **`--sigma-mpa`** — far-field tensile stress at the crack. Structural
   input from loading analysis, not vision.
3. **`--member-width/height-mm`** — the member dimensions the SENT model is
   embedded in. Defaults to `W = 5a` (the notebook's validated a/W = 0.2).

## Multi-crack handling (bridge/cv_features.py)

- YOLO instance identity is preserved (masks are never OR-merged), so two
  crossing cracks detected separately stay separate.
- Skeletons are cut at branchpoints; collinear nearby branches re-link, so
  an X-junction resolves into two through-going cracks and a Y-junction
  into a main crack plus a side branch.
- Fragmented masks re-link when endpoints are close, orientations agree,
  and the gap direction matches; linked gaps count toward length
  (conservative — K grows with length).
- Crack length is a true arc-length walk along the ordered skeleton path.
- Each crack's measured angle resolves the load into per-crack
  K_I = F·σ·√(πa)·cos²δ and K_II = F·σ·√(πa)·cosδ·sinδ; ranking uses
  K_eff = √(K_I² + K_II²).
- Similarly-oriented cracks closer than the shorter one's length are
  flagged as interacting (isolated-crack K underestimates there).

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
   member's edges are in frame; MobileSAM (prompted with points offset
   perpendicular to each crack — prompting ON the crack segments the crack
   itself) returns the member mask; a chord cast through each crack along
   its growth direction = W. Same mask classifies **edge vs center**
   cracks (endpoint touching the boundary or not — center cracks use half
   length + the secant F, ~1.6x less severe). Chords exiting through the
   image border report W as a lower bound (conservative). Validated on a
   synthetic beam: W = 198 px vs truth 200, correct edge/center/clipped
   classification. Requires perspective rectification for skewed views.
   Tolerance is friendly: ~10% W error ≈ only 2–4% K error at small a/W.
3. **Survey** — COLMAP SfM over an overlapping sweep + ONE scale anchor
   (taped distance / marker / depth estimate) → metric 3D of the member;
   W, crack atlas, and growth monitoring all come from the same
   reconstruction. COLMAP gives geometry, not semantics — SAM still labels
   which plane is the member. Offline, post-survey.

## Known gaps / next steps

- [x] ~~Depth model for automatic mm_per_pixel~~ — done: `--depth monocular`
  (Depth Anything V2) and `--depth depth-file` (depth camera) plugins
- [ ] Interacting crack pairs get flagged but still use isolated-crack K;
  a true two-crack PINN domain is the research-grade fix
- [ ] K_II uses the centre-crack resolved-stress form with the edge-crack F
  as an approximation; a Mode-II Williams term in the PINN would do it properly
- [ ] YOLO recall unquantified — run `model.val()` on the test split
- [ ] Plane stress vs plane strain decision for thick members
  (`pinn_strain.ipynb` has the plane-strain variant)
