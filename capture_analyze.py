#!/usr/bin/env python3
"""Live capture -> full crack pipeline -> one annotated result figure.

Point the camera at a cracked surface, press space/c to FREEZE a frame, and
the whole pipeline runs on that still:

  YOLO-seg crack detection -> skeleton geometry (bridge.cv_features)
  MobileSAM member surface  -> W per crack          (bridge.member_extent)
  Depth Anything V2         -> scale (mm/px) + map   (bridge.depth_sources idea)
  LEFM scaling law          -> K_I / K_eff / severity (bridge.pinn_solver)

The result figure has four panels:
  1. crack analysis  — cracks coloured by severity, ids, and the load /
     Mode-I direction drawn for reference
  2. SAM surface     — member mask + the W chord measured per crack
  3. depth map       — Depth Anything V2 (skip with --no-depth)
  4. metrics table   — per-crack a, width, W, a/W, type, K_eff, K/Kic

Run `setup_models.py` once first so the depth model is already cached.

    PY=~/miniconda3/bin/python

    # live, scale from depth (give the camera's focal length in px)
    $PY capture_analyze.py --camera 0 --sigma-mpa 2.0 --focal-px 1400

    # live, manual scale instead of depth
    $PY capture_analyze.py --camera 0 --sigma-mpa 2.0 --mm-per-pixel 0.2

    # one saved photo
    $PY capture_analyze.py --image photo.jpg --sigma-mpa 2.0 --focal-px 1400

Keys (live):  space / c = capture & analyse,   q = quit
"""

import argparse
import datetime
import math
from pathlib import Path

import cv2
import numpy as np

from bridge.cv_features import analyse_frame
from bridge.member_extent import measure_member
from bridge.pinn_solver import LEFM_MAX_RATIO, evaluate_crack, load_bank

OUT_DIR = Path("output") / "capture"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"

# 35mm-equivalent focal lengths of the iPhone main rear camera (Center Stage
# OFF). f_px = stream_width_px * equiv_mm / 36.
CAMERA_PRESETS = {"iphone15": 26.0, "iphone16": 26.0, "iphone17": 24.0}

_pipe = None


# ── Depth (full map; scale is derived from it) ─────────────────────────────

def estimate_depth(image_bgr, model_name, device=None):
    global _pipe
    if _pipe is None:
        import torch
        from transformers import pipeline
        dev = device or ("cuda" if torch.cuda.is_available()
                         else "mps" if torch.backends.mps.is_available()
                         else "cpu")
        print(f"[depth] loading {model_name} on {dev} "
              "(run setup_models.py once to pre-cache) ...")
        _pipe = pipeline("depth-estimation", model=model_name, device=dev)
    from PIL import Image
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    out = _pipe(Image.fromarray(rgb))
    pd = out["predicted_depth"]
    depth = (pd.squeeze().cpu().numpy() if hasattr(pd, "cpu")
             else np.asarray(pd).squeeze())
    if depth.shape[:2] != image_bgr.shape[:2]:
        depth = cv2.resize(depth, (image_bgr.shape[1], image_bgr.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
    return depth.astype(np.float64)


def _central_region(shape_hw):
    h, w = shape_hw
    r = np.zeros((h, w), bool)
    r[h // 4:3 * h // 4, w // 4:3 * w // 4] = True
    return r


# ── Colours ────────────────────────────────────────────────────────────────

def sev_rgb(ratio):
    if ratio is None:
        return (90, 200, 90)
    if ratio >= 1.0:
        return (235, 60, 60)
    if ratio >= 0.5:
        return (245, 180, 40)
    return (90, 200, 90)


# ── The pipeline on one still frame ────────────────────────────────────────

def detect_segmentation(frame):
    """Raw YOLO instance masks (for the segmentation panel)."""
    from bridge.cv_features import get_default_detector, preprocess_for_yolo
    det = get_default_detector()
    dets = det.detect(frame, preprocess_for_yolo(frame))
    return [d.mask > 0 for d in dets if d.mask is not None]


def run_pipeline(frame, args, bank):
    seg_masks = detect_segmentation(frame)
    report, cracks = analyse_frame(
        frame, mm_per_pixel=None, detector=None,
        load_point_norm=tuple(args.load_point),
        load_direction_deg=args.load_direction_deg)

    member = measure_member(frame, cracks, mm_per_pixel=None) if cracks else None

    # depth map (for the panel and/or scale)
    depth = None
    if not args.no_depth:
        try:
            depth = estimate_depth(frame, args.depth_model, args.device)
        except Exception as e:
            print(f"[depth] skipped ({e})")

    # scale: manual wins; else derive from depth + focal length
    focal = args.focal_px
    if focal is None and args.phone:
        focal = frame.shape[1] * CAMERA_PRESETS[args.phone] / 36.0
    mmpp = args.mm_per_pixel
    if mmpp is None and depth is not None and focal:
        region = (member["mask"] if member is not None
                  and member["mask"].sum() else _central_region(depth.shape))
        z_mm = float(np.median(depth[region])) * 1000.0      # metric ~ metres
        mmpp = z_mm / focal

    # per-crack metrics
    rows = []
    for c in report["cracks"]:
        cid = c["id"]
        a_px = c["arc_length_px"]
        w_px = c["width_mean_px"]
        ang_dev = c.get("angular_dev_deg") or 0.0
        rec = member["per_crack"].get(cid) if member else None
        W_px = rec["W_px"] if rec else 5.0 * a_px
        ctype = rec["crack_type"] if rec else "edge"
        W_lb = rec["W_lower_bound"] if rec else False
        a_mm = a_px * mmpp if mmpp else None
        W_mm = W_px * mmpp if mmpp else None
        w_mm = w_px * mmpp if mmpp else None
        ev = None
        if mmpp and args.sigma_mpa is not None:
            ev = evaluate_crack(a_mm, W_mm, args.sigma_mpa,
                                angular_dev_deg=ang_dev, bank=bank,
                                crack_type=ctype)
            ev["severity"] = round(ev["K_eff_MPa_sqrt_mm"] / args.kic, 3)
        rows.append({"id": cid, "a_px": a_px, "w_px": w_px, "angle": c["angle_deg"],
                     "ang_dev": ang_dev, "W_px": W_px, "W_lb": W_lb,
                     "type": ctype, "a_mm": a_mm, "W_mm": W_mm, "w_mm": w_mm,
                     "ev": ev})
    rows.sort(key=lambda r: -(r["ev"]["severity"] if r["ev"] else r["a_px"]))
    return {"report": report, "cracks": cracks, "member": member,
            "depth": depth, "rows": rows, "mmpp": mmpp,
            "seg_masks": seg_masks}


# ── Figure ─────────────────────────────────────────────────────────────────

def build_figure(frame, bundle, args, show):
    import matplotlib.pyplot as plt

    report, cracks = bundle["report"], bundle["cracks"]
    member, depth, rows, mmpp = (bundle["member"], bundle["depth"],
                                 bundle["rows"], bundle["mmpp"])
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]
    ev_by_id = {r["id"]: r for r in rows}
    seg_masks = bundle["seg_masks"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    ax_seg, ax_skel, ax_c, ax_s, ax_d, ax_t = axes.ravel()
    PALETTE = [(255, 80, 80), (80, 160, 255), (120, 230, 120),
               (240, 200, 60), (210, 120, 240), (80, 220, 220)]

    # 1 — raw YOLO crack segmentation (per instance)
    seg = rgb.copy()
    for i, m in enumerate(seg_masks):
        seg[m] = (0.45 * seg[m]
                  + 0.55 * np.array(PALETTE[i % len(PALETTE)])).astype(np.uint8)
    ax_seg.imshow(seg)
    ax_seg.set_title(f"1. YOLO segmentation  [{len(seg_masks)} instance(s)]",
                     fontsize=11)
    ax_seg.axis("off")

    # 2 — skeleton (branch-decomposed centreline)
    skel = np.zeros((h, w), bool)
    for c in cracks:
        p = c["_pixels_rc"]
        skel[p[:, 0], p[:, 1]] = True
    skel = cv2.dilate(skel.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    sk = (0.35 * rgb).astype(np.uint8)
    sk[skel] = (60, 255, 130)
    ax_skel.imshow(sk)
    ax_skel.set_title("2. skeleton (branch-decomposed)", fontsize=11)
    ax_skel.axis("off")

    # 3 — crack analysis (severity) + load direction
    over = rgb.copy()
    for c in cracks:
        r = ev_by_id.get(c["id"])
        col = sev_rgb(r["ev"]["severity"] if r and r["ev"] else None)
        cm = np.zeros((h, w), bool)
        cm[c["_pixels_rc"][:, 0], c["_pixels_rc"][:, 1]] = True
        cm = cv2.dilate(cm.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        over[cm] = col
    ax_c.imshow(over)
    for c in cracks:
        cx, cy = c["centroid_xy"]
        ax_c.text(cx + 5, cy - 5, f"#{c['id']}", color="white", fontsize=10,
                  weight="bold",
                  bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5))
    lp = report["load"]["point_px"]
    L = 0.16 * min(h, w)
    rad = math.radians(report["load"]["direction_deg"])
    dx, dy = math.cos(rad) * L, math.sin(rad) * L
    ax_c.annotate("", xy=(lp[0] + dx, lp[1] + dy),
                  xytext=(lp[0] - dx, lp[1] - dy),
                  arrowprops=dict(arrowstyle="<->", color="cyan", lw=2.2))
    ax_c.text(lp[0] + dx, lp[1] + dy, " load", color="cyan", fontsize=9)
    prad = math.radians(report["load"]["direction_deg"] + 90)
    pdx, pdy = math.cos(prad) * L, math.sin(prad) * L
    ax_c.plot([lp[0] - pdx, lp[0] + pdx], [lp[1] - pdy, lp[1] + pdy],
              "--", color="white", lw=1.2)
    ax_c.text(lp[0] + pdx, lp[1] + pdy, " Mode-I", color="white", fontsize=8)
    ax_c.set_title(f"3. cracks (severity-coloured) + load dir   "
                   f"[{len(cracks)} crack(s)]", fontsize=11)
    ax_c.axis("off")

    # 2 — SAM surface + W chords
    surf = rgb.copy()
    if member is not None:
        m = member["mask"]
        surf[m] = (0.5 * surf[m] + 0.5 * np.array([60, 130, 246])).astype(np.uint8)
    ax_s.imshow(surf)
    for c in cracks:
        r = ev_by_id.get(c["id"])
        if r is None:
            continue
        W_px, W_lb = r["W_px"], r["W_lb"]
        rec = member["per_crack"].get(c["id"]) if member is not None else None
        if rec and rec.get("chord_ends_xy"):
            (x0, y0), (x1, y1) = rec["chord_ends_xy"]
            colr, dash, tag = "#ff2d55", "-", ""          # measured (solid red)
        else:                                             # 5×a fallback (dashed)
            ang = c["angle_deg"]
            if ang is None:
                continue
            rad = math.radians(ang)
            hx, hy = math.cos(rad) * W_px / 2, -math.sin(rad) * W_px / 2
            cx, cy = c["centroid_xy"]
            x0, y0, x1, y1 = cx - hx, cy - hy, cx + hx, cy + hy
            colr, dash, tag = "#ffd60a", "--", " (5a)"    # fallback (dashed amber)
        ax_s.plot([x0, x1], [y0, y1], dash, color=colr, lw=2.6,
                  solid_capstyle="round", zorder=5)
        ax_s.plot([x0, x1], [y0, y1], "o", color=colr, ms=7, mec="white",
                  mew=1.2, zorder=6)
        lab = (f"W = {W_px*mmpp:.0f} mm" if mmpp else f"W = {W_px:.0f} px")
        if W_lb:
            lab = "≥ " + lab
        ax_s.annotate(lab + tag, ((x0 + x1) / 2, (y0 + y1) / 2),
                      color="white", fontsize=9, weight="bold",
                      ha="center", va="center",
                      bbox=dict(boxstyle="round,pad=0.25", fc=colr,
                                ec="white", alpha=0.9), zorder=7)
    ax_s.set_title("4. SAM surface + W line  (red = measured, "
                   "dashed amber = 5a default)", fontsize=11)
    ax_s.axis("off")

    # 3 — depth
    if depth is not None:
        im = ax_d.imshow(depth, cmap="magma")
        fig.colorbar(im, ax=ax_d, fraction=0.046, pad=0.04)
        ax_d.set_title("5. depth — Depth Anything V2", fontsize=11)
    else:
        ax_d.text(0.5, 0.5, "depth off (--no-depth)", ha="center",
                  va="center", fontsize=12)
        ax_d.set_title("5. depth", fontsize=11)
    ax_d.axis("off")

    # 4 — metrics table
    ax_t.axis("off")
    ax_t.set_title("6. per-crack metrics", fontsize=11)
    if rows:
        if mmpp and args.sigma_mpa is not None:
            cols = ["#", "a mm", "w mm", "Δθ°", "W mm", "a/W", "ty", "K_eff", "K/Kic"]
            cells, colours = [], []
            for r in rows:
                ev = r["ev"]
                cells.append([
                    r["id"], f"{r['a_mm']:.1f}", f"{r['w_mm']:.2f}",
                    f"{r['ang_dev']:.0f}",
                    f"{r['W_mm']:.0f}{'+' if r['W_lb'] else ''}",
                    f"{ev['a_over_W']:.2f}", r["type"][0].upper(),
                    f"{ev['K_eff_MPa_sqrt_mm']:.1f}",
                    f"{ev['severity']:.2f}"
                    + ("" if ev["lefm_valid"] else "*")])
                c = sev_rgb(ev["severity"])
                colours.append(tuple(v / 255 for v in c))
        else:
            cols = ["#", "a px", "w px", "θ°", "W px", "type"]
            cells, colours = [], []
            for r in rows:
                cells.append([r["id"], f"{r['a_px']:.0f}", f"{r['w_px']:.1f}",
                              f"{r['angle']:.0f}" if r["angle"] is not None else "-",
                              f"{r['W_px']:.0f}{'+' if r['W_lb'] else ''}",
                              r["type"]])
                colours.append((0.9, 0.9, 0.9))
        tbl = ax_t.table(cellText=cells, colLabels=cols, loc="center",
                         cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.5)
        for i, col in enumerate(colours):           # tint the # cell per severity
            tbl[i + 1, 0].set_facecolor(col)
    else:
        ax_t.text(0.5, 0.5, "no structural cracks detected", ha="center",
                  va="center", fontsize=12)

    scale_txt = (f"{mmpp:.4f} mm/px" if mmpp else "no scale (px only)")
    sig = (f", σ={args.sigma_mpa} MPa" if args.sigma_mpa is not None
           else ", no σ → no K")
    fig.suptitle(f"crack severity   |   scale: {scale_txt}{sig}   |   "
                 f"* a/W > {LEFM_MAX_RATIO} (LEFM caution)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%H-%M-%S")
    path = OUT_DIR / f"capture_{stamp}.png"
    fig.savefig(path, dpi=110)
    print(f"[saved] {path}")
    if show:
        try:
            plt.show()
        except Exception as e:
            print(f"(no window: {e}; PNG saved)")
    plt.close(fig)


def print_metrics(bundle, args):
    rows, mmpp = bundle["rows"], bundle["mmpp"]
    if not rows:
        print("  (no structural cracks)")
        return
    if mmpp and args.sigma_mpa is not None:
        print(f"  {'#':>2} {'a(mm)':>7} {'w(mm)':>7} {'Δθ':>5} {'W(mm)':>7} "
              f"{'a/W':>5} {'type':>6} {'K_eff':>7} {'K/Kic':>6}")
        for r in rows:
            ev = r["ev"]
            print(f"  {r['id']:>2} {r['a_mm']:>7.1f} {r['w_mm']:>7.2f} "
                  f"{r['ang_dev']:>5.0f} "
                  f"{r['W_mm']:>6.0f}{'+' if r['W_lb'] else ' '} "
                  f"{ev['a_over_W']:>5.2f} {r['type']:>6} "
                  f"{ev['K_eff_MPa_sqrt_mm']:>7.1f} {ev['severity']:>6.2f}"
                  + ("" if ev["lefm_valid"] else "  !LEFM"))
    else:
        print(f"  {'#':>2} {'a(px)':>7} {'w(px)':>7} {'W(px)':>7} {'type':>6}  "
              "(give --mm-per-pixel or --focal-px + --sigma-mpa for K)")
        for r in rows:
            print(f"  {r['id']:>2} {r['a_px']:>7.0f} {r['w_px']:>7.1f} "
                  f"{r['W_px']:>6.0f}{'+' if r['W_lb'] else ' '} {r['type']:>6}")


def analyse(frame, args, bank):
    if args.max_width and frame.shape[1] > args.max_width:
        s = args.max_width / frame.shape[1]
        frame = cv2.resize(frame, None, fx=s, fy=s)
        if args.mm_per_pixel:               # rescale manual mm/px to match
            args = argparse.Namespace(**{**vars(args),
                                         "mm_per_pixel": args.mm_per_pixel / s})
    print("[run] detecting + measuring ...")
    bundle = run_pipeline(frame, args, bank)
    print(f"[run] {len(bundle['rows'])} crack(s):")
    print_metrics(bundle, args)
    build_figure(frame, bundle, args, not args.no_show)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, help="camera index (live)")
    src.add_argument("--image", type=Path, help="analyse a saved photo")

    p.add_argument("--sigma-mpa", type=float, default=None,
                   help="far-field tensile stress (MPa); needed for K_I")
    p.add_argument("--kic", type=float, default=31.6,
                   help="fracture toughness K_IC (MPa·√mm); default 31.6")
    p.add_argument("--mm-per-pixel", type=float, default=None,
                   help="manual scale (overrides depth-derived scale)")
    p.add_argument("--focal-px", type=float, default=None,
                   help="camera focal length in px → depth gives mm/px")
    p.add_argument("--phone", choices=sorted(CAMERA_PRESETS),
                   help="derive focal-px from an iPhone main-camera preset")
    p.add_argument("--no-depth", action="store_true",
                   help="skip Depth Anything V2 (no depth panel; needs "
                        "--mm-per-pixel for K)")
    p.add_argument("--depth-model", default=DEPTH_MODEL)
    p.add_argument("--device", default=None, help="cuda | mps | cpu (auto)")
    p.add_argument("--load-point", type=float, nargs=2, default=(0.5, 0.0),
                   metavar=("X", "Y"), help="normalised load point (0..1)")
    p.add_argument("--load-direction-deg", type=float, default=90,
                   help="load direction: 0=right, 90=down (default 90)")
    p.add_argument("--max-width", type=int, default=960,
                   help="downscale wide frames to this width for speed")
    p.add_argument("--no-show", action="store_true",
                   help="save the figure without opening a window")
    args = p.parse_args()

    bank = load_bank()
    if bank:
        print(f"[k_i] PINN bank: {len(bank['entries'])} ratios "
              f"(trained {bank['created']})")

    if args.image is not None:
        frame = cv2.imread(str(args.image))
        if frame is None:
            raise SystemExit(f"cannot read image: {args.image}")
        analyse(frame, args, bank)
        return

    cam = args.camera if args.camera is not None else 0
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        raise SystemExit(
            f"cannot open camera {cam} — check System Settings > Privacy & "
            "Security > Camera for your terminal app (try a different index)")
    win = "live — space/c capture, q quit"
    print(f"camera {cam} open — space/c = capture & analyse, q = quit")
    misses = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            misses += 1
            if misses > 30:                 # tolerate transient grab failures
                print("camera stopped delivering frames — exiting")
                break
            cv2.waitKey(30)
            continue
        misses = 0
        cv2.imshow(win, frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k in (ord("c"), ord(" ")):
            cv2.destroyWindow(win)
            analyse(frame.copy(), args, bank)
            for _ in range(5):              # flush stale frames after the figure
                cap.read()
            print("camera live again — space/c = capture, q = quit")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
