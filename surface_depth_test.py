#!/usr/bin/env python3
"""Standalone surface + depth + SAM probe — NO crack pipeline.

Purpose: test the "W from the surface" idea in isolation. Capture one
frame (live camera or a saved --image), click a point ON the surface, and
see four things together:

  1. SAM mask          — MobileSAM, point-prompted exactly where you click
                         (left-click = positive, right-click = negative).
  2. Depth map         — Depth Anything V2 (same model the pipeline uses).
  3. Depth consistency — of the pixels SAM called "surface", which actually
                         share the surface's depth (green) vs bled onto the
                         background (red). This is "SAM proposes, depth
                         disposes" made visible.
  4. Horizontal width  — a left↔right chord across the mask through its
                         centroid row, in px, and in mm if --focal-px given.

There is deliberately no crack detection here — you place the prompt by
hand so you can see the surface/depth/mask behaviour on its own.

Examples
--------
    PY=~/miniconda3/bin/python

    # live, built-in camera: capture, click the surface, ENTER to analyse
    $PY surface_depth_test.py --camera 0

    # one saved photo, with scale so the width prints in mm
    $PY surface_depth_test.py --image photo.jpg --focal-px 1400

    # skip the depth model (just SAM + width) for a quick first run
    $PY surface_depth_test.py --camera 0 --no-depth

Keys (live):    space / c = capture & analyse,   q = quit
Clicks:         left = positive point ("surface here")
                right = negative point ("not the surface")
                ENTER with no clicks = prompt the image centre
"""

import argparse
import datetime
from pathlib import Path

import cv2
import numpy as np

from bridge.member_extent import get_sam

OUT_DIR = Path("output") / "surface_test"


# ── SAM (point-prompted, supports negative points) ─────────────────────────

def sam_mask(image_bgr, pos_pts, neg_pts):
    """MobileSAM mask from positive/negative click points."""
    model = get_sam()
    pts = list(pos_pts) + list(neg_pts)
    labels = [1] * len(pos_pts) + [0] * len(neg_pts)
    if not pts:
        return None
    res = model(image_bgr, points=pts, labels=labels, verbose=False)[0]
    if res.masks is None or len(res.masks.data) == 0:
        return None
    m = res.masks.data[0].cpu().numpy().astype(bool)
    if m.shape != image_bgr.shape[:2]:
        m = cv2.resize(m.astype(np.uint8),
                       (image_bgr.shape[1], image_bgr.shape[0]),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
    return m


# ── Depth (Depth Anything V2, full map) ────────────────────────────────────

_pipe = None


def estimate_depth(image_bgr, model_name, device=None):
    """Full per-pixel depth map (model units; metric model ≈ metres)."""
    global _pipe
    if _pipe is None:
        import torch
        from transformers import pipeline
        dev = device or ("cuda" if torch.cuda.is_available()
                         else "mps" if torch.backends.mps.is_available()
                         else "cpu")
        print(f"[depth] loading {model_name} on {dev} "
              "(first run downloads ~100 MB) ...")
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


# ── Horizontal width across the mask ───────────────────────────────────────

def horizontal_width(mask):
    """Left↔right run of mask through its centroid row (px)."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    row = int(round(ys.mean()))
    ccol = int(round(xs.mean()))
    line = mask[row]
    w = mask.shape[1]
    if not line[min(max(ccol, 0), w - 1)]:
        # centroid fell in a hole — take the longest run on this row instead
        idx = np.flatnonzero(line)
        if len(idx) == 0:
            return None
        splits = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        run = max(splits, key=len)
        xl, xr = int(run[0]), int(run[-1])
    else:
        xl = ccol
        while xl - 1 >= 0 and line[xl - 1]:
            xl -= 1
        xr = ccol
        while xr + 1 < w and line[xr + 1]:
            xr += 1
    return {"row": row, "x_left": xl, "x_right": xr,
            "width_px": xr - xl + 1,
            "hit_border": xl == 0 or xr == w - 1}


# ── Figure ─────────────────────────────────────────────────────────────────

def build_figure(image_bgr, mask, depth, width, pos_pts, neg_pts,
                 focal_px, depth_tol, show):
    import matplotlib.pyplot as plt

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = mask.shape

    # surface depth + consistency band
    surf = float(np.median(depth[mask])) if depth is not None else None
    on_surface = off = None
    frac_on = None
    if depth is not None:
        band = abs(surf) * depth_tol
        within = np.abs(depth - surf) <= band
        on_surface = mask & within
        off = mask & ~within
        frac_on = on_surface.sum() / max(mask.sum(), 1)

    # width → mm
    width_mm = None
    if width and focal_px and surf is not None:
        z_mm = abs(surf) * 1000.0            # metric model ≈ metres → mm
        width_mm = width["width_px"] * z_mm / focal_px

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()

    # 1 — original + prompts + width line
    ax = axes[0]
    ax.imshow(rgb)
    if width:
        ax.plot([width["x_left"], width["x_right"]],
                [width["row"], width["row"]], "-", color="yellow", lw=2)
        ax.plot([width["x_left"], width["x_right"]],
                [width["row"], width["row"]], "|", color="yellow",
                markersize=12, mew=2)
    for (x, y) in pos_pts:
        ax.plot(x, y, "o", color="#1de975", mec="black", ms=9)
    for (x, y) in neg_pts:
        ax.plot(x, y, "X", color="#ff4b4b", mec="black", ms=10)
    title = "1. photo + prompts + width"
    if width:
        wtxt = f"{width['width_px']} px"
        if width_mm:
            wtxt += f" ≈ {width_mm:.0f} mm"
        if width["hit_border"]:
            wtxt += "  (hits frame → lower bound)"
        title += f"\nwidth: {wtxt}"
    ax.set_title(title, fontsize=11)
    ax.axis("off")

    # 2 — SAM mask overlay
    ax = axes[1]
    overlay = rgb.copy()
    overlay[mask] = (0.45 * overlay[mask] +
                     0.55 * np.array([60, 130, 246])).astype(np.uint8)
    ax.imshow(overlay)
    ax.set_title(f"2. SAM mask  ({mask.sum()} px, "
                 f"{100 * mask.sum() / (h * w):.0f}% of frame)", fontsize=11)
    ax.axis("off")

    if depth is not None:
        # 3 — depth map
        ax = axes[2]
        im = ax.imshow(depth, cmap="magma")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("3. depth (Depth Anything V2)", fontsize=11)
        ax.axis("off")

        # 4 — depth consistency within the SAM mask
        ax = axes[3]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        base = np.stack([gray] * 3, -1).astype(np.float64)
        base[on_surface] = 0.4 * base[on_surface] + 0.6 * np.array([30, 220, 90])
        base[off] = 0.4 * base[off] + 0.6 * np.array([235, 60, 60])
        ax.imshow(base.astype(np.uint8))
        ax.set_title(f"4. depth check: green = on-surface "
                     f"({100 * frac_on:.0f}%), red = bled", fontsize=11)
        ax.axis("off")
    else:
        axes[2].axis("off")
        axes[3].axis("off")
        axes[2].set_title("(depth skipped: --no-depth)", fontsize=11)

    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%H-%M-%S")
    path = OUT_DIR / f"surface_{stamp}.png"
    fig.savefig(path, dpi=110)
    print(f"[saved] {path}")
    if width:
        wmm = f" ≈ {width_mm:.0f} mm" if width_mm else " (give --focal-px for mm)"
        print(f"[width] {width['width_px']} px{wmm}"
              + ("  LOWER BOUND (mask hits frame edge)"
                 if width["hit_border"] else ""))
    if frac_on is not None:
        print(f"[depth] {100 * frac_on:.0f}% of the SAM mask shares the "
              f"surface depth (band ±{100 * depth_tol:.0f}%)"
              + ("" if frac_on > 0.85 else
                 "  ← SAM likely bled onto background"))
    if show:
        try:
            plt.show()
        except Exception as e:
            print(f"(could not open a window: {e}; PNG was saved)")
    plt.close(fig)


# ── Click collection on a frozen frame ─────────────────────────────────────

def collect_prompts(frame):
    win = "click surface (L=yes R=no), ENTER=run, q=cancel"
    pos, neg = [], []
    disp = frame.copy()

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pos.append((x, y))
            cv2.circle(disp, (x, y), 6, (80, 230, 120), -1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            neg.append((x, y))
            cv2.drawMarker(disp, (x, y), (60, 60, 235),
                           cv2.MARKER_TILTED_CROSS, 16, 2)

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    cancelled = False
    while True:
        cv2.imshow(win, disp)
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10):            # ENTER
            break
        if k == ord("q"):
            cancelled = True
            break
    cv2.destroyWindow(win)
    return (None, None) if cancelled else (pos, neg)


def analyse(frame, args):
    if args.center:
        h, w = frame.shape[:2]
        pos, neg = [(w // 2, h // 2)], []
        print("[sam] --center: prompting the image centre")
    else:
        pos, neg = collect_prompts(frame)
        if pos is None:                  # cancelled
            return
        if not pos:                      # nothing clicked → centre point
            h, w = frame.shape[:2]
            pos = [(w // 2, h // 2)]
            print("[sam] no clicks — prompting the image centre")

    print(f"[sam] {len(pos)} positive, {len(neg)} negative prompt(s) ...")
    mask = sam_mask(frame, pos, neg)
    if mask is None or mask.sum() == 0:
        print("[sam] no mask returned — try clicking a flatter area")
        return

    width = horizontal_width(mask)
    depth = None
    if not args.no_depth:
        depth = estimate_depth(frame, args.depth_model, args.device)

    build_figure(frame, mask, depth, width, pos, neg,
                 args.focal_px, args.depth_tol, not args.no_show)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, help="camera index (live mode)")
    src.add_argument("--image", type=Path, help="analyse a saved photo")
    p.add_argument("--focal-px", type=float, default=None,
                   help="focal length in px → prints width in mm")
    p.add_argument("--no-depth", action="store_true",
                   help="skip the depth model (SAM + width only)")
    p.add_argument("--depth-model",
                   default="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
                   help="HF depth-estimation model id")
    p.add_argument("--depth-tol", type=float, default=0.05,
                   help="depth band as a fraction of surface depth (0.05=5%%)")
    p.add_argument("--device", default=None, help="cuda | mps | cpu (auto)")
    p.add_argument("--no-show", action="store_true",
                   help="save the PNG without opening a window")
    p.add_argument("--center", action="store_true",
                   help="skip clicking; prompt the image centre (headless)")
    args = p.parse_args()

    if args.image is not None:
        frame = cv2.imread(str(args.image))
        if frame is None:
            raise SystemExit(f"cannot read image: {args.image}")
        analyse(frame, args)
        return

    cam = args.camera if args.camera is not None else 0
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        raise SystemExit(
            f"cannot open camera {cam} — check System Settings > Privacy & "
            "Security > Camera for your terminal app")
    print(f"camera {cam} open — space/c = capture & analyse, q = quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("frame grab failed")
            break
        cv2.imshow("live — space/c capture, q quit", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k in (ord("c"), ord(" ")):
            cv2.destroyWindow("live — space/c capture, q quit")
            analyse(frame.copy(), args)
            print("camera live again — space/c = capture, q = quit")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
