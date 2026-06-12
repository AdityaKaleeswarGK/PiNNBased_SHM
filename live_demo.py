#!/usr/bin/env python3
"""Live crack severity from a webcam or iPhone (Continuity Camera).

    python live_demo.py --camera 0 --mm-per-pixel 0.2 --sigma-mpa 2.0

iPhone as camera: keep it near the Mac, same Apple ID, Bluetooth + Wi-Fi
on — it appears as an extra camera device. List devices with --list, then
pick its index with --camera.

Keys: q = quit, s = save annotated frame + crack_report.json to output/live/

Frames are downscaled to --max-width for processing (mm/px is rescaled to
match). Per-crack K_I comes from the PINN F(a/W) bank via the scaling law,
so evaluation adds nothing to the per-frame cost.
"""

import argparse
import datetime
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from bridge.cv_features import analyse_frame
from bridge.depth_sources import get_depth_source
from bridge.detectors import get_detector
from bridge.member_extent import measure_member
from bridge.pinn_solver import evaluate_crack, load_bank

GREEN, YELLOW, RED = (80, 220, 80), (40, 200, 255), (60, 60, 255)

# 35mm-equivalent focal lengths of the MAIN rear camera. Continuity Camera
# uses the main camera when Center Stage is OFF (Center Stage switches to a
# dynamically cropped ultra-wide — turn it off or the scale is wrong).
# f_px = stream_width_px * equiv_mm / 36, computed from the live stream.
CAMERA_PRESETS = {
    "iphone17": 24.0,        # 48MP Fusion main, f/1.78
    "iphone16": 26.0,        # 48MP Fusion main, f/1.6
    "iphone15": 26.0,
}


def list_cameras():
    try:
        out = subprocess.run(["system_profiler", "SPCameraDataType"],
                             capture_output=True, text=True, timeout=15).stdout
        names = [ln.strip().rstrip(":") for ln in out.splitlines()
                 if ln.endswith(":") and not ln.startswith(" " * 8)
                 and "Camera" not in ln.strip()[:1]]
        names = [n for n in names if n and n != "Camera"]
        print("Cameras reported by macOS (OpenCV index usually follows "
              "this order):")
        for i, n in enumerate(names):
            print(f"  {i}: {n}")
    except Exception as e:
        print(f"could not query cameras: {e}")


def severity_colour(ratio):
    if ratio is None:
        return GREEN
    return RED if ratio >= 1.0 else YELLOW if ratio >= 0.5 else GREEN


def draw_overlay(frame, cracks, evals, fps, mm_per_pixel, sigma,
                 member=None):
    """Annotate the processed frame in place."""
    ev_by_id = {e["crack_id"]: e for e in evals}
    for crack in cracks:
        ev = ev_by_id.get(crack["id"])
        ratio = ev["severity"] if ev else None
        colour = severity_colour(ratio)
        pts = crack["_pixels_rc"]
        frame[pts[:, 0], pts[:, 1]] = colour
        cx, cy = crack["centroid_xy"]
        label = f"#{crack['id']}"
        if ev:
            label += f" K/Kic={ratio:.2f}"
        elif mm_per_pixel is None:
            label += f" {crack['arc_length_px']:.0f}px"
        if member and crack["id"] in member["per_crack"]:
            rec = member["per_crack"][crack["id"]]
            label += f" {rec['crack_type'][:1].upper()}"
            if rec.get("W_mm"):
                label += (f" W{'>=' if rec['W_lower_bound'] else '='}"
                          f"{rec['W_mm']:.0f}mm")
        cv2.putText(frame, label, (int(cx) + 6, int(cy) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(frame, label, (int(cx) + 6, int(cy) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

    worst = max((e["severity"] for e in evals), default=None)
    header = f"{fps:.1f} fps | cracks: {len(cracks)}"
    header += (f" | scale: {mm_per_pixel:.3f} mm/px" if mm_per_pixel
               else " | NO SCALE (px only)")
    if sigma is None:
        header += " | no sigma -> no K"
    cv2.putText(frame, header, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 3)
    cv2.putText(frame, header, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)
    if worst is not None:
        verdict = ("CRITICAL" if worst >= 1.0 else
                   "elevated" if worst >= 0.5 else "stable")
        colour = severity_colour(worst)
        cv2.putText(frame, f"worst K/Kic = {worst:.2f}  {verdict}",
                    (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(frame, f"worst K/Kic = {worst:.2f}  {verdict}",
                    (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", type=int, default=0, help="camera index")
    p.add_argument("--list", action="store_true", help="list cameras and exit")
    p.add_argument("--detector", default="ultralytics-seg")
    p.add_argument("--weights", default=None)
    p.add_argument("--mm-per-pixel", type=float, default=None,
                   help="manual scale at the surface you are pointing at")
    p.add_argument("--depth", default=None, choices=["monocular"],
                   help="estimate scale live with Depth Anything V2 "
                        "(needs --focal-px)")
    p.add_argument("--focal-px", type=float, default=None,
                   help="camera focal length in pixels at FULL resolution "
                        "(overrides --phone)")
    p.add_argument("--phone", default=None, choices=sorted(CAMERA_PRESETS),
                   help="compute focal-px from the phone's main-camera "
                        "specs and the live stream width (Center Stage "
                        "must be OFF)")
    p.add_argument("--depth-every", type=int, default=30,
                   help="re-estimate depth every N frames (default 30)")
    p.add_argument("--sigma-mpa", type=float, default=None)
    p.add_argument("--member-width-mm", type=float, default=None)
    p.add_argument("--member", default="off", choices=["off", "auto"],
                   help="auto: measure W per crack with MobileSAM (member "
                        "edges visible in frame); classifies edge/center")
    p.add_argument("--member-every", type=int, default=30,
                   help="re-run MobileSAM every N frames (default 30); "
                        "chords are recast against the cached mask between")
    p.add_argument("--kic", type=float, default=31.6)
    p.add_argument("--max-width", type=int, default=640,
                   help="processing resolution (smaller = faster)")
    p.add_argument("--snapshot", action="store_true",
                   help="grab ONE frame, analyse, save, exit (no window)")
    args = p.parse_args()

    if args.list:
        list_cameras()
        return

    kwargs = {}
    if args.weights:
        kwargs["weights"] = args.weights
    detector = (get_detector(args.detector, **kwargs)
                if args.detector != "classical" else get_detector("classical"))
    bank = load_bank()
    out_dir = Path("output") / "live"
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"cannot open camera {args.camera} — try --list, and check "
              "System Settings > Privacy & Security > Camera for your "
              "terminal app")
        return
    print(f"camera {args.camera} open — "
          + ("snapshot mode" if args.snapshot else "q quits, s saves"))

    if args.mm_per_pixel is not None:
        args.depth = None                      # manual scale wins
    if (args.depth == "monocular" and args.focal_px is None
            and args.phone is None):
        print("--depth monocular needs --focal-px or --phone <model> "
              "(calibrate focal once: f_px = Z_mm * size_px / size_mm)")
        return
    depth_src = None
    cached_mm_pp = None          # scale in PROCESSED-frame px
    cached_mask = None           # MobileSAM member mask (--member auto)

    t_prev = time.time()
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("frame grab failed — camera disconnected?")
            break

        # Downscale for processing; rescale mm/px (and focal) to match
        scale = min(1.0, args.max_width / frame.shape[1])
        proc = (cv2.resize(frame, None, fx=scale, fy=scale)
                if scale < 1.0 else frame.copy())

        if args.depth == "monocular" and depth_src is None:
            if args.focal_px is None:
                equiv = CAMERA_PRESETS[args.phone]
                args.focal_px = frame.shape[1] * equiv / 36.0
                print(f"[scale] {args.phone} main camera ({equiv:.0f} mm "
                      f"equiv) at {frame.shape[1]}px stream -> "
                      f"focal {args.focal_px:.0f} px (Center Stage OFF!)")
            depth_src = get_depth_source("monocular",
                                         focal_px=args.focal_px * scale)

        mm_pp = (args.mm_per_pixel / scale
                 if args.mm_per_pixel is not None else cached_mm_pp)

        use_depth_now = (depth_src is not None
                         and (cached_mm_pp is None
                              or frame_idx % args.depth_every == 0))
        report, cracks = analyse_frame(
            proc, mm_per_pixel=mm_pp,
            depth_source=depth_src if use_depth_now else None,
            detector=detector)
        if use_depth_now and report["calibration"]["mm_per_pixel"]:
            cached_mm_pp = report["calibration"]["mm_per_pixel"]
            mm_pp = cached_mm_pp

        member = None
        if args.member == "auto" and cracks:
            refresh = (cached_mask is None
                       or frame_idx % args.member_every == 0)
            member = measure_member(proc, cracks, mm_per_pixel=mm_pp,
                                    mask=None if refresh else cached_mask)
            if member is not None:
                cached_mask = member["mask"]
        frame_idx += 1

        evals = []
        if mm_pp is not None and args.sigma_mpa is not None:
            for crack in report["cracks"]:
                a_mm = crack["length_mm"]
                W_mm = args.member_width_mm or 5.0 * a_mm
                crack_type = "edge"
                if member and crack["id"] in member["per_crack"]:
                    rec = member["per_crack"][crack["id"]]
                    if rec.get("W_mm"):
                        W_mm = rec["W_mm"]
                    crack_type = rec["crack_type"]
                ev = evaluate_crack(a_mm, W_mm, args.sigma_mpa,
                                    angular_dev_deg=crack.get("angular_dev_deg") or 0.0,
                                    bank=bank, crack_type=crack_type)
                ev["crack_id"] = crack["id"]
                ev["severity"] = round(ev["K_eff_MPa_sqrt_mm"] / args.kic, 3)
                evals.append(ev)

        now = time.time()
        fps = 1.0 / max(now - t_prev, 1e-6)
        t_prev = now
        if member is not None:
            contours, _ = cv2.findContours(
                member["mask"].astype("uint8"), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(proc, contours, -1, (255, 200, 60), 1)
        draw_overlay(proc, cracks, evals, fps, mm_pp, args.sigma_mpa,
                     member=member)

        if args.snapshot:
            stamp = datetime.datetime.now().strftime("%H-%M-%S")
            img_path = out_dir / f"snap_{stamp}.jpg"
            cv2.imwrite(str(img_path), proc)
            report["evaluations"] = evals
            (out_dir / f"snap_{stamp}.json").write_text(
                json.dumps(report, indent=2))
            print(f"saved {img_path}  ({len(cracks)} crack(s))")
            break

        cv2.imshow("crack severity — q quits, s saves", proc)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            stamp = datetime.datetime.now().strftime("%H-%M-%S")
            cv2.imwrite(str(out_dir / f"snap_{stamp}.jpg"), proc)
            report["evaluations"] = evals
            (out_dir / f"snap_{stamp}.json").write_text(
                json.dumps(report, indent=2))
            print(f"saved snap_{stamp}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
