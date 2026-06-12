#!/usr/bin/env python3
"""Camera image -> multi-crack geometry -> per-crack K_I -> ranked severity.

    python run_end_to_end.py test_images/foo.jpg \
        --mm-per-pixel 0.25 --sigma-mpa 2.0 --member-width-mm 1000

Stages
------
1. CV    : YOLO-seg, skeleton branch decomposition, fragment linking,
           interaction screening (bridge.cv_features) -> crack_report.json
2. Scale : pixel -> mm via a bridge.depth_sources plugin (manual, reference
           object, standoff pinhole, depth-camera file, or monocular
           Depth Anything V2)
3. K_I   : EVERY crack evaluated instantly via the LEFM scaling law
           K_I = F(a/W)*sigma_eff*sqrt(pi*a), with F from the PINN-trained
           bank (bridge/pinn_bank.json, built once offline by
           `python -m bridge.build_bank`) or the Tada handbook as fallback.
           Inclined cracks get resolved-stress K_I + K_II from their
           CV-measured angle. -> severity_report.json
4. Verdict: cracks ranked by K_eff/K_IC; worst crack drives the verdict.

`--validate-pinn` additionally runs a full PINN training solve on the worst
crack and reports its deviation from the scaling-law value (slow path,
~18 min at 6000 epochs on Apple MPS).
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

import cv2

from bridge.cv_features import analyse_frame
from bridge.depth_sources import available_depth_sources, get_depth_source
from bridge.detectors import available_detectors, get_detector
from bridge.member_extent import measure_member
from bridge.pinn_solver import (LEFM_MAX_RATIO, evaluate_crack, load_bank,
                                solve_mode_I)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", type=Path, help="crack image to analyse")

    det = p.add_argument_group(
        f"detector plugin ({', '.join(available_detectors())})")
    det.add_argument("--detector", default="ultralytics-seg",
                     help="detection backend (default ultralytics-seg)")
    det.add_argument("--weights", default=None,
                     help="model weights/checkpoint for the detector")
    det.add_argument("--arch", default="ssd300_vgg16",
                     help="torchvision architecture (torchvision backend)")
    det.add_argument("--det-conf", type=float, default=0.20,
                     help="detector confidence threshold")

    cal = p.add_argument_group(
        f"scale plugin ({', '.join(available_depth_sources())})")
    cal.add_argument("--mm-per-pixel", type=float, default=None,
                     help="manual scale (shortcut for --depth manual)")
    cal.add_argument("--depth", default=None,
                     help="depth source backend")
    cal.add_argument("--ref-mm", type=float, default=None,
                     help="reference object true size (mm), with --ref-px")
    cal.add_argument("--ref-px", type=float, default=None,
                     help="reference object measured size (px)")
    cal.add_argument("--standoff-mm", type=float, default=None,
                     help="camera-to-surface distance (mm), with --focal-px")
    cal.add_argument("--focal-px", type=float, default=None,
                     help="focal length in pixels (pinhole conversions)")
    cal.add_argument("--depth-map", type=Path, default=None,
                     help="aligned depth map file (depth-file backend)")
    cal.add_argument("--depth-model", default=None,
                     help="HF model id override (monocular backend)")

    load = p.add_argument_group("structural inputs")
    load.add_argument("--sigma-mpa", type=float, default=None,
                      help="far-field tensile stress (MPa); required for K_I")
    load.add_argument("--member-width-mm", type=float, default=None,
                      help="member dimension W along crack growth (mm); "
                           "default 5x each crack's length (a/W = 0.2)")
    load.add_argument("--member", default="off", choices=["off", "auto"],
                      help="auto: measure W per crack with MobileSAM "
                           "(member edges must be visible in frame); also "
                           "classifies edge vs center cracks")
    load.add_argument("--load-point", type=float, nargs=2, default=(0.5, 0.0),
                      metavar=("X", "Y"), help="normalised load point")
    load.add_argument("--load-direction-deg", type=float, default=90,
                      help="load direction: 0=right 90=down (default 90)")
    load.add_argument("--kic", type=float, default=31.6,
                      help="fracture toughness K_IC in MPa*sqrt(mm) "
                           "(default 31.6 = 1.0 MPa*sqrt(m), plain concrete)")

    pinn = p.add_argument_group("pinn")
    pinn.add_argument("--bank", type=Path, default=None,
                      help="path to F(a/W) bank json (default bridge/pinn_bank.json)")
    pinn.add_argument("--validate-pinn", action="store_true",
                      help="also run a full PINN training solve on the worst "
                           "crack and compare with the scaling-law value")
    pinn.add_argument("--epochs", type=int, default=6000,
                      help="epochs for --validate-pinn")
    pinn.add_argument("--device", default=None, help="cuda | mps | cpu (auto)")
    pinn.add_argument("--skip-pinn", action="store_true",
                      help="stop after the CV report")

    p.add_argument("--out-dir", type=Path, default=None,
                   help="output directory (default output/<timestamp>_bridge)")
    return p.parse_args()


def build_depth_source(args):
    """Map CLI flags onto a depth-source plugin, or None."""
    name = args.depth
    if name is None:                     # infer from the shortcut flags
        if args.mm_per_pixel is not None:
            name = "manual"
        elif args.ref_mm is not None and args.ref_px is not None:
            name = "reference"
        elif args.standoff_mm is not None and args.focal_px is not None:
            name = "standoff"
        elif args.depth_map is not None:
            name = "depth-file"
        else:
            return None
    if name == "manual":
        return get_depth_source("manual", mm_per_pixel=args.mm_per_pixel)
    if name == "reference":
        return get_depth_source("reference", ref_mm=args.ref_mm,
                                ref_px=args.ref_px)
    if name == "standoff":
        return get_depth_source("standoff", standoff_mm=args.standoff_mm,
                                focal_px=args.focal_px)
    if name == "depth-file":
        if args.focal_px is None:
            sys.exit("--depth depth-file needs --focal-px")
        return get_depth_source("depth-file", depth_path=args.depth_map,
                                focal_px=args.focal_px)
    if name == "monocular":
        if args.focal_px is None:
            sys.exit("--depth monocular needs --focal-px")
        kwargs = {"focal_px": args.focal_px}
        if args.depth_model:
            kwargs["model"] = args.depth_model
        return get_depth_source("monocular", **kwargs)
    return get_depth_source(name)


def build_detector(args):
    kwargs = {"conf": args.det_conf}
    if args.detector in ("ultralytics-seg", "ultralytics-box"):
        if args.weights:
            kwargs["weights"] = args.weights
    elif args.detector == "torchvision":
        kwargs = {"arch": args.arch, "conf": args.det_conf,
                  "checkpoint": args.weights}
    elif args.detector == "classical":
        kwargs = {}
    return get_detector(args.detector, **kwargs)


def assess(ratio):
    if ratio >= 1.0:
        return "CRITICAL: K exceeds K_IC — propagation expected"
    if ratio >= 0.5:
        return "elevated: K above 50% of K_IC"
    return "stable under the given load"


def main():
    args = parse_args()
    if not args.image.exists():
        sys.exit(f"image not found: {args.image}")

    out_dir = args.out_dir or Path("output") / (
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_bridge")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1+2. Plugins + CV analysis ───────────────────────────────────────
    depth_source = build_depth_source(args)
    if depth_source is None:
        print("! no scale source given — geometry will be pixels only and "
              "the K_I stage will be skipped.\n  Provide --mm-per-pixel, "
              "--depth <backend>, --ref-mm/--ref-px, or "
              "--standoff-mm/--focal-px.")
    detector = build_detector(args)

    print(f"[cv] analysing {args.image}  (detector={args.detector}, "
          f"scale={getattr(depth_source, 'name', 'none')}) ...")
    img = cv2.imread(str(args.image))
    if img is None:
        sys.exit(f"cannot read image: {args.image}")
    report, cracks_internal = analyse_frame(
        img, detector=detector, depth_source=depth_source,
        load_point_norm=tuple(args.load_point),
        load_direction_deg=args.load_direction_deg)
    report = {"image": str(args.image), **report}
    mm_per_pixel = report["calibration"]["mm_per_pixel"]
    if mm_per_pixel is not None:
        print(f"[cv] scale: {mm_per_pixel:.4f} mm/px "
              f"({report['calibration']['source']})")

    report_path = out_dir / "crack_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[cv] {report['n_detections']} detection(s) -> "
          f"{report['n_segments']} segment(s) -> "
          f"{len(report['cracks'])} crack(s), "
          f"{len(report['rejected'])} rejected -> {report_path}")

    for flag in report["interactions"]:
        print(f"! cracks #{flag['pair'][0]} and #{flag['pair'][1]} are "
              f"{flag['min_dist_px']} px apart (< {flag['limit_px']} px): "
              f"{flag['note']}")

    if not report["cracks"]:
        print("[cv] nothing structural detected — stopping.")
        return
    if args.skip_pinn or mm_per_pixel is None:
        return
    if args.sigma_mpa is None:
        print("! --sigma-mpa not given — skipping the K_I stage. The far-"
              "field stress is a structural input the CV side cannot see.")
        return

    # ── 3. Every crack -> K_I via the scaling law ─────────────────────────
    bank = load_bank(args.bank) if args.bank else load_bank()
    if bank:
        print(f"[k_i] using PINN bank ({len(bank['entries'])} ratios, "
              f"trained {bank['created']})")
    else:
        print("[k_i] no PINN bank found — using Tada handbook F(a/W). "
              "Build the bank once with:  python -m bridge.build_bank")

    # ── 3b. Optional: measure W per crack from the frame (MobileSAM) ──────
    member = None
    if args.member == "auto":
        member = measure_member(img, cracks_internal,
                                mm_per_pixel=mm_per_pixel)
        if member is None:
            print("! --member auto: MobileSAM found no usable member mask "
                  "— falling back to width defaults")
        else:
            n_lb = sum(r["W_lower_bound"] for r in member["per_crack"].values())
            print(f"[member] W measured for {len(member['per_crack'])} "
                  f"crack(s){f', {n_lb} clipped by frame (lower bound)' if n_lb else ''}")
            report["member"] = {str(k): {kk: vv for kk, vv in r.items()}
                                for k, r in member["per_crack"].items()}
            report_path.write_text(json.dumps(report, indent=2))

    evaluations = []
    for crack in report["cracks"]:
        a_mm = crack["length_mm"]
        crack_type = "edge"
        W_source = "user" if args.member_width_mm else "5a-default"
        W_mm = args.member_width_mm or 5.0 * a_mm
        if member and crack["id"] in member["per_crack"]:
            rec = member["per_crack"][crack["id"]]
            if rec.get("W_mm"):
                W_mm = rec["W_mm"]
                W_source = ("sam-lower-bound" if rec["W_lower_bound"]
                            else "sam-measured")
            crack_type = rec["crack_type"]
        ev = evaluate_crack(a_mm, W_mm, args.sigma_mpa,
                            angular_dev_deg=crack.get("angular_dev_deg") or 0.0,
                            bank=bank, crack_type=crack_type)
        ev["crack_id"] = crack["id"]
        ev["W_mm"] = round(W_mm, 1)
        ev["W_source"] = W_source
        ev["severity"] = round(ev["K_eff_MPa_sqrt_mm"] / args.kic, 3)
        ev["assessment"] = assess(ev["severity"])
        if not ev["lefm_valid"]:
            ev["warning"] = (f"a/W = {ev['a_over_W']} exceeds the LEFM "
                             f"validity limit {LEFM_MAX_RATIO} — treat with caution")
        evaluations.append(ev)

    evaluations.sort(key=lambda e: -e["severity"])
    worst = evaluations[0]

    # ── 4. Ranked severity table ──────────────────────────────────────────
    print("\n  rank  crack  type   a(mm)   W(mm)   a/W    dev(deg)   K_I  "
          "   K_II    K_eff   K/K_IC")
    print("  " + "-" * 86)
    for rank, ev in enumerate(evaluations, start=1):
        print(f"  {rank:>4}  #{ev['crack_id']:<4} {ev['crack_type'][:1].upper()}   "
              f"{ev['a_mm']:>7.1f} {ev['W_mm']:>7.1f}  "
              f"{ev['a_over_W']:>5.2f}  {ev['angular_dev_deg']:>7.1f}  "
              f"{ev['KI_MPa_sqrt_mm']:>7.1f} {ev['KII_MPa_sqrt_mm']:>7.1f} "
              f"{ev['K_eff_MPa_sqrt_mm']:>7.1f}  {ev['severity']:>6.2f}"
              + ("  !LEFM" if not ev["lefm_valid"] else "")
              + ("  W>=" if ev["W_source"] == "sam-lower-bound" else ""))
    print(f"\n  worst crack #{worst['crack_id']}: {worst['assessment']}")

    result = {
        "sigma_MPa": args.sigma_mpa,
        "KIC_MPa_sqrt_mm": args.kic,
        "member_width_policy": (f"{args.member_width_mm} mm (user)"
                                if args.member_width_mm else "5*a per crack"),
        "F_source": worst["F_source"],
        "cracks_ranked": evaluations,
        "interactions": report["interactions"],
        "worst": {"crack_id": worst["crack_id"],
                  "severity": worst["severity"],
                  "assessment": worst["assessment"]},
    }

    # ── Optional: full PINN solve on the worst crack as a cross-check ─────
    if args.validate_pinn:
        a_mm = worst["a_mm"]
        W_mm = worst["W_mm"]
        print(f"\n[pinn] full training solve on worst crack "
              f"(a = {a_mm:.1f} mm, W = {W_mm:.1f} mm) ...")
        full = solve_mode_I(a_mm=a_mm, W_mm=W_mm, H_half_mm=W_mm / 2.0,
                            sigma_MPa=args.sigma_mpa, epochs=args.epochs,
                            device=args.device)
        # Compare against the pure Mode-I scaling value (the full solve
        # models a perpendicular crack — no inclination resolution).
        pure = evaluate_crack(a_mm, W_mm, args.sigma_mpa,
                              angular_dev_deg=0.0, bank=bank)
        dev_pct = (abs(full["KI_param_MPa_sqrt_mm"] - pure["KI_MPa_sqrt_mm"])
                   / full["KI_param_MPa_sqrt_mm"] * 100)
        print(f"[pinn] full solve K_I = {full['KI_param_MPa_sqrt_mm']:.2f} "
              f"vs scaling-law {pure['KI_MPa_sqrt_mm']:.2f}  "
              f"(dev {dev_pct:.1f}%)")
        result["validation"] = {"full_solve": full,
                                "scaling_law_pure_modeI_KI":
                                    pure["KI_MPa_sqrt_mm"],
                                "deviation_pct": round(dev_pct, 2)}

    result_path = out_dir / "severity_report.json"
    result_path.write_text(json.dumps(result, indent=2))
    print(f"\n[done] {result_path}")


if __name__ == "__main__":
    main()
