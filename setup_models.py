#!/usr/bin/env python3
"""One-time setup: install deps and pre-fetch every downloadable model.

For a fresh clone:

    python setup_models.py

This installs requirements.txt, then downloads + warms the two models that
fetch automatically — MobileSAM (~39 MB via ultralytics) and Depth Anything
V2 (~100 MB via HuggingFace). Both cache locally and are NOT re-downloaded
on later runs.

It does NOT supply the YOLO crack detector (`best-2.pt`): that is your own
trained model. Drop your YOLO-seg weights in the repo root as `best-2.pt`,
or pass `--weights /path/to/your.pt` to the run scripts.

    python setup_models.py --no-install   # skip pip, just fetch models
"""

import argparse
import subprocess
import sys
from pathlib import Path

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
ROOT = Path(__file__).resolve().parent


def mark(label, ok, hint=""):
    line = f"  [{'OK ' if ok else 'MISSING'}] {label}"
    if not ok and hint:
        line += f"   -> {hint}"
    print(line)
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-install", action="store_true",
                   help="skip pip install -r requirements.txt")
    args = p.parse_args()

    if not args.no_install:
        print("1. installing requirements.txt ...")
        req = ROOT / "requirements.txt"
        rc = subprocess.run([sys.executable, "-m", "pip", "install",
                             "-r", str(req)]).returncode
        if rc != 0:
            sys.exit("pip install failed — fix the errors above and re-run")
    else:
        print("1. skipping pip install (--no-install)")

    print("\n2. verifying python dependencies")
    missing = []
    for m in ("cv2", "numpy", "scipy", "skimage", "torch", "ultralytics",
              "transformers", "matplotlib", "PIL"):
        try:
            __import__(m)
            mark(m, True)
        except Exception:
            missing.append(m)
            mark(m, False, "see requirements.txt")
    if missing:
        sys.exit(f"\nstill missing: {', '.join(missing)} — install and re-run")

    print("\n3. YOLO crack detector (your own model)")
    yolo = ROOT / "best-2.pt"
    if yolo.exists():
        mark(f"{yolo.name} found", True)
    else:
        mark("best-2.pt", False, "BRING YOUR OWN")
        print("     The crack detector is not shipped. Put your trained "
              "YOLO-seg\n     weights here as best-2.pt, or pass "
              "--weights /path/to/your.pt\n     to run_end_to_end.py / "
              "capture_analyze.py / live_demo.py.")

    print("\n4. warming MobileSAM (auto-downloads mobile_sam.pt, ~39 MB)")
    try:
        from bridge.member_extent import get_sam
        get_sam()
        mark("MobileSAM ready", True)
    except Exception as e:
        mark("MobileSAM", False, str(e))

    print(f"\n5. downloading + warming Depth Anything V2\n   {DEPTH_MODEL}")
    print("   first run pulls ~100 MB into ~/.cache/huggingface ...")
    try:
        import numpy as np
        import torch
        from PIL import Image
        from transformers import pipeline
        dev = ("mps" if torch.backends.mps.is_available()
               else "cuda" if torch.cuda.is_available() else "cpu")
        pipe = pipeline("depth-estimation", model=DEPTH_MODEL, device=dev)
        pipe(Image.fromarray(np.zeros((64, 64, 3), np.uint8)))   # warm inference
        mark(f"Depth Anything V2 ready on {dev}", True)
    except Exception as e:
        mark("Depth Anything V2", False, str(e))

    print("\nsetup complete.")
    if not yolo.exists():
        print("REMINDER: add your best-2.pt (or --weights) before running.")
    print("then:  python capture_analyze.py --camera 0 --sigma-mpa 2.0 "
          "--focal-px 1400")


if __name__ == "__main__":
    main()
