#!/usr/bin/env python3
"""Train the F(a/W) reference bank — the once-offline PINN work.

    python -m bridge.build_bank --ratios 0.05 0.1 0.2 0.3 --epochs 6000

Each ratio is one full PINN solve at the reference scale (W=100 mm,
H/W=1, sigma=100 MPa — the configuration validated against FEniCS FEM in
the team's rundown). The dimensionless geometry factor

    F(a/W) = K_I / (sigma * sqrt(pi * a))

is what gets stored; by LEFM similarity it applies to every crack with
that a/W at any physical size and load. ~18 min per ratio on Apple MPS at
6000 epochs.
"""

import argparse
import datetime
import json
import math
from pathlib import Path

from .pinn_solver import (DEFAULT_BANK_PATH, sent_geometry_factor,
                          solve_mode_I)

REF_W = 100.0
REF_H_HALF = 50.0
REF_SIGMA = 100.0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ratios", type=float, nargs="+",
                   default=[0.05, 0.10, 0.20, 0.30])
    p.add_argument("--epochs", type=int, default=6000)
    p.add_argument("--device", default=None)
    p.add_argument("--out", type=Path, default=DEFAULT_BANK_PATH)
    args = p.parse_args()

    entries = []
    for ratio in sorted(args.ratios):
        a = ratio * REF_W
        print(f"\n=== a/W = {ratio:.2f}  (a = {a:.1f} mm) ===")
        res = solve_mode_I(a_mm=a, W_mm=REF_W, H_half_mm=REF_H_HALF,
                           sigma_MPa=REF_SIGMA, epochs=args.epochs,
                           device=args.device, log_every=2000)
        denom = REF_SIGMA * math.sqrt(math.pi * a)
        entry = {
            "a_over_W": ratio,
            "F_param": round(res["KI_param_MPa_sqrt_mm"] / denom, 4),
            "F_J": round(res["KI_J_integral_MPa_sqrt_mm"] / denom, 4),
            "F_williams": round(res["KI_williams_fit_MPa_sqrt_mm"] / denom, 4),
            "F_tada": round(sent_geometry_factor(ratio), 4),
        }
        entry["dev_vs_tada_pct"] = round(
            abs(entry["F_param"] - entry["F_tada"]) / entry["F_tada"] * 100, 2)
        print(f"    F_pinn = {entry['F_param']}  F_tada = {entry['F_tada']}  "
              f"(dev {entry['dev_vs_tada_pct']}%)")
        entries.append(entry)

    bank = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "epochs": args.epochs,
        "reference": {"W_mm": REF_W, "H_half_mm": REF_H_HALF,
                      "sigma_MPa": REF_SIGMA},
        "entries": entries,
    }
    args.out.write_text(json.dumps(bank, indent=2))
    print(f"\n[bank] {len(entries)} entries -> {args.out}")


if __name__ == "__main__":
    main()
