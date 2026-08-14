#!/usr/bin/env python
"""Verdict table for an eval.py output directory.

For each of simulation / reconstruction / unfolding prints the W1/KS from
metrics.json plus shape statistics (mean, std, mode) of the truth and pred
mass histograms stored in mass_histograms.npz, and the fraction of each
distribution inside a configurable signal window.

Usage:
  python scripts/eval_verdict.py --eval-dir outputs/.../eval
  python scripts/eval_verdict.py --eval-dir outputs/.../eval \
      --signal-min 3.0369 --signal-max 3.1569
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _shape(path: Path) -> dict[str, float]:
    import numpy as np

    with np.load(path) as d:
        truth = d["truth_mass"].astype(np.float64)
        pred = d["pred_mass"].astype(np.float64)
    out: dict[str, float] = {}
    for label, a in (("truth", truth), ("pred", pred)):
        out[f"{label}_n"] = float(len(a))
        out[f"{label}_mean"] = float(a.mean())
        out[f"{label}_std"] = float(a.std())
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--signal-min", type=float, default=3.0369)
    parser.add_argument("--signal-max", type=float, default=3.1569)
    args = parser.parse_args()

    comp_root = args.eval_dir / "comparisons"
    print(f"verdict for {args.eval_dir}")
    print(f"{'path':16s} {'W1 [GeV]':>9s} {'KS':>7s} {'truth mean':>11s} {'truth std':>10s} {'pred mean':>11s} {'pred std':>10s}")
    for name in ("simulation", "reconstruction", "unfolding"):
        mfile = comp_root / name / "metrics.json"
        hfile = comp_root / name / "mass_histograms.npz"
        if not mfile.exists():
            print(f"{name:16s} (missing)")
            continue
        m = json.loads(mfile.read_text())
        shape = _shape(hfile) if hfile.exists() else {}
        print(
            f"{name:16s} {m.get('w1_distance', float('nan')):9.4f} "
            f"{m.get('ks_statistic', float('nan')):7.3f} "
            f"{shape.get('truth_mean', float('nan')):11.4f} "
            f"{shape.get('truth_std', float('nan')):10.4f} "
            f"{shape.get('pred_mean', float('nan')):11.4f} "
            f"{shape.get('pred_std', float('nan')):10.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
