#!/usr/bin/env python
"""Run the kinematics-only Upsilon probe.

For true Upsilon(1S) events, change the pair pT while keeping the E-based mass
fixed (Lorentz-boost based), decode with frozen Run E checkpoints, and write
summary JSONs used by the kinematics plots.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for directory in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from scripts.physics import invariant_mass_np

MAKE_VARIANT = HERE / "make_kinematics_variant_upsilon.py"
DECODE = REPO_ROOT / "scripts_joint" / "upsilon" / "decode_prior.py"
RUN_DIR = REPO_ROOT / "outputs" / "cms_Joint" / "Run_E"
OUT = HERE / "outputs"
DECODED = OUT / "decoded"
MUON_MASS = 0.105658
TARGET_PTS = [0.0, 5.0, 10.0, 20.0, 40.0, 80.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-events", type=int, default=20000)
    parser.add_argument("--pair-pts", type=float, nargs="+", default=TARGET_PTS)
    parser.add_argument("--checkpoints", nargs="+", choices=["best", "last"], default=["best", "last"])
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("\n$", " ".join(str(item) for item in cmd))
    subprocess.run([str(item) for item in cmd], cwd=REPO_ROOT, check=True)


def stable_mass(p4):
    return invariant_mass_np(p4, daughter_masses=(MUON_MASS, MUON_MASS), stable=True)


def decoded_summary(path: Path) -> dict:
    with h5py.File(path, "r") as handle:
        x = np.asarray(handle["FDL/xData"], dtype=np.float64)
    m = stable_mass(x)
    in_window = (m >= 8.5) & (m <= 11.5)
    mw = m[in_window]
    return {
        "median_gev": float(np.median(mw)) if len(mw) else float("nan"),
        "mean_gev": float(np.mean(mw)) if len(mw) else float("nan"),
        "robust_width_mev": float((np.percentile(mw,75)-np.percentile(mw,25))/1.349*1000.0) if len(mw) else float("nan"),
        "window_retention": float(np.mean(in_window)),
    }


def main() -> int:
    args = parse_args()
    DECODED.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = {"best": RUN_DIR / "best_model.pt", "last": RUN_DIR / "last_model.pt"}

    for checkpoint in args.checkpoints:
        true_decoded = DECODED / f"{checkpoint}_true_upsilon1s.hdf5"
        if not true_decoded.exists():
            make_true = HERE / "make_true_upsilon1s_subset.py"
            subset = OUT / "true_upsilon1s_subset.hdf5"
            run([sys.executable, make_true, "--max-events", args.max_events, "--output", subset])
            run([sys.executable, DECODE, "--prior", subset, "--checkpoint", checkpoint_paths[checkpoint],
                 "--output", true_decoded, "--device", args.device, "--max-events", args.max_events, "--overwrite"])

        rows = []
        for pt in args.pair_pts:
            variant_prior = OUT / f"true_upsilon1s_kinematics_pt{pt:g}.hdf5"
            variant_decoded = DECODED / f"{checkpoint}_true_upsilon1s_kinematics_pt{pt:g}.hdf5"
            run([sys.executable, MAKE_VARIANT, "--pair-pt", str(pt),
                 "--max-events", args.max_events, "--output", variant_prior])
            run([sys.executable, DECODE, "--prior", variant_prior, "--checkpoint", checkpoint_paths[checkpoint],
                 "--output", variant_decoded, "--device", args.device,
                 "--max-events", args.max_events, "--overwrite"])
            summary = decoded_summary(variant_decoded)
            summary["pair_pt_gev"] = float(pt)
            rows.append(summary)

        report = {"checkpoint": checkpoint, "kinematics_probes": rows,
                  "method": "Lorentz boost preserving E-based mass and rapidity; target pair pT changed"}
        out_json = OUT / f"kinematics_{checkpoint}.json"
        out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("\nCheckpoint:", checkpoint)
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
