#!/usr/bin/env python
"""Run the mass-only Upsilon probe.

This keeps the true Upsilon(1S) momenta fixed, rescales only the stored
energies to move the E-based mass by +/- a few percent, decodes with frozen
Run E checkpoints, and compares the decoded stable-mass response to the
unshifted true Upsilon decode.
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

MAKE_SHIFT = HERE / "make_mass_shifted_upsilon.py"
DECODE = REPO_ROOT / "scripts_joint" / "upsilon" / "decode_prior.py"
RUN_DIR = REPO_ROOT / "outputs" / "cms_Joint" / "Run_E"
OUT = HERE / "outputs"
DECODED = OUT / "decoded"

MUON_MASS = 0.105658


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-events", type=int, default=20000)
    parser.add_argument("--shifts", type=float, nargs="+",
                        default=[0.03, -0.03, 0.05, -0.05])
    parser.add_argument("--checkpoints", nargs="+", choices=["best", "last"],
                        default=["best", "last"])
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("\n$", " ".join(str(item) for item in cmd))
    subprocess.run([str(item) for item in cmd], cwd=REPO_ROOT, check=True)


def stable_mass(p4):
    return invariant_mass_np(p4, daughter_masses=(MUON_MASS, MUON_MASS), stable=True)


def decoded_mass_summary(path: Path) -> dict:
    with h5py.File(path, "r") as handle:
        x = np.asarray(handle["FDL/xData"], dtype=np.float64)
    m = stable_mass(x)
    m = m[(m >= 8.5) & (m <= 11.5)]
    return {
        "median_gev": float(np.median(m)),
        "mean_gev": float(np.mean(m)),
        "robust_width_mev": float((np.percentile(m,75)-np.percentile(m,25))/1.349*1000.0),
        "window_retention": float(np.mean((stable_mass(x)>=8.5)&(stable_mass(x)<=11.5))),
    }


def main() -> int:
    args = parse_args()
    DECODED.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = {"best": RUN_DIR / "best_model.pt", "last": RUN_DIR / "last_model.pt"}

    for checkpoint in args.checkpoints:
        true_decoded = DECODED / f"{checkpoint}_true_upsilon1s.hdf5"
        if not true_decoded.exists():
            # Build the same true subset used by the J/psi scaling probe.
            make_true = HERE / "make_true_upsilon1s_subset.py"
            run([sys.executable, make_true, "--max-events", args.max_events,
                 "--output", OUT / "true_upsilon1s_subset.hdf5"])
            run([sys.executable, DECODE, "--prior", OUT / "true_upsilon1s_subset.hdf5",
                 "--checkpoint", checkpoint_paths[checkpoint],
                 "--output", true_decoded, "--device", args.device,
                 "--max-events", args.max_events, "--overwrite"])

        true_summary = decoded_mass_summary(true_decoded)
        rows = []
        for shift in args.shifts:
            shifted_prior = OUT / f"true_upsilon1s_mass_{shift:+.3f}.hdf5"
            shifted_decoded = DECODED / f"{checkpoint}_true_upsilon1s_mass_{shift:+.3f}.hdf5"
            run([sys.executable, MAKE_SHIFT, "--shift", str(shift),
                 "--max-events", args.max_events, "--output", shifted_prior])
            run([sys.executable, DECODE, "--prior", shifted_prior,
                 "--checkpoint", checkpoint_paths[checkpoint],
                 "--output", shifted_decoded, "--device", args.device,
                 "--max-events", args.max_events, "--overwrite"])
            summary = decoded_mass_summary(shifted_decoded)
            summary["shift"] = float(shift)
            summary["median_response_mev"] = (summary["median_gev"] - true_summary["median_gev"]) * 1000.0
            summary["mean_response_mev"] = (summary["mean_gev"] - true_summary["mean_gev"]) * 1000.0
            rows.append(summary)

        report = {
            "checkpoint": checkpoint,
            "unshifted_true_decoded": true_summary,
            "mass_only_probes": rows,
            "method": "stored energies rescaled so E-based pair mass moves by shift; px/py/pz unchanged",
        }
        out_json = OUT / f"mass_only_{checkpoint}.json"
        out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("\nCheckpoint:", checkpoint)
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
