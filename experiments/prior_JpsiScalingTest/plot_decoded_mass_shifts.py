#!/usr/bin/env python
"""Plot decoded Upsilon(1S) mass distributions for all mass shifts on one plot.

For each Run E checkpoint, overlay the decoded stable-mass distributions of:
  0%, +3%, -3%, +5%, -5% mass-only shifted true Upsilon priors.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for directory in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from scripts.physics import invariant_mass_np

HERE = Path(__file__).resolve().parent
DECODED = HERE / "outputs" / "decoded"
PLOT_DIR = HERE / "plots"
MUON_MASS = 0.105658
MASS_MIN = 8.5
MASS_MAX = 11.5
SHIFTS = [0.0, 0.03, -0.03, 0.05, -0.05]
CHECKPOINTS = ["best", "last"]
COLORS = {
    0.0: "black",
    0.03: "deepskyblue",
    -0.03: "forestgreen",
    0.05: "darkviolet",
    -0.05: "darkorange",
}
STYLES = {
    0.0: "-",
    0.03: "--",
    -0.03: ":",
    0.05: "-.",
    -0.05: "--",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", choices=CHECKPOINTS, default=CHECKPOINTS)
    parser.add_argument("--bins", type=int, default=161)
    return parser.parse_args()


def stable_mass(p4):
    return invariant_mass_np(p4, daughter_masses=(MUON_MASS, MUON_MASS), stable=True)


def decoded_mass_file(ckpt: str, shift: float) -> np.ndarray:
    if shift == 0.0:
        path = DECODED / f"{ckpt}_true_upsilon1s.hdf5"
    else:
        path = DECODED / f"{ckpt}_true_upsilon1s_mass_{shift:+.3f}.hdf5"
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        x = np.asarray(handle["FDL/xData"], dtype=np.float64)
    m = stable_mass(x)
    return m[(m >= MASS_MIN) & (m <= MASS_MAX)]


def main() -> int:
    args = parse_args()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = np.linspace(MASS_MIN, MASS_MAX, args.bins + 1)
    for ckpt in args.checkpoints:
        fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
        for shift in SHIFTS:
            masses = decoded_mass_file(ckpt, shift)
            ax.hist(
                masses,
                bins=bins,
                histtype="step",
                density=True,
                color=COLORS[shift],
                linestyle=STYLES[shift],
                linewidth=1.8,
                label=rf"{shift*100:+.0f}% mass shift",
            )
        ax.axvline(9.4603, color="gray", linestyle="--", linewidth=1.0,
                   label=r"$m_{\Upsilon(1S)}=9.4603$ GeV")
        ax.set_xlim(MASS_MIN, MASS_MAX)
        ax.set_xlabel(r"$m_{\mu\mu}$ [GeV]")
        ax.set_ylabel("Normalized density")
        ax.set_title(f"Run E {ckpt}: decoded Upsilon(1S) mass for mass-only shifts")
        ax.legend(frameon=False, fontsize=9)
        ax.grid(alpha=0.2)
        out = PLOT_DIR / f"decoded_mass_by_shift_{ckpt}.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print("Saved:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
