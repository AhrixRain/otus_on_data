#!/usr/bin/env python
"""Plot kinematics-only Upsilon probe results.

(1) decoded mass distributions for each target pair pT on one plot;
(2) prior pair-pT distributions for each target pair pT on one plot.
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
OUT = HERE / "outputs"
DECODED = OUT / "decoded"
PLOT_DIR = HERE / "plots"
MUON_MASS = 0.105658
TARGET_PTS = [0.0, 5.0, 10.0, 20.0, 40.0, 80.0]
COLORS = ["black", "deepskyblue", "forestgreen", "darkorange", "darkviolet", "crimson"]
STYLES = ["-", "--", ":", "-.", "--", "-"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-events", type=int, default=5000)
    return parser.parse_args()


def stable_mass(p4):
    return invariant_mass_np(p4, daughter_masses=(MUON_MASS, MUON_MASS), stable=True)


def ebased_mass(p4):
    e = p4[:, 3] + p4[:, 7]
    p = p4[:, :3] + p4[:, 4:7]
    return np.sqrt(np.maximum(e * e - np.sum(p * p, axis=1), 0.0))


def pair_pt(p4):
    return np.hypot(p4[:, 0] + p4[:, 4], p4[:, 1] + p4[:, 5])


def prior_pair_pt_file(pt: float) -> np.ndarray:
    path = OUT / f"true_upsilon1s_kinematics_pt{pt:g}.hdf5"
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        z = np.asarray(handle["FDL/zData"], dtype=np.float64)
    return pair_pt(z)


def decoded_mass_file(ckpt: str, pt: float) -> np.ndarray:
    path = DECODED / f"{ckpt}_true_upsilon1s_kinematics_pt{pt:g}.hdf5"
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        x = np.asarray(handle["FDL/xData"], dtype=np.float64)
    m = stable_mass(x)
    return m[(m >= 8.5) & (m <= 11.5)]


def main() -> int:
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 1: decoded mass by kinematics shift, one panel per checkpoint.
    for ckpt in ["best", "last"]:
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        bins = np.linspace(8.5, 11.5, 161)
        for pt, color, style in zip(TARGET_PTS, COLORS, STYLES):
            masses = decoded_mass_file(ckpt, pt)
            ax.hist(masses, bins=bins, histtype="step", density=True,
                    color=color, linestyle=style, linewidth=1.8,
                    label=rf"pair $p_T$ = {pt:g} GeV")
        ax.axvline(9.4603, color="gray", linestyle="--", linewidth=1.0,
                   label=r"$m_{\Upsilon(1S)}=9.4603$ GeV")
        ax.set_xlim(8.5, 11.5)
        ax.set_xlabel(r"$m_{\mu\mu}$ [GeV]")
        ax.set_ylabel("Normalized density")
        ax.set_title(f"Run E {ckpt}: decoded mass vs kinematics-only pair-pT shift")
        ax.legend(frameon=False, fontsize=9)
        ax.grid(alpha=0.2)
        out = PLOT_DIR / f"decoded_mass_by_kinematics_shift_{ckpt}.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print("Saved:", out)

    # Figure 2: prior pair-pT distributions by kinematics shift.
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    bins_pt = np.linspace(0, 100, 121)
    for pt, color, style in zip(TARGET_PTS, COLORS, STYLES):
        values = prior_pair_pt_file(pt)
        ax.hist(values, bins=bins_pt, histtype="step", density=True,
                color=color, linestyle=style, linewidth=1.8,
                label=rf"pair $p_T$ target = {pt:g} GeV")
    ax.set_xlabel(r"Prior pair $p_T$ [GeV]")
    ax.set_ylabel("Normalized density")
    ax.set_title("Kinematics-only shifted Upsilon priors: pair pT distributions")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.2)
    out = PLOT_DIR / "kinematics_shifted_priors.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
