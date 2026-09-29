#!/usr/bin/env python
"""Plot all mass-only shifted Upsilon priors on the same plot.

The input mass here is the stored-energy / E-based pair mass, which is the
quantity the mass-shift tool moves while leaving px,py,pz fixed.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
PLOT_DIR = HERE / "plots"
SHIFTS = [0.0, 0.03, -0.03, 0.05, -0.05]
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


def ebased_mass_from_file(path: Path):
    with h5py.File(path, "r") as handle:
        z = np.asarray(handle["FDL/zData"], dtype=np.float64)
    e = z[:, 3] + z[:, 7]
    p = z[:, :3] + z[:, 4:7]
    m2 = e * e - np.sum(p * p, axis=1)
    m = np.sqrt(np.maximum(m2, 0.0))
    return m[(m >= 8.0) & (m <= 11.5)]


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    bins = np.linspace(8.5, 11.5, 241)
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for shift in SHIFTS:
        if shift == 0.0:
            path = OUT / "true_upsilon1s_subset.hdf5"
        else:
            path = OUT / f"true_upsilon1s_mass_{shift:+.3f}.hdf5"
        if not path.exists():
            raise FileNotFoundError(path)
        masses = ebased_mass_from_file(path)
        ax.hist(masses, bins=bins, histtype="step", density=True,
                color=COLORS[shift], linestyle=STYLES[shift], linewidth=1.8,
                label=rf"{shift*100:+.0f}% mass shift")
    ax.axvline(9.4603, color="gray", linestyle="--", linewidth=1.0,
               label=r"$m_{\Upsilon(1S)}=9.4603$ GeV")
    ax.set_xlim(8.5, 11.5)
    ax.set_xlabel(r"$m_{\mu\mu}$ [GeV] (E-based prior mass)")
    ax.set_ylabel("Normalized density")
    ax.set_title("Mass-only shifted Upsilon(1S) priors")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    out = PLOT_DIR / "mass_shifted_priors.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
