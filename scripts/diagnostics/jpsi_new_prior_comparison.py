#!/usr/bin/env python
"""Side-by-side comparison: CMS signal-window data vs the OLD MG5 prior vs the NEW mixed MG5 prior.

Loads the version-3 split caches in outputs/cms_JpsiDoubleMuons/archive/.plot_cache:
  - 0ed04817d72bb34f84ed : v3.9 signal-region selection (muon pT in (2,100], |eta|<2.4,
                           mass in [3.0369,3.1569])
  - 410ce075cc7cc2e067c8 : full window [2.6,3.5] (composition sideband estimate)

Prior files:
  - OLD: data/cms_jpsi_mumu_mg5_8tev_1M.hdf5        (signal-only delta, 1M events)
  - NEW: data/cms_jpsi_mumu_mg5_8tev_mixed_ptj5.hdf5 (85/15 mixed, ptj=5, 94,880 events)

Mass convention: training-exact x-space (stable p-based formula, physical muon masses).
Outputs a 3x2 panel figure to experiments/cms_Jpsi_ee/jpsi_new_prior_vs_cms_mumu_ptj5.png
and prints the same summary numbers to stdout.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.cms_data import load_theory_prior_z  # noqa: E402
from scripts.physics import MUON_MASS_GEV, invariant_mass_np  # noqa: E402

CACHE_DIR = ROOT / "outputs/cms_JpsiDoubleMuons/archive/.plot_cache"
SIGNAL_CACHE = "0ed04817d72bb34f84ed"
FULL_CACHE = "410ce075cc7cc2e067c8"
OLD_PRIOR = ROOT / "data/cms_jpsi_mumu_mg5_8tev_1M.hdf5"
NEW_PRIOR = ROOT / "data/cms_jpsi_mumu_mg5_8tev_mixed_ptj5.hdf5"
OUT_PNG = ROOT / "experiments/cms_Jpsi_ee/jpsi_new_prior_vs_cms_mumu_ptj5.png"

MASS_LO, MASS_HI = 3.0369, 3.1569
M_PSI = 3.0969
MU = (MUON_MASS_GEV, MUON_MASS_GEV)


def load_cache(key):
    with np.load(CACHE_DIR / ("selected_split_" + key + ".npz"), allow_pickle=False) as d:
        return {k: np.asarray(d[k]) for k in d.keys()}


def mass(pairs):
    return invariant_mass_np(pairs, daughter_masses=MU, stable=True)


def kin(pairs):
    px1, py1, pz1 = pairs[:, 0], pairs[:, 1], pairs[:, 2]
    px2, py2, pz2 = pairs[:, 4], pairs[:, 5], pairs[:, 6]
    pt1 = np.hypot(px1, py1)
    pt2 = np.hypot(px2, py2)
    eta1 = np.arcsinh(pz1 / np.maximum(pt1, 1e-9))
    eta2 = np.arcsinh(pz2 / np.maximum(pt2, 1e-9))
    pair_pt = np.hypot(px1 + px2, py1 + py2)
    return pt1, pt2, eta1, eta2, pair_pt


def data_selection_mask(pairs):
    pt1, pt2, eta1, eta2, _ = kin(pairs)
    m = mass(pairs)
    return (
        (pt1 > 3) & (pt2 > 3) & (np.abs(eta1) < 2.4) & (np.abs(eta2) < 2.4)
        & (m > MASS_LO) & (m < MASS_HI)
    )


def density_hist(values, n, bins, lo, hi):
    counts, edges = np.histogram(values, bins=bins, range=(lo, hi))
    width = edges[1] - edges[0]
    return counts / (n * width), edges


def step_plot(ax, values, n, bins, lo, hi, color, label, log_y=False, alpha=1.0):
    dens, edges = density_hist(values, n, bins, lo, hi)
    xs = 0.5 * (edges[:-1] + edges[1:])
    if log_y:
        dens = np.clip(dens, 1e-5, None)
    ax.step(xs, dens, where="mid", color=color, linewidth=1.8, label=label, alpha=alpha)


def support_coverage(data, prior):
    lo = np.percentile(prior, 0.5, axis=0)
    hi = np.percentile(prior, 99.5, axis=0)
    inside = np.all((data >= lo) & (data <= hi), axis=1)
    return float(np.mean(inside))


def main():
    print("=" * 78)
    print("CMS DATA vs MG5 PRIORS - J/psi signal window comparison")
    print("=" * 78)

    # ---------------- data (signal window) ----------------
    sig = load_cache(SIGNAL_CACHE)
    x = np.concatenate([sig["x_train"], sig["x_val"], sig["x_test"]]).astype(np.float64)
    mx = mass(x)
    xpt1, xpt2, xeta1, xeta2, xpair_pt = kin(x)
    print("CMS data, signal window [3.0369,3.1569] (v3.9 cache): n = " + f"{len(x):,}")
    print("  mass: mean " + f"{mx.mean():.5f}" + "  std " + f"{mx.std()*1000:.1f}" + " MeV")
    print("  muon pT: median " + f"{np.median(np.concatenate([xpt1, xpt2])):.2f}")
    print("  pair pT: median " + f"{np.median(xpair_pt):.2f}")

    # ---------------- old prior (unfiltered -> filter) ----------------
    zold_full = load_theory_prior_z(OLD_PRIOR).astype(np.float64)
    keep_old = data_selection_mask(zold_full)
    zold = zold_full[keep_old]
    mold = mass(zold)
    opt1, opt2, oeta1, oeta2, opair_pt = kin(zold)
    old_pass = len(zold) / len(zold_full) * 100
    print("OLD MG5 prior: " + f"{len(zold):,}" + " / " + f"{len(zold_full):,}"
          + " events pass the data selection (" + f"{old_pass:.2f}" + "%)")
    print("  mass: mean " + f"{mold.mean():.5f}" + "  std " + f"{mold.std()*1000:.2f}" + " MeV (delta)")
    print("  muon pT: median " + f"{np.median(np.concatenate([opt1, opt2])):.2f}")
    print("  pair pT: median " + f"{np.median(opair_pt):.2f}")

    # ---------------- new prior ----------------
    znew = load_theory_prior_z(NEW_PRIOR).astype(np.float64)
    keep_new = data_selection_mask(znew)
    mnew = mass(znew)
    npt1, npt2, neta1, neta2, npair_pt = kin(znew)
    new_pass = keep_new.mean() * 100
    print("NEW MG5 prior (ptj=5, mixed 85/15): n = " + f"{len(znew):,}" + "; "
          + f"{keep_new.sum():,}" + " pass the data selection (" + f"{new_pass:.2f}" + "%)")
    print("  mass: mean " + f"{mnew.mean():.5f}" + "  std " + f"{mnew.std()*1000:.1f}" + " MeV")
    print("  muon pT: median " + f"{np.median(np.concatenate([npt1, npt2])):.2f}")
    print("  pair pT: median " + f"{np.median(npair_pt):.2f}")

    # ---------------- composition (full window sideband estimate) ----------------
    full = load_cache(FULL_CACHE)
    xfull = np.concatenate([full["x_train"], full["x_val"], full["x_test"]]).astype(np.float64)
    mfull = mass(xfull)
    n_lo = int(np.sum(mfull < MASS_LO))
    n_sig = int(np.sum((mfull >= MASS_LO) & (mfull <= MASS_HI)))
    n_hi = int(np.sum(mfull > MASS_HI))
    bkg_density = (n_lo / (MASS_LO - 2.6) + n_hi / (3.5 - MASS_HI)) / 2.0
    bkg_under = bkg_density * (MASS_HI - MASS_LO)
    cont_frac = bkg_under / n_sig
    print("Full window [2.6,3.5]: low " + f"{n_lo:,}" + " | signal region " + f"{n_sig:,}"
          + " | high " + f"{n_hi:,}")
    print("  linear-sideband continuum fraction under the signal window: " + f"{cont_frac*100:.1f}" + "%")

    # ---------------- support coverage ----------------
    cov_old = support_coverage(x, zold)
    cov_new = support_coverage(x, znew)
    print("Joint [p0.5,p99.5] box coverage of data: old prior " + f"{cov_old*100:.1f}"
          + "% | new prior " + f"{cov_new*100:.1f}" + "%")

    # ================= figure =================
    plt.rcParams["font.size"] = 11
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 12.0), constrained_layout=True)
    c_data, c_old, c_new = "black", "crimson", "steelblue"
    lab_data = "CMS data (signal window)"
    lab_old = "old MG5 prior (filtered, delta)"
    lab_new = "new MG5 prior (ptj=5, 85/15 mixed)"

    # (0,0) mass, full window
    ax = axes[0, 0]
    step_plot(ax, mx, len(x), 60, MASS_LO, MASS_HI, c_data, lab_data)
    step_plot(ax, mnew, len(znew), 60, MASS_LO, MASS_HI, c_new, lab_new)
    ax.axvline(M_PSI, color=c_old, linestyle="--", linewidth=2.0, label=lab_old)
    ax.set_xlabel("m(mumu) [GeV]")
    ax.set_ylabel("events / GeV (norm.)")
    ax.set_title("Pair mass, signal window (2 MeV bins)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)

    # (0,1) mass, zoomed
    ax = axes[0, 1]
    step_plot(ax, mx, len(x), 110, 3.04, 3.15, c_data, lab_data)
    step_plot(ax, mnew, len(znew), 110, 3.04, 3.15, c_new, lab_new)
    ax.axvline(M_PSI, color=c_old, linestyle="--", linewidth=2.0, label=lab_old)
    ax.set_xlabel("m(mumu) [GeV]")
    ax.set_ylabel("events / GeV (norm.)")
    ax.set_title("Pair mass zoomed (1 MeV bins) - width: data " + f"{mx.std()*1000:.0f}"
                 + ", old " + f"{mold.std()*1000:.1f}" + ", new " + f"{mnew.std()*1000:.0f}" + " MeV")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)

    # (1,0) muon pT (log y)
    ax = axes[1, 0]
    step_plot(ax, np.concatenate([xpt1, xpt2]), 2 * len(x), 60, 0.0, 60.0, c_data, lab_data, log_y=True)
    step_plot(ax, np.concatenate([opt1, opt2]), 2 * len(zold), 60, 0.0, 60.0, c_old, lab_old, log_y=True, alpha=0.85)
    step_plot(ax, np.concatenate([npt1, npt2]), 2 * len(znew), 60, 0.0, 60.0, c_new, lab_new, log_y=True)
    ax.set_yscale("log")
    ax.set_xlabel("muon pT [GeV]")
    ax.set_ylabel("events / GeV (norm., log)")
    ax.set_title("Muon pT - CMS trigger floor at 3 GeV (log y)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, which="both")

    # (1,1) pair pT (log y)
    ax = axes[1, 1]
    step_plot(ax, xpair_pt, len(x), 60, 0.0, 80.0, c_data, lab_data, log_y=True)
    step_plot(ax, opair_pt, len(zold), 60, 0.0, 80.0, c_old, lab_old, log_y=True, alpha=0.85)
    step_plot(ax, npair_pt, len(znew), 60, 0.0, 80.0, c_new, lab_new, log_y=True)
    ax.set_yscale("log")
    ax.set_xlabel("pair pT [GeV]")
    ax.set_ylabel("events / GeV (norm., log)")
    ax.set_title("Pair pT (log y)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, which="both")

    # (2,0) muon |eta|
    ax = axes[2, 0]
    step_plot(ax, np.abs(np.concatenate([xeta1, xeta2])), 2 * len(x), 50, 0.0, 2.5, c_data, lab_data)
    step_plot(ax, np.abs(np.concatenate([oeta1, oeta2])), 2 * len(zold), 50, 0.0, 2.5, c_old, lab_old, alpha=0.85)
    step_plot(ax, np.abs(np.concatenate([neta1, neta2])), 2 * len(znew), 50, 0.0, 2.5, c_new, lab_new)
    ax.set_xlabel("muon |eta|")
    ax.set_ylabel("events / unit eta (norm.)")
    ax.set_title("Muon |eta|")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)

    # (2,1) summary table
    ax = axes[2, 1]
    ax.axis("off")
    rows = [
        ("", "DATA", "OLD", "NEW"),
        ("events", f"{len(x):,}", f"{len(zold):,}", f"{len(znew):,}"),
        ("mass mean [GeV]", f"{mx.mean():.4f}", f"{mold.mean():.4f}", f"{mnew.mean():.4f}"),
        ("mass std [MeV]", f"{mx.std()*1000:.1f}", f"{mold.std()*1000:.1f}", f"{mnew.std()*1000:.1f}"),
        ("muon pT median [GeV]",
         f"{np.median(np.concatenate([xpt1, xpt2])):.2f}",
         f"{np.median(np.concatenate([opt1, opt2])):.2f}",
         f"{np.median(np.concatenate([npt1, npt2])):.2f}"),
        ("pair pT median [GeV]", f"{np.median(xpair_pt):.2f}",
         f"{np.median(opair_pt):.2f}", f"{np.median(npair_pt):.2f}"),
        ("passes data selection", "-", f"{old_pass:.2f}%", f"{new_pass:.1f}%"),
        ("data inside prior box", "-", f"{cov_old*100:.1f}%", f"{cov_new*100:.1f}%"),
        ("continuum in window", f"~{cont_frac*100:.0f}%", "0%", "15% (by design)"),
    ]
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            if i == 0 or j == 0:
                ax.text(0.02 + j * 0.33, 0.97 - i * 0.115, cell, transform=ax.transAxes,
                        fontsize=10, fontweight="bold",
                        ha="left" if j == 0 else "center")
            else:
                ax.text(0.02 + j * 0.33, 0.97 - i * 0.115, cell, transform=ax.transAxes,
                        fontsize=10, ha="center")
    ax.set_title("Mismatch summary (measured in this script)", fontsize=12)
    ax.text(0.02, 0.01, "mass convention: x-space stable p-based, m(mu)=0.105658 GeV",
            transform=ax.transAxes, fontsize=8, style="italic", color="gray")

    fig.suptitle(
        "CMS 2012 DoubleMuParked data vs MG5 priors in the J/psi signal window [3.0369, 3.1569] GeV",
        fontsize=13,
    )
    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
    print("Saved: " + str(OUT_PNG))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
