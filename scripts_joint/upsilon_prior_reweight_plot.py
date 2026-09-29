#!/usr/bin/env python
"""Paper-style comparison of the original and continuum-reweighted Upsilon priors.

Panels:
  1. total prior density, original vs reweighted vs CMS data;
  2. continuum component, original vs reweighted vs CMS background;
  3. ratio to the original prior, with the fitted sideband correction overlaid.

The reweighted prior is a resampled version of the original, produced by
``upsilon_continuum_reweight.py``; both are normalized over 8.5-11.2 GeV.
"""

from __future__ import annotations

import argparse
import json
from math import erf, sqrt
from pathlib import Path

import h5py
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORIGINAL = (
    REPO_ROOT
    / "data"
    / "cms_upsilon_mumu_mg5py8_ckkwl_8tev_inclusive_0j1j_fiducial_8p5_11p5_1M.hdf5"
)
DEFAULT_REWEIGHTED = (
    REPO_ROOT
    / "outputs"
    / "cms_Joint"
    / "Run_F"
    / "upsilon_continuum_reweight"
    / "upsilon_prior_continuumReweighted.hdf5"
)
DEFAULT_CMS = REPO_ROOT / "experiments" / "cms_upsilon" / "data" / "Ymumu.csv"
FIT_SUMMARY = (
    REPO_ROOT
    / "outputs"
    / "cms_Joint"
    / "Run_F"
    / "upsilon_transfer_best_model"
    / "decoded_vs_cms"
    / "comparison_summary.json"
)
WINDOW = (8.5, 11.2)
SIDEBANDS = ((8.5, 9.1), (10.7, 11.2))
REGION = (8.5, 9.25)
BIN_WIDTH = 0.02
CONTINUUM_ID = 3


def mass8(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    energy = values[:, 3] + values[:, 7]
    momentum = values[:, :3] + values[:, 4:7]
    return np.sqrt(np.maximum(energy**2 - np.sum(momentum**2, axis=1), 0.0))


def cms_mass(path: Path) -> np.ndarray:
    table = np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8")
    energy = table["E1"] + table["E2"]
    px = table["px1"] + table["px2"]
    py = table["py1"] + table["py2"]
    pz = table["pz1"] + table["pz2"]
    return np.sqrt(np.maximum(energy**2 - px**2 - py**2 - pz**2, 0.0))


def phi(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def peak_histogram(edges: np.ndarray, peaks: list[dict]) -> np.ndarray:
    counts = np.zeros(len(edges) - 1)
    for peak in peaks:
        mu = float(peak["cms_fitted_mass_gev"])
        sigma = float(peak["cms_sigma_gev"])
        yield_in_window = float(peak["cms_fitted_yield"])
        norm = phi((WINDOW[1] - mu) / sigma) - phi((WINDOW[0] - mu) / sigma)
        if norm <= 0:
            continue
        for index in range(len(counts)):
            lo, hi = edges[index], edges[index + 1]
            counts[index] += yield_in_window * (phi((hi - mu) / sigma) - phi((lo - mu) / sigma)) / norm
    return counts


def normalized(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    return counts / (counts.sum() * np.diff(edges))


def region_fraction(values: np.ndarray, lo: float, hi: float) -> float:
    in_window = (values >= WINDOW[0]) & (values < WINDOW[1])
    in_region = (values >= lo) & (values < hi)
    return float(in_region.sum() / in_window.sum())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--reweighted", type=Path, default=DEFAULT_REWEIGHTED)
    parser.add_argument("--cms", type=Path, default=DEFAULT_CMS)
    parser.add_argument("--fit-summary", type=Path, default=FIT_SUMMARY)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "cms_Joint" / "Run_F" / "upsilon_continuum_reweight",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.original, "r") as handle:
        original_z = handle["FDL/zData"][:]
        original_id = handle["FDL/component_id"][:]
    with h5py.File(args.reweighted, "r") as handle:
        reweighted_z = handle["FDL/zData"][:]
        reweighted_id = handle["FDL/component_id"][:]

    original_mass = mass8(original_z)
    reweighted_mass = mass8(reweighted_z)
    original_cont = original_mass[original_id == CONTINUUM_ID]
    reweighted_cont = reweighted_mass[reweighted_id == CONTINUUM_ID]

    cms = cms_mass(args.cms)
    edges = np.arange(WINDOW[0], WINDOW[1] + 1e-9, BIN_WIDTH)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fit = json.loads(args.fit_summary.read_text(encoding="utf-8"))
    cms_hist = normalized(cms, edges)
    cms_background = normalized(cms, edges)  # placeholder, recomputed as counts below
    peak_counts = peak_histogram(edges, fit.get("cms_peak_fit", []))
    cms_counts, _ = np.histogram(cms, bins=edges)
    cms_background_counts = np.clip(cms_counts - peak_counts, 0.0, None)
    cms_background = cms_background_counts / (
        cms_background_counts.sum() * np.diff(edges)
    )

    orig_total = normalized(original_mass, edges)
    rew_total = normalized(reweighted_mass, edges)
    orig_cont = normalized(original_cont, edges)
    rew_cont = normalized(reweighted_cont, edges)
    ratio_total = np.divide(rew_total, orig_total, out=np.full_like(rew_total, np.nan), where=orig_total > 0)
    ratio_cont = np.divide(rew_cont, orig_cont, out=np.full_like(rew_cont, np.nan), where=orig_cont > 0)

    # fitted weight curve, if the diagnostic JSON is present
    weight_curve = None
    weight_json = args.output_dir / "continuum_reweight.json"
    if weight_json.exists():
        data = json.loads(weight_json.read_text(encoding="utf-8"))
        weight_curve = data.get("samples", {}).get("last_model", {}).get("weight_curve")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "legend.frameon": False,
        }
    )
    figure, (top, middle, bottom) = plt.subplots(
        3,
        1,
        figsize=(8, 8.6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 3, 2], "hspace": 0.08},
    )
    for axis in (top, middle):
        for lo, hi in SIDEBANDS:
            axis.axvspan(lo, hi, color="#f2f2f2", zorder=0)

    top.step(centers, orig_total, where="mid", color="#2b6cb0", linewidth=1.3, label="original prior (total)")
    top.step(centers, rew_total, where="mid", color="#dd6b20", linewidth=1.3, label="reweighted prior (total)")
    top.step(centers, cms_hist, where="mid", color="black", linewidth=1.1, linestyle=":", label="CMS data")
    top.set_ylabel("normalized / 20 MeV")
    top.set_ylim(bottom=0)
    top.legend(loc="upper right", fontsize=9)
    top.text(0.02, 0.93, "all components", transform=top.transAxes, fontsize=10)

    middle.step(centers, orig_cont, where="mid", color="#2b6cb0", linewidth=1.3, label="original continuum")
    middle.step(centers, rew_cont, where="mid", color="#dd6b20", linewidth=1.3, label="reweighted continuum")
    middle.step(centers, cms_background, where="mid", color="black", linewidth=1.1, linestyle=":", label="CMS background (data $-$ peaks)")
    middle.set_ylabel("normalized / 20 MeV")
    middle.set_ylim(bottom=0)
    middle.legend(loc="upper right", fontsize=9)
    middle.text(0.02, 0.93, "continuum component", transform=middle.transAxes, fontsize=10)

    bottom.plot(centers, ratio_cont, color="#dd6b20", linewidth=1.4, label="reweighted / original (continuum)")
    bottom.plot(centers, ratio_total, color="#2b6cb0", linewidth=1.0, linestyle="--", label="reweighted / original (total)")
    if weight_curve is not None:
        bottom.plot(
            weight_curve["mass_gev"],
            weight_curve["cms_over_continuum_ratio"],
            color="black",
            linewidth=0.9,
            linestyle=":",
            label="fitted sideband ratio",
        )
    bottom.axhline(1.0, color="black", linewidth=0.7)
    for lo, hi in SIDEBANDS:
        bottom.axvspan(lo, hi, color="#f2f2f2", zorder=0)
    bottom.set_ylabel("shape ratio")
    bottom.set_xlabel(r"$m_{\mu\mu}$ [GeV]")
    bottom.set_xlim(*WINDOW)
    bottom.set_ylim(0.0, 2.4)
    bottom.legend(loc="lower right", fontsize=8, ncol=2)
    figure.savefig(args.output_dir / "prior_reweight_comparison.png", dpi=150, bbox_inches="tight")
    figure.savefig(args.output_dir / "prior_reweight_comparison.pdf", bbox_inches="tight")
    plt.close(figure)

    rows = []
    for name, values in (
        ("original total", original_mass),
        ("reweighted total", reweighted_mass),
        ("original continuum", original_cont),
        ("reweighted continuum", reweighted_cont),
    ):
        rows.append(
            {
                "sample": name,
                "density_8p50_9p25_over_window": region_fraction(values, *REGION),
                "mean_gev": float(values.mean()),
                "std_gev": float(values.std()),
            }
        )
    lines = [
        "# Original vs continuum-reweighted Upsilon prior",
        "",
        "Both normalized over 8.5-11.2 GeV. CMS background = data minus the fitted",
        "1S/2S/3S Gaussians. Shaded bands are the sidebands used to fit the weight.",
        "",
        "| sample | 8.5-9.25 / window | mean [GeV] | std [GeV] |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {sample} | {density_8p50_9p25_over_window:.4f} | {mean_gev:.4f} | {std_gev:.4f} |".format(**row)
        )
    lines += [
        "",
        "Figure: `prior_reweight_comparison.png` / `.pdf`.",
        "Top: total; middle: continuum; bottom: reweighted/original shape ratio",
        "with the fitted sideband ratio overlaid.",
    ]
    (args.output_dir / "prior_reweight_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "prior_reweight_comparison.json").write_text(
        json.dumps({"window": list(WINDOW), "region": list(REGION), "rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
