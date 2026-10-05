#!/usr/bin/env python
"""Paperstyle Upsilon transfer comparison: A2frozen vs D3b, one panel each.

Read-only.  Draws the decoded x-space dimuon mass spectra of two checkpoints
against the CMS data on the same continuum-reweighted prior, with the ratio
panel underneath.  This is the figure the transfer claim needs after D3b: the
inclusive spectrum is closed by both arms, so the discriminating information is
the peak height/width and the low-mass shoulder.

    python scripts_joint/plot_upsilon_arm_comparison.py

Nothing is trained; no checkpoint, data file or existing output is modified.
New files are written to --output-dir only.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import h5py  # noqa: E402

HERE = Path(__file__).resolve().parent


def find_repo_root() -> Path:
    for candidate in HERE.parents:
        if (candidate / "scripts_joint").is_dir() and (candidate / "scripts_sota").is_dir():
            return candidate
    raise RuntimeError("Could not locate the OTUS repository root")


REPO_ROOT = find_repo_root()

MASS_RANGE = (8.5, 11.2)
BINS = 135


def mass8(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    energy = values[:, 3] + values[:, 7]
    momentum = values[:, :3] + values[:, 4:7]
    return np.sqrt(np.maximum(energy * energy - np.sum(momentum * momentum, axis=1), 0.0))


def load_cms(path: Path) -> np.ndarray:
    """Read the CMS Upsilon sample and build [p1, E1, p2, E2] four-vectors.

    The CSV is the two-muon export (E1/px1/.../E2/px2/...) used by
    `compare_prior_cms.py`; the invariant mass uses the same formula.
    """
    columns = ("E1", "px1", "py1", "pz1", "E2", "px2", "py2", "pz2")
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        index = {name.strip(): position for position, name in enumerate(header)}
        missing = [name for name in columns if name not in index]
        if missing:
            raise KeyError(f"{path} is missing columns {missing}")
        rows = []
        for row in reader:
            if len(row) < len(header):
                continue
            try:
                rows.append([float(row[index[name]]) for name in columns])
            except ValueError:
                continue
    raw = np.asarray(rows, dtype=np.float64)
    # [E1, px1, py1, pz1, E2, px2, py2, pz2] -> [p1, E1, p2, E2]
    values = np.column_stack([raw[:, 1:4], raw[:, 0], raw[:, 5:8], raw[:, 4]])
    masses = mass8(values)
    return masses[(masses >= MASS_RANGE[0]) & (masses <= MASS_RANGE[1])]


def load_decoded(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as source:
        values = np.asarray(source["FDL/xData"][:], dtype=np.float64)
    masses = mass8(values)
    return masses[(masses >= MASS_RANGE[0]) & (masses <= MASS_RANGE[1])]


def hist(masses: np.ndarray, edges: np.ndarray, *, normalize: bool = True):
    counts, _ = np.histogram(masses, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    if normalize:
        integral = counts.sum() * width
        density = counts / integral if integral > 0 else counts
    else:
        density = counts
    return centers, density, counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--a2frozen",
        type=Path,
        default=REPO_ROOT
        / "outputs/cms_Joint/Run_H_A2frozen/upsilon_transfer_continuumReweighted_stage3_best/decoded",
    )
    parser.add_argument(
        "--d3b",
        type=Path,
        default=REPO_ROOT
        / "outputs/cms_Joint/Run_H_D3b/upsilon_transfer_continuumReweighted_best_model/decoded",
    )
    parser.add_argument(
        "--prior",
        type=Path,
        default=REPO_ROOT / "data/upsilon_prior_continuumReweighted.hdf5",
    )
    parser.add_argument(
        "--cms",
        type=Path,
        default=REPO_ROOT / "experiments/cms_upsilon/data/Ymumu.csv",
    )
    parser.add_argument("--component-dataset", default="FDL/component_id")
    parser.add_argument("--prior-dataset", default="FDL/zData")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/cms_Joint/Run_H_D3b/upsilon_arm_comparison",
    )
    args = parser.parse_args()

    def resolve(folder: Path) -> Path:
        if folder.is_file():
            return folder
        candidates = sorted(folder.glob("*xspace*.hdf5")) or sorted(folder.glob("*.hdf5"))
        if not candidates:
            raise FileNotFoundError(f"no decoded hdf5 under {folder}")
        return candidates[0]

    a2_path = resolve(args.a2frozen)
    d3b_path = resolve(args.d3b)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    edges = np.linspace(MASS_RANGE[0], MASS_RANGE[1], BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]

    cms_masses = load_cms(args.cms)
    a2_masses = load_decoded(a2_path)
    d3_masses = load_decoded(d3b_path)

    _, cms_density, cms_counts = hist(cms_masses, edges)
    _, a2_density, a2_counts = hist(a2_masses, edges)
    _, d3_density, d3_counts = hist(d3_masses, edges)

    with h5py.File(args.prior, "r") as source:
        prior_z = np.asarray(source[args.prior_dataset][:], dtype=np.float64)
        component = np.asarray(source[args.component_dataset][:])
    prior_masses = mass8(prior_z)
    window = (prior_masses >= MASS_RANGE[0]) & (prior_masses <= MASS_RANGE[1])
    components = {
        "1S": int(component[window][prior_masses[window] < 9.7].size),
        "2S": int(((prior_masses[window] >= 9.7) & (prior_masses[window] < 10.2)).sum()),
        "3S": int((prior_masses[window] >= 10.2).sum()),
    }

    fig, (ax, axr) = plt.subplots(
        2,
        1,
        figsize=(11.0, 8.6),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.06},
    )

    # Both samples are scaled to the CMS total in the window, so the y axis is
    # in CMS-like expected events and the ratio panel is a shape comparison.
    cms_total = float(cms_counts.sum())
    scale_a2 = cms_total / max(float(a2_counts.sum()), 1.0)
    scale_d3 = cms_total / max(float(d3_counts.sum()), 1.0)

    ax.errorbar(
        centers,
        cms_counts,
        yerr=np.sqrt(np.maximum(cms_counts, 1.0)),
        fmt="o",
        ms=3.0,
        lw=1.0,
        color="black",
        label=f"CMS 2012 open data ({int(cms_total)} events in window)",
        zorder=3,
    )
    ax.step(
        centers,
        a2_counts * scale_a2,
        where="mid",
        lw=2.0,
        color="#4c72b0",
        label="A2frozen stage-3 best (g110), scaled to CMS",
        zorder=2,
    )
    ax.step(
        centers,
        d3_counts * scale_d3,
        where="mid",
        lw=2.0,
        color="#c44e52",
        label="D3b best_model (g180), scaled to CMS",
        zorder=2,
    )
    for _, mass in (("1S", 9.4603), ("2S", 10.0233), ("3S", 10.3552)):
        ax.axvline(mass, color="0.6", ls=":", lw=0.9, zorder=1)
    ax.text(
        0.015,
        0.965,
        "both samples scaled to the CMS yield in the window; the 1S peak is the\n"
        "discriminating feature (height, width, low-mass shoulder)",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color="0.25",
    )
    ax.set_ylabel(f"Events / {width * 1000:.0f} MeV")
    ax.set_ylim(0, max(float(cms_counts.max()), float(a2_counts.max()) * scale_a2,
                      float(d3_counts.max()) * scale_d3) * 1.35)
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    ax.set_title("Zero-shot Upsilon transfer on one prior: A2frozen vs D3b")

    axr.plot(centers, a2_counts * scale_a2 / np.maximum(cms_counts, 1e-12), lw=1.6, color="#4c72b0")
    axr.plot(centers, d3_counts * scale_d3 / np.maximum(cms_counts, 1e-12), lw=1.6, color="#c44e52")
    axr.axhline(1.0, color="0.4", lw=1.0, ls="--")
    axr.set_ylabel("Ratio to CMS")
    axr.set_xlabel(r"$m_{\mu\mu}$ [GeV]")
    axr.set_ylim(0.0, 2.4)
    axr.grid(alpha=0.2)

    png = args.output_dir / "upsilon_arm_comparison_peaknormalised.png"
    pdf = args.output_dir / "upsilon_arm_comparison_peaknormalised.pdf"
    fig.savefig(png, dpi=160, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    def peak_region(masses: np.ndarray) -> dict:
        selection = (masses > 9.3) & (masses < 9.62)
        values = masses[selection]
        return {
            "n": int(values.size),
            "median_gev": float(np.median(values)) if values.size else None,
            "std_gev": float(np.std(values)) if values.size else None,
            "robust_half_width_gev": float(
                (np.quantile(values, 0.84) - np.quantile(values, 0.16)) / 2.0
            )
            if values.size
            else None,
        }

    payload = {
        "schema_version": 1,
        "diagnostic": "Upsilon transfer arm comparison (paperstyle)",
        "prior": str(args.prior),
        "cms": str(args.cms),
        "mass_range_gev": list(MASS_RANGE),
        "bin_width_gev": width,
        "prior_component_counts_in_window": components,
        "cms_1s_region": peak_region(cms_masses),
        "a2frozen_1s_region": peak_region(a2_masses),
        "d3b_1s_region": peak_region(d3_masses),
        "median_ratio_1s_region": {
            "a2frozen": float(
                np.median(a2_masses[(a2_masses > 9.3) & (a2_masses < 9.62)])
            ),
            "d3b": float(np.median(d3_masses[(d3_masses > 9.3) & (d3_masses < 9.62)])),
        },
        "files": [str(png), str(pdf)],
    }
    (args.output_dir / "arm_comparison.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    print(f"wrote {png}")
    print(f"wrote {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
