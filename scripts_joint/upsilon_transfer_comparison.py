#!/usr/bin/env python
"""Compare the Run F Upsilon-transfer evaluations in one table and one figure.

Reads the standard ``upsilon_transfer_*/quantitative_z_to_x/z_to_x_metrics.json``
and ``decoded_vs_cms/peak_summary.csv`` products, computes the normalized
8.5-9.25 GeV density ratio directly from the decoded HDF5s, writes a markdown
table plus JSON, and draws an overlay mass spectrum with a ratio panel.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import h5py
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = REPO_ROOT / "outputs" / "cms_Joint" / "Run_F"
CMS_CSV = REPO_ROOT / "experiments" / "cms_upsilon" / "data" / "Ymumu.csv"
PRIOR = (
    REPO_ROOT
    / "data"
    / "cms_upsilon_mumu_mg5py8_ckkwl_8tev_inclusive_0j1j_fiducial_8p5_11p5_1M.hdf5"
)
WINDOW = (8.5, 11.2)
REGION = (8.5, 9.25)
BIN_WIDTH = 0.02

SAMPLES = (
    ("best_model (stage-1 ep20, det.)", "upsilon_transfer_best_model", None),
    ("stage-2 best (ep45)", "upsilon_transfer_stage2best", None),
    ("last_model (stage-2 ep100)", "upsilon_transfer_last_model", None),
    (
        "last_model + continuum reweight",
        "upsilon_transfer_continuumReweighted_last",
        RUN_DIR / "upsilon_continuum_reweight" / "decoded_reweighted_last_model.hdf5",
    ),
    ("best_model + continuum reweight", "upsilon_transfer_continuumReweighted_best", None),
)


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


def decoded_mass(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return mass8(handle["FDL/xData"][:])


def normalized_hist(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    density = counts / (counts.sum() * np.diff(edges))
    return density


def region_fraction(values: np.ndarray, lo: float, hi: float) -> float:
    in_window = (values >= WINDOW[0]) & (values < WINDOW[1])
    in_region = (values >= lo) & (values < hi)
    return float(in_region.sum() / in_window.sum())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=RUN_DIR,
        help="run directory used for the default sample table (default: Run_F)",
    )
    parser.add_argument(
        "--sample",
        action="append",
        default=None,
        metavar="LABEL=TRANSFER_DIR",
        help=(
            "override the sample table (repeatable). Each value is a transfer "
            "directory containing decoded/, decoded_vs_cms/ and "
            "quantitative_z_to_x/; relative paths resolve against the cwd."
        ),
    )
    parser.add_argument(
        "--prior",
        type=Path,
        default=PRIOR,
        help="reference truth prior drawn as the grey dashed curve (default: raw 1M prior)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="default: <run-dir>/upsilon_transfer_comparison",
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.run_dir / "upsilon_transfer_comparison"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.sample:
        samples: list[tuple[str, Path, Path | None]] = []
        for raw in args.sample:
            if "=" not in raw:
                raise SystemExit(f"--sample expects LABEL=TRANSFER_DIR, got {raw!r}")
            label, value = raw.split("=", 1)
            decoded_override = None
            if "::" in value:
                value, decoded_override = value.split("::", 1)
            path = Path(value)
            if not path.is_absolute():
                path = (Path.cwd() / path).resolve()
            if decoded_override is not None:
                decoded_path = Path(decoded_override)
                if not decoded_path.is_absolute():
                    decoded_path = (Path.cwd() / decoded_path).resolve()
            else:
                decoded_path = None
            samples.append((label, path, decoded_path))
    else:
        samples = [
            (label, args.run_dir / subdir, override)
            for label, subdir, override in SAMPLES
        ]

    cms = cms_mass(CMS_CSV)
    with h5py.File(args.prior, "r") as handle:
        prior_mass = mass8(handle["FDL/zData"][:])

    edges = np.arange(WINDOW[0], WINDOW[1] + 1e-9, BIN_WIDTH)
    centers = 0.5 * (edges[:-1] + edges[1:])
    cms_hist = normalized_hist(cms, edges)
    prior_hist = normalized_hist(prior_mass, edges)
    cms_ratio_region = region_fraction(cms, *REGION)

    rows = []
    curves = {"CMS data": (cms_hist, "black"), "prior (z)": (prior_hist, "#999999")}
    colors = ["#2b6cb0", "#dd6b20", "#2f855a", "#6b46c1", "#b83280"]
    for (label, base, override), color in zip(samples, itertools.cycle(colors)):
        decoded_path = override or (
            base / "decoded" / "upsilon_0j1j_prior_decoded_xspace.hdf5"
        )
        if not decoded_path.exists():
            continue
        masses = decoded_mass(decoded_path)
        hist = normalized_hist(masses, edges)
        curves[label] = (hist, color)
        metrics = json.loads(
            (base / "quantitative_z_to_x" / "z_to_x_metrics.json").read_text(encoding="utf-8")
        )
        mass = metrics["distribution_report"]["mass"]
        pair_pt = metrics["distribution_report"]["pair_pt"]
        c2st = metrics["distribution_report"]["c2st"]["mlp"]
        peak_rows = (base / "decoded_vs_cms" / "peak_summary.csv").read_text(encoding="utf-8").splitlines()
        first = peak_rows[1].split(",")
        rows.append(
            {
                "sample": label,
                "mass_w1_gev": float(mass["w1_gev"]),
                "mass_ks": float(mass["ks"]),
                "pair_pt_ks": float(pair_pt["ks"]),
                "c2st_mlp": float(c2st["c2st_auc_mean"]),
                "decoded_mass_mean_gev": float(mass["fake_mean_gev"]),
                "decoded_mass_std_gev": float(mass["fake_std_gev"]),
                "upsilon1s_decoded_mean_gev": float(first[6]),
                "upsilon1s_decoded_std_mev": float(first[7]) * 1000.0,
                "density_8p50_9p25_over_cms": region_fraction(masses, *REGION) / cms_ratio_region,
            }
        )

    # ---- figure: spectra + ratio ------------------------------------------
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )
    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(8, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
    )
    top.step(centers, cms_hist, where="mid", color="black", linewidth=1.4, label="CMS data")
    top.step(centers, prior_hist, where="mid", color="#999999", linewidth=1.0, linestyle="--", label="prior (z)")
    for label, (hist, color) in curves.items():
        if label in ("CMS data", "prior (z)"):
            continue
        top.step(centers, hist, where="mid", color=color, linewidth=1.1, label=label, alpha=0.9)
    top.set_ylabel("normalized events / 20 MeV")
    top.set_xlim(*WINDOW)
    top.set_ylim(bottom=0)
    top.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    top.text(0.02, 0.94, "CMS Open Data, 8 TeV", transform=top.transAxes, fontsize=9)

    for label, (hist, color) in curves.items():
        if label == "CMS data":
            continue
        ratio = np.divide(hist, cms_hist, out=np.full_like(hist, np.nan), where=cms_hist > 0)
        bottom.step(centers, ratio, where="mid", color=color, linewidth=1.0, label=label)
    bottom.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
    bottom.set_ylabel("model / CMS")
    bottom.set_xlabel(r"$m_{\mu\mu}$ [GeV]")
    bottom.set_ylim(0.0, 2.6)
    bottom.set_xlim(*WINDOW)
    figure.savefig(args.output_dir / "upsilon_mass_comparison.png", dpi=150, bbox_inches="tight")
    figure.savefig(args.output_dir / "upsilon_mass_comparison.pdf", bbox_inches="tight")
    plt.close(figure)

    # ---- table ------------------------------------------------------------
    lines = [
        "# Run F Upsilon-transfer comparison",
        "",
        "All four evaluations decode the same labelled Upsilon prior; the last row",
        "uses the continuum-reweighted prior. Mass W1/KS, pair-pT KS and C2ST come",
        "from `z_to_x_metrics.json`; the 8.5-9.25 density ratio is computed here",
        "from the decoded mass spectra normalized over 8.5-11.2 GeV.",
        "",
        "| sample | mass W1 [GeV] | mass KS | pair-pT KS | C2ST MLP | decoded mass mean [GeV] | 1S mean [GeV] | 1S std [MeV] | 8.5-9.25 / CMS |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {sample} | {mass_w1_gev:.4f} | {mass_ks:.4f} | {pair_pt_ks:.4f} | "
            "{c2st_mlp:.3f} | {decoded_mass_mean_gev:.4f} | {upsilon1s_decoded_mean_gev:.4f} | "
            "{upsilon1s_decoded_std_mev:.1f} | {density_8p50_9p25_over_cms:.3f} |".format(**row)
        )
    lines += [
        "",
        "CMS fitted 1S mean = 9.4451 GeV, sigma = 84.4 MeV.",
        "",
        "Figure: `upsilon_mass_comparison.png` / `.pdf` (top: normalized spectra;",
        "bottom: ratio to CMS).",
    ]
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "comparison.json").write_text(
        json.dumps({"window": list(WINDOW), "region": list(REGION), "rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
