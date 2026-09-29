#!/usr/bin/env python
"""Upsilon continuum-shape diagnostic: does a sideband-derived reweight close
the 8.5-9.25 GeV excess?

Context (2026-09-10). The Run F Upsilon transfer shows the prior/decoded
spectrum is ~1.7x the CMS data in 8.5-9.25 GeV and ~0.7x above 10.5 GeV, and
the excess is 97% continuum component. This script asks the diagnostic
question: if the continuum shape is corrected to the CMS sidebands, does the
total spectrum close -- i.e. is the excess prior-shape rather than response?

Method
------
1. CMS background per bin = data - fitted 1S/2S/3S Gaussians (the same fit as
   the transfer comparison).
2. For a decoded sample (which retains the input ``FDL/zData`` and
   ``component_id`` in the same order as the prior), build the continuum
   density from the *decoded* mass and the signal density from the decoded
   signal components.
3. Fit ``log(density)`` vs mass with a quadratic on two resonance-free
   sidebands (default 8.5-9.1 and 10.7-11.2 GeV) for both CMS background and
   decoded continuum; the weight is ``R(m) = exp(fit_cms - fit_cont)``.
4. Reweight the continuum events by ``R`` (signal untouched), preserving the
   continuum total so the correction is shape-only, and compare the total to
   CMS in 8.5-9.25 GeV and its sub-bins.

The weight is derived only from the sidebands. 8.5-9.1 is inside a fitted
sideband, 9.1-9.25 is interpolated, and >10.7 is fitted -- the report keeps
those three regions separate so the closure is not self-fulfilling.

``--write-reweighted-prior`` additionally resamples the prior with the same
weights for a fresh decode with the project's ``upsilon/decode_prior.py``.
"""

from __future__ import annotations

import argparse
import json
from math import erf, sqrt
from pathlib import Path

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "outputs" / "cms_Joint" / "Run_F"
DEFAULT_PRIOR = (
    REPO_ROOT
    / "data"
    / "cms_upsilon_mumu_mg5py8_ckkwl_8tev_inclusive_0j1j_fiducial_8p5_11p5_1M.hdf5"
)
DEFAULT_CMS = REPO_ROOT / "experiments" / "cms_upsilon" / "data" / "Ymumu.csv"
CONTINUUM_ID = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--cms", type=Path, default=DEFAULT_CMS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--window", type=float, nargs=2, default=(8.5, 11.2))
    parser.add_argument("--region", type=float, nargs=2, default=(8.5, 9.25))
    parser.add_argument(
        "--sidebands",
        type=float,
        nargs=4,
        default=(8.5, 9.1, 10.7, 11.2),
        metavar=("LO1", "HI1", "LO2", "HI2"),
    )
    parser.add_argument("--bin-width", type=float, default=0.02)
    parser.add_argument(
        "--write-reweighted-prior",
        type=Path,
        default=None,
        help="Resample the prior with the last-model weights and write it here.",
    )
    parser.add_argument(
        "--seed", type=int, default=20260910
    )
    return parser.parse_args()


# ----------------------------------------------------------------------------
# mass / histogram helpers
# ----------------------------------------------------------------------------

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


def peak_histogram(edges: np.ndarray, peaks: list[dict], window: tuple[float, float]) -> np.ndarray:
    """Per-bin counts of the fitted Gaussians inside ``window``."""
    counts = np.zeros(len(edges) - 1)
    for peak in peaks:
        mu = float(peak["cms_fitted_mass_gev"])
        sigma = float(peak["cms_sigma_gev"])
        yield_in_window = float(peak["cms_fitted_yield"])
        norm = phi((window[1] - mu) / sigma) - phi((window[0] - mu) / sigma)
        if norm <= 0:
            continue
        for index in range(len(counts)):
            lo, hi = edges[index], edges[index + 1]
            fraction = phi((hi - mu) / sigma) - phi((lo - mu) / sigma)
            counts[index] += yield_in_window * fraction / norm
    return counts


def histogram(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    return counts.astype(np.float64)


def normalized_density(counts: np.ndarray, edges: np.ndarray) -> np.ndarray:
    width = np.diff(edges)
    area = float((counts * width).sum())
    if area <= 0:
        return np.zeros_like(counts)
    return counts / area


def fit_log_density(
    edges: np.ndarray,
    values: np.ndarray,
    sidebands: tuple[float, float, float, float],
    degree: int = 2,
) -> np.poly1d:
    """Quadratic fit of log(density) vs mass on the sidebands."""
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = (
        ((centers >= sidebands[0]) & (centers < sidebands[1]))
        | ((centers >= sidebands[2]) & (centers < sidebands[3]))
    )
    density = normalized_density(values, edges)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_density = np.log(np.clip(density, 1e-12, None))
    weights = np.sqrt(np.clip(values, 1.0, None))
    coefficients = np.polyfit(
        centers[mask], log_density[mask], degree, w=weights[mask]
    )
    return np.poly1d(coefficients)


def subregion_metrics(
    masses: np.ndarray,
    weights: np.ndarray | None,
    edges: np.ndarray,
    window: tuple[float, float],
    region: tuple[float, float],
    bins: list[tuple[float, float]],
) -> dict:
    if weights is None:
        weights = np.ones_like(masses)
    in_window = (masses >= window[0]) & (masses < window[1])
    w = weights[in_window]
    density = {}
    for lo, hi in bins:
        sel = (masses >= lo) & (masses < hi)
        numerator = float(weights[sel].sum())
        denominator = float((w.sum()) * (window[1] - window[0]))
        density[f"{lo:.2f}-{hi:.2f}"] = numerator / denominator if denominator else 0.0
    region_sel = (masses >= region[0]) & (masses < region[1])
    region_fraction = float(weights[region_sel].sum() / w.sum()) if w.sum() else 0.0
    return {"density": density, "region_fraction": region_fraction}


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: list[float]) -> np.ndarray:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    cumulative = (cumulative - 0.5 * weights) / cumulative[-1]
    return np.interp(quantiles, cumulative, values)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def resolve_samples(run_dir: Path) -> dict[str, Path]:
    samples = {}
    for label, subdir in (
        ("best_model", "upsilon_transfer_best_model"),
        ("last_model", "upsilon_transfer_last_model"),
    ):
        path = run_dir / subdir / "decoded" / "upsilon_0j1j_prior_decoded_xspace.hdf5"
        if path.exists():
            samples[label] = path
    return samples


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = (args.output_dir or (run_dir / "upsilon_continuum_reweight")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    window = tuple(args.window)
    region = tuple(args.region)
    edges = np.arange(window[0], window[1] + 1e-9, args.bin_width)
    sub_bins = [(8.50, 8.75), (8.75, 9.00), (9.00, 9.25), (9.25, 9.60), (10.50, 11.20)]

    # ---- CMS data and background ------------------------------------------
    cms = cms_mass(args.cms.expanduser().resolve())
    cms_hist = histogram(cms, edges)
    fit_summary_path = (
        run_dir / "upsilon_transfer_best_model" / "decoded_vs_cms" / "comparison_summary.json"
    )
    peaks: list[dict] = []
    if fit_summary_path.exists():
        peaks = json.loads(fit_summary_path.read_text(encoding="utf-8")).get("cms_peak_fit", [])
    peak_hist = peak_histogram(edges, peaks, window) if peaks else np.zeros_like(cms_hist)
    cms_background_hist = np.clip(cms_hist - peak_hist, 0.0, None)
    cms_background_density = normalized_density(cms_background_hist, edges)
    cms_total_density = normalized_density(cms_hist, edges)

    # ---- prior (for the z-space control) ----------------------------------
    with h5py.File(args.prior.expanduser().resolve(), "r") as handle:
        prior_z = handle["FDL/zData"][:]
        prior_component = handle["FDL/component_id"][:]
        prior_attrs = dict(handle["FDL"].attrs)
    prior_mass = mass8(prior_z)
    prior_cont_hist = histogram(prior_mass[prior_component == CONTINUUM_ID], edges)

    samples = resolve_samples(run_dir)
    if not samples:
        raise SystemExit(f"No decoded samples found under {run_dir}")

    fit_cms = fit_log_density(edges, cms_background_hist, tuple(args.sidebands))
    centers = 0.5 * (edges[:-1] + edges[1:])
    report: dict = {
        "window": list(window),
        "region": list(region),
        "sidebands": list(args.sidebands),
        "bin_width": args.bin_width,
        "cms_events_in_window": int(((cms >= window[0]) & (cms < window[1])).sum()),
        "fit_summary": str(fit_summary_path),
        "samples": {},
    }

    for label, decoded_path in samples.items():
        with h5py.File(decoded_path, "r") as handle:
            decoded_x = handle["FDL/xData"][:]
            decoded_z = handle["FDL/zData"][:]
            decoded_component = handle["FDL/component_id"][:]
        decoded_mass = mass8(decoded_x)
        continuum = decoded_component == CONTINUUM_ID
        cont_hist = histogram(decoded_mass[continuum], edges)
        fit_cont = fit_log_density(edges, cont_hist, tuple(args.sidebands))

        # Sideband-derived shape ratio, clipped away from absurd values.
        ratio = np.exp(fit_cms(centers) - fit_cont(centers))
        ratio = np.clip(ratio, 0.05, 20.0)
        ratio_at_event = np.interp(
            decoded_mass, centers, ratio, left=ratio[0], right=ratio[-1]
        )

        weights = np.ones_like(decoded_mass)
        weights[continuum] = ratio_at_event[continuum]
        # Preserve the continuum total: the correction is shape-only.
        if weights[continuum].sum() > 0:
            weights[continuum] *= continuum.sum() / weights[continuum].sum()

        # z-space control: reweight the prior continuum by the same ratio,
        # evaluated on the decoded mass of the matching event (same order).
        z_weight = np.ones_like(prior_mass)
        z_weight[prior_component == CONTINUUM_ID] = ratio_at_event[prior_component == CONTINUUM_ID]
        if z_weight[prior_component == CONTINUUM_ID].sum() > 0:
            z_weight[prior_component == CONTINUUM_ID] *= (
                (prior_component == CONTINUUM_ID).sum()
                / z_weight[prior_component == CONTINUUM_ID].sum()
            )

        def density_block(masses, w):
            return subregion_metrics(masses, w, edges, window, region, sub_bins)

        report["samples"][label] = {
            "decoded_path": str(decoded_path),
            "decoded": density_block(decoded_mass, None),
            "reweighted_decoded": density_block(decoded_mass, weights),
            "continuum_reweighted": density_block(
                decoded_mass[continuum], weights[continuum]
            ),
            "continuum_unweighted": density_block(decoded_mass[continuum], None),
            "prior_continuum_reweighted": density_block(
                prior_mass[prior_component == CONTINUUM_ID],
                z_weight[prior_component == CONTINUUM_ID],
            ),
            "ratio_at_events": {
                "min": float(ratio_at_event[continuum].min()),
                "median": float(np.median(ratio_at_event[continuum])),
                "max": float(ratio_at_event[continuum].max()),
            },
            # Persist the smooth shape ratio itself, not just its summary, so the
            # correction is auditable and reproducible from this JSON alone.
            "weight_curve": {
                "mass_gev": centers.tolist(),
                "cms_over_continuum_ratio": ratio.tolist(),
                "log_density_fit_cms_over_continuum_coefficients_desc": (
                    (fit_cms - fit_cont).coefficients.tolist()
                ),
            },
        }

    report["weight_source"] = {
        "method": "quadratic fit of log(density) vs mass on the CMS background and",
        "sidebands_gev": list(args.sidebands),
        "window_gev": list(window),
        "resample_seed": int(args.seed),
        "note": "ratio = exp(fit_cms(m) - fit_continuum(m)), clipped to [0.05, 20]",
    }

    report["cms_background"] = subregion_metrics(
        np.repeat(centers, np.clip(cms_background_hist.astype(int), 0, None)),
        None,
        edges,
        window,
        region,
        sub_bins,
    )
    report["cms_total"] = subregion_metrics(cms, None, edges, window, region, sub_bins)
    report["prior_continuum_unweighted"] = subregion_metrics(
        prior_mass[prior_component == CONTINUUM_ID], None, edges, window, region, sub_bins
    )

    # ---- report ------------------------------------------------------------
    lines = [
        "# Upsilon continuum-shape diagnostic",
        "",
        f"Window {window[0]}-{window[1]} GeV, bin {args.bin_width} GeV. CMS background =",
        "data minus the fitted 1S/2S/3S Gaussians. The continuum weight is fitted",
        f"on the sidebands {args.sidebands[0]}-{args.sidebands[1]} and "
        f"{args.sidebands[2]}-{args.sidebands[3]} GeV only.",
        "",
        "## Normalized density in 8.5-9.25 GeV (ratio to CMS data)",
        "",
        "| sample | 8.50-8.75 | 8.75-9.00 | 9.00-9.25 | 8.50-9.25 |",
        "|---|---|---|---|---|",
    ]

    def row(name, block):
        d = block["density"]
        cells = []
        for lo, hi in sub_bins[:3]:
            key = f"{lo:.2f}-{hi:.2f}"
            value = d[key]
            cms_value = report["cms_total"]["density"][key]
            cells.append(f"{value / cms_value:.3f}" if cms_value else "-")
        region_value = block["region_fraction"]
        cms_region = report["cms_total"]["region_fraction"]
        cells.append(f"{region_value / cms_region:.3f}" if cms_region else "-")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    row("CMS data (reference)", report["cms_total"])
    for label in samples:
        block = report["samples"][label]
        row(f"{label} (unweighted)", block["decoded"])
        row(f"{label} (continuum reweighted)", block["reweighted_decoded"])

    baseline_excess = {
        label: report["samples"][label]["decoded"]["region_fraction"]
        / report["cms_total"]["region_fraction"]
        for label in samples
    }
    fixed_excess = {
        label: report["samples"][label]["reweighted_decoded"]["region_fraction"]
        / report["cms_total"]["region_fraction"]
        for label in samples
    }
    lines += [
        "",
        "## 8.5-9.25 GeV closure",
        "",
        "| sample | unweighted / CMS | reweighted / CMS | change |",
        "|---|---|---|---|",
    ]
    for label in samples:
        lines.append(
            f"| {label} | {baseline_excess[label]:.3f} | {fixed_excess[label]:.3f} | "
            f"{fixed_excess[label] - baseline_excess[label]:+.3f} |"
        )
    lines += [
        "",
        "## Reading",
        "",
        "- The weight is derived from the sidebands, so 8.5-9.1 is a fitted region and",
        "  9.1-9.25 is interpolated. If both close together, the excess is one smooth",
        "  continuum-shape error. If only the fitted region closes, the correction is",
        "  not transferable.",
        "- `last_model (continuum reweighted)` shows whether the residual after the",
        "  shape fix is in the response or elsewhere.",
        "",
        "Full numbers: `continuum_reweight.json` in this directory.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "continuum_reweight.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output_dir / 'REPORT.md'}")

    # ---- optional resampled prior for a fresh decode ----------------------
    if args.write_reweighted_prior is not None and "last_model" in samples:
        with h5py.File(samples["last_model"], "r") as handle:
            decoded_mass_last = mass8(handle["FDL/xData"][:])
            component_last = handle["FDL/component_id"][:]
        cont = component_last == CONTINUUM_ID
        cont_hist_last = histogram(decoded_mass_last[cont], edges)
        fit_cont_last = fit_log_density(edges, cont_hist_last, tuple(args.sidebands))
        ratio_last = np.clip(np.exp(fit_cms(centers) - fit_cont_last(centers)), 0.05, 20.0)
        ratio_event = np.interp(decoded_mass_last, centers, ratio_last, left=ratio_last[0], right=ratio_last[-1])
        event_weight = np.ones_like(decoded_mass_last)
        event_weight[cont] = ratio_event[cont]
        probability = event_weight / event_weight.sum()
        rng = np.random.default_rng(args.seed)
        index = rng.choice(len(probability), size=len(probability), replace=True, p=probability)
        out = args.write_reweighted_prior.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(out, "w") as handle:
            group = handle.create_group("FDL")
            z_dataset = group.create_dataset("zData", data=prior_z[index])
            group.create_dataset("component_id", data=prior_component[index])
            # The decode/evaluate readers look for component_* on the root and on
            # the zData dataset (and the FDL group), so write all three.
            for key, value in prior_attrs.items():
                for target in (handle.attrs, group.attrs, z_dataset.attrs):
                    try:
                        target[key] = value
                    except Exception:
                        target[key] = str(value)
            for target in (handle.attrs, group.attrs, z_dataset.attrs):
                target["continuum_shape_reweight"] = (
                    "resampled to the CMS sideband continuum shape, 2026-09-10"
                )
        print(f"Wrote reweighted prior {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
