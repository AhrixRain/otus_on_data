#!/usr/bin/env python
"""Compare an Upsilon prior (or decoded sample) with CMS dimuon mass peaks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance


HERE = Path(__file__).resolve().parent


def find_repo_root() -> Path:
    for candidate in (HERE, *HERE.parents):
        if (candidate / "scripts_joint").is_dir() and (candidate / "experiments").is_dir():
            return candidate
    raise RuntimeError("Could not locate the OTUS repository root")


REPO_ROOT = find_repo_root()
UPSILON_EXPERIMENT = REPO_ROOT / "experiments" / "cms_upsilon"
if str(UPSILON_EXPERIMENT) not in sys.path:
    sys.path.insert(0, str(UPSILON_EXPERIMENT))

from plot_upsilon import (  # noqa: E402
    PDG_MASSES_GEV,
    STATE_LABELS,
    fit_spectrum,
    load_csv_masses,
)


DEFAULT_PRIOR = (
    REPO_ROOT
    / "data"
    / "cms_upsilon_mumu_mg5_8tev_inclusive_3S_continuum_ptj5_fiducial_8p5_11p5_1M.hdf5"
)
DEFAULT_CMS = UPSILON_EXPERIMENT / "data" / "Ymumu.csv"
DEFAULT_OUTPUT_DIR = HERE / "preliminary_prior_vs_cms"
MASS_RANGE = (8.5, 11.2)
DEFAULT_BINS = 135  # Exactly 20 MeV over the default 2.7 GeV window.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--prior-dataset", default="FDL/zData")
    parser.add_argument("--component-dataset", default="FDL/component_id")
    parser.add_argument("--cms", type=Path, default=DEFAULT_CMS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mass-min", type=float, default=MASS_RANGE[0])
    parser.add_argument("--mass-max", type=float, default=MASS_RANGE[1])
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS)
    parser.add_argument(
        "--sample-label",
        default=None,
        help="Legend label for the HDF5 sample (inferred when omitted)",
    )
    return parser.parse_args()


def _decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def invariant_mass(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    energy = values[:, 3] + values[:, 7]
    momentum = values[:, 0:3] + values[:, 4:7]
    return np.sqrt(np.maximum(energy * energy - np.sum(momentum * momentum, axis=1), 0.0))


def load_hdf5_sample(
    path: Path, sample_dataset: str, component_dataset: str
) -> tuple[np.ndarray, np.ndarray | None, dict, dict[str, int]]:
    with h5py.File(path, "r") as handle:
        if sample_dataset not in handle:
            raise KeyError(f"Missing dataset {sample_dataset!r} in {path}")
        values = np.asarray(handle[sample_dataset])
        if values.ndim != 2 or values.shape[1] < 8:
            raise ValueError(f"Expected [N, >=8] four-vectors, found {values.shape}")
        components = (
            np.asarray(handle[component_dataset], dtype=np.uint8)
            if component_dataset in handle
            else None
        )
        attrs = {key: _decode_attr(value) for key, value in handle.attrs.items()}
        attrs.update(
            {
                key: _decode_attr(value)
                for key, value in handle[sample_dataset].attrs.items()
                if key not in attrs
            }
        )
    if components is not None and len(components) != len(values):
        raise ValueError("component_id and four-vector datasets have different lengths")
    raw_mapping = attrs.get("component_id_mapping", "{}")
    mapping = json.loads(raw_mapping) if isinstance(raw_mapping, str) else dict(raw_mapping)
    mapping = {str(name): int(identifier) for name, identifier in mapping.items()}
    return values[:, :8], components, attrs, mapping


def fit_cms_peaks(masses: np.ndarray, edges: np.ndarray) -> list[dict[str, float | str]]:
    counts, _ = np.histogram(masses, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = float(edges[1] - edges[0])
    parameters, covariance = fit_spectrum(centers, counts, bin_width)
    errors = np.sqrt(np.diag(covariance))
    rows = []
    for index, label in enumerate(STATE_LABELS):
        offset = 3 + 3 * index
        rows.append(
            {
                "state": label,
                "component": f"upsilon{index + 1}s",
                "pdg_mass_gev": float(PDG_MASSES_GEV[index]),
                "cms_fitted_mass_gev": float(parameters[offset + 1]),
                "cms_mass_error_gev": float(errors[offset + 1]),
                "cms_sigma_gev": float(parameters[offset + 2]),
                "cms_sigma_error_gev": float(errors[offset + 2]),
                "cms_fitted_yield": float(parameters[offset]),
                "cms_yield_error": float(errors[offset]),
            }
        )
    return rows


def component_summary(
    masses: np.ndarray,
    component_ids: np.ndarray | None,
    mapping: dict[str, int],
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    if component_ids is None:
        return result
    for name, identifier in sorted(mapping.items(), key=lambda item: item[1]):
        selected = masses[component_ids == identifier]
        if not len(selected):
            continue
        result[name] = {
            "events": int(len(selected)),
            "fraction": float(len(selected) / len(masses)),
            "mass_mean_gev": float(np.mean(selected)),
            "mass_std_gev": float(np.std(selected)),
            "mass_median_gev": float(np.median(selected)),
        }
    return result


def write_peak_csv(
    path: Path,
    cms_rows: list[dict[str, float | str]],
    prior_components: dict[str, dict[str, float | int]],
) -> None:
    fields = [
        "state",
        "pdg_mass_gev",
        "cms_fitted_mass_gev",
        "cms_mass_error_gev",
        "cms_sigma_gev",
        "cms_sigma_error_gev",
        "prior_mass_mean_gev",
        "prior_mass_std_gev",
        "prior_events",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in cms_rows:
            component = prior_components.get(str(row["component"]), {})
            writer.writerow(
                {
                    "state": row["state"],
                    "pdg_mass_gev": f'{row["pdg_mass_gev"]:.6f}',
                    "cms_fitted_mass_gev": f'{row["cms_fitted_mass_gev"]:.6f}',
                    "cms_mass_error_gev": f'{row["cms_mass_error_gev"]:.6f}',
                    "cms_sigma_gev": f'{row["cms_sigma_gev"]:.6f}',
                    "cms_sigma_error_gev": f'{row["cms_sigma_error_gev"]:.6f}',
                    "prior_mass_mean_gev": f'{float(component.get("mass_mean_gev", float("nan"))):.6f}',
                    "prior_mass_std_gev": f'{float(component.get("mass_std_gev", float("nan"))):.6f}',
                    "prior_events": int(component.get("events", 0)),
                }
            )


def make_plot(
    cms_masses: np.ndarray,
    prior_masses: np.ndarray,
    components: np.ndarray | None,
    mapping: dict[str, int],
    edges: np.ndarray,
    output_dir: Path,
    sample_label: str,
    composition_scope: str,
    decoded_x_space: bool,
) -> None:
    cms_counts, _ = np.histogram(cms_masses, bins=edges)
    prior_counts, _ = np.histogram(prior_masses, bins=edges)
    scale = cms_counts.sum() / max(prior_counts.sum(), 1)
    prior_scaled = prior_counts * scale
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width_mev = (edges[1] - edges[0]) * 1000.0

    figure, (axis, ratio_axis) = plt.subplots(
        2,
        1,
        figsize=(10.0, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3.4, 1.0]},
        constrained_layout=True,
    )
    axis.errorbar(
        centers,
        cms_counts,
        yerr=np.sqrt(np.maximum(cms_counts, 1)),
        fmt="o",
        markersize=3.2,
        linewidth=0.9,
        color="#1f4e79",
        label=f"CMS data ({cms_counts.sum():,} events)",
        zorder=4,
    )
    axis.stairs(
        prior_scaled,
        edges,
        color="#d97706",
        linewidth=1.8,
        label=f"{sample_label} (scaled to CMS total)",
        zorder=3,
    )
    if components is not None and "continuum" in mapping:
        continuum_counts, _ = np.histogram(
            prior_masses[components == mapping["continuum"]], bins=edges
        )
        axis.stairs(
            continuum_counts * scale,
            edges,
            color="#2f855a",
            linestyle="--",
            linewidth=1.3,
            label="Sample continuum component",
            zorder=2,
        )
    for mass, label in zip(PDG_MASSES_GEV, ("1S", "2S", "3S")):
        axis.axvline(mass, color="0.45", linestyle=":", linewidth=1.0)
        axis.text(mass + 0.012, 0.96, label, transform=axis.get_xaxis_transform(), va="top")

    ratio = np.divide(
        prior_scaled,
        cms_counts,
        out=np.full_like(prior_scaled, np.nan, dtype=float),
        where=cms_counts > 0,
    )
    ratio_axis.axhline(1.0, color="0.35", linewidth=1.0)
    ratio_axis.plot(centers, ratio, color="#d97706", marker=".", markersize=3, linewidth=0.8)
    ratio_axis.set_ylim(0.0, min(max(3.0, float(np.nanquantile(ratio, 0.95)) * 1.25), 12.0))
    ratio_axis.set_ylabel("Sample / CMS")
    ratio_axis.set_xlabel(r"$m_{\mu\mu}$ [GeV]")
    ratio_axis.grid(alpha=0.2)

    axis.set_ylabel(f"Events / {bin_width_mev:.0f} MeV")
    axis.set_title(
        rf"$\Upsilon(1S,2S,3S)\rightarrow\mu^+\mu^-$: {sample_label}"
    )
    axis.legend(frameon=False, fontsize=9, loc="upper right")
    axis.grid(alpha=0.16)
    axis.text(
        0.015,
        0.97,
        (
            "Decoded x-space from the frozen joint checkpoint\n"
            if decoded_x_space
            else "Raw z-space peak widths are not detector resolution\n"
        )
        + f"Composition scope: {composition_scope}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
    )
    axis.set_xlim(edges[0], edges[-1])

    png_path = output_dir / "prior_vs_cms_upsilon_peaks.png"
    figure.savefig(png_path, dpi=190)
    figure.savefig(png_path.with_suffix(".pdf"))
    plt.close(figure)


def main() -> int:
    args = parse_args()
    prior_path = args.prior.expanduser().resolve()
    cms_path = args.cms.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not prior_path.exists():
        raise FileNotFoundError(f"Prior/sample file not found: {prior_path}")
    if not cms_path.exists():
        raise FileNotFoundError(f"CMS CSV not found: {cms_path}")
    if args.mass_max <= args.mass_min or args.bins < 2:
        raise ValueError("Require mass_max > mass_min and at least two bins")

    values, components, attrs, mapping = load_hdf5_sample(
        prior_path, args.prior_dataset, args.component_dataset
    )
    prior_masses_all = invariant_mass(values)
    cms_masses_all = load_csv_masses(cms_path)
    prior_keep = (
        np.isfinite(prior_masses_all)
        & (prior_masses_all >= args.mass_min)
        & (prior_masses_all <= args.mass_max)
    )
    cms_keep = (
        np.isfinite(cms_masses_all)
        & (cms_masses_all >= args.mass_min)
        & (cms_masses_all <= args.mass_max)
    )
    prior_masses = prior_masses_all[prior_keep]
    cms_masses = cms_masses_all[cms_keep]
    components_window = components[prior_keep] if components is not None else None
    if not len(prior_masses) or not len(cms_masses):
        raise ValueError("The requested mass window contains no events")

    edges = np.linspace(args.mass_min, args.mass_max, args.bins + 1)
    cms_rows = fit_cms_peaks(cms_masses, edges)
    prior_components = component_summary(prior_masses_all, components, mapping)
    sample_label = args.sample_label or (
        "Decoded prior" if args.prior_dataset.endswith("xData") else "Raw theory prior"
    )
    composition_scope = str(attrs.get("composition_scope", "unspecified"))

    output_dir.mkdir(parents=True, exist_ok=True)
    make_plot(
        cms_masses,
        prior_masses,
        components_window,
        mapping,
        edges,
        output_dir,
        sample_label,
        composition_scope,
        args.prior_dataset.endswith("xData"),
    )
    write_peak_csv(output_dir / "peak_summary.csv", cms_rows, prior_components)

    summary = {
        "schema_version": 1,
        "prior_or_sample": str(prior_path),
        "prior_or_sample_sha256": file_sha256(prior_path),
        "sample_dataset": args.prior_dataset,
        "component_dataset": args.component_dataset if components is not None else None,
        "cms_data": str(cms_path),
        "cms_data_sha256": file_sha256(cms_path),
        "mass_range_gev": [float(args.mass_min), float(args.mass_max)],
        "bins": int(args.bins),
        "bin_width_gev": float(edges[1] - edges[0]),
        "events": {
            "sample_total": int(len(prior_masses_all)),
            "sample_in_window": int(len(prior_masses)),
            "cms_total": int(len(cms_masses_all)),
            "cms_in_window": int(len(cms_masses)),
        },
        "unbinned_shape_metrics_in_window": {
            "ks": float(ks_2samp(cms_masses, prior_masses).statistic),
            "w1_gev": float(wasserstein_distance(cms_masses, prior_masses)),
        },
        "component_mapping": mapping,
        "sample_components": prior_components,
        "cms_peak_fit": cms_rows,
        "composition_scope": composition_scope,
        "interpretation_note": (
            "The inclusive prior mixture fractions are CMS-fit-derived, so total-composition "
            "agreement is not a strict zero-shot result. "
            + (
                "This sample is decoded x-space output from a frozen checkpoint."
                if args.prior_dataset.endswith("xData")
                else "Raw z-space resonance widths are not expected to match detector-smeared "
                "CMS widths before decoding."
            )
        ),
    }
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Loaded {len(prior_masses_all):,} sample events and {len(cms_masses_all):,} CMS events")
    print(f"Common plotted window: {args.mass_min:.3f}-{args.mass_max:.3f} GeV")
    print(f"Bin width: {(edges[1] - edges[0]) * 1000:.1f} MeV")
    print(f"Wrote {output_dir / 'prior_vs_cms_upsilon_peaks.png'}")
    print(f"Wrote {output_dir / 'comparison_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
