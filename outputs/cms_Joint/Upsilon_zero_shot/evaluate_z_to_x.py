#!/usr/bin/env python
"""Quantitative Upsilon evaluation for the direct z-to-x path only.

The script consumes an already-decoded HDF5 file. It never loads an encoder,
forms a cycle, changes a checkpoint, or trains on Upsilon.
"""

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
import torch
from scipy.stats import ks_2samp, wasserstein_distance


HERE = Path(__file__).resolve().parent


def find_repo_root() -> Path:
    for candidate in (HERE, *HERE.parents):
        if (candidate / "scripts").is_dir() and (candidate / "scripts_sota").is_dir():
            return candidate
    raise RuntimeError("Could not locate the OTUS repository root")


REPO_ROOT = find_repo_root()
for directory in (
    REPO_ROOT / "scripts",
    REPO_ROOT / "scripts_sota",
    REPO_ROOT / "experiments" / "cms_upsilon",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import plot as pplot  # noqa: E402
from evaluation import c2st_suite  # noqa: E402
from g0_contract import distribution_report  # noqa: E402
from physics import invariant_mass_np  # noqa: E402
from plot_upsilon import PDG_MASSES_GEV, STATE_LABELS, fit_spectrum  # noqa: E402


DEFAULT_DECODED = HERE / "decoded" / "upsilon_prior_decoded_xspace.hdf5"
DEFAULT_CMS = REPO_ROOT / "experiments" / "cms_upsilon" / "data" / "Ymumu.csv"
DEFAULT_OUTPUT = HERE / "quantitative_z_to_x"
MUON_MASSES = [0.1056583755, 0.1056583755]
P4_NAMES = (
    "mu_minus_px",
    "mu_minus_py",
    "mu_minus_pz",
    "mu_minus_E",
    "mu_plus_px",
    "mu_plus_py",
    "mu_plus_pz",
    "mu_plus_E",
)
P4_LABELS = (
    r"$\mu^-\ p_x$ [GeV]",
    r"$\mu^-\ p_y$ [GeV]",
    r"$\mu^-\ p_z$ [GeV]",
    r"$\mu^-\ E$ [GeV]",
    r"$\mu^+\ p_x$ [GeV]",
    r"$\mu^+\ p_y$ [GeV]",
    r"$\mu^+\ p_z$ [GeV]",
    r"$\mu^+\ E$ [GeV]",
)
STATE_COMPONENTS = ("upsilon1s", "upsilon2s", "upsilon3s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoded", type=Path, default=DEFAULT_DECODED)
    parser.add_argument("--cms", type=Path, default=DEFAULT_CMS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mass-min", type=float, default=8.5)
    parser.add_argument("--mass-max", type=float, default=11.5)
    parser.add_argument("--mass-bin-width", type=float, default=0.020)
    parser.add_argument("--component-bins", type=int, default=80)
    parser.add_argument("--bootstrap-replicates", type=int, default=200)
    parser.add_argument("--c2st-max-samples", type=int, default=5000)
    parser.add_argument("--c2st-folds", type=int, default=5)
    parser.add_argument("--c2st-mlp-iterations", type=int, default=250)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=20260822)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def load_cms_p4(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    data = np.atleast_1d(data)
    required = ("E1", "px1", "py1", "pz1", "Q1", "E2", "px2", "py2", "pz2", "Q2")
    missing = [name for name in required if name not in (data.dtype.names or ())]
    if missing:
        raise ValueError(f"CMS CSV is missing columns: {missing}")
    p1 = np.stack([data["px1"], data["py1"], data["pz1"], data["E1"]], axis=1)
    p2 = np.stack([data["px2"], data["py2"], data["pz2"], data["E2"]], axis=1)
    q1 = np.asarray(data["Q1"])
    q2 = np.asarray(data["Q2"])
    opposite_sign = q1 * q2 < 0
    p1, p2, q1 = p1[opposite_sign], p2[opposite_sign], q1[opposite_sign]
    x = np.empty((len(p1), 8), dtype=np.float32)
    first_is_minus = q1 < 0
    x[first_is_minus, :4] = p1[first_is_minus]
    x[first_is_minus, 4:] = p2[first_is_minus]
    x[~first_is_minus, :4] = p2[~first_is_minus]
    x[~first_is_minus, 4:] = p1[~first_is_minus]
    if not np.isfinite(x).all():
        raise ValueError("CMS four-vectors contain non-finite values")
    return x


def load_decoded(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int], dict]:
    with h5py.File(path, "r") as handle:
        for dataset in ("FDL/zData", "FDL/xData", "FDL/component_id"):
            if dataset not in handle:
                raise KeyError(f"Decoded file is missing {dataset}")
        z = np.asarray(handle["FDL/zData"], dtype=np.float32)
        x = np.asarray(handle["FDL/xData"], dtype=np.float32)
        component_id = np.asarray(handle["FDL/component_id"])
        raw_mapping = handle.attrs.get("component_id_mapping", "{}")
        if isinstance(raw_mapping, bytes):
            raw_mapping = raw_mapping.decode("utf-8")
        mapping = json.loads(raw_mapping) if isinstance(raw_mapping, str) else dict(raw_mapping)
        provenance_raw = handle.attrs.get("decode_provenance", "{}")
        if isinstance(provenance_raw, bytes):
            provenance_raw = provenance_raw.decode("utf-8")
        provenance = (
            json.loads(provenance_raw) if isinstance(provenance_raw, str) else dict(provenance_raw)
        )
    if z.shape != x.shape or z.ndim != 2 or z.shape[1] != 8:
        raise ValueError(f"Expected aligned z/x arrays with shape [N,8], got {z.shape}, {x.shape}")
    if len(component_id) != len(x) or not np.isfinite(x).all() or not np.isfinite(z).all():
        raise ValueError("Decoded arrays are misaligned or contain non-finite values")
    return z, x, component_id, {str(k): int(v) for k, v in mapping.items()}, provenance


def mass(values: np.ndarray) -> np.ndarray:
    return invariant_mass_np(values, daughter_masses=MUON_MASSES, stable=True)


def physics_observables(values: np.ndarray) -> dict[str, np.ndarray]:
    p1, p2 = values[:, :4], values[:, 4:8]
    pt1 = np.hypot(p1[:, 0], p1[:, 1])
    pt2 = np.hypot(p2[:, 0], p2[:, 1])
    eta1 = np.arcsinh(p1[:, 2] / np.maximum(pt1, 1e-8))
    eta2 = np.arcsinh(p2[:, 2] / np.maximum(pt2, 1e-8))
    phi1 = np.arctan2(p1[:, 1], p1[:, 0])
    phi2 = np.arctan2(p2[:, 1], p2[:, 0])
    delta_phi = np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2))
    pair_e = p1[:, 3] + p2[:, 3]
    pair_pz = p1[:, 2] + p2[:, 2]
    pair_y = 0.5 * np.log(
        np.maximum(pair_e + pair_pz, 1e-8) / np.maximum(pair_e - pair_pz, 1e-8)
    )
    return {
        "mass": mass(values),
        "pair_pt": np.hypot(p1[:, 0] + p2[:, 0], p1[:, 1] + p2[:, 1]),
        "pt_minus": pt1,
        "pt_plus": pt2,
        "eta_minus": eta1,
        "eta_plus": eta2,
        "phi_minus": phi1,
        "phi_plus": phi2,
        "pair_rapidity": pair_y,
        "delta_eta": eta1 - eta2,
        "delta_phi": delta_phi,
    }


def equal_subset(real: np.ndarray, fake: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(real), len(fake))
    rng = np.random.default_rng(seed)
    real_index = rng.choice(len(real), n, replace=False)
    fake_index = rng.choice(len(fake), n, replace=False)
    return real[real_index], fake[fake_index]


def pooled_bins(
    real: np.ndarray,
    fake: np.ndarray,
    count: int,
    *,
    nonnegative: bool = False,
    fixed: tuple[float, float] | None = None,
) -> np.ndarray:
    if fixed is not None:
        low, high = fixed
    else:
        pooled = np.concatenate([real[np.isfinite(real)], fake[np.isfinite(fake)]])
        low, high = np.quantile(pooled, [0.002, 0.998])
        if nonnegative:
            low = 0.0
        padding = 0.03 * max(float(high - low), 1e-8)
        if not nonnegative:
            low -= padding
        high += padding
    return np.linspace(float(low), float(high), int(count) + 1)


def observable_metric(real: np.ndarray, fake: np.ndarray) -> dict[str, float]:
    return {
        "ks": float(ks_2samp(real, fake).statistic),
        "w1": float(wasserstein_distance(real, fake)),
        "cms_mean": float(np.mean(real)),
        "decoded_mean": float(np.mean(fake)),
        "cms_std": float(np.std(real)),
        "decoded_std": float(np.std(fake)),
    }


def fit_cms(masses: np.ndarray, edges: np.ndarray) -> list[dict[str, float | str]]:
    counts, _ = np.histogram(masses, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    parameters, covariance = fit_spectrum(centers, counts, float(edges[1] - edges[0]))
    errors = np.sqrt(np.diag(covariance))
    rows = []
    for index, state in enumerate(STATE_LABELS):
        offset = 3 + 3 * index
        rows.append(
            {
                "state": state,
                "mass_gev": float(parameters[offset + 1]),
                "mass_error_gev": float(errors[offset + 1]),
                "sigma_gev": float(parameters[offset + 2]),
                "sigma_error_gev": float(errors[offset + 2]),
                "yield": float(parameters[offset]),
                "yield_error": float(errors[offset]),
            }
        )
    return rows


def state_response(
    z: np.ndarray,
    x: np.ndarray,
    component_id: np.ndarray,
    mapping: dict[str, int],
    mass_min: float,
    mass_max: float,
    bin_width: float,
) -> dict[str, dict[str, float | int]]:
    z_mass, x_mass = mass(z), mass(x)
    result = {}
    edges = np.arange(mass_min, mass_max + 0.5 * bin_width, bin_width)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for name in STATE_COMPONENTS:
        selected = component_id == mapping[name]
        values = x_mass[selected]
        residual = values - z_mass[selected]
        counts, _ = np.histogram(values, bins=edges)
        in_window = (values >= mass_min) & (values <= mass_max)
        result[name] = {
            "events": int(np.sum(selected)),
            "x_window_efficiency": float(np.mean(in_window)),
            "decoded_mass_mean_gev": float(np.mean(values)),
            "decoded_mass_std_gev": float(np.std(values)),
            "decoded_mass_median_gev": float(np.median(values)),
            "decoded_mass_mode_bin_center_gev": float(centers[np.argmax(counts)]),
            "response_bias_mean_gev": float(np.mean(residual)),
            "response_bias_median_gev": float(np.median(residual)),
            "response_q16_gev": float(np.quantile(residual, 0.16)),
            "response_q84_gev": float(np.quantile(residual, 0.84)),
            "response_q025_gev": float(np.quantile(residual, 0.025)),
            "response_q975_gev": float(np.quantile(residual, 0.975)),
        }
    return result


def plot_state_response(
    z: np.ndarray,
    x: np.ndarray,
    component_id: np.ndarray,
    mapping: dict[str, int],
    output_dir: Path,
) -> None:
    z_mass, x_mass = mass(z), mass(x)
    colors = ("#2563eb", "#d97706", "#059669")
    figure, (mass_axis, residual_axis) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for name, label, color in zip(STATE_COMPONENTS, STATE_LABELS, colors):
        selected = component_id == mapping[name]
        mass_axis.hist(
            x_mass[selected],
            bins=np.linspace(8.5, 10.7, 221),
            density=True,
            histtype="step",
            linewidth=1.7,
            color=color,
            label=f"Decoded {label}",
        )
        residual_axis.hist(
            x_mass[selected] - z_mass[selected],
            bins=np.linspace(-1.2, 0.35, 156),
            density=True,
            histtype="step",
            linewidth=1.7,
            color=color,
            label=label,
        )
    for reference in PDG_MASSES_GEV:
        mass_axis.axvline(reference, color="0.5", linestyle=":", linewidth=1.0)
    residual_axis.axvline(0.0, color="0.5", linestyle=":", linewidth=1.0)
    mass_axis.set(xlabel=r"Decoded $m_{\mu\mu}$ [GeV]", ylabel="Normalized density")
    residual_axis.set(
        xlabel=r"$m_{\mu\mu}^{x}-m_{\mu\mu}^{z}$ [GeV]",
        ylabel="Normalized density",
    )
    mass_axis.set_title("State-resolved decoded mass")
    residual_axis.set_title("Direct decoder mass response")
    mass_axis.legend(frameon=False)
    residual_axis.legend(frameon=False)
    mass_axis.grid(alpha=0.18)
    residual_axis.grid(alpha=0.18)
    figure.savefig(output_dir / "z_to_x_state_resolved_mass_response.png", dpi=180)
    plt.close(figure)


def write_observable_csv(path: Path, metrics: dict[str, dict[str, float]]) -> None:
    fields = ["observable", "ks", "w1", "cms_mean", "decoded_mean", "cms_std", "decoded_std"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, values in metrics.items():
            writer.writerow({"observable": name, **values})


def main() -> int:
    args = parse_args()
    decoded_path = args.decoded.expanduser().resolve()
    cms_path = args.cms.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not decoded_path.exists() or not cms_path.exists():
        raise FileNotFoundError("Decoded HDF5 or CMS CSV input is missing")
    if args.mass_max <= args.mass_min or args.mass_bin_width <= 0:
        raise ValueError("Invalid mass range or bin width")
    output_dir.mkdir(parents=True, exist_ok=True)

    z_all, x_all, component_id, mapping, provenance = load_decoded(decoded_path)
    cms_all = load_cms_p4(cms_path)
    cms_mass_all = mass(cms_all)
    decoded_mass_all = mass(x_all)
    cms_keep = (cms_mass_all >= args.mass_min) & (cms_mass_all <= args.mass_max)
    decoded_keep = (decoded_mass_all >= args.mass_min) & (decoded_mass_all <= args.mass_max)
    cms_selected = cms_all[cms_keep]
    decoded_selected = x_all[decoded_keep]
    cms_eval, decoded_eval = equal_subset(cms_selected, decoded_selected, args.seed)
    print(
        f"z->x only: CMS selected={len(cms_selected):,}, decoded selected={len(decoded_selected):,}, "
        f"equal evaluation={len(cms_eval):,}"
    )

    mass_bins = np.arange(
        args.mass_min,
        args.mass_max + 0.5 * args.mass_bin_width,
        args.mass_bin_width,
    )
    cms_obs = physics_observables(cms_eval)
    decoded_obs = physics_observables(decoded_eval)
    truth_label = "CMS data: x"
    prediction_label = r"Decoded prior: $z\rightarrow\tilde{x}$"
    run_label = str(provenance.get("run_label") or "cms_Joint")
    title_prefix = rf"{run_label}: direct $z\rightarrow x$"

    pplot.plot_all_components(
        [
            (cms_eval, truth_label, pplot.TRUTH_STYLE),
            (decoded_eval, prediction_label, pplot.DEC_STYLE),
        ],
        f"{title_prefix}: all x-space four-vector elements",
        output_dir / "z_to_x_all8_xspace_components_density.png",
        args.component_bins,
        True,
        list(P4_LABELS),
    )

    plot_specs: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, str]] = [
        ("mass", cms_obs["mass"], decoded_obs["mass"], mass_bins, r"$m_{\mu\mu}$ [GeV]"),
        (
            "pair_pt",
            cms_obs["pair_pt"],
            decoded_obs["pair_pt"],
            pooled_bins(cms_obs["pair_pt"], decoded_obs["pair_pt"], args.component_bins, nonnegative=True),
            r"$p_T(\mu\mu)$ [GeV]",
        ),
        (
            "pt_minus",
            cms_obs["pt_minus"],
            decoded_obs["pt_minus"],
            pooled_bins(cms_obs["pt_minus"], decoded_obs["pt_minus"], args.component_bins, nonnegative=True),
            r"$p_T(\mu^-)$ [GeV]",
        ),
        (
            "pt_plus",
            cms_obs["pt_plus"],
            decoded_obs["pt_plus"],
            pooled_bins(cms_obs["pt_plus"], decoded_obs["pt_plus"], args.component_bins, nonnegative=True),
            r"$p_T(\mu^+)$ [GeV]",
        ),
        (
            "eta_minus",
            cms_obs["eta_minus"],
            decoded_obs["eta_minus"],
            pooled_bins(cms_obs["eta_minus"], decoded_obs["eta_minus"], args.component_bins),
            r"$\eta(\mu^-)$",
        ),
        (
            "eta_plus",
            cms_obs["eta_plus"],
            decoded_obs["eta_plus"],
            pooled_bins(cms_obs["eta_plus"], decoded_obs["eta_plus"], args.component_bins),
            r"$\eta(\mu^+)$",
        ),
        (
            "pair_rapidity",
            cms_obs["pair_rapidity"],
            decoded_obs["pair_rapidity"],
            pooled_bins(cms_obs["pair_rapidity"], decoded_obs["pair_rapidity"], args.component_bins),
            r"$y(\mu\mu)$",
        ),
        (
            "delta_eta",
            cms_obs["delta_eta"],
            decoded_obs["delta_eta"],
            pooled_bins(cms_obs["delta_eta"], decoded_obs["delta_eta"], args.component_bins),
            r"$\Delta\eta(\mu^-,\mu^+)$",
        ),
        (
            "delta_phi",
            cms_obs["delta_phi"],
            decoded_obs["delta_phi"],
            pooled_bins(
                cms_obs["delta_phi"], decoded_obs["delta_phi"], args.component_bins, fixed=(-np.pi, np.pi)
            ),
            r"$\Delta\phi(\mu^-,\mu^+)$",
        ),
    ]
    for index, (name, label) in enumerate(zip(P4_NAMES, P4_LABELS)):
        plot_specs.append(
            (
                name,
                cms_eval[:, index],
                decoded_eval[:, index],
                pooled_bins(
                    cms_eval[:, index],
                    decoded_eval[:, index],
                    args.component_bins,
                    nonnegative=index in (3, 7),
                ),
                label,
            )
        )

    observable_metrics = {}
    files = ["z_to_x_all8_xspace_components_density.png"]
    for name, real, fake, bins, xlabel in plot_specs:
        path = output_dir / f"paperstyle_z_to_x_{name}_density_ratio.png"
        pplot.paper_ratio_plot_single(
            truth=real,
            pred=fake,
            bins=bins,
            xlabel=xlabel,
            title=f"{title_prefix}: {xlabel}",
            path=path,
            density=True,
            truth_label=truth_label,
            pred_label=prediction_label,
            pred_style=pplot.DEC_STYLE,
            xlim=(float(bins[0]), float(bins[-1])),
            ratio_ylim=(0.0, 2.5),
            residual_ylim=(-1.0, 1.5),
        )
        observable_metrics[name] = observable_metric(real, fake)
        files.append(path.name)

    plot_state_response(z_all, x_all, component_id, mapping, output_dir)
    files.append("z_to_x_state_resolved_mass_response.png")
    state_metrics = state_response(
        z_all,
        x_all,
        component_id,
        mapping,
        args.mass_min,
        args.mass_max,
        args.mass_bin_width,
    )
    cms_peak_fit = fit_cms(cms_obs["mass"], mass_bins)

    device = choose_device(args.device)
    report = distribution_report(
        cms_eval,
        decoded_eval,
        daughter_masses=MUON_MASSES,
        stable_mass=True,
        max_events=None,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed + 100,
    )
    report["c2st"] = c2st_suite(
        cms_eval,
        decoded_eval,
        daughter_masses=MUON_MASSES,
        max_samples=args.c2st_max_samples,
        folds=args.c2st_folds,
        seed=args.seed + 200,
        device=device,
        mlp_iterations=args.c2st_mlp_iterations,
        stable_mass=True,
    )
    summary = {
        "schema_version": 1,
        "direction": "z_to_x_only",
        "decoded_file": str(decoded_path),
        "decoded_file_sha256": file_sha256(decoded_path),
        "cms_file": str(cms_path),
        "cms_file_sha256": file_sha256(cms_path),
        "decode_provenance": provenance,
        "evaluation_scope": provenance.get("evaluation_scope", "unspecified"),
        "selection": {
            "mass_range_gev": [float(args.mass_min), float(args.mass_max)],
            "mass_bin_width_gev": float(args.mass_bin_width),
            "cms_events_before_mass_window": int(len(cms_all)),
            "cms_events_after_mass_window": int(len(cms_selected)),
            "decoded_events_before_mass_window": int(len(x_all)),
            "decoded_events_after_mass_window": int(len(decoded_selected)),
            "decoded_x_window_efficiency": float(np.mean(decoded_keep)),
            "equal_count_events_per_distribution": int(len(cms_eval)),
            "equal_subset_seed": int(args.seed),
        },
        "distribution_report": report,
        "observable_metrics": observable_metrics,
        "state_response": state_metrics,
        "cms_three_peak_fit": cms_peak_fit,
        "files": sorted(files),
        "interpretation": (
            "Every comparison is direct decoded prior versus CMS x space. No encoder, cycle, "
            "or Upsilon tuning is used. Inclusive mixture yields are not strict zero-shot because "
            "the source prior component fractions are CMS-fit-derived. The zero-shot versus "
            "post-unblinding scope is recorded in decode_provenance.evaluation_scope."
        ),
    }
    (output_dir / "z_to_x_metrics.json").write_text(
        json.dumps(pplot.json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_observable_csv(output_dir / "z_to_x_observable_metrics.csv", observable_metrics)
    (output_dir / "z_to_x_manifest.json").write_text(
        json.dumps(
            {
                "direction": "z_to_x_only",
                "files": sorted(files + ["z_to_x_metrics.json", "z_to_x_observable_metrics.csv"]),
                "events_per_distribution": int(len(cms_eval)),
                "mass_bin_width_gev": float(args.mass_bin_width),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(files)} z->x-only plots and quantitative reports to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
