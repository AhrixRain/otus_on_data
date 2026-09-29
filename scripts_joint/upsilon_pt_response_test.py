#!/usr/bin/env python
"""Two decisive checks for the asymmetric decoded Upsilon peaks.

CHECK 1 (model): true Upsilon(1S) events with pair-pT set to 0/5/10/20/40/80 GeV
at fixed E-based mass are decoded with frozen Run F checkpoints, both at noise 0
(the mean map) and at the checkpoint's native noise. We measure the decoded mass
shift and width versus the input pair-pT.

CHECK 2 (data): the CMS Upsilon(1S) peak position and width are fitted in
pair-pT bins, giving the same slope in data. A physical response has a roughly
flat peak position and a pT-dependent width; a pT-dependent peak position is a
scale bug, and the width-vs-pT curve is what a stochastic response must match.

Outputs JSON + markdown + a two-panel figure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
SCALING_DIR = REPO_ROOT / "experiments" / "prior_JpsiScalingTest" / "outputs"
RUN_DIR = REPO_ROOT / "outputs" / "cms_Joint" / "Run_F"
CMS_CSV = REPO_ROOT / "experiments" / "cms_upsilon" / "data" / "Ymumu.csv"
MUON_MASS = 0.1056583755
U1S = 9.4603
CMS_U1S_FIT = 9.4451
TARGET_PTS = (0.0, 5.0, 10.0, 20.0, 40.0, 80.0)
PT_EDGES = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0, 60.0)
FIT_WINDOW = (9.20, 9.70)

for directory in (
    REPO_ROOT / "scripts",
    REPO_ROOT / "scripts_sota",
    REPO_ROOT / "scripts_joint",
    REPO_ROOT / "scripts_joint" / "upsilon",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from decode_prior import load_frozen_model  # noqa: E402


def mass8(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    energy = values[:, 3] + values[:, 7]
    momentum = values[:, :3] + values[:, 4:7]
    return np.sqrt(np.maximum(energy**2 - np.sum(momentum**2, axis=1), 0.0))


def pair_pt(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.hypot(values[:, 0] + values[:, 4], values[:, 1] + values[:, 5])


def cms_pairs(path: Path) -> np.ndarray:
    table = np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8")
    return np.column_stack(
        [
            table["px1"], table["py1"], table["pz1"], table["E1"],
            table["px2"], table["py2"], table["pz2"], table["E2"],
        ]
    )


def gauss_bg(x, amplitude, mu, sigma, b0, b1):
    centred = x - U1S
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + b0 + b1 * centred


def fit_peak(masses: np.ndarray, bootstrap: int = 0, seed: int = 0, width: float = 0.015):
    """Gaussian + linear background fit of a mass peak. Returns position/width."""
    masses = np.asarray(masses, dtype=np.float64)
    masses = masses[(masses > FIT_WINDOW[0]) & (masses < FIT_WINDOW[1])]
    if len(masses) < 60:
        return None
    edges = np.arange(FIT_WINDOW[0], FIT_WINDOW[1] + width, width)
    centres = 0.5 * (edges[:-1] + edges[1:])

    def fit(sample):
        from scipy.optimize import curve_fit

        counts, _ = np.histogram(sample, bins=edges)
        p0 = [max(counts.max(), 5.0), float(np.median(sample)), 0.09, float(np.median(counts)), 0.0]
        popt, pcov = curve_fit(gauss_bg, centres, counts, p0=p0, maxfev=20000)
        return float(popt[1]), abs(float(popt[2])), float(np.sqrt(pcov[1, 1])), float(np.sqrt(pcov[2, 2]))

    try:
        mu, sigma, mu_err, sigma_err = fit(masses)
    except Exception:
        mu = float(np.median(masses))
        sigma = float((np.percentile(masses, 84) - np.percentile(masses, 16)) / 2.0)
        mu_err = sigma / max(np.sqrt(len(masses)), 1.0)
        sigma_err = mu_err
    result = {
        "n": int(len(masses)),
        "mu_gev": mu,
        "mu_err_gev": mu_err,
        "sigma_gev": sigma,
        "sigma_err_gev": sigma_err,
        "median_gev": float(np.median(masses)),
        "robust_width_gev": float((np.percentile(masses, 84) - np.percentile(masses, 16)) / 2.0),
    }
    if bootstrap > 0:
        rng = np.random.default_rng(seed)
        mus, sigmas = [], []
        for _ in range(bootstrap):
            sample = rng.choice(masses, size=len(masses), replace=True)
            try:
                m, s, _, _ = fit(sample)
                mus.append(m)
                sigmas.append(s)
            except Exception:
                continue
        result["mu_bootstrap_sd_gev"] = float(np.std(mus)) if mus else None
        result["sigma_bootstrap_sd_gev"] = float(np.std(sigmas)) if sigmas else None
    return result


def weighted_slope(xs, ys, errs):
    mask = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(errs) & (errs > 0)
    if mask.sum() < 2:
        return None
    return float(np.polyfit(xs[mask], ys[mask], 1, w=1.0 / errs[mask])[0])


def decode_variants(checkpoint: Path, variants: dict[float, np.ndarray], mode: str, device, seed: int):
    model, _, _ = load_frozen_model(checkpoint, device)
    model.eval()
    if mode == "zero":
        model.set_noise_multipliers(0.0, 0.0)
    torch.manual_seed(seed)
    out = {}
    with torch.no_grad():
        for pt, values in variants.items():
            tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
            out[pt] = model.decode(tensor).cpu().numpy()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def discover_checkpoints(run_dir: Path) -> dict[str, tuple[Path, tuple[str, ...]]]:
    """Best / stage-2 best / global last checkpoints present in a run directory."""
    checkpoints: dict[str, tuple[Path, tuple[str, ...]]] = {}
    best = run_dir / "best_model.pt"
    if best.exists():
        checkpoints["best_model"] = (best, ("zero",))
    stage_best = sorted(run_dir.glob("best_*stage2*.pt"))
    if stage_best:
        checkpoints["stage2 best"] = (stage_best[0], ("zero", "native"))
    last = run_dir / "last_model.pt"
    if last.exists():
        checkpoints["last_model"] = (last, ("zero", "native"))
    return checkpoints


PALETTE = ("#2b6cb0", "#dd6b20", "#2f855a", "#6b46c1", "#b83280")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap", type=int, default=60)
    return parser.parse_args()


def run(args) -> int:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    run_dir = args.run_dir.expanduser().resolve()
    args.output_dir = (
        args.output_dir or (run_dir / "upsilon_pt_response")
    ).expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = discover_checkpoints(run_dir)
    if not checkpoints:
        raise SystemExit(f"No checkpoints found under {run_dir}")
    label_colors = {
        label: PALETTE[index % len(PALETTE)]
        for index, label in enumerate(checkpoints)
    }

    # ---- inputs: fixed-mass pair-pT variants ------------------------------
    variants, input_meta = {}, {}
    for target in TARGET_PTS:
        path = SCALING_DIR / f"true_upsilon1s_kinematics_pt{int(target)}.hdf5"
        with h5py.File(path, "r") as handle:
            z = handle["FDL/zData"][:]
        variants[target] = z
        masses = mass8(z)
        input_meta[target] = {
            "input_pair_pt_median_gev": float(np.median(pair_pt(z))),
            "input_mass_median_gev": float(np.median(masses)),
            "input_robust_width_gev": float((np.percentile(masses, 84) - np.percentile(masses, 16)) / 2.0),
        }

    # ---- CHECK 1: model response ------------------------------------------
    model_rows, model_slopes = [], {}
    for label, (checkpoint, modes) in checkpoints.items():
        for mode in modes:
            decoded = decode_variants(checkpoint, variants, mode, device, seed=20260910)
            shift, width, xs = [], [], []
            for target in TARGET_PTS:
                m = mass8(decoded[target])
                fit = fit_peak(m, bootstrap=0)
                if fit is None:
                    continue
                decoded_shift = (fit["median_gev"] - U1S) * 1000.0
                model_rows.append(
                    {
                        "checkpoint": label,
                        "mode": mode,
                        "target_pair_pt_gev": target,
                        "input_pair_pt_gev": input_meta[target]["input_pair_pt_median_gev"],
                        "decoded_mu_gev": fit["mu_gev"],
                        "decoded_median_gev": fit["median_gev"],
                        "shift_mev": decoded_shift,
                        "sigma_gev": fit["sigma_gev"],
                        "robust_width_gev": fit["robust_width_gev"],
                    }
                )
                xs.append(input_meta[target]["input_pair_pt_median_gev"])
                shift.append(decoded_shift)
                width.append(fit["robust_width_gev"] * 1000.0)
            xs = np.asarray(xs)
            key = f"{label} [{mode}]"
            model_slopes[key] = {
                "shift_slope_mev_per_gev_all": float(np.polyfit(xs, np.asarray(shift), 1)[0]),
                "shift_slope_mev_per_gev_pt_le_20": float(
                    np.polyfit(xs[xs <= 20.5], np.asarray(shift)[xs <= 20.5], 1)[0]
                ),
                "sigma_slope_mev_per_gev_all": float(np.polyfit(xs, np.asarray(width), 1)[0]),
            }

    # ---- CHECK 2: CMS peak position and width vs pair-pT ------------------
    cms = cms_pairs(CMS_CSV)
    cms_mass = mass8(cms)
    cms_pt = pair_pt(cms)
    in_1s = (cms_mass > FIT_WINDOW[0]) & (cms_mass < FIT_WINDOW[1])
    data_rows = []
    for lo, hi in zip(PT_EDGES[:-1], PT_EDGES[1:]):
        sel = in_1s & (cms_pt >= lo) & (cms_pt < hi)
        if sel.sum() < 60:
            continue
        fit = fit_peak(cms_mass[sel], bootstrap=args.bootstrap, seed=int(lo * 100) + 1)
        if fit is None:
            continue
        data_rows.append(
            {
                "pt_lo_gev": lo,
                "pt_hi_gev": hi,
                "pt_median_gev": float(np.median(cms_pt[sel])),
                "n": int(sel.sum()),
                **{k: v for k, v in fit.items() if k != "n"},
            }
        )

    x_data = np.array([row["pt_median_gev"] for row in data_rows])
    mu_data = np.array([row["mu_gev"] for row in data_rows])
    mu_err = np.array([row["mu_bootstrap_sd_gev"] or row["mu_err_gev"] for row in data_rows])
    sig_data = np.array([row["sigma_gev"] * 1000.0 for row in data_rows])
    sig_err = np.array([row["sigma_bootstrap_sd_gev"] or row["sigma_err_gev"] for row in data_rows]) * 1000.0
    low = x_data <= 20.5
    data_slopes = {
        "position_slope_mev_per_gev_all": weighted_slope(x_data, (mu_data - CMS_U1S_FIT) * 1000.0, mu_err),
        "position_slope_mev_per_gev_pt_le_20": weighted_slope(
            x_data[low], (mu_data[low] - CMS_U1S_FIT) * 1000.0, mu_err[low]
        ),
        "sigma_slope_mev_per_gev_all": weighted_slope(x_data, sig_data, sig_err),
        "reference_global_1s_fit_gev": CMS_U1S_FIT,
    }

    report = {
        "input": input_meta,
        "model_rows": model_rows,
        "model_slopes": model_slopes,
        "data_rows": data_rows,
        "data_slopes": data_slopes,
    }

    # ---- figure -----------------------------------------------------------
    plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.linewidth": 0.8,
                         "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False})
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(8, 7.2), sharex=True)
    for key, rows in _group(model_rows).items():
        label, mode = key.rsplit(" [", 1)
        color = label_colors.get(label, "black")
        style = "--" if mode.rstrip("]") == "native" else "-"
        top.plot(rows["x"], rows["shift"], color=color, linestyle=style, marker="o", ms=3.5, lw=1.2,
                 label=f"{key}  (slope {model_slopes[key]['shift_slope_mev_per_gev_all']:+.1f} MeV/GeV)")
        bottom.plot(rows["x"], rows["sigma"], color=color, linestyle=style, marker="o", ms=3.5, lw=1.2,
                    label=key)
    top.errorbar(x_data, (mu_data - CMS_U1S_FIT) * 1000.0, yerr=mu_err * 1000.0, fmt="ks", ms=5,
                 capsize=3, label=f"CMS data (slope {data_slopes['position_slope_mev_per_gev_all']:+.1f} MeV/GeV)")
    bottom.errorbar(x_data, sig_data, yerr=sig_err, fmt="ks", ms=5, capsize=3, label="CMS data")
    top.axhline(0.0, color="black", lw=0.7, ls=":")
    top.set_ylabel("decoded peak position $-$ truth [MeV]")
    bottom.set_ylabel("peak width $\\sigma$ [MeV]")
    bottom.set_xlabel("pair $p_T$ [GeV]")
    bottom.set_xlim(-2, 85)
    top.legend(fontsize=7.5, loc="best")
    bottom.legend(fontsize=8, loc="best")
    figure.savefig(args.output_dir / "upsilon_pt_response.png", dpi=150, bbox_inches="tight")
    figure.savefig(args.output_dir / "upsilon_pt_response.pdf", bbox_inches="tight")
    plt.close(figure)

    lines = [
        "# Upsilon pair-pT response test",
        "",
        "CHECK 1: true 1S events at fixed mass, pair-pT set to 0/5/10/20/40/80 GeV,",
        "decoded with frozen Run F checkpoints. `zero` = noise 0 (mean map),",
        "`native` = the checkpoint's own noise. Decoded peaks are often non-Gaussian,",
        "so slopes use the robust median position and 68% half-width.",
        "",
        "| checkpoint | mode | shift slope [MeV/GeV] (all) | shift slope (pT<=20) | sigma slope [MeV/GeV] |",
        "|---|---|---|---|---|",
    ]
    for key, value in model_slopes.items():
        label, mode = key.rsplit(" [", 1)
        lines.append(
            f"| {label} | {mode.rstrip(']')} | {value['shift_slope_mev_per_gev_all']:+.1f} | "
            f"{value['shift_slope_mev_per_gev_pt_le_20']:+.1f} | {value['sigma_slope_mev_per_gev_all']:+.1f} |"
        )
    lines += [
        "",
        "CHECK 2: CMS Upsilon(1S) peak fits in pair-pT bins (Gaussian + linear background).",
        "",
        "| pair-pT [GeV] | N | peak position [GeV] | sigma [MeV] |",
        "|---|---|---|---|",
    ]
    for row in data_rows:
        lines.append(
            f"| {row['pt_lo_gev']:.0f}-{row['pt_hi_gev']:.0f} | {row['n']} | "
            f"{row['mu_gev']:.4f} +/- {row['mu_err_gev']:.4f} | {row['sigma_gev']*1000:.1f} |"
        )
    lines += [
        "",
        f"CMS position slope (all bins, weighted): **{data_slopes['position_slope_mev_per_gev_all']:+.2f} MeV/GeV**; "
        f"pT<=20: {data_slopes['position_slope_mev_per_gev_pt_le_20']:+.2f} MeV/GeV.",
        f"CMS sigma slope: {data_slopes['sigma_slope_mev_per_gev_all']:+.1f} MeV/GeV.",
        "",
        "Figure: `upsilon_pt_response.png` / `.pdf`.",
    ]
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "upsilon_pt_response.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))
    return 0


def _group(model_rows):
    grouped = {}
    for row in model_rows:
        key = f"{row['checkpoint']} [{row['mode']}]"
        grouped.setdefault(key, {"x": [], "shift": [], "sigma": []})
        grouped[key]["x"].append(row["input_pair_pt_gev"])
        grouped[key]["shift"].append(row["shift_mev"])
        grouped[key]["sigma"].append(row["robust_width_gev"] * 1000.0)
    return grouped


if __name__ == "__main__":
    raise SystemExit(run(main()))
