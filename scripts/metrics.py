from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.stats import ks_2samp, wasserstein_distance

    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


def invariant_mass(pairs: np.ndarray) -> np.ndarray:
    p4 = pairs[:, 0:4] + pairs[:, 4:8]
    mass2 = p4[:, 3] ** 2 - p4[:, 2] ** 2 - p4[:, 1] ** 2 - p4[:, 0] ** 2
    return np.sqrt(np.where(mass2 > 0.0, mass2, 0.0))


def residual_metrics(
    truth_mass: np.ndarray,
    pred_mass: np.ndarray,
    bins: int,
    mass_range: tuple[float, float],
    min_truth_count: int = 20,
) -> tuple[dict, dict[str, np.ndarray]]:
    edges = np.linspace(mass_range[0], mass_range[1], int(bins) + 1)
    truth_counts, _ = np.histogram(truth_mass, bins=edges)
    pred_counts, _ = np.histogram(pred_mass, bins=edges)

    truth_norm = truth_counts / max(1, truth_counts.sum())
    pred_norm = pred_counts / max(1, pred_counts.sum())
    valid = (
        (truth_counts >= int(min_truth_count))
        & np.isfinite(truth_norm)
        & (truth_norm != 0.0)
    )
    residual = np.full_like(truth_norm, np.nan, dtype=float)
    residual[valid] = (pred_norm[valid] - truth_norm[valid]) / truth_norm[valid]
    ratio = np.full_like(truth_norm, np.nan, dtype=float)
    ratio[valid] = pred_norm[valid] / truth_norm[valid]
    valid_residual = residual[valid]

    stat_error = np.full_like(truth_norm, np.nan, dtype=float)
    stat_mask = valid & (pred_counts > 0)
    stat_error[stat_mask] = np.sqrt(1.0 / pred_counts[stat_mask] + 1.0 / truth_counts[stat_mask])

    chi2_ndf = None
    if np.any(valid):
        chi2_ndf = float(
            np.sum((pred_counts[valid] - truth_counts[valid]) ** 2 / (truth_counts[valid] + 1e-12))
            / max(1, int(valid.sum()) - 1)
        )
    ks_statistic = None
    ks_pvalue = None
    w1_distance = None
    if HAS_SCIPY:
        ks = ks_2samp(truth_mass, pred_mass)
        ks_statistic = float(ks.statistic)
        ks_pvalue = float(ks.pvalue)
        w1_distance = float(wasserstein_distance(truth_mass, pred_mass))
    metrics = {
        "mass_range": [float(mass_range[0]), float(mass_range[1])],
        "bins": int(bins),
        "total_bins": int(len(truth_counts)),
        "min_truth_count": int(min_truth_count),
        "truth_entries": int(len(truth_mass)),
        "pred_entries": int(len(pred_mass)),
        "valid_bins": int(valid.sum()),
        "frac_within_1pct": None if valid_residual.size == 0 else float(np.mean(np.abs(valid_residual) <= 0.01)),
        "rms_rel_residual": None if valid_residual.size == 0 else float(np.sqrt(np.nanmean(valid_residual**2))),
        "mae_rel_residual": None if valid_residual.size == 0 else float(np.nanmean(np.abs(valid_residual))),
        "max_abs_rel_residual": None if valid_residual.size == 0 else float(np.nanmax(np.abs(valid_residual))),
        "median_stat_error": None if not np.any(np.isfinite(stat_error[valid])) else float(np.nanmedian(stat_error[valid])),
        "ks_statistic": ks_statistic,
        "ks_pvalue": ks_pvalue,
        "w1_distance": w1_distance,
        "chi2_ndf": chi2_ndf,
        "max_abs_residual": None if valid_residual.size == 0 else float(np.nanmax(np.abs(valid_residual))),
        "mean_abs_residual": None if valid_residual.size == 0 else float(np.nanmean(np.abs(valid_residual))),
        "rms_residual": None if valid_residual.size == 0 else float(np.sqrt(np.nanmean(valid_residual**2))),
        "chi2_like": chi2_ndf,
    }
    arrays = {
        "edges": edges,
        "centers": 0.5 * (edges[:-1] + edges[1:]),
        "truth_counts": truth_counts,
        "pred_counts": pred_counts,
        "truth_norm": truth_norm,
        "pred_norm": pred_norm,
        "residual": residual,
        "ratio": ratio,
        "valid": valid,
    }
    return metrics, arrays


def write_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")


def plot_mass_ratio(
    arrays: dict[str, np.ndarray],
    path: Path,
    truth_label: str = "CMS held-out",
    pred_label: str = "OTUS decoded MG5",
    x_label: str = "m(ee) [GeV]",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    centers = arrays["centers"]
    width = np.diff(arrays["edges"])
    fig, (ax_top, ax_ratio) = plt.subplots(
        2,
        1,
        figsize=(8, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax_top.step(centers, arrays["truth_norm"], where="mid", label=truth_label, color="black")
    ax_top.step(centers, arrays["pred_norm"], where="mid", label=pred_label, color="#0072B2")
    ax_top.set_ylabel("Normalized entries")
    ax_top.legend()
    ax_top.grid(alpha=0.25)

    ax_ratio.bar(centers, arrays["ratio"], width=width, align="center", color="#0072B2", alpha=0.75)
    ax_ratio.axhline(1.0, color="black", linewidth=1)
    ax_ratio.set_ylabel("Pred / truth")
    ax_ratio.set_xlabel(x_label)
    ax_ratio.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_residual(
    arrays: dict[str, np.ndarray],
    path: Path,
    x_label: str = "m(ee) [GeV]",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    centers = arrays["centers"]
    width = np.diff(arrays["edges"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(centers, arrays["residual"], width=width, align="center", color="#D55E00", alpha=0.75)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.axhline(0.01, color="gray", linestyle="--", linewidth=1)
    ax.axhline(-0.01, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel(x_label)
    ax.set_ylabel("(pred - truth) / truth")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
