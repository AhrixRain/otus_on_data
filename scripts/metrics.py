from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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
    valid = truth_counts >= int(min_truth_count)
    residual = np.full_like(truth_norm, np.nan, dtype=float)
    residual[valid] = (pred_norm[valid] - truth_norm[valid]) / truth_norm[valid]
    ratio = np.full_like(truth_norm, np.nan, dtype=float)
    ratio[valid] = pred_norm[valid] / truth_norm[valid]
    valid_residual = residual[valid]

    chi2_like = None
    if np.any(valid):
        chi2_like = float(
            np.sum((pred_norm[valid] - truth_norm[valid]) ** 2 / (truth_norm[valid] + 1e-12))
            / max(1, int(valid.sum()) - 1)
        )
    metrics = {
        "mass_range": [float(mass_range[0]), float(mass_range[1])],
        "bins": int(bins),
        "min_truth_count": int(min_truth_count),
        "truth_entries": int(len(truth_mass)),
        "pred_entries": int(len(pred_mass)),
        "valid_bins": int(valid.sum()),
        "max_abs_residual": None if valid_residual.size == 0 else float(np.nanmax(np.abs(valid_residual))),
        "mean_abs_residual": None if valid_residual.size == 0 else float(np.nanmean(np.abs(valid_residual))),
        "rms_residual": None if valid_residual.size == 0 else float(np.sqrt(np.nanmean(valid_residual**2))),
        "chi2_like": chi2_like,
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
