from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cms_data import load_and_split, load_config, resolve_config, save_resolved_config
from cms_model import load_model_from_checkpoint
from cms_training import first_tensor
from device_utils import device_report, select_device

try:
    from scipy.stats import ks_2samp

    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


TRUTH_STYLE = dict(color="black", linewidth=1.8, linestyle="-")
ENC_STYLE = dict(color="deepskyblue", linewidth=1.8, linestyle="--")
ENCDEC_STYLE = dict(color="forestgreen", linewidth=1.8, linestyle=":")
DEC_STYLE = dict(color="darkviolet", linewidth=1.8, linestyle="--")

TRUTH_LABEL_Z = "Ground truth: z"
PRED_LABEL_Z = r"OTUS encoder: $x \rightarrow \tilde{z}$"
TRUTH_LABEL_X = "Ground truth: x / CMS data"
PRED_LABEL_X_RECO = r"OTUS encoder-decoder: $x \rightarrow \tilde{z} \rightarrow \tilde{x}$"
PRED_LABEL_X_GEN = r"OTUS decoder: $z \rightarrow \tilde{x}^{\prime}$"

MZ_REF = 91.1880
COMPONENT_TITLES = [
    r"$e^-\ p_x$",
    r"$e^-\ p_y$",
    r"$e^-\ p_z$",
    r"$e^-\ E$",
    r"$e^+\ p_x$",
    r"$e^+\ p_y$",
    r"$e^+\ p_z$",
    r"$e^+\ E$",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create CMS DoubleElectron paper-style OTUS plots with ratio and residual panels."
    )
    parser.add_argument("--config", type=Path, required=True, help="YAML/JSON config path.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="best_model.pt or last_model.pt.")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Plot output directory.")
    parser.add_argument("--num-samples", type=int, default=None, help="Limit selected CMS and MG5 rows before splitting.")
    parser.add_argument("--split", choices=("val", "test"), default="test", help="Held-out split to plot.")
    parser.add_argument("--max-x-events", type=int, default=5000000, help="Maximum CMS x events to plot.")
    parser.add_argument("--max-z-events", type=int, default=5000000, help="Maximum MG5 z events to plot.")
    parser.add_argument("--batch-size", type=int, default=None, help="Inference batch size.")
    parser.add_argument("--seed", type=int, default=None, help="Plot subsampling seed.")
    parser.add_argument(
        "--counts",
        action="store_true",
        help="Plot raw counts instead of normalized densities.",
    )
    parser.add_argument("--mass-low", type=float, default=70.0, help="Low edge for mass plots.")
    parser.add_argument("--mass-high", type=float, default=110.0, help="High edge for mass plots.")
    parser.add_argument("--mass-bin-width", type=float, default=1.0, help="Mass bin width in GeV.")
    return parser.parse_args()


def finite_values(a: np.ndarray) -> np.ndarray:
    values = np.asarray(a).reshape(-1)
    return values[np.isfinite(values)]


def finite_8d(a: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(a)
    if values.ndim != 2 or values.shape[1] != 8:
        raise ValueError(f"{name} must have shape (N, 8). Got {values.shape}.")
    mask = np.isfinite(values).all(axis=1)
    dropped = int(len(values) - mask.sum())
    if dropped:
        print(f"Dropped {dropped} non-finite rows from {name}.")
    return values[mask]


def random_subset(a: np.ndarray, n: int | None, seed: int) -> np.ndarray:
    values = np.asarray(a)
    if n is None or int(n) <= 0 or len(values) <= int(n):
        return values
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(values), size=int(n), replace=False)
    return values[idx]


def inv_mass_ee(a: np.ndarray, eps: float = 0.0) -> np.ndarray:
    values = np.asarray(a)
    px = values[:, 0] + values[:, 4]
    py = values[:, 1] + values[:, 5]
    pz = values[:, 2] + values[:, 6]
    energy = values[:, 3] + values[:, 7]
    mass2 = energy**2 - px**2 - py**2 - pz**2
    return np.sqrt(np.maximum(mass2, eps))


def pt_ee(a: np.ndarray) -> np.ndarray:
    values = np.asarray(a)
    px = values[:, 0] + values[:, 4]
    py = values[:, 1] + values[:, 5]
    return np.sqrt(px**2 + py**2)


def w2_1d_np(a: np.ndarray, b: np.ndarray) -> float:
    left = finite_values(a)
    right = finite_values(b)
    n = min(len(left), len(right))
    if n == 0:
        return float("nan")
    return float(np.mean((np.sort(left)[:n] - np.sort(right)[:n]) ** 2))


def maybe_ks(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if not HAS_SCIPY:
        return float("nan"), float("nan")
    result = ks_2samp(np.asarray(a), np.asarray(b))
    return float(result.statistic), float(result.pvalue)


def hist_for_plot(values: np.ndarray, bins: np.ndarray, density: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = finite_values(values)
    counts, edges = np.histogram(values, bins=bins)
    counts = counts.astype(float)
    widths = np.diff(edges)
    if density:
        total = counts.sum()
        heights = counts / (total * widths) if total > 0 else np.zeros_like(counts)
    else:
        heights = counts.copy()
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, heights, counts


def ratio_and_error(
    pred_heights: np.ndarray,
    truth_heights: np.ndarray,
    pred_counts: np.ndarray,
    truth_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pred_heights = np.asarray(pred_heights, dtype=float)
    truth_heights = np.asarray(truth_heights, dtype=float)
    pred_counts = np.asarray(pred_counts, dtype=float)
    truth_counts = np.asarray(truth_counts, dtype=float)

    ratio = np.full_like(pred_heights, np.nan, dtype=float)
    err = np.full_like(pred_heights, np.nan, dtype=float)

    mask = (truth_heights > 0) & (truth_counts > 0) & np.isfinite(truth_heights)
    ratio[mask] = pred_heights[mask] / truth_heights[mask]

    emask = mask & (pred_counts > 0)
    err[emask] = ratio[emask] * np.sqrt(1.0 / pred_counts[emask] + 1.0 / truth_counts[emask])
    return ratio, err


def draw_hist_step(ax: plt.Axes, centers: np.ndarray, heights: np.ndarray, label: str, style: dict[str, Any]) -> None:
    ax.step(centers, heights, where="mid", label=label, **style)


def draw_relative_error_target(ax: plt.Axes, target_rel_err: float = 0.01) -> None:
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax.axhspan(-target_rel_err, target_rel_err, color="gray", alpha=0.12, linewidth=0)
    ax.axhline(+target_rel_err, color="gray", linestyle=":", linewidth=0.9)
    ax.axhline(-target_rel_err, color="gray", linestyle=":", linewidth=0.9)
    ax.text(
        0.985,
        0.90,
        fr"$\pm {100 * target_rel_err:.1f}\%$ target",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="dimgray",
    )


def residual_quality_summary(
    residual: np.ndarray,
    residual_err: np.ndarray,
    truth_counts: np.ndarray,
    pred_counts: np.ndarray,
    target_rel_err: float = 0.01,
    sigma_max: float = 0.01,
    min_truth_counts: int = 100,
    min_pred_counts: int = 100,
) -> dict[str, float | int | None]:
    residual = np.asarray(residual, dtype=float)
    residual_err = np.asarray(residual_err, dtype=float)
    truth_counts = np.asarray(truth_counts, dtype=float)
    pred_counts = np.asarray(pred_counts, dtype=float)

    mask = (
        np.isfinite(residual)
        & np.isfinite(residual_err)
        & (truth_counts >= min_truth_counts)
        & (pred_counts >= min_pred_counts)
        & (residual_err < sigma_max)
    )
    if not np.any(mask):
        return {
            "valid_bins": 0,
            "total_bins": int(len(residual)),
            "frac_within_target": None,
            "mae": None,
            "rms": None,
            "max_abs": None,
            "median_sigma": None,
        }

    selected = residual[mask]
    abs_selected = np.abs(selected)
    return {
        "valid_bins": int(mask.sum()),
        "total_bins": int(len(residual)),
        "frac_within_target": float(np.mean(abs_selected < target_rel_err)),
        "mae": float(np.mean(abs_selected)),
        "rms": float(np.sqrt(np.mean(selected**2))),
        "max_abs": float(np.max(abs_selected)),
        "median_sigma": float(np.median(residual_err[mask])),
    }


def paper_ratio_plot_single(
    truth: np.ndarray,
    pred: np.ndarray,
    bins: np.ndarray,
    xlabel: str,
    title: str,
    path: Path,
    density: bool,
    truth_label: str = TRUTH_LABEL_Z,
    pred_label: str = PRED_LABEL_Z,
    pred_style: dict[str, Any] = ENC_STYLE,
    xlim: tuple[float, float] | None = None,
    ratio_ylim: tuple[float, float] = (0.5, 1.5),
    residual_ylim: tuple[float, float] = (-0.05, 0.05),
    residual_ylabel: str = "Rel. diff.\nto truth",
    reference_x: float | None = None,
    reference_label: str | None = None,
) -> dict[str, np.ndarray]:
    truth = finite_values(truth)
    pred = finite_values(pred)

    centers, h_truth, c_truth = hist_for_plot(truth, bins, density=density)
    _, h_pred, c_pred = hist_for_plot(pred, bins, density=density)
    ratio, rerr = ratio_and_error(h_pred, h_truth, c_pred, c_truth)
    residual = ratio - 1.0

    fig = plt.figure(figsize=(7.0, 7.4))
    gs = fig.add_gridspec(3, 1, height_ratios=[3.0, 1.05, 1.05], hspace=0.05)
    ax = fig.add_subplot(gs[0])
    rax = fig.add_subplot(gs[1], sharex=ax)
    dax = fig.add_subplot(gs[2], sharex=ax)

    draw_hist_step(ax, centers, h_truth, truth_label, TRUTH_STYLE)
    draw_hist_step(ax, centers, h_pred, pred_label, pred_style)
    if reference_x is not None:
        ax.axvline(
            reference_x,
            linestyle="--",
            linewidth=1.2,
            color="gray",
            label=reference_label,
        )
    ax.text(
        0.97,
        0.92,
        fr"$W_2^2 = {w2_1d_np(truth, pred):.3e}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=pred_style.get("color", "black"),
        fontsize=10,
    )
    ax.set_ylabel(r"Normalized density" if density else "Counts")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.18)

    rax.errorbar(
        centers,
        ratio,
        yerr=rerr,
        fmt="o",
        markersize=3.5,
        color=pred_style.get("color", "black"),
        ecolor=pred_style.get("color", "black"),
        elinewidth=1.0,
        capsize=0,
    )
    rax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
    rax.set_ylabel("Ratio\nto truth")
    rax.set_ylim(*ratio_ylim)
    rax.grid(alpha=0.18)

    dax.errorbar(
        centers,
        residual,
        yerr=rerr,
        fmt="o",
        markersize=3.5,
        color=pred_style.get("color", "black"),
        ecolor=pred_style.get("color", "black"),
        elinewidth=1.0,
        capsize=0,
    )
    draw_relative_error_target(dax)
    dax.set_ylabel(residual_ylabel)
    dax.set_xlabel(xlabel)
    dax.set_ylim(*residual_ylim)
    dax.grid(alpha=0.18)

    if xlim is not None:
        ax.set_xlim(*xlim)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(rax.get_xticklabels(), visible=False)
    fig.subplots_adjust(hspace=0.05)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)
    return {
        "centers": centers,
        "truth": h_truth,
        "pred": h_pred,
        "truth_counts": c_truth,
        "pred_counts": c_pred,
        "ratio": ratio,
        "ratio_err": rerr,
        "residual": residual,
        "residual_err": rerr,
    }


def paper_ratio_plot_double(
    truth: np.ndarray,
    pred1: np.ndarray,
    pred2: np.ndarray,
    bins: np.ndarray,
    xlabel: str,
    title: str,
    path: Path,
    density: bool,
    truth_label: str = TRUTH_LABEL_X,
    pred1_label: str = PRED_LABEL_X_RECO,
    pred2_label: str = PRED_LABEL_X_GEN,
    pred1_style: dict[str, Any] = ENCDEC_STYLE,
    pred2_style: dict[str, Any] = DEC_STYLE,
    xlim: tuple[float, float] | None = None,
    ratio_ylim: tuple[float, float] = (0.5, 1.5),
    residual_ylim: tuple[float, float] = (-0.05, 0.05),
    residual_ylabel: str = "Rel. diff.\nto truth",
    reference_x: float | None = None,
    reference_label: str | None = None,
) -> dict[str, np.ndarray]:
    truth = finite_values(truth)
    pred1 = finite_values(pred1)
    pred2 = finite_values(pred2)

    centers, h_truth, c_truth = hist_for_plot(truth, bins, density=density)
    _, h_pred1, c_pred1 = hist_for_plot(pred1, bins, density=density)
    _, h_pred2, c_pred2 = hist_for_plot(pred2, bins, density=density)
    ratio1, err1 = ratio_and_error(h_pred1, h_truth, c_pred1, c_truth)
    ratio2, err2 = ratio_and_error(h_pred2, h_truth, c_pred2, c_truth)
    residual1 = ratio1 - 1.0
    residual2 = ratio2 - 1.0

    fig = plt.figure(figsize=(7.0, 7.4))
    gs = fig.add_gridspec(3, 1, height_ratios=[3.0, 1.05, 1.05], hspace=0.05)
    ax = fig.add_subplot(gs[0])
    rax = fig.add_subplot(gs[1], sharex=ax)
    dax = fig.add_subplot(gs[2], sharex=ax)

    draw_hist_step(ax, centers, h_truth, truth_label, TRUTH_STYLE)
    draw_hist_step(ax, centers, h_pred1, pred1_label, pred1_style)
    draw_hist_step(ax, centers, h_pred2, pred2_label, pred2_style)
    if reference_x is not None:
        ax.axvline(
            reference_x,
            linestyle="--",
            linewidth=1.2,
            color="gray",
            label=reference_label,
        )
    ax.text(
        0.97,
        0.92,
        fr"$W_2^2(x,\tilde{{x}}) = {w2_1d_np(truth, pred1):.3e}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=pred1_style.get("color", "black"),
        fontsize=9,
    )
    ax.text(
        0.97,
        0.84,
        fr"$W_2^2(x,\tilde{{x}}^\prime) = {w2_1d_np(truth, pred2):.3e}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=pred2_style.get("color", "black"),
        fontsize=9,
    )
    ax.set_ylabel(r"Normalized density" if density else "Counts")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.18)

    rax.errorbar(
        centers,
        ratio1,
        yerr=err1,
        fmt="o",
        markersize=3.5,
        color=pred1_style.get("color", "black"),
        ecolor=pred1_style.get("color", "black"),
        elinewidth=1.0,
        capsize=0,
        label=pred1_label,
    )
    rax.errorbar(
        centers,
        ratio2,
        yerr=err2,
        fmt="s",
        markersize=3.0,
        color=pred2_style.get("color", "black"),
        ecolor=pred2_style.get("color", "black"),
        elinewidth=1.0,
        capsize=0,
        label=pred2_label,
    )
    rax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
    rax.set_ylabel("Ratio\nto data")
    rax.set_ylim(*ratio_ylim)
    rax.grid(alpha=0.18)

    dax.errorbar(
        centers,
        residual1,
        yerr=err1,
        fmt="o",
        markersize=3.5,
        color=pred1_style.get("color", "black"),
        ecolor=pred1_style.get("color", "black"),
        elinewidth=1.0,
        capsize=0,
        label=pred1_label,
    )
    dax.errorbar(
        centers,
        residual2,
        yerr=err2,
        fmt="s",
        markersize=3.0,
        color=pred2_style.get("color", "black"),
        ecolor=pred2_style.get("color", "black"),
        elinewidth=1.0,
        capsize=0,
        label=pred2_label,
    )
    draw_relative_error_target(dax)
    dax.set_ylabel(residual_ylabel)
    dax.set_xlabel(xlabel)
    dax.set_ylim(*residual_ylim)
    dax.grid(alpha=0.18)

    if xlim is not None:
        ax.set_xlim(*xlim)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(rax.get_xticklabels(), visible=False)
    fig.subplots_adjust(hspace=0.05)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)
    return {
        "centers": centers,
        "truth": h_truth,
        "pred1": h_pred1,
        "pred2": h_pred2,
        "truth_counts": c_truth,
        "pred1_counts": c_pred1,
        "pred2_counts": c_pred2,
        "ratio1": ratio1,
        "ratio2": ratio2,
        "ratio1_err": err1,
        "ratio2_err": err2,
        "residual1": residual1,
        "residual2": residual2,
        "residual1_err": err1,
        "residual2_err": err2,
    }


def predict_batches(model: torch.nn.Module, mode: str, arr: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(arr), batch_size):
            batch_np = np.asarray(arr[start : start + batch_size], dtype=np.float32)
            batch = torch.as_tensor(batch_np, dtype=torch.float32, device=device)
            if mode == "encode":
                out = model.encode(batch)
            elif mode == "decode":
                out = model.decode(batch)
            elif mode == "reconstruct":
                out = model.decode(first_tensor(model.encode(batch)))
            else:
                raise ValueError(f"Unknown prediction mode: {mode}")
            outputs.append(first_tensor(out).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def plot_all_components(
    arrays: list[tuple[np.ndarray, str, dict[str, Any]]],
    title: str,
    path: Path,
    component_bins: int,
    density: bool,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(20, 7))
    fig.suptitle(title, y=1.02, fontsize=14)
    for j, ax in enumerate(axes.ravel()):
        all_j = np.concatenate([values[:, j] for values, _, _ in arrays])
        lo, hi = np.nanpercentile(all_j, [0.5, 99.5])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = np.nanmin(all_j), np.nanmax(all_j)
        bins = np.linspace(lo, hi, component_bins)
        for values, label, style in arrays:
            ax.hist(
                values[:, j],
                bins=bins,
                density=density,
                histtype="step",
                linewidth=style.get("linewidth", 1.8),
                linestyle=style.get("linestyle", "-"),
                color=style.get("color"),
                label=label,
            )
        ax.set_title(COMPONENT_TITLES[j])
        ax.grid(alpha=0.25)
        if j in {0, 4}:
            ax.set_ylabel("Normalized density" if density else "Counts")
        if j == 0:
            ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


def write_summary(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Saved:", path)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    config = resolve_config(load_config(args.config))
    seed = int(config.get("seed", 0) if args.seed is None else args.seed)
    density = not args.counts

    device = select_device(args.device)
    report = device_report(device)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    model, model_config, _stats, checkpoint = load_model_from_checkpoint(
        checkpoint_path,
        config=config,
        map_location=torch.device("cpu"),
    )
    model.to(device)

    arrays = load_and_split(model_config, num_samples=args.num_samples)
    split = args.split
    x_split = finite_8d(arrays[f"x_{split}"], f"x_{split}")
    z_split = finite_8d(arrays[f"z_{split}"], f"z_{split}")

    x_plot = random_subset(x_split, args.max_x_events, seed=seed)
    z_plot = random_subset(z_split, args.max_z_events, seed=seed + 1)
    z_for_x = random_subset(z_split, len(x_plot), seed=seed + 2)

    batch_size = int(args.batch_size or model_config.get("loaders", {}).get("eval_batch_size", 20000))
    batch_size = max(1, batch_size)
    z_encoded = predict_batches(model, "encode", x_plot, device, batch_size)
    x_reco = predict_batches(model, "reconstruct", x_plot, device, batch_size)
    x_from_z = predict_batches(model, "decode", z_for_x, device, batch_size)

    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else checkpoint_path.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(model_config, output_dir / "config.resolved.json")

    mass_bins = np.arange(args.mass_low, args.mass_high + args.mass_bin_width, args.mass_bin_width)
    pt_bins = np.linspace(0.0, 100.0, 101)
    bins_y = np.array([-100, -60] + [-50 + 5 * i for i in range(21)] + [60, 100], dtype=float)
    bins_z = np.array([-400] + [-250 + 20 * i for i in range(26)] + [400], dtype=float)
    bins_e = np.array([0] + [20 + 10 * i for i in range(26)] + [400], dtype=float)

    m_x = inv_mass_ee(x_plot)
    m_x_reco = inv_mass_ee(x_reco)
    m_x_from_z = inv_mass_ee(x_from_z)
    xmass_info = paper_ratio_plot_double(
        truth=m_x,
        pred1=m_x_reco,
        pred2=m_x_from_z,
        bins=mass_bins,
        xlabel=r"$m_{ee}$ [GeV]",
        title=r"CMS DoubleElectron: x-space $Z \rightarrow ee$ mass",
        path=output_dir / "paperstyle_xspace_mass_density_ratio.png",
        density=density,
        xlim=(args.mass_low, args.mass_high),
        ratio_ylim=(0.5, 1.5),
        residual_ylim=(-0.05, 0.05),
        residual_ylabel=r"$(\mathrm{OTUS}-\mathrm{data})/\mathrm{data}$",
        reference_x=MZ_REF,
        reference_label=fr"$m_Z={MZ_REF:.4f}$ GeV",
    )

    pt_x = pt_ee(x_plot)
    pt_x_reco = pt_ee(x_reco)
    pt_x_from_z = pt_ee(x_from_z)
    paper_ratio_plot_double(
        truth=pt_x,
        pred1=pt_x_reco,
        pred2=pt_x_from_z,
        bins=pt_bins,
        xlabel=r"$p_T(ee)$ [GeV]",
        title=r"CMS DoubleElectron: dilepton transverse momentum",
        path=output_dir / "paperstyle_xspace_pt_density_ratio.png",
        density=density,
        xlim=(0.0, 100.0),
        ratio_ylim=(0.0, 2.0),
    )

    principal_specs_z = [
        (5, bins_y, r"Positron $p_y$ [GeV]", (-100.0, 100.0), (0.5, 1.5), "paperstyle_zspace_pos_py_ratio.png"),
        (6, bins_z, r"Positron $p_z$ [GeV]", (-400.0, 400.0), (0.5, 1.5), "paperstyle_zspace_pos_pz_ratio.png"),
        (7, bins_e, r"Positron $E$ [GeV]", (0.0, 400.0), (0.5, 1.5), "paperstyle_zspace_pos_E_ratio.png"),
    ]
    for idx, bins, xlabel, xlim, ratio_ylim, filename in principal_specs_z:
        paper_ratio_plot_single(
            truth=z_plot[:, idx],
            pred=z_encoded[:, idx],
            bins=bins,
            xlabel=xlabel,
            title=fr"z-space closure: {xlabel}",
            path=output_dir / filename,
            density=density,
            xlim=xlim,
            ratio_ylim=ratio_ylim,
        )

    principal_specs_x = [
        (5, bins_y, r"Positron $p_y$ [GeV]", (-100.0, 100.0), (0.5, 1.5), "paperstyle_xspace_pos_py_ratio.png"),
        (6, bins_z, r"Positron $p_z$ [GeV]", (-400.0, 400.0), (0.5, 1.5), "paperstyle_xspace_pos_pz_ratio.png"),
        (7, bins_e, r"Positron $E$ [GeV]", (0.0, 400.0), (0.5, 1.5), "paperstyle_xspace_pos_E_ratio.png"),
    ]
    for idx, bins, xlabel, xlim, ratio_ylim, filename in principal_specs_x:
        paper_ratio_plot_double(
            truth=x_plot[:, idx],
            pred1=x_reco[:, idx],
            pred2=x_from_z[:, idx],
            bins=bins,
            xlabel=xlabel,
            title=fr"x-space closure: {xlabel}",
            path=output_dir / filename,
            density=density,
            xlim=xlim,
            ratio_ylim=ratio_ylim,
        )

    m_z_prior = inv_mass_ee(z_plot)
    m_x_to_z = inv_mass_ee(z_encoded)
    paper_ratio_plot_single(
        truth=m_z_prior,
        pred=m_x_to_z,
        bins=mass_bins,
        xlabel=r"$m_{ee}^{z}$ [GeV]",
        title=r"z-space mass check: MG5 z vs OTUS $x \rightarrow \tilde{z}$",
        path=output_dir / "paperstyle_zspace_mass_density_ratio.png",
        density=density,
        xlim=(args.mass_low, args.mass_high),
        ratio_ylim=(0.0, 2.0),
        reference_x=MZ_REF,
        reference_label=fr"$m_Z={MZ_REF:.4f}$ GeV",
    )

    plot_all_components(
        arrays=[(z_plot, "MG5 z", TRUTH_STYLE), (z_encoded, r"CMS x $\rightarrow$ z", ENC_STYLE)],
        title="z-space component check: MG5 z vs OTUS x -> z",
        path=output_dir / "all8_zspace_components_density.png",
        component_bins=80,
        density=density,
    )
    plot_all_components(
        arrays=[
            (x_plot, "CMS x", TRUTH_STYLE),
            (x_reco, r"x $\rightarrow$ z $\rightarrow$ x", ENCDEC_STYLE),
            (x_from_z, r"z $\rightarrow$ x", DEC_STYLE),
        ],
        title="x-space component closure: CMS x vs OTUS outputs",
        path=output_dir / "all8_xspace_components_density.png",
        component_bins=80,
        density=density,
    )

    zspace_validation_dir = output_dir / "zspace_validation"
    paper_ratio_plot_single(
        truth=m_z_prior,
        pred=m_x_to_z,
        bins=mass_bins,
        xlabel=r"$m_{ee}^{z}$ [GeV]",
        title="z-space mass check: MG5 z vs OTUS x -> z",
        path=zspace_validation_dir / "zspace_mass_mg5_vs_x2z.png",
        density=density,
        xlim=(args.mass_low, args.mass_high),
        ratio_ylim=(0.0, 2.0),
        reference_x=MZ_REF,
        reference_label=fr"$m_Z={MZ_REF:.4f}$ GeV",
    )
    plot_all_components(
        arrays=[(z_plot, "MG5 z", TRUTH_STYLE), (z_encoded, "CMS x -> z", ENC_STYLE)],
        title="z-space component check: MG5 z vs OTUS x -> z",
        path=zspace_validation_dir / "zspace_components_mg5_vs_x2z.png",
        component_bins=80,
        density=density,
    )

    ks_mass, p_mass = maybe_ks(m_z_prior, m_x_to_z)
    ks_pt_reco, p_pt_reco = maybe_ks(pt_x, pt_x_reco)
    ks_pt_zx, p_pt_zx = maybe_ks(pt_x, pt_x_from_z)
    residual_summary_reco = residual_quality_summary(
        xmass_info["residual1"],
        xmass_info["residual1_err"],
        xmass_info["truth_counts"],
        xmass_info["pred1_counts"],
    )
    residual_summary_zx = residual_quality_summary(
        xmass_info["residual2"],
        xmass_info["residual2_err"],
        xmass_info["truth_counts"],
        xmass_info["pred2_counts"],
    )

    summary_lines = [
        f"NORMALIZE_DENSITY: {density}",
        f"CHECKPOINT: {checkpoint_path}",
        f"CHECKPOINT_EPOCH: {checkpoint.get('epoch')}",
        f"CHECKPOINT_EVAL_LOSS: {checkpoint.get('eval_loss')}",
        f"SPLIT: {split}",
        f"DEVICE: {device}",
        f"HAS_SCIPY: {HAS_SCIPY}",
        f"z_mg5 shape: {z_plot.shape}",
        f"z_encoded shape: {z_encoded.shape}",
        f"x_plot shape: {x_plot.shape}",
        f"x_reco shape: {x_reco.shape}",
        f"x_from_z shape: {x_from_z.shape}",
        f"m_mg5 mean/std: {m_z_prior.mean():.6g} / {m_z_prior.std():.6g}",
        f"m_x_to_z mean/std: {m_x_to_z.mean():.6g} / {m_x_to_z.std():.6g}",
        f"m_x mean/std: {m_x.mean():.6g} / {m_x.std():.6g}",
        f"m_x_reco mean/std: {m_x_reco.mean():.6g} / {m_x_reco.std():.6g}",
        f"m_x_from_z mean/std: {m_x_from_z.mean():.6g} / {m_x_from_z.std():.6g}",
        f"mass KS statistic/pvalue: {ks_mass:.6g} / {p_mass:.6g}",
        f"pT KS x vs x->z->x: {ks_pt_reco:.6g} / {p_pt_reco:.6g}",
        f"pT KS x vs z->x: {ks_pt_zx:.6g} / {p_pt_zx:.6g}",
    ]
    for j in range(8):
        ks_j, p_j = maybe_ks(z_plot[:, j], z_encoded[:, j])
        summary_lines.append(f"z dim {j:02d} KS statistic/pvalue: {ks_j:.6g} / {p_j:.6g}")
    write_summary(output_dir / "paperstyle_summary.txt", summary_lines)
    write_summary(zspace_validation_dir / "zspace_summary.txt", summary_lines)

    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_eval_loss": checkpoint.get("eval_loss"),
        "device_report": report,
        "split": split,
        "density": density,
        "has_scipy": HAS_SCIPY,
        "shapes": {
            "x_plot": list(x_plot.shape),
            "z_plot": list(z_plot.shape),
            "z_encoded": list(z_encoded.shape),
            "x_reco": list(x_reco.shape),
            "x_from_z": list(x_from_z.shape),
        },
        "mass": {
            "z_mg5_mean": float(m_z_prior.mean()),
            "z_mg5_std": float(m_z_prior.std()),
            "x_to_z_mean": float(m_x_to_z.mean()),
            "x_to_z_std": float(m_x_to_z.std()),
            "x_mean": float(m_x.mean()),
            "x_std": float(m_x.std()),
            "x_reco_mean": float(m_x_reco.mean()),
            "x_reco_std": float(m_x_reco.std()),
            "x_from_z_mean": float(m_x_from_z.mean()),
            "x_from_z_std": float(m_x_from_z.std()),
            "zspace_ks_statistic": ks_mass,
            "zspace_ks_pvalue": p_mass,
        },
        "pt": {
            "x_vs_reco_ks_statistic": ks_pt_reco,
            "x_vs_reco_ks_pvalue": p_pt_reco,
            "x_vs_zx_ks_statistic": ks_pt_zx,
            "x_vs_zx_ks_pvalue": p_pt_zx,
        },
        "xspace_mass_residual": {
            "x_to_z_to_x": residual_summary_reco,
            "z_to_x": residual_summary_zx,
        },
    }
    (output_dir / "paperstyle_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("Saved:", output_dir / "paperstyle_summary.json")

    np.savez_compressed(
        output_dir / "paperstyle_loaded_model_outputs.npz",
        x_plot=x_plot,
        z_plot=z_plot,
        z_encoded=z_encoded,
        x_reco=x_reco,
        x_from_z=x_from_z,
        m_z_prior=m_z_prior,
        m_x_to_z=m_x_to_z,
        m_x=m_x,
        m_x_reco=m_x_reco,
        m_x_from_z=m_x_from_z,
        pt_x=pt_x,
        pt_x_reco=pt_x_reco,
        pt_x_from_z=pt_x_from_z,
    )
    print("Saved:", output_dir / "paperstyle_loaded_model_outputs.npz")
    print("Using device:", device)
    print("Device report:", json.dumps(report, sort_keys=True))
    print("Wrote plots:", output_dir)


if __name__ == "__main__":
    main()
