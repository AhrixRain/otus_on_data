"""A/B comparison plots for two CMS dilepton OTUS networks.

This script mirrors scripts/plot.py's content (x/z-space mass, dilepton pT,
principal-component, all-8 component, and summary outputs) but draws both
networks on every graph.  Both models are evaluated on the same cached
selected/split rows, so the comparison is apples-to-apples.

Example:
    .venv/bin/python scripts/test_plot.py \
        --config-a configs/cms_JpsiDoubleMuons_ab_A_mass.yaml \
        --checkpoint-a outputs/cms_JpsiDoubleMuons/jpsi_ab_A/best_model.pt \
        --config-b configs/cms_JpsiDoubleMuons_ab_B_no_mass.yaml \
        --checkpoint-b outputs/cms_JpsiDoubleMuons/jpsi_ab_B/best_model.pt \
        --num-samples 50000 --split test --device auto
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# plot.py's module-level code configures the thread environment from --threads
# before importing numpy/torch/matplotlib, so import it first.
import plot as plot_lib  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from cms_data import (  # noqa: E402
    load_and_split_cached,
    load_config,
    resolve_config,
    save_resolved_config,
)
from cms_model import load_model_from_checkpoint  # noqa: E402
from device_utils import device_report, select_device  # noqa: E402


plt = plot_lib.plt

TRUTH_STYLE = plot_lib.TRUTH_STYLE
TRUTH_LABEL_X = plot_lib.TRUTH_LABEL_X
TRUTH_LABEL_Z = plot_lib.TRUTH_LABEL_Z

# Network color distinguishes A from B; linestyle preserves plot.py's mode
# conventions (":" for x -> z -> x reconstruction, "--" for z -> x generation
# and for z-space encoder output).
STYLE_A_RECO = dict(color="crimson", linewidth=1.8, linestyle=":")
STYLE_A_GEN = dict(color="crimson", linewidth=1.8, linestyle="--")
STYLE_A_Z = dict(color="crimson", linewidth=1.8, linestyle="--")
STYLE_B_RECO = dict(color="royalblue", linewidth=1.8, linestyle=":")
STYLE_B_GEN = dict(color="royalblue", linewidth=1.8, linestyle="--")
STYLE_B_Z = dict(color="royalblue", linewidth=1.8, linestyle="--")

MUON_CHANNELS = {"muon", "doublemuon", "doublemuons", "mumu", "jpsi_mumu"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "A/B comparison plots for two CMS dilepton OTUS networks. "
            "Same content as plot.py, with both networks on every graph."
        )
    )
    parser.add_argument("--config-a", type=Path, required=True, help="YAML/JSON config for network A.")
    parser.add_argument(
        "--checkpoint-a",
        type=Path,
        required=True,
        help="best_model.pt or last_model.pt for network A.",
    )
    parser.add_argument("--label-a", default=None, help="Short legend label for network A (defaults to mass loss weights).")
    parser.add_argument("--config-b", type=Path, required=True, help="YAML/JSON config for network B.")
    parser.add_argument(
        "--checkpoint-b",
        type=Path,
        required=True,
        help="best_model.pt or last_model.pt for network B.",
    )
    parser.add_argument("--label-b", default=None, help="Short legend label for network B (defaults to mass loss weights).")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Plot output directory.")
    parser.add_argument("--num-samples", type=int, default=None, help="Limit selected CMS and MG5 rows before splitting.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory for selected/split data cache. Defaults to <output_root>/.plot_cache.",
    )
    parser.add_argument(
        "--no-data-cache",
        action="store_true",
        help="Disable selected/split data caching and reread ROOT/HDF5 inputs.",
    )
    parser.add_argument(
        "--split",
        choices=("all", "val", "test"),
        default="all",
        help="Data split to plot. Use all selected rows by default.",
    )
    parser.add_argument("--max-x-events", type=int, default=5000000, help="Maximum CMS x events to plot.")
    parser.add_argument("--max-z-events", type=int, default=5000000, help="Maximum MG5 z events to plot.")
    parser.add_argument("--batch-size", type=int, default=None, help="Inference batch size.")
    parser.add_argument(
        "--progress-log-steps",
        type=int,
        default=10,
        help="Log inference progress every N batches. Use 0 to log only phase boundaries.",
    )
    parser.add_argument(
        "--threads",
        default="auto",
        help="CPU worker threads for NumPy/OpenMP/Torch. Use auto/max/all for all logical CPUs.",
    )
    parser.add_argument(
        "--interop-threads",
        default="auto",
        help="PyTorch inter-op threads. Use auto/max/all for all logical CPUs.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Plot subsampling seed.")
    parser.add_argument(
        "--counts",
        action="store_true",
        help="Plot raw counts instead of normalized densities.",
    )
    parser.add_argument("--mass-low", type=float, default=None, help="Low edge for mass plots.")
    parser.add_argument("--mass-high", type=float, default=None, help="High edge for mass plots.")
    parser.add_argument(
        "--mass-bin-width",
        type=float,
        default=None,
        help="Mass bin width in GeV. Defaults to evaluation.mass_bin_width.",
    )
    return parser.parse_args()


def default_label(prefix: str, config: dict[str, Any]) -> str:
    loss = config.get("loss", {})
    mw1 = loss.get("mass_w1")
    pair_w1 = loss.get("pair_mass_w1")
    if mw1 is not None and pair_w1 is not None and float(mw1) == float(pair_w1):
        return f"{prefix} (mass_w1={float(mw1):g})"
    parts: list[str] = []
    if mw1 is not None:
        parts.append(f"mass_w1={float(mw1):g}")
    if pair_w1 is not None:
        parts.append(f"pair_mass_w1={float(pair_w1):g}")
    if parts:
        return f"{prefix} ({', '.join(parts)})"
    return prefix


def selection_key(config: dict[str, Any]) -> str:
    channel = str(config.get("data", {}).get("channel", "electron")).lower()
    return "muon_selection" if channel in MUON_CHANNELS else "electron_selection"


def data_config_payload(config: dict[str, Any], num_samples: int | None) -> dict[str, Any]:
    paths = config["paths"]
    key = selection_key(config)
    payload: dict[str, Any] = {
        "data": config.get("data", {}),
        "data_split": config.get("data_split"),
        "seed": config.get("seed"),
        "float_type": config.get("float_type"),
        "cms_root_file": paths.get("cms_root_file"),
        "theory_prior_file": paths.get("theory_prior_file"),
        key: config.get(key),
        "num_samples": num_samples,
    }
    return payload


def verify_same_data_config(config_a: dict[str, Any], config_b: dict[str, Any], num_samples: int | None) -> None:
    payload_a = data_config_payload(config_a, num_samples)
    payload_b = data_config_payload(config_b, num_samples)
    if payload_a != payload_b:
        raise ValueError(
            "A and B configs must select identical data for a fair A/B comparison.\n"
            f"A: {payload_a}\nB: {payload_b}"
        )


def verify_same_evaluation_config(config_a: dict[str, Any], config_b: dict[str, Any]) -> None:
    keys = (
        "mass_range",
        "mass_bin_width",
        "mass_label",
        "channel_title",
        "reference_mass",
        "reference_label",
        "min_truth_count",
    )
    eval_a = config_a.get("evaluation", {})
    eval_b = config_b.get("evaluation", {})
    diffs = [key for key in keys if eval_a.get(key) != eval_b.get(key)]
    if diffs:
        raise ValueError(
            "A and B evaluation settings must match for comparable plots. "
            f"Different keys: {diffs}"
        )


def paper_ratio_plot_multi(
    truth: np.ndarray,
    series: list[dict[str, Any]],
    bins: np.ndarray,
    xlabel: str,
    title: str,
    path: Path,
    density: bool,
    truth_label: str = TRUTH_LABEL_X,
    xlim: tuple[float, float] | None = None,
    ratio_ylim: tuple[float, float] = (0.5, 1.5),
    residual_ylim: tuple[float, float] = (-0.05, 0.05),
    residual_ylabel: str = "Rel. diff.\nto truth",
    reference_x: float | None = None,
    reference_label: str | None = None,
) -> dict[str, Any]:
    """plot.py's paper ratio plot, with an arbitrary number of prediction series."""
    truth = plot_lib.finite_values(truth)
    centers, h_truth, c_truth = plot_lib.hist_for_plot(truth, bins, density=density)

    prepared: list[dict[str, Any]] = []
    for item in series:
        values = plot_lib.finite_values(item["values"])
        _, h_pred, c_pred = plot_lib.hist_for_plot(values, bins, density=density)
        ratio, err = plot_lib.ratio_and_error(h_pred, h_truth, c_pred, c_truth)
        w2_shown = plot_lib.w2_1d_in_histogram_range_np(truth, values, bins)
        prepared.append(
            {
                "label": item["label"],
                "short": item.get("short", item["label"]),
                "style": item["style"],
                "marker": item.get("marker", "o"),
                "heights": h_pred,
                "counts": c_pred,
                "ratio": ratio,
                "err": err,
                "w2_shown": w2_shown,
            }
        )

    fig = plt.figure(figsize=(7.0, 7.4))
    gs = fig.add_gridspec(3, 1, height_ratios=[3.0, 1.05, 1.05], hspace=0.05)
    ax = fig.add_subplot(gs[0])
    rax = fig.add_subplot(gs[1], sharex=ax)
    dax = fig.add_subplot(gs[2], sharex=ax)

    plot_lib.draw_hist_step(ax, centers, h_truth, truth_label, TRUTH_STYLE)
    for item in prepared:
        plot_lib.draw_hist_step(ax, centers, item["heights"], item["label"], item["style"])
    if reference_x is not None:
        ax.axvline(
            reference_x,
            linestyle="--",
            linewidth=1.2,
            color="gray",
            label=reference_label,
        )
    for idx, item in enumerate(prepared):
        ax.text(
            0.97,
            0.90 - idx * 0.065,
            fr"$W_{{2,\mathrm{{shown}}}}^2$({item['short']}) = {item['w2_shown']:.3e}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            color=item["style"].get("color", "black"),
            fontsize=7.5,
        )
    ax.set_ylabel(r"Normalized density" if density else "Counts")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.18)

    for item in prepared:
        rax.errorbar(
            centers,
            item["ratio"],
            yerr=item["err"],
            fmt=item["marker"],
            markersize=3.0,
            color=item["style"].get("color", "black"),
            ecolor=item["style"].get("color", "black"),
            elinewidth=1.0,
            capsize=0,
            label=item["label"],
        )
    rax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
    rax.set_ylabel("Ratio\nto truth")
    rax.set_ylim(*ratio_ylim)
    rax.grid(alpha=0.18)
    rax.legend(loc="best", fontsize=7.5)

    for item in prepared:
        dax.errorbar(
            centers,
            item["ratio"] - 1.0,
            yerr=item["err"],
            fmt=item["marker"],
            markersize=3.0,
            color=item["style"].get("color", "black"),
            ecolor=item["style"].get("color", "black"),
            elinewidth=1.0,
            capsize=0,
            label=item["label"],
        )
    plot_lib.draw_relative_error_target(dax)
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
        "truth_counts": c_truth,
        "series": prepared,
    }


def run_inference(
    model: torch.nn.Module,
    x_plot: np.ndarray,
    z_for_x: np.ndarray,
    device: torch.device,
    batch_size: int,
    progress_log_steps: int,
    tag: str,
) -> dict[str, np.ndarray]:
    plot_lib.log_progress(f"Starting {tag} network inference.")
    return {
        "z_encoded": plot_lib.predict_batches(model, "encode", x_plot, device, batch_size, progress_log_steps),
        "x_reco": plot_lib.predict_batches(model, "reconstruct", x_plot, device, batch_size, progress_log_steps),
        "x_from_z": plot_lib.predict_batches(model, "decode", z_for_x, device, batch_size, progress_log_steps),
    }


def x_space_series(
    run_letter: str,
    run_label: str,
    out: dict[str, np.ndarray],
    reco_style: dict[str, Any],
    gen_style: dict[str, Any],
    reco_marker: str,
    gen_marker: str,
) -> list[dict[str, Any]]:
    return [
        {
            "label": run_label + r": $x \rightarrow \tilde{z} \rightarrow \tilde{x}$",
            "short": f"{run_letter}-reco",
            "values": out["x_reco"],
            "style": reco_style,
            "marker": reco_marker,
        },
        {
            "label": run_label + r": $z \rightarrow \tilde{x}^{\prime}$",
            "short": f"{run_letter}-gen",
            "values": out["x_from_z"],
            "style": gen_style,
            "marker": gen_marker,
        },
    ]


def z_space_series(
    run_letter: str,
    run_label: str,
    out: dict[str, np.ndarray],
    style: dict[str, Any],
    marker: str,
) -> list[dict[str, Any]]:
    return [
        {
            "label": run_label + r": $x \rightarrow \tilde{z}$",
            "short": run_letter,
            "values": out["z_encoded"],
            "style": style,
            "marker": marker,
        }
    ]


def summary_row(item: dict[str, Any]) -> str:
    return (
        f"{item['valid_bins']}/{item['total_bins']} "
        f"{item['frac_within_1pct']} {item['rms_rel_residual']} "
        f"{item['mae_rel_residual']} {item['max_abs_rel_residual']} "
        f"{item['median_stat_error']} {item['ks_statistic']} "
        f"{item['w1_distance']} {item['chi2_ndf']}"
    )


def main() -> None:
    args = parse_args()
    threads = plot_lib.parse_thread_count(args.threads, plot_lib.MAX_THREADS)
    interop_threads = plot_lib.parse_thread_count(args.interop_threads, plot_lib.MAX_THREADS)
    thread_report = plot_lib.configure_torch_threads(threads, interop_threads)
    plot_lib.log_progress("Starting CMS dilepton A/B plot generation.")
    plot_lib.log_progress(
        "Thread configuration: "
        f"threads={thread_report['torch_num_threads']}, "
        f"interop_threads={thread_report['torch_num_interop_threads']}, "
        f"max_available={plot_lib.MAX_THREADS}"
    )

    config_a = resolve_config(load_config(args.config_a))
    config_b = resolve_config(load_config(args.config_b))
    verify_same_data_config(config_a, config_b, args.num_samples)
    verify_same_evaluation_config(config_a, config_b)

    seed = int(config_a.get("seed", 0) if args.seed is None else args.seed)
    density = not args.counts
    device = select_device(args.device)
    report = device_report(device)
    plot_lib.log_progress(f"Using device: {device}")

    checkpoint_a = args.checkpoint_a.expanduser().resolve()
    checkpoint_b = args.checkpoint_b.expanduser().resolve()
    if args.output_dir:
        output_dir = args.output_dir.expanduser().resolve()
    else:
        output_dir = Path(config_a["paths"]["output_root"]) / "ab_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_a, model_config_a, _stats_a, ckpt_a = load_model_from_checkpoint(
        checkpoint_a,
        config=config_a,
        map_location=torch.device("cpu"),
    )
    model_b, model_config_b, _stats_b, ckpt_b = load_model_from_checkpoint(
        checkpoint_b,
        config=config_b,
        map_location=torch.device("cpu"),
    )
    model_a.to(device)
    model_b.to(device)
    save_resolved_config(model_config_a, output_dir / "config_a.resolved.json")
    save_resolved_config(model_config_b, output_dir / "config_b.resolved.json")

    evaluation_config = model_config_a.get("evaluation", {})
    configured_mass_range = evaluation_config.get("mass_range", [70.0, 110.0])
    mass_low = float(configured_mass_range[0] if args.mass_low is None else args.mass_low)
    mass_high = float(configured_mass_range[1] if args.mass_high is None else args.mass_high)
    mass_bin_width = float(
        evaluation_config.get("mass_bin_width", 1.0)
        if args.mass_bin_width is None
        else args.mass_bin_width
    )
    if mass_high <= mass_low or mass_bin_width <= 0.0:
        raise ValueError("Mass plot range and bin width must be positive and ordered.")
    min_truth_count = int(evaluation_config.get("min_truth_count", 20))
    mass_label = str(evaluation_config.get("mass_label", "m(ll) [GeV]"))
    channel_title = str(evaluation_config.get("channel_title", "CMS dilepton"))
    reference_mass = evaluation_config.get("reference_mass")
    reference_mass = None if reference_mass is None else float(reference_mass)
    reference_label = str(evaluation_config.get("reference_label", "resonance mass"))
    channel = str(model_config_a.get("data", {}).get("channel", "electron")).lower()
    is_muon = channel in MUON_CHANNELS
    pair_symbol = r"\mu\mu" if is_muon else "ee"
    positive_particle = "Positive muon" if is_muon else "Positron"
    component_symbol = r"\mu" if is_muon else "e"
    component_titles = [
        fr"${component_symbol}^-\ p_x$",
        fr"${component_symbol}^-\ p_y$",
        fr"${component_symbol}^-\ p_z$",
        fr"${component_symbol}^-\ E$",
        fr"${component_symbol}^+\ p_x$",
        fr"${component_symbol}^+\ p_y$",
        fr"${component_symbol}^+\ p_z$",
        fr"${component_symbol}^+\ E$",
    ]

    arrays, cache_info = load_and_split_cached(
        model_config_a,
        num_samples=args.num_samples,
        cache_dir=args.cache_dir,
        use_cache=not args.no_data_cache,
        log=plot_lib.log_progress,
    )
    split = args.split
    plot_lib.log_progress(f"Selecting split={split}.")
    x_split = plot_lib.finite_8d(plot_lib.select_split(arrays, "x", split), f"x_{split}")
    z_split = plot_lib.finite_8d(plot_lib.select_split(arrays, "z", split), f"z_{split}")
    plot_lib.log_progress(f"Selected rows before caps: x={x_split.shape}, z={z_split.shape}")

    x_plot = plot_lib.random_subset(x_split, args.max_x_events, seed=seed)
    z_plot = plot_lib.random_subset(z_split, args.max_z_events, seed=seed + 1)
    z_for_x = plot_lib.random_subset(z_split, len(x_plot), seed=seed + 2)
    plot_lib.log_progress(f"Rows after caps/subsampling: x_plot={x_plot.shape}, z_plot={z_plot.shape}, z_for_x={z_for_x.shape}")

    batch_size = int(args.batch_size or model_config_a.get("loaders", {}).get("eval_batch_size", 20000))
    batch_size = max(1, batch_size)
    out_a = run_inference(model_a, x_plot, z_for_x, device, batch_size, args.progress_log_steps, "A")
    out_b = run_inference(model_b, x_plot, z_for_x, device, batch_size, args.progress_log_steps, "B")

    label_a = args.label_a or default_label("A", model_config_a)
    label_b = args.label_b or default_label("B", model_config_b)
    x_series = x_space_series("A", label_a, out_a, STYLE_A_RECO, STYLE_A_GEN, "o", "^") + x_space_series(
        "B", label_b, out_b, STYLE_B_RECO, STYLE_B_GEN, "s", "D"
    )
    z_series = z_space_series("A", label_a, out_a, STYLE_A_Z, "o") + z_space_series(
        "B", label_b, out_b, STYLE_B_Z, "s"
    )

    plot_lib.log_progress("Computing observables.")
    mass_bins = np.arange(mass_low, mass_high + mass_bin_width, mass_bin_width)
    pt_bins = np.linspace(0.0, 100.0, 101)
    bins_y = np.array([-100, -60] + [-50 + 5 * i for i in range(21)] + [60, 100], dtype=float)
    bins_z = np.array([-400] + [-250 + 20 * i for i in range(26)] + [400], dtype=float)
    bins_e = np.array([0] + [20 + 10 * i for i in range(26)] + [400], dtype=float)

    m_x = plot_lib.inv_mass_ee(x_plot)
    m_x_reco_a = plot_lib.inv_mass_ee(out_a["x_reco"])
    m_x_reco_b = plot_lib.inv_mass_ee(out_b["x_reco"])
    m_x_from_z_a = plot_lib.inv_mass_ee(out_a["x_from_z"])
    m_x_from_z_b = plot_lib.inv_mass_ee(out_b["x_from_z"])
    m_z_prior = plot_lib.inv_mass_ee(z_plot)
    m_x_to_z_a = plot_lib.inv_mass_ee(out_a["z_encoded"])
    m_x_to_z_b = plot_lib.inv_mass_ee(out_b["z_encoded"])

    plot_lib.log_progress("Writing x-space mass plot.")
    mass_info = paper_ratio_plot_multi(
        truth=m_x,
        series=x_series,
        bins=mass_bins,
        xlabel=mass_label,
        title=f"{channel_title}: x-space mass (A vs B)",
        path=output_dir / "paperstyle_xspace_mass_density_ratio_ab.png",
        density=density,
        xlim=(mass_low, mass_high),
        ratio_ylim=(0.5, 1.5),
        residual_ylim=(-0.05, 0.05),
        residual_ylabel=r"$(\mathrm{OTUS}-\mathrm{data})/\mathrm{data}$",
        reference_x=reference_mass,
        reference_label=None if reference_mass is None else f"{reference_label}={reference_mass:.4f} GeV",
    )

    pt_x = plot_lib.pt_ee(x_plot)
    pt_x_reco_a = plot_lib.pt_ee(out_a["x_reco"])
    pt_x_reco_b = plot_lib.pt_ee(out_b["x_reco"])
    pt_x_from_z_a = plot_lib.pt_ee(out_a["x_from_z"])
    pt_x_from_z_b = plot_lib.pt_ee(out_b["x_from_z"])
    plot_lib.log_progress("Writing x-space pT plot.")
    paper_ratio_plot_multi(
        truth=pt_x,
        series=x_series,
        bins=pt_bins,
        xlabel=fr"$p_T({pair_symbol})$ [GeV]",
        title=f"{channel_title}: dilepton transverse momentum (A vs B)",
        path=output_dir / "paperstyle_xspace_pt_density_ratio_ab.png",
        density=density,
        xlim=(0.0, 100.0),
        ratio_ylim=(0.0, 2.0),
    )

    principal_specs_z = [
        (5, bins_y, f"{positive_particle} $p_y$ [GeV]", (-100.0, 100.0), (0.5, 1.5), "paperstyle_zspace_pos_py_ratio_ab.png"),
        (6, bins_z, f"{positive_particle} $p_z$ [GeV]", (-400.0, 400.0), (0.5, 1.5), "paperstyle_zspace_pos_pz_ratio_ab.png"),
        (7, bins_e, f"{positive_particle} $E$ [GeV]", (0.0, 400.0), (0.5, 1.5), "paperstyle_zspace_pos_E_ratio_ab.png"),
    ]
    plot_lib.log_progress("Writing selected z-space component plots.")
    for idx, bins, xlabel, xlim, ratio_ylim, filename in principal_specs_z:
        paper_ratio_plot_multi(
            truth=z_plot[:, idx],
            series=z_series,
            bins=bins,
            xlabel=xlabel,
            title=fr"z-space closure: {xlabel} (A vs B)",
            path=output_dir / filename,
            density=density,
            truth_label=TRUTH_LABEL_Z,
            xlim=xlim,
            ratio_ylim=ratio_ylim,
        )

    principal_specs_x = [
        (5, bins_y, f"{positive_particle} $p_y$ [GeV]", (-100.0, 100.0), (0.5, 1.5), "paperstyle_xspace_pos_py_ratio_ab.png"),
        (6, bins_z, f"{positive_particle} $p_z$ [GeV]", (-400.0, 400.0), (0.5, 1.5), "paperstyle_xspace_pos_pz_ratio_ab.png"),
        (7, bins_e, f"{positive_particle} $E$ [GeV]", (0.0, 400.0), (0.5, 1.5), "paperstyle_xspace_pos_E_ratio_ab.png"),
    ]
    plot_lib.log_progress("Writing selected x-space component plots.")
    for idx, bins, xlabel, xlim, ratio_ylim, filename in principal_specs_x:
        paper_ratio_plot_multi(
            truth=x_plot[:, idx],
            series=x_series,
            bins=bins,
            xlabel=xlabel,
            title=fr"x-space closure: {xlabel} (A vs B)",
            path=output_dir / filename,
            density=density,
            xlim=xlim,
            ratio_ylim=ratio_ylim,
        )

    plot_lib.log_progress("Writing z-space mass plot.")
    paper_ratio_plot_multi(
        truth=m_z_prior,
        series=z_series,
        bins=mass_bins,
        xlabel=mass_label,
        title="z-space mass check: MG5 z vs OTUS x -> z (A vs B)",
        path=output_dir / "paperstyle_zspace_mass_density_ratio_ab.png",
        density=density,
        truth_label=TRUTH_LABEL_Z,
        xlim=(mass_low, mass_high),
        ratio_ylim=(0.0, 2.0),
        reference_x=reference_mass,
        reference_label=None if reference_mass is None else f"{reference_label}={reference_mass:.4f} GeV",
    )

    plot_lib.log_progress("Writing all-component density checks.")
    plot_lib.plot_all_components(
        arrays=[
            (z_plot, "MG5 z", TRUTH_STYLE),
            (out_a["z_encoded"], f"{label_a}: " + r"$x \rightarrow \tilde{z}$", STYLE_A_Z),
            (out_b["z_encoded"], f"{label_b}: " + r"$x \rightarrow \tilde{z}$", STYLE_B_Z),
        ],
        title="z-space component check: MG5 z vs OTUS x -> z (A vs B)",
        path=output_dir / "all8_zspace_components_density_ab.png",
        component_bins=80,
        density=density,
        component_titles=component_titles,
    )
    plot_lib.plot_all_components(
        arrays=[
            (x_plot, "CMS x", TRUTH_STYLE),
            (out_a["x_reco"], f"{label_a}: " + r"$x \rightarrow \tilde{z} \rightarrow \tilde{x}$", STYLE_A_RECO),
            (out_a["x_from_z"], f"{label_a}: " + r"$z \rightarrow \tilde{x}^{\prime}$", STYLE_A_GEN),
            (out_b["x_reco"], f"{label_b}: " + r"$x \rightarrow \tilde{z} \rightarrow \tilde{x}$", STYLE_B_RECO),
            (out_b["x_from_z"], f"{label_b}: " + r"$z \rightarrow \tilde{x}^{\prime}$", STYLE_B_GEN),
        ],
        title="x-space component closure: CMS x vs OTUS outputs (A vs B)",
        path=output_dir / "all8_xspace_components_density_ab.png",
        component_bins=80,
        density=density,
        component_titles=component_titles,
    )

    zspace_validation_dir = output_dir / "zspace_validation"
    plot_lib.log_progress("Writing z-space validation plots.")
    paper_ratio_plot_multi(
        truth=m_z_prior,
        series=z_series,
        bins=mass_bins,
        xlabel=mass_label,
        title="z-space mass check: MG5 z vs OTUS x -> z (A vs B)",
        path=zspace_validation_dir / "zspace_mass_mg5_vs_x2z_ab.png",
        density=density,
        truth_label=TRUTH_LABEL_Z,
        xlim=(mass_low, mass_high),
        ratio_ylim=(0.0, 2.0),
        reference_x=reference_mass,
        reference_label=None if reference_mass is None else f"{reference_label}={reference_mass:.4f} GeV",
    )
    plot_lib.plot_all_components(
        arrays=[
            (z_plot, "MG5 z", TRUTH_STYLE),
            (out_a["z_encoded"], f"{label_a}: " + r"$x \rightarrow \tilde{z}$", STYLE_A_Z),
            (out_b["z_encoded"], f"{label_b}: " + r"$x \rightarrow \tilde{z}$", STYLE_B_Z),
        ],
        title="z-space component check: MG5 z vs OTUS x -> z (A vs B)",
        path=zspace_validation_dir / "zspace_components_mg5_vs_x2z_ab.png",
        component_bins=80,
        density=density,
        component_titles=component_titles,
    )

    plot_lib.log_progress("Computing residual and statistical summaries.")
    ks_mass_a, p_mass_a = plot_lib.maybe_ks(m_z_prior, m_x_to_z_a)
    ks_mass_b, p_mass_b = plot_lib.maybe_ks(m_z_prior, m_x_to_z_b)
    ks_pt_reco_a, p_pt_reco_a = plot_lib.maybe_ks(pt_x, pt_x_reco_a)
    ks_pt_reco_b, p_pt_reco_b = plot_lib.maybe_ks(pt_x, pt_x_reco_b)
    ks_pt_zx_a, p_pt_zx_a = plot_lib.maybe_ks(pt_x, pt_x_from_z_a)
    ks_pt_zx_b, p_pt_zx_b = plot_lib.maybe_ks(pt_x, pt_x_from_z_b)

    mass_component = "m_mumu" if is_muon else "m_ee"
    pt_component = f"pT_{'mumu' if is_muon else 'ee'}"
    lepton_prefix = "mu" if is_muon else "e"
    observable_specs = [
        (mass_component, m_x, m_x_from_z_a, m_x_from_z_b, mass_bins),
        (pt_component, pt_x, pt_x_from_z_a, pt_x_from_z_b, pt_bins),
        (f"{lepton_prefix}_minus_E", x_plot[:, 3], out_a["x_from_z"][:, 3], out_b["x_from_z"][:, 3], bins_e),
        (f"{lepton_prefix}_plus_E", x_plot[:, 7], out_a["x_from_z"][:, 7], out_b["x_from_z"][:, 7], bins_e),
        (f"{lepton_prefix}_minus_py", x_plot[:, 1], out_a["x_from_z"][:, 1], out_b["x_from_z"][:, 1], bins_y),
        (f"{lepton_prefix}_plus_py", x_plot[:, 5], out_a["x_from_z"][:, 5], out_b["x_from_z"][:, 5], bins_y),
        (f"{lepton_prefix}_minus_pz", x_plot[:, 2], out_a["x_from_z"][:, 2], out_b["x_from_z"][:, 2], bins_z),
        (f"{lepton_prefix}_plus_pz", x_plot[:, 6], out_a["x_from_z"][:, 6], out_b["x_from_z"][:, 6], bins_z),
        ("zspace_mass", m_z_prior, m_x_to_z_a, m_x_to_z_b, mass_bins),
    ]
    observable_summaries: dict[str, dict[str, Any]] = {}
    for name, truth_values, pred_a, pred_b, obs_bins in observable_specs:
        observable_summaries[name] = {
            "a": plot_lib.histogram_observable_summary(
                truth_values, pred_a, obs_bins, density, min_truth_count
            ),
            "b": plot_lib.histogram_observable_summary(
                truth_values, pred_b, obs_bins, density, min_truth_count
            ),
            "w2_shown": {
                "a": plot_lib.w2_1d_in_histogram_range_np(truth_values, pred_a, obs_bins),
                "b": plot_lib.w2_1d_in_histogram_range_np(truth_values, pred_b, obs_bins),
            },
        }

    z_component_summaries: dict[str, dict[str, Any]] = {}
    for j in range(8):
        values = np.concatenate([z_plot[:, j], out_a["z_encoded"][:, j], out_b["z_encoded"][:, j]])
        lo, hi = np.nanpercentile(values, [0.5, 99.5])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = np.nanmin(values), np.nanmax(values)
        z_bins = np.linspace(lo, hi, 81)
        z_component_summaries[f"z_component_{j:02d}"] = {
            "a": plot_lib.histogram_observable_summary(
                z_plot[:, j], out_a["z_encoded"][:, j], z_bins, density, min_truth_count
            ),
            "b": plot_lib.histogram_observable_summary(
                z_plot[:, j], out_b["z_encoded"][:, j], z_bins, density, min_truth_count
            ),
        }

    summary_lines = [
        f"NORMALIZE_DENSITY: {density}",
        f"CHECKPOINT_A: {checkpoint_a}",
        f"CHECKPOINT_A_EPOCH: {ckpt_a.get('epoch')}",
        f"CHECKPOINT_A_EVAL_LOSS: {ckpt_a.get('eval_loss')}",
        f"CHECKPOINT_B: {checkpoint_b}",
        f"CHECKPOINT_B_EPOCH: {ckpt_b.get('epoch')}",
        f"CHECKPOINT_B_EVAL_LOSS: {ckpt_b.get('eval_loss')}",
        f"LABEL_A: {label_a}",
        f"LABEL_B: {label_b}",
        f"SPLIT: {split}",
        f"DATA_CACHE: {cache_info}",
        f"DEVICE: {device}",
        f"THREADS: {thread_report}",
        f"HAS_SCIPY: {plot_lib.HAS_SCIPY}",
        f"z_mg5 shape: {z_plot.shape}",
        f"x_plot shape: {x_plot.shape}",
        f"m_mg5 mean/std: {m_z_prior.mean():.6g} / {m_z_prior.std():.6g}",
        f"m_x mean/std: {m_x.mean():.6g} / {m_x.std():.6g}",
        f"m_x_to_z A mean/std: {m_x_to_z_a.mean():.6g} / {m_x_to_z_a.std():.6g}",
        f"m_x_to_z B mean/std: {m_x_to_z_b.mean():.6g} / {m_x_to_z_b.std():.6g}",
        f"m_x_reco A mean/std: {m_x_reco_a.mean():.6g} / {m_x_reco_a.std():.6g}",
        f"m_x_reco B mean/std: {m_x_reco_b.mean():.6g} / {m_x_reco_b.std():.6g}",
        f"m_x_from_z A mean/std: {m_x_from_z_a.mean():.6g} / {m_x_from_z_a.std():.6g}",
        f"m_x_from_z B mean/std: {m_x_from_z_b.mean():.6g} / {m_x_from_z_b.std():.6g}",
        f"mass KS z vs x->z A: {ks_mass_a:.6g} / {p_mass_a:.6g}",
        f"mass KS z vs x->z B: {ks_mass_b:.6g} / {p_mass_b:.6g}",
        f"pT KS x vs x->z->x A: {ks_pt_reco_a:.6g} / {p_pt_reco_a:.6g}",
        f"pT KS x vs x->z->x B: {ks_pt_reco_b:.6g} / {p_pt_reco_b:.6g}",
        f"pT KS x vs z->x A: {ks_pt_zx_a:.6g} / {p_pt_zx_a:.6g}",
        f"pT KS x vs z->x B: {ks_pt_zx_b:.6g} / {p_pt_zx_b:.6g}",
        "",
        "RESIDUAL SUMMARY TABLE (z->x for x-space; x->z for z-space):",
        "observable valid/total frac_1pct rms mae max_abs median_stat_err ks w1 chi2_ndf",
    ]
    for name, item in observable_summaries.items():
        summary_lines.append(f"{name} A: {summary_row(item['a'])}")
        summary_lines.append(f"{name} B: {summary_row(item['b'])}")
        summary_lines.append(
            f"{name} W2_shown A/B: {item['w2_shown']['a']:.6g} / {item['w2_shown']['b']:.6g}"
        )
    for name, item in z_component_summaries.items():
        summary_lines.append(f"{name} A: {summary_row(item['a'])}")
        summary_lines.append(f"{name} B: {summary_row(item['b'])}")
    for j in range(8):
        ks_a, p_a = plot_lib.maybe_ks(z_plot[:, j], out_a["z_encoded"][:, j])
        ks_b, p_b = plot_lib.maybe_ks(z_plot[:, j], out_b["z_encoded"][:, j])
        summary_lines.append(f"z dim {j:02d} KS A statistic/pvalue: {ks_a:.6g} / {p_a:.6g}")
        summary_lines.append(f"z dim {j:02d} KS B statistic/pvalue: {ks_b:.6g} / {p_b:.6g}")
    plot_lib.write_summary(output_dir / "paperstyle_summary_ab.txt", summary_lines)
    plot_lib.write_summary(zspace_validation_dir / "zspace_summary_ab.txt", summary_lines)

    summary = {
        "checkpoint_a": str(checkpoint_a),
        "checkpoint_a_epoch": ckpt_a.get("epoch"),
        "checkpoint_a_eval_loss": ckpt_a.get("eval_loss"),
        "checkpoint_b": str(checkpoint_b),
        "checkpoint_b_epoch": ckpt_b.get("epoch"),
        "checkpoint_b_eval_loss": ckpt_b.get("eval_loss"),
        "labels": {"a": label_a, "b": label_b},
        "device_report": report,
        "thread_report": thread_report,
        "split": split,
        "data_cache": cache_info,
        "density": density,
        "has_scipy": plot_lib.HAS_SCIPY,
        "shapes": {
            "x_plot": list(x_plot.shape),
            "z_plot": list(z_plot.shape),
            "z_encoded_a": list(out_a["z_encoded"].shape),
            "z_encoded_b": list(out_b["z_encoded"].shape),
            "x_reco_a": list(out_a["x_reco"].shape),
            "x_reco_b": list(out_b["x_reco"].shape),
            "x_from_z_a": list(out_a["x_from_z"].shape),
            "x_from_z_b": list(out_b["x_from_z"].shape),
        },
        "mass": {
            "z_mg5_mean": float(m_z_prior.mean()),
            "z_mg5_std": float(m_z_prior.std()),
            "x_mean": float(m_x.mean()),
            "x_std": float(m_x.std()),
            "x_to_z_a_mean": float(m_x_to_z_a.mean()),
            "x_to_z_a_std": float(m_x_to_z_a.std()),
            "x_to_z_b_mean": float(m_x_to_z_b.mean()),
            "x_to_z_b_std": float(m_x_to_z_b.std()),
            "x_reco_a_mean": float(m_x_reco_a.mean()),
            "x_reco_a_std": float(m_x_reco_a.std()),
            "x_reco_b_mean": float(m_x_reco_b.mean()),
            "x_reco_b_std": float(m_x_reco_b.std()),
            "x_from_z_a_mean": float(m_x_from_z_a.mean()),
            "x_from_z_a_std": float(m_x_from_z_a.std()),
            "x_from_z_b_mean": float(m_x_from_z_b.mean()),
            "x_from_z_b_std": float(m_x_from_z_b.std()),
            "zspace_ks_statistic": {"a": ks_mass_a, "b": ks_mass_b},
            "zspace_ks_pvalue": {"a": p_mass_a, "b": p_mass_b},
        },
        "pt": {
            "x_vs_reco_ks_statistic": {"a": ks_pt_reco_a, "b": ks_pt_reco_b},
            "x_vs_reco_ks_pvalue": {"a": p_pt_reco_a, "b": p_pt_reco_b},
            "x_vs_zx_ks_statistic": {"a": ks_pt_zx_a, "b": ks_pt_zx_b},
            "x_vs_zx_ks_pvalue": {"a": p_pt_zx_a, "b": p_pt_zx_b},
        },
        "observable_residuals": observable_summaries,
        "zspace_component_residuals": z_component_summaries,
    }
    plot_lib.log_progress("Writing summary files.")
    (output_dir / "paperstyle_summary_ab.json").write_text(
        json.dumps(plot_lib.json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("Saved:", output_dir / "paperstyle_summary_ab.json")

    plot_lib.log_progress("Writing compressed model output arrays.")
    np.savez_compressed(
        output_dir / "paperstyle_ab_loaded_model_outputs.npz",
        x_plot=x_plot,
        z_plot=z_plot,
        z_encoded_a=out_a["z_encoded"],
        z_encoded_b=out_b["z_encoded"],
        x_reco_a=out_a["x_reco"],
        x_reco_b=out_b["x_reco"],
        x_from_z_a=out_a["x_from_z"],
        x_from_z_b=out_b["x_from_z"],
        m_z_prior=m_z_prior,
        m_x_to_z_a=m_x_to_z_a,
        m_x_to_z_b=m_x_to_z_b,
        m_x=m_x,
        m_x_reco_a=m_x_reco_a,
        m_x_reco_b=m_x_reco_b,
        m_x_from_z_a=m_x_from_z_a,
        m_x_from_z_b=m_x_from_z_b,
        pt_x=pt_x,
        pt_x_reco_a=pt_x_reco_a,
        pt_x_reco_b=pt_x_reco_b,
        pt_x_from_z_a=pt_x_from_z_a,
        pt_x_from_z_b=pt_x_from_z_b,
    )
    print("Saved:", output_dir / "paperstyle_ab_loaded_model_outputs.npz")
    print("Using device:", device)
    print("Device report:", json.dumps(report, sort_keys=True))
    print("Wrote plots:", output_dir)
    plot_lib.log_progress("Finished CMS dilepton A/B plot generation.")


if __name__ == "__main__":
    main()
