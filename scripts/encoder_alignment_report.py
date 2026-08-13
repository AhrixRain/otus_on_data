#!/usr/bin/env python
"""Aggregate encoder-alignment diagnostics into a machine-readable report.

Example:
    conda run -n cms python scripts/encoder_alignment_report.py \
        --runs control=outputs/cms_JpsiDoubleMuons/encoder_diag_control \
        --runs candidateA=outputs/cms_JpsiDoubleMuons/encoder_diag_candidateA \
        --runs candidateB=outputs/cms_JpsiDoubleMuons/encoder_diag_candidateB \
        --output-dir outputs/cms_JpsiDoubleMuons/encoder_alignment_diagnostic
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from physics import invariant_mass_np


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def parse_run_args(values: list[str]) -> list[tuple[str, Path]]:
    runs = []
    for value in values:
        if "=" in value:
            label, path = value.split("=", 1)
        else:
            label = Path(value).name
            path = value
        runs.append((label.strip(), Path(path).expanduser().resolve()))
    return runs


def plot_trajectory(
    runs: list[tuple[str, Path]],
    output_path: Path,
    metric: str,
    ylabel: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, run_dir in runs:
        rows = read_csv(run_dir / "encoder_alignment_diagnostic" / "metric_trajectory.csv")
        epochs = [num(row["global_epoch"]) for row in rows]
        values = [num(row.get(metric)) for row in rows]
        ax.plot(epochs, values, marker="o", markersize=3, label=label)
    ax.set_xlabel("global epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_metric_trajectory(runs: list[tuple[str, Path]], out_dir: Path) -> None:
    metrics = [
        ("mean_z_ks", "mean 8D latent KS"),
        ("z_mass_ks", "latent invariant-mass KS"),
        ("z_pt_ks", "latent dilepton pT KS"),
        ("cycle_mass_ks", "cycle invariant-mass KS"),
        ("cycle_pt_ks", "cycle dilepton pT KS"),
        ("z_to_x_mass_ks", "z->x invariant-mass KS"),
        ("z_to_x_pt_ks", "z->x dilepton pT KS"),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(13, 16))
    for ax, (metric, label) in zip(axes.ravel(), metrics):
        for run_label, run_dir in runs:
            rows = read_csv(run_dir / "encoder_alignment_diagnostic" / "metric_trajectory.csv")
            epochs = [num(row["global_epoch"]) for row in rows]
            values = [num(row.get(metric)) for row in rows]
            ax.plot(epochs, values, marker="o", markersize=2, label=run_label)
        ax.set_title(label)
        ax.set_xlabel("global epoch")
        ax.set_ylabel("KS")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes.ravel()[-1].axis("off")
    fig.suptitle("Encoder/cycle metric trajectory (Stage 1 + Stage 2)", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(out_dir / "plots" / "metric_trajectory.png", dpi=160)
    plt.close(fig)


def plot_gradient_norms(runs: list[tuple[str, Path]], out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    for label, run_dir in runs:
        rows = read_csv(run_dir / "encoder_alignment_diagnostic" / "gradient_trajectory.csv")
        epochs = [num(row["epoch"]) for row in rows]
        latent = [num(row.get("grad_norm_encoder_latent")) for row in rows]
        reco = [num(row.get("grad_norm_encoder_reco")) for row in rows]
        axes[0].plot(epochs, np.log10(np.maximum(1e-6, latent)), marker="o", markersize=3, label=f"{label} latent")
        axes[0].plot(epochs, np.log10(np.maximum(1e-6, reco)), marker="s", markersize=3, label=f"{label} reco")
        ratios = [
            (latent[i] / reco[i]) if (latent[i] and reco[i]) else None
            for i in range(len(epochs))
        ]
        axes[1].plot(
            [e for e, r in zip(epochs, ratios) if r is not None],
            [np.log10(r) for r in ratios if r is not None],
            marker="o",
            markersize=3,
            label=f"{label} latent/reco",
        )
    axes[0].set_ylabel("log10 encoder grad norm")
    axes[0].set_title("Encoder gradient norm by loss family")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("log10 latent/reco ratio")
    axes[1].set_xlabel("global epoch")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "gradient_norms.png", dpi=160)
    plt.close(fig)


def plot_gradient_cosine(runs: list[tuple[str, Path]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, run_dir in runs:
        rows = read_csv(run_dir / "encoder_alignment_diagnostic" / "gradient_trajectory.csv")
        epochs = [num(row["epoch"]) for row in rows]
        cosines = [num(row.get("grad_cosine_latent_reco")) for row in rows]
        ax.plot(epochs, cosines, marker="o", markersize=3, label=label)
    ax.axhline(0.0, color="gray", linewidth=1)
    ax.set_xlabel("global epoch")
    ax.set_ylabel(r"$\cos(\nabla L_{\mathrm{latent}}, \nabla L_{\mathrm{reco}})$")
    ax.set_title("Encoder gradient cosine between latent and reconstruction")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "gradient_cosine.png", dpi=160)
    plt.close(fig)


def plot_loss_trajectory(runs: list[tuple[str, Path]], out_dir: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    fields = [
        ("train_z_loss", "latent total (z prior)"),
        ("train_x_loss", "reconstruction (x reco)"),
        ("train_alt_x_loss", "simulator x loss (z -> x)"),
    ]
    for ax, (field, label) in zip(axes, fields):
        for run_label, run_dir in runs:
            rows = read_csv(run_dir / "encoder_alignment_diagnostic" / "loss_trajectory.csv")
            epochs = [num(row["epoch"]) for row in rows]
            values = [num(row.get(field)) for row in rows]
            ax.plot(epochs, values, marker="o", markersize=2, label=run_label)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("global epoch")
    fig.suptitle("Training loss trajectory")
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "loss_trajectory.png", dpi=160)
    plt.close(fig)


def invariant_mass(pairs: np.ndarray) -> np.ndarray:
    """Cancellation-free pair invariant mass (float64, massless convention)."""
    return invariant_mass_np(pairs, daughter_masses=None, stable=True)


def pair_pt(pairs: np.ndarray) -> np.ndarray:
    return np.sqrt((pairs[:, 0] + pairs[:, 4]) ** 2 + (pairs[:, 1] + pairs[:, 5]) ** 2)


def load_npz(output_dir: Path, label: str) -> dict[str, np.ndarray]:
    path = output_dir / label / "plots" / "paperstyle_loaded_model_outputs.npz"
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def plot_hist_comparison(
    arrays_by_run: dict[str, dict[str, np.ndarray]],
    output_path: Path,
    *,
    truth_key: str,
    pred_key: str,
    xlabel: str,
    bins: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, len(arrays_by_run), figsize=(15, 4), sharey=True)
    if len(arrays_by_run) == 1:
        axes = [axes]
    for ax, (label, data) in zip(axes, arrays_by_run.items()):
        truth = np.asarray(data[truth_key]).reshape(-1)
        pred = np.asarray(data[pred_key]).reshape(-1)
        ax.hist(truth, bins=bins, density=True, histtype="step", label="truth", color="black")
        ax.hist(pred, bins=bins, density=True, histtype="step", label=label, color="#0072B2")
        ax.set_title(label)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("density")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def copy_plot(output_dir: Path, label: str, source: str, destination: str) -> None:
    src = output_dir / label / "plots" / source
    dst = output_dir / "plots" / destination
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def write_combined_csv(output_dir: Path, run_dir_map: list[tuple[str, Path]], filename: str) -> None:
    rows: list[dict[str, Any]] = []
    for label, run_dir in run_dir_map:
        for row in read_csv(run_dir / "encoder_alignment_diagnostic" / filename):
            out = {"run": label}
            out.update(row)
            rows.append(out)
    path = output_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()) if rows else ["run"])
        writer.writeheader()
        writer.writerows(rows)


def final_metrics_from_stage_summary(output_dir: Path, label: str) -> dict[str, Any]:
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        return {}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for item in summary.get("checkpoints", []):
        if item["label"] == label:
            return dict(item["metrics"])
    return {}


def write_summary_md(
    output_dir: Path,
    runs: list[tuple[str, Path]],
    final: dict[str, dict[str, Any]],
    settings: dict[str, Any],
) -> str:
    lines = [
        "# Encoder alignment diagnostic",
        "",
        f"- Config: `{settings['config']}`",
        f"- Split: `{settings['split']}`, num-samples: {settings['num_samples']}",
        f"- Seed: {settings['seed']}",
        "- Runs: " + ", ".join(label for label, _ in runs),
        "",
        "## Final epoch-100 metrics (same fixed test sample and seed)",
        "",
        "| run | cycle mass KS | cycle pT KS | z mass KS | z pT KS | mean 8D KS | max 8D KS | z->x mass KS | z->x pT KS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    metric_keys = [
        "cycle_mass_ks",
        "cycle_pt_ks",
        "zspace_mass_ks",
        "z_pt_ks",
        "z_component_ks_mean",
        "max_z_ks",
        "z_to_x_mass_ks",
        "z_to_x_pt_ks",
    ]
    for label, _ in runs:
        m = final.get(label, {})
        values = [m.get(key) for key in metric_keys]
        lines.append(
            "| " + label + " | "
            + " | ".join("-" if v is None else f"{v:.4f}" for v in values)
            + " |"
        )
    lines.append("")
    lines.append(
        "Note: `z_pt_ks` and `max_z_ks` come from the deterministic training "
        "trajectory (metric_trajectory.csv); the other metrics come from the "
        "shared plot/eval pipeline."
    )
    lines.append("")
    lines.append("## Gradient evidence at epoch 100")
    lines.append("")
    lines.append("| run | latent grad norm | reco grad norm | latent/reco | cosine |")
    lines.append("|---|---:|---:|---:|---:|")
    for label, run_dir in runs:
        rows = read_csv(run_dir / "encoder_alignment_diagnostic" / "gradient_trajectory.csv")
        if not rows:
            continue
        row = rows[-1]
        lat = num(row.get("grad_norm_encoder_latent"))
        rec = num(row.get("grad_norm_encoder_reco"))
        cos = num(row.get("grad_cosine_latent_reco"))
        ratio = (lat / rec) if (lat is not None and rec) else None
        lines.append(
            f"| {label} | {lat:.4g} | {rec:.4g} | "
            f"{'-' if ratio is None else f'{ratio:.4g}'} | "
            f"{'-' if cos is None else f'{cos:.4f}'} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", action="append", required=True, help="label=run_dir")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/cms_JpsiDoubleMuons_encoder_control.yaml"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    runs = parse_run_args(args.runs)
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for label, _ in runs:
        copy_plot(out_dir, label, "all8_zspace_components_density.png", f"latent_components_{label}.png")
        copy_plot(out_dir, label, "paperstyle_zspace_mass_density_ratio.png", f"latent_mass_{label}.png")
        copy_plot(out_dir, label, "paperstyle_zspace_pos_pz_ratio.png", f"latent_pz_{label}.png")
        copy_plot(out_dir, label, "paperstyle_zspace_pos_E_ratio.png", f"latent_E_{label}.png")
        copy_plot(out_dir, label, "paperstyle_xspace_mass_density_ratio.png", f"cycle_mass_{label}.png")
        copy_plot(out_dir, label, "paperstyle_xspace_pt_density_ratio.png", f"cycle_pt_{label}.png")
        copy_plot(out_dir, label, "all8_xspace_components_density.png", f"cycle_components_{label}.png")

    # Comparison plots from the shared npz outputs.
    arrays_by_run = {label: load_npz(out_dir, label) for label, _ in runs}
    mass_bins = np.arange(2.6, 3.51, 0.01)
    pt_bins = np.linspace(0.0, 100.0, 101)
    plot_hist_comparison(
        arrays_by_run,
        plot_dir / "latent_mass_comparison.png",
        truth_key="z_plot",
        pred_key="z_encoded",
        xlabel="m(mumu) [GeV]",
        bins=mass_bins,
    )
    plot_hist_comparison(
        arrays_by_run,
        plot_dir / "latent_pt_comparison.png",
        truth_key="z_plot",
        pred_key="z_encoded",
        xlabel="pT(mumu) [GeV]",
        bins=pt_bins,
    )
    plot_hist_comparison(
        arrays_by_run,
        plot_dir / "cycle_mass_comparison.png",
        truth_key="x_plot",
        pred_key="x_reco",
        xlabel="m(mumu) [GeV]",
        bins=mass_bins,
    )
    plot_hist_comparison(
        arrays_by_run,
        plot_dir / "cycle_pt_comparison.png",
        truth_key="x_plot",
        pred_key="x_reco",
        xlabel="pT(mumu) [GeV]",
        bins=pt_bins,
    )
    plot_hist_comparison(
        arrays_by_run,
        plot_dir / "decoder_mass_comparison.png",
        truth_key="x_plot",
        pred_key="x_from_z",
        xlabel="m(mumu) [GeV]",
        bins=mass_bins,
    )
    plot_hist_comparison(
        arrays_by_run,
        plot_dir / "decoder_pt_comparison.png",
        truth_key="x_plot",
        pred_key="x_from_z",
        xlabel="pT(mumu) [GeV]",
        bins=pt_bins,
    )

    plot_metric_trajectory(runs, out_dir)
    plot_gradient_norms(runs, out_dir)
    plot_gradient_cosine(runs, out_dir)
    plot_loss_trajectory(runs, out_dir)
    write_combined_csv(out_dir, runs, "metric_trajectory.csv")
    write_combined_csv(out_dir, runs, "gradient_trajectory.csv")
    write_combined_csv(out_dir, runs, "loss_trajectory.csv")

    final: dict[str, dict[str, Any]] = {}
    for label, run_dir in runs:
        metrics = final_metrics_from_stage_summary(out_dir, label)
        metric_rows = read_csv(run_dir / "encoder_alignment_diagnostic" / "metric_trajectory.csv")
        if metric_rows:
            last = metric_rows[-1]
            for key in ("z_pt_ks", "max_z_ks", "mean_z_ks", "z_ks_00", "z_ks_01", "z_ks_02",
                        "z_ks_03", "z_ks_04", "z_ks_05", "z_ks_06", "z_ks_07"):
                if key in last:
                    metrics[key] = num(last[key])
        final[label] = metrics

    settings = {
        "config": str(args.config.expanduser().resolve()),
        "split": args.split,
        "num_samples": args.num_samples,
        "seed": args.seed,
    }
    summary = {
        "settings": settings,
        "runs": {
            label: {
                "run_dir": str(run_dir),
                "metrics": final.get(label, {}),
            }
            for label, run_dir in runs
        },
    }
    (out_dir / "encoder_alignment_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md = write_summary_md(out_dir, runs, final, settings)
    (out_dir / "encoder_alignment_summary.md").write_text(md, encoding="utf-8")
    print("Wrote:", out_dir / "encoder_alignment_summary.json")
    print("Wrote:", out_dir / "encoder_alignment_summary.md")
    print("Wrote plots in:", plot_dir)


if __name__ == "__main__":
    main()
