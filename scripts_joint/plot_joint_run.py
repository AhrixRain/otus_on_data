#!/usr/bin/env python
"""Complete per-component diagnostics for a trained cms_Joint checkpoint."""


from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages


REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (
    REPO_ROOT,
    REPO_ROOT / "scripts",
    REPO_ROOT / "scripts_sota",
    REPO_ROOT / "scripts_joint",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from cms_data import load_config  # noqa: E402
from device_utils import select_device  # noqa: E402
from joint_data import load_joint_regions, resolve_joint_config  # noqa: E402
from joint_metrics import ks_distance, transform_batches, wasserstein_1d  # noqa: E402
from joint_model import build_joint_autoencoder  # noqa: E402
from joint_trainer import restore_joint_checkpoint  # noqa: E402
from physics import invariant_mass_np  # noqa: E402


P4_LABELS = (
    r"$p_{x}^{\mu^-}$ [GeV]",
    r"$p_{y}^{\mu^-}$ [GeV]",
    r"$p_{z}^{\mu^-}$ [GeV]",
    r"$E^{\mu^-}$ [GeV]",
    r"$p_{x}^{\mu^+}$ [GeV]",
    r"$p_{y}^{\mu^+}$ [GeV]",
    r"$p_{z}^{\mu^+}$ [GeV]",
    r"$E^{\mu^+}$ [GeV]",
)

DIRECTION_LABELS = {
    "z_to_x": r"Direct decoder: $z \rightarrow x$",
    "x_to_z": r"Encoder: $x \rightarrow z$",
    "x_to_z_to_x": r"Data cycle: $x \rightarrow z \rightarrow x$",
    "z_to_x_to_z": r"Prior cycle: $z \rightarrow x \rightarrow z$",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs_joint" / "cms_Joint_runA.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "outputs" / "cms_Joint" / "Run_A" / "best_model.pt",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-events", type=int, default=15000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--bins", type=int, default=70)
    parser.add_argument("--seed", type=int, default=20260821)
    return parser.parse_args()


def _equal_subset(a: np.ndarray, b: np.ndarray, nmax: int, seed: int):
    n = min(len(a), len(b), int(nmax))
    rng = np.random.default_rng(seed)
    return (
        np.asarray(a[rng.choice(len(a), n, replace=False)], dtype=np.float32),
        np.asarray(b[rng.choice(len(b), n, replace=False)], dtype=np.float32),
    )


def _physics(values: np.ndarray, masses, stable_mass: bool) -> dict[str, np.ndarray]:
    p1, p2 = values[:, :4], values[:, 4:]
    pt1 = np.hypot(p1[:, 0], p1[:, 1])
    pt2 = np.hypot(p2[:, 0], p2[:, 1])
    eta1 = np.arcsinh(p1[:, 2] / np.maximum(pt1, 1.0e-8))
    eta2 = np.arcsinh(p2[:, 2] / np.maximum(pt2, 1.0e-8))
    phi1 = np.arctan2(p1[:, 1], p1[:, 0])
    phi2 = np.arctan2(p2[:, 1], p2[:, 0])
    dphi = np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2))
    pair = p1 + p2
    pair_pt = np.hypot(pair[:, 0], pair[:, 1])
    rapidity = 0.5 * np.log(
        np.maximum(pair[:, 3] + pair[:, 2], 1.0e-8)
        / np.maximum(pair[:, 3] - pair[:, 2], 1.0e-8)
    )
    return {
        r"$m_{\mu\mu}$ [GeV]": invariant_mass_np(
            values, daughter_masses=masses, stable=stable_mass
        ),
        r"$p_T^{\mu\mu}$ [GeV]": pair_pt,
        r"$p_T^{\mu^-}$ [GeV]": pt1,
        r"$p_T^{\mu^+}$ [GeV]": pt2,
        r"$\eta^{\mu^-}$": eta1,
        r"$\eta^{\mu^+}$": eta2,
        r"$\phi^{\mu^-}$": phi1,
        r"$\phi^{\mu^+}$": phi2,
        r"$\Delta\phi$": dphi,
        r"$\Delta\eta$": eta1 - eta2,
        r"$y_{\mu\mu}$": rapidity,
    }


def _range(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    pooled = np.concatenate([a[np.isfinite(a)], b[np.isfinite(b)]])
    low, high = np.quantile(pooled, [0.001, 0.999])
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        low, high = float(np.min(pooled)), float(np.max(pooled))
    padding = 0.03 * max(high - low, 1.0e-8)
    return float(low - padding), float(high + padding)


def _hist_panel(ax, reference, prediction, xlabel, bins, reference_label, prediction_label):
    bounds = _range(reference, prediction)
    edges = np.linspace(bounds[0], bounds[1], int(bins) + 1)
    ax.hist(
        reference,
        bins=edges,
        density=True,
        histtype="step",
        linewidth=1.7,
        color="#2b6cb0",
        label=reference_label,
    )
    ax.hist(
        prediction,
        bins=edges,
        density=True,
        histtype="step",
        linewidth=1.7,
        color="#dd6b20",
        label=prediction_label,
    )
    ax.set_xlim(bounds)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalized density")
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.text(
        0.98,
        0.96,
        f"KS={ks_distance(reference, prediction):.3f}\nW1={wasserstein_1d(reference, prediction):.3g}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )


def _p4_figure(region, direction, reference, prediction, labels, bins):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    for index, ax in enumerate(axes.flat):
        _hist_panel(
            ax,
            reference[:, index],
            prediction[:, index],
            P4_LABELS[index],
            bins,
            labels[0],
            labels[1],
        )
    axes.flat[0].legend(loc="upper left", frameon=False, fontsize=8)
    fig.suptitle(
        f"{region.upper()} — {DIRECTION_LABELS[direction]} — all p4 components",
        y=0.997,
        ha="center",
    )
    return fig


def _physics_figure(region, direction, reference, prediction, masses, stable_mass, labels, bins):
    real = _physics(reference, masses, stable_mass)
    fake = _physics(prediction, masses, stable_mass)
    fig, axes = plt.subplots(3, 4, figsize=(18, 11), constrained_layout=True)
    for ax, name in zip(axes.flat, real):
        _hist_panel(ax, real[name], fake[name], name, bins, labels[0], labels[1])
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(
        handles,
        legend_labels,
        loc="center",
        frameon=False,
    )
    fig.suptitle(
        f"{region.upper()} — {DIRECTION_LABELS[direction]} — physics observables",
        y=0.997,
        ha="center",
    )
    return fig


def _history_figure(history: list[dict], run_label: str):
    rows = [row for row in history if "joint_selection" in row]
    epochs = np.asarray([row["global_epoch"] for row in rows])
    raw = np.asarray([row["joint_selection"]["raw_worst_region_score"] for row in rows])
    jpsi = np.asarray([row["joint_selection"]["regions"]["jpsi"]["score"] for row in rows])
    z = np.asarray([row["joint_selection"]["regions"]["z"]["score"] for row in rows])
    core = np.asarray([row["core_noise_multiplier"] for row in rows])
    tail = np.asarray([row["tail_noise_multiplier"] for row in rows])
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    axes[0].plot(epochs, raw, label="Worst region", color="#222222", linewidth=2)
    axes[0].plot(epochs, jpsi, label="J/psi", color="#2b6cb0")
    axes[0].plot(epochs, z, label="Z", color="#dd6b20")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Normalized selection score")
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False, ncol=3)
    axes[1].plot(epochs, core, label="Gaussian core", color="#2f855a")
    axes[1].plot(epochs, tail, label="Student-t tail", color="#805ad5")
    axes[1].set_xlabel("Global epoch")
    axes[1].set_ylabel("Noise multiplier")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False, ncol=2)
    fig.suptitle(f"{run_label} checkpoint score and stochasticity schedule")
    return fig


def main() -> int:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else checkpoint_path.parent / "plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    config = resolve_joint_config(load_config(args.config))
    arrays, _, _, _ = load_joint_regions(config, num_samples=None, use_cache=True)
    device = select_device(args.device)
    model = build_joint_autoencoder(
        config["model"],
        arrays,
        float(config.get("muon_mass_gev", 0.1056583755)),
        config["model"].get("daughter_masses"),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    restore_joint_checkpoint(model, checkpoint)
    model.eval()
    masses = config["model"].get("daughter_masses")
    manifest = {
        "checkpoint": str(checkpoint_path),
        "stage": (checkpoint.get("stage") or {}).get("name"),
        "stage_epoch": checkpoint.get("stage_epoch"),
        "noise_multipliers": checkpoint.get("noise_multipliers"),
        "regions": {},
        "files": [],
    }

    for offset, region in enumerate(config["region_order"]):
        x, z = _equal_subset(
            arrays[region]["x_test"],
            arrays[region]["z_test"],
            args.max_events,
            args.seed + 1000 * offset,
        )
        torch.manual_seed(args.seed + 1000 * offset)
        x_from_z = transform_batches(
            model.decode, z, device=device, batch_size=args.batch_size
        )
        torch.manual_seed(args.seed + 1000 * offset + 1)
        z_from_x = transform_batches(
            model.encode, x, device=device, batch_size=args.batch_size
        )
        torch.manual_seed(args.seed + 1000 * offset + 2)
        x_cycle = transform_batches(
            model.decode, z_from_x, device=device, batch_size=args.batch_size
        )
        torch.manual_seed(args.seed + 1000 * offset + 3)
        z_cycle = transform_batches(
            model.encode, x_from_z, device=device, batch_size=args.batch_size
        )
        comparisons = {
            "z_to_x": (x, x_from_z, ("CMS data", "Decoded prior"), True),
            "x_to_z": (z, z_from_x, ("Theory prior", "Encoded data"), False),
            "x_to_z_to_x": (x, x_cycle, ("CMS data", "Data cycle"), True),
            "z_to_x_to_z": (z, z_cycle, ("Theory prior", "Prior cycle"), False),
        }
        pdf_path = output_dir / f"{region}_all_elements.pdf"
        with PdfPages(pdf_path) as pdf:
            for direction, (reference, prediction, labels, stable_mass) in comparisons.items():
                p4_fig = _p4_figure(
                    region, direction, reference, prediction, labels, args.bins
                )
                p4_path = output_dir / f"{region}_{direction}_p4.png"
                p4_fig.savefig(p4_path, dpi=150, bbox_inches="tight")
                pdf.savefig(p4_fig)
                plt.close(p4_fig)
                physics_fig = _physics_figure(
                    region,
                    direction,
                    reference,
                    prediction,
                    masses,
                    stable_mass,
                    labels,
                    args.bins,
                )
                physics_path = output_dir / f"{region}_{direction}_physics.png"
                physics_fig.savefig(physics_path, dpi=150, bbox_inches="tight")
                pdf.savefig(physics_fig)
                plt.close(physics_fig)
                manifest["files"].extend([str(p4_path), str(physics_path)])
        manifest["files"].append(str(pdf_path))
        manifest["regions"][region] = {"events_per_distribution": int(len(x))}

    history_path = checkpoint_path.parent / "history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        fig = _history_figure(history, str(config.get("run_label", "cms_Joint")))
        path = output_dir / "training_selection_history.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        manifest["files"].append(str(path))
    manifest_path = output_dir / "plot_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(manifest['files'])} plot artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
