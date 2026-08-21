#!/usr/bin/env python
"""Plot current Run E cylindrical-flow checkpoint without disturbing training.

Usage:

    python scripts_sota/plot_run_e.py \
      --config configs_sota/cms_Jpsi_ptj5_runE_tierA.yaml \
      --checkpoint outputs/cms_Jpsi_sota/runE_full_20260816_213140/best_model.pt \
      --device cpu --threads 4 --interop-threads 4 \
      --output-dir outputs/cms_Jpsi_sota/runE_full_20260816_213140/plots_best_while_training

By default this script uses the CPU so the training process keeps MPS.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "scripts_sota"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cms_data import load_and_split_cached, load_config, resolve_config  # noqa: E402
from physics import daughter_masses_from_config, invariant_mass_np  # noqa: E402
from run_e import build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--interop-threads", type=int, default=4)
    parser.add_argument("--mass-low", type=float, default=2.9)
    parser.add_argument("--mass-high", type=float, default=3.3)
    parser.add_argument("--mass-bin-width", type=float, default=0.005)
    return parser.parse_args()


@torch.inference_mode()
def transform_in_batches(function, values, device, batch_size):
    outputs = []
    for start in range(0, len(values), batch_size):
        batch = torch.as_tensor(values[start : start + batch_size], dtype=torch.float32, device=device)
        outputs.append(function(batch).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def w1(a, b):
    try:
        from scipy.stats import wasserstein_distance
        return float(wasserstein_distance(a, b))
    except Exception:
        q = np.linspace(0, 1, 4097)
        return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def ks(a, b):
    try:
        from scipy.stats import ks_2samp
        return float(ks_2samp(a, b).statistic)
    except Exception:
        x = np.sort(np.concatenate([a, b]))
        ca = np.searchsorted(np.sort(a), x, side="right") / len(a)
        cb = np.searchsorted(np.sort(b), x, side="right") / len(b)
        return float(np.max(np.abs(ca - cb)))


def pair_pt(p4):
    return np.hypot(p4[:, 0] + p4[:, 4], p4[:, 1] + p4[:, 5])


def plot_mass(ax, truth, pred, edges, label):
    ax.hist(truth, bins=edges, density=True, histtype="step", color="black", linewidth=1.8, label="CMS data")
    ax.hist(pred, bins=edges, density=True, histtype="step", color="#0072B2", linewidth=1.8, label=label)
    ax.set_xlabel("m(mumu) [GeV]")
    ax.set_ylabel("normalized density")
    ax.grid(alpha=0.25)
    ax.legend()


def main() -> int:
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.interop_threads)
    device = torch.device("mps" if args.device == "auto" and torch.backends.mps.is_available() else args.device)
    config = resolve_config(load_config(args.config))
    ck = torch.load(args.checkpoint.expanduser().resolve(), map_location="cpu", weights_only=False)
    ck_config = ck.get("config") or config
    masses = daughter_masses_from_config(ck_config)

    arrays, cache_info = load_and_split_cached(
        ck_config, num_samples=args.num_samples, cache_dir=None, use_cache=True
    )
    x_test = arrays["x_test"]
    z_test = arrays["z_test"]

    model = build_model(ck_config, arrays).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    z_decoded = transform_in_batches(model.decode, z_test, device, args.batch_size)
    z_encoded = transform_in_batches(model.encode, x_test, device, args.batch_size)
    x_reco = transform_in_batches(model.decode, z_encoded, device, args.batch_size)

    m_x = invariant_mass_np(x_test, daughter_masses=masses, stable=True)
    m_sim = invariant_mass_np(z_decoded, daughter_masses=masses, stable=True)
    m_z = invariant_mass_np(z_test, daughter_masses=masses, stable=True)
    m_enc = invariant_mass_np(z_encoded, daughter_masses=masses, stable=True)
    m_reco = invariant_mass_np(x_reco, daughter_masses=masses, stable=True)

    out_dir = args.output_dir or (args.checkpoint.parent / "plots_run_e_while_training")
    out_dir.mkdir(parents=True, exist_ok=True)

    edges = np.arange(args.mass_low, args.mass_high + 0.5 * args.mass_bin_width, args.mass_bin_width)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    plot_mass(axes[0], m_x, m_sim, edges, "D(z_test)")
    plot_mass(axes[1], m_x, m_reco, edges, "D(E(x_test))")
    axes[0].set_title("Generator z -> x")
    axes[1].set_title("Cycle x -> z -> x")
    fig.tight_layout()
    fig.savefig(out_dir / "mass_generator_cycle.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    plot_mass(axes[0], m_z, m_enc, edges, "E(x_test)")
    axes[0].set_title("Latent z prior vs encoder")
    pt_edges = np.quantile(np.concatenate([pair_pt(x_test), pair_pt(z_decoded)]), np.linspace(0, 1, 81))
    axes[1].hist(pair_pt(x_test), bins=pt_edges, density=True, histtype="step", color="black", label="CMS data")
    axes[1].hist(pair_pt(z_decoded), bins=pt_edges, density=True, histtype="step", color="#0072B2", label="D(z_test)")
    axes[1].set_xlabel("pair pT [GeV]")
    axes[1].set_ylabel("normalized density")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    axes[1].set_title("Pair pT")
    fig.tight_layout()
    fig.savefig(out_dir / "latent_mass_pair_pt.png", dpi=160)
    plt.close(fig)

    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": ck.get("epoch"),
        "stage": (ck.get("stage") or {}).get("name"),
        "data_cache": cache_info,
        "shapes": {
            "x_test": list(x_test.shape),
            "z_test": list(z_test.shape),
        },
        "simulation": {"w1_gev": w1(m_x, m_sim), "ks": ks(m_x, m_sim),
                       "mean_gev": float(m_sim.mean()), "std_gev": float(m_sim.std())},
        "reconstruction": {"w1_gev": w1(m_x, m_reco), "ks": ks(m_x, m_reco),
                           "mean_gev": float(m_reco.mean()), "std_gev": float(m_reco.std())},
        "latent": {"w1_gev": w1(m_z, m_enc), "ks": ks(m_z, m_enc),
                   "mean_gev": float(m_enc.mean()), "std_gev": float(m_enc.std())},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    np.savez_compressed(out_dir / "arrays.npz", x_test=x_test, z_test=z_test,
                        z_decoded=z_decoded, z_encoded=z_encoded, x_reco=x_reco)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("Wrote plots to", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
