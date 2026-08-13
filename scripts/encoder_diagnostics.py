#!/usr/bin/env python
"""Deterministic encoder-alignment diagnostics for the J/psi dimuon pipeline.

This module evaluates the x -> z encoder against the MG5 latent prior with a
fixed sample and seed, and writes the same KS/SW trajectory used by the
Stage-1/2 encoder-alignment diagnostic.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from cms_data import array_stats
from cms_training import first_tensor
from metrics import invariant_mass

try:
    from scipy.stats import ks_2samp

    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


COMPONENT_LABELS = [
    "mu_minus_px",
    "mu_minus_py",
    "mu_minus_pz",
    "mu_minus_E",
    "mu_plus_px",
    "mu_plus_py",
    "mu_plus_pz",
    "mu_plus_E",
]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _snapshot_rng() -> dict[str, Any]:
    """Snapshot Python/NumPy/torch RNG state (CUDA and MPS included)."""
    state: dict[str, Any] = {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = [value.clone() for value in torch.cuda.get_rng_state_all()]
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        try:
            state["mps"] = torch.mps.get_rng_state()
        except Exception:
            pass
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["random"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state:
        try:
            torch.mps.set_rng_state(state["mps"])
        except Exception:
            pass


def finite(a: np.ndarray) -> np.ndarray:
    values = np.asarray(a).reshape(-1)
    return values[np.isfinite(values)]


def maybe_ks(a: np.ndarray, b: np.ndarray) -> float | None:
    if not HAS_SCIPY:
        return None
    a = finite(a)
    b = finite(b)
    if len(a) == 0 or len(b) == 0:
        return None
    return float(ks_2samp(a, b).statistic)


def pair_pt(pairs: np.ndarray) -> np.ndarray:
    px = pairs[:, 0] + pairs[:, 4]
    py = pairs[:, 1] + pairs[:, 5]
    return np.sqrt(px**2 + py**2)


def transform_in_batches(
    transform: Callable[[torch.Tensor], Any],
    values: np.ndarray,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    set_seed(seed)
    outputs = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.as_tensor(
                values[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            outputs.append(first_tensor(transform(batch)).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def encode_x(model: torch.nn.Module, x: np.ndarray, device: torch.device, batch_size: int, seed: int) -> np.ndarray:
    model.eval()
    return transform_in_batches(model.encode, x, device, batch_size, seed)


def decode_z(model: torch.nn.Module, z: np.ndarray, device: torch.device, batch_size: int, seed: int) -> np.ndarray:
    model.eval()
    return transform_in_batches(model.decode, z, device, batch_size, seed)


def latent_sw_loss(
    loss_factory: Any,
    z_true: np.ndarray,
    z_encoded: np.ndarray,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    """Training-style z-prior objective on the fixed diagnostic sample."""
    set_seed(seed)
    z_true_t = torch.as_tensor(z_true, dtype=torch.float32, device=device)
    z_enc_t = torch.as_tensor(z_encoded, dtype=torch.float32, device=device)
    with torch.no_grad():
        loss = loss_factory.z_prior_loss(z_true_t, z_enc_t)
        components = dict(getattr(loss_factory, "latest_components", {}))
    out = {
        "z_prior_loss_raw": float(loss.detach().cpu()),
        "z_sw_raw": float(components.get("z_raw_swd", float("nan"))),
        "z_marginal_w1_raw": float(components.get("z_marginal_w1", float("nan"))),
    }
    return out


def evaluate_encoder(
    model: torch.nn.Module,
    x_eval: np.ndarray,
    z_eval: np.ndarray,
    device: torch.device,
    *,
    seed: int,
    batch_size: int = 8192,
    loss_factory: Any | None = None,
) -> dict[str, Any]:
    """Evaluate encoder/cycle/decoder closure on one fixed sample."""
    z_encoded = encode_x(model, x_eval, device, batch_size, seed)
    x_reco = decode_z(model, z_encoded, device, batch_size, seed + 1)
    x_from_z = decode_z(model, z_eval, device, batch_size, seed + 2)

    metrics: dict[str, Any] = {
        "n_eval": int(len(x_eval)),
        "z_ks": {},
        "mean_z_ks": None,
        "max_z_ks": None,
        "z_mass_ks": maybe_ks(
            invariant_mass(z_eval, stable=False),
            invariant_mass(z_encoded, stable=False),
        ),
        "z_pt_ks": maybe_ks(pair_pt(z_eval), pair_pt(z_encoded)),
        "cycle_mass_ks": maybe_ks(invariant_mass(x_eval), invariant_mass(x_reco)),
        "cycle_pt_ks": maybe_ks(pair_pt(x_eval), pair_pt(x_reco)),
        "z_to_x_mass_ks": maybe_ks(invariant_mass(x_eval), invariant_mass(x_from_z)),
        "z_to_x_pt_ks": maybe_ks(pair_pt(x_eval), pair_pt(x_from_z)),
    }
    z_ks_values = []
    for j in range(8):
        value = maybe_ks(z_eval[:, j], z_encoded[:, j])
        metrics["z_ks"][f"z_ks_{j:02d}"] = value
        if value is not None:
            z_ks_values.append(value)
    if z_ks_values:
        metrics["mean_z_ks"] = float(np.mean(z_ks_values))
        metrics["max_z_ks"] = float(np.max(z_ks_values))

    # Moment checks in standardized units, useful for diagnosing scale domination.
    z_mean = np.mean(z_eval, axis=0)
    z_std = np.std(z_eval, axis=0)
    enc_mean = np.mean(z_encoded, axis=0)
    enc_std = np.std(z_encoded, axis=0)
    std = np.where(z_std > 0.0, z_std, 1.0)
    metrics["z_mean_abs_diff_std"] = float(np.mean(np.abs((enc_mean - z_mean) / std)))
    metrics["z_std_abs_diff_std"] = float(np.mean(np.abs((enc_std - z_std) / std)))

    if loss_factory is not None:
        loss_metrics = latent_sw_loss(loss_factory, z_eval, z_encoded, device, seed + 3)
        metrics.update(loss_metrics)
    return metrics


def write_metric_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow({key: row[key] for key in sorted(row.keys())})


def make_training_callback(
    run_dir: Path,
    x_eval: np.ndarray,
    z_eval: np.ndarray,
    device: torch.device,
    *,
    seed: int,
    batch_size: int = 8192,
    independent_rng: bool = False,
) -> Callable[[torch.nn.Module, int, str], None]:
    """Return a callback suitable for train_all_stages diagnostics_callback."""
    metric_path = run_dir / "encoder_alignment_diagnostic" / "metric_trajectory.csv"

    def callback(model: torch.nn.Module, global_epoch: int, stage: str, loss_factory: Any) -> None:
        def run() -> dict[str, Any]:
            return evaluate_encoder(
                model,
                x_eval,
                z_eval,
                device,
                seed=seed,
                batch_size=batch_size,
                loss_factory=loss_factory,
            )

        if independent_rng:
            rng_state = _snapshot_rng()
            try:
                metrics = run()
            finally:
                _restore_rng(rng_state)
        else:
            metrics = run()
        row = {"global_epoch": int(global_epoch), "stage": stage}
        row.update(metrics)
        row.pop("z_ks", None)
        row.update(metrics["z_ks"])
        write_metric_row(metric_path, row)
        print(
            f"[encoder_diag] epoch {global_epoch:04d} {stage} | "
            f"mean_z_ks={row.get('mean_z_ks'):.4f} z_mass_ks={row.get('z_mass_ks'):.4f} "
            f"cycle_mass_ks={row.get('cycle_mass_ks'):.4f}",
            flush=True,
        )

    return callback


def final_metric_row(
    checkpoint_path: Path,
    model: torch.nn.Module,
    config: dict[str, Any],
    x_test: np.ndarray,
    z_test: np.ndarray,
    device: torch.device,
    *,
    seed: int,
) -> dict[str, Any]:
    """Evaluate a saved checkpoint and return a flat metric row."""
    loss_factory = None
    try:
        from cms_training import build_loss_factory

        loss_config = config.get("loss", {})
        loss_factory = build_loss_factory(
            x_test,
            z_test,
            loss_config,
            daughter_masses=(config.get("model") or {}).get("daughter_masses"),
        )
    except Exception:
        loss_factory = None
    batch_size = int(config.get("loaders", {}).get("eval_batch_size", 8192))
    metrics = evaluate_encoder(
        model,
        x_test,
        z_test,
        device,
        seed=seed,
        batch_size=batch_size,
        loss_factory=loss_factory,
    )
    row = {"checkpoint": str(checkpoint_path), "epoch": int(config.get("epoch", -1))}
    row.update(metrics)
    row.pop("z_ks", None)
    row.update(metrics["z_ks"])
    return row
