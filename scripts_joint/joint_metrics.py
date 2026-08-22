"""Deterministic validation and worst-region selection for ``cms_Joint``."""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import torch

from physics import invariant_mass_np, validate_daughter_masses


def _first(value):
    return value[0] if isinstance(value, (tuple, list)) else value


def wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    """Dependency-free empirical W1, including unequal sample counts."""
    a = np.sort(np.asarray(a, dtype=np.float64).reshape(-1))
    b = np.sort(np.asarray(b, dtype=np.float64).reshape(-1))
    if not len(a) or not len(b):
        return float("nan")
    n = max(len(a), len(b))
    q = (np.arange(n, dtype=np.float64) + 0.5) / n
    aq = np.interp(q, (np.arange(len(a)) + 0.5) / len(a), a)
    bq = np.interp(q, (np.arange(len(b)) + 0.5) / len(b), b)
    return float(np.mean(np.abs(aq - bq)))


def ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(np.asarray(a, dtype=np.float64).reshape(-1))
    b = np.sort(np.asarray(b, dtype=np.float64).reshape(-1))
    if not len(a) or not len(b):
        return float("nan")
    grid = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def pair_pt(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    return np.hypot(values[:, 0] + values[:, 4], values[:, 1] + values[:, 5])


def _width_relative_error(real: np.ndarray, fake: np.ndarray) -> float:
    real_width = float(np.std(real))
    fake_width = float(np.std(fake))
    return abs(fake_width - real_width) / max(real_width, 1.0e-12)


def _equal_subset(
    a: np.ndarray,
    b: np.ndarray,
    *,
    max_events: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    if max_events is not None:
        n = min(n, int(max_events))
    if n < 2:
        raise ValueError("Joint validation needs at least two events per distribution")
    rng = np.random.default_rng(seed)
    return (
        np.asarray(a[rng.choice(len(a), n, replace=False)], dtype=np.float32),
        np.asarray(b[rng.choice(len(b), n, replace=False)], dtype=np.float32),
    )


@torch.inference_mode()
def transform_batches(
    function: Callable[[torch.Tensor], torch.Tensor],
    values: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    outputs = []
    for start in range(0, len(values), int(batch_size)):
        tensor = torch.as_tensor(
            np.ascontiguousarray(values[start : start + int(batch_size)]),
            dtype=torch.float32,
            device=device,
        )
        outputs.append(_first(function(tensor)).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _distribution_metrics(
    prefix: str,
    real: np.ndarray,
    fake: np.ndarray,
    *,
    daughter_masses,
    stable_mass: bool,
) -> dict[str, float]:
    real_mass = invariant_mass_np(
        real, daughter_masses=daughter_masses, stable=stable_mass
    )
    fake_mass = invariant_mass_np(
        fake, daughter_masses=daughter_masses, stable=stable_mass
    )
    return {
        f"{prefix}_mass_w1_gev": wasserstein_1d(real_mass, fake_mass),
        f"{prefix}_mass_ks": ks_distance(real_mass, fake_mass),
        f"{prefix}_mass_width_rel_error": _width_relative_error(real_mass, fake_mass),
        f"{prefix}_mass_mean_shift_gev": float(fake_mass.mean() - real_mass.mean()),
        f"{prefix}_pair_pt_ks": ks_distance(pair_pt(real), pair_pt(fake)),
    }


@torch.inference_mode()
def evaluate_region(
    model,
    arrays: dict[str, np.ndarray],
    loss_factory,
    *,
    daughter_masses,
    device: torch.device,
    batch_size: int,
    max_events: int | None,
    decoder_draws: int,
    seed: int,
) -> dict[str, float]:
    """Evaluate direct, latent, and cycle closure on one fixed region subset."""
    masses = validate_daughter_masses(daughter_masses)
    x, z = _equal_subset(
        arrays["x_val"], arrays["z_val"], max_events=max_events, seed=seed
    )
    model.eval()

    direct_draws = []
    for draw in range(max(1, int(decoder_draws))):
        torch.manual_seed(seed + 100 * draw)
        direct_draws.append(
            transform_batches(model.decode, z, device=device, batch_size=batch_size)
        )
    x_direct = np.concatenate(direct_draws, axis=0)
    x_direct_truth = np.tile(x, (len(direct_draws), 1))

    torch.manual_seed(seed + 1)
    z_encoded = transform_batches(model.encode, x, device=device, batch_size=batch_size)
    torch.manual_seed(seed + 2)
    x_cycle = transform_batches(
        model.decode, z_encoded, device=device, batch_size=batch_size
    )

    metrics = {}
    metrics.update(
        _distribution_metrics(
            "direct",
            x_direct_truth,
            x_direct,
            daughter_masses=masses,
            stable_mass=True,
        )
    )
    metrics.update(
        _distribution_metrics(
            "latent", z, z_encoded, daughter_masses=masses, stable_mass=False
        )
    )
    metrics.update(
        _distribution_metrics(
            "cycle", x, x_cycle, daughter_masses=masses, stable_mass=True
        )
    )

    # A compact fixed-weight validation loss complements the interpretable
    # peak metrics. It is reported, but is not used by default for selection.
    n_loss = min(len(x), int(batch_size))
    x_t = torch.as_tensor(x[:n_loss], dtype=torch.float32, device=device)
    z_t = torch.as_tensor(z[:n_loss], dtype=torch.float32, device=device)
    torch.manual_seed(seed + 3)
    z_tilde = _first(model.encode(x_t))
    x_reco = _first(model.decode(z_tilde))
    x_from_z = _first(model.decode(z_t))
    x_loss = loss_factory.x_reco_loss(x_t, x_reco)
    z_loss = loss_factory.z_prior_loss(z_t, z_tilde)
    direct_loss = loss_factory.x_sim_loss(x_t, x_from_z)
    base = loss_factory.validation_score(
        {
            "x_loss": x_loss,
            "z_loss": z_loss,
            "alt_x_loss": direct_loss,
            "cycle_loss": x_loss,
        }
    )
    metrics["base_loss"] = float(base.detach().cpu())
    return metrics


def score_joint_metrics(
    region_metrics: dict[str, dict[str, float]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Normalize targets, reduce within each region, then select the worst region."""
    selection = config.get("checkpoint_selection", {})
    hard_gates = bool(selection.get("hard_gates", True))
    base_weight = float(selection.get("base_loss_weight", 0.0))
    gate_fail_penalty = float(selection.get("gate_fail_penalty", 1.0e6))
    region_reduction = str(selection.get("region_reduction", "max")).lower()
    if region_reduction not in {"max", "mean", "rms"}:
        raise ValueError(
            "checkpoint_selection.region_reduction must be one of: max, mean, rms"
        )
    region_reports: dict[str, Any] = {}
    all_passed = True

    for name in config["region_order"]:
        metrics = region_metrics[name]
        region_selection = config["regions"][name].get("selection", {})
        targets = region_selection.get("targets", {})
        if not targets:
            raise ValueError(f"Region {name!r} has no checkpoint selection targets")
        normalized = {}
        for key, target in targets.items():
            target = float(target)
            value = float(metrics[key])
            if not math.isfinite(value) or target <= 0.0:
                normalized[key] = float("inf")
            else:
                normalized[key] = abs(value) / target
        gates = {}
        for key, threshold in region_selection.get("gates", {}).items():
            value = float(metrics[key])
            gates[key] = bool(math.isfinite(value) and abs(value) <= float(threshold))
        passed = all(gates.values()) if gates else True
        all_passed = all_passed and passed
        worst_metric = max(normalized, key=normalized.get)
        normalized_values = list(normalized.values())
        if region_reduction == "mean":
            reduced_score = sum(normalized_values) / len(normalized_values)
        elif region_reduction == "rms":
            reduced_score = math.sqrt(
                sum(value * value for value in normalized_values)
                / len(normalized_values)
            )
        else:
            reduced_score = float(normalized[worst_metric])
        score = float(reduced_score) + base_weight * float(
            metrics.get("base_loss", 0.0)
        )
        region_reports[name] = {
            "score": score,
            "reduction": region_reduction,
            "worst_metric": worst_metric,
            "normalized_metrics": normalized,
            "gates": gates,
            "gate_passed": passed,
        }

    worst_region = max(region_reports, key=lambda key: region_reports[key]["score"])
    raw_score = float(region_reports[worst_region]["score"])
    selection_score = raw_score
    if hard_gates and not all_passed:
        selection_score += gate_fail_penalty
    return {
        "selection_score": selection_score,
        "raw_worst_region_score": raw_score,
        "worst_region": worst_region,
        "all_gates_passed": all_passed,
        "regions": region_reports,
    }
