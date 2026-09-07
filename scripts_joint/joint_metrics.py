"""Deterministic validation and worst-region selection for ``cms_Joint``.

Every ``{direction}_*`` metric is reported alongside two references computed by
``identity_reference``: ``{direction}_identity_*`` (the score of doing nothing
at all) and ``{direction}_floor_*`` (the score of two independent draws of the
same distribution). The derived ``{direction}_*_vs_identity`` gauge is 1.0 for
a no-op map, 0.0 at the finite-sample floor, and is what a gate should be
placed on. See ``identity_reference`` for why: on the 2026-09-04 smeared-prior
A/B arm the identity map passed every strict latent target, so the raw metrics
alone cannot certify that any unfolding took place.

The cycle direction gets no identity reference on purpose: the identity cycle
is x -> x, whose metrics are identically zero, so cycle gates have no power
against a no-op by construction and a gauge there would divide by zero.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import torch

from physics import invariant_mass_np, validate_daughter_masses

# Re-exported so callers keep importing the 1-D statistics from this module.
from identity_reference import (  # noqa: F401
    equal_subset as _equal_subset,
    ks_distance,
    pair_pt,
    reference_block,
    wasserstein_1d,
    width_relative_error as _width_relative_error,
)


def _first(value):
    return value[0] if isinstance(value, (tuple, list)) else value


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

    # Identity and finite-sample references. These make every gated number
    # readable: a metric is only evidence of unfolding if it beats what the
    # input itself scores. The mass convention is passed in rather than
    # re-derived, so the references can never drift from the metrics above.
    #
    # latent: the map is x -> z. Doing nothing means emitting x unchanged.
    # direct: the map is z -> x. Doing nothing means emitting z unchanged,
    #         tiled to match the decoder-draw count so the floor is estimated
    #         at the sample size the model was actually scored at.
    metrics.update(
        reference_block(
            "latent",
            real_values=z,
            identity_values=x,
            reference_pool=arrays["z_val"],
            model_metrics=metrics,
            mass_fn=lambda values: invariant_mass_np(
                values, daughter_masses=masses, stable=False
            ),
            seed=seed + 500,
        )
    )
    metrics.update(
        reference_block(
            "direct",
            real_values=x_direct_truth,
            identity_values=np.tile(z, (len(direct_draws), 1)),
            reference_pool=arrays["x_val"],
            model_metrics=metrics,
            mass_fn=lambda values: invariant_mass_np(
                values, daughter_masses=masses, stable=True
            ),
            seed=seed + 600,
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
