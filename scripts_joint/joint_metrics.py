"""Deterministic validation and worst-region selection for ``cms_Joint``.

Every ``{direction}_*`` metric is reported alongside two references computed by
``identity_reference``: ``{direction}_identity_*`` (the score of doing nothing
at all) and ``{direction}_floor_*`` (the score of two independent draws of the
same distribution). The derived ``{direction}_*_vs_identity`` gauge is 1.0 for
a no-op map, 0.0 at the finite-sample floor. These references are retained as
readable diagnostics.

The cycle direction gets no identity reference on purpose: the identity cycle
is x -> x, whose metrics are identically zero, so a gauge there would divide
by zero.
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


_NOISE_COMPONENTS = ("encoder_core", "encoder_tail", "decoder_core", "decoder_tail")


def resolve_noise_multipliers(spec: Any) -> dict[str, float]:
    """Resolve a noise-multiplier mapping with deterministic defaults.

    Missing keys default to zero, so ``None`` means "score the deterministic
    map". A shared ``core``/``tail`` pair is accepted for symmetry with the
    stage schedule; explicit ``encoder_*``/``decoder_*`` keys override it.
    Validation must never inherit whatever noise level the current training
    stage happens to use: that makes stage scores incomparable and hides
    deterministic-map degeneration (see the Run E stage-3/4 audit).
    """
    if spec is None:
        spec = {}
    if not isinstance(spec, dict):
        raise ValueError("noise_multipliers must be a mapping or None")
    shared = {"core": 0.0, "tail": 0.0}
    for key in shared:
        if key in spec:
            shared[key] = float(spec[key])
    resolved = {
        "encoder_core": shared["core"],
        "encoder_tail": shared["tail"],
        "decoder_core": shared["core"],
        "decoder_tail": shared["tail"],
    }
    for key in _NOISE_COMPONENTS:
        if key in spec:
            resolved[key] = float(spec[key])
    for key, value in resolved.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"noise multiplier {key!r} must be finite and non-negative, got {value!r}"
            )
    return resolved


def current_noise_multipliers(model) -> dict[str, float]:
    """Read the encoder/decoder multipliers the model is currently using."""
    return {
        "encoder_core": float(model.encoder.steps[0].core_noise_multiplier),
        "encoder_tail": float(model.encoder.steps[0].tail_noise_multiplier),
        "decoder_core": float(model.decoder.steps[0].core_noise_multiplier),
        "decoder_tail": float(model.decoder.steps[0].tail_noise_multiplier),
    }


def set_noise_multipliers(model, multipliers: dict[str, float]) -> None:
    """Apply resolved multipliers to both components (all flow steps)."""
    if hasattr(model, "set_component_noise_multipliers"):
        model.set_component_noise_multipliers(**multipliers)
    else:
        # Legacy/SOTA models expose only the shared pair. The joint model,
        # which is the only caller of evaluate_region, always has the split API.
        model.set_noise_multipliers(
            multipliers["encoder_core"], multipliers["encoder_tail"]
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
def _evaluate_region(
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

    # Identity and finite-sample references. These make every reported metric
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
    noise_multipliers: dict[str, float] | None = None,
) -> dict[str, float]:
    """Score one region under an explicit, restored noise policy.

    ``noise_multipliers=None`` scores the fully deterministic map (all four
    multipliers zero). The model's previous multipliers are always restored, so
    this is safe to call mid-training. Keeping validation deterministic makes
    stage scores comparable and exposes deterministic-map degeneration that the
    training noise would otherwise mask.
    """
    previous = current_noise_multipliers(model)
    set_noise_multipliers(model, resolve_noise_multipliers(noise_multipliers))
    try:
        return _evaluate_region(
            model,
            arrays,
            loss_factory,
            daughter_masses=daughter_masses,
            device=device,
            batch_size=batch_size,
            max_events=max_events,
            decoder_draws=decoder_draws,
            seed=seed,
        )
    finally:
        set_noise_multipliers(model, previous)


@torch.inference_mode()
def noise_budget_metrics(
    model,
    z_values: np.ndarray,
    *,
    daughter_masses,
    device: torch.device,
    batch_size: int,
    draws: int,
    max_events: int | None,
    seed: int,
    core: float,
    tail: float,
) -> dict[str, float]:
    """A0.3 fixed-z noise budget for validation (diagnostic only).

    Decodes a fixed subset of truth events ``draws`` times under an explicit
    decoder noise policy and reports the per-event within-z decoded-mass width,
    the exact noise-carried variance fraction and the equivalent
    ``sigma_noise_only``. It never changes the selection score by itself; the
    caller merges the returned keys into the region metrics.

    The model's previous multipliers are always restored.
    """
    draws = int(draws)
    if draws < 2:
        raise ValueError("validation_noise_budget.draws must be at least 2")
    z = np.asarray(z_values)
    if max_events is not None and len(z) > int(max_events):
        rng = np.random.default_rng(int(seed))
        chosen = rng.choice(len(z), size=int(max_events), replace=False)
        chosen.sort()
        z = z[chosen]
    previous = current_noise_multipliers(model)
    policy = dict(previous)
    policy["decoder_core"] = float(core)
    policy["decoder_tail"] = float(tail)
    set_noise_multipliers(model, policy)
    try:
        indices = np.repeat(np.arange(len(z)), draws)
        masses = np.empty(len(indices), dtype=np.float64)
        for start in range(0, len(indices), int(batch_size)):
            block = indices[start : start + int(batch_size)]
            values = torch.as_tensor(
                np.ascontiguousarray(z[block]), dtype=torch.float32, device=device
            )
            decoded = _first(model.decode(values))
            masses[start : start + len(block)] = invariant_mass_np(
                decoded.detach().cpu().numpy(),
                daughter_masses=daughter_masses,
                stable=True,
            )
        matrix = masses.reshape(len(z), draws)
    finally:
        set_noise_multipliers(model, previous)

    within_std = matrix.std(axis=1)
    within_robust = (
        np.quantile(matrix, 0.84, axis=1) - np.quantile(matrix, 0.16, axis=1)
    ) / 2.0
    within_variance = within_std**2
    total_variance = float(matrix.var())
    noise_fraction = (
        float(within_variance.mean() / total_variance) if total_variance > 0.0 else 0.0
    )
    return {
        "noise_budget_events": float(len(z)),
        "noise_budget_draws": float(draws),
        "noise_budget_decoder_core": float(core),
        "noise_budget_decoder_tail": float(tail),
        "noise_budget_within_std_median_gev": float(np.median(within_std)),
        "noise_budget_within_std_mean_gev": float(within_std.mean()),
        "noise_budget_within_robust_median_gev": float(np.median(within_robust)),
        "noise_budget_sigma_only_within_gev": float(
            np.sqrt(max(within_variance.mean(), 0.0))
        ),
        "noise_budget_noise_variance_fraction": noise_fraction,
        "noise_budget_ensemble_std_gev": float(matrix.std()),
    }


def score_joint_metrics(
    region_metrics: dict[str, dict[str, float]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Normalize targets, reduce within each region, then select the worst region.

    Checkpoint selection is not pass/fail: the score is lower-is-better and is
    based on how far each metric is from its configured target.
    """
    selection = config.get("checkpoint_selection", {})
    base_weight = float(selection.get("base_loss_weight", 0.0))
    region_reduction = str(selection.get("region_reduction", "max")).lower()
    if region_reduction not in {"max", "mean", "rms"}:
        raise ValueError(
            "checkpoint_selection.region_reduction must be one of: max, mean, rms"
        )
    region_reports: dict[str, Any] = {}

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
        }

    worst_region = max(region_reports, key=lambda key: region_reports[key]["score"])
    raw_score = float(region_reports[worst_region]["score"])
    return {
        "selection_score": raw_score,
        "raw_worst_region_score": raw_score,
        "worst_region": worst_region,
        "regions": region_reports,
    }
