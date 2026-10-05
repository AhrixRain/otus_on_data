"""Strictly proper scoring rules for conditional distributions.

One implementation, used by the training loss, the offline diagnostics and the
unit tests, so any number quoted anywhere comes from the same code.

What is here
------------
:func:`gaussian_log_score` and :func:`gaussian_log_score_from_draws` - the
strictly proper scoring rule for a Gaussian conditional, in the negative
orientation:

    ``S(mu, sigma, x) = (x - mu)^2 / (2 sigma^2) + log sigma``, lower is better.

Minimising it drives ``mu -> x`` and ``sigma -> |x - mu|`` together. It is the
only rule in this module that penalises a collapsed channel, and it is what the
training term uses. Two rejected alternatives are documented below because both
look reasonable and neither works.

Why not the energy score
------------------------
``E|X~ - x| - (1/2) E|X~ - X~'|`` is strictly proper over the predictive
distribution and has the right minimum over a Gaussian scale family
(analytically ``c = sigma``; measured 0.567 at the correct spread versus 0.699
and 0.657 either side). But with ``m`` identical draws - a fully collapsed
channel - its within term vanishes and it returns **zero**, better than the
correct value, so a trainer minimising it drives ``sigma -> 0``. Measured
directly: identical draws at a unit-Gaussian target score 0.0 against 0.567 for
the correct spread. In the 1-D toy study the marginal term happened to hold the
scale up, which is exactly why the toy alone is not sufficient evidence.

Why not a Gauss-kernel score
----------------------------
With a fixed plug-in bandwidth and a scale family the kernel score *decreases*
with the spread: a too-narrow model (``c = 0.2``) scored 0.216 against 0.286 for
the correct model (``c = 1``), and a shuffled target scored 0.097 because the
bounded kernel simply vanishes when model and data do not overlap. A bounded
kernel score is a similarity, not a divergence.

Relationship to the marginal
----------------------------
A marginal-matching objective constrains only the law of ``x``. A deterministic
decoder can reproduce any target marginal by transporting the prior, so such an
objective has no barrier at ``sigma(z) = 0`` and the conditional spread is
invisible to it. The log score is evaluated at the realised ``x`` of each event,
which is the structural difference that lets it see the spread.
"""

from __future__ import annotations

import torch

__all__ = [
    "gaussian_log_score",
    "gaussian_log_score_from_draws",
    "median_bandwidth",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _as_columns(values: torch.Tensor) -> torch.Tensor:
    """Return a 2-D (events, dimensions) view of ``values``."""
    if values.dim() == 1:
        return values.reshape(-1, 1)
    if values.dim() != 2:
        raise ValueError(f"expected 1-D or 2-D values, got shape {tuple(values.shape)}")
    return values


def _stack_draws(draws: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalise ``draws`` to (m, n, d) and ``target`` to (n, d)."""
    target_columns = _as_columns(target)
    if draws.dim() == 2:
        if draws.shape[0] != target_columns.shape[0]:
            raise ValueError(
                "a single draw needs one row per target: "
                f"{tuple(draws.shape)} vs {tuple(target_columns.shape)}"
            )
        return draws.unsqueeze(0), target_columns
    if draws.dim() == 3:
        if draws.shape[1] != target_columns.shape[0]:
            raise ValueError(
                "draws and targets must share the event axis: "
                f"{tuple(draws.shape)} vs {tuple(target_columns.shape)}"
            )
        return draws, target_columns
    raise ValueError(f"draws must be (n, d) or (m, n, d), got {tuple(draws.shape)}")


def _l1_diagonal(draw: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """Per-event mean-absolute distance between two aligned blocks.

    Both terms of the score are *per event*: the cross term uses each event's own
    realised value and the within term pairs draws of the same event. Averaging
    over all (draw, event) pairs instead scores the ensemble against the
    empirical marginal, which is exactly the quantity that cannot see the
    conditional spread.
    """
    return (draw - other).abs().mean(dim=1)


# ---------------------------------------------------------------------------
# scale estimator
# ---------------------------------------------------------------------------

def median_bandwidth(
    values: torch.Tensor,
    *,
    max_events: int = 4096,
    floor: float = 1e-6,
) -> torch.Tensor:
    """Robust scale of ``values``, usable as a kernel bandwidth or a report scale.

    The normalised median absolute deviation, ``1.4826 * median(|x - median|)``,
    so standard-normal data returns the standard deviation. The MAD is used
    rather than a median pairwise distance because the latter is inflated at
    finite ``n`` (the ``n^2`` pair sample is not the ``n -> inf`` limit; measured
    +5% at ``n = 3000``), which would bias any scale away from the data.
    """
    columns = _as_columns(values)
    if columns.shape[0] > int(max_events):
        step = columns.shape[0] // int(max_events)
        columns = columns[:: max(step, 1)]
    if columns.shape[0] < 2:
        return torch.as_tensor(
            max(float(floor), 1e-3), dtype=values.dtype, device=values.device
        )
    centred = columns - columns.median(dim=0).values
    mad = torch.median(centred.abs(), dim=0).values
    return torch.clamp(1.4826 * mad.mean(), min=float(floor))


# ---------------------------------------------------------------------------
# the score
# ---------------------------------------------------------------------------

def gaussian_log_score(
    mean: torch.Tensor, scale: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Gaussian log score ``(x - mu)^2 / (2 sigma^2) + log sigma``, lower is better.

    This is the strictly proper scoring rule for a Gaussian conditional, in the
    negative orientation: minimising it drives ``mu -> x`` and ``sigma -> |x - mu|``
    jointly. The ``log sigma`` term is the only term in this module that actually
    penalises a collapsed channel, and it is why this rule, not the energy score,
    is what the training term uses.

    Why not the energy score: with ``m`` identical draws (a fully collapsed
    channel) it returns exactly zero, which is *better* than the value at the
    correct spread (measured 0.0 versus 0.567 for a unit Gaussian), so minimising
    it drives ``sigma -> 0``. The same trap applies to any rule whose within-model
    term collapses with the spread. See the module docstring.
    """
    residual = target - mean
    variance = scale.pow(2).clamp_min(1e-12)
    return (0.5 * residual.pow(2) / variance + torch.log(scale.clamp_min(1e-12))).mean()


def gaussian_log_score_from_draws(
    draws: torch.Tensor, target: torch.Tensor, *, floor: float = 1e-9
) -> torch.Tensor:
    """Gaussian log score with the moments estimated from the draws themselves.

    ``draws`` is (m, n, ...) with at least two independent decodes per event; the
    per-event mean and standard deviation are taken over the draw axis, then
    scored against the realised target. Reported per coordinate and averaged, so
    a collapsed channel (all draws identical) gives ``sigma = floor`` and a
    diverging penalty rather than the free pass the energy score hands out.

    ``floor`` is a **relative** floor, applied to the per-coordinate scale of the
    target sample, not an absolute one. An absolute floor cannot work here: the
    score is scale-free, so a floor that is large compared to the data makes the
    first term negligible everywhere and the term degenerates to ``log sigma``.
    Callers must standardise (or otherwise scale) their coordinates so that a
    relative floor is meaningful; ``scripts/loss.py`` divides by the target
    sample's per-coordinate standard deviation before calling this.
    """
    draws, target = _stack_draws(draws, target)
    m = draws.shape[0]
    if m < 2:
        raise ValueError(
            "gaussian_log_score_from_draws needs at least two draws per event; "
            f"got {m}."
        )
    reference = target.detach().std(dim=0, unbiased=False).clamp_min(1e-8)
    minimum = (float(floor) if float(floor) > 0.0 else 1e-9) * reference
    mean = draws.mean(dim=0)
    scale = draws.std(dim=0, unbiased=False).clamp_min(minimum)
    return gaussian_log_score(mean, scale, target)
