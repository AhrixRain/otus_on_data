"""Identity-map references and finite-sample floors for the joint metrics.

Why this module exists
----------------------
The 2026-09-04/05 prior-width A/B (memory.md section 7.2) measured this: on the
smeared J/psi prior the trivial map ``z~ = x`` -- no model at all -- scores
latent mass KS 0.0359, W1 0.00219 and width relative error 0.071 against
strict targets of 0.04, 0.003 and 0.1. It passes all three. A gate that a
no-op passes cannot certify that any unfolding happened, so every gate number
has to be published next to two references:

``identity``
    what the metric reads when the map is the identity (the model output is
    replaced by its own input). This is the "did nothing" line.
``floor``
    what the metric reads between two independent draws of the *same*
    distribution at the sample size actually used. This is the best score any
    map can achieve; it is pure finite-sample noise.

and one gauge that turns the three numbers into a decision:

    vs_identity = (model - floor) / (identity - floor)

    1.0  the model did exactly as well as doing nothing
    0.0  the model reached the finite-sample floor (perfect, given n)
    >1   the model is worse than doing nothing

``vs_identity`` is lower-is-better and non-negative, so it drops into the
existing ``abs(value) <= threshold`` gate machinery in
``joint_metrics.score_joint_metrics`` with no change to the selection logic.

    headroom = (identity - floor) / floor

is a property of the DATASET, not of the model: it says how much separation the
comparison could detect at all. Small headroom means the region cannot tell an
unfolding model from a no-op whatever it scores. On the A/B: narrow headroom
~18, smeared headroom ~1, which is the whole story of that experiment in one
number.

Design constraints
------------------
* NumPy only. No torch import, so finished runs can be re-scored offline and
  so the pure statistics can be unit-tested without a GPU environment.
* The invariant-mass convention is injected as ``mass_fn`` rather than
  imported, so this module can never drift from the caller's definition.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np


# Metrics that are gauged against the identity map. All are non-negative and
# lower-is-better. ``mass_mean_shift_gev`` is deliberately absent: it is signed,
# so a ratio against the identity is not interpretable.
GAUGED_SUFFIXES = (
    "mass_ks",
    "mass_w1_gev",
    "mass_width_rel_error",
    "pair_pt_ks",
)


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


def width_relative_error(real: np.ndarray, fake: np.ndarray) -> float:
    real_width = float(np.std(real))
    fake_width = float(np.std(fake))
    return abs(fake_width - real_width) / max(real_width, 1.0e-12)


def equal_subset(
    a: np.ndarray,
    b: np.ndarray,
    *,
    max_events: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic equal-count subsets of two arrays.

    Lives here rather than in ``joint_metrics`` so that offline re-scoring can
    reproduce a finished run's validation geometry exactly without importing
    torch. ``joint_metrics`` imports this definition; there is only one.
    """
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


def gauge(model: float, identity: float, floor: float) -> float:
    """(model - floor) / (identity - floor), clamped at zero from below.

    Returns +inf when the identity map is already at or below the floor. That
    is not a failure of the model, it is a failure of the COMPARISON: there is
    nothing left to detect, so nothing can be certified. Reporting +inf makes a
    gate on this metric refuse to pass rather than pass for free; read it
    together with ``headroom``, which says why.
    """
    if not (math.isfinite(model) and math.isfinite(identity) and math.isfinite(floor)):
        return float("inf")
    denominator = identity - floor
    if denominator <= 0.0:
        return float("inf")
    return max(0.0, (model - floor) / denominator)


def headroom(identity: float, floor: float) -> float:
    """(identity - floor) / floor. How much separation the test can resolve."""
    if not (math.isfinite(identity) and math.isfinite(floor)) or floor <= 0.0:
        return float("inf")
    return (identity - floor) / floor


def _floor_pair(
    pool: np.ndarray, *, n: int, seed: int
) -> tuple[np.ndarray, np.ndarray, bool, float]:
    """Two same-distribution samples for the finite-sample floor.

    Preferred: two DISJOINT draws of exactly ``n`` events from ``pool``, which
    reproduces the model comparison's sample geometry exactly (n vs n, no
    shared events) and needs no rescaling.

    Fallback, when the pool holds fewer than ``2n`` events: two disjoint halves
    of size ``m = len(pool) // 2``. Both KS and W1 between two samples of size
    m scale as ``1 / sqrt(m)``, so the measurement is rescaled by
    ``sqrt(m / n)`` to the sample size the model was scored at. That rescaling
    is an asymptotic argument, not an identity; the returned flag records which
    branch ran so a reader can tell.

    Returns (a, b, disjoint_exact, scale).
    """
    pool = np.asarray(pool)
    rng = np.random.default_rng(seed)
    total = len(pool)
    if total >= 2 * n:
        index = rng.choice(total, 2 * n, replace=False)
        return pool[index[:n]], pool[index[n:]], True, 1.0
    half = total // 2
    if half < 2:
        raise ValueError("Floor estimate needs at least four reference events")
    index = rng.permutation(total)
    return pool[index[:half]], pool[index[half : 2 * half]], False, math.sqrt(half / n)


def reference_block(
    prefix: str,
    *,
    real_values: np.ndarray,
    identity_values: np.ndarray,
    reference_pool: np.ndarray,
    model_metrics: dict[str, float],
    mass_fn: Callable[[np.ndarray], np.ndarray],
    seed: int,
) -> dict[str, float]:
    """Identity, floor, gauge and headroom for one direction.

    ``real_values``     the target-distribution sample the model was scored
                        against ([N, 8] four-vector pairs).
    ``identity_values`` the model's own INPUT, i.e. what its output would be if
                        the map were the identity.
    ``reference_pool``  a larger sample of the target distribution, used only
                        for the floor. Pass the full split array; the function
                        takes disjoint draws from it.
    ``model_metrics``   the already-computed ``{prefix}_*`` metrics, so the
                        gauge divides exactly the numbers that are gated.

    Every returned value is a float, including the flags, so downstream code
    that iterates the metrics dict never meets a non-numeric entry.
    """
    n = int(len(real_values))
    real_mass = mass_fn(real_values)
    identity_mass = mass_fn(identity_values)
    floor_a, floor_b, disjoint, scale = _floor_pair(
        reference_pool, n=n, seed=seed
    )
    floor_a_mass = mass_fn(floor_a)
    floor_b_mass = mass_fn(floor_b)

    identity_values_by_suffix = {
        "mass_ks": ks_distance(real_mass, identity_mass),
        "mass_w1_gev": wasserstein_1d(real_mass, identity_mass),
        "mass_width_rel_error": width_relative_error(real_mass, identity_mass),
        "pair_pt_ks": ks_distance(pair_pt(real_values), pair_pt(identity_values)),
    }
    floor_values_by_suffix = {
        "mass_ks": scale * ks_distance(floor_a_mass, floor_b_mass),
        "mass_w1_gev": scale * wasserstein_1d(floor_a_mass, floor_b_mass),
        "mass_width_rel_error": scale
        * width_relative_error(floor_a_mass, floor_b_mass),
        "pair_pt_ks": scale * ks_distance(pair_pt(floor_a), pair_pt(floor_b)),
    }

    block: dict[str, float] = {
        f"{prefix}_floor_events": float(n),
        f"{prefix}_floor_disjoint": 1.0 if disjoint else 0.0,
        f"{prefix}_floor_scale": float(scale),
    }
    for suffix in GAUGED_SUFFIXES:
        identity_value = float(identity_values_by_suffix[suffix])
        floor_value = float(floor_values_by_suffix[suffix])
        model_value = float(model_metrics[f"{prefix}_{suffix}"])
        block[f"{prefix}_identity_{suffix}"] = identity_value
        block[f"{prefix}_floor_{suffix}"] = floor_value
        block[f"{prefix}_{suffix}_vs_identity"] = gauge(
            model_value, identity_value, floor_value
        )
        block[f"{prefix}_{suffix}_headroom"] = headroom(identity_value, floor_value)
    return block
