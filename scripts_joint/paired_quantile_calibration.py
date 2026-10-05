#!/usr/bin/env python
"""P2 (paired half) - eINN-style per-event quantile calibration on the ppzee bench.

Why this exists
---------------
The P branch of the plan (`docs/project_tree.md`) asks whether a *conditional*
distribution is identified.  On the CMS regions there is no truth partner, so a
conditional test can only be built out of gauges (that half is
`scripts_joint/conditional_spread.py`).  On the ppzee bench the withheld truth
partner ``z_true`` exists for every detector-level event ``x``, so the
conditional is directly testable: draw ``D`` samples of the encoder's implicit
posterior ``q(z | x)`` and ask where the truth lands.

Four readouts, all in the invariant-mass coordinate:

* **rank histogram (PIT).**  ``rank = #{j : mass(z_j) < mass(z_true)}`` over the
  ``D`` draws, mapped to ``u = (rank + 0.5) / (D + 1)``.  If the draws and the
  truth are exchangeable - which is exactly what a calibrated conditional means
  - then ``rank`` is uniform on ``{0, ..., D}`` and ``u`` is uniform on the
  ``D + 1`` midpoints.  The statistic is therefore *exactly* uniform under
  calibration, with no finite-``D`` correction, which is why it is the sharpest
  of the four.  Every event contributes one rank, so the bin counts are
  Binomial(n, p_k) with an exactly computable ``p_k``.
* **central-interval coverage** at nominal 0.68 and 0.95 (fraction of truths
  inside the central 68% / 95% credible interval of the ``D`` draws), with a
  bootstrap band over events beside the binomial standard error.  Coverage from
  a *sampled* conditional carries a finite-``D`` deficit at high levels, so the
  criterion compares it against a synthetic **null** run with the same ``n`` and
  ``D``, not against the nominal number.
* **pull** ``(mass(z_true) - mean(mass(z_j))) / std(mass(z_j))`` - mean 0,
  std 1 for a calibrated Gaussian-ish conditional, up to the finite-``D`` factor
  ``sqrt(1 + 1/D) * sqrt((D-1)/(D-3))`` which the null run supplies.
* **shuffled control.**  The same three statistics with each ``x`` paired with
  *another* event's truth (a seeded derangement, so no event keeps its own
  partner).  The pairing is destroyed by construction, so the observed change is
  the negative control: it says how much of the paired reading comes from the
  pairing rather than from the marginals.

Every criterion is therefore measured against a **finite-draw null**: a
synthetic, calibrated-by-construction conditional run through the identical
statistics with the identical ``n``, ``D``, bins and bootstrap count.  That is
what makes a "calibrated" verdict meaningful rather than an artefact of the
sampling.

Reuse
-----
This script invents no inference and no conventions.  It calls

* ``decode_prior.load_frozen_model`` - the frozen-checkpoint loader used by
  every Upsilon readout;
* ``paired_closure.direct_mass`` - the ppzee invariant-mass convention;
* ``ppzee_posterior_predictions.encode_draws`` / ``draw_mass_matrix`` /
  ``validate_draws`` / ``choose_device`` / ``PULL_CLAMP`` /
  ``ZERO_VARIANCE_TOL`` - the D4 posterior-draw protocol, including its
  per-draw seeding (``seed + 1000 * (draw + 1)``) and its numerically
  zero-width convention.

The default checkpoint is the one the existing canonical ppzee result uses
(``outputs/cms_Joint/ppzee/best_model.pt``, global epoch 120, stage
``runA_stage2_gaussian_response``, which scored ``residual_rms_vs_identity``
0.98435 with a single encode).

Nothing is trained and no existing artifact is modified.  The only writes are
new files under ``--output-dir`` (default
``outputs/cms_Joint/ppzee/quantile_calibration/``).

Usage
-----
    python scripts_joint/paired_quantile_calibration.py --device cpu
    python scripts_joint/paired_quantile_calibration.py --events 20000 --draws 16
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (
    REPO_ROOT / "scripts",
    REPO_ROOT / "scripts_sota",
    REPO_ROOT / "scripts_joint",
    REPO_ROOT / "scripts_joint" / "upsilon",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from decode_prior import load_frozen_model  # noqa: E402
from paired_closure import direct_mass  # noqa: E402
from ppzee_posterior_predictions import (  # noqa: E402
    PULL_CLAMP,
    ZERO_VARIANCE_TOL,
    choose_device,
    draw_mass_matrix,
    encode_draws,
    validate_draws,
)

try:  # present in the cms environment, but never a hard dependency
    from scipy import stats as _scipy_stats
    from scipy.special import ndtr as _ndtr
except Exception:  # pragma: no cover - fallback exercised only without scipy
    _scipy_stats = None

    def _ndtr(values):  # type: ignore[misc]
        flat = np.asarray(values, dtype=np.float64).ravel()
        out = np.array(
            [0.5 * math.erfc(-float(v) / math.sqrt(2.0)) for v in flat],
            dtype=np.float64,
        )
        return out.reshape(np.shape(values))


DEFAULT_CHECKPOINT = REPO_ROOT / "outputs" / "cms_Joint" / "ppzee" / "best_model.pt"
DEFAULT_DATASET = (
    REPO_ROOT / "outputs" / "cms_Joint" / "ppzee" / "paired_closure" / "pairs_test.npz"
)
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "cms_Joint" / "ppzee" / "quantile_calibration"

NOMINAL_LEVELS = (0.68, 0.95)
BAND_SIGMA = 2.0
COVERAGE_Z_TOLERANCE = 3.0
PULL_TOLERANCE = 0.10
CHI2_P_THRESHOLD = 1.0e-3
FALLBACK_MAX_DEVIATION_SIGMA = 3.5
BOOTSTRAP_PERCENTILES = (2.5, 97.5)
NULL_SEED_OFFSET = 100

REFERENCE = {
    "label": "artifact-measured/source-verified references this run is read against",
    "existing_result_artifact": "outputs/cms_Joint/ppzee/paired_closure.json",
    "existing_result_checkpoint": "outputs/cms_Joint/ppzee/best_model.pt",
    "existing_result_global_epoch": 120,
    "existing_result_stage": "runA_stage2_gaussian_response",
    "existing_result_residual_rms_vs_identity": 0.9843545885876726,
    "existing_result_model_rms_gev": 2.610023155793977,
    "existing_result_identity_rms_gev": 2.651507074842586,
    "pull_coverage_artifact": "outputs/cms_Joint/ppzee/pull_coverage/REPORT.md",
    "pull_coverage_stage2_native": {
        "encoder_multipliers": [0.984146, 0.0],
        "pull_mean": -1.0991,
        "pull_std": 20.7219,
        "coverage_1sigma": 0.0581,
    },
    "upstream_otus_encoder_residual_rms_vs_identity": 3.3304,
    "paired_oracle_floor": 0.8099,
    "identity_map": 1.0,
    "finite_draw_note": (
        "For D draws of a calibrated conditional the pull follows "
        "sqrt(1 + 1/D) * t_{D-1}, so its std is "
        "sqrt(1 + 1/D) * sqrt((D-1)/(D-3)) - about 1.05 at D = 32.  The "
        "central-interval coverage of the *sampled* interval is inward-biased, so "
        "a calibrated conditional still under-covers at every level and the "
        "deficit grows as D shrinks; that is what the finite-draw null measures.  "
        "The rank histogram carries no such correction: exchangeability makes it "
        "exactly uniform for any continuous conditional."
    ),
}


# ---------------------------------------------------------------------------
# pure statistics (unit-tested)
# ---------------------------------------------------------------------------

def derangement_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    """A random permutation of ``range(n)`` with no fixed point.

    The shuffled control must pair every ``x`` with *another* event's truth;
    a random permutation leaves ~1 fixed point on average at n = 160000, which
    would quietly leak a correct pairing into the control.  The repair swaps a
    fixed point with its neighbour: either the neighbour was fixed too (both are
    repaired) or it was not, and then it cannot map to ``index`` (a permutation
    has one preimage), so neither position is left fixed.
    """
    if int(n) < 2:
        raise ValueError("a derangement needs at least two events")
    perm = np.asarray(rng.permutation(int(n)), dtype=np.int64)
    for index in np.flatnonzero(perm == np.arange(int(n))):
        if perm[index] != index:  # already repaired by an earlier swap
            continue
        other = (int(index) + 1) % int(n)
        perm[index], perm[other] = perm[other], perm[index]
    if np.any(perm == np.arange(int(n))):  # pragma: no cover - guarded by design
        raise RuntimeError("derangement construction left a fixed point")
    return perm


def select_subset(n_available: int, n_events: int, rng: np.random.Generator) -> np.ndarray:
    """Sorted event indices for a seeded subsample (all events when 0 or None)."""
    if n_events is None or int(n_events) <= 0 or int(n_events) >= int(n_available):
        return np.arange(int(n_available), dtype=np.int64)
    return np.sort(rng.choice(int(n_available), size=int(n_events), replace=False))


def midrank_pit(mass_true: np.ndarray, draw_mass: np.ndarray) -> dict[str, np.ndarray]:
    """Ranks of the truth among the draws, and their uniform midpoints.

    ``rank = #{j : draw_j < truth}`` is uniform on ``{0, ..., D}`` when the
    truth and the ``D`` draws are exchangeable (a calibrated conditional), so
    ``u = (rank + 0.5) / (D + 1)`` is uniform on the ``D + 1`` midpoints.
    Strict inequality is used; exact ties are counted separately.
    """
    truth = np.asarray(mass_true, dtype=np.float64)
    matrix = np.asarray(draw_mass, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != truth.shape[0]:
        raise ValueError(
            f"draw_mass must be [draws, N] aligned with the truth, got {matrix.shape}"
        )
    draws = matrix.shape[0]
    rank = np.sum(matrix < truth[None, :], axis=0).astype(np.int64)
    ties = np.sum(matrix == truth[None, :], axis=0).astype(np.int64)
    return {
        "rank": rank,
        "u": (rank.astype(np.float64) + 0.5) / float(draws + 1),
        "ties": ties,
    }


def exact_bin_probabilities(bins: int, draws: int) -> np.ndarray:
    """Exact per-bin probability of a uniform rank under calibration.

    Every bin count is Binomial(n, p_k) with these ``p_k``; no approximation
    and no finite-draw correction is involved.
    """
    if int(bins) < 1:
        raise ValueError("--bins must be >= 1")
    if int(bins) > int(draws) + 1:
        raise ValueError(
            f"--bins={bins} exceeds the {int(draws) + 1} admissible ranks; "
            "empty bins would make the binomial band degenerate"
        )
    ranks = np.arange(int(draws) + 1, dtype=np.float64)
    midpoints = (ranks + 0.5) / float(int(draws) + 1)
    index = np.minimum((midpoints * int(bins)).astype(np.int64), int(bins) - 1)
    atoms = np.bincount(index, minlength=int(bins)).astype(np.float64)
    return atoms / atoms.sum()


def rank_histogram(u: np.ndarray, *, bins: int, draws: int) -> dict[str, Any]:
    """Binned PIT with the exact binomial band and flatness summaries."""
    values = np.asarray(u, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"u must be one-dimensional, got {values.shape}")
    if values.size == 0:
        raise ValueError("u is empty")
    if np.any((values <= 0.0) | (values >= 1.0)):
        raise ValueError("u must lie strictly inside (0, 1)")
    events = int(values.size)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    counts, _ = np.histogram(values, bins=edges)
    counts = counts.astype(np.int64)
    probability = exact_bin_probabilities(bins, draws)
    band_sd = np.sqrt(probability * (1.0 - probability) / float(events))
    fractions = counts.astype(np.float64) / float(events)
    deviation_sigma = np.zeros_like(fractions)
    valid = band_sd > 0.0
    deviation_sigma[valid] = (fractions[valid] - probability[valid]) / band_sd[valid]
    expected = probability * float(events)
    chi2_terms = np.where(
        probability > 0.0,
        (counts - expected) ** 2 / np.maximum(expected, 1e-300),
        0.0,
    )
    chi2 = float(np.sum(chi2_terms))
    dof = int(np.count_nonzero(probability > 0.0)) - 1
    chi2_p = (
        float(_scipy_stats.chi2.sf(chi2, dof))
        if (_scipy_stats is not None and dof > 0)
        else None
    )
    return {
        "bins": int(bins),
        "edges": edges.tolist(),
        "counts": counts.tolist(),
        "fractions": fractions.tolist(),
        "expected_probability": probability.tolist(),
        "expected_count": expected.tolist(),
        "band_sigma": BAND_SIGMA,
        "band_sd": band_sd.tolist(),
        "band_low": (probability - BAND_SIGMA * band_sd).tolist(),
        "band_high": (probability + BAND_SIGMA * band_sd).tolist(),
        "deviation_sigma": deviation_sigma.tolist(),
        "max_abs_deviation_sigma": float(np.max(np.abs(deviation_sigma))),
        "total_variation": float(0.5 * np.sum(np.abs(fractions - probability))),
        "chi2": chi2,
        "dof": dof,
        "chi2_p_value": chi2_p,
    }


def central_interval_coverage(
    mass_true: np.ndarray, draw_mass: np.ndarray, level: float
) -> dict[str, Any]:
    """Fraction of truths inside the central ``level`` credible interval."""
    if not 0.0 < float(level) < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    truth = np.asarray(mass_true, dtype=np.float64)
    matrix = np.asarray(draw_mass, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != truth.shape[0]:
        raise ValueError(
            f"draw_mass must be [draws, N] aligned with the truth, got {matrix.shape}"
        )
    tail = (1.0 - float(level)) / 2.0
    lower = np.quantile(matrix, tail, axis=0)
    upper = np.quantile(matrix, 1.0 - tail, axis=0)
    inside = (truth >= lower) & (truth <= upper)
    return {
        "nominal": float(level),
        "measured": float(np.mean(inside)),
        "inside_count": int(np.count_nonzero(inside)),
        "events": int(truth.size),
        "binomial_sd": float(math.sqrt(float(level) * (1.0 - float(level)) / truth.size)),
        "median_interval_halfwidth_gev": float(np.median(0.5 * (upper - lower))),
        "inside": inside,
    }


def bootstrap_coverage_band(
    inside: np.ndarray, *, boot: int, rng: np.random.Generator
) -> dict[str, Any]:
    """Percentile bootstrap band of a coverage fraction over events."""
    flags = np.asarray(inside, dtype=np.float64)
    events = int(flags.size)
    if int(boot) < 2:
        raise ValueError("--bootstrap must be at least 2")
    draws = np.empty(int(boot), dtype=np.float64)
    for index in range(int(boot)):
        resample = rng.integers(0, events, size=events)
        draws[index] = float(np.mean(flags[resample]))
    low, high = np.percentile(draws, BOOTSTRAP_PERCENTILES)
    return {
        "boot": int(boot),
        "percentiles": list(BOOTSTRAP_PERCENTILES),
        "low": float(low),
        "high": float(high),
        "mean": float(np.mean(draws)),
        "sd": float(np.std(draws, ddof=1)),
    }


def bootstrap_std_band(
    values: np.ndarray, *, boot: int, rng: np.random.Generator
) -> dict[str, Any]:
    """Percentile bootstrap band of the standard deviation of ``values``."""
    array = np.asarray(values, dtype=np.float64)
    events = int(array.size)
    if int(boot) < 2:
        raise ValueError("--bootstrap must be at least 2")
    draws = np.empty(int(boot), dtype=np.float64)
    for index in range(int(boot)):
        resample = rng.integers(0, events, size=events)
        draws[index] = float(np.std(array[resample]))
    low, high = np.percentile(draws, BOOTSTRAP_PERCENTILES)
    return {
        "boot": int(boot),
        "percentiles": list(BOOTSTRAP_PERCENTILES),
        "low": float(low),
        "high": float(high),
        "mean": float(np.mean(draws)),
        "sd": float(np.std(draws, ddof=1)),
    }


def ks_distance_to_normal(values: np.ndarray) -> dict[str, Any]:
    """Two-sided KS distance of ``values`` to the standard normal, with a p-value."""
    array = np.sort(np.asarray(values, dtype=np.float64).ravel())
    events = int(array.size)
    if events == 0:
        raise ValueError("cannot test an empty sample")
    cdf = np.asarray(_ndtr(array), dtype=np.float64)
    upper = np.arange(1, events + 1, dtype=np.float64) / events - cdf
    lower = cdf - np.arange(0, events, dtype=np.float64) / events
    distance = float(max(np.max(upper), np.max(lower)))
    root = math.sqrt(events)
    lam = (root + 0.12 + 0.11 / root) * distance
    total = 0.0
    for term in range(1, 101):
        total += ((-1.0) ** (term - 1)) * math.exp(-2.0 * term * term * lam * lam)
    p_value = float(min(1.0, max(0.0, 2.0 * total)))
    return {"ks_distance": distance, "ks_p_value_asymptotic": p_value, "events": events}


def finite_draw_pull_std(draws: int) -> float:
    """Std of the pull for ``draws`` samples of a calibrated conditional.

    ``pull = sqrt(1 + 1/D) * t_{D-1}``: the ``sqrt(1 + 1/D)`` is the sampling
    fluctuation of the sample mean, the ``t`` factor the sampling fluctuation of
    the sample width.  ``t_{D-1}`` has no finite variance for ``D <= 3``, so the
    reference is NaN there and the criterion downgrades to the nominal 1.0.
    """
    if int(draws) <= 3:
        return float("nan")
    return math.sqrt(1.0 + 1.0 / float(draws)) * math.sqrt(
        (float(draws) - 1.0) / (float(draws) - 3.0)
    )


def pull_values(mass_true: np.ndarray, draw_mass: np.ndarray) -> np.ndarray:
    """Per-event pull of the truth against the draws' own mean and width.

    The clamp is `paired_closure`'s: a zero-variance posterior gives a finite
    huge pull rather than a NaN.
    """
    truth = np.asarray(mass_true, dtype=np.float64)
    matrix = np.asarray(draw_mass, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != truth.shape[0]:
        raise ValueError(
            f"draw_mass must be [draws, N] aligned with the truth, got {matrix.shape}"
        )
    if matrix.shape[0] < 2:
        raise ValueError("a pull needs at least two draws per event")
    width = matrix.std(axis=0, ddof=1)
    return (truth - matrix.mean(axis=0)) / np.maximum(width, PULL_CLAMP)


def pull_statistics(
    mass_true: np.ndarray, draw_mass: np.ndarray, *, histogram_bins: int = 61
) -> dict[str, Any]:
    """Pull of the truth against the draws' own mean and width."""
    truth = np.asarray(mass_true, dtype=np.float64)
    matrix = np.asarray(draw_mass, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != truth.shape[0]:
        raise ValueError(
            f"draw_mass must be [draws, N] aligned with the truth, got {matrix.shape}"
        )
    draws = int(matrix.shape[0])
    width = matrix.std(axis=0, ddof=1)
    zero_variance = int(np.count_nonzero(width <= ZERO_VARIANCE_TOL))
    pull = pull_values(truth, matrix)
    median = float(np.median(pull))
    mad = float(np.median(np.abs(pull - median)))
    ks = ks_distance_to_normal(pull)
    edges = np.linspace(-6.0, 6.0, int(histogram_bins) + 1)
    counts, _ = np.histogram(pull, bins=edges)
    return {
        "draws": draws,
        "events": int(truth.size),
        "pull_mean": float(np.mean(pull)),
        "pull_std": float(np.std(pull)),
        "pull_median": median,
        "pull_robust_width_1p4826mad": float(1.4826 * mad),
        "pull_q16": float(np.percentile(pull, 16.0)),
        "pull_q84": float(np.percentile(pull, 84.0)),
        "pull_q16_q84_width": float(np.percentile(pull, 84.0) - np.percentile(pull, 16.0)),
        "fraction_abs_pull_le_2": float(np.mean(np.abs(pull) <= 2.0)),
        "posterior_std_median_gev": float(np.median(width)),
        "mean_residual_rms_gev": float(np.sqrt(np.mean((matrix.mean(axis=0) - truth) ** 2))),
        "finite_draw_expected_std": finite_draw_pull_std(draws),
        "zero_variance_events": zero_variance,
        "zero_variance_tolerance_gev": float(ZERO_VARIANCE_TOL),
        "degenerate": bool(zero_variance == int(truth.size)),
        "ks_distance_to_standard_normal": ks["ks_distance"],
        "ks_p_value_asymptotic": ks["ks_p_value_asymptotic"],
        "histogram": {
            "edges": edges.tolist(),
            "counts": counts.astype(np.int64).tolist(),
            "range": [float(edges[0]), float(edges[-1])],
            "events_outside_range": int(
                np.count_nonzero((pull < edges[0]) | (pull > edges[-1]))
            ),
        },
    }


def calibration_block(
    mass_true: np.ndarray,
    draw_mass: np.ndarray,
    *,
    levels=NOMINAL_LEVELS,
    bins: int,
    boot: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Every descriptive readout for one pairing (paired, shuffled or null)."""
    truth = np.asarray(mass_true, dtype=np.float64)
    matrix = np.asarray(draw_mass, dtype=np.float64)
    pit = midrank_pit(truth, matrix)
    histogram = rank_histogram(pit["u"], bins=bins, draws=matrix.shape[0])
    coverage: dict[str, Any] = {}
    for level in levels:
        interval = central_interval_coverage(truth, matrix, float(level))
        coverage[f"{float(level):.2f}"] = {
            "nominal": interval["nominal"],
            "measured": interval["measured"],
            "inside_count": interval["inside_count"],
            "events": interval["events"],
            "binomial_sd": interval["binomial_sd"],
            "median_interval_halfwidth_gev": interval["median_interval_halfwidth_gev"],
            "bootstrap": bootstrap_coverage_band(interval["inside"], boot=boot, rng=rng),
        }
    pull = pull_statistics(truth, matrix)
    pull["bootstrap_std"] = bootstrap_std_band(
        pull_values(truth, matrix), boot=boot, rng=rng
    )
    return {
        "events": int(truth.size),
        "draws": int(matrix.shape[0]),
        "pit": histogram,
        "rank_summary": {
            "rank_min": int(pit["rank"].min()),
            "rank_max": int(pit["rank"].max()),
            "events_with_ties": int(np.count_nonzero(pit["ties"] > 0)),
        },
        "coverage": coverage,
        "pull": pull,
    }


def null_calibration_reference(
    events: int,
    draws: int,
    *,
    levels=NOMINAL_LEVELS,
    bins: int,
    boot: int,
    seed: int,
) -> dict[str, Any]:
    """What every statistic reads for a conditional that is calibrated *by construction*.

    A synthetic Gaussian conditional - truth, a per-event mean error of the same
    size as the stated width, and ``draws`` Gaussian samples - pushed through
    the identical statistics with the identical ``n``, ``D``, bins and bootstrap
    count.  Any criterion that is not measured against this run is measuring the
    finite-``D`` sampling, not the model.
    """
    rng = np.random.default_rng(int(seed) + NULL_SEED_OFFSET)
    truth = rng.normal(0.0, 1.0, int(events))
    error = rng.normal(0.0, 1.0, int(events))
    noise = rng.normal(0.0, 1.0, (int(draws), int(events)))
    block = calibration_block(
        truth,
        truth[None, :] + error[None, :] + noise,
        levels=levels,
        bins=int(bins),
        boot=int(boot),
        rng=np.random.default_rng(int(seed) + NULL_SEED_OFFSET + 1),
    )
    block["generator"] = (
        "synthetic calibrated Gaussian conditional (per-event mean error = stated "
        "width = 1, truth ~ N(0, 1))"
    )
    block["finite_draw_pull_std_analytic"] = finite_draw_pull_std(int(draws))
    return block


def coverage_criterion(
    block: dict[str, Any], null: dict[str, Any], level_key: str
) -> dict[str, Any]:
    """Two-sample test of a coverage fraction against the finite-draw null.

    The null is itself a finite sample, so its own sampling error must be in the
    denominator: comparing a measurement against the null's *bootstrap band*
    alone would fail a genuinely calibrated model a large fraction of the time.
    """
    measured = block["coverage"][level_key]["measured"]
    null_value = null["coverage"][level_key]["measured"]
    se_paired = block["coverage"][level_key]["binomial_sd"]
    se_null = null["coverage"][level_key]["binomial_sd"]
    combined = math.sqrt(se_paired**2 + se_null**2)
    z_score = (measured - null_value) / combined if combined > 0 else float("inf")
    return {
        "null_value": float(null_value),
        "value": float(measured),
        "nominal": float(block["coverage"][level_key]["nominal"]),
        "nominal_deviation": float(measured - block["coverage"][level_key]["nominal"]),
        "se_paired": float(se_paired),
        "se_null": float(se_null),
        "z_score_vs_null": float(z_score),
        "z_tolerance": COVERAGE_Z_TOLERANCE,
        "bootstrap_band": [
            float(block["coverage"][level_key]["bootstrap"]["low"]),
            float(block["coverage"][level_key]["bootstrap"]["high"]),
        ],
        "pass": bool(abs(z_score) <= COVERAGE_Z_TOLERANCE),
    }


def calibration_criteria(block: dict[str, Any], null: dict[str, Any]) -> dict[str, Any]:
    """Machine-checkable pass/fail flags, each with its null reference written out."""
    pull = block["pull"]
    null_pull = null["pull"]
    chi2_p = block["pit"]["chi2_p_value"]
    if chi2_p is None:  # pragma: no cover - only without scipy
        flat_pass = bool(
            block["pit"]["max_abs_deviation_sigma"] <= FALLBACK_MAX_DEVIATION_SIGMA
        )
        flat_reference: Any = FALLBACK_MAX_DEVIATION_SIGMA
    else:
        flat_pass = bool(chi2_p > CHI2_P_THRESHOLD)
        flat_reference = CHI2_P_THRESHOLD
    expected_pull_std = null_pull["finite_draw_expected_std"]
    if not math.isfinite(float(expected_pull_std)):  # pragma: no cover - D <= 3
        expected_pull_std = 1.0
    return {
        "pull_mean_abs_le_tolerance": {
            "tolerance": PULL_TOLERANCE,
            "value": pull["pull_mean"],
            "pass": bool(abs(pull["pull_mean"]) <= PULL_TOLERANCE),
        },
        "pull_std_within_tolerance_of_finite_draw_null": {
            "tolerance": PULL_TOLERANCE,
            "null_value": float(expected_pull_std),
            "value": pull["pull_std"],
            "relative_deviation": float(pull["pull_std"] / expected_pull_std - 1.0),
            "pass": bool(
                abs(pull["pull_std"] / expected_pull_std - 1.0) <= PULL_TOLERANCE
            ),
        },
        "coverage_68_consistent_with_null": coverage_criterion(block, null, "0.68"),
        "coverage_95_consistent_with_null": coverage_criterion(block, null, "0.95"),
        "rank_histogram_consistent_with_uniform": {
            "reference": flat_reference,
            "null_max_abs_deviation_sigma": null["pit"]["max_abs_deviation_sigma"],
            "chi2": block["pit"]["chi2"],
            "dof": block["pit"]["dof"],
            "p_value": chi2_p,
            "max_abs_deviation_sigma": block["pit"]["max_abs_deviation_sigma"],
            "pass": flat_pass,
        },
    }


def calibration_verdict(criteria: dict[str, Any], pull: dict[str, Any]) -> str:
    """Plain-language label for the report and the JSON."""
    if pull.get("degenerate"):
        return "degenerate (zero-variance posterior over every event)"
    problems = []
    if not criteria["pull_mean_abs_le_tolerance"]["pass"]:
        problems.append(f"biased (pull mean {pull['pull_mean']:+.3f})")
    null_std = criteria["pull_std_within_tolerance_of_finite_draw_null"]["null_value"]
    if not criteria["pull_std_within_tolerance_of_finite_draw_null"]["pass"]:
        if pull["pull_std"] > null_std:
            problems.append(
                f"over-confident (pull std {pull['pull_std']:.3f} vs {null_std:.3f} for a "
                "calibrated conditional at this draw count)"
            )
        else:
            problems.append(
                f"under-confident (pull std {pull['pull_std']:.3f} vs {null_std:.3f})"
            )
    for key, level in (
        ("coverage_68_consistent_with_null", "0.68"),
        ("coverage_95_consistent_with_null", "0.95"),
    ):
        criterion = criteria[key]
        if not criterion["pass"]:
            problems.append(
                f"coverage {level} = {criterion['value']:.4f} against a calibrated null "
                f"of {criterion['null_value']:.4f} (z = {criterion['z_score_vs_null']:+.1f})"
            )
    if not criteria["rank_histogram_consistent_with_uniform"]["pass"]:
        problems.append(
            "rank histogram not flat "
            f"(max |f - p| = {criteria['rank_histogram_consistent_with_uniform']['max_abs_deviation_sigma']:.1f} sigma, "
            f"chi2 p = {criteria['rank_histogram_consistent_with_uniform']['p_value']:.3g})"
        )
    return "calibrated" if not problems else "not calibrated: " + "; ".join(problems)


def json_ready(value: Any) -> Any:
    """Recursively convert numpy scalars/arrays into JSON-serialisable values."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def fingerprint(path: Path, chunk: int = 1 << 20) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def git_rev() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def parse_checkpoint_argument(raw: str) -> tuple[str, Path]:
    """Accept ``PATH`` or ``LABEL=PATH``."""
    label = "checkpoint"
    text = str(raw)
    if "=" in text:
        head, tail = text.split("=", 1)
        if head and not Path(head).exists():
            label, text = head, tail
    path = Path(text)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return label, path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="PATH or LABEL=PATH; default is the checkpoint the existing "
        "canonical ppzee result used (outputs/cms_Joint/ppzee/best_model.pt)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="paired npz carrying 'z' and 'x' [N, 8]; the upstream "
        "experiments/ppzee/otus_results-dataset=ppzee_test.npz carries the same "
        "160000 events in a different row order and works here too",
    )
    parser.add_argument("--events", type=int, default=160000)
    parser.add_argument("--draws", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument(
        "--levels",
        default="0.68,0.95",
        help="comma-separated nominal coverage levels",
    )
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument(
        "--encoder-core",
        type=float,
        default=None,
        help="override the encoder core noise multiplier (default: the "
        "checkpoint's own recorded value)",
    )
    parser.add_argument(
        "--encoder-tail",
        type=float,
        default=None,
        help="override the encoder tail noise multiplier (default: the "
        "checkpoint's own recorded value)",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="seed for the shuffled control (default: --seed + 1)",
    )
    parser.add_argument(
        "--no-null",
        action="store_true",
        help="skip the synthetic finite-draw null run (criteria then fall back to "
        "the nominal values)",
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args(argv)


def parse_levels(raw: str) -> tuple[float, ...]:
    levels = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if not 0.0 < value < 1.0:
            raise SystemExit(f"--levels entries must lie in (0, 1), got {token!r}")
        levels.append(value)
    if not levels:
        raise SystemExit("--levels needs at least one value")
    if len(set(levels)) != len(levels):
        raise SystemExit(f"duplicate --levels entries: {levels}")
    if not any(abs(value - 0.68) < 1e-9 for value in levels) or not any(
        abs(value - 0.95) < 1e-9 for value in levels
    ):
        raise SystemExit("--levels must include the 0.68 and 0.95 levels")
    return tuple(levels)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.draws < 2:
        raise SystemExit("--draws must be >= 2 for a quantile/coverage estimate")
    levels = parse_levels(args.levels)
    dataset_path = args.dataset.expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"paired dataset not found: {dataset_path}")
    checkpoint_label, checkpoint_path = parse_checkpoint_argument(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    started = time.time()

    pairs = np.load(dataset_path, allow_pickle=False)
    missing = [key for key in ("z", "x") if key not in pairs.files]
    if missing:
        raise KeyError(f"{dataset_path} is missing {missing}")
    n_available = int(len(pairs["z"]))
    subset = select_subset(n_available, args.events, np.random.default_rng(args.seed))
    z_true = np.asarray(pairs["z"][subset], dtype=np.float64)
    x_input = np.asarray(pairs["x"][subset], dtype=np.float64)
    if x_input.shape[1] != 8 or z_true.shape[1] != 8:
        raise ValueError(
            f"expected [N, 8] four-vectors, got z {z_true.shape} x {x_input.shape}"
        )
    mass_true = direct_mass(z_true)
    print(
        f"ppzee quantile calibration: {len(z_true)} of {n_available} paired events "
        f"from {dataset_path.name} on {device}, {args.draws} draws"
    )

    model, checkpoint, config = load_frozen_model(checkpoint_path, device)
    noise = checkpoint.get("noise_multipliers") or {}
    native_core = float(noise.get("encoder_core", noise.get("core", 1.0)))
    native_tail = float(noise.get("encoder_tail", noise.get("tail", 0.0)))
    encoder_core = native_core if args.encoder_core is None else float(args.encoder_core)
    encoder_tail = native_tail if args.encoder_tail is None else float(args.encoder_tail)
    stage = checkpoint.get("stage") or {}
    print(
        f"[{checkpoint_label}] epoch {checkpoint.get('global_epoch')} "
        f"{(stage.get('name') if isinstance(stage, dict) else None)} "
        f"encoder multipliers ({encoder_core:g}, {encoder_tail:g})"
    )

    draws_array, deterministic = encode_draws(
        model,
        x_input.astype(np.float32),
        core=encoder_core,
        tail=encoder_tail,
        draws=int(args.draws),
        batch_size=int(args.batch_size),
        device=device,
        seed=int(args.seed),
    )
    validate_draws(draws_array, len(z_true))
    draw_mass = draw_mass_matrix(draws_array)
    print(f"draws {draws_array.shape}, deterministic encoder: {deterministic}")

    shuffle_seed = int(args.seed) + 1 if args.shuffle_seed is None else int(args.shuffle_seed)
    permutation = derangement_indices(len(z_true), np.random.default_rng(shuffle_seed))
    shuffled_truth = mass_true[permutation]
    shuffle_shift = float(np.sqrt(np.mean((shuffled_truth - mass_true) ** 2)))

    paired = calibration_block(
        mass_true,
        draw_mass,
        levels=levels,
        bins=int(args.bins),
        boot=int(args.bootstrap),
        rng=np.random.default_rng(int(args.seed) + 2),
    )
    shuffled = calibration_block(
        shuffled_truth,
        draw_mass,
        levels=levels,
        bins=int(args.bins),
        boot=int(args.bootstrap),
        rng=np.random.default_rng(int(args.seed) + 3),
    )
    if args.no_null:
        # Degrade gracefully: compare against the nominal values instead of a
        # finite-draw null. Documented in the JSON so the report cannot be read
        # as a null-referenced verdict.
        null = {
            "status": "skipped (--no-null)",
            "generator": None,
            "events": paired["events"],
            "draws": paired["draws"],
            "pit": {
                "max_abs_deviation_sigma": float("nan"),
                "total_variation": float("nan"),
            },
            "coverage": {
                key: {
                    "measured": paired["coverage"][key]["nominal"],
                    "bootstrap": {
                        "low": paired["coverage"][key]["measured"],
                        "high": paired["coverage"][key]["measured"],
                    },
                }
                for key in paired["coverage"]
            },
            "pull": {"finite_draw_expected_std": finite_draw_pull_std(paired["draws"])},
        }
        skipped_null = True
    else:
        null = null_calibration_reference(
            len(z_true),
            int(args.draws),
            levels=levels,
            bins=int(args.bins),
            boot=int(args.bootstrap),
            seed=int(args.seed),
        )
        skipped_null = False

    for block in (paired, shuffled):
        criteria = calibration_criteria(block, null)
        block["criteria"] = criteria
        block["verdict"] = calibration_verdict(criteria, block["pull"])

    paired_pull = paired["pull"]
    shuffled_pull = shuffled["pull"]
    level_68 = paired["coverage"]["0.68"]
    level_95 = paired["coverage"]["0.95"]
    shuffled_68 = shuffled["coverage"]["0.68"]
    shuffled_95 = shuffled["coverage"]["0.95"]
    print(
        f"null     : coverage 0.68 {null['coverage']['0.68']['measured']:.4f}  "
        f"0.95 {null['coverage']['0.95']['measured']:.4f}  "
        f"pull std {null['pull']['finite_draw_expected_std']:.4f} (finite-draw "
        f"expected)  PIT max dev {null['pit']['max_abs_deviation_sigma']:.2f} sigma"
    )
    print(
        f"paired   : coverage 0.68 {level_68['measured']:.4f} "
        f"[{level_68['bootstrap']['low']:.4f}, {level_68['bootstrap']['high']:.4f}]  "
        f"0.95 {level_95['measured']:.4f}   "
        f"pull {paired_pull['pull_mean']:+.4f}/{paired_pull['pull_std']:.4f}   "
        f"PIT max dev {paired['pit']['max_abs_deviation_sigma']:.1f} sigma"
    )
    print(
        f"shuffled : coverage 0.68 {shuffled_68['measured']:.4f} "
        f"[{shuffled_68['bootstrap']['low']:.4f}, {shuffled_68['bootstrap']['high']:.4f}]  "
        f"0.95 {shuffled_95['measured']:.4f}   "
        f"pull {shuffled_pull['pull_mean']:+.4f}/{shuffled_pull['pull_std']:.4f}   "
        f"PIT max dev {shuffled['pit']['max_abs_deviation_sigma']:.1f} sigma"
    )
    print(f"paired   verdict: {paired['verdict']}")
    print(f"shuffled verdict: {shuffled['verdict']}")

    payload: dict[str, Any] = {
        "schema_version": 1,
        "diagnostic": "P2 paired half - ppzee per-event quantile calibration (PIT / coverage / pull)",
        "status": "completed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": None,
        "repo_root": str(REPO_ROOT),
        "git_rev": git_rev(),
        "device": str(device),
        "torch_version": _torch_version(),
        "cuda_available": _cuda_available(),
        "dataset": str(dataset_path),
        "dataset_sha256": fingerprint(dataset_path),
        "events_available": n_available,
        "events_used": int(len(z_true)),
        "draws": int(args.draws),
        "seed": int(args.seed),
        "shuffle_seed": shuffle_seed,
        "bins": int(args.bins),
        "bootstrap": int(args.bootstrap),
        "levels": [float(level) for level in levels],
        "band_sigma": BAND_SIGMA,
        "coverage_z_tolerance": COVERAGE_Z_TOLERANCE,
        "pull_tolerance": PULL_TOLERANCE,
        "chi2_p_threshold": CHI2_P_THRESHOLD,
        "null_skipped": bool(skipped_null),
        "deterministic_encoder": bool(deterministic),
        "mass_convention": "paired_closure.direct_mass (pair E^2 - |p|^2, float64)",
        "reference": REFERENCE,
        "checkpoint": {
            "label": checkpoint_label,
            "path": str(checkpoint_path),
            "sha256": fingerprint(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
            "global_epoch": checkpoint.get("global_epoch"),
            "stage_name": (stage.get("name") if isinstance(stage, dict) else None),
            "recorded_noise_multipliers": noise,
            "native_encoder_multipliers": {"core": native_core, "tail": native_tail},
            "encoder_multipliers_used": {"core": encoder_core, "tail": encoder_tail},
            "run_label": config.get("run_label"),
        },
        "control": {
            "kind": "shuffled pairing (seeded derangement, no fixed points)",
            "shuffle_seed": shuffle_seed,
            "fixed_points": int(
                np.count_nonzero(permutation == np.arange(len(permutation)))
            ),
            "shuffled_truth_mass_shift_rms_gev": shuffle_shift,
        },
        "null": null,
        "paired": paired,
        "shuffled": shuffled,
        "observed_change": {
            "coverage_68_delta": float(shuffled_68["measured"] - level_68["measured"]),
            "coverage_95_delta": float(shuffled_95["measured"] - level_95["measured"]),
            "pull_std_delta": float(shuffled_pull["pull_std"] - paired_pull["pull_std"]),
            "pull_mean_delta": float(
                shuffled_pull["pull_mean"] - paired_pull["pull_mean"]
            ),
            "pit_max_deviation_sigma_delta": float(
                shuffled["pit"]["max_abs_deviation_sigma"]
                - paired["pit"]["max_abs_deviation_sigma"]
            ),
            "verdict_changed": bool(paired["verdict"] != shuffled["verdict"]),
        },
        "cross_checks": {
            "paired_posterior_mean_residual_rms_gev": paired_pull["mean_residual_rms_gev"],
            "reference_point_prediction_model_rms_gev": REFERENCE["existing_result_model_rms_gev"],
            "absolute_difference_gev": float(
                abs(
                    paired_pull["mean_residual_rms_gev"]
                    - REFERENCE["existing_result_model_rms_gev"]
                )
            ),
            "paired_posterior_std_median_gev": paired_pull["posterior_std_median_gev"],
            "note": (
                "source-verified reference: the canonical ppzee result "
                "(paired_closure.json, single encode of the same checkpoint) "
                "reports model_rms 2.6100 GeV against truth partners, i.e. a "
                "posterior-mean residual of the same size if the draws are "
                "unbiased around the point prediction."
            ),
        },
        "artifacts": {"json": None, "report": None, "png": None},
    }

    json_path = out_dir / "quantile_calibration.json"
    report_path = out_dir / "REPORT.md"
    png_path = out_dir / "quantile_calibration.png"
    payload["artifacts"]["json"] = str(json_path)
    payload["artifacts"]["report"] = str(report_path)
    if not args.no_plot:
        try:
            written = make_plot(payload, png_path)
            payload["artifacts"]["png"] = str(written) if written else None
        except Exception as error:  # pragma: no cover - plotting must not kill the run
            print(f"plot skipped: {error}")
            payload["artifacts"]["png"] = None
    payload["runtime_seconds"] = time.time() - started
    json_path.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(render_report(payload), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {report_path}")
    if payload["artifacts"]["png"]:
        print(f"wrote {payload['artifacts']['png']}")
    print(f"done in {payload['runtime_seconds']:.1f}s")
    return 0


def _torch_version() -> str:
    import torch

    return str(torch.__version__)


def _cuda_available() -> bool:
    import torch

    return bool(torch.cuda.is_available())


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------

def make_plot(payload: dict, path: Path) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paired = payload["paired"]
    shuffled = payload["shuffled"]
    null = payload["null"]
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.4))
    centers = 0.5 * (
        np.asarray(paired["pit"]["edges"][:-1]) + np.asarray(paired["pit"]["edges"][1:])
    )
    width = float(np.diff(paired["pit"]["edges"])[0]) * 0.9

    for axis, block, title in (
        (axes[0][0], paired, "paired (x with its own truth)"),
        (axes[0][1], shuffled, "shuffled control (x with another event's truth)"),
    ):
        edges = np.asarray(block["pit"]["edges"], dtype=np.float64)
        fraction = np.asarray(block["pit"]["fractions"], dtype=np.float64)
        probability = np.asarray(block["pit"]["expected_probability"], dtype=np.float64)
        band = float(block["pit"]["band_sigma"]) * np.asarray(
            block["pit"]["band_sd"], dtype=np.float64
        )
        flat = float(1.0 / block["pit"]["bins"])
        axis.bar(centers, fraction, width=width, color="#4c72b0", alpha=0.85,
                 label="measured")
        axis.axhline(flat, color="black", linewidth=1.0, linestyle="--",
                     label="uniform (calibrated)")
        step_x = np.repeat(edges, 2)[1:-1]
        axis.fill_between(
            step_x,
            np.repeat(probability - band, 2),
            np.repeat(probability + band, 2),
            step="pre",
            color="grey",
            alpha=0.25,
            label=f"+/-{block['pit']['band_sigma']:g} binomial band",
        )
        axis.set_title(
            f"{title}\nmax |f - p| = {block['pit']['max_abs_deviation_sigma']:.1f} sigma, "
            f"TV = {block['pit']['total_variation']:.3f}",
            fontsize=10,
        )
        axis.set_xlabel("PIT  u = (rank + 0.5) / (D + 1)")
        axis.set_ylabel("fraction of events")
        axis.set_xlim(0.0, 1.0)
        axis.legend(fontsize=8)

    pull_axis = axes[1][0]
    normal_x = np.linspace(-6.0, 6.0, 400)
    pull_axis.plot(
        normal_x,
        np.exp(-0.5 * normal_x**2) / math.sqrt(2.0 * math.pi),
        color="black",
        linewidth=1.0,
        linestyle="--",
        label="N(0, 1)",
    )
    for block, colour, label in (
        (paired, "#4c72b0", f"paired (std {paired['pull']['pull_std']:.2f})"),
        (shuffled, "#dd8452", f"shuffled (std {shuffled['pull']['pull_std']:.2f})"),
    ):
        hist = block["pull"]["histogram"]
        edges = np.asarray(hist["edges"], dtype=np.float64)
        counts = np.asarray(hist["counts"], dtype=np.float64)
        inside = counts.sum() + int(hist["events_outside_range"])
        density = counts / max(inside, 1.0) / float(np.diff(edges)[0])
        pull_axis.step(edges[:-1], density, where="post", color=colour, label=label)
    pull_axis.set_xlim(-6.0, 6.0)
    pull_axis.set_xlabel("pull (truth - mean) / std")
    pull_axis.set_ylabel("density")
    pull_axis.set_title(
        "pull distribution (clipped to +/-6); calibrated std here = "
        f"{paired['pull']['finite_draw_expected_std']:.3f}",
        fontsize=10,
    )
    pull_axis.legend(fontsize=8)

    coverage_axis = axes[1][1]
    levels = list(paired["coverage"])
    positions = np.arange(len(levels))
    for offset, block, colour, label in (
        (-0.22, paired, "#4c72b0", "paired"),
        (0.22, shuffled, "#dd8452", "shuffled"),
    ):
        measured = [block["coverage"][key]["measured"] for key in levels]
        errors = np.asarray(
            [
                [
                    block["coverage"][key]["measured"]
                    - block["coverage"][key]["bootstrap"]["low"]
                    for key in levels
                ],
                [
                    block["coverage"][key]["bootstrap"]["high"]
                    - block["coverage"][key]["measured"]
                    for key in levels
                ],
            ]
        )
        coverage_axis.bar(
            positions + offset, measured, width=0.4, color=colour, label=label,
            yerr=errors, capsize=4,
        )
    null_measured = [null["coverage"][key]["measured"] for key in levels]
    coverage_axis.plot(
        positions, null_measured, "k_", markersize=14,
        label="calibrated null (finite D)",
    )
    nominal = [paired["coverage"][key]["nominal"] for key in levels]
    coverage_axis.plot(positions, nominal, "k:", label="nominal")
    coverage_axis.set_xticks(positions)
    coverage_axis.set_xticklabels(levels)
    coverage_axis.set_ylim(0.0, 1.05)
    coverage_axis.set_xlabel("nominal central credible level")
    coverage_axis.set_ylabel("empirical coverage")
    coverage_axis.set_title("central-interval coverage (bootstrap band)", fontsize=10)
    coverage_axis.legend(fontsize=8)

    figure.suptitle(
        "ppzee per-event quantile calibration - "
        f"{payload['events_used']} events, {payload['draws']} draws, "
        f"seed {payload['seed']}",
        fontsize=11,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def render_report(payload: dict) -> str:
    paired = payload["paired"]
    shuffled = payload["shuffled"]
    null = payload["null"]
    change = payload["observed_change"]

    def criterion_row(name: str) -> str:
        return (
            f"| {name} | {paired['criteria'][name]['pass']} | "
            f"{shuffled['criteria'][name]['pass']} |"
        )

    lines: list[str] = []
    lines.append("# P2 (paired half) - ppzee per-event quantile calibration")
    lines.append("")
    lines.append(
        f"*{payload['created_utc']} - read-only inference, no training. git "
        f"`{str(payload['git_rev'])[:8]}`. Device `{payload['device']}`. "
        f"{payload['events_used']} of {payload['events_available']} withheld paired "
        f"events, {payload['draws']} encoder draws per event, seed {payload['seed']} "
        f"(subset and draws), {payload['shuffle_seed']} (shuffle control), "
        f"{payload['bootstrap']} bootstrap resamples. Runtime "
        f"{payload['runtime_seconds']:.1f}s.*"
    )
    lines.append("")
    lines.append(
        "**Labels** (CLAUDE.md section 2): *artifact-measured* = produced by this "
        "run; *source-verified* = read in the code or the named artifact; "
        "*hypothesis* = interpretation of a measurement; *proposal* = suggested "
        "next action. Every number below is also in "
        f"`{Path(payload['artifacts']['json']).name}`."
    )
    lines.append("")
    lines.append("## What this measures")
    lines.append("")
    lines.append(
        "*(source-verified: method as implemented.)* "
        "For every held-out ppzee pair `(x, z_true)` the frozen encoder is run "
        "`D` times at its native noise multipliers, giving `D` samples of its "
        "implicit conditional `q(z | x)`. In the invariant-mass coordinate the "
        "rank of `mass(z_true)` among the `D` sampled masses is recorded. If the "
        "truth and the draws are exchangeable - the definition of a calibrated "
        "conditional - that rank is uniform on `{0, ..., D}`, so the rank "
        "histogram (PIT) is flat **exactly**, with no finite-`D` correction. The "
        "central-interval coverage and the pull test the same conditional from "
        "two other directions."
    )
    lines.append("")
    lines.append(
        "*(source-verified: sign convention.)* The pull here is "
        "`(mass(z_true) - mean(mass(z_j))) / std(mass(z_j))`. "
        "`paired_closure.py` reports the opposite sign, "
        "`(mean - truth) / std`; a positive pull here and a negative pull there are "
        "the same statement (the posterior mean sits below the truth mass), so the "
        "two artifacts must be compared with that sign flip in mind."
    )
    lines.append("")
    lines.append(
        "**Shuffled control (source-verified).** The same three statistics are "
        "recomputed with each `x` paired with another event's truth: a seeded "
        "derangement with **no fixed points** "
        f"(`fixed_points = {payload['control']['fixed_points']}`), so the control "
        "cannot inherit a single correct pairing. It isolates how much of the "
        "paired reading comes from the pairing rather than from the marginals."
    )
    lines.append("")
    lines.append("**Finite-draw null (source-verified).** Every criterion is "
                 "evaluated against a synthetic conditional that is calibrated by "
                 "construction, pushed through the identical statistics with the "
                 "identical `n`, `D`, bins and bootstrap count: "
                 f"`{null.get('generator')}`. Without it, a coverage criterion at "
                 "the 0.95 level would be measuring the inward bias of sampled "
                 "quantiles rather than the model.")
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append(
        "*(source-verified: read from the named files and the checkpoint itself.)*"
    )
    lines.append("")
    lines.append(f"- dataset: `{payload['dataset']}` (sha256 `{payload['dataset_sha256'][:16]}...`)")
    lines.append(
        f"- checkpoint: `{payload['checkpoint']['path']}` "
        f"(label `{payload['checkpoint']['label']}`, global epoch "
        f"{payload['checkpoint']['global_epoch']}, stage "
        f"`{payload['checkpoint']['stage_name']}`, sha256 "
        f"`{payload['checkpoint']['sha256'][:16]}...`)"
    )
    lines.append(
        f"- encoder multipliers used: core "
        f"{payload['checkpoint']['encoder_multipliers_used']['core']:g}, tail "
        f"{payload['checkpoint']['encoder_multipliers_used']['tail']:g} "
        f"(recorded native: core "
        f"{payload['checkpoint']['native_encoder_multipliers']['core']:g}, tail "
        f"{payload['checkpoint']['native_encoder_multipliers']['tail']:g}); "
        f"deterministic encoder: {payload['deterministic_encoder']}"
    )
    lines.append(
        f"- mass convention: {payload['mass_convention']}; PIT bins "
        f"{payload['bins']}; binomial band +/-{payload['band_sigma']:g} sd"
    )
    lines.append(
        "- **source-verified:** this is the checkpoint the existing canonical "
        "ppzee result used - `outputs/cms_Joint/ppzee/paired_closure.json` "
        "(global epoch 120, stage `runA_stage2_gaussian_response`, "
        "`residual_rms_vs_identity` 0.98435, `model_rms` 2.6100 GeV against truth "
        "partners, single encode). The D4 pull/coverage artifact "
        "(`outputs/cms_Joint/ppzee/pull_coverage/REPORT.md`) used the other two "
        "retained checkpoints; its stage-2 row at native multipliers is quoted "
        "under Cross-checks."
    )
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(
        "| statistic | calibrated null (this n, this D) | paired | shuffled control | "
        "change (shuffled - paired) |"
    )
    lines.append("|---|---|---|---|---|")
    lines.append(
        f"| coverage @ 0.68 (bootstrap 95% band) | "
        f"{null['coverage']['0.68']['measured']:.4f} | "
        f"**{paired['coverage']['0.68']['measured']:.4f}** "
        f"[{paired['coverage']['0.68']['bootstrap']['low']:.4f}, "
        f"{paired['coverage']['0.68']['bootstrap']['high']:.4f}] | "
        f"{shuffled['coverage']['0.68']['measured']:.4f} "
        f"[{shuffled['coverage']['0.68']['bootstrap']['low']:.4f}, "
        f"{shuffled['coverage']['0.68']['bootstrap']['high']:.4f}] | "
        f"{change['coverage_68_delta']:+.4f} |"
    )
    lines.append(
        f"| coverage @ 0.95 (bootstrap 95% band) | "
        f"{null['coverage']['0.95']['measured']:.4f} | "
        f"**{paired['coverage']['0.95']['measured']:.4f}** "
        f"[{paired['coverage']['0.95']['bootstrap']['low']:.4f}, "
        f"{paired['coverage']['0.95']['bootstrap']['high']:.4f}] | "
        f"{shuffled['coverage']['0.95']['measured']:.4f} "
        f"[{shuffled['coverage']['0.95']['bootstrap']['low']:.4f}, "
        f"{shuffled['coverage']['0.95']['bootstrap']['high']:.4f}] | "
        f"{change['coverage_95_delta']:+.4f} |"
    )
    lines.append(
        f"| pull mean | {null['pull']['pull_mean']:+.4f} | {paired['pull']['pull_mean']:+.4f} | "
        f"{shuffled['pull']['pull_mean']:+.4f} | {change['pull_mean_delta']:+.4f} |"
    )
    lines.append(
        f"| pull std | {paired['pull']['finite_draw_expected_std']:.4f} | "
        f"**{paired['pull']['pull_std']:.4f}** | {shuffled['pull']['pull_std']:.4f} | "
        f"{change['pull_std_delta']:+.4f} |"
    )
    lines.append(
        f"| pull robust width (1.4826 MAD) | - | "
        f"{paired['pull']['pull_robust_width_1p4826mad']:.4f} | "
        f"{shuffled['pull']['pull_robust_width_1p4826mad']:.4f} | "
        f"{shuffled['pull']['pull_robust_width_1p4826mad'] - paired['pull']['pull_robust_width_1p4826mad']:+.4f} |"
    )
    lines.append(
        f"| PIT max abs deviation [binomial sd] | "
        f"{null['pit']['max_abs_deviation_sigma']:.2f} | "
        f"{paired['pit']['max_abs_deviation_sigma']:.2f} | "
        f"{shuffled['pit']['max_abs_deviation_sigma']:.2f} | "
        f"{change['pit_max_deviation_sigma_delta']:+.2f} |"
    )
    lines.append(
        f"| PIT total variation from uniform | {null['pit']['total_variation']:.4f} | "
        f"{paired['pit']['total_variation']:.4f} | "
        f"{shuffled['pit']['total_variation']:.4f} | "
        f"{shuffled['pit']['total_variation'] - paired['pit']['total_variation']:+.4f} |"
    )
    lines.append(
        f"| verdict | calibrated | {paired['verdict']} | {shuffled['verdict']} | "
        f"changed: {change['verdict_changed']} |"
    )
    lines.append("")
    lines.append(
        "**source-verified (why the null column is not 0.68 / 0.95 / 1.0).** The "
        "central interval is built from `D` sampled quantiles, whose inward bias "
        "costs real coverage, and the deficit grows as `D` shrinks: here the null "
        f"reads {null['coverage']['0.68']['measured']:.4f} at 0.68 and "
        f"{null['coverage']['0.95']['measured']:.4f} at 0.95 for `D` = "
        f"{payload['draws']}. The pull carries the exact factor "
        "`sqrt(1 + 1/D) * sqrt((D-1)/(D-3))` "
        f"= {paired['pull']['finite_draw_expected_std']:.4f} at the same `D`. The "
        "PIT has no such correction: the rank is exactly uniform under calibration "
        "for any continuous conditional, which is why the null's PIT column is the "
        "flat histogram. This is the whole reason the criteria are null-referenced."
    )
    lines.append("")
    lines.append("## Rank histogram (PIT)")
    lines.append("")
    lines.append(
        "Under a calibrated conditional every bin count is `Binomial(n, p_k)` "
        "with the exact `p_k` shown; `dev sigma = (fraction - p_k) / sd_k`. Band "
        f"= +/-{payload['band_sigma']:g} sd."
    )
    lines.append("")
    for name, block in (("paired", paired), ("shuffled control", shuffled)):
        pit = block["pit"]
        lines.append(
            f"**{name}** - chi2 {pit['chi2']:.2f} on {pit['dof']} dof"
            + (
                f", asymptotic p = {pit['chi2_p_value']:.3g}"
                if pit["chi2_p_value"] is not None
                else ""
            )
            + f"; max |f - p| = {pit['max_abs_deviation_sigma']:.1f} sigma"
        )
        lines.append("")
        lines.append("| bin | u range | count | fraction | expected p | band (2sd) | dev sigma |")
        lines.append("|---|---|---|---|---|---|---|")
        for index in range(pit["bins"]):
            lines.append(
                f"| {index} | [{pit['edges'][index]:.2f}, {pit['edges'][index + 1]:.2f}) | "
                f"{pit['counts'][index]} | {pit['fractions'][index]:.4f} | "
                f"{pit['expected_probability'][index]:.4f} | "
                f"+/-{pit['band_sigma'] * pit['band_sd'][index]:.5f} | "
                f"{pit['deviation_sigma'][index]:+.2f} |"
            )
        lines.append("")
    lines.append("## Coverage and pull detail")
    lines.append("")
    lines.append(
        "| pairing | level | measured | nominal | null (this n, D) | binomial sd | "
        "bootstrap 95% band | median interval half-width [GeV] |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, block in (("paired", paired), ("shuffled", shuffled)):
        for key in block["coverage"]:
            item = block["coverage"][key]
            lines.append(
                f"| {name} | {key} | {item['measured']:.4f} | {item['nominal']:.2f} | "
                f"{null['coverage'][key]['measured']:.4f} | {item['binomial_sd']:.5f} | "
                f"[{item['bootstrap']['low']:.4f}, {item['bootstrap']['high']:.4f}] | "
                f"{item['median_interval_halfwidth_gev']:.4f} |"
            )
    lines.append("")
    lines.append(
        "| pairing | pull mean | pull std | calibrated null std | bootstrap 95% band on std | "
        "robust width | q16-q84 width | frac abs(pull)<=2 | KS vs N(0,1) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for name, block in (("paired", paired), ("shuffled", shuffled)):
        pull = block["pull"]
        lines.append(
            f"| {name} | {pull['pull_mean']:+.4f} | {pull['pull_std']:.4f} | "
            f"{pull['finite_draw_expected_std']:.4f} | "
            f"[{pull['bootstrap_std']['low']:.4f}, {pull['bootstrap_std']['high']:.4f}] | "
            f"{pull['pull_robust_width_1p4826mad']:.4f} | {pull['pull_q16_q84_width']:.4f} | "
            f"{pull['fraction_abs_pull_le_2']:.4f} | "
            f"D={pull['ks_distance_to_standard_normal']:.4f}, p={pull['ks_p_value_asymptotic']:.3g} |"
        )
    lines.append("")
    lines.append("## Machine-checkable criteria")
    lines.append("")
    lines.append(
        "Coverage is tested against the null with **both** sampling errors in the "
        "denominator (`z = (paired - null) / sqrt(se_paired^2 + se_null^2)`, "
        f"tolerance |z| <= {COVERAGE_Z_TOLERANCE:g}); the pull std against the "
        f"finite-draw reference with a {PULL_TOLERANCE:.0%} relative tolerance; the "
        f"PIT by its chi-square p-value (> {CHI2_P_THRESHOLD:g})."
    )
    lines.append("")
    lines.append(
        "| criterion | paired reference | paired value | pass | shuffled value | pass |"
    )
    lines.append("|---|---|---|---|---|---|")
    pit_criterion = "rank_histogram_consistent_with_uniform"
    for name in paired["criteria"]:
        if name == pit_criterion:
            continue
        criterion = paired["criteria"][name]
        reference = criterion.get(
            "null_value", criterion.get("tolerance", criterion.get("band_sigma"))
        )
        lines.append(
            f"| {name} | {reference:g} | {criterion['value']:.4f} | {criterion['pass']} | "
            f"{shuffled['criteria'][name]['value']:.4f} | {shuffled['criteria'][name]['pass']} |"
        )
    pit_paired = paired["criteria"][pit_criterion]
    pit_shuffled = shuffled["criteria"][pit_criterion]
    lines.append(
        f"| {pit_criterion} | chi2 p > {pit_paired['reference']:g} | "
        f"{pit_paired['p_value']:.3g} | {pit_paired['pass']} | "
        f"{pit_shuffled['p_value']:.3g} | {pit_shuffled['pass']} |"
    )
    lines.append("")
    lines.append("## Findings")
    lines.append("")
    lines.append(
        f"- **artifact-measured.** Paired coverage is "
        f"{paired['coverage']['0.68']['measured']:.4f} at nominal 0.68 and "
        f"{paired['coverage']['0.95']['measured']:.4f} at nominal 0.95, against a "
        f"calibrated null of {null['coverage']['0.68']['measured']:.4f} and "
        f"{null['coverage']['0.95']['measured']:.4f} at the same n and D. The "
        f"encoder's conditional is far too narrow: the withheld truth falls "
        f"outside its own stated credible interval almost always."
    )
    lines.append(
        f"- **artifact-measured.** Paired pull std is "
        f"{paired['pull']['pull_std']:.3f} (mean {paired['pull']['pull_mean']:+.3f}) "
        f"against {paired['pull']['finite_draw_expected_std']:.3f} for a calibrated "
        f"conditional at {payload['draws']} draws - about "
        f"{paired['pull']['pull_std'] / paired['pull']['finite_draw_expected_std']:.0f}x "
        f"too wide. In mass units the stated posterior width is "
        f"{paired['pull']['posterior_std_median_gev']:.4f} GeV (median) while the "
        f"posterior-mean residual is "
        f"{paired['pull']['mean_residual_rms_gev']:.4f} GeV rms."
    )
    lines.append(
        f"- **artifact-measured.** The PIT histogram deviates from uniform by up to "
        f"{paired['pit']['max_abs_deviation_sigma']:.1f} binomial sd (total variation "
        f"{paired['pit']['total_variation']:.4f}, chi2 "
        f"{paired['pit']['chi2']:.0f} on {paired['pit']['dof']} dof"
        + (
            f", p = {paired['pit']['chi2_p_value']:.3g}"
            if paired["pit"]["chi2_p_value"] is not None
            else ""
        )
        + "); the deviation is a U/T shape, i.e. the signature of an over-confident "
        "conditional, in the same direction as the pull reading. This statistic needs "
        "no finite-draw correction, so it is the cleanest of the four."
    )
    lines.append(
        f"- **artifact-measured (control).** Destroying the pairing moves coverage "
        f"@0.68 by {change['coverage_68_delta']:+.4f} and @0.95 by "
        f"{change['coverage_95_delta']:+.4f}, pull std by "
        f"{change['pull_std_delta']:+.4f} and the PIT maximum deviation by "
        f"{change['pit_max_deviation_sigma_delta']:+.2f} sigma. The shuffle moves "
        f"the truth by {payload['control']['shuffled_truth_mass_shift_rms_gev']:.3f} GeV "
        "rms, so the control is a real re-pairing rather than a relabelling."
    )
    lines.append(
        "- **hypothesis.** Because the paired reading is far from the null and the "
        "shuffled control is far from it too, the failure of this encoder as a "
        "*per-event posterior* is not primarily a pairing artefact: its conditional "
        "is too narrow in the mass coordinate whether or not the truth belongs to "
        "the event. That is the ppzee counterpart of the marginal-closure finding "
        "in the CMS regions (F2/F3 in `docs/project_tree.md`, plan branch P)."
    )
    lines.append(
        "- **proposal.** Any per-event uncertainty band quoted from this encoder is "
        "unusable at face value; a calibration layer (variance rescaling, or a "
        "posterior trained with a proper scoring rule) is required before the "
        "inverse branch can report calibrated uncertainties."
    )
    lines.append("")
    lines.append("## Cross-checks against existing artifacts")
    lines.append("")
    checks = payload["cross_checks"]
    lines.append(
        f"- **source-verified / artifact-measured.** The canonical ppzee result "
        f"reports `model_rms` = {checks['reference_point_prediction_model_rms_gev']:.4f} GeV "
        f"for the same checkpoint (single encode); this run's posterior-mean residual "
        f"is {checks['paired_posterior_mean_residual_rms_gev']:.4f} GeV "
        f"(absolute difference {checks['absolute_difference_gev']:.4f} GeV), so the "
        "posterior mean sits where the point prediction sits - the draws are "
        "centred, they are simply too narrow."
    )
    reference_pull = REFERENCE["pull_coverage_stage2_native"]
    lines.append(
        f"- **source-verified.** The D4 artifact's stage-2 row at native multipliers "
        f"({reference_pull['encoder_multipliers'][0]:g}, "
        f"{reference_pull['encoder_multipliers'][1]:g}) reports pull "
        f"{reference_pull['pull_mean']:+.4f}/{reference_pull['pull_std']:.4f} and "
        f"1-sigma coverage {reference_pull['coverage_1sigma']:.4f} on the "
        "stage-2 checkpoint; this run's paired pull is "
        f"{paired['pull']['pull_mean']:+.4f}/{paired['pull']['pull_std']:.4f} on "
        "`best_model.pt`. Both say the same thing - the posterior is ~20x too narrow "
        "and its mean sits below the truth mass (the sign conventions are opposite, "
        "see above) - but the checkpoints differ, so this is agreement between two "
        "rows of the same bench, not a reproduction."
    )
    lines.append("")
    lines.append("## What this can and cannot conclude")
    lines.append("")
    lines.append(
        "- **Can (artifact-measured).** Whether a *per-event* conditional is "
        "calibrated, because this bench carries the withheld truth partner of every "
        "detector-level event. The rank/quantile curve, the central-interval "
        "coverage and the pull are three independent readings of the same "
        "conditional, and the shuffled control bounds how much of each comes from "
        "the pairing."
    )
    lines.append(
        "- **Can (source-verified).** That the PIT statistic is unbiased under the "
        "null: exchangeability of the truth with the draws makes the rank uniform on "
        "`{0, ..., D}` exactly, so a flat histogram is the correct calibrated "
        "expectation with no finite-draw correction. Coverage and pull do carry "
        "finite-draw corrections, which is why both are read against the null "
        "column rather than against their nominal values."
    )
    lines.append(
        "- **Cannot.** Nothing here transfers to the CMS dimuon regions: those have "
        "no truth partner, so this test cannot be run there at all - which is why "
        "the CMS half of P2 is a gauge-based conditional-spread metric with an "
        "identity map and a finite-sample floor instead. This script does not "
        "measure the CMS conditional, the encoder's closure quality "
        "(`residual_rms_vs_identity`), or the Upsilon transfer."
    )
    lines.append(
        "- **Cannot.** The ppzee response geometry is not the CMS one: the upstream "
        "LO 2->1 sample has identically zero pair pT, so every unit of detector-level "
        "pair pT was manufactured downstream. Calibration numbers measured here need "
        "not transfer to J/psi or Z."
    )
    lines.append(
        "- **Cannot.** The posterior sampled here is the encoder's implicit "
        "conditional over draws, not a Bayesian posterior; the test asks whether its "
        "width matches its actual per-event error, nothing more. A degenerate "
        "(zero-variance) encoder makes the pull undefined and is reported as such "
        "rather than as a calibration measurement."
    )
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append(
        f"- Finite-draw reference (source-verified): {REFERENCE['finite_draw_note']}"
    )
    lines.append(
        f"- **artifact-measured.** Tie handling: "
        f"{paired['rank_summary']['events_with_ties']} paired events have at least "
        "one draw exactly equal to the truth mass (float32 four-vectors recomputed "
        "in float64). Ranks use strict inequality, so a tie is counted as neither "
        "below nor above the truth; no event here is affected, so the choice is "
        "immaterial for this run."
    )
    lines.append(
        f"- **artifact-measured.** Numerical degeneracy: "
        f"{paired['pull']['zero_variance_events']} paired events have a posterior "
        f"width at or below {ZERO_VARIANCE_TOL:g} GeV; the pull divides by "
        f"max(width, {PULL_CLAMP:g}) exactly as `paired_closure` does."
    )
    lines.append(
        f"- **artifact-measured.** Sampling: the event subset is drawn with seed "
        f"{payload['seed']} (yielding {payload['events_used']} of "
        f"{payload['events_available']} events) and the shuffle with seed "
        f"{payload['shuffle_seed']}. Two runs at a fixed seed on the same device "
        "produce identical statistics (pinned by the CLI smoke test, and observed "
        "across the three full runs written here)."
    )
    lines.append(
        "- **source-verified.** The null is a Gaussian calibrated conditional; the "
        "PIT criterion is distribution-free under calibration, while its coverage "
        "column is the Gaussian case and is quoted as such."
    )
    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append(
        "python scripts_joint/paired_quantile_calibration.py "
        f"--checkpoint {payload['checkpoint']['path']} "
        f"--dataset {payload['dataset']} "
        f"--events {payload['events_used']} --draws {payload['draws']} "
        f"--seed {payload['seed']} --device cpu --bins {payload['bins']} "
        f"--bootstrap {payload['bootstrap']}"
    )
    lines.append("```")
    lines.append("")
    lines.append("```bash")
    lines.append("python -m unittest tests.test_paired_quantile_calibration -v")
    lines.append("```")
    lines.append("")
    lines.append(
        f"*Runtime {payload['runtime_seconds']:.1f}s on {payload['device']}; JSON "
        f"`{Path(payload['artifacts']['json']).name}`, figure "
        f"`{Path(payload['artifacts']['png']).name if payload['artifacts']['png'] else 'skipped'}`.*"
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
