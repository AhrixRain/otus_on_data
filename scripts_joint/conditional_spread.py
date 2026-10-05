#!/usr/bin/env python
"""Conditional-spread criterion: does the model's claimed per-event spread match the data?

P2 of the approved programme (`docs/project_tree.md`, branch P). Read-only: no
training, nothing under ``data/``, ``outputs/`` or any checkpoint is modified,
every artifact goes to a new ``--output-dir``.

Why this exists
---------------
The project's central failure (``memory.md`` F1/F2, five refuted mechanisms) is
that marginal agreement does not identify where the resolution lives: a
deterministic mean map can transport the prior and reproduce any detector-level
marginal exactly, so every marginal gauge is blind to a dead noise channel. This
tool measures the object the marginal gauges cannot see, and it reports the
controls that make the measurement falsifiable:

1. **the exact split** ``Var(x) = E_z[Var(x|z)] + Var_z[E(x|z)]`` -- the
   *claim* (within-z spread from repeated decodes at a fixed truth) against the
   *mean-map* spread;
2. **the claim versus the requirement** -- the median claimed per-event scale
   against the data-driven quadrature requirement
   ``sqrt(Var(x_data) - Var(x_prior))`` for the region;
3. **coverage of the cycle residual** -- for held-out detector events ``x``,
   form the model's own pseudo-pair ``z = encode(x)`` and measure
   ``r = m(x) - m(M(z))`` against the claimed per-event scale ``s(z)``: a
   calibrated conditional covers 68% / 95% of residuals at the nominal levels;
4. **a permutation control** -- the same statistic under a shuffled
   ``(x, z)`` assignment. A statistic that a permutation cannot change is not
   measuring the map. This is the criterion's teeth;
5. **an identity rail and finite-sample floors** -- a point map that hands the
   detector event back claims *zero* spread, so its coverage at any nominal
   level is 0 by construction; two independent draws of the same distribution
   bound the no-information reference; the binomial and bootstrap floors say
   how much of any coverage number is sampling noise.

What this criterion can and cannot conclude (stated, not implied)
----------------------------------------------------------------
* **It can falsify.** A checkpoint whose claimed scale is zero (or far below the
  requirement) while the mean map carries the region's width is rejected here on
  two independent legs, and its coverage collapses under the permutation
  control.
* **It cannot confirm.** In the unpaired setting the cycle residual mixes the
  decoder's conditional noise, the pair-formation error of the encoder and the
  model's own misfit, so a *passing* coverage is necessary, not sufficient. The
  only place a per-event conditional can be confirmed is a bench with pairs --
  see ``scripts_joint/paired_quantile_calibration.py`` (ppzee).
* The residual is measured around the **zero-noise decode** of ``encode(x)``
  (the mean map, by definition of the decomposition), while the scale comes from
  **native-noise** decodes at the same ``z``: the pair ``(r, s)`` is the claim
  and the outcome for the same per-event conditioning.

Usage
-----
    python scripts_joint/conditional_spread.py \\
        --checkpoint D3b_best=outputs/cms_Joint/Run_H_D3b/best_model.pt \\
        --checkpoint SR_best=outputs/cms_Joint/Run_H_SR/best_model.pt \\
        --region jpsi --events 8000 --draws 16 \\
        --output-dir outputs/cms_Joint/conditional_spread/Run_H_D3b_vs_SR
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

import numpy as np

SCHEMA_VERSION = 1
NOMINAL_LEVELS = (0.50, 0.68, 0.80, 0.90, 0.95)
#: The tolerance the approved plan pre-declared for the 68% coverage (S4).
COVERAGE_TOLERANCE = 0.10
#: A pairing sensitivity below this is treated as "the statistic cannot see the
#: map": the shuffle would have to move the explained variance by less than a
#: tenth of the data's own variance.
MIN_PAIRING_SENSITIVITY = 0.10
#: The band the A0.4 scorecard's Axis R uses for within/required. The same band is
#: applied here to the claimed scale over the requirement, twice: once against the
#: std quadrature target and once against the robust one, because which one is
#: adopted changes verdicts (see docs/calibration_target_2026-10-04.md, P4).
CLAIM_BAND = (0.8, 1.25)


# --------------------------------------------------------------------------- #
# pure-numpy core (importable and testable without torch)
# --------------------------------------------------------------------------- #
def nominal_z(level: float) -> float:
    """Two-sided normal quantile for a central coverage ``level``."""
    if not 0.0 < level < 1.0:
        raise ValueError(f"coverage level must be in (0, 1), got {level!r}")
    return NormalDist().inv_cdf(0.5 + 0.5 * float(level))


def robust_half_width(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Half the 16-84 interquantile range: a 1-sigma-like robust scale."""
    values = np.asarray(values, dtype=np.float64)
    lower = np.quantile(values, 0.16, axis=axis)
    upper = np.quantile(values, 0.84, axis=axis)
    return (upper - lower) / 2.0


def per_event_scale(matrix: np.ndarray, *, mode: str = "std") -> np.ndarray:
    """Per-event claimed spread from a ``(n_events, draws)`` decode matrix.

    ``std`` (the default) uses the unbiased variance (``ddof=1``): the 16-84
    robust half width is the project's usual scale, but from a few dozen draws it
    is biased low (sample-quantile bias, about -5% at 32 draws and -2.5% at 64),
    which would silently understate the claim and shift the coverage. Both are
    available; ``robust`` is reported alongside for comparability with the rest
    of the project.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2-D decode matrix, got shape {matrix.shape}")
    if mode == "robust":
        return np.asarray(robust_half_width(matrix, axis=1), dtype=np.float64)
    if mode == "std":
        return matrix.std(axis=1, ddof=1 if matrix.shape[1] > 1 else 0)
    raise ValueError(f"unknown scale mode {mode!r}")


def central_coverage(
    residual: np.ndarray, scale: np.ndarray, level: float, *, center: float = 0.0
) -> float:
    """Fraction of residuals inside ``nominal_z(level) * scale`` around ``center``.

    ``center`` defaults to **zero and not to the empirical median**: the residual
    is already measured around the model's own mean map, so the claimed
    distribution of the residual is zero-mean by construction. Recentring on the
    empirical median would let a biased mean map look calibrated, which is the
    opposite of what this criterion is for; the bias is reported separately as
    ``residual_center_gev``.

    A zero scale (a deterministic checkpoint, or the identity rail) can only
    cover an exactly zero residual, so it scores 0 for every level -- that is the
    degenerate limit this criterion exists to expose, not a numerical accident.
    """
    residual = np.asarray(residual, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    if residual.shape != scale.shape:
        raise ValueError("residual and scale must have the same shape")
    finite = np.isfinite(residual) & np.isfinite(scale)
    if not finite.any():
        return float("nan")
    inside = np.abs(residual[finite] - float(center)) <= nominal_z(level) * scale[finite]
    return float(np.mean(inside))


def binomial_floor_sd(level: float, n: int) -> float:
    """Sampling sd of a coverage estimate at ``n`` independent events."""
    if n <= 0:
        return float("nan")
    return float(np.sqrt(max(level * (1.0 - level), 0.0) / float(n)))


def bootstrap_band(
    values: np.ndarray,
    statistic,
    *,
    n_boot: int = 200,
    seed: int = 0,
    alpha: float = 0.05,
) -> list[float] | None:
    """Percentile bootstrap band of ``statistic`` over resampled events."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or n_boot <= 0:
        return None
    rng = np.random.default_rng(int(seed))
    draws = []
    for _ in range(int(n_boot)):
        sample = values[rng.integers(0, values.size, size=values.size)]
        value = statistic(sample)
        if np.isfinite(value):
            draws.append(float(value))
    if not draws:
        return None
    return [
        float(np.quantile(draws, alpha / 2.0)),
        float(np.quantile(draws, 1.0 - alpha / 2.0)),
    ]


def normalised_reduction(
    model_value: float, floor_value: float, reference_value: float, *, eps: float = 1e-12
) -> float | None:
    """``(model - floor) / (reference - floor)``; lower is better, 1.0 = no better
    than the reference, 0.0 = at the floor. ``None`` when the denominator is
    degenerate, so no ratio is ever quoted without its reference.
    """
    denominator = float(reference_value) - float(floor_value)
    if not np.isfinite(denominator) or abs(denominator) < eps:
        return None
    return float((float(model_value) - float(floor_value)) / denominator)


def finite_draw_null_coverage(
    scale: np.ndarray,
    level: float,
    *,
    draws: int,
    seed: int = 0,
    replication: int = 1,
) -> float:
    """What coverage this estimator reports for a *perfectly* calibrated model.

    The per-event scale is itself estimated from ``draws`` decodes, so the central
    interval is fitted to the same sample it is tested on: with few draws a
    calibrated model cannot reach its nominal coverage. The null simulates exactly
    that -- residuals drawn from ``N(0, s_i)`` and a fresh scale estimate from
    ``draws`` samples -- and is the reference the measured coverage must be read
    against. (The paired bench measures the same effect analytically; see
    ``scripts_joint/paired_quantile_calibration.py``.)
    """
    scale = np.asarray(scale, dtype=np.float64)
    if scale.size == 0 or draws < 2 or float(np.median(scale)) <= 0.0:
        # A zero scale is the degenerate rail, not a null: every synthetic
        # residual is exactly zero and would report coverage 1.0.
        return float("nan")
    rng = np.random.default_rng(int(seed))
    z = float(nominal_z(level))
    covered = 0.0
    total = 0
    for _ in range(max(1, int(replication))):
        residual = rng.normal(0.0, 1.0, size=scale.size) * scale
        noise = rng.normal(0.0, 1.0, size=(scale.size, int(draws))) * scale[:, None]
        estimate = noise.std(axis=1, ddof=1)
        covered += float(np.mean(np.abs(residual) <= z * estimate))
        total += 1
    return covered / max(total, 1)


def calibration_block(
    residual: np.ndarray,
    scale: np.ndarray,
    *,
    levels: tuple[float, ...] = NOMINAL_LEVELS,
    n_boot: int = 200,
    seed: int = 0,
    draws: int | None = None,
) -> dict:
    """Coverage, spread, ratio and floors for one ``(residual, scale)`` pairing.

    ``coverage`` is the calibration test (centred on zero, see
    ``central_coverage``). The ``ratio_*`` entries are *dispersion* diagnostics
    measured around the median residual; for a zero-mean Gaussian the MAD-style
    median ratio is ~0.674 at calibration and the robust ratio is ~1.0, which is
    why the verdict uses the robust one.
    """
    residual = np.asarray(residual, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    n = int(residual.size)
    center = float(np.median(residual)) if n else float("nan")
    abs_residual = np.abs(residual - center) if n else residual
    scale_median = float(np.median(scale)) if n else float("nan")
    residual_robust = float(robust_half_width(residual)) if n else float("nan")
    ratio_median = (
        float(np.median(abs_residual) / scale_median)
        if n and scale_median > 0.0
        else None
    )
    coverage: dict[str, float] = {}
    coverage_band: dict[str, list[float] | None] = {}
    coverage_pooled: dict[str, float] = {}
    scale_pooled = scale_median
    for level in levels:
        key = f"{level:.2f}"
        coverage[key] = central_coverage(residual, scale, level)
        # A pooled scale has no per-event estimation noise, so this second
        # statement isolates the *size* of the claim from the quality of each
        # per-event estimate. With a small `draws` the per-event robust scale is
        # biased low by the sample-quantile bias (~5% at 32 draws, ~2.5% at 64),
        # which is why both are reported.
        coverage_pooled[key] = (
            central_coverage(residual, np.full_like(scale, scale_pooled), level)
            if scale_pooled > 0.0
            else float("nan")
        )
        coverage_band[key] = bootstrap_band(
            residual,
            lambda sample, _level=level, _scale=scale: central_coverage(
                sample, _scale, _level
            ),
            n_boot=n_boot,
            seed=seed,
        )
    return {
        "n_events": n,
        "residual_center_gev": center,
        "residual_std_gev": float(np.std(residual)) if n else float("nan"),
        "residual_robust_half_width_gev": residual_robust,
        "median_abs_residual_gev": float(np.median(abs_residual)) if n else float("nan"),
        "scale_median_gev": scale_median,
        "scale_robust_half_width_gev": float(robust_half_width(scale)) if n else float("nan"),
        "scale_zero_fraction": float(np.mean(scale <= 0.0)) if n else float("nan"),
        "claim_is_degenerate": bool(n > 0 and scale_median <= 0.0),
        "ratio_median_abs_over_scale": ratio_median,
        "ratio_robust_over_median_scale": (
            float(residual_robust / scale_median) if scale_median > 0.0 else None
        ),
        "coverage": coverage,
        "coverage_pooled_scale": coverage_pooled,
        "coverage_null_at_this_n_and_draws": (
            {
                f"{level:.2f}": finite_draw_null_coverage(
                    scale, level, draws=int(draws), seed=seed + 23
                )
                for level in levels
            }
            if draws
            else None
        ),
        "coverage_bootstrap_band": coverage_band,
        "binomial_floor_sd": {
            f"{level:.2f}": binomial_floor_sd(level, n) for level in levels
        },
        "residual_robust_bootstrap_band_gev": bootstrap_band(
            residual, lambda sample: float(robust_half_width(sample)), n_boot=n_boot, seed=seed
        ),
    }


def explained_variance(residual: np.ndarray, reference: np.ndarray) -> float:
    """``1 - Var(residual) / Var(reference)``: how much of the data's spread the
    pairing leaves unexplained. 1.0 = the map reproduces the data exactly,
    0.0 = the map explains nothing, negative = the residual is wider than the
    data itself (an over-claimed channel)."""
    residual = np.asarray(residual, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    denominator = float(np.var(reference))
    if denominator <= 0.0:
        return float("nan")
    return float(1.0 - float(np.var(residual)) / denominator)


def pairing_sensitivity(
    true_residual: np.ndarray, shuffled_residual: np.ndarray, reference: np.ndarray
) -> float:
    """How much the explained variance depends on the ``(x, z)`` assignment.

    This is the criterion's control leg, and it is deliberately **not** a
    coverage difference: coverage saturates and has too little power when the
    mean-map spread is small against the channel's noise. For a constant map the
    two pairings are identical and the sensitivity is exactly 0; for a map whose
    cycle closes under the true pairing but not under a shuffle it approaches
    ``1 + (Var(mean map) - Var(noise)) / Var(data)``.
    """
    return float(
        explained_variance(true_residual, reference)
        - explained_variance(shuffled_residual, reference)
    )


def permutation_sensitivity(true_coverage: float, shuffled_coverage: float) -> float:
    """Coverage difference between the true and the shuffled pairing.

    Reported as a diagnostic only -- see ``pairing_sensitivity`` for the leg the
    verdict uses.
    """
    if not np.isfinite(true_coverage) or not np.isfinite(shuffled_coverage):
        return float("nan")
    return float(true_coverage - shuffled_coverage)


# --------------------------------------------------------------------------- #
# repository plumbing
# --------------------------------------------------------------------------- #
def find_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "scripts_joint").is_dir() and (candidate / "scripts_sota").is_dir():
            return candidate
    raise RuntimeError("Could not locate the OTUS repository root")


REPO_ROOT = find_repo_root()
for directory in (
    REPO_ROOT / "scripts",
    REPO_ROOT / "scripts_sota",
    REPO_ROOT / "scripts_joint",
    REPO_ROOT / "scripts_joint" / "upsilon",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


def mass8(values: np.ndarray) -> np.ndarray:
    """Stable dimuon invariant mass from an ``(N, 8)`` ``[p-, E-, p+, E+]`` array."""
    values = np.asarray(values, dtype=np.float64)
    energy = values[:, 3] + values[:, 7]
    momentum = values[:, :3] + values[:, 4:7]
    return np.sqrt(np.maximum(energy * energy - np.sum(momentum * momentum, axis=1), 0.0))


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def parse_checkpoint(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise SystemExit(f"--checkpoint expects LABEL=RELATIVE_PATH, got {raw!r}")
    label, relative = raw.split("=", 1)
    return label, (REPO_ROOT / relative)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="LABEL=RELATIVE_PATH",
        help="checkpoint to score (repeatable); the first one supplies the config",
    )
    parser.add_argument(
        "--region",
        action="append",
        default=None,
        help="mass region to score (repeatable; default jpsi)",
    )
    parser.add_argument("--events", type=int, default=8000)
    parser.add_argument(
        "--draws",
        type=int,
        default=64,
        help=(
            "decode draws per event for the per-event scale; the robust scale from "
            "a small n is biased low (about -5%% at 32 draws, -2.5%% at 64)"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--encoder-noise",
        choices=("zero", "native"),
        default="zero",
        help="pairing noise for z = encode(x); zero keeps the pseudo-pairs deterministic",
    )
    parser.add_argument(
        "--required-resolution-std",
        type=float,
        default=None,
        help="override the data-driven quadrature requirement for every region [GeV]",
    )
    parser.add_argument(
        "--scale-mode",
        choices=("std", "robust"),
        default="std",
        help="per-event scale estimator for the claim (default std, unbiased variance)",
    )
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "cms_Joint" / "conditional_spread",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# torch-dependent measurement
# --------------------------------------------------------------------------- #
def encode_values(model, values: np.ndarray, *, batch_size: int, device) -> np.ndarray:
    import torch

    outputs = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            block = torch.as_tensor(
                np.ascontiguousarray(values[start : start + batch_size]),
                dtype=torch.float32,
                device=device,
            )
            outputs.append(model.encode(block).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def decode_masses(
    model,
    z: np.ndarray,
    *,
    batch_size: int,
    device,
    draws: int,
    seed: int,
) -> np.ndarray:
    """Decode each ``z`` ``draws`` times; return an ``(n_events, draws)`` mass matrix."""
    import torch

    torch.manual_seed(int(seed))
    indices = np.repeat(np.arange(len(z)), draws)
    flat = np.empty(len(indices), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            block = indices[start : start + batch_size]
            values = torch.as_tensor(
                np.ascontiguousarray(z[block]), dtype=torch.float32, device=device
            )
            decoded = model.decode(values)
            energy = decoded[:, 3] + decoded[:, 7]
            momentum = decoded[:, :3] + decoded[:, 4:7]
            mass = torch.sqrt(
                torch.clamp(energy * energy - torch.sum(momentum * momentum, dim=1), min=0.0)
            )
            flat[start : start + len(block)] = mass.detach().cpu().numpy()
    return flat.reshape(len(z), draws)


def matrix_decomposition(matrix: np.ndarray) -> dict:
    """The exact split ``Var = E_z[Var(x|z)] + Var_z[E(x|z)]`` of a decode matrix."""
    within = matrix.std(axis=1, ddof=0)
    conditional_mean = matrix.mean(axis=1)
    within_var = float(np.mean(within**2))
    across_var = float(np.var(conditional_mean, ddof=0))
    total_var = float(np.var(matrix.ravel(), ddof=0))
    return {
        "within_std_gev": float(np.sqrt(max(within_var, 0.0))),
        "across_std_gev": float(np.sqrt(max(across_var, 0.0))),
        "total_std_gev": float(np.sqrt(max(total_var, 0.0))),
        "noise_variance_fraction": float(within_var / total_var) if total_var > 0 else None,
    }


def region_measurement(
    model,
    *,
    x_data: np.ndarray,
    z_prior: np.ndarray,
    x_reference: np.ndarray,
    args: argparse.Namespace,
    device,
    seed: int,
    required_override: float | None,
    encoder_multipliers: tuple[float, float],
    decoder_multipliers: tuple[float, float],
) -> dict:
    """One checkpoint on one region: every number the criterion reports."""
    x_mass = mass8(x_data)
    prior_mass = mass8(z_prior)
    reference_mass = mass8(x_reference)

    # The mean map is the zero-noise decode of the encoded event: deterministic
    # by definition, which is what makes `r` the residual around the mean map.
    model.set_component_noise_multipliers(
        encoder_core=encoder_multipliers[0],
        encoder_tail=encoder_multipliers[1],
        decoder_core=0.0,
        decoder_tail=0.0,
    )
    z_hat = encode_values(model, x_data, batch_size=args.batch_size, device=device)
    mean_map = decode_masses(
        model, z_hat, batch_size=args.batch_size, device=device, draws=1, seed=seed + 11
    )[:, 0]

    # The claim: repeated native-noise decodes of the same z.
    model.set_component_noise_multipliers(
        encoder_core=encoder_multipliers[0],
        encoder_tail=encoder_multipliers[1],
        decoder_core=decoder_multipliers[0],
        decoder_tail=decoder_multipliers[1],
    )
    draws_matrix = decode_masses(
        model,
        z_hat,
        batch_size=args.batch_size,
        device=device,
        draws=args.draws,
        seed=seed + 101,
    )

    # ---- leg 1: the claim (within-z) and the mean map (across-z) ---------------
    decomposition = matrix_decomposition(draws_matrix)
    scale = per_event_scale(draws_matrix, mode=args.scale_mode)
    robust_scale = per_event_scale(draws_matrix, mode="robust")

    # ---- leg 2: claim versus the data-driven requirement ----------------------
    # Two denominators, because the project adopts one and the A0.4 R gate uses
    # the other; every identification number has to be quoted with its own.
    required_std = (
        float(required_override)
        if required_override is not None
        else float(np.sqrt(max(np.var(reference_mass) - np.var(prior_mass), 0.0)))
    )
    required_robust = (
        float(required_override)
        if required_override is not None
        else float(
            np.sqrt(
                max(
                    robust_half_width(reference_mass) ** 2
                    - robust_half_width(prior_mass) ** 2,
                    0.0,
                )
            )
        )
    )
    required = required_std
    scale_median = float(np.median(scale))
    robust_scale_median = float(np.median(robust_scale))
    claim_over_required = float(scale_median / required_std) if required_std > 0 else None
    claim_over_required_robust = (
        float(robust_scale_median / required_robust) if required_robust > 0 else None
    )

    # ---- leg 3: coverage of the cycle residual, and its permutation control ----
    residual = x_mass - mean_map
    true_block = calibration_block(
        residual, scale, n_boot=args.bootstrap, seed=seed + 3, draws=int(args.draws)
    )
    rng = np.random.default_rng(seed + 5)
    permutation = rng.permutation(len(residual))
    shuffled_residual = x_mass - mean_map[permutation]
    shuffled_block = calibration_block(
        shuffled_residual,
        scale[permutation],
        n_boot=args.bootstrap,
        seed=seed + 7,
    )
    coverage_sensitivity = permutation_sensitivity(
        true_block["coverage"]["0.68"], shuffled_block["coverage"]["0.68"]
    )
    explained = {
        "true_pairing": explained_variance(residual, x_mass),
        "shuffled_pairing": explained_variance(shuffled_residual, x_mass),
        "mean_map_variance_gev2": float(np.var(mean_map)),
        "data_variance_gev2": float(np.var(x_mass)),
    }
    sensitivity = pairing_sensitivity(residual, shuffled_residual, x_mass)

    # ---- leg 4: the identity rail and the no-information floors ---------------
    n_reference = min(len(reference_mass), len(prior_mass))
    identity_residual = reference_mass[:n_reference] - prior_mass[:n_reference]
    half = n_reference // 2
    disjoint_residual = (
        reference_mass[:half] - reference_mass[half : 2 * half] if half > 0 else np.zeros(0)
    )
    identity_scale = np.zeros(n_reference)
    identity_block = calibration_block(
        identity_residual, identity_scale, n_boot=min(args.bootstrap, 50), seed=seed + 13
    )
    disjoint_block = calibration_block(
        disjoint_residual,
        np.zeros(len(disjoint_residual)),
        n_boot=min(args.bootstrap, 50),
        seed=seed + 17,
    )
    identity_robust = identity_block["residual_robust_half_width_gev"]
    floor_robust = disjoint_block["residual_robust_half_width_gev"]
    reduction = normalised_reduction(
        true_block["residual_robust_half_width_gev"], floor_robust, identity_robust
    )

    # ---- verdict -------------------------------------------------------------
    coverage_68 = true_block["coverage"]["0.68"]
    coverage_95 = true_block["coverage"]["0.95"]
    band_68 = true_block["coverage_bootstrap_band"]["0.68"]
    low, high = CLAIM_BAND
    reasons = []
    claim_reasons = []
    if true_block["claim_is_degenerate"]:
        reasons.append("claimed per-event scale is zero (deterministic map)")
    if claim_over_required is None or claim_over_required < 0.5:
        reasons.append("claimed scale below half the data-driven requirement")
    if not np.isfinite(coverage_68) or abs(coverage_68 - 0.68) > COVERAGE_TOLERANCE:
        reasons.append(
            f"68% coverage {coverage_68:.3f} outside 0.68 +/- {COVERAGE_TOLERANCE:.2f}"
        )
    if not np.isfinite(sensitivity) or sensitivity < MIN_PAIRING_SENSITIVITY:
        reasons.append(
            f"pairing sensitivity {sensitivity:.3f} below {MIN_PAIRING_SENSITIVITY:.2f}"
        )
    for name, ratio in (
        ("std", claim_over_required),
        ("robust", claim_over_required_robust),
    ):
        if ratio is None or not (low <= ratio <= high):
            shown = "n/a" if ratio is None else f"{ratio:.3f}"
            claim_reasons.append(f"claimed/required ({name} target) {shown} outside [{low}, {high}]")
    calibration_verdict = "calibrated" if not reasons else "rejected"
    claim_band_verdict = "in_band" if not claim_reasons else "off_band"
    verdict = (
        "calibrated"
        if calibration_verdict == "calibrated" and claim_band_verdict == "in_band"
        else "rejected"
    )

    return {
        "reference": {
            "n_prior": int(len(prior_mass)),
            "n_cms": int(len(reference_mass)),
            "prior_mass_std_gev": float(np.std(prior_mass)),
            "prior_mass_robust_half_width_gev": float(robust_half_width(prior_mass)),
            "cms_mass_std_gev": float(np.std(reference_mass)),
            "cms_mass_robust_half_width_gev": float(robust_half_width(reference_mass)),
            "required_quadrature_std_gev": required_std,
            "required_quadrature_robust_gev": required_robust,
            "required_source": "override" if required_override is not None else "data_driven",
        },
        "claim": {
            **decomposition,
            "scale_mode": args.scale_mode,
            "scale_median_gev": scale_median,
            "scale_robust_median_gev": float(np.median(robust_scale)),
            "scale_robust_half_width_gev": float(robust_half_width(scale)),
            "claim_over_required": claim_over_required,
            "claim_over_required_robust": claim_over_required_robust,
            "claim_is_degenerate": bool(true_block["claim_is_degenerate"]),
        },
        "cycle_residual": true_block,
        "cycle_residual_permuted": shuffled_block,
        "pairing_sensitivity": sensitivity,
        "explained_variance": explained,
        "permutation_sensitivity_coverage_68": coverage_sensitivity,
        "identity_rail": {
            **identity_block,
            "no_model_residual_robust_half_width_gev": identity_robust,
            "claims_zero_spread": True,
        },
        "no_information_floor": {
            **disjoint_block,
            "disjoint_data_residual_robust_half_width_gev": floor_robust,
        },
        "reduction_vs_no_model": reduction,
        "coverage_95": coverage_95,
        "coverage_68_bootstrap_band": band_68,
        "calibration_verdict": calibration_verdict,
        "claim_band_verdict": claim_band_verdict,
        "calibration_reasons": reasons,
        "claim_band_reasons": claim_reasons,
        "verdict": verdict,
        "verdict_reasons": reasons + claim_reasons,
    }


def make_plot(payload: dict, path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - plotting is optional
        return False
    levels = list(NOMINAL_LEVELS)
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    labels, ratios = [], []
    for index, (label, entry) in enumerate(payload["checkpoints"].items()):
        color = palette[index % len(palette)]
        for region, block in entry["regions"].items():
            coverage = [block["cycle_residual"]["coverage"][f"{level:.2f}"] for level in levels]
            shuffled = [
                block["cycle_residual_permuted"]["coverage"][f"{level:.2f}"] for level in levels
            ]
            axes[0].plot(
                levels, coverage, marker="o", color=color, label=f"{label} / {region}"
            )
            axes[0].plot(levels, shuffled, marker="x", ls="--", color=color, alpha=0.45)
            labels.append(f"{label}\n{region}")
            ratios.append(block["claim"]["claim_over_required"] or 0.0)
    axes[0].plot([0.0, 1.0], [0.0, 1.0], color="0.6", lw=1.0, label="nominal")
    axes[0].set_xlabel("nominal central coverage")
    axes[0].set_ylabel("empirical coverage")
    axes[0].set_title("cycle residual (solid) vs permuted pairing (dashed)")
    axes[0].legend(fontsize=7, loc="upper left")
    axes[1].bar(range(len(ratios)), ratios, color="#1f77b4")
    axes[1].axhline(1.0, color="0.4", lw=1.0, ls="--")
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=90, fontsize=7)
    axes[1].set_ylabel("claimed scale / required resolution")
    axes[1].set_title("leg 2: is the claim the right size?")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def write_report(payload: dict, path: Path) -> None:
    lines = [
        "# Conditional-spread criterion",
        "",
        f"*artifact-measured, {payload['created_utc'][:10]}. Read-only: "
        f"`scripts_joint/conditional_spread.py`, events {payload['events']}, draws "
        f"{payload['draws']}, encoder noise `{payload['encoder_noise']}`, "
        f"device `{payload['device']}`. Labels follow `CLAUDE.md` section 2.*",
        "",
        "Legs: (1) the exact split, (2) claimed scale versus the data-driven "
        "requirement under **both** denominators (the std quadrature target adopted "
        "in `docs/calibration_target_2026-10-04.md`, P4, and the robust target the "
        "A0.4 R gate uses), (3) coverage of the cycle residual, (4) the permutation "
        "control on the explained variance, (5) the identity rail and floors. "
        f"`calibration_verdict` needs the 68% coverage inside 0.68 +/- "
        f"{COVERAGE_TOLERANCE:.2f} and a pairing sensitivity above "
        f"{MIN_PAIRING_SENSITIVITY:.2f}; `claim_band_verdict` needs the claimed "
        f"scale between {CLAIM_BAND[0]:.2f} and {CLAIM_BAND[1]:.2f} of **both** "
        "requirements. An off-band claim is a failure of *size*, not of coherence.",
        "",
    ]
    for label, entry in payload["checkpoints"].items():
        lines += [f"## {label}", "", f"- checkpoint: `{entry['path']}`",
                  f"- global epoch: {entry['global_epoch']}, stage: `{entry['stage_name']}`",
                  f"- native decoder multipliers: {entry['native_decoder_multipliers']}",
                  ""]
        for region, block in entry["regions"].items():
            reference = block["reference"]
            claim = block["claim"]
            true_block = block["cycle_residual"]
            shuffled = block["cycle_residual_permuted"]
            lines += [
                f"### {label} / {region}",
                "",
                f"| quantity | value |", "|---|---|",
                f"| prior mass std [GeV] | {reference['prior_mass_std_gev']:.5g} |",
                f"| CMS mass std [GeV] | {reference['cms_mass_std_gev']:.5g} |",
                f"| required quadrature std [GeV] ({reference['required_source']}) | "
                f"{reference['required_quadrature_std_gev']:.5g} |",
                f"| required quadrature robust [GeV] | "
                f"{reference['required_quadrature_robust_gev']:.5g} |",
                f"| claimed scale, median [GeV] ({claim['scale_mode']} estimator) | "
                f"{claim['scale_median_gev']:.5g} |",
                f"| claimed scale, robust 16-84 median [GeV] | "
                f"{claim['scale_robust_median_gev']:.5g} |",
                f"| claimed / required (std target, adopted) | "
                f"{claim['claim_over_required']:.3f} |"
                if claim["claim_over_required"] is not None
                else "| claimed / required (std target) | n/a |",
                f"| claimed / required (robust target, A0.4 band) | "
                f"{claim['claim_over_required_robust']:.3f} |"
                if claim["claim_over_required_robust"] is not None
                else "| claimed / required (robust target) | n/a |",
                f"| mean-map (across-z) std [GeV] | {claim['across_std_gev']:.5g} |",
                f"| within-z (noise) std [GeV] | {claim['within_std_gev']:.5g} |",
                f"| noise variance fraction | {claim['noise_variance_fraction']:.3g} |"
                if claim["noise_variance_fraction"] is not None
                else "| noise variance fraction | n/a |",
                f"| cycle residual robust width [GeV] | "
                f"{true_block['residual_robust_half_width_gev']:.5g} |",
                f"| coverage 68% (true pairing) | {true_block['coverage']['0.68']:.3f} "
                f"(bootstrap {true_block['coverage_bootstrap_band']['0.68']}) |",
                f"| coverage 68% (pooled scale, no per-event noise) | "
                f"{true_block['coverage_pooled_scale']['0.68']:.3f} |",
                f"| coverage 68% (finite-draw null at this n, D) | "
                f"{(true_block['coverage_null_at_this_n_and_draws'] or {}).get('0.68', float('nan')):.3f} |",
                f"| coverage 95% (true pairing) | {true_block['coverage']['0.95']:.3f} |",
                f"| coverage 68% (permuted) | {shuffled['coverage']['0.68']:.3f} |",
                f"| explained variance, true pairing | "
                f"{block['explained_variance']['true_pairing']:.3f} |",
                f"| explained variance, shuffled pairing | "
                f"{block['explained_variance']['shuffled_pairing']:.3f} |",
                f"| **pairing sensitivity** (control leg) | "
                f"**{block['pairing_sensitivity']:.3f}** |",
                f"| coverage sensitivity (diagnostic) | "
                f"{block['permutation_sensitivity_coverage_68']:.3f} |",
                f"| identity rail: no-model residual robust width [GeV] | "
                f"{block['identity_rail']['no_model_residual_robust_half_width_gev']:.5g} |",
                f"| floor: disjoint-data residual robust width [GeV] | "
                f"{block['no_information_floor']['disjoint_data_residual_robust_half_width_gev']:.5g} |",
                f"| reduction `(model - floor) / (identity - floor)` | "
                f"{block['reduction_vs_no_model']} |",
                f"| binomial floor sd at 68% | "
                f"{true_block['binomial_floor_sd']['0.68']:.4f} |",
                f"| calibration verdict (legs 3-4) | {block['calibration_verdict']} |",
                f"| claim-band verdict (leg 2, both targets) | {block['claim_band_verdict']} |",
                f"| **overall verdict** | **{block['verdict']}** |",
                "",
            ]
            if block["verdict_reasons"]:
                lines += ["Reasons: " + "; ".join(block["verdict_reasons"]) + ".", ""]
    lines += [
        "## What this can and cannot conclude",
        "",
        "- A `rejected` verdict is a falsification: the claim and the outcome "
        "disagree, or the statistic cannot see the map (permutation control).",
        "- A `calibrated` verdict is **necessary, not sufficient**: in the "
        "unpaired setting the cycle residual mixes decoder noise, the encoder's "
        "pair-formation error and model misfit. The per-event conditional can "
        "only be confirmed on a bench with pairs "
        "(`scripts_joint/paired_quantile_calibration.py`, ppzee).",
        "- The identity rail scores 0 coverage by construction: a map that hands "
        "the detector event back claims zero spread. That is the degenerate limit "
        "the criterion exists to expose.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    started = time.time()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"

    def log(message: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    import torch

    from decode_prior import load_frozen_model
    from joint_data import load_joint_regions, resolve_joint_config

    device = torch.device(args.device)
    checkpoints = [parse_checkpoint(raw) for raw in args.checkpoint]
    for label, path in checkpoints:
        if not path.exists():
            raise FileNotFoundError(f"checkpoint {label} not found: {path}")
    regions = list(args.region or ["jpsi"])

    log(f"conditional-spread criterion on {device}; regions {regions}; output {output_dir}")
    _, _, config = load_frozen_model(checkpoints[0][1], torch.device("cpu"))
    resolved = resolve_joint_config(config)
    arrays, cache_info, region_configs, _ = load_joint_regions(
        resolved, num_samples=None, use_cache=True, log=lambda message: None
    )
    for region in regions:
        if region not in arrays:
            raise SystemExit(f"region {region!r} not in the locked split: {sorted(arrays)}")

    count = int(args.events)
    results: dict[str, dict] = {}
    for index, (label, path) in enumerate(checkpoints):
        model, checkpoint, _ = load_frozen_model(path, device)
        model.eval()
        recorded = checkpoint.get("noise_multipliers") or {}
        shared_core = float(recorded.get("core", 1.0))
        shared_tail = float(recorded.get("tail", 1.0))
        native_core = float(recorded.get("decoder_core", shared_core))
        native_tail = float(recorded.get("decoder_tail", shared_tail))
        encoder_core = float(recorded.get("encoder_core", shared_core))
        encoder_tail = float(recorded.get("encoder_tail", shared_tail))
        if args.encoder_noise == "zero":
            encoder_core = encoder_tail = 0.0
        stage = checkpoint.get("stage") or {}

        entry = {
            "path": str(path),
            "global_epoch": checkpoint.get("global_epoch"),
            "stage_name": stage.get("name") if isinstance(stage, dict) else None,
            "recorded_noise_multipliers": json_ready(recorded),
            "native_decoder_multipliers": {"core": native_core, "tail": native_tail},
            "regions": {},
        }
        for region in regions:
            region_seed = args.seed + index * 1000 + 10 * regions.index(region)
            x_test = np.asarray(arrays[region]["x_test"], dtype=np.float32)
            z_test = np.asarray(arrays[region]["z_test"], dtype=np.float32)
            selection = (region_configs[region].get("theory_prior_selection") or {})
            rng = np.random.default_rng(region_seed)
            n_events = min(count, len(x_test), len(z_test))
            x_chosen = np.sort(rng.choice(len(x_test), size=n_events, replace=False))
            z_chosen = np.sort(rng.choice(len(z_test), size=n_events, replace=False))
            measurement = region_measurement(
                model,
                x_data=np.ascontiguousarray(x_test[x_chosen]),
                z_prior=np.ascontiguousarray(z_test[z_chosen]),
                x_reference=np.ascontiguousarray(x_test[: min(200_000, len(x_test))]),
                args=args,
                device=device,
                seed=region_seed,
                required_override=args.required_resolution_std,
                encoder_multipliers=(encoder_core, encoder_tail),
                decoder_multipliers=(native_core, native_tail),
            )
            measurement["region_config"] = {
                "prior_window_gev": [selection.get("mass_min"), selection.get("mass_max")],
                "cache_dir": (
                    cache_info[region].get("cache_dir")
                    if isinstance(cache_info[region], dict)
                    else None
                ),
            }
            entry["regions"][region] = measurement
            log(
                f"[{index + 1}/{len(checkpoints)}] {label} / {region}: claim "
                f"{measurement['claim']['scale_median_gev']:.5g} GeV "
                f"({measurement['claim']['claim_over_required']:.2f}x required), "
                f"residual {measurement['cycle_residual']['residual_robust_half_width_gev']:.5g} GeV, "
                f"coverage68 {measurement['cycle_residual']['coverage']['0.68']:.3f} "
                f"(permuted {measurement['cycle_residual_permuted']['coverage']['0.68']:.3f}), "
                f"pairing sensitivity {measurement['pairing_sensitivity']:.3f} "
                f"-> {measurement['calibration_verdict']}/{measurement['claim_band_verdict']} "
                f"= {measurement['verdict']}"
            )
        results[label] = entry
        del model

    payload = {
        "schema_version": SCHEMA_VERSION,
        "diagnostic": "conditional-spread criterion (claim, coverage, permutation control)",
        "status": "completed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.time() - started,
        "device": str(device),
        "events": count,
        "draws": int(args.draws),
        "seed": int(args.seed),
        "encoder_noise": args.encoder_noise,
        "scale_mode": args.scale_mode,
        "coverage_tolerance": COVERAGE_TOLERANCE,
        "min_pairing_sensitivity": MIN_PAIRING_SENSITIVITY,
        "checkpoints": results,
    }
    json_path = output_dir / "conditional_spread.json"
    json_path.write_text(json.dumps(json_ready(payload), indent=2) + "\n", encoding="utf-8")
    log(f"wrote {json_path}")
    write_report(payload, output_dir / "REPORT.md")
    log(f"wrote {output_dir / 'REPORT.md'}")
    if not args.no_plot:
        plot_path = output_dir / "calibration_curves.png"
        if make_plot(payload, plot_path):
            log(f"wrote {plot_path}")
    log(f"done in {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
