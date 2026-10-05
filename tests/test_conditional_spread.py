#!/usr/bin/env python
"""Tests for the conditional-spread criterion (`scripts_joint/conditional_spread.py`).

Two layers:

1. **pure functions** -- the scale, coverage, floor and reduction helpers;
2. **the criterion end to end** on a duck-typed fake response whose conditional
   is known exactly, so every leg of the verdict can be exercised deliberately:
   a calibrated model, a dead noise channel, a vacuous (permutation-invariant)
   map, an over-claiming channel and an under-claiming one.

The fake avoids any real checkpoint or dataset: the mass coordinate is packed
into the two energies, so ``x = M(z) + eps`` is exact and ``eps`` is known.
"""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (REPO_ROOT / "scripts", REPO_ROOT / "scripts_joint"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from conditional_spread import (  # noqa: E402
    COVERAGE_TOLERANCE,
    MIN_PAIRING_SENSITIVITY,
    binomial_floor_sd,
    bootstrap_band,
    calibration_block,
    central_coverage,
    explained_variance,
    finite_draw_null_coverage,
    nominal_z,
    normalised_reduction,
    pairing_sensitivity,
    per_event_scale,
    permutation_sensitivity,
    region_measurement,
    robust_half_width,
)


def _pack(mass_values):
    import torch

    half = mass_values / 2.0
    zeros = torch.zeros_like(half)
    return torch.stack(
        [zeros, zeros, zeros, half, zeros, zeros, zeros, half], dim=1
    )


def _mass_torch(values):
    import torch

    energy = values[:, 3] + values[:, 7]
    momentum = values[:, :3] + values[:, 4:7]
    return torch.sqrt(torch.clamp(energy * energy - torch.sum(momentum * momentum, dim=1), min=0.0))


class FakeResponse:
    """Duck-typed stand-in for the joint autoencoder, in the mass coordinate only.

    ``encode`` is an **oracle inverse**: it subtracts the realised noise that
    produced the detector event, so ``z_hat`` is the truth and the cycle residual
    ``x - M(z_hat)`` is the true noise draw -- which is exactly what a converged
    cycle term produces in the real model. ``decode`` adds Gaussian noise scaled
    by the decoder multiplier and the channel's own ``sigma``.
    """

    def __init__(self, *, mean_fn=None, inverse_fn=None, sigma=0.0, denoise=None):
        self.mean_fn = mean_fn if mean_fn is not None else (lambda m: m)
        self.inverse_fn = inverse_fn if inverse_fn is not None else (lambda m: m)
        self.sigma = float(sigma)
        self.denoise = None if denoise is None else np.asarray(denoise, dtype=np.float32)
        self.multipliers: dict[str, float] = {
            "encoder_core": 0.0,
            "encoder_tail": 0.0,
            "decoder_core": 0.0,
            "decoder_tail": 0.0,
        }

    def set_component_noise_multipliers(self, **kwargs):
        self.multipliers.update({key: float(value) for key, value in kwargs.items()})

    def encode(self, x):
        import torch

        mass = _mass_torch(x)
        if self.denoise is not None:
            mass = mass - torch.as_tensor(
                self.denoise[: mass.shape[0]], dtype=mass.dtype, device=mass.device
            )
        return _pack(self.inverse_fn(mass))

    def decode(self, z):
        import torch

        mass = self.mean_fn(_mass_torch(z))
        multiplier = float(self.multipliers.get("decoder_core", 0.0))
        if multiplier > 0.0:
            mass = mass + torch.randn_like(mass) * self.sigma * multiplier
        return _pack(mass)


def _args(**overrides) -> argparse.Namespace:
    values = {
        "batch_size": 512,
        "draws": 64,
        "bootstrap": 40,
        "scale_mode": "std",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _pack_np(masses: np.ndarray) -> np.ndarray:
    """Pack masses into ``(N, 8)`` four-vectors whose stable mass is ``masses``."""
    masses = np.asarray(masses, dtype=np.float32)
    values = np.zeros((len(masses), 8), dtype=np.float32)
    values[:, 3] = masses / 2.0
    values[:, 7] = masses / 2.0
    return values


def _synthetic_problem(*, n=4000, sigma_true=0.02, prior_std=0.01, seed=7):
    """Packed detector events, packed prior events, the reference data and the noise."""
    generator = np.random.default_rng(seed)
    truth = 4.0 + generator.normal(0.0, prior_std, size=n)
    noise = generator.normal(0.0, sigma_true, size=n)
    detector = truth + noise
    prior = 4.0 + generator.normal(0.0, prior_std, size=3 * n)
    return {
        "x_data": _pack_np(detector),
        "z_prior": _pack_np(prior),
        "x_reference": _pack_np(detector),
        "noise": noise,
        "sigma_true": sigma_true,
        "detector_mass": detector,
    }


class PureFunctionTests(unittest.TestCase):
    def test_nominal_z_matches_the_normal_quantiles(self):
        self.assertAlmostEqual(nominal_z(0.68), 0.9944578832, places=6)
        self.assertAlmostEqual(nominal_z(0.95), 1.9599639845, places=6)
        self.assertAlmostEqual(nominal_z(0.50), 0.6744897502, places=6)

    def test_nominal_z_rejects_impossible_levels(self):
        for level in (0.0, 1.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                nominal_z(level)

    def test_robust_half_width_is_the_sixteen_eighty_four_half_range(self):
        values = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
        lower = np.quantile(values, 0.16)
        upper = np.quantile(values, 0.84)
        self.assertAlmostEqual(float(robust_half_width(values)), (upper - lower) / 2.0)

    def test_per_event_scale_modes(self):
        matrix = np.array([[0.0, 1.0], [2.0, 2.0], [1.0, 3.0]])
        std = per_event_scale(matrix, mode="std")
        robust = per_event_scale(matrix, mode="robust")
        # Unbiased variance (ddof=1): std of {0, 1} is sqrt(0.5), not 0.5.
        self.assertAlmostEqual(float(std[0]), np.sqrt(0.5))
        self.assertAlmostEqual(float(std[1]), 0.0)
        self.assertTrue(np.all(robust >= 0.0))
        with self.assertRaises(ValueError):
            per_event_scale(np.zeros(3))
        with self.assertRaises(ValueError):
            per_event_scale(matrix, mode="nonsense")

    def test_central_coverage_of_a_calibrated_gaussian(self):
        rng = np.random.default_rng(0)
        residual = rng.normal(0.0, 1.0, size=200_000)
        scale = np.ones_like(residual)
        self.assertAlmostEqual(central_coverage(residual, scale, 0.68), 0.68, places=2)
        self.assertAlmostEqual(central_coverage(residual, scale, 0.95), 0.95, places=2)

    def test_central_coverage_under_and_over_dispersion(self):
        rng = np.random.default_rng(1)
        residual = rng.normal(0.0, 1.0, size=200_000)
        under = central_coverage(residual, np.full_like(residual, 0.5), 0.68)
        over = central_coverage(residual, np.full_like(residual, 2.0), 0.68)
        self.assertAlmostEqual(under, 0.383, places=2)
        self.assertAlmostEqual(over, 0.954, places=2)

    def test_central_coverage_with_zero_scale_is_the_degenerate_rail(self):
        residual = np.array([0.1, -0.2, 0.05, 0.0])
        scale = np.zeros_like(residual)
        # Only the exactly-zero residual can be covered by a zero-scale claim.
        self.assertAlmostEqual(central_coverage(residual, scale, 0.68), 0.25)

    def test_central_coverage_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            central_coverage(np.zeros(3), np.zeros(4), 0.68)

    def test_binomial_floor_sd(self):
        self.assertAlmostEqual(binomial_floor_sd(0.68, 10_000), np.sqrt(0.68 * 0.32 / 10_000))
        self.assertTrue(np.isnan(binomial_floor_sd(0.68, 0)))

    def test_bootstrap_band_brackets_the_point_estimate(self):
        rng = np.random.default_rng(3)
        values = rng.normal(0.0, 1.0, size=500)
        band = bootstrap_band(values, lambda sample: float(np.mean(sample)), n_boot=60, seed=1)
        self.assertIsNotNone(band)
        self.assertLess(band[0], float(np.mean(values)))
        self.assertGreater(band[1], float(np.mean(values)))

    def test_bootstrap_band_is_none_when_disabled_or_empty(self):
        self.assertIsNone(bootstrap_band(np.zeros(0), lambda sample: 0.0, n_boot=10))
        self.assertIsNone(bootstrap_band(np.zeros(10), lambda sample: 0.0, n_boot=0))

    def test_normalised_reduction_conventions(self):
        self.assertAlmostEqual(normalised_reduction(5.0, 1.0, 5.0), 1.0)
        self.assertAlmostEqual(normalised_reduction(1.0, 1.0, 5.0), 0.0)
        self.assertAlmostEqual(normalised_reduction(9.0, 1.0, 5.0), 2.0)
        self.assertIsNone(normalised_reduction(3.0, 1.0, 1.0))

    def test_permutation_sensitivity_is_the_coverage_difference(self):
        self.assertAlmostEqual(permutation_sensitivity(0.68, 0.10), 0.58)
        self.assertTrue(np.isnan(permutation_sensitivity(float("nan"), 0.1)))

    def test_finite_draw_null_is_below_nominal_and_rises_with_draws(self):
        """A calibrated model cannot reach nominal coverage when the per-event
        scale is estimated from few draws; the null quantifies that gap."""
        scale = np.full(20_000, 0.02)
        low = finite_draw_null_coverage(scale, 0.68, draws=8, seed=0)
        mid = finite_draw_null_coverage(scale, 0.68, draws=32, seed=0)
        high = finite_draw_null_coverage(scale, 0.68, draws=512, seed=0)
        self.assertLess(low, mid)
        self.assertLess(mid, high)
        self.assertLess(low, 0.66)
        self.assertGreater(low, 0.62)
        self.assertAlmostEqual(mid, 0.674, delta=0.01)
        self.assertAlmostEqual(high, 0.678, delta=0.01)

    def test_finite_draw_null_needs_at_least_two_draws(self):
        scale = np.full(100, 0.5)
        self.assertTrue(np.isnan(finite_draw_null_coverage(scale, 0.68, draws=1)))
        self.assertTrue(np.isnan(finite_draw_null_coverage(np.zeros(0), 0.68, draws=32)))

    def test_calibration_block_reports_the_null_when_draws_are_given(self):
        rng = np.random.default_rng(11)
        scale = np.full(4000, 0.02)
        residual = rng.normal(0.0, 0.02, size=4000)
        block = calibration_block(residual, scale, n_boot=5, seed=0, draws=8)
        null = block["coverage_null_at_this_n_and_draws"]["0.68"]
        self.assertLess(null, 0.67)
        self.assertGreater(null, 0.63)
        without = calibration_block(residual, scale, n_boot=5, seed=0)
        self.assertIsNone(without["coverage_null_at_this_n_and_draws"])

    def test_explained_variance_conventions(self):
        reference = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(explained_variance(np.zeros(4), reference), 1.0)
        self.assertAlmostEqual(explained_variance(reference, reference), 0.0)
        self.assertTrue(np.isnan(explained_variance(np.zeros(4), np.zeros(4))))

    def test_pairing_sensitivity_is_zero_for_a_constant_map(self):
        rng = np.random.default_rng(5)
        reference = rng.normal(0.0, 1.0, size=2000)
        residual = reference - 3.0
        self.assertAlmostEqual(pairing_sensitivity(residual, residual, reference), 0.0)

    def test_pairing_sensitivity_is_positive_when_the_pairing_matters(self):
        rng = np.random.default_rng(6)
        truth = rng.normal(0.0, 1.0, size=4000)
        noise = rng.normal(0.0, 0.5, size=4000)
        reference = truth + noise
        true_residual = reference - truth
        shuffled_residual = reference - truth[rng.permutation(len(truth))]
        sensitivity = pairing_sensitivity(true_residual, shuffled_residual, reference)
        self.assertGreater(sensitivity, MIN_PAIRING_SENSITIVITY)
        # With the pairing destroyed, the residual is (truth + noise) - truth',
        # i.e. two whole data spreads, so the explained variance is about -0.8.
        self.assertLess(explained_variance(shuffled_residual, reference), -0.5)

    def test_calibration_block_reports_floors_and_degeneracy(self):
        residual = np.zeros(100)
        scale = np.zeros(100)
        block = calibration_block(residual, scale, n_boot=5, seed=0)
        self.assertTrue(block["claim_is_degenerate"])
        self.assertAlmostEqual(block["coverage"]["0.68"], 1.0)
        self.assertIsNone(block["ratio_median_abs_over_scale"])
        self.assertGreater(block["binomial_floor_sd"]["0.68"], 0.0)


class CriterionTests(unittest.TestCase):
    """The five legs, each isolated with a fake whose truth is known."""

    def _measure(self, problem, model, *, seed, required_override, draws=64):
        return region_measurement(
            model,
            x_data=problem["x_data"],
            z_prior=problem["z_prior"],
            x_reference=problem["x_reference"],
            args=_args(draws=draws),
            device="cpu",
            seed=seed,
            required_override=required_override,
            encoder_multipliers=(0.0, 0.0),
            decoder_multipliers=(1.0, 0.0),
        )

    def test_calibrated_response_is_accepted(self):
        problem = _synthetic_problem(sigma_true=0.02)
        model = FakeResponse(sigma=0.02, denoise=problem["noise"])
        block = self._measure(problem, model, seed=11, required_override=None)
        self.assertAlmostEqual(block["claim"]["claim_over_required"], 1.0, delta=0.08)
        self.assertAlmostEqual(
            block["cycle_residual"]["coverage_pooled_scale"]["0.68"], 0.68, delta=0.02
        )
        self.assertAlmostEqual(
            block["cycle_residual"]["coverage"]["0.68"], 0.68, delta=0.05
        )
        self.assertGreater(block["pairing_sensitivity"], MIN_PAIRING_SENSITIVITY)
        self.assertEqual(block["verdict"], "calibrated")
        self.assertEqual(block["verdict_reasons"], [])

    def test_dead_noise_channel_is_rejected(self):
        problem = _synthetic_problem()
        model = FakeResponse(sigma=0.02, denoise=problem["noise"])
        block = region_measurement(
            model,
            x_data=problem["x_data"],
            z_prior=problem["z_prior"],
            x_reference=problem["x_reference"],
            args=_args(),
            device="cpu",
            seed=12,
            required_override=None,
            encoder_multipliers=(0.0, 0.0),
            decoder_multipliers=(0.0, 0.0),
        )
        self.assertTrue(block["claim"]["claim_is_degenerate"])
        self.assertEqual(block["verdict"], "rejected")
        self.assertIn(
            "claimed per-event scale is zero (deterministic map)", block["verdict_reasons"]
        )
        self.assertGreater(block["claim"]["across_std_gev"], 0.0)

    def test_vacuous_map_is_rejected_by_the_permutation_control_alone(self):
        problem = _synthetic_problem(sigma_true=0.02)
        const = 4.0
        model = FakeResponse(
            mean_fn=lambda mass: mass * 0.0 + const,
            inverse_fn=lambda mass: mass * 0.0 + const,
            sigma=0.02,
            denoise=problem["noise"],
        )
        block = self._measure(
            problem, model, seed=13, required_override=problem["sigma_true"]
        )
        # The claim and the coverage are deliberately made to pass, so only the
        # control leg can reject this map.
        self.assertGreater(block["cycle_residual"]["coverage"]["0.68"], 0.58)
        self.assertAlmostEqual(block["claim"]["claim_over_required"], 1.0, delta=0.05)
        self.assertLess(block["pairing_sensitivity"], MIN_PAIRING_SENSITIVITY)
        self.assertEqual(block["verdict"], "rejected")
        self.assertTrue(
            any("pairing sensitivity" in reason for reason in block["verdict_reasons"])
        )

    def test_over_claiming_channel_is_rejected(self):
        problem = _synthetic_problem(sigma_true=0.02)
        model = FakeResponse(sigma=0.08, denoise=problem["noise"])
        block = self._measure(
            problem, model, seed=14, required_override=problem["sigma_true"]
        )
        self.assertGreater(block["cycle_residual"]["coverage"]["0.68"], 0.9)
        self.assertGreater(
            abs(block["cycle_residual"]["coverage"]["0.68"] - 0.68), COVERAGE_TOLERANCE
        )
        self.assertEqual(block["verdict"], "rejected")

    def test_under_claiming_channel_is_rejected(self):
        problem = _synthetic_problem(sigma_true=0.02)
        model = FakeResponse(sigma=0.006, denoise=problem["noise"])
        block = self._measure(
            problem, model, seed=15, required_override=problem["sigma_true"]
        )
        self.assertLess(block["cycle_residual"]["coverage"]["0.68"], 0.4)
        self.assertEqual(block["verdict"], "rejected")

    def test_identity_rail_and_floor_are_reported_with_references(self):
        problem = _synthetic_problem(sigma_true=0.02)
        model = FakeResponse(sigma=0.02, denoise=problem["noise"])
        block = self._measure(problem, model, seed=16, required_override=None)
        rail = block["identity_rail"]
        floor = block["no_information_floor"]
        self.assertTrue(rail["claims_zero_spread"])
        self.assertGreater(rail["no_model_residual_robust_half_width_gev"], 0.0)
        self.assertGreater(floor["disjoint_data_residual_robust_half_width_gev"], 0.0)
        # Two independent draws of the same distribution are sqrt(2) wider.
        ratio = (
            floor["disjoint_data_residual_robust_half_width_gev"]
            / block["reference"]["cms_mass_robust_half_width_gev"]
        )
        self.assertAlmostEqual(ratio, np.sqrt(2.0), delta=0.15)
        self.assertIsNotNone(block["reduction_vs_no_model"])


if __name__ == "__main__":
    unittest.main()
