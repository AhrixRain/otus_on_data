"""Unit tests for the Gaussian log score and the robust scale estimator.

Data-free: every test builds its own tensors, so the suite runs anywhere.

Two properties are load-bearing and pinned numerically rather than asserted:

* ``test_is_minimised_at_the_true_scale`` - strict propriety. Without it the
  training term cannot make the conditional spread learnable. The first version of
  this module used a Gauss-kernel score, which failed here.
* ``test_interpolating_mean_is_the_degenerate_limit`` - the mean map, not the
  loss, is what breaks identification. If the mean can hit its own observation the
  score is minimised at ``sigma -> 0`` whatever the rule, which is the limit
  measured end to end in ``scripts_joint/single_observation_limits.py``.
"""
import math
import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (
    REPO_ROOT,
    REPO_ROOT / "scripts",
    REPO_ROOT / "scripts_sota",
    REPO_ROOT / "scripts_joint",
    REPO_ROOT / "scripts_joint" / "upsilon",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from scoring_rules import (  # noqa: E402
    gaussian_log_score,
    gaussian_log_score_from_draws,
    median_bandwidth,
)

torch.manual_seed(20261004)


def draws_with_scale(sigma: float, events: int, draws: int, *, seed: int):
    generator = torch.Generator().manual_seed(seed)
    return sigma * torch.randn(draws, events, 1, dtype=torch.float64, generator=generator)


class MedianBandwidthTests(unittest.TestCase):
    def test_recovers_the_standard_deviation_in_one_dimension(self):
        values = 2.5 * torch.randn(20000, 1, dtype=torch.float64)
        self.assertAlmostEqual(float(median_bandwidth(values)), 2.5, delta=0.15)

    def test_recovers_the_standard_deviation_in_two_dimensions(self):
        values = 1.7 * torch.randn(20000, 2, dtype=torch.float64)
        self.assertAlmostEqual(float(median_bandwidth(values)), 1.7, delta=0.2)

    def test_is_scale_equivariant(self):
        values = torch.randn(4000, 3, dtype=torch.float64)
        base = float(median_bandwidth(values))
        self.assertAlmostEqual(float(median_bandwidth(values * 10.0)) / base, 10.0, places=3)

    def test_is_stable_at_small_sample_size(self):
        small = torch.randn(3000, 1, dtype=torch.float64)
        large = torch.randn(60000, 1, dtype=torch.float64)
        self.assertAlmostEqual(
            float(median_bandwidth(small)) / float(median_bandwidth(large)), 1.0, delta=0.1
        )

    def test_clamps_a_degenerate_ensemble(self):
        self.assertGreater(float(median_bandwidth(torch.zeros(100, 1, dtype=torch.float64))), 0.0)

    def test_handles_a_single_event(self):
        self.assertGreater(float(median_bandwidth(torch.zeros(1, 1))), 0.0)


class GaussianLogScoreTests(unittest.TestCase):
    def test_is_minimised_at_the_true_scale(self):
        """Strict propriety: the minimum sits at sigma_hat = sigma."""
        sigma, events = 1.0, 20000
        generator = torch.Generator().manual_seed(7)
        residual = sigma * torch.randn(events, 1, dtype=torch.float64, generator=generator)
        mean = torch.zeros_like(residual)
        scores = {
            scale: float(
                gaussian_log_score(mean, torch.full_like(residual, scale), residual)
            )
            for scale in (0.2, 0.5, 1.0, 1.5, 2.0, 4.0)
        }
        best = min(scores, key=scores.get)
        self.assertAlmostEqual(best, 1.0, delta=0.01, msg=str(scores))

    def test_matches_the_closed_form(self):
        """E[S] = sigma_truth^2 / (2 sigma_hat^2) + log(sigma_hat) for a correct mean.

        With a unit-variance target and a predicted scale ``s`` the first term is
        ``1 / (2 s^2)``, so a too-small scale is penalised like ``1 / s^2``.
        """
        generator = torch.Generator().manual_seed(3)
        residual = torch.randn(200000, 1, dtype=torch.float64, generator=generator)
        mean = torch.zeros_like(residual)
        for scale in (0.5, 1.0, 2.0):
            value = float(gaussian_log_score(mean, torch.full_like(residual, scale), residual))
            expected = 1.0 / (2.0 * scale**2) + math.log(scale)
            self.assertAlmostEqual(value, expected, delta=0.02)

    def test_is_zero_residual_plus_log_scale(self):
        mean = torch.zeros(4, 1, dtype=torch.float64)
        target = torch.zeros(4, 1, dtype=torch.float64)
        self.assertAlmostEqual(
            float(gaussian_log_score(mean, torch.tensor(0.5, dtype=torch.float64), target)),
            math.log(0.5),
            places=6,
        )

    def test_penalises_collapse_with_a_wide_target(self):
        generator = torch.Generator().manual_seed(5)
        target = torch.randn(10000, 1, dtype=torch.float64, generator=generator)
        mean = torch.zeros_like(target)
        collapsed = float(gaussian_log_score(mean, torch.full_like(target, 1e-6), target))
        correct = float(gaussian_log_score(mean, torch.full_like(target, 1.0), target))
        self.assertGreater(collapsed, correct)

    def test_interpolating_mean_is_the_degenerate_limit(self):
        """With mu = x only log(sigma) is left, so sigma -> 0 wins."""
        generator = torch.Generator().manual_seed(11)
        target = torch.randn(1000, 1, dtype=torch.float64, generator=generator)
        scores = [
            float(gaussian_log_score(target, torch.full_like(target, scale), target))
            for scale in (1.0, 0.1, 0.01)
        ]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_propagates_gradients_to_the_scale(self):
        residual = torch.randn(512, 1, dtype=torch.float64)
        mean = torch.zeros_like(residual)
        scale = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        gaussian_log_score(mean, scale.expand_as(residual), residual).backward()
        self.assertIsNotNone(scale.grad)
        self.assertNotEqual(float(scale.grad), 0.0)

    def test_from_draws_recovers_the_moments(self):
        generator = torch.Generator().manual_seed(13)
        draws = 1.5 * torch.randn(8, 4000, 1, dtype=torch.float64, generator=generator)
        target = 1.5 * torch.randn(4000, 1, dtype=torch.float64, generator=generator)
        value = float(gaussian_log_score_from_draws(draws, target))
        direct = float(
            gaussian_log_score(
                draws.mean(dim=0),
                draws.std(dim=0, unbiased=False),
                target,
            )
        )
        self.assertAlmostEqual(value, direct, places=10)

    def test_from_draws_requires_two_draws(self):
        with self.assertRaises(ValueError):
            gaussian_log_score_from_draws(torch.zeros(1, 10, 1), torch.zeros(10, 1))

    def test_rejects_mismatched_event_axes(self):
        with self.assertRaises(ValueError):
            gaussian_log_score_from_draws(torch.zeros(3, 11, 1), torch.zeros(10, 1))

    def test_keeps_the_input_dtype(self):
        residual = torch.randn(64, 1, dtype=torch.float32)
        value = gaussian_log_score(torch.zeros_like(residual), torch.ones_like(residual), residual)
        self.assertEqual(value.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
