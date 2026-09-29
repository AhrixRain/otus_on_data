"""Unit tests for the A0 fixed-z noise-budget estimator (pure numerics)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (
    REPO_ROOT / "scripts",
    REPO_ROOT / "scripts_sota",
    REPO_ROOT / "scripts_joint",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from fixed_z_noise_budget import (  # noqa: E402
    deterministic_statistics,
    draw_statistics,
    merge_budget,
    robust_half_width,
)


class RobustHalfWidthTests(unittest.TestCase):
    def test_matches_normal_quantiles(self):
        rng = np.random.default_rng(7)
        sample = rng.normal(size=200_000)
        self.assertAlmostEqual(float(robust_half_width(sample)), 0.9945, places=2)

    def test_axis_reduction(self):
        values = np.array([[0.0, 1.0], [2.0, 3.0]])
        widths = robust_half_width(values, axis=1)
        self.assertEqual(widths.shape, (2,))
        np.testing.assert_allclose(widths, [0.34, 0.34], atol=1e-12)


class DrawStatisticsTests(unittest.TestCase):
    def test_variance_decomposition_and_noise_fraction(self):
        rng = np.random.default_rng(11)
        events, draws = 4000, 200
        across = rng.normal(scale=2.0, size=events)
        sigma = 0.7
        matrix = across[:, None] + sigma * rng.normal(size=(events, draws))
        stats = draw_statistics(matrix)
        expected_fraction = sigma**2 / (sigma**2 + across.var(ddof=0))
        self.assertAlmostEqual(stats["noise_variance_fraction"], expected_fraction, places=2)
        self.assertAlmostEqual(
            stats["mean_map_variance_fraction"] + stats["noise_variance_fraction"],
            1.0,
            places=10,
        )
        self.assertAlmostEqual(stats["within_std_median_gev"], sigma, places=2)
        self.assertAlmostEqual(stats["conditional_mean_std_gev"], float(across.std(ddof=0)), places=1)

    def test_total_variance_is_within_plus_across(self):
        rng = np.random.default_rng(3)
        matrix = rng.normal(size=(500, 9))
        stats = draw_statistics(matrix)
        self.assertAlmostEqual(
            stats["total_variance_gev2"],
            stats["within_variance_mean_gev2"] + stats["across_variance_gev2"],
            places=10,
        )

    def test_rejects_single_draw(self):
        with self.assertRaises(ValueError):
            draw_statistics(np.zeros((4, 1)))


class DeterministicAndMergeTests(unittest.TestCase):
    def test_deterministic_statistics_have_zero_within_width(self):
        stats = deterministic_statistics(np.array([1.0, 2.0, 3.0, 4.0]))
        self.assertEqual(stats["within_std_median_gev"], 0.0)
        self.assertEqual(stats["within_variance_mean_gev2"], 0.0)
        self.assertAlmostEqual(stats["ensemble_std_gev"], np.std([1.0, 2.0, 3.0, 4.0]))

    def test_merge_budget_reports_both_estimators(self):
        draw = {
            "ensemble_std_gev": 1.3,
            "within_variance_mean_gev2": 0.25,
            "within_std_median_gev": 0.5,
        }
        zero = {"ensemble_std_gev": 1.2}
        merged = merge_budget(draw, zero)
        self.assertAlmostEqual(merged["sigma_noise_only_gev"], np.sqrt(1.3**2 - 1.2**2))
        self.assertAlmostEqual(merged["sigma_noise_only_from_within_gev"], 0.5)
        self.assertEqual(merged["within_std_median_gev"], 0.5)

    def test_merge_budget_clamps_negative_ensemble_difference(self):
        draw = {"ensemble_std_gev": 1.0, "within_variance_mean_gev2": 0.0, "within_std_median_gev": 0.0}
        zero = {"ensemble_std_gev": 1.5}
        merged = merge_budget(draw, zero)
        self.assertEqual(merged["sigma_noise_only_gev"], 0.0)


if __name__ == "__main__":
    unittest.main()
