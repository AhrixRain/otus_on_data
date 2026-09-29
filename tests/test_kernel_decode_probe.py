"""Unit tests for the A2.3 decode-only kernel calibration probe."""

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

from kernel_decode_probe import apply_scale, recommend_scale, within_metrics  # noqa: E402


class ApplyScaleTests(unittest.TestCase):
    def test_sets_scale_without_mutating_input(self):
        spec = {"eta_bins": [0.0, 1.0], "logpt_a": [0.01], "logpt_b": [0.5], "scale": 1.0}
        scaled = apply_scale(spec, 2.5)
        self.assertEqual(scaled["scale"], 2.5)
        self.assertEqual(spec["scale"], 1.0)
        self.assertIsNot(scaled, spec)


class WithinMetricsTests(unittest.TestCase):
    def test_decomposition_and_fraction(self):
        rng = np.random.default_rng(5)
        events, draws = 2000, 200
        across = rng.normal(scale=1.5, size=events)
        sigma = 0.4
        matrix = across[:, None] + sigma * rng.normal(size=(events, draws))
        metrics = within_metrics(matrix)
        expected_fraction = sigma**2 / (sigma**2 + across.var(ddof=0))
        self.assertAlmostEqual(metrics["noise_variance_fraction"], expected_fraction, places=2)
        self.assertAlmostEqual(metrics["within_std_median_gev"], sigma, places=2)
        self.assertAlmostEqual(metrics["sigma_only_within_gev"], sigma, places=2)

    def test_zero_variance_matrix(self):
        matrix = np.zeros((10, 4))
        metrics = within_metrics(matrix)
        self.assertEqual(metrics["within_std_median_gev"], 0.0)
        self.assertEqual(metrics["noise_variance_fraction"], 0.0)


class RecommendScaleTests(unittest.TestCase):
    def test_interpolates_the_crossing(self):
        points = [(0.5, 0.05), (1.0, 0.10), (2.0, 0.20)]
        self.assertAlmostEqual(recommend_scale(points, 0.125), 1.25, places=6)

    def test_target_below_all_returns_smallest_scale(self):
        points = [(0.5, 0.05), (1.0, 0.10)]
        self.assertAlmostEqual(recommend_scale(points, 0.01), 0.5)

    def test_target_above_all_returns_closest(self):
        points = [(0.5, 0.05), (2.0, 0.11)]
        self.assertAlmostEqual(recommend_scale(points, 1.0), 2.0)

    def test_unsorted_input_is_sorted(self):
        points = [(2.0, 0.20), (0.5, 0.05), (1.0, 0.10)]
        self.assertAlmostEqual(recommend_scale(points, 0.15), 1.5, places=6)

    def test_empty_input_rejected(self):
        with self.assertRaises(ValueError):
            recommend_scale([], 1.0)


if __name__ == "__main__":
    unittest.main()
