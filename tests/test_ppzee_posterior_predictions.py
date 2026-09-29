"""Tests for the D4 ppzee posterior-calibration logic (pure numerics)."""

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
    REPO_ROOT / "scripts_joint" / "upsilon",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from paired_closure import direct_mass, paired_closure  # noqa: E402
from ppzee_posterior_predictions import (  # noqa: E402
    calibration_verdict,
    draw_mass_matrix,
    parse_settings,
    posterior_calibration,
    validate_draws,
)


class ParseSettingsTests(unittest.TestCase):
    def test_native_resolves_to_checkpoint_values(self):
        settings = parse_settings("native", 0.984, 0.0)
        self.assertEqual(settings, [{"label": "native", "core": 0.984, "tail": 0.0}])

    def test_explicit_pair_and_combination(self):
        settings = parse_settings("native,1.0:0.25", 0.5, 0.0)
        self.assertEqual(len(settings), 2)
        self.assertEqual(settings[0]["label"], "native")
        self.assertEqual(settings[1]["label"], "core1_tail0.25")
        self.assertAlmostEqual(settings[1]["core"], 1.0)
        self.assertAlmostEqual(settings[1]["tail"], 0.25)

    def test_rejects_unknown_negative_duplicate_and_empty(self):
        with self.assertRaises(ValueError):
            parse_settings("banana", 1.0, 0.0)
        with self.assertRaises(ValueError):
            parse_settings("-1:0", 1.0, 0.0)
        with self.assertRaises(ValueError):
            parse_settings("native,native", 1.0, 0.0)
        with self.assertRaises(ValueError):
            parse_settings(" , ", 1.0, 0.0)


class DrawAggregationTests(unittest.TestCase):
    def test_validate_accepts_draws_n_8(self):
        draws = np.zeros((4, 7, 8), dtype=np.float32)
        out = validate_draws(draws, 7)
        self.assertEqual(out.shape, (4, 7, 8))

    def test_validate_rejects_bad_shapes_and_values(self):
        with self.assertRaises(ValueError):
            validate_draws(np.zeros((4, 7)), 7)
        with self.assertRaises(ValueError):
            validate_draws(np.zeros((4, 6, 8)), 7)
        with self.assertRaises(ValueError):
            validate_draws(np.zeros((4, 7, 9)), 7)
        with self.assertRaises(ValueError):
            validate_draws(np.zeros((1, 7, 8)), 7)  # one draw cannot give a pull
        bad = np.zeros((4, 7, 8))
        bad[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            validate_draws(bad, 7)

    def test_draw_mass_matrix_uses_the_first_component(self):
        draws = np.arange(2 * 5 * 8, dtype=np.float64).reshape(2, 5, 8)
        masses = draw_mass_matrix(draws, mass_fn=lambda values: values[:, 0])
        np.testing.assert_allclose(masses, draws[:, :, 0])
        self.assertEqual(masses.shape, (2, 5))

    def test_draw_mass_matrix_rejects_non_eight_columns(self):
        with self.assertRaises(ValueError):
            draw_mass_matrix(np.zeros((2, 5, 4)))


class PosteriorCalibrationTests(unittest.TestCase):
    @staticmethod
    def _calibrated_sample(seed: int, n_events: int, draws: int, sigma: float = 1.0):
        rng = np.random.default_rng(seed)
        center = rng.normal(size=n_events)
        # truth is a draw from the same posterior that produces the draws
        truth = center + sigma * rng.normal(size=n_events)
        draw_mass = center[None, :] + sigma * rng.normal(size=(draws, n_events))
        return truth, draw_mass

    def test_calibrated_posterior_has_unit_pull_and_nominal_coverage(self):
        truth, draw_mass = self._calibrated_sample(3, n_events=40000, draws=64)
        stats = posterior_calibration(truth, draw_mass)
        self.assertAlmostEqual(stats["pull_mean"], 0.0, delta=0.03)
        # finite-draw correction: pull ~ sqrt(1 + 1/D) * t_{D-1}
        expected_pull_std = np.sqrt(1.0 + 1.0 / 64.0) * np.sqrt(63.0 / 61.0)
        self.assertAlmostEqual(stats["pull_std"], expected_pull_std, delta=0.05)
        self.assertAlmostEqual(stats["coverage_1sigma"], 0.6827, delta=0.02)
        self.assertFalse(stats["degenerate"])
        self.assertEqual(stats["zero_variance_events"], 0)

    def test_overconfident_posterior_shows_large_pull_std_and_low_coverage(self):
        rng = np.random.default_rng(5)
        n_events, draws, sigma = 40000, 64, 1.0
        center = rng.normal(size=n_events)
        truth = center + sigma * rng.normal(size=n_events)
        # posterior stated 2x too narrow
        draw_mass = center[None, :] + 0.5 * sigma * rng.normal(size=(draws, n_events))
        stats = posterior_calibration(truth, draw_mass)
        self.assertGreater(stats["pull_std"], 1.8)
        self.assertLess(stats["coverage_1sigma"], 0.45)
        self.assertIn("over-confident", calibration_verdict(stats))

    def test_biased_posterior_shows_shifted_pull_mean(self):
        rng = np.random.default_rng(7)
        n_events, draws = 40000, 64
        center = rng.normal(size=n_events)
        truth = center + 0.5 + rng.normal(size=n_events)
        draw_mass = center[None, :] + rng.normal(size=(draws, n_events))
        stats = posterior_calibration(truth, draw_mass)
        self.assertAlmostEqual(stats["pull_mean"], -0.5, delta=0.05)
        self.assertIn("biased", calibration_verdict(stats))

    def test_zero_variance_posterior_is_finite_and_degenerate(self):
        n_events, draws = 1000, 8
        truth = np.linspace(1.0, 2.0, n_events)
        constant = np.zeros((draws, n_events))
        stats = posterior_calibration(truth, constant)
        for key in ("pull_mean", "pull_std", "coverage_1sigma", "mean_residual_rms"):
            self.assertTrue(np.isfinite(stats[key]), key)
        self.assertEqual(stats["posterior_std_median_gev"], 0.0)
        self.assertEqual(stats["zero_variance_events"], n_events)
        self.assertTrue(stats["degenerate"])
        self.assertEqual(stats["coverage_1sigma"], 0.0)
        self.assertIn("degenerate", calibration_verdict(stats))

    def test_numerically_zero_variance_at_realistic_scale_is_degenerate(self):
        # Identical draws at ~91 GeV can carry a ~1e-14 float64 width from
        # summation rounding in the variance, so the degenerate branch must use
        # an absolute tolerance rather than an exact zero test.
        n_events, draws = 500, 8
        truth = np.full(n_events, 91.5)
        constant = np.full((draws, n_events), 91.0)
        stats = posterior_calibration(truth, constant)
        self.assertTrue(stats["degenerate"])
        self.assertEqual(stats["zero_variance_events"], n_events)
        self.assertIn("degenerate", calibration_verdict(stats))
        for key in ("pull_mean", "pull_std", "coverage_1sigma"):
            self.assertTrue(np.isfinite(stats[key]), key)

    def test_zero_variance_with_exact_truth_is_covered(self):
        truths = np.array([1.0, 2.0, 3.0])
        draws = np.vstack([truths, truths, truths])
        stats = posterior_calibration(truths, draws)
        self.assertTrue(np.isfinite(stats["pull_mean"]))
        self.assertEqual(stats["coverage_1sigma"], 1.0)
        self.assertTrue(stats["degenerate"])

    def test_single_draw_is_rejected_by_validate_but_not_nan_in_helper(self):
        with self.assertRaises(ValueError):
            validate_draws(np.zeros((1, 5, 8)), 5)
        stats = posterior_calibration(np.arange(5.0), np.arange(5.0)[None, :])
        self.assertTrue(np.isfinite(stats["pull_mean"]))
        self.assertTrue(stats["degenerate"])


class PairClosureConsistencyTests(unittest.TestCase):
    def test_helper_matches_paired_closure_posterior_block(self):
        rng = np.random.default_rng(11)
        n_events, draws = 300, 6
        z_true = rng.normal(size=(n_events, 8))
        x_input = rng.normal(size=(n_events, 8))
        z_pred = rng.normal(size=(n_events, 8))
        draws_array = rng.normal(size=(draws, n_events, 8))
        report = paired_closure(z_true, x_input, z_pred, z_pred_draws=draws_array)
        local = posterior_calibration(direct_mass(z_true), draw_mass_matrix(draws_array))
        for key in (
            "mean_residual_rms",
            "posterior_std_median_gev",
            "pull_mean",
            "pull_std",
            "coverage_1sigma",
        ):
            self.assertAlmostEqual(
                report["posterior"][key],
                local[key],
                places=10,
                msg=f"{key} disagrees with paired_closure",
            )


if __name__ == "__main__":
    unittest.main()
