"""Tests for the A2.0 resolution-kernel calibration pure functions."""

from __future__ import annotations

import math
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

from resolution_kernel_calibration import (  # noqa: E402
    fit_ab,
    fit_power_law,
    logpt_amplitude,
    merge_pt_groups,
    quadrature_resolution,
    resolution_from_logpt,
    robust_half_width,
    validate_kernel_spec,
)


class RobustWidthTests(unittest.TestCase):
    def test_matches_normal_quantiles(self):
        rng = np.random.default_rng(7)
        self.assertAlmostEqual(robust_half_width(rng.normal(size=200_000)), 0.9945, places=2)

    def test_two_points(self):
        self.assertAlmostEqual(robust_half_width([0.0, 1.0]), 0.34, places=10)

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            robust_half_width([])


class ConversionTests(unittest.TestCase):
    def test_quadrature(self):
        self.assertAlmostEqual(quadrature_resolution(2.0, 1.0), math.sqrt(3.0), places=12)
        self.assertEqual(quadrature_resolution(1.0, 2.0), 0.0)

    def test_quadrature_rejects_non_finite(self):
        with self.assertRaises(ValueError):
            quadrature_resolution(float("nan"), 1.0)

    def test_logpt_amplitude_and_inverse(self):
        value = logpt_amplitude(0.084, 9.4603)
        self.assertAlmostEqual(value, 0.012557, places=5)
        self.assertAlmostEqual(resolution_from_logpt(value, 9.4603), 0.084, places=12)

    def test_logpt_amplitude_rejects_bad_mass(self):
        with self.assertRaises(ValueError):
            logpt_amplitude(0.01, 0.0)


class FitTests(unittest.TestCase):
    def test_fit_ab_recovers_known_parameters(self):
        pT = np.array([4.0, 6.0, 10.0, 20.0, 40.0, 80.0])
        a_true, b_true = 0.008, 0.05
        sigma = np.sqrt(a_true**2 + (b_true / pT) ** 2)
        fit = fit_ab(list(zip(pT, sigma)))
        self.assertAlmostEqual(fit["a"], a_true, places=4)
        self.assertAlmostEqual(fit["b"], b_true, places=3)
        self.assertLess(fit["rms"], 1e-6)

    def test_fit_ab_degenerate_constant_marks_bound(self):
        pT = np.array([4.0, 10.0, 40.0])
        sigma = np.full_like(pT, 0.01)
        fit = fit_ab(list(zip(pT, sigma)))
        self.assertTrue(fit["b_at_bound"])

    def test_fit_ab_needs_two_points(self):
        with self.assertRaises(ValueError):
            fit_ab([(4.0, 0.01)])

    def test_fit_power_law_recovers_known_parameters(self):
        pT = np.array([4.0, 6.0, 10.0, 20.0, 40.0])
        c_true, alpha_true = 0.01, 0.5
        sigma = c_true * (pT / 10.0) ** alpha_true
        fit = fit_power_law(list(zip(pT, sigma)))
        self.assertIsNotNone(fit)
        self.assertAlmostEqual(fit["c"], c_true, places=8)
        self.assertAlmostEqual(fit["alpha"], alpha_true, places=8)

    def test_fit_power_law_needs_two_points(self):
        self.assertIsNone(fit_power_law([(4.0, 0.01)]))


class MergeTests(unittest.TestCase):
    def test_merges_short_leading_and_trailing_groups(self):
        groups = merge_pt_groups(
            [10, 200, 300, 5], [5, 80, 150, 5], min_x=250, min_z=100
        )
        self.assertEqual(groups, [(0, 3)])

    def test_empty_row_returns_empty(self):
        self.assertEqual(merge_pt_groups([0, 0], [0, 0], min_x=1, min_z=1), [])

    def test_length_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            merge_pt_groups([1, 2], [1], min_x=1, min_z=1)


class KernelSpecTests(unittest.TestCase):
    def _spec(self, **overrides):
        spec = {
            "schema_version": 1,
            "eta_bins": [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4],
            "logpt_a": [0.001] * 6,
            "logpt_b": [0.01] * 6,
            "phi_a": [0.0] * 6,
            "phi_b": [0.0] * 6,
            "eta_a": [0.0] * 6,
            "eta_b": [0.0] * 6,
            "tail_ratio": 0.25,
            "units": {"logpt": "log(pT/GeV)", "logpt_b": "GeV"},
            "provenance": {"method": "test"},
        }
        spec.update(overrides)
        return spec

    def test_valid_spec_passes(self):
        validate_kernel_spec(self._spec())

    def test_missing_key_rejected(self):
        spec = self._spec()
        del spec["logpt_a"]
        with self.assertRaises(ValueError):
            validate_kernel_spec(spec)

    def test_wrong_length_rejected(self):
        with self.assertRaises(ValueError):
            validate_kernel_spec(self._spec(logpt_b=[0.01] * 5))

    def test_negative_value_rejected(self):
        with self.assertRaises(ValueError):
            validate_kernel_spec(self._spec(phi_a=[-1.0] + [0.0] * 5))

    def test_bad_units_rejected(self):
        with self.assertRaises(ValueError):
            validate_kernel_spec(self._spec(units={"logpt": "GeV", "logpt_b": "GeV"}))

    def test_bad_tail_ratio_rejected(self):
        with self.assertRaises(ValueError):
            validate_kernel_spec(self._spec(tail_ratio=1.5))

    def test_non_monotone_eta_rejected(self):
        with self.assertRaises(ValueError):
            validate_kernel_spec(
                self._spec(eta_bins=[0.0, 0.8, 0.4, 1.2, 1.6, 2.0, 2.4])
            )


if __name__ == "__main__":
    unittest.main()
