"""Unit tests for the S5 systematics study (data-free core)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (
    REPO_ROOT,
    REPO_ROOT / "scripts",
    REPO_ROOT / "scripts_sota",
    REPO_ROOT / "scripts_joint",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from systematics_study import (  # noqa: E402
    COMPONENT,
    bootstrap_median,
    claim_verdict,
    combine_systematics,
    region_ratio,
    spread,
    state_stats,
)


class SpreadTests(unittest.TestCase):
    def test_known_values(self):
        result = spread([1.0, 2.0, 3.0])
        self.assertEqual(result["n"], 3)
        self.assertAlmostEqual(result["mean"], 2.0)
        self.assertAlmostEqual(result["std"], 1.0, places=9)
        self.assertAlmostEqual(result["half_range"], 1.0)

    def test_filters_none_and_nan(self):
        result = spread([1.0, None, float("nan"), 3.0])
        self.assertEqual(result["n"], 2)
        self.assertAlmostEqual(result["mean"], 2.0)

    def test_empty_is_reported_not_zero(self):
        result = spread([None, float("nan")])
        self.assertEqual(result["n"], 0)
        self.assertIsNone(result["mean"])

    def test_single_value_has_zero_width(self):
        result = spread([5.0])
        self.assertEqual(result["std"], 0.0)
        self.assertEqual(result["half_range"], 0.0)


class CombineTests(unittest.TestCase):
    def test_max_half_range_and_quadrature(self):
        a = spread([1.0, 3.0])      # half_range 1.0
        b = spread([10.0, 10.5])    # half_range 0.25
        total = combine_systematics(a, b)
        self.assertAlmostEqual(total["half_range_max"], 1.0)
        expected = (a["std"] ** 2 + b["std"] ** 2) ** 0.5
        self.assertAlmostEqual(total["quadrature_std"], expected, places=9)
        self.assertEqual(len(total["axes"]), 2)

    def test_empty_inputs_are_safe(self):
        total = combine_systematics(spread([]), {"half_range": None, "std": None})
        self.assertIsNone(total["half_range_max"])


class ClaimVerdictTests(unittest.TestCase):
    def test_robust_when_the_budget_fits(self):
        total = combine_systematics(spread([0.0, 4.0]))  # half range 2
        verdict = claim_verdict(3.0, 0.0, 10.0, total)
        self.assertEqual(verdict["status"], "robust")
        self.assertAlmostEqual(verdict["budget"], 5.0)

    def test_fragile_when_the_budget_overflows(self):
        total = combine_systematics(spread([0.0, 20.0]))  # half range 10
        verdict = claim_verdict(25.0, 0.0, 30.0, total)   # 25 + 10 > 30
        self.assertEqual(verdict["status"], "fragile")
        self.assertLess(verdict["headroom"], 0.0)

    def test_unmeasured_without_a_spread(self):
        verdict = claim_verdict(None, 0.0, 30.0, {"half_range_max": None})
        self.assertEqual(verdict["status"], "not_measured")


class BootstrapTests(unittest.TestCase):
    def test_is_deterministic_for_a_seed(self):
        rng = np.random.default_rng(3)
        values = rng.normal(10.0, 0.1, 500)
        first = bootstrap_median(values, draws=50, seed=11)
        second = bootstrap_median(values, draws=50, seed=11)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["median"], float(np.median(values)), places=12)
        self.assertGreater(first["std"], 0.0)

    def test_different_seeds_give_different_draws(self):
        rng = np.random.default_rng(4)
        values = rng.normal(10.0, 0.1, 500)
        a = bootstrap_median(values, draws=50, seed=1)
        b = bootstrap_median(values, draws=50, seed=2)
        self.assertNotEqual(a["std"], b["std"])


class StateStatsTests(unittest.TestCase):
    def _sample(self):
        rng = np.random.default_rng(5)
        masses = np.concatenate([rng.normal(9.46, 0.10, 4000), rng.normal(10.02, 0.12, 3000)])
        component = np.concatenate(
            [np.full(4000, COMPONENT["upsilon1s"]), np.full(3000, COMPONENT["upsilon2s"])]
        )
        return masses, component

    def test_separates_states(self):
        masses, component = self._sample()
        stats = state_stats(masses, component, bootstrap_draws=40, seed=7)
        self.assertEqual(stats["upsilon1s"]["events"], 4000)
        self.assertEqual(stats["upsilon2s"]["events"], 3000)
        self.assertEqual(stats["upsilon3s"]["events"], 0)
        # median - CMS: 1S is 15 MeV above the CMS fit at 9.445 GeV
        self.assertAlmostEqual(stats["upsilon1s"]["median_minus_cms_mev"], 15.0, delta=5.0)
        self.assertGreater(stats["upsilon1s"]["median_bootstrap_std_mev"], 0.0)

    def test_window_efficiency_is_bounded(self):
        masses, component = self._sample()
        stats = state_stats(masses, component, bootstrap_draws=20, seed=7)
        self.assertGreaterEqual(stats["upsilon1s"]["window_efficiency"], 0.0)
        self.assertLessEqual(stats["upsilon1s"]["window_efficiency"], 1.0)


class RegionRatioTests(unittest.TestCase):
    def test_identical_distributions_ratio_one(self):
        rng = np.random.default_rng(9)
        values = rng.uniform(8.5, 11.2, 20000)
        self.assertAlmostEqual(region_ratio(values, values.copy()), 1.0, places=6)

    def test_low_mass_excess_gives_ratio_above_one(self):
        rng = np.random.default_rng(9)
        reference = rng.uniform(8.5, 11.2, 20000)
        shifted = np.concatenate([rng.uniform(8.5, 9.0, 6000), rng.uniform(9.0, 11.2, 14000)])
        self.assertGreater(region_ratio(shifted, reference), 1.0)


if __name__ == "__main__":
    unittest.main()
