"""Unit tests for the per-event paired closure metrics (NumPy only)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (REPO_ROOT / "scripts_joint",):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from paired_closure import direct_mass, paired_closure  # noqa: E402


def _pairs(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Truth pairs and a smeared detector-level partner for each."""
    z = np.zeros((n, 8))
    pt = rng.uniform(20.0, 60.0, size=n)
    phi = rng.uniform(-np.pi, np.pi, size=n)
    z[:, 0] = pt * np.cos(phi)
    z[:, 1] = pt * np.sin(phi)
    z[:, 2] = rng.normal(0.0, 30.0, size=n)
    z[:, 4] = -pt * np.cos(phi)
    z[:, 5] = -pt * np.sin(phi)
    z[:, 6] = rng.normal(0.0, 30.0, size=n)
    for start in (0, 4):
        z[:, start + 3] = np.sqrt(np.sum(z[:, start : start + 3] ** 2, axis=1))
    x = z.copy()
    x[:, [0, 1, 2, 4, 5, 6]] *= 1.0 + 0.02 * rng.standard_normal((n, 6))
    for start in (0, 4):
        x[:, start + 3] = np.sqrt(np.sum(x[:, start : start + 3] ** 2, axis=1))
    return z, x


class PairedClosureTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20260905)
        self.z, self.x = _pairs(rng, 5000)

    def test_perfect_inverse_scores_zero(self):
        report = paired_closure(self.z, self.x, self.z)
        self.assertAlmostEqual(report["per_event_mass"]["model_rms"], 0.0, places=9)
        self.assertAlmostEqual(
            report["per_event_mass"]["residual_rms_vs_identity"], 0.0, places=9
        )

    def test_identity_map_scores_exactly_one(self):
        report = paired_closure(self.z, self.x, self.x)
        self.assertAlmostEqual(
            report["per_event_mass"]["residual_rms_vs_identity"], 1.0, places=9
        )
        for value in report["per_event_components"]["rms_vs_identity_per_column"]:
            self.assertAlmostEqual(value, 1.0, places=9)

    def test_a_map_that_matches_the_marginal_can_still_fail_per_event(self):
        """The whole reason this test exists: shuffling preserves the marginal."""
        rng = np.random.default_rng(5)
        shuffled = self.z[rng.permutation(len(self.z))]
        report = paired_closure(self.z, self.x, shuffled)
        self.assertLess(report["marginal_mass"]["model_ks"], 0.05)
        self.assertGreater(report["per_event_mass"]["residual_rms_vs_identity"], 1.0)

    def test_cycle_mode_without_identity_reference(self):
        report = paired_closure(self.x, None, self.x)
        self.assertFalse(report["has_identity_reference"])
        self.assertNotIn("residual_rms_vs_identity", report["per_event_mass"])

    def test_posterior_coverage_is_reported(self):
        rng = np.random.default_rng(9)
        draws = np.stack(
            [
                self.z * (1.0 + 0.002 * rng.standard_normal(self.z.shape))
                for _ in range(16)
            ]
        )
        report = paired_closure(
            self.z, self.x, draws.mean(axis=0), z_pred_draws=draws
        )
        self.assertEqual(report["posterior"]["draws"], 16)
        self.assertGreater(report["posterior"]["coverage_1sigma"], 0.0)
        self.assertLessEqual(report["posterior"]["coverage_1sigma"], 1.0)

    def test_event_alignment_is_enforced(self):
        with self.assertRaises(ValueError):
            paired_closure(self.z, self.x[:10], self.z)

    def test_mass_helper_matches_the_direct_formula(self):
        pair = self.z[:, 0:4] + self.z[:, 4:8]
        expected = np.sqrt(
            np.clip(pair[:, 3] ** 2 - np.sum(pair[:, 0:3] ** 2, axis=1), 0.0, None)
        )
        np.testing.assert_allclose(direct_mass(self.z), expected, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
