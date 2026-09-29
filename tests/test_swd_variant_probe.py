"""Unit tests for the C1 sliced-Wasserstein estimator probe (non-training)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (
    REPO_ROOT / "scripts",
    REPO_ROOT / "scripts_sota",
    REPO_ROOT / "scripts_joint",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from loss import sliced_wasserstein  # noqa: E402
from swd_variant_probe import (  # noqa: E402
    DEFAULT_CONFIG,
    aggregate_variant,
    detection_power,
    hard_direction_sample,
    json_ready,
    learned_max_swd,
    mass8,
    random_swd,
    resonance_sample,
    run_probe,
    scale_mass,
    smear_cylindrical,
    summarize,
    swd_with_directions,
)


def _tiny_config() -> dict:
    config = dict(DEFAULT_CONFIG)
    config.update(
        {
            "scenarios": ["hard_direction"],
            "sample_sizes": [96],
            "replicates": 2,
            "random_slices": [16],
            "fixed_random_slices": 16,
            "orthogonal_slices": 4,
            "max_directions": [8],
            "max_steps": 5,
        }
    )
    return config


class SyntheticDataTests(unittest.TestCase):
    def test_resonance_sample_has_the_requested_mass_peak(self):
        rng = np.random.default_rng(3)
        values = resonance_sample(rng, 8192, mass_gev=3.0969, width_gev=0.005)
        masses = mass8(values)
        self.assertAlmostEqual(float(masses.mean()), 3.0969, places=4)
        self.assertLess(abs(float(masses.std()) - 0.005), 1.5e-3)

    def test_smearing_broadens_the_mass(self):
        rng = np.random.default_rng(4)
        truth = resonance_sample(rng, 4096, width_gev=0.005)
        detector = smear_cylindrical(truth, np.random.default_rng(5))
        self.assertGreater(float(mass8(detector).std()), 3.0 * float(mass8(truth).std()))

    def test_scale_mass_shifts_the_peak_by_the_requested_amount(self):
        rng = np.random.default_rng(6)
        truth = resonance_sample(rng, 4096, width_gev=0.002)
        shifted = scale_mass(truth, 0.010, 3.0969)
        shift = float(mass8(shifted).mean() - mass8(truth).mean())
        self.assertAlmostEqual(shift, 0.010, places=4)

    def test_fixed_seed_reproducibility(self):
        first = resonance_sample(np.random.default_rng(9), 512)
        second = resonance_sample(np.random.default_rng(9), 512)
        third = resonance_sample(np.random.default_rng(10), 512)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, third))

    def test_hard_direction_sample_shift_is_exact(self):
        base, shifted, direction = hard_direction_sample(
            np.random.default_rng(11), 512, 8, 0.5
        )
        expected = np.broadcast_to(0.5 * direction[None, :], base.shape)
        np.testing.assert_allclose(shifted - base, expected, atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.norm(direction)), 1.0, places=6)


class EstimatorTests(unittest.TestCase):
    def test_identical_samples_are_zero_for_every_estimator(self):
        torch.manual_seed(0)
        values = torch.randn(64, 8)
        directions = F.normalize(torch.randn(32, 8), dim=1)
        self.assertEqual(float(sliced_wasserstein(values, values, 64, 1)), 0.0)
        self.assertEqual(float(swd_with_directions(values, values, directions, 1)), 0.0)
        self.assertEqual(float(random_swd(values, values, 64, 1, seed=1)), 0.0)

    def test_swd_with_directions_matches_the_public_random_slice_loss(self):
        torch.manual_seed(123)
        truth = torch.randn(512, 8)
        pred = torch.randn(512, 8) + 0.2
        torch.manual_seed(7)
        directions = F.normalize(torch.randn(64, 8), dim=1)
        torch.manual_seed(7)
        expected = float(sliced_wasserstein(truth, pred, 64, 1).detach())
        observed = float(swd_with_directions(truth, pred, directions, 1))
        self.assertAlmostEqual(observed, expected, places=10)

    def test_max_swd_beats_random_slices_on_a_hard_direction(self):
        torch.manual_seed(21)
        shift_direction = F.normalize(torch.arange(1.0, 9.0), dim=0)
        truth = torch.randn(256, 8)
        pred = truth + 0.5 * shift_direction
        random_values = [random_swd(truth, pred, 32, 1, seed=seed) for seed in range(6)]
        learned_values = [
            learned_max_swd(
                truth,
                pred,
                num_directions=32,
                steps=100,
                lr=0.5,
                diversity_weight=0.01,
                update_every=1,
                direction_grad_clip=1.0,
                p=1,
                seed=seed,
            )[0]
            for seed in range(2)
        ]
        random_mean = float(np.mean(random_values))
        learned_mean = float(np.mean(learned_values))
        self.assertGreaterEqual(learned_mean, random_mean)
        self.assertGreater(learned_mean, 1.5 * random_mean)

    def test_max_swd_directions_stay_normalized(self):
        torch.manual_seed(31)
        truth = torch.randn(64, 8)
        pred = torch.randn(64, 8) + 0.5
        _, directions = learned_max_swd(
            truth,
            pred,
            num_directions=8,
            steps=5,
            lr=0.5,
            diversity_weight=0.01,
            update_every=1,
            direction_grad_clip=1.0,
            p=1,
            seed=0,
        )
        norms = directions.norm(dim=1)
        np.testing.assert_allclose(norms.numpy(), np.ones(8), atol=1e-5)


class AggregationTests(unittest.TestCase):
    def test_summarize_known_values(self):
        stats = summarize([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(stats["n"], 4)
        self.assertAlmostEqual(stats["mean"], 2.5)
        self.assertAlmostEqual(stats["median"], 2.5)
        self.assertAlmostEqual(stats["std"], np.std([1.0, 2.0, 3.0, 4.0], ddof=1))
        self.assertAlmostEqual(stats["q05"], float(np.quantile([1.0, 2.0, 3.0, 4.0], 0.05)))

    def test_detection_power_is_one_when_the_shift_is_large(self):
        result = detection_power([2.0, 3.0, 4.0, 5.0, 6.0], [-1.0, 0.0, 1.0])
        self.assertEqual(result["power_empirical"], 1.0)
        self.assertEqual(result["power_gaussian"], 1.0)

    def test_detection_power_is_zero_when_the_shift_is_below_the_null(self):
        result = detection_power([-1.0, -2.0, -3.0], [0.0, 1.0, 2.0])
        self.assertEqual(result["power_empirical"], 0.0)
        self.assertEqual(result["power_gaussian"], 0.0)

    def test_aggregate_variant_reports_the_expected_fields(self):
        spec = {"name": "swd_16", "kind": "random", "num_slices": 16}
        records = [
            {"signal": 0.5, "shift": 0.6, "shift_large": 0.8, "null": 0.4, "null_b": 0.51},
            {"signal": 0.55, "shift": 0.65, "shift_large": 0.85, "null": 0.42, "null_b": 0.56},
            {"signal": 0.45, "shift": 0.55, "shift_large": 0.75, "null": 0.38, "null_b": 0.46},
        ]
        result = aggregate_variant(records, spec, DEFAULT_CONFIG)
        self.assertEqual(result["name"], "swd_16")
        self.assertEqual(result["num_slices"], 16)
        self.assertIn("signal_to_floor", result)
        self.assertIn("detection", result)
        self.assertIn("detection_large", result)
        self.assertGreater(result["signal"]["mean"], 0.0)

    def test_variant_specs_include_both_max_protocols(self):
        from swd_variant_probe import variant_specs

        names = [spec["name"] for spec in variant_specs(DEFAULT_CONFIG)]
        self.assertIn("maxswd_8_in", names)
        self.assertIn("maxswd_8_heldout", names)
        self.assertIn("swd_256", names)
        self.assertIn("swd_fixed_256", names)
        self.assertIn("swd_orthogonal_8", names)


class ReproducibilityTests(unittest.TestCase):
    def test_tiny_probe_is_reproducible(self):
        config = _tiny_config()
        first = run_probe(config, device=torch.device("cpu"), log=lambda message: None)
        second = run_probe(config, device=torch.device("cpu"), log=lambda message: None)
        for payload in (first, second):
            payload.pop("created_utc", None)
            payload.pop("runtime_seconds", None)
        self.assertEqual(
            json.dumps(json_ready(first), sort_keys=True),
            json.dumps(json_ready(second), sort_keys=True),
        )

    def test_tiny_probe_variants_have_alignment_for_hard_direction(self):
        payload = run_probe(_tiny_config(), device=torch.device("cpu"), log=lambda message: None)
        variants = payload["scenarios"]["hard_direction"]["by_sample_size"]["96"]["variants"]
        self.assertIn("maxswd_8_in_alignment", variants)
        self.assertIn("maxswd_8_heldout_alignment", variants)


if __name__ == "__main__":
    unittest.main()
