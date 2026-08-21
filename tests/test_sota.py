"""Unit tests for scripts_sota Tier A components.

These tests use small synthetic tensors only; no ROOT/HDF5/cache access.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (REPO_ROOT / "scripts", REPO_ROOT / "scripts_sota"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from ot import (  # noqa: E402
    PhysicsGroundCost,
    SinkhornConfig,
    cylindrical_physics_features,
    sinkhorn_divergence,
)
from max_swd import MaxSlicedWasserstein  # noqa: E402
from cylindrical_flow import (  # noqa: E402
    CylindricalFlowAutoencoder,
    cylindrical_to_p4,
    p4_to_cylindrical,
)
from evaluation import c2st_score, cylindrical_features_np  # noqa: E402
from selection import GateConfig, gated_score  # noqa: E402


MUON_MASS = 0.1056583755
MASSES = (MUON_MASS, MUON_MASS)


def _p4_back_to_back(pt=10.0, eta=0.0, phi=0.0, delta=0.0):
    """Two back-to-back muons with a controlled opening angle."""
    p1 = np.array([pt, 0.0, pt * np.sinh(eta), 0.0], dtype=np.float32)
    p1[3] = np.sqrt(np.sum(p1[:3] ** 2) + MUON_MASS**2)
    phi2 = phi + np.pi + delta
    p2 = np.array([pt * np.cos(phi2), pt * np.sin(phi2), -pt * np.sinh(eta), 0.0], dtype=np.float32)
    p2[3] = np.sqrt(np.sum(p2[:3] ** 2) + MUON_MASS**2)
    return np.concatenate([p1, p2]).astype(np.float32)[None, :]


class TestPhysicsGroundCost(unittest.TestCase):
    def test_z_features_use_stored_energy_when_requested(self):
        values = _p4_back_to_back(pt=1.5)
        shifted_energy = values.copy()
        shifted_energy[:, 3] += 0.1
        shifted_energy[:, 7] += 0.1
        stable_a = cylindrical_physics_features(torch.tensor(values), MASSES)
        stable_b = cylindrical_physics_features(torch.tensor(shifted_energy), MASSES)
        stored_b = cylindrical_physics_features(
            torch.tensor(shifted_energy), MASSES, mass_from_energy=True
        )
        self.assertTrue(torch.allclose(stable_a[:, 8], stable_b[:, 8]))
        self.assertFalse(torch.allclose(stable_a[:, 8], stored_b[:, 8]))

    def test_feature_shape_and_standardization(self):
        x = np.concatenate([_p4_back_to_back(pt=p) for p in [8, 10, 12, 14]])
        cost = PhysicsGroundCost(x, daughter_masses=MASSES)
        features = cost.features(torch.as_tensor(x, dtype=torch.float32))
        self.assertEqual(tuple(features.shape), (4, 14))
        self.assertTrue(torch.isfinite(features).all())
        self.assertTrue(torch.allclose(features.mean(dim=0), torch.zeros(14), atol=1e-5))

    def test_mass_shift_is_expensive(self):
        train = np.concatenate([_p4_back_to_back(pt=10.0)] * 16)
        cost = PhysicsGroundCost(train, daughter_masses=MASSES)
        a = _p4_back_to_back(pt=10.0)
        b = _p4_back_to_back(pt=10.0, delta=0.010)
        caa = cost.pairwise_cost(torch.tensor(a), torch.tensor(a))
        cab = cost.pairwise_cost(torch.tensor(a), torch.tensor(b))
        self.assertGreater(float(cab[0, 0]), 0.0)
        self.assertAlmostEqual(float(caa[0, 0]), 0.0, places=5)


class TestSinkhorn(unittest.TestCase):
    def test_sinkhorn_same_sample_small_and_finite(self):
        x = np.concatenate([_p4_back_to_back(pt=p) for p in [8, 9, 10, 11, 12]])
        ground = PhysicsGroundCost(x, daughter_masses=MASSES)
        value = sinkhorn_divergence(
            ground,
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(x, dtype=torch.float32),
            SinkhornConfig(regularization=0.05, max_iter=100, max_batch=64),
        )
        self.assertTrue(torch.isfinite(value))
        self.assertGreaterEqual(float(value), -1e-4)
        self.assertLess(float(value), 0.2)

    def test_sinkhorn_grows_with_mass_shift(self):
        x = np.concatenate([_p4_back_to_back(pt=p) for p in [8, 9, 10, 11, 12]])
        ground = PhysicsGroundCost(x, daughter_masses=MASSES)
        shifted = np.concatenate([_p4_back_to_back(pt=p, delta=0.010) for p in [8, 9, 10, 11, 12]])
        base = sinkhorn_divergence(
            ground,
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(x, dtype=torch.float32),
            SinkhornConfig(regularization=0.05, max_iter=100, max_batch=64),
        )
        apart = sinkhorn_divergence(
            ground,
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(shifted, dtype=torch.float32),
            SinkhornConfig(regularization=0.05, max_iter=100, max_batch=64),
        )
        self.assertGreater(float(apart), float(base))


class TestMaxSWD(unittest.TestCase):
    def test_adversarial_step_increases_separation(self):
        torch.manual_seed(0)
        module = MaxSlicedWasserstein(feature_dim=8, num_directions=8, lr=0.2, diversity_weight=0.01)
        a = torch.zeros(32, 8)
        b = torch.ones(32, 8)
        before = float(module(a, b).detach())
        for _ in range(20):
            module.adversarial_step(a, b)
        after = float(module(a, b).detach())
        self.assertGreater(after, before)


class TestCylindricalFlow(unittest.TestCase):
    def _model(self):
        cfg = {
            "hidden_dims": [32, 32],
            "activation": "SiLU",
            "flow_steps": 2,
            "mean_residual_limits": [0.25, 0.3, 0.3, 0.25, 0.3, 0.3],
            "core_sigma_floors": [0.001, 0.0005, 0.0005, 0.001, 0.0005, 0.0005],
            "core_sigma_scales": [0.05, 0.025, 0.025, 0.05, 0.025, 0.025],
            "tail_sigma_scales": [0.15, 0.08, 0.08, 0.15, 0.08, 0.08],
            "core_log_sigma_bias": -3.0,
            "tail_log_sigma_bias": -5.0,
            "student_t_degrees_of_freedom": 4.0,
            "maximum_heavy_noise": 12.0,
        }
        x = np.concatenate([_p4_back_to_back(pt=p) for p in [8, 10, 12, 14]])
        model = CylindricalFlowAutoencoder(
            (np.zeros(14, dtype=np.float32), np.ones(14, dtype=np.float32)),
            (np.zeros(14, dtype=np.float32), np.ones(14, dtype=np.float32)),
            cfg,
            MUON_MASS,
            MASSES,
        )
        return model, x

    def test_round_trip_shape_and_mass_shell(self):
        model, x = self._model()
        model.set_noise_multipliers(0.0, 0.0)
        _z = model.encode(torch.tensor(x, dtype=torch.float32))
        reco = model.decode(_z)
        self.assertIsNotNone(_z)
        self.assertEqual(tuple(_z.shape), (4, 8))
        self.assertEqual(tuple(reco.shape), (4, 8))
        for values in (_z, reco):
            for start in (0, 4):
                p = values[:, start : start + 3]
                e = torch.sqrt((p * p).sum(dim=1) + MUON_MASS**2)
                self.assertTrue(torch.allclose(values[:, start + 3], e, atol=1e-5))

    def test_cylindrical_round_trip(self):
        x = _p4_back_to_back(pt=11.0, eta=1.2, phi=0.7)
        coords = p4_to_cylindrical(torch.tensor(x))
        back = cylindrical_to_p4(coords, MUON_MASS)
        self.assertTrue(torch.allclose(torch.tensor(x), back, atol=1e-4))

    def test_noise_multipliers_change_samples(self):
        model, x = self._model()
        torch.manual_seed(123)
        model.set_noise_multipliers(0.0, 0.0)
        d1 = model.decode(torch.tensor(x, dtype=torch.float32))
        d2 = model.decode(torch.tensor(x, dtype=torch.float32))
        self.assertTrue(torch.allclose(d1, d2))
        model.set_noise_multipliers(1.0, 0.0)
        d3 = model.decode(torch.tensor(x, dtype=torch.float32))
        d4 = model.decode(torch.tensor(x, dtype=torch.float32))
        self.assertFalse(torch.allclose(d3, d4))


class TestEvaluation(unittest.TestCase):
    def test_c2st_same_distribution_near_chance(self):
        rng = np.random.default_rng(2)
        def sample(n):
            p = rng.uniform(8, 14, n)
            e = rng.uniform(-1, 1, n)
            return np.stack([
                np.concatenate([
                    [pp, 0, pp * np.sinh(ee), np.sqrt(pp**2 + (pp * np.sinh(ee)) ** 2 + MUON_MASS**2)],
                    [-pp, 0, -pp * np.sinh(ee), np.sqrt(pp**2 + (pp * np.sinh(ee)) ** 2 + MUON_MASS**2)],
                ])
                for pp, ee in zip(p, e)
            ]).astype(np.float32)
        x = sample(400)
        report = c2st_score(x, sample(400), daughter_masses=MASSES, max_samples=400, folds=2)
        self.assertLess(abs(report["c2st_auc_mean"] - 0.5), 0.15)

    def test_cylindrical_features_shape(self):
        x = np.concatenate([_p4_back_to_back(pt=p) for p in [8, 10, 12]])
        features = cylindrical_features_np(x, MASSES)
        self.assertEqual(features.shape, (3, 14))
        self.assertTrue(np.isfinite(features).all())


class TestSelectionGates(unittest.TestCase):
    def test_gated_score(self):
        self.assertEqual(gated_score(1.0, {"a": 1}, True), 1.0)
        self.assertTrue(np.isinf(gated_score(1.0, {"a": 1}, False)))

    def test_gate_config_defaults(self):
        cfg = GateConfig.from_config({})
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.max_sim_mass_w1_gev, 0.015)


if __name__ == "__main__":
    unittest.main()
