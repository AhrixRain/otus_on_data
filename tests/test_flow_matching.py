"""Unit tests for the Run F bidirectional OT-flow-matching model."""

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

from flow_matching import (  # noqa: E402
    build_flow_matching_autoencoder,
    cylindrical_difference,
    interpolate_cylindrical,
)


MUON_MASS = 0.1056583755
MASSES = (MUON_MASS, MUON_MASS)


def _p4_back_to_back(pt=10.0, eta=0.0, phi=0.0, delta=0.0):
    p1 = np.array([pt, 0.0, pt * np.sinh(eta), 0.0], dtype=np.float32)
    p1[3] = np.sqrt(np.sum(p1[:3] ** 2) + MUON_MASS**2)
    phi2 = phi + np.pi + delta
    p2 = np.array([pt * np.cos(phi2), pt * np.sin(phi2), -pt * np.sinh(eta), 0.0], dtype=np.float32)
    p2[3] = np.sqrt(np.sum(p2[:3] ** 2) + MUON_MASS**2)
    return np.concatenate([p1, p2]).astype(np.float32)[None, :]


class TestCylindricalInterpolation(unittest.TestCase):
    def test_phi_wraps_on_s1(self):
        u0 = torch.tensor([[1.0, 0.0, 3.1, 1.0, 0.0, -3.1]])
        u1 = torch.tensor([[2.0, 1.0, -3.1, 2.0, 1.0, 3.1]])
        u = interpolate_cylindrical(u0, u1, torch.tensor([[0.5]]))
        self.assertTrue(torch.isfinite(u).all())
        self.assertTrue((u[:, 2].abs() <= np.pi).all())
        d = cylindrical_difference(u0, u1)
        self.assertTrue((d[:, 2].abs() <= np.pi).all())


class TestFlowMatchingModel(unittest.TestCase):
    def _model(self):
        cfg = {
            "hidden_dims": [32, 32],
            "activation": "SiLU",
            "integration_steps": 2,
            "noise_dim": 4,
            "sigma": 0.1,
            "ot_regularization": 0.05,
            "ot_max_iter": 50,
            "ot_max_batch": 128,
            "daughter_masses": list(MASSES),
        }
        x = np.concatenate([_p4_back_to_back(pt=p) for p in [8, 10, 12, 14]])
        z = np.concatenate([_p4_back_to_back(pt=p, delta=0.01) for p in [8, 10, 12, 14]])
        model = build_flow_matching_autoencoder(cfg, x, z, MUON_MASS, MASSES)
        return model, x, z

    def test_shapes_and_mass_shell(self):
        model, x, _z = self._model()
        torch.manual_seed(0)
        z = model.encode(torch.tensor(x))
        reco = model.decode(z)
        self.assertEqual(tuple(z.shape), (4, 8))
        self.assertEqual(tuple(reco.shape), (4, 8))
        for values in (z, reco):
            for start in (0, 4):
                p = values[:, start : start + 3]
                e = torch.sqrt((p * p).sum(dim=1) + MUON_MASS**2)
                self.assertTrue(torch.allclose(values[:, start + 3], e, atol=1e-5))

    def test_flow_loss_finite_and_differentiable(self):
        model, x, z = self._model()
        x = torch.tensor(x); z = torch.tensor(z)
        loss = model.flow_matching_loss(x, z)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss.detach()), 0.0)
        params = list(model.parameters())
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        self.assertTrue(any(g is not None for g in grads))

    def test_different_noise_gives_different_samples(self):
        model, _x, z = self._model()
        z = torch.tensor(z)
        torch.manual_seed(1)
        a = model.decode(z)
        torch.manual_seed(2)
        b = model.decode(z)
        self.assertFalse(torch.allclose(a, b))


if __name__ == "__main__":
    unittest.main()
