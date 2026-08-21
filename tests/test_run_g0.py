"""Synthetic-only tests for the Run G0 bridge and locked evaluator."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (REPO_ROOT / "scripts", REPO_ROOT / "scripts_sota"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from g0_contract import array_fingerprint, build_split_manifest, distribution_report  # noqa: E402
from reciprocal_bridge import build_reciprocal_bridge_autoencoder  # noqa: E402
from selection import checkpoint_selection_score  # noqa: E402


MUON_MASS = 0.1056583755
MASSES = (MUON_MASS, MUON_MASS)


def _p4(pt: float, eta: float = 0.0, delta: float = 0.0) -> np.ndarray:
    pz = pt * np.sinh(eta)
    e = np.sqrt(pt**2 + pz**2 + MUON_MASS**2)
    phi2 = np.pi + delta
    p1 = [pt, 0.0, pz, e]
    p2 = [pt * np.cos(phi2), pt * np.sin(phi2), -pz, e]
    return np.asarray([*p1, *p2], dtype=np.float32)


class TestReciprocalBridge(unittest.TestCase):
    def _model(self):
        x = np.stack([_p4(1.50 + 0.02 * i, eta=0.05 * i) for i in range(8)])
        z = np.stack([_p4(1.48 + 0.02 * i, eta=0.04 * i, delta=0.01) for i in range(8)])
        config = {
            "hidden_dims": [16, 16],
            "activation": "SiLU",
            "integration_steps": 2,
            "noise_dim": 4,
            "sigma": 0.1,
            "score_weight": 0.1,
            "reciprocity_weight": 0.25,
            "ot_regularization": 0.1,
            "ot_max_iter": 20,
            "ot_max_batch": 8,
        }
        return build_reciprocal_bridge_autoencoder(
            config, x, z, MUON_MASS, MASSES
        ), x, z

    def test_shared_field_shapes_and_mass_shell(self):
        model, x, z = self._model()
        self.assertIs(model.encoder.field, model.decoder.field)
        encoded = model.encode(torch.tensor(x))
        decoded = model.decode(torch.tensor(z))
        self.assertEqual(tuple(encoded.shape), (8, 8))
        self.assertEqual(tuple(decoded.shape), (8, 8))
        for values in (encoded, decoded):
            for start in (0, 4):
                momentum = values[:, start : start + 3]
                expected = torch.sqrt((momentum**2).sum(dim=1) + MUON_MASS**2)
                self.assertTrue(torch.allclose(values[:, start + 3], expected, atol=1e-5))

    def test_bridge_loss_is_finite_and_differentiable(self):
        model, x, z = self._model()
        loss = model.flow_matching_loss(torch.tensor(x), torch.tensor(z))
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss.detach()), 0.0)
        gradients = torch.autograd.grad(loss, list(model.parameters()), allow_unused=True)
        self.assertTrue(any(value is not None for value in gradients))
        self.assertEqual(
            set(model.latest_bridge_components),
            {"bridge_flow", "bridge_score", "bridge_reciprocity"},
        )


class TestG0Contract(unittest.TestCase):
    def test_array_fingerprint_is_content_sensitive(self):
        values = np.arange(24, dtype=np.float32).reshape(3, 8)
        first = array_fingerprint(values)
        self.assertEqual(first, array_fingerprint(values.copy()))
        changed = values.copy()
        changed[0, 0] += 1
        self.assertNotEqual(first["sha256"], array_fingerprint(changed)["sha256"])

    def test_manifest_is_stable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cms = root / "cms.root"
            prior = root / "prior.h5"
            cms.write_bytes(b"cms")
            prior.write_bytes(b"prior")
            config = {
                "paths": {
                    "cms_root_file": str(cms),
                    "theory_prior_file": str(prior),
                },
                "data": {"channel": "muon"},
                "data_split": {"train_ratio": 0.8, "val_ratio": 0.1},
                "muon_selection": {
                    "muon_pt_min": 3.0,
                    "muon_abs_eta_max": 2.4,
                    "jpsi_mass_min": 3.0,
                    "jpsi_mass_max": 3.2,
                },
                "seed": 0,
                "float_type": "float32",
            }
            values = np.stack([_p4(1.5 + 0.01 * i) for i in range(6)])
            arrays = {key: values.copy() for key in (
                "x_train", "x_val", "x_test", "z_train", "z_val", "z_test"
            )}
            a = build_split_manifest(config, arrays, {"key": "test"}, num_samples=None)
            b = build_split_manifest(config, arrays, {"key": "test"}, num_samples=None)
            self.assertEqual(a["contract_sha256"], b["contract_sha256"])

    def test_distribution_report_equal_samples(self):
        values = np.stack([_p4(1.45 + 0.01 * i) for i in range(120)])
        report = distribution_report(
            values,
            values.copy(),
            daughter_masses=MASSES,
            stable_mass=True,
            max_events=None,
            bootstrap_replicates=4,
            seed=1,
        )
        self.assertAlmostEqual(report["mass"]["w1_gev"], 0.0, places=7)
        self.assertAlmostEqual(report["mass"]["ks"], 0.0, places=7)
        self.assertAlmostEqual(report["feature_ks"]["max"], 0.0, places=7)

    def test_worst_direction_checkpoint_selection(self):
        config = {
            "checkpoint_selection": {"mode": "worst_direction"},
            "sota_selection_targets": {"sim_mass_ks": 0.02, "latent_mass_ks": 0.04},
        }
        score = checkpoint_selection_score(
            100.0,
            {"sim_mass_ks": 0.01, "latent_mass_ks": 0.06},
            True,
            config,
        )
        self.assertAlmostEqual(score, 1.5)


if __name__ == "__main__":
    unittest.main()
