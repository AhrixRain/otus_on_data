from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from cms_data import load_config, resolve_config  # noqa: E402
from cms_training import build_loss_factory  # noqa: E402
from loss import CANONICAL_LOSS_KIND, build_ee_physics_features  # noqa: E402
from metrics import residual_metrics  # noqa: E402


def fake_ee_batch(n: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    momenta = torch.randn(n, 6, generator=generator) * 20.0
    px_m, py_m, pz_m = momenta[:, 0], momenta[:, 1], momenta[:, 2]
    px_p, py_p, pz_p = momenta[:, 3], momenta[:, 4], momenta[:, 5]
    e_m = torch.sqrt(px_m**2 + py_m**2 + pz_m**2 + 0.000511**2) + 5.0
    e_p = torch.sqrt(px_p**2 + py_p**2 + pz_p**2 + 0.000511**2) + 5.0
    return torch.stack([px_m, py_m, pz_m, e_m, px_p, py_p, pz_p, e_p], dim=1)


class CmsLossSmokeTest(unittest.TestCase):
    def test_build_ee_physics_features_is_finite(self) -> None:
        features = build_ee_physics_features(fake_ee_batch(32))
        self.assertIn("m_ee", features)
        self.assertIn("physics_coord_features", features)
        for value in features.values():
            self.assertTrue(torch.isfinite(value).all())

    def test_canonical_loss_forward_backward(self) -> None:
        truth_x = fake_ee_batch(48, seed=1).numpy().astype("float32")
        truth_z = fake_ee_batch(48, seed=2).numpy().astype("float32")
        loss_factory = build_loss_factory(
            truth_x,
            truth_z,
            {
                "kind": CANONICAL_LOSS_KIND,
                "num_slices": 8,
                "pair_pt_w1": 2.0,
                "physics_coord_swd": 0.5,
            },
        )
        pred = fake_ee_batch(48, seed=3).requires_grad_(True)
        loss = loss_factory.x_sim_loss(torch.as_tensor(truth_x), pred)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(pred.grad)
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertIn("x_pair_pt_w1", loss_factory.latest_components)
        self.assertIn("x_physics_coord_swd", loss_factory.latest_components)

    def test_new_config_loads_with_three_stages(self) -> None:
        config = resolve_config(
            load_config(REPO_ROOT / "configs/cms_doubleelectron_mps.yaml")
        )
        self.assertEqual(config["loss"]["kind"], CANONICAL_LOSS_KIND)
        self.assertEqual(
            [stage["name"] for stage in config["stages"]],
            [
                "stage1_anchor_warmup",
                "stage2_joint_transport",
                "stage3_decoder_response_mass_protected",
            ],
        )

    def test_residual_metrics_valid_bins_use_truth_counts(self) -> None:
        truth = np.concatenate([np.full(25, 80.0), np.full(25, 90.0)])
        pred = np.concatenate([np.full(25, 80.0), np.full(25, 90.0)])
        metrics, _arrays = residual_metrics(
            truth,
            pred,
            bins=4,
            mass_range=(70.0, 110.0),
            min_truth_count=20,
        )
        self.assertGreater(metrics["valid_bins"], 0)
        self.assertEqual(metrics["total_bins"], 4)


if __name__ == "__main__":
    unittest.main()
