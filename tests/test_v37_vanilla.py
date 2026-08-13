from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from cms_data import load_config, split_unpaired  # noqa: E402
from cms_model import build_model  # noqa: E402
from cms_training import build_loss_factory, first_tensor  # noqa: E402
from loss import sliced_wasserstein  # noqa: E402


def make_arrays(n: int = 256, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = (rng.normal(size=(n, 8)) * 10.0 + 30.0).astype(np.float32)
    z = (rng.normal(size=(n, 8)) * 3.0 + 10.0).astype(np.float32)
    return x, z


def vanilla_loss_config() -> dict:
    return {
        "kind": "cms_jpsi_doublemuon_loss",
        "vanilla_v3_7": True,
        "raw_swd": 0.0,
        "marginal_w1": 0.0,
        "mass_w1": 0.0,
        "resonance_mass_w1": 0.0,
        "physics_swd": 0.0,
        "mass_kin_swd": 0.0,
        "transverse_w1": 0.0,
        "longitudinal_w1": 0.0,
        "tail_w1": 0.0,
        "pair_mass_w1": 0.0,
        "pair_pt_w1": 0.0,
        "lepton_pt_w1": 0.0,
        "delta_phi_w1": 0.0,
        "delta_eta_w1": 0.0,
        "pair_rapidity_w1": 0.0,
        "physics_coord_swd": 0.0,
        "mmd": 0.0,
        "x_reco_physics_w1": 0.0,
        "num_slices": 16,
    }


def tiny_model_config() -> dict:
    return {
        "model": {
            "class": "CondNoiseAutoencoder",
            "num_hidden_layers": 1,
            "dim_per_hidden_layer": 16,
            "activation": "ReLU",
            "sigma_fun": "softplus",
            "sigma_floor": 0.001,
            "stoch_enc": False,
            "stoch_dec": False,
            "raw_io": True,
        }
    }


class TestVanillaV37Objective(unittest.TestCase):
    def test_z_prior_loss_is_only_sliced_wasserstein(self):
        x, z = make_arrays(128)
        factory = build_loss_factory(x, z, vanilla_loss_config())
        factory.set_num_slices(16)
        z_true = torch.as_tensor(z)
        z_encoded = torch.randn(128, 8, requires_grad=True)
        torch.manual_seed(7)
        loss = factory.z_prior_loss(z_true, z_encoded)
        torch.manual_seed(7)
        expected = sliced_wasserstein(
            factory.standardize_z_raw(z_true),
            factory.standardize_z_raw(z_encoded),
            16,
            2,
        )
        self.assertAlmostEqual(float(loss), float(expected), places=6)
        self.assertTrue(
            set(factory.latest_components).issubset(
                {"z_raw_swd", "z_raw_swd_raw", "z_raw_swd_weighted"}
            )
        )

    def test_x_reco_loss_is_raw_mse(self):
        x, z = make_arrays(128)
        factory = build_loss_factory(x, z, vanilla_loss_config())
        x_true = torch.as_tensor(x)
        x_hat = torch.as_tensor(x + np.random.default_rng(3).normal(size=x.shape))
        loss = factory.x_reco_loss(x_true, x_hat)
        self.assertAlmostEqual(
            float(loss),
            float(torch.mean((x_true - x_hat) ** 2)),
            places=6,
        )
        self.assertTrue("x_reco_mse_raw" in factory.latest_components)

    def test_gradient_paths(self):
        x, z = make_arrays(64)
        model_config = tiny_model_config()
        x_mean = x.mean(axis=0)
        x_std = np.where(x.std(axis=0) == 0.0, 1.0, x.std(axis=0))
        z_mean = z.mean(axis=0)
        z_std = np.where(z.std(axis=0) == 0.0, 1.0, z.std(axis=0))
        model = build_model(
            model_config,
            x_mean,
            x_std,
            z_mean,
            z_std,
        )
        factory = build_loss_factory(x, z, vanilla_loss_config())
        factory.set_num_slices(8)
        x_batch = torch.as_tensor(x[:32])
        z_batch = torch.as_tensor(z[:32])
        z_hat = first_tensor(model.encode(x_batch))
        x_hat = first_tensor(model.decode(z_hat))
        reco = factory.x_reco_loss(x_batch, x_hat)
        latent = factory.z_prior_loss(z_batch, z_hat)
        total = reco + 1.0 * latent

        encoder_params = list(model.encoder.parameters())
        decoder_params = list(model.decoder.parameters())
        all_params = [*encoder_params, *decoder_params]
        n_enc = len(encoder_params)

        def norms(loss):
            grads = torch.autograd.grad(
                loss,
                all_params,
                retain_graph=True,
                allow_unused=True,
            )
            enc = [grad for grad in grads[:n_enc] if grad is not None]
            dec = [grad for grad in grads[n_enc:] if grad is not None]
            return (
                float(torch.cat([g.reshape(-1) for g in enc]).norm()) if enc else 0.0,
                float(torch.cat([g.reshape(-1) for g in dec]).norm()) if dec else 0.0,
            )

        reco_enc, reco_dec = norms(reco)
        lat_enc, lat_dec = norms(latent)
        total_enc, total_dec = norms(total)
        self.assertGreater(reco_enc, 0.0)
        self.assertGreater(reco_dec, 0.0)
        self.assertGreater(lat_enc, 0.0)
        self.assertEqual(lat_dec, 0.0)
        self.assertGreater(total_enc, 0.0)
        self.assertGreater(total_dec, 0.0)


class TestV37ConfigHelpers(unittest.TestCase):
    def test_split_caps_clamp_not_inflate(self):
        arr = np.arange(1000, dtype=np.float32).reshape(-1, 1)
        train, val, test = split_unpaired(
            arr,
            0.8,
            0.1,
            0,
            train_max=100,
            val_max=50,
            test_max=25,
        )
        self.assertEqual(train.shape, (100, 1))
        self.assertEqual(val.shape, (50, 1))
        self.assertEqual(test.shape, (25, 1))

        small = np.arange(200, dtype=np.float32).reshape(-1, 1)
        train2, val2, test2 = split_unpaired(
            small,
            0.8,
            0.1,
            0,
            train_max=500,
            val_max=100,
            test_max=100,
        )
        self.assertEqual(train2.shape, (160, 1))
        self.assertEqual(val2.shape, (20, 1))
        self.assertEqual(test2.shape, (20, 1))

    def test_load_config_extends_merges_named_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "base.yaml").write_text(
                "run_name: base\n"
                "epochs_total: 300\n"
                "stages:\n"
                "  - name: s1\n"
                "    lamb: 1.0\n"
                "    epochs: 300\n"
                "    lr_decay: false\n"
                "loss:\n"
                "  kind: cms_jpsi_doublemuon_loss\n",
                encoding="utf-8",
            )
            (tmp_path / "child.yaml").write_text(
                "extends: base.yaml\n"
                "run_name: v3.7_lambda0p3\n"
                "stages:\n"
                "  - name: s1\n"
                "    lamb: 0.3\n",
                encoding="utf-8",
            )
            config = load_config(tmp_path / "child.yaml")
            self.assertEqual(config["run_name"], "v3.7_lambda0p3")
            self.assertEqual(config["epochs_total"], 300)
            self.assertEqual(config["loss"]["kind"], "cms_jpsi_doublemuon_loss")
            self.assertEqual(config["stages"][0]["lamb"], 0.3)
            self.assertEqual(config["stages"][0]["epochs"], 300)
            self.assertFalse(config["stages"][0]["lr_decay"])


if __name__ == "__main__":
    unittest.main()
