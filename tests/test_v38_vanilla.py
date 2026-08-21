"""v3.8 vanilla SWAE (raw-coordinate latent SWD) correctness tests.

All tests use synthetic arrays; no ROOT/HDF5 source data is required.
"""

from __future__ import annotations

import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from cms_data import (  # noqa: E402
    inspect_cms_root,
    inspect_theory_prior,
    load_config,
    resolve_config,
    sha256_fingerprint,
)

from cms_training import (  # noqa: E402
    DeterministicSequentialLoader,
    _aligned_batch_sizes,
    build_loaders,
    build_loss_factory,
    first_tensor,
    make_stage_loss_config,
    train_standard_epoch,
)
from cms_model import build_model, checkpoint_payload, load_model_from_checkpoint  # noqa: E402
from loss import DEFAULT_SPACE_WEIGHTS, sliced_wasserstein  # noqa: E402


V38_CONFIG_PATH = REPO_ROOT / "configs/archive/cms_JpsiDoubleMuons_v3.8_vanilla_paper.yaml"
V37_CONFIG_PATHS = [
    REPO_ROOT / "configs/archive/cms_JpsiDoubleMuons_v3.7_vanilla.yaml",
    REPO_ROOT / "configs/archive/cms_JpsiDoubleMuons_v3.7_lambda0p1.yaml",
    REPO_ROOT / "configs/archive/cms_JpsiDoubleMuons_v3.7_lambda0p3.yaml",
    REPO_ROOT / "configs/archive/cms_JpsiDoubleMuons_v3.7_lambda1.yaml",
    REPO_ROOT / "configs/archive/cms_JpsiDoubleMuons_v3.7_lambda3.yaml",
    REPO_ROOT / "configs/archive/cms_JpsiDoubleMuons_v3.7_lambda10.yaml",
]


def make_arrays(n: int = 256, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = (rng.normal(size=(n, 8)) * 10.0 + 30.0).astype(np.float32)
    z = (rng.normal(size=(n, 8)) * 3.0 + 10.0).astype(np.float32)
    return x, z


def v38_loss_config(num_slices: int = 16) -> dict:
    config = resolve_config(load_config(V38_CONFIG_PATH))
    loss = dict(config["loss"])
    loss["num_slices"] = int(num_slices)
    return loss


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


def loader_config(n: int) -> dict:
    return {
        "paths": {"output_root": str(REPO_ROOT / "outputs" / "unused")},
        "seed": 0,
        "loaders": {
            "train_batch_size": n,
            "eval_batch_size": n,
            "preload_data_to_accelerator": False,
            "train_sampler": "shuffled_without_replacement",
            "eval_sampler": "deterministic_sequential",
        },
    }


def split_arrays(
    x_train: int,
    z_train: int,
    x_val: int = 20,
    z_val: int = 20,
) -> dict[str, np.ndarray]:
    def rows(n: int) -> np.ndarray:
        return np.repeat(np.arange(n, dtype=np.float32)[:, None], 8, axis=1)

    return {
        "x_train": rows(x_train),
        "z_train": rows(z_train),
        "x_val": rows(x_val),
        "z_val": rows(z_val),
    }


class TestV38Config(unittest.TestCase):
    def test_config_resolves_v38_semantics(self) -> None:
        config = resolve_config(load_config(V38_CONFIG_PATH))
        self.assertEqual(config["run_name"], "Jpsi_v3.8_vanilla_paper_all_data_seed0")
        self.assertIs(config["loss"]["vanilla_swae"], True)
        self.assertIs(config["loss"]["vanilla_v3_7"], False)
        self.assertIs(config["loss"]["standardize_raw_matching"], False)
        self.assertEqual(config["loss"]["p"], 2)
        self.assertEqual(config["loss"]["num_slices"], 1000)
        self.assertIs(config["model"]["raw_io"], True)
        self.assertEqual(config["model"]["daughter_masses"], [0.105658, 0.105658])
        self.assertEqual(config["loaders"]["train_sampler"], "shuffled_without_replacement")
        self.assertEqual(config["loaders"]["eval_sampler"], "deterministic_sequential")
        self.assertEqual(config["loaders"]["train_batch_size"], 20000)
        self.assertEqual(config["loaders"]["eval_batch_size"], 20000)
        stage = config["stages"][0]
        self.assertEqual(stage["epochs"], 300)
        self.assertEqual(stage["lr"], 0.001)
        self.assertEqual(stage["beta"], 1.0)
        self.assertEqual(stage["lamb"], 1.0)
        self.assertEqual(stage["tau"], 0.0)
        self.assertEqual(stage["rho"], 0.0)
        self.assertEqual(stage["nu_e"], 0.0)
        self.assertEqual(stage["nu_d"], 0.0)
        self.assertEqual(stage["num_slices"], 1000)
        self.assertFalse(stage["lr_decay"])
        self.assertFalse(stage["early_stopping"]["enabled"])
        self.assertIs(config["evaluation"]["independent_diagnostic_rng"], True)

    def test_only_reconstruction_and_latent_weights_nonzero(self) -> None:
        config = resolve_config(load_config(V38_CONFIG_PATH))
        loss = config["loss"]
        for key in DEFAULT_SPACE_WEIGHTS:
            self.assertEqual(loss[key], 0.0, key)
        for space in ("x", "z"):
            for key in DEFAULT_SPACE_WEIGHTS:
                self.assertEqual(loss["space_weights"][space][key], 0.0, f"{space}.{key}")
        self.assertEqual(loss["x_reco_physics_w1"], 0.0)
        self.assertEqual(loss["selection_score"]["x_sim"], 0.0)
        self.assertEqual(loss["selection_score"]["cycle"], 0.0)
        self.assertGreater(loss["selection_score"]["z_prior"], 0.0)
        self.assertGreater(loss["selection_score"]["x_reco"], 0.0)
        stage = config["stages"][0]
        for key in ("tau", "rho", "nu_e", "nu_d"):
            self.assertEqual(stage[key], 0.0, key)
        self.assertGreater(stage["beta"], 0.0)
        self.assertGreater(stage["lamb"], 0.0)

    def test_v38_has_no_data_caps_and_no_v37_inheritance(self) -> None:
        config = resolve_config(load_config(V38_CONFIG_PATH))
        split = config["data_split"]
        self.assertNotIn("train_max", split)
        self.assertNotIn("val_max", split)
        self.assertNotIn("test_max", split)
        raw_text = V38_CONFIG_PATH.read_text(encoding="utf-8")
        self.assertNotIn("extends:", raw_text)

    def test_resolved_config_records_raw_swd_and_loader_policy(self) -> None:
        config = resolve_config(load_config(V38_CONFIG_PATH))
        self.assertIs(config["loss"]["standardize_raw_matching"], False)
        self.assertIs(config["loss"]["vanilla_swae"], True)
        self.assertEqual(config["loaders"]["train_sampler"], "shuffled_without_replacement")
        self.assertEqual(config["loaders"]["eval_sampler"], "deterministic_sequential")


class TestV38Objective(unittest.TestCase):
    def test_z_prior_loss_is_raw_sliced_wasserstein(self) -> None:
        x, z = make_arrays(128)
        factory = build_loss_factory(x, z, v38_loss_config())
        factory.set_num_slices(16)
        z_true = torch.as_tensor(z)
        z_encoded = torch.randn(128, 8)
        torch.manual_seed(7)
        actual = factory.z_prior_loss(z_true, z_encoded)
        torch.manual_seed(7)
        expected = sliced_wasserstein(z_true, z_encoded, 16, 2)
        self.assertAlmostEqual(float(actual), float(expected), places=6)
        self.assertTrue(
            set(factory.latest_components).issubset(
                {"z_raw_swd", "z_raw_swd_raw", "z_raw_swd_weighted"}
            )
        )

    def test_z_prior_loss_differs_from_standardized_version(self) -> None:
        x, z = make_arrays(128)
        factory = build_loss_factory(x, z, v38_loss_config())
        factory.set_num_slices(16)
        z_true = torch.as_tensor(z)
        z_encoded = torch.randn(128, 8)
        torch.manual_seed(11)
        raw = factory.z_prior_loss(z_true, z_encoded)
        torch.manual_seed(11)
        standardized = sliced_wasserstein(
            factory.standardize_z_raw(z_true),
            factory.standardize_z_raw(z_encoded),
            16,
            2,
        )
        self.assertFalse(torch.allclose(raw, standardized, atol=1e-5, rtol=1e-5))

    def test_x_reco_loss_is_raw_mse(self) -> None:
        x, z = make_arrays(128)
        factory = build_loss_factory(x, z, v38_loss_config())
        x_true = torch.as_tensor(x)
        x_hat = torch.as_tensor(x + 1.5)
        loss = factory.x_reco_loss(x_true, x_hat)
        self.assertAlmostEqual(
            float(loss),
            float(torch.mean((x_true - x_hat) ** 2)),
            places=6,
        )
        self.assertIn("x_reco_mse_raw", factory.latest_components)

    def test_total_loss_is_beta_reco_plus_lambda_swd(self) -> None:
        x, z = make_arrays(64)
        factory = build_loss_factory(x, z, v38_loss_config())
        factory.set_num_slices(8)
        x_true = torch.as_tensor(x[:32])
        z_true = torch.as_tensor(z[:32])
        z_hat = torch.randn(32, 8)
        x_hat = x_true + 0.25
        torch.manual_seed(3)
        reco = factory.x_reco_loss(x_true, x_hat)
        torch.manual_seed(3)
        latent = factory.z_prior_loss(z_true, z_hat)
        beta, lamb = 1.0, 1.0
        total = beta * reco + lamb * latent
        self.assertAlmostEqual(
            float(total),
            float(beta * torch.mean((x_true - x_hat) ** 2) + lamb * latent),
            places=6,
        )

    def test_latent_cardinality_mismatch_fails_fast(self) -> None:
        x, z = make_arrays(64)
        factory = build_loss_factory(x, z, v38_loss_config())
        with self.assertRaises(ValueError):
            factory.z_prior_loss(torch.as_tensor(z), torch.randn(63, 8))

    def test_gradient_paths(self) -> None:
        x, z = make_arrays(64)
        model_config = tiny_model_config()
        x_mean = x.mean(axis=0)
        x_std = np.where(x.std(axis=0) == 0.0, 1.0, x.std(axis=0))
        z_mean = z.mean(axis=0)
        z_std = np.where(z.std(axis=0) == 0.0, 1.0, z.std(axis=0))
        model = build_model(model_config, x_mean, x_std, z_mean, z_std)
        factory = build_loss_factory(x, z, v38_loss_config())
        factory.set_num_slices(8)
        x_batch = torch.as_tensor(x[:32])
        z_batch = torch.as_tensor(z[:32])
        z_hat = first_tensor(model.encode(x_batch))
        x_hat = first_tensor(model.decode(z_hat))
        reco = factory.x_reco_loss(x_batch, x_hat)
        latent = factory.z_prior_loss(z_batch, z_hat)

        encoder_params = list(model.encoder.parameters())
        decoder_params = list(model.decoder.parameters())
        all_params = [*encoder_params, *decoder_params]
        n_enc = len(encoder_params)

        def norms(loss):
            grads = torch.autograd.grad(loss, all_params, retain_graph=True, allow_unused=True)
            enc = [g for g in grads[:n_enc] if g is not None]
            dec = [g for g in grads[n_enc:] if g is not None]
            enc_norm = float(torch.cat([g.reshape(-1) for g in enc]).norm()) if enc else 0.0
            dec_norm = float(torch.cat([g.reshape(-1) for g in dec]).norm()) if dec else 0.0
            return enc_norm, dec_norm

        reco_enc, reco_dec = norms(reco)
        lat_enc, lat_dec = norms(latent)
        self.assertGreater(reco_enc, 0.0)
        self.assertGreater(reco_dec, 0.0)
        self.assertGreater(lat_enc, 0.0)
        self.assertEqual(lat_dec, 0.0)

    def test_standard_epoch_logs_three_raw_losses_and_fixed_reference(self) -> None:
        x, z = make_arrays(64)
        model_config = tiny_model_config()
        model = build_model(
            model_config,
            x.mean(axis=0),
            np.where(x.std(axis=0) == 0, 1.0, x.std(axis=0)),
            z.mean(axis=0),
            np.where(z.std(axis=0) == 0, 1.0, z.std(axis=0)),
        )
        factory = build_loss_factory(x, z, v38_loss_config())
        factory.set_num_slices(4)
        x_train = torch.as_tensor(x[:16])
        z_train = torch.as_tensor(z[:16])
        x_loader = DeterministicSequentialLoader(x_train, [16])
        z_loader = DeterministicSequentialLoader(z_train, [16])
        stage = {
            "beta": 1.0,
            "lamb": 1.0,
            "tau": 0.0,
            "rho": 0.0,
            "nu_e": 0.0,
            "nu_d": 0.0,
            "num_slices": 4,
        }
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        sums = train_standard_epoch(
            model,
            optimizer,
            x_loader,
            z_loader,
            stage,
            factory,
            torch.device("cpu"),
        )
        # The direct z -> x path is evaluated for the fixed reference even
        # though tau=0 means it does not contribute a training gradient.
        self.assertGreater(sums["alt_x_loss"], 0.0)
        self.assertEqual(sums["x_constraint_loss"], 0.0)
        self.assertEqual(sums["grad_norm_encoder_anchor"], 0.0)
        self.assertAlmostEqual(
            sums["reference_loss"],
            sums["x_loss"] + sums["z_loss"] + sums["alt_x_loss"],
            places=4,
        )
        # Vanilla mode records only the raw SWD terms plus the raw cycle MSE.
        self.assertTrue(
            set(factory.latest_components).issubset(
                {
                    "z_raw_swd", "z_raw_swd_raw", "z_raw_swd_weighted",
                    "x_raw_swd", "x_raw_swd_raw", "x_raw_swd_weighted",
                    "x_reco_mse_raw",
                }
            )
        )
        self.assertTrue(np.isfinite(sums["loss"]))
        self.assertTrue(np.isfinite(sums["grad_norm_encoder_total"]))
        self.assertTrue(np.isfinite(sums["grad_norm_decoder_total"]))

    def test_vanilla_x_sim_loss_is_raw_swd_only(self) -> None:
        x, z = make_arrays(64)
        factory = build_loss_factory(x, z, v38_loss_config(num_slices=8))
        factory.set_num_slices(8)
        truth = torch.as_tensor(x)
        pred = torch.as_tensor(z)
        torch.manual_seed(123)
        expected = sliced_wasserstein(truth, pred, 8, 2)
        torch.manual_seed(123)
        loss = factory.x_sim_loss(truth, pred)
        self.assertAlmostEqual(float(loss), float(expected), places=6)
        self.assertEqual(
            set(factory.latest_components),
            {"x_raw_swd", "x_raw_swd_raw", "x_raw_swd_weighted"},
        )

    def test_pair_swd_and_cycle_mass_huber_are_opt_in_and_finite(self) -> None:
        x, z = make_arrays(64)
        config = v38_loss_config(num_slices=8)
        config.update({
            "pair_swd_weight": 0.5,
            "cycle_mass_huber_weight": 50.0,
            "cycle_mass_huber_delta": 0.02,
        })
        factory = build_loss_factory(x, z, config)
        factory.set_num_slices(8)
        x_true = torch.as_tensor(x)
        z_true = torch.as_tensor(z)

        z_encoded = torch.as_tensor(z) + 0.05
        z_loss = factory.z_prior_loss(z_true, z_encoded)
        self.assertIn("z_pair_swd", factory.latest_components)
        self.assertGreater(float(z_loss), float(factory.latest_components["z_raw_swd"]))
        self.assertTrue(torch.isfinite(z_loss))

        x_from_z = torch.as_tensor(x) + 0.05
        x_sim_loss = factory.x_sim_loss(x_true, x_from_z)
        self.assertIn("x_pair_swd", factory.latest_components)
        self.assertGreater(float(x_sim_loss), float(factory.latest_components["x_raw_swd"]))
        self.assertTrue(torch.isfinite(x_sim_loss))

        x_reco = torch.as_tensor(x) + 0.05
        reco_loss = factory.x_reco_loss(x_true, x_reco)
        self.assertIn("x_reco_mass_huber_raw", factory.latest_components)
        self.assertGreater(float(reco_loss), float(factory.latest_components["x_reco_mse_raw"]))
        self.assertTrue(torch.isfinite(reco_loss))

    def test_pair_swd_default_weight_preserves_vanilla_components(self) -> None:
        x, z = make_arrays(64)
        factory = build_loss_factory(x, z, v38_loss_config(num_slices=8))
        x_true = torch.as_tensor(x)
        _ = factory.x_sim_loss(x_true, x_true)
        self.assertEqual(
            set(factory.latest_components),
            {"x_raw_swd", "x_raw_swd_raw", "x_raw_swd_weighted"},
        )
        factory.reset_components()
        _ = factory.x_reco_loss(x_true, x_true)
        self.assertEqual(set(factory.latest_components), {"x_reco_mse_raw"})

    def test_cycle_mass_huber_uses_relative_residual(self) -> None:
        x, z = make_arrays(64)
        config = v38_loss_config(num_slices=8)
        config.update({"cycle_mass_huber_weight": 1.0, "cycle_mass_huber_delta": 0.02})
        factory = build_loss_factory(x, z, config)
        x_true = torch.as_tensor(x)
        scale = 1.5
        x_reco = x_true * scale
        loss = factory.x_reco_loss(x_true, x_reco)
        # Relative residual is (1.5*m - m)/m = 0.5 everywhere; Huber at 0.5
        # is linear, so this should be finite and substantially larger than MSE.
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(factory.latest_components["x_reco_mass_huber_raw"]), 0.0)

    def test_alpha_stage_coefficient_sets_both_distribution_paths(self) -> None:
        stage = {
            "name": "core",
            "epochs": 100,
            "alpha": 1.0,
            "lambda_cycle": 1.0,
            "rho": 0.0,
            "nu_e": 0.0,
            "nu_d": 0.0,
            "num_slices": 500,
        }
        cfg = make_stage_loss_config(stage, 1, 100)
        self.assertEqual(cfg["alpha"], 1.0)
        self.assertEqual(cfg["lamb"], cfg["tau"])
        self.assertEqual(cfg["lamb"], 1.0)
        self.assertEqual(cfg["lambda_cycle"], 1.0)
        self.assertEqual(cfg["beta"], 1.0)

    def test_cosine_schedule_reaches_endpoints_and_stays_nonzero(self) -> None:
        stage = {
            "name": "core",
            "epochs": 220,
            "alpha": {"start": 1.0, "end": 0.25, "schedule": "cosine"},
            "lambda_cycle": 1.0,
        }
        first = make_stage_loss_config(stage, 1, 220)
        middle = make_stage_loss_config(stage, 111, 220)
        last = make_stage_loss_config(stage, 220, 220)
        self.assertAlmostEqual(first["alpha"], 1.0, places=12)
        self.assertAlmostEqual(last["alpha"], 0.25, places=12)
        self.assertLess(middle["alpha"], 1.0)
        self.assertGreater(middle["alpha"], 0.25)
        self.assertGreater(last["alpha"], 0.0)

    def test_alpha_conflicts_with_legacy_distribution_keys(self) -> None:
        stage = {"alpha": 1.0, "lamb": 1.0, "lambda_cycle": 1.0}
        with self.assertRaises(ValueError):
            make_stage_loss_config(stage, 1, 10)

    def test_legacy_scalar_stage_weights_are_unchanged(self) -> None:
        stage = {"beta": 2.0, "lamb": 1.0, "tau": 0.5, "rho": 0.0,
                 "nu_e": 0.0, "nu_d": 0.0, "num_slices": 100}
        cfg = make_stage_loss_config(stage, 1, 10)
        self.assertEqual(cfg["beta"], 2.0)
        self.assertEqual(cfg["lamb"], 1.0)
        self.assertEqual(cfg["tau"], 0.5)
        self.assertEqual(cfg["lambda_cycle"], 2.0)
        self.assertIsNone(cfg["alpha"])


class TestV38BackwardCompat(unittest.TestCase):
    def test_v37_retains_standardized_latent_swd(self) -> None:
        x, z = make_arrays(128)
        config = resolve_config(load_config(V37_CONFIG_PATHS[0]))
        factory = build_loss_factory(x, z, config["loss"])
        factory.set_num_slices(16)
        z_true = torch.as_tensor(z)
        z_encoded = torch.randn(128, 8)
        torch.manual_seed(7)
        actual = factory.z_prior_loss(z_true, z_encoded)
        torch.manual_seed(7)
        expected = sliced_wasserstein(
            factory.standardize_z_raw(z_true),
            factory.standardize_z_raw(z_encoded),
            16,
            2,
        )
        self.assertAlmostEqual(float(actual), float(expected), places=6)
        # The generalized flag is true for legacy vanilla_v3_7 configs too.
        self.assertIs(factory.vanilla_swae, True)
        self.assertIs(factory.vanilla_v3_7, True)

    def test_all_v37_configs_still_load(self) -> None:
        for path in V37_CONFIG_PATHS:
            config = resolve_config(load_config(path))
            self.assertIs(config["loss"]["vanilla_v3_7"], True, path.name)

    def test_vanilla_flags_accepted_by_strict_validation(self) -> None:
        from loss import validate_loss_config

        validate_loss_config({"kind": "cms_jpsi_doublemuon_loss", "vanilla_v3_7": True})
        validate_loss_config({"kind": "cms_jpsi_doublemuon_loss", "vanilla_swae": True})

    def test_non_vanilla_weighted_behavior_unchanged(self) -> None:
        x, z = make_arrays(64)
        factory = build_loss_factory(
            x,
            z,
            {
                "kind": "cms_jpsi_doublemuon_loss",
                "num_slices": 8,
                "raw_swd": 0.5,
            },
        )
        self.assertIs(factory.vanilla_swae, False)
        loss = factory.z_prior_loss(torch.as_tensor(z), torch.randn(64, 8))
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("z_raw_swd_weighted", factory.latest_components)
        self.assertIn("z_mass_w1", factory.latest_components)


class TestV38Loaders(unittest.TestCase):
    def test_every_train_event_visited_before_reshuffle(self) -> None:
        config = loader_config(25)
        arrays = split_arrays(x_train=100, z_train=60)
        train_loaders, _eval, info = build_loaders(
            config, arrays, None, torch.device("cpu")
        )
        x_loader, z_loader = train_loaders
        x_seen: list[int] = []
        z_seen: list[int] = []
        for x_batch, z_batch in zip(x_loader, z_loader):
            self.assertEqual(x_batch.shape[0], z_batch.shape[0])
            x_seen.extend(x_batch[:, 0].tolist())
            z_seen.extend(z_batch[:, 0].tolist())
        # Larger (x) domain: each of its 100 events exactly once.
        self.assertEqual(sorted(x_seen), sorted(range(100)))
        # Smaller (z) domain: all 60 events at least once.
        self.assertEqual(set(range(60)).issubset(set(z_seen)), True)
        self.assertEqual(info["train_repeated_x"], 0)
        self.assertEqual(info["train_repeated_z"], 40)

    def test_final_incomplete_batch_equal_cardinality(self) -> None:
        config = loader_config(25)
        arrays = split_arrays(x_train=103, z_train=61)
        train_loaders, _eval, info = build_loaders(
            config, arrays, None, torch.device("cpu")
        )
        batches = list(zip(train_loaders[0], train_loaders[1]))
        self.assertEqual(len(batches), 5)
        self.assertEqual(batches[-1][0].shape[0], 3)
        self.assertEqual(batches[-1][1].shape[0], 3)
        self.assertEqual(sum(b.shape[0] for b, _ in batches), 103)
        self.assertEqual(sum(b.shape[0] for _, b in batches), 103)
        self.assertEqual(info["train_repeated_z"], 42)

    def test_epoch_reshuffle_is_deterministic_and_changes(self) -> None:
        config = loader_config(25)
        arrays = split_arrays(x_train=100, z_train=100)
        t1, _e1, _ = build_loaders(config, arrays, None, torch.device("cpu"))
        t2, _e2, _ = build_loaders(config, arrays, None, torch.device("cpu"))
        epoch1_a = [b[:, 0].tolist() for b in t1[0]]
        epoch1_b = [b[:, 0].tolist() for b in t2[0]]
        self.assertEqual(epoch1_a, epoch1_b)
        epoch2_a = [b[:, 0].tolist() for b in t1[0]]
        self.assertNotEqual(epoch1_a, epoch2_a)

    def test_eval_loaders_deterministic_and_cover_splits(self) -> None:
        config = loader_config(20)
        arrays = split_arrays(x_train=30, z_train=30, x_val=53, z_val=41)
        _train, eval_loaders, info = build_loaders(
            config, arrays, None, torch.device("cpu")
        )
        self.assertEqual(info["eval_repeated_x"], 0)
        self.assertEqual(info["eval_repeated_z"], 12)

        def collect(loader):
            return [b[:, 0].tolist() for b in loader]

        x_pass1 = collect(eval_loaders[0])
        z_pass1 = collect(eval_loaders[1])
        x_pass2 = collect(eval_loaders[0])
        z_pass2 = collect(eval_loaders[1])
        self.assertEqual(x_pass1, x_pass2)
        self.assertEqual(z_pass1, z_pass2)
        x_seen = [item for batch in x_pass1 for item in batch]
        z_seen = [item for batch in z_pass1 for item in batch]
        self.assertEqual(sorted(x_seen), sorted(range(53)))
        self.assertEqual(set(range(41)).issubset(set(z_seen)), True)

    def test_unknown_sampler_policy_rejected(self) -> None:
        config = loader_config(25)
        config["loaders"]["train_sampler"] = "with_replacement_typo"
        arrays = split_arrays(x_train=10, z_train=10)
        with self.assertRaises(ValueError):
            build_loaders(config, arrays, None, torch.device("cpu"))

    def test_aligned_batch_sizes_schedule(self) -> None:
        self.assertEqual(_aligned_batch_sizes(103, 25), [25, 25, 25, 25, 3])
        self.assertEqual(_aligned_batch_sizes(100, 25), [25, 25, 25, 25])
        self.assertEqual(_aligned_batch_sizes(7, 25), [7])
        with self.assertRaises(ValueError):
            _aligned_batch_sizes(0, 25)

    def test_loader_rejects_batch_larger_than_domain(self) -> None:
        tensor = torch.zeros((5, 8))
        with self.assertRaises(ValueError):
            DeterministicSequentialLoader(tensor, [6])


class TestV38Operational(unittest.TestCase):
    def test_make_run_dir_refuses_overwrite(self) -> None:
        import train

        with tempfile.TemporaryDirectory() as td:
            output_root = Path(td) / "outputs"
            run_dir = output_root / "Jpsi_v3.8_vanilla_paper_all_data_seed0"
            run_dir.mkdir(parents=True)
            (run_dir / "checkpoint_final.pt").write_bytes(b"checkpoint")
            config = {"paths": {"output_root": str(output_root)}}
            with self.assertRaises(FileExistsError):
                train.make_run_dir(config, "Jpsi_v3.8_vanilla_paper_all_data_seed0", dry_run=False)
            # A dry run does not raise and does not create the directory.
            empty = output_root / "smoke_dry"
            train.make_run_dir(config, "smoke_dry", dry_run=True)
            self.assertFalse(empty.exists())

    def test_validate_loaded_arrays_fail_fast(self) -> None:
        import train

        good = {
            key: np.zeros((4, 8), dtype=np.float32)
            for key in ("x_train", "x_val", "x_test", "z_train", "z_val", "z_test")
        }
        train.validate_loaded_arrays(good)
        bad = dict(good)
        bad["x_train"] = np.zeros((4, 7), dtype=np.float32)
        with self.assertRaises(ValueError):
            train.validate_loaded_arrays(bad)
        nan = dict(good)
        nan["z_test"][0, 0] = np.nan
        with self.assertRaises(ValueError):
            train.validate_loaded_arrays(nan)

    def test_checkpoint_roundtrip_with_metadata(self) -> None:
        x, z = make_arrays(32)
        model = build_model(
            tiny_model_config(),
            x.mean(axis=0),
            np.ones(8, dtype=np.float32),
            z.mean(axis=0),
            np.ones(8, dtype=np.float32),
        )
        config = {
            "model": tiny_model_config()["model"],
            "float_type": "float32",
        }
        stats = {
            "x_train_mean": x.mean(axis=0),
            "x_train_std": np.ones(8, dtype=np.float32),
            "z_train_mean": z.mean(axis=0),
            "z_train_std": np.ones(8, dtype=np.float32),
        }
        metadata = {"seed": 0, "loader_policy": "shuffled_without_replacement"}
        payload = checkpoint_payload(
            model, config, stats, 1, 0.5, {"selected": "cpu"}, metadata=metadata
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "checkpoint.pt"
            torch.save(payload, path)
            reloaded, _cfg, _stats, checkpoint = load_model_from_checkpoint(
                path, config=None, map_location=torch.device("cpu")
            )
        self.assertEqual(checkpoint["run_metadata"], metadata)
        for key, expected in model.state_dict().items():
            self.assertTrue(torch.equal(reloaded.state_dict()[key], expected), key)

    def test_diagnostics_rng_snapshot_restore(self) -> None:
        from encoder_diagnostics import _restore_rng, _snapshot_rng

        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        state = _snapshot_rng()
        expected = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        # Perturb every RNG stream.
        random.random()
        np.random.rand()
        torch.rand(1)
        _restore_rng(state)
        actual = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        self.assertEqual(actual, expected)

    def test_sha256_fingerprint_stable(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "file.bin"
            path.write_bytes(b"preflight-payload")
            first = sha256_fingerprint(path)
            second = sha256_fingerprint(path)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 64)

    def test_inspect_theory_prior(self) -> None:
        import h5py

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "prior.hdf5"
            with h5py.File(path, "w") as handle:
                handle.create_dataset("FDL/zData", data=np.zeros((10, 8), dtype=np.float32))
            report = inspect_theory_prior(path)
            self.assertEqual(report["dataset_key"], "FDL/zData")
            self.assertEqual(report["shape"], [10, 8])
            self.assertTrue(report["eight_dimensional_compatible"])
            self.assertTrue(report["all_finite"])
            self.assertNotIn("error", report)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.hdf5"
            with h5py.File(path, "w") as handle:
                handle.create_dataset("wrong_key", data=np.zeros((4, 4)))
            report = inspect_theory_prior(path)
            self.assertIn("error", report)

    def test_inspect_cms_root_mocked(self) -> None:
        class FakeEvents:
            num_entries = 1234

            def keys(self):
                return {
                    "nMuon",
                    "Muon_pt",
                    "Muon_eta",
                    "Muon_phi",
                    "Muon_mass",
                    "Muon_charge",
                }

        class FakeRoot:
            def __init__(self, events):
                self.events = events

            def keys(self):
                return ["Events"]

            def __getitem__(self, key):
                return self.events

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "cms.root"
            path.write_bytes(b"fake-root")
            with mock.patch("uproot.open", return_value=FakeRoot(FakeEvents())):
                report = inspect_cms_root(path, {}, channel="muon")
            self.assertEqual(report["total_event_count"], 1234)
            self.assertEqual(report["missing_branches"], [])
            self.assertNotIn("error", report)


if __name__ == "__main__":
    unittest.main()
