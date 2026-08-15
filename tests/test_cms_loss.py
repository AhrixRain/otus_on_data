from __future__ import annotations

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

import cms_data  # noqa: E402
from cms_data import load_config, resolve_config  # noqa: E402
from cms_training import build_loss_factory  # noqa: E402
from cms_model import build_model  # noqa: E402
from loss import (  # noqa: E402
    CANONICAL_LOSS_KIND,
    JPSI_DIMUON_LOSS_KIND,
    CmsJpsiDoubleMuonLossFactory,
    build_ee_physics_features,
    sliced_wasserstein,
)
from metrics import residual_metrics  # noqa: E402


def fake_ee_batch(n: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    momenta = torch.randn(n, 6, generator=generator) * 20.0
    px_m, py_m, pz_m = momenta[:, 0], momenta[:, 1], momenta[:, 2]
    px_p, py_p, pz_p = momenta[:, 3], momenta[:, 4], momenta[:, 5]
    e_m = torch.sqrt(px_m**2 + py_m**2 + pz_m**2 + 0.000511**2) + 5.0
    e_p = torch.sqrt(px_p**2 + py_p**2 + pz_p**2 + 0.000511**2) + 5.0
    return torch.stack([px_m, py_m, pz_m, e_m, px_p, py_p, pz_p, e_p], dim=1)


def reference_w1_sorted(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Old per-column 1D Wasserstein-1 used before the batched-sort refactor."""
    a_sorted = torch.sort(a.reshape(-1))[0]
    b_sorted = torch.sort(b.reshape(-1))[0]
    n = min(a_sorted.numel(), b_sorted.numel())
    return torch.mean(torch.abs(a_sorted[:n] - b_sorted[:n]))


def reference_marginal_w1(factory, truth: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    truth_std = factory.standardize_raw(truth)
    pred_std = factory.standardize_raw(pred)
    loss = truth_std.new_tensor(0.0)
    for dim in range(truth_std.shape[1]):
        loss = loss + reference_w1_sorted(truth_std[:, dim], pred_std[:, dim])
    return loss / truth_std.shape[1]


def reference_feature_w1(factory, truth_f, pred_f, mean, std) -> torch.Tensor:
    truth_std = factory.standardize_features(truth_f, mean, std)
    pred_std = factory.standardize_features(pred_f, mean, std)
    loss = truth_std.new_tensor(0.0)
    for dim in range(truth_std.shape[1]):
        loss = loss + reference_w1_sorted(truth_std[:, dim], pred_std[:, dim])
    return loss / truth_std.shape[1]


def reference_tail_w1(factory, truth: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    truth_std = factory.standardize_raw(truth)
    pred_std = factory.standardize_raw(pred)
    loss = truth_std.new_tensor(0.0)
    for dim in (0, 1, 2, 4, 5, 6):
        a = torch.sort(torch.abs(truth_std[:, dim].reshape(-1)))[0]
        b = torch.sort(torch.abs(pred_std[:, dim].reshape(-1)))[0]
        n = min(a.numel(), b.numel())
        start = max(0, min(int((1.0 - factory.tail_frac) * n), n - 1))
        loss = loss + torch.mean(torch.abs(a[start:n] - b[start:n]))
    return loss / 6.0


def reference_distribution_components(factory, truth: torch.Tensor, pred: torch.Tensor):
    """Pre-refactor algorithm: per-column sorts and repeated feature builds."""
    truth_std = factory.standardize_raw(truth)
    pred_std = factory.standardize_raw(pred)
    truth_named = build_ee_physics_features(
        truth,
        factory.eps,
        factory.daughter_masses,
        mass_from_energy=factory.mass_from_energy,
    )
    pred_named = build_ee_physics_features(
        pred,
        factory.eps,
        factory.daughter_masses,
        mass_from_energy=factory.mass_from_energy,
    )
    truth_features_std = factory.standardize_features(
        truth_named["physics_features"], factory.feature_mean, factory.feature_std
    )
    pred_features_std = factory.standardize_features(
        pred_named["physics_features"], factory.feature_mean, factory.feature_std
    )
    truth_coord_std = factory.standardize_features(
        truth_named["physics_coord_features"], factory.physics_coord_mean, factory.physics_coord_std
    )
    pred_coord_std = factory.standardize_features(
        pred_named["physics_coord_features"], factory.physics_coord_mean, factory.physics_coord_std
    )
    return {
        "raw_swd": sliced_wd_fixed_seed(truth_std, pred_std, factory),
        "marginal_w1": reference_marginal_w1(factory, truth, pred),
        "mass_w1": reference_w1_sorted(
            factory.standardize_mass(truth_named["m_ee"]),
            factory.standardize_mass(pred_named["m_ee"]),
        ),
        "resonance_mass_w1": factory.resonance_mass_w1(
            truth_named["m_ee"],
            pred_named["m_ee"],
        ),
        "physics_swd": sliced_wd_fixed_seed(truth_features_std, pred_features_std, factory),
        "mass_kin_swd": sliced_wd_fixed_seed(
            truth_features_std[:, 4:9],
            pred_features_std[:, 4:9],
            factory,
        ),
        "transverse_w1": reference_feature_w1(
            factory,
            factory.transverse_features(truth),
            factory.transverse_features(pred),
            factory.transverse_mean,
            factory.transverse_std,
        ),
        "longitudinal_w1": reference_feature_w1(
            factory,
            factory.longitudinal_features(truth),
            factory.longitudinal_features(pred),
            factory.longitudinal_mean,
            factory.longitudinal_std,
        ),
        "tail_w1": reference_tail_w1(factory, truth, pred),
        "pair_mass_w1": reference_w1_sorted(
            factory.standardize_mass(truth_named["m_ee"]),
            factory.standardize_mass(pred_named["m_ee"]),
        ),
        "pair_pt_w1": reference_w1_sorted(
            factory.standardized_physics_column(truth, 5),
            factory.standardized_physics_column(pred, 5),
        ),
        "lepton_pt_w1": 0.5
        * (
            reference_w1_sorted(
                factory.standardized_physics_column(truth, 0),
                factory.standardized_physics_column(pred, 0),
            )
            + reference_w1_sorted(
                factory.standardized_physics_column(truth, 1),
                factory.standardized_physics_column(pred, 1),
            )
        ),
        "delta_phi_w1": reference_w1_sorted(
            truth_named["delta_phi"] / torch.pi,
            pred_named["delta_phi"] / torch.pi,
        ),
        "delta_eta_w1": reference_w1_sorted(
            truth_named["delta_eta"]
            / (factory.to_like(factory.delta_eta_std, truth) + factory.eps),
            pred_named["delta_eta"]
            / (factory.to_like(factory.delta_eta_std, pred) + factory.eps),
        ),
        "pair_rapidity_w1": reference_w1_sorted(
            factory.standardized_physics_column(truth, 6),
            factory.standardized_physics_column(pred, 6),
        ),
        "physics_coord_swd": sliced_wd_fixed_seed(truth_coord_std, pred_coord_std, factory),
    }


def sliced_wd_fixed_seed(a: torch.Tensor, b: torch.Tensor, factory):
    from loss import sliced_wasserstein

    return sliced_wasserstein(a, b, factory.num_slices, factory.p)


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
            load_config(REPO_ROOT / "configs/archive/cms_doubleelectron_mps.yaml")
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

    def test_jpsi_muon_config_reuses_v5_network_and_data_directory(self) -> None:
        config = resolve_config(
            load_config(REPO_ROOT / "configs/archive/cms_JpsiDoubleMuons_mps.yaml")
        )
        self.assertEqual(config["data"]["channel"], "muon")
        self.assertEqual(config["loss"]["kind"], JPSI_DIMUON_LOSS_KIND)
        self.assertEqual(config["model"]["num_hidden_layers"], 4)
        self.assertEqual(config["model"]["dim_per_hidden_layer"], 512)
        self.assertEqual(Path(config["paths"]["cms_root_file"]).parent, REPO_ROOT / "data")
        self.assertEqual(Path(config["paths"]["theory_prior_file"]).parent, REPO_ROOT / "data")
        self.assertEqual(config["evaluation"]["mass_range"], [2.6, 3.5])

    def test_jpsi_v36a_config_removes_only_explicit_mass_supervision(self) -> None:
        base = resolve_config(load_config(REPO_ROOT / "configs/archive/cms_JpsiDoubleMuons_mps.yaml"))
        v36a = resolve_config(
            load_config(
                REPO_ROOT / "configs/archive/cms_JpsiDoubleMuons_Jpsi_v3.6A_no_explicit_mass.yaml"
            )
        )
        base_loss = base["loss"]
        v36a_loss = v36a["loss"]
        # Explicit mass weights: top level and both space overrides must be 0.
        self.assertEqual(v36a_loss["mass_w1"], 0.0)
        self.assertEqual(v36a_loss["pair_mass_w1"], 0.0)
        self.assertEqual(v36a_loss["space_weights"]["x"]["mass_w1"], 0.0)
        self.assertEqual(v36a_loss["space_weights"]["x"]["resonance_mass_w1"], 0.0)
        self.assertEqual(v36a_loss["space_weights"]["z"]["mass_w1"], 0.0)
        self.assertEqual(v36a_loss["space_weights"]["z"]["resonance_mass_w1"], 0.0)
        self.assertEqual(v36a_loss["mass_kin_swd_components"]["mll"], 0.0)
        # The four non-mass pair-kinematics columns stay active.
        for name in ("ptll", "yll", "cos_dphi", "sin_dphi"):
            self.assertEqual(v36a_loss["mass_kin_swd_components"][name], 1.0)
        # Everything else must be unchanged from v3.5 (controlled ablation).
        v36a_only_mass = dict(v36a_loss)
        v36a_only_mass.pop("mass_kin_swd_components")
        expected_only_mass = {
            "mass_w1": 0.0,
            "pair_mass_w1": 0.0,
            "space_weights": {
                "x": {"mass_w1": 0.0, "resonance_mass_w1": 0.0},
                "z": {"mass_w1": 0.0, "resonance_mass_w1": 0.0},
            },
        }
        for key, value in expected_only_mass.items():
            self.assertEqual(v36a_loss[key], value)
        base_loss_massless = {
            key: value
            for key, value in base_loss.items()
            if key
            not in {
                "mass_w1",
                "pair_mass_w1",
                "space_weights",
                "resonance_mass_center",
                "resonance_mass_half_width",
            }
        }
        v36a_loss_massless = {
            key: value
            for key, value in v36a_only_mass.items()
            if key
            not in {
                "mass_w1",
                "pair_mass_w1",
                "space_weights",
                "resonance_mass_center",
                "resonance_mass_half_width",
            }
        }
        self.assertEqual(v36a_loss_massless, base_loss_massless)

    def test_jpsi_muon_loss_alias_uses_dilepton_objective(self) -> None:
        truth_x = fake_ee_batch(24, seed=10).numpy().astype("float32")
        truth_z = fake_ee_batch(24, seed=11).numpy().astype("float32")
        loss_factory = build_loss_factory(
            truth_x,
            truth_z,
            {"kind": JPSI_DIMUON_LOSS_KIND, "num_slices": 4},
        )
        self.assertIsInstance(loss_factory, CmsJpsiDoubleMuonLossFactory)
        pred = fake_ee_batch(24, seed=12)
        loss = loss_factory.x_sim_loss(torch.as_tensor(truth_x), pred)
        self.assertTrue(torch.isfinite(loss))

    def test_space_weights_override_per_space(self) -> None:
        truth_x = fake_ee_batch(32, seed=20).numpy().astype("float32")
        truth_z = fake_ee_batch(32, seed=21).numpy().astype("float32")
        loss_factory = build_loss_factory(
            truth_x,
            truth_z,
            {
                "kind": JPSI_DIMUON_LOSS_KIND,
                "num_slices": 4,
                "space_weights": {
                    "z": {"mass_w1": 0.6, "resonance_mass_w1": 2.0},
                    "x": {"mass_w1": 0.3, "resonance_mass_w1": 0.0},
                },
            },
        )
        self.assertEqual(loss_factory.z_space.weights["mass_w1"], 0.6)
        self.assertEqual(loss_factory.z_space.weights["resonance_mass_w1"], 2.0)
        self.assertEqual(loss_factory.x_space.weights["mass_w1"], 0.3)
        self.assertEqual(loss_factory.x_space.weights["resonance_mass_w1"], 0.0)
        components = loss_factory.z_space.distribution_components(
            torch.as_tensor(truth_z),
            fake_ee_batch(32, seed=22),
        )
        self.assertIn("resonance_mass_w1", components)
        self.assertIn("mass_kin_swd", components)
        self.assertTrue(torch.isfinite(components["resonance_mass_w1"]))
        self.assertTrue(torch.isfinite(components["mass_kin_swd"]))

    def test_resonance_mass_w1_only_penalizes_window(self) -> None:
        truth_x = fake_ee_batch(16, seed=23).numpy().astype("float32")
        truth_z = fake_ee_batch(16, seed=24).numpy().astype("float32")
        factory = build_loss_factory(
            truth_x,
            truth_z,
            {"kind": CANONICAL_LOSS_KIND, "num_slices": 4},
        ).x_space
        center = 3.0969
        half = 0.06
        truth_mass = torch.tensor(
            [center - 0.01, center, center + 0.01, 2.5, 4.0],
            dtype=torch.float32,
        )
        # In-window values identical, out-of-window values wildly different:
        # the loss must be exactly zero because the window masks them out.
        pred_close = torch.tensor(
            [center - 0.01, center, center + 0.01, 6.0, 0.5],
            dtype=torch.float32,
        )
        loss_close = float(factory.resonance_mass_w1(truth_mass, pred_close))
        self.assertEqual(loss_close, 0.0)
        # A shift inside the window should be penalized.
        pred_shifted = torch.tensor(
            [center + 0.03, center + 0.04, center + 0.05, 2.5, 4.0],
            dtype=torch.float32,
        )
        loss_shifted = float(factory.resonance_mass_w1(truth_mass, pred_shifted))
        self.assertGreater(loss_shifted, 0.01)

    def test_mass_kin_swd_mass_component_is_independently_disableable(self) -> None:
        truth_x = fake_ee_batch(64, seed=50).numpy().astype("float32")
        truth_z = fake_ee_batch(64, seed=51).numpy().astype("float32")
        no_mass_config = {
            "kind": JPSI_DIMUON_LOSS_KIND,
            "num_slices": 16,
            "mass_kin_swd_components": {
                "mll": 0.0,
                "ptll": 1.0,
                "yll": 1.0,
                "cos_dphi": 1.0,
                "sin_dphi": 1.0,
            },
        }
        default = build_loss_factory(
            truth_x,
            truth_z,
            {"kind": JPSI_DIMUON_LOSS_KIND, "num_slices": 16},
        ).x_space
        no_mass = build_loss_factory(truth_x, truth_z, no_mass_config).x_space
        truth = torch.as_tensor(truth_x)
        pred = fake_ee_batch(64, seed=52)

        # Disabling mll must equal a fresh SWD over the remaining columns 5:9
        # (pT_ll, y_ll, cos dphi, sin dphi) with the same slice count.
        truth_std = no_mass.standardize_features(
            build_ee_physics_features(truth, no_mass.eps)["physics_features"],
            no_mass.feature_mean,
            no_mass.feature_std,
        )
        pred_std = no_mass.standardize_features(
            build_ee_physics_features(pred, no_mass.eps)["physics_features"],
            no_mass.feature_mean,
            no_mass.feature_std,
        )
        torch.manual_seed(0)
        expected = sliced_wasserstein(truth_std[:, 5:9], pred_std[:, 5:9], 16, 2)
        torch.manual_seed(0)
        actual = no_mass._mass_kin_swd(truth_std, pred_std)
        self.assertTrue(torch.isfinite(actual))
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        # With all five components active the value stays finite and, by the
        # default-config reference test, matches the pre-refactor 4:9 bundle.
        default_value = float(default.distribution_components(truth, pred)["mass_kin_swd"])
        self.assertTrue(np.isfinite(default_value))
        self.assertEqual(default.mass_kin_swd_component_weights["mll"], 1.0)
        self.assertEqual(no_mass.mass_kin_swd_component_weights["mll"], 0.0)

    def test_mass_kin_swd_all_components_disabled_is_zero(self) -> None:
        truth_x = fake_ee_batch(32, seed=53).numpy().astype("float32")
        truth_z = fake_ee_batch(32, seed=54).numpy().astype("float32")
        factory = build_loss_factory(
            truth_x,
            truth_z,
            {
                "kind": JPSI_DIMUON_LOSS_KIND,
                "num_slices": 8,
                "mass_kin_swd_components": {
                    "mll": 0.0,
                    "ptll": 0.0,
                    "yll": 0.0,
                    "cos_dphi": 0.0,
                    "sin_dphi": 0.0,
                },
            },
        ).x_space
        value = float(
            factory.distribution_components(
                torch.as_tensor(truth_x),
                fake_ee_batch(32, seed=55),
            )["mass_kin_swd"]
        )
        self.assertEqual(value, 0.0)

    def test_x_reco_physics_mse_is_config_gated(self) -> None:
        truth_x = fake_ee_batch(32, seed=25).numpy().astype("float32")
        truth_z = fake_ee_batch(32, seed=26).numpy().astype("float32")
        x_true = torch.as_tensor(truth_x)
        x_reco = fake_ee_batch(32, seed=27)
        off = build_loss_factory(
            truth_x,
            truth_z,
            {"kind": CANONICAL_LOSS_KIND, "num_slices": 4},
        )
        on = build_loss_factory(
            truth_x,
            truth_z,
            {
                "kind": CANONICAL_LOSS_KIND,
                "num_slices": 4,
                "x_reco_physics_w1": 1.0,
            },
        )
        base = float(off.x_reco_loss(x_true, x_reco))
        with_physics = float(on.x_reco_loss(x_true, x_reco))
        self.assertGreater(with_physics, base)
        self.assertTrue(torch.isfinite(torch.as_tensor(with_physics)))

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

    def test_batched_w1_matches_reference(self) -> None:
        truth = fake_ee_batch(64, seed=31)
        pred = fake_ee_batch(64, seed=32)
        factory = build_loss_factory(
            truth.numpy().astype("float32"),
            pred.numpy().astype("float32"),
            {"kind": CANONICAL_LOSS_KIND, "num_slices": 8},
        )
        space = factory.x_space
        self.assertTrue(
            torch.allclose(
                space.marginal_w1(truth, pred),
                reference_marginal_w1(space, truth, pred),
                atol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                space.tail_w1(truth, pred),
                reference_tail_w1(space, truth, pred),
                atol=1e-5,
            )
        )

    def test_distribution_components_match_pre_refactor_reference(self) -> None:
        truth_x = fake_ee_batch(96, seed=41).numpy().astype("float32")
        truth_z = fake_ee_batch(96, seed=42).numpy().astype("float32")
        factory = build_loss_factory(
            truth_x,
            truth_z,
            {"kind": CANONICAL_LOSS_KIND, "num_slices": 16},
        )
        truth = torch.as_tensor(truth_x)
        pred = fake_ee_batch(96, seed=43)
        torch.manual_seed(0)
        new_components = factory.x_space.distribution_components(truth, pred)
        torch.manual_seed(0)
        ref_components = reference_distribution_components(factory.x_space, truth, pred)
        self.assertEqual(set(new_components.keys()), set(ref_components.keys()) | {"mmd"})
        for key in ref_components:
            self.assertTrue(
                torch.allclose(new_components[key], ref_components[key], atol=1e-4, rtol=1e-3),
                f"component {key} diverged from the pre-refactor reference",
            )

    def test_history_logger_covers_all_loss_components(self) -> None:
        from cms_training import HistoryLogger

        fields = set(HistoryLogger.component_fields)
        for expected in (
            "train_x_raw_swd",
            "train_x_marginal_w1",
            "train_x_mass_w1",
            "train_x_resonance_mass_w1",
            "train_x_physics_swd",
            "train_z_mass_kin_swd",
            "train_x_transverse_w1",
            "train_x_longitudinal_w1",
            "train_x_tail_w1",
            "train_x_pair_mass_w1",
            "train_x_physics_coord_swd",
            "train_z_physics_swd",
        ):
            self.assertIn(expected, fields)
        self.assertEqual(len(fields), 32)

    def test_data_cache_roundtrip_and_keying(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            root_file = tmp / "cms.root"
            prior_file = tmp / "prior.hdf5"
            root_file.write_bytes(b"root")
            prior_file.write_bytes(b"prior")
            config = {
                "paths": {
                    "output_root": str(tmp / "outputs"),
                    "cms_root_file": str(root_file),
                    "theory_prior_file": str(prior_file),
                },
                "data": {"channel": "electron"},
                "electron_selection": {"electron_pt_min": 20.0, "z_mass_max": 110.0},
                "float_type": "float32",
                "seed": 0,
                "data_split": {"train_ratio": 0.8, "val_ratio": 0.1},
            }
            fake_arrays = {
                key: np.full((4, 8), float(i), dtype=np.float32)
                for i, key in enumerate(
                    ("x_train", "x_val", "x_test", "z_train", "z_val", "z_test")
                )
            }
            with mock.patch.object(cms_data, "load_and_split", return_value=fake_arrays) as loader:
                _arrays, info1 = cms_data.load_and_split_cached(config, None, None, True)
                arrays2, info2 = cms_data.load_and_split_cached(config, None, None, True)
                self.assertFalse(info1["hit"])
                self.assertTrue(info2["hit"])
                self.assertEqual(loader.call_count, 1)
                for key, value in fake_arrays.items():
                    np.testing.assert_array_equal(arrays2[key], value)
                _arrays, info3 = cms_data.load_and_split_cached(config, 5, None, True)
                self.assertNotEqual(info3["key"], info1["key"])

    def test_model_build_forward_backward_with_cached_stats(self) -> None:
        model_config = {
            "model": {
                "class": "CondNoiseAutoencoder",
                "num_hidden_layers": 1,
                "dim_per_hidden_layer": 8,
                "activation": "ReLU",
                "stoch_enc": True,
                "stoch_dec": True,
                "raw_io": True,
                "sigma_fun": "softplus",
                "sigma_floor": 0.001,
            }
        }
        model = build_model(
            model_config,
            np.zeros(8, dtype=np.float32),
            np.ones(8, dtype=np.float32),
            np.zeros(8, dtype=np.float32),
            np.ones(8, dtype=np.float32),
        )
        model.eval()
        with torch.no_grad():
            z, x_tilde = model(torch.zeros(4, 8))
        self.assertEqual(z.shape, (4, 8))
        self.assertEqual(x_tilde.shape, (4, 8))
        self.assertTrue(torch.isfinite(z).all())
        self.assertTrue(torch.isfinite(x_tilde).all())
        cached_input, _cached_output = model.encoder._stats_for(torch.device("cpu"))
        self.assertIsNotNone(cached_input)
        self.assertTrue(torch.equal(cached_input, model.encoder.input_stats))
        x = torch.randn(4, 8, requires_grad=True)
        _z, x_tilde = model(x)
        x_tilde.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())


if __name__ == "__main__":
    unittest.main()
