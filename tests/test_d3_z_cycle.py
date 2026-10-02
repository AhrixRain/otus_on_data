"""D3 z-space consistency cycle: loss contract, noise policy, one-change config."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (
    REPO_ROOT / "scripts",
    REPO_ROOT / "scripts_sota",
    REPO_ROOT / "scripts_joint",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from cms_data import load_config  # noqa: E402
from joint_data import resolve_joint_config  # noqa: E402
from joint_metrics import current_noise_multipliers, set_noise_multipliers  # noqa: E402
from joint_model import build_joint_autoencoder  # noqa: E402
from joint_trainer import _resolve_z_cycle_noise, _z_cycle_roundtrip, train_joint_epoch  # noqa: E402
from loss import CmsJpsiDoubleMuonLossFactory  # noqa: E402


MUON_MASS = 0.1056583755
FROZEN_CONFIG = REPO_ROOT / "configs_joint" / "cms_Joint_runH_A2frozen.yaml"
D3_CONFIG = REPO_ROOT / "configs_joint" / "cms_Joint_runH_D3zcycle.yaml"


def _pairs(n: int, mass: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    momentum = np.sqrt((mass / 2.0) ** 2 - MUON_MASS**2)
    phi = rng.uniform(-np.pi, np.pi, n)
    eta = rng.normal(0.0, 0.35, n)
    pt = momentum / np.cosh(eta)
    recoil_x = rng.normal(0.0, 0.02 * mass, n)
    recoil_y = rng.normal(0.0, 0.02 * mass, n)
    px1 = pt * np.cos(phi) + recoil_x / 2
    py1 = pt * np.sin(phi) + recoil_y / 2
    pz1 = pt * np.sinh(eta)
    px2 = -pt * np.cos(phi) + recoil_x / 2
    py2 = -pt * np.sin(phi) + recoil_y / 2
    pz2 = -pz1
    e1 = np.sqrt(px1**2 + py1**2 + pz1**2 + MUON_MASS**2)
    e2 = np.sqrt(px2**2 + py2**2 + pz2**2 + MUON_MASS**2)
    return np.stack([px1, py1, pz1, e1, px2, py2, pz2, e2], axis=1).astype(np.float32)


def _model_config() -> dict:
    return {
        "hidden_dims": [16],
        "activation": "SiLU",
        "flow_steps": 1,
        "condition_features": [i for i in range(14) if i != 8],
        "condition_stats_max_events_per_region": 32,
        "mean_residual_limits": [0.1] * 6,
        "core_sigma_floors": [0.001] * 6,
        "core_sigma_scales": [0.01] * 6,
        "tail_sigma_scales": [0.01] * 6,
        "core_log_sigma_bias": -5.0,
        "tail_log_sigma_bias": -6.0,
        "student_t_degrees_of_freedom": 4.0,
        "maximum_heavy_noise": 4.0,
        "daughter_masses": [MUON_MASS, MUON_MASS],
    }


def _arrays() -> dict[str, dict[str, np.ndarray]]:
    return {
        "jpsi": {"x_train": _pairs(32, 3.0969, 1), "z_train": _pairs(32, 3.0969, 2)},
        "z": {"x_train": _pairs(32, 91.1876, 3), "z_train": _pairs(32, 91.1876, 4)},
    }


def _loss_config() -> dict:
    return {
        "kind": "cms_jpsi_doublemuon_loss",
        "num_slices": 4,
        "p": 1,
        "raw_swd": 0.1,
        "marginal_w1": 0.0,
        "mass_w1": 0.0,
        "resonance_mass_w1": 0.0,
        "physics_swd": 0.0,
        "mass_kin_swd": 0.0,
        "transverse_w1": 0.0,
        "longitudinal_w1": 0.0,
        "tail_w1": 0.0,
        "pair_mass_w1": 0.1,
        "pair_pt_w1": 0.0,
        "lepton_pt_w1": 0.0,
        "delta_phi_w1": 0.0,
        "delta_eta_w1": 0.0,
        "pair_rapidity_w1": 0.0,
        "physics_coord_swd": 0.0,
        "mmd": 0.0,
        "x_reco_physics_w1": 0.1,
    }


def _factories(arrays: dict) -> dict:
    return {
        name: CmsJpsiDoubleMuonLossFactory(
            region["x_train"], region["z_train"], _loss_config(), [MUON_MASS, MUON_MASS]
        )
        for name, region in arrays.items()
    }


class ZCycleLossTests(unittest.TestCase):
    def test_exact_inverse_scores_zero(self):
        arrays = _arrays()
        factories = _factories(arrays)
        z = torch.as_tensor(arrays["jpsi"]["z_train"])
        value = factories["jpsi"].z_cycle_loss(z, z.clone())
        self.assertLess(abs(float(value)), 1e-9)

    def test_perturbed_inverse_is_positive(self):
        arrays = _arrays()
        factories = _factories(arrays)
        z = torch.as_tensor(arrays["jpsi"]["z_train"])
        value = factories["jpsi"].z_cycle_loss(z, z + 0.05)
        self.assertGreater(float(value), 0.0)

    def test_gradient_flows_through_the_roundtrip_to_the_model(self):
        # A shifted copy gives a constant paired residual, so the meaningful
        # gradient test goes through encode(decode(z)) into the model parameters
        # - exactly the training path.
        arrays = _arrays()
        model = build_joint_autoencoder(
            _model_config(), arrays, MUON_MASS, [MUON_MASS, MUON_MASS]
        )
        factories = _factories(arrays)
        z = torch.as_tensor(arrays["jpsi"]["z_train"])
        z_reco = _z_cycle_roundtrip(model, z, "native")
        value = factories["jpsi"].z_cycle_loss(z, z_reco)
        value.backward()
        total = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(total, 0.0)

    def test_components_are_registered(self):
        arrays = _arrays()
        factories = _factories(arrays)
        z = torch.as_tensor(arrays["jpsi"]["z_train"])
        factories["jpsi"].reset_components()
        factories["jpsi"].z_cycle_loss(z, z + 0.02)
        self.assertIn("z_cycle_mse_raw", factories["jpsi"].latest_components)


class ZCycleNoisePolicyTests(unittest.TestCase):
    def test_default_is_native(self):
        self.assertEqual(_resolve_z_cycle_noise({}, {}), "native")

    def test_stage_overrides_the_top_level_key(self):
        self.assertEqual(
            _resolve_z_cycle_noise({"z_cycle_noise": "native"}, {"z_cycle_noise": "zero"}),
            "zero",
        )

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            _resolve_z_cycle_noise({}, {"z_cycle_noise": "sometimes"})

    def test_zero_mode_restores_the_multipliers(self):
        arrays = _arrays()
        model = build_joint_autoencoder(
            _model_config(), arrays, MUON_MASS, [MUON_MASS, MUON_MASS]
        )
        schedule = {
            "encoder_core": 1.0,
            "encoder_tail": 0.5,
            "decoder_core": 1.0,
            "decoder_tail": 0.5,
        }
        set_noise_multipliers(model, schedule)
        z = torch.as_tensor(arrays["jpsi"]["z_train"])
        zeroed = _z_cycle_roundtrip(model, z, "zero")
        native = _z_cycle_roundtrip(model, z, "native")
        self.assertEqual(tuple(zeroed.shape), tuple(z.shape))
        self.assertEqual(tuple(native.shape), tuple(z.shape))
        self.assertEqual(current_noise_multipliers(model), schedule)
        with self.assertRaises(ValueError):
            _z_cycle_roundtrip(model, z, "half")


class ZCycleEpochTests(unittest.TestCase):
    def _epoch(self, zeta: float):
        arrays = _arrays()
        model = build_joint_autoencoder(
            _model_config(), arrays, MUON_MASS, [MUON_MASS, MUON_MASS]
        )
        factories = _factories(arrays)
        config = {
            "region_order": ["jpsi", "z"],
            "region_weights": {"jpsi": 1.0, "z": 1.0},
            "loaders": {"train_batch_size": 8, "steps_per_epoch": 1},
        }
        stage = {
            "name": "synthetic_d3_stage",
            "beta": 1.0,
            "lamb": 1.0,
            "tau": 1.0,
            "nu_e": 0.0,
            "nu_d": 0.0,
            "gradient_clip_norm": 1.0,
            "zeta": zeta,
        }
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
        return train_joint_epoch(
            model, optimizer, arrays, factories, config, stage,
            device=torch.device("cpu"), seed=7,
        )

    def test_term_is_absent_when_the_weight_is_zero(self):
        result = self._epoch(0.0)
        self.assertNotIn("jpsi_z_cycle_loss", result)
        self.assertNotIn("z_z_cycle_loss", result)

    def test_term_is_reported_and_finite_when_weighted(self):
        result = self._epoch(0.5)
        self.assertIn("jpsi_z_cycle_loss", result)
        self.assertIn("z_z_cycle_loss", result)
        self.assertTrue(np.isfinite(result["jpsi_z_cycle_loss"]))
        self.assertTrue(np.isfinite(result["z_z_cycle_loss"]))
        self.assertGreater(result["jpsi_z_cycle_loss"], 0.0)
        self.assertTrue(np.isfinite(result["loss"]))


class ShippedD3ConfigTests(unittest.TestCase):
    def test_config_changes_only_the_new_term(self):
        frozen = resolve_joint_config(load_config(FROZEN_CONFIG))
        d3 = resolve_joint_config(load_config(D3_CONFIG))
        self.assertEqual(d3["run_name"], "Run_H_D3zcycle")
        self.assertEqual(d3["comparison"]["baseline_run"], "Run_H_A2frozen")
        self.assertEqual(d3["z_cycle_noise"], "native")
        for key in (
            "model",
            "regions",
            "loaders",
            "cycle_decoder_noise",
            "mean_map_anchor",
            "prior_components",
            "paths",
        ):
            self.assertEqual(d3.get(key), frozen.get(key), f"{key} changed unexpectedly")
        self.assertEqual(
            [stage["name"] for stage in d3["stages"]],
            [stage["name"] for stage in frozen["stages"]],
        )
        for d3_stage, frozen_stage in zip(d3["stages"], frozen["stages"]):
            self.assertNotIn("zeta", frozen_stage)
            for key in (
                "beta",
                "lamb",
                "tau",
                "nu_e",
                "nu_d",
                "epochs",
                "lr",
                "num_slices",
                "core_noise_multiplier",
                "tail_noise_multiplier",
            ):
                self.assertEqual(
                    d3_stage.get(key),
                    frozen_stage.get(key),
                    f"{key} changed in {d3_stage[chr(39) + chr(39)] if False else d3_stage['name']}",
                )

    def test_shipped_weights_are_pinned(self):
        d3 = resolve_joint_config(load_config(D3_CONFIG))
        zetas = {stage["name"]: float(stage.get("zeta", 0.0)) for stage in d3["stages"]}
        self.assertEqual(zetas["runH_stage1_deterministic_warmup"], 0.0)
        self.assertEqual(zetas["runH_stage2_stochastic_core"], 0.5)
        self.assertEqual(zetas["runH_stage3_stochastic_tail"], 0.5)


if __name__ == "__main__":
    unittest.main()
