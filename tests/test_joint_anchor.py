"""Contract tests for the frozen mean-map anchor (Run H drift control)."""

from __future__ import annotations

import copy
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
from joint_anchor import MeanMapAnchor  # noqa: E402
from joint_data import resolve_joint_config  # noqa: E402
from joint_metrics import current_noise_multipliers, set_noise_multipliers  # noqa: E402
from joint_model import build_joint_autoencoder  # noqa: E402
from joint_trainer import train_joint_epoch  # noqa: E402
from loss import CmsJpsiDoubleMuonLossFactory  # noqa: E402
from run_joint import assert_mean_map_anchor_contract  # noqa: E402


MUON_MASS = 0.1056583755
RUN_H_CONFIG = REPO_ROOT / "configs_joint" / "cms_Joint_runH.yaml"
RUN_H_ANCHOR_CONFIG = REPO_ROOT / "configs_joint" / "cms_Joint_runH_anchor.yaml"
RUN_H_FIX_CONFIG = REPO_ROOT / "configs_joint" / "cms_Joint_runH_fix.yaml"


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


def _build_model_anchor():
    arrays = _arrays()
    model = build_joint_autoencoder(
        _model_config(), arrays, MUON_MASS, [MUON_MASS, MUON_MASS]
    )
    anchor = MeanMapAnchor(events_per_region=8)
    anchor.build_inputs(arrays, ["jpsi", "z"], torch.device("cpu"))
    anchor.capture(model, "stage1")
    return arrays, model, anchor


def test_mean_map_anchor_contract_accepts_the_run_h_arms():
    clean = resolve_joint_config(load_config(RUN_H_CONFIG))
    assert_mean_map_anchor_contract(clean)  # Run H itself has no anchor

    for path in (RUN_H_ANCHOR_CONFIG, RUN_H_FIX_CONFIG):
        config = resolve_joint_config(load_config(path))
        assert config["mean_map_anchor"]["enabled"] is True
        assert (
            config["mean_map_anchor"]["reference_stage"]
            == "runH_stage1_deterministic_warmup"
        )
        anchored = [
            stage
            for stage in config["stages"]
            if float(stage.get("mean_map_anchor_weight", 0.0) or 0.0) > 0.0
        ]
        assert len(anchored) == 2, path
        # The encoder noise is deliberately kept: the model stays stochastic
        # in both directions while the mean map is anchored.
        for stage in anchored:
            for component in ("encoder", "decoder"):
                core = stage.get(
                    f"{component}_core_noise_multiplier",
                    stage.get("core_noise_multiplier", 1.0),
                )
                tail = stage.get(
                    f"{component}_tail_noise_multiplier",
                    stage.get("tail_noise_multiplier", 0.0),
                )
                core = float(core.get("end", core.get("start")) if isinstance(core, dict) else core)
                tail = float(tail.get("end", tail.get("start")) if isinstance(tail, dict) else tail)
                assert core > 0.0 or tail > 0.0, (path, stage["name"], component)
        assert_mean_map_anchor_contract(config, path.name)


def test_mean_map_anchor_contract_rejects_violations():
    base = resolve_joint_config(load_config(RUN_H_ANCHOR_CONFIG))
    assert_mean_map_anchor_contract(base)  # clean declaration is accepted

    undeclared = copy.deepcopy(base)
    undeclared["mean_map_anchor"]["enabled"] = False
    with unittest.TestCase().assertRaises(RuntimeError):
        assert_mean_map_anchor_contract(undeclared)

    bad_reference = copy.deepcopy(base)
    bad_reference["mean_map_anchor"]["reference_stage"] = "not_a_stage"
    with unittest.TestCase().assertRaises(RuntimeError):
        assert_mean_map_anchor_contract(bad_reference)

    too_few = copy.deepcopy(base)
    too_few["mean_map_anchor"]["events_per_region"] = 1
    with unittest.TestCase().assertRaises(RuntimeError):
        assert_mean_map_anchor_contract(too_few)

    # The anchor may not be used to switch either direction's noise off.
    for component in ("encoder", "decoder"):
        frozen = copy.deepcopy(base)
        for stage in frozen["stages"]:
            if float(stage.get("mean_map_anchor_weight", 0.0) or 0.0) > 0.0:
                stage[f"{component}_core_noise_multiplier"] = 0.0
                stage[f"{component}_tail_noise_multiplier"] = 0.0
        with unittest.TestCase().assertRaises(RuntimeError):
            assert_mean_map_anchor_contract(frozen)


def test_mean_map_anchor_is_zero_at_reference_and_grows_with_drift():
    _arrays_, model, anchor = _build_model_anchor()
    model.train()
    loss, terms = anchor.loss(model)
    assert float(loss.detach()) < 1e-8  # identical map -> exact zero
    assert all(abs(value) < 1e-8 for value in terms.values())

    with torch.no_grad():
        model.decoder.steps[0].head.bias[0] += 0.05
    loss, terms = anchor.loss(model)
    assert float(loss.detach()) > 0.0
    loss.backward()
    gradient = model.decoder.steps[0].head.bias.grad
    assert gradient is not None and float(gradient.abs().sum()) > 0.0


def test_mean_map_anchor_does_not_consume_or_zero_the_noise_schedule():
    _arrays_, model, anchor = _build_model_anchor()
    schedule = {
        "encoder_core": 1.0,
        "encoder_tail": 0.5,
        "decoder_core": 0.25,
        "decoder_tail": 0.125,
    }
    set_noise_multipliers(model, schedule)
    anchor.loss(model)
    assert current_noise_multipliers(model) == schedule
    # And the encoder noise is never silently zeroed by the anchor.
    assert current_noise_multipliers(model)["encoder_core"] > 0.0


def test_train_joint_epoch_reports_the_mean_map_anchor():
    arrays, model, anchor = _build_model_anchor()
    loss_config = _loss_config()
    factories = {
        name: CmsJpsiDoubleMuonLossFactory(
            region["x_train"], region["z_train"], loss_config, [MUON_MASS, MUON_MASS]
        )
        for name, region in arrays.items()
    }
    config = {
        "region_order": ["jpsi", "z"],
        "region_weights": {"jpsi": 1.0, "z": 1.0},
        "loaders": {"train_batch_size": 8, "steps_per_epoch": 1},
    }
    stage = {
        "name": "synthetic_anchor_stage",
        "beta": 1.0,
        "lamb": 1.0,
        "tau": 1.0,
        "nu_e": 0.0,
        "nu_d": 0.0,
        "gradient_clip_norm": 1.0,
        "mean_map_anchor_weight": 0.5,
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
    original = next(model.parameters()).detach().clone()
    result = train_joint_epoch(
        model,
        optimizer,
        arrays,
        factories,
        config,
        stage,
        device=torch.device("cpu"),
        seed=123,
        mean_map_anchor=anchor,
    )
    assert np.isfinite(result["loss"])
    assert "mean_map_anchor" in result
    assert np.isfinite(result["mean_map_anchor"])
    assert not torch.equal(original, next(model.parameters()).detach())

    # Without a weight the term is absent, so every earlier config is unchanged.
    model2 = build_joint_autoencoder(
        _model_config(), arrays, MUON_MASS, [MUON_MASS, MUON_MASS]
    )
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1.0e-4)
    no_anchor_stage = dict(stage)
    no_anchor_stage["mean_map_anchor_weight"] = 0.0
    result2 = train_joint_epoch(
        model2,
        optimizer2,
        arrays,
        factories,
        config,
        no_anchor_stage,
        device=torch.device("cpu"),
        seed=123,
        mean_map_anchor=anchor,
    )
    assert "mean_map_anchor" not in result2


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for function in (
        test_mean_map_anchor_contract_accepts_the_run_h_arms,
        test_mean_map_anchor_contract_rejects_violations,
        test_mean_map_anchor_is_zero_at_reference_and_grows_with_drift,
        test_mean_map_anchor_does_not_consume_or_zero_the_noise_schedule,
        test_train_joint_epoch_reports_the_mean_map_anchor,
    ):
        suite.addTest(unittest.FunctionTestCase(function))
    return suite


if __name__ == "__main__":
    unittest.main()
