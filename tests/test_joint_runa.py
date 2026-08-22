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
from joint_data import (  # noqa: E402
    build_joint_split_manifest,
    region_data_config,
    resolve_joint_config,
)
from joint_metrics import score_joint_metrics  # noqa: E402
from joint_model import build_joint_autoencoder, resolve_condition_indices  # noqa: E402
from joint_trainer import _coverage_indices, train_joint_epoch  # noqa: E402
from loss import CmsJpsiDoubleMuonLossFactory  # noqa: E402


MUON_MASS = 0.1056583755
RUN_A_CONFIG = REPO_ROOT / "configs_joint" / "cms_Joint_runA.yaml"
RUN_B_CONFIG = REPO_ROOT / "configs_joint" / "cms_Joint_runB.yaml"
RUN_C_CONFIG = REPO_ROOT / "configs_joint" / "cms_Joint_runC.yaml"
RUN_C_FULL_CONFIG = REPO_ROOT / "configs_joint" / "cms_Joint_runC_fullScale.yaml"


def _pairs(n: int, mass: float, seed: int) -> np.ndarray:
    """Small on-shell dimuon sample with varied direction and pair recoil."""
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
    return np.stack([px1, py1, pz1, e1, px2, py2, pz2, e2], axis=1).astype(
        np.float32
    )


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


def test_run_a_config_is_two_region_mass_blind_contract():
    config = resolve_joint_config(load_config(RUN_A_CONFIG))
    assert config["run_label"] == "Run A"
    assert config["region_order"] == ["jpsi", "z"]
    assert all(config["regions"][name]["data"]["channel"] == "muon" for name in config["region_order"])
    assert 8 not in resolve_condition_indices(config["model"]["condition_features"])
    assert config["holdout"]["resonance"] == "Upsilon"
    assert config["holdout"]["status"] == "locked_zero_shot"


def test_run_b_changes_only_the_z_prior_and_run_identity():
    run_a = resolve_joint_config(load_config(RUN_A_CONFIG))
    run_b = resolve_joint_config(load_config(RUN_B_CONFIG))

    assert run_b["run_label"] == "Run B"
    assert run_b["run_name"] == "Run_B"
    assert run_b["comparison"] == {
        "baseline_run": "Run_A",
        "controlled_change": "dedicated_dymumu_z_prior",
    }
    assert Path(run_b["regions"]["z"]["paths"]["theory_prior_file"]).name == (
        "cms_dymumu_mg5_8tev_dy1j_ptj5_fiducial_70_110_1M.hdf5"
    )
    assert run_b["regions"]["jpsi"] == run_a["regions"]["jpsi"]
    assert run_b["regions"]["z"]["muon_selection"] == run_a["regions"]["z"]["muon_selection"]
    assert run_b["regions"]["z"]["theory_prior_selection"] == run_a["regions"]["z"]["theory_prior_selection"]
    assert run_b["model"] == run_a["model"]
    assert run_b["loss"] == run_a["loss"]
    assert run_b["stages"] == run_a["stages"]
    assert run_b["holdout"]["status"] == "locked_zero_shot"


def test_run_c_contract_has_requested_depth_duration_batch_and_selection():
    run_b = resolve_joint_config(load_config(RUN_B_CONFIG))
    run_c = resolve_joint_config(load_config(RUN_C_CONFIG))

    assert run_c["run_label"] == "Run C"
    assert run_c["run_name"] == "Run_C"
    assert run_c["checkpoint_selection"]["region_reduction"] == "rms"
    assert run_c["model"]["hidden_dims"] == [512] * 6
    assert run_c["loaders"]["train_batch_size"] == 24576
    assert run_c["loaders"]["eval_batch_size"] == 4096
    assert run_c["loaders"]["steps_per_epoch"] == 4
    assert run_c["final_evaluation"]["batch_size"] == 4096
    assert [stage["epochs"] for stage in run_c["stages"]] == [72, 120, 144, 96]
    assert all(
        current["epochs"] == round(1.2 * previous["epochs"])
        for previous, current in zip(run_b["stages"], run_c["stages"])
    )
    assert run_c["regions"] == run_b["regions"]
    assert run_c["region_weights"] == run_b["region_weights"]
    assert run_c["loss"] == run_b["loss"]
    assert run_c["holdout"]["status"] == "locked_zero_shot"


def test_rms_checkpoint_selection_balances_all_target_metrics():
    metrics = {
        "jpsi": {"metric_a": 1.0, "metric_b": 3.0, "base_loss": 0.0},
        "z": {"metric_a": 2.0, "metric_b": 2.0, "base_loss": 0.0},
    }
    config = {
        "region_order": ["jpsi", "z"],
        "regions": {
            name: {
                "selection": {
                    "targets": {"metric_a": 1.0, "metric_b": 1.0},
                    "gates": {"metric_a": 10.0, "metric_b": 10.0},
                }
            }
            for name in metrics
        },
        "checkpoint_selection": {
            "hard_gates": True,
            "base_loss_weight": 0.0,
            "region_reduction": "rms",
        },
    }
    report = score_joint_metrics(metrics, config)
    assert np.isclose(report["regions"]["jpsi"]["score"], np.sqrt(5.0))
    assert np.isclose(report["regions"]["z"]["score"], 2.0)
    assert report["worst_region"] == "jpsi"
    assert np.isclose(report["selection_score"], np.sqrt(5.0))


def test_run_c_fullscale_is_uncapped_and_preserves_run_c_model_contract():
    run_c = resolve_joint_config(load_config(RUN_C_CONFIG))
    full = resolve_joint_config(load_config(RUN_C_FULL_CONFIG))

    assert full["run_label"] == "Run C full-scale"
    assert full["run_name"] == "Run_C_fullScale"
    assert full["loaders"]["sampling"] == "cycling_without_replacement"
    for name in full["region_order"]:
        assert full["regions"][name]["data_split"]["train_max"] is None
        assert full["regions"][name]["data_split"]["val_max"] is None
        assert full["regions"][name]["data_split"]["test_max"] is None
    assert full["model"] == run_c["model"]
    assert full["loss"] == run_c["loss"]
    assert full["stages"] == run_c["stages"]
    assert full["region_weights"] == run_c["region_weights"]


def test_fullscale_coverage_stream_visits_every_row_before_repeating():
    first = _coverage_indices(17, 0, 17, seed=1701, stream=0)
    resumed = np.concatenate(
        [
            _coverage_indices(17, 0, 8, seed=1701, stream=0),
            _coverage_indices(17, 8, 9, seed=1701, stream=0),
        ]
    )
    second = _coverage_indices(17, 17, 17, seed=1701, stream=0)

    assert sorted(first.tolist()) == list(range(17))
    assert np.array_equal(first, resumed)
    assert sorted(second.tolist()) == list(range(17))
    assert not np.array_equal(first, second)


def test_joint_model_is_shared_and_preserves_muon_mass_shell():
    arrays = _arrays()
    model = build_joint_autoencoder(
        _model_config(), arrays, MUON_MASS, [MUON_MASS, MUON_MASS]
    )
    model.set_noise_multipliers(0.0, 0.0)
    jpsi = torch.as_tensor(arrays["jpsi"]["x_train"][:8])
    z = torch.as_tensor(arrays["z"]["x_train"][:8])
    # Both scales use the exact same encoder object and parameter identities.
    before = id(next(model.encoder.parameters()))
    encoded_jpsi = model.encode(jpsi)
    encoded_z = model.encode(z)
    assert id(next(model.encoder.parameters())) == before
    assert encoded_jpsi.shape == encoded_z.shape == (8, 8)
    for output in (encoded_jpsi, encoded_z, model.decode(encoded_jpsi)):
        for start in (0, 4):
            p4 = output[:, start : start + 4]
            expected_energy = torch.sqrt(
                torch.sum(p4[:, :3] ** 2, dim=1) + MUON_MASS**2
            )
            assert torch.allclose(
                p4[:, 3], expected_energy, atol=1.0e-5, rtol=1.0e-6
            )


def test_worst_region_checkpoint_score_cannot_be_hidden_by_other_region():
    metrics = {
        "jpsi": {"direct_mass_ks": 0.01, "base_loss": 0.0},
        "z": {"direct_mass_ks": 0.09, "base_loss": 0.0},
    }
    config = {
        "region_order": ["jpsi", "z"],
        "regions": {
            "jpsi": {"selection": {"targets": {"direct_mass_ks": 0.02}, "gates": {}}},
            "z": {"selection": {"targets": {"direct_mass_ks": 0.03}, "gates": {}}},
        },
        "checkpoint_selection": {"hard_gates": True, "base_loss_weight": 0.0},
    }
    report = score_joint_metrics(metrics, config)
    assert report["worst_region"] == "z"
    assert report["selection_score"] == 3.0


def test_joint_contract_hash_is_independent_of_cache_hit_state():
    config = resolve_joint_config(load_config(RUN_A_CONFIG))
    arrays = {}
    for offset, name in enumerate(config["region_order"]):
        mass = 3.0969 if name == "jpsi" else 91.1876
        values = _pairs(6, mass, 20 + offset)
        arrays[name] = {
            "x_train": values[:2],
            "x_val": values[2:4],
            "x_test": values[4:],
            "z_train": values[:2].copy(),
            "z_val": values[2:4].copy(),
            "z_test": values[4:].copy(),
        }
    region_configs = {
        name: region_data_config(config, name) for name in config["region_order"]
    }
    miss = {name: {"hit": False, "status": "written"} for name in config["region_order"]}
    hit = {name: {"hit": True, "status": "hit"} for name in config["region_order"]}
    first = build_joint_split_manifest(
        config, arrays, miss, region_configs, num_samples=6
    )
    second = build_joint_split_manifest(
        config, arrays, hit, region_configs, num_samples=6
    )
    assert first["contract_sha256"] == second["contract_sha256"]


def test_one_joint_step_updates_shared_parameters():
    arrays = _arrays()
    model = build_joint_autoencoder(
        _model_config(), arrays, MUON_MASS, [MUON_MASS, MUON_MASS]
    )
    loss_config = {
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
        "beta": 1.0,
        "lamb": 1.0,
        "tau": 1.0,
        "nu_e": 0.0,
        "nu_d": 0.0,
        "gradient_clip_norm": 1.0,
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
        seed=99,
    )
    assert np.isfinite(result["loss"])
    assert not torch.equal(original, next(model.parameters()).detach())


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for function in (
        test_run_a_config_is_two_region_mass_blind_contract,
        test_run_b_changes_only_the_z_prior_and_run_identity,
        test_run_c_contract_has_requested_depth_duration_batch_and_selection,
        test_rms_checkpoint_selection_balances_all_target_metrics,
        test_run_c_fullscale_is_uncapped_and_preserves_run_c_model_contract,
        test_fullscale_coverage_stream_visits_every_row_before_repeating,
        test_joint_model_is_shared_and_preserves_muon_mass_shell,
        test_worst_region_checkpoint_score_cannot_be_hidden_by_other_region,
        test_joint_contract_hash_is_independent_of_cache_hit_state,
        test_one_joint_step_updates_shared_parameters,
    ):
        suite.addTest(unittest.FunctionTestCase(function))
    return suite


if __name__ == "__main__":
    unittest.main()
