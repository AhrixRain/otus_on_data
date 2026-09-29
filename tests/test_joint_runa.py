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
from joint_metrics import (  # noqa: E402
    current_noise_multipliers,
    evaluate_region,
    resolve_noise_multipliers,
    score_joint_metrics,
    set_noise_multipliers,
)
from joint_model import build_joint_autoencoder, resolve_condition_indices  # noqa: E402
from joint_trainer import (  # noqa: E402
    _coverage_indices,
    _full_pass_batches,
    _full_pass_step_count,
    _stage_last_path,
    train_joint_epoch,
    validate_joint,
)
from joint_train_utils import _expand_indices, _scheduled_value  # noqa: E402
from loss import CmsJpsiDoubleMuonLossFactory  # noqa: E402
from run_joint import assert_mass_not_conditioned, assert_no_mass_anchor  # noqa: E402


MUON_MASS = 0.1056583755
RUN_A_CONFIG = REPO_ROOT / "configs_joint" / "cms_Joint_runA.yaml"
RUN_B_CONFIG = REPO_ROOT / "configs_joint" / "cms_Joint_runB.yaml"
RUN_C_CONFIG = REPO_ROOT / "configs_joint" / "cms_Joint_runC.yaml"
RUN_C_FULL_CONFIG = REPO_ROOT / "configs_joint" / "cms_Joint_runC_fullScale.yaml"
RUN_D_CONFIG = REPO_ROOT / "configs_joint" / "cms_Joint_runD.yaml"
RUN_E_CONFIG = REPO_ROOT / "configs_joint" / "cms_Joint_runE.yaml"
CONST_NOISE_CONFIG = REPO_ROOT / "configs_joint" / "cms_Joint_runE_constNoise.yaml"
RUN_F_CONFIG = REPO_ROOT / "configs_joint" / "cms_Joint_runF.yaml"
RUN_G_CONFIG = REPO_ROOT / "configs_joint" / "cms_Joint_runG.yaml"
RUN_H_CONFIG = REPO_ROOT / "configs_joint" / "cms_Joint_runH.yaml"
RUN_F_PRIOR_CONFIG = REPO_ROOT / "configs_joint" / "cms_Joint_runF_priorCKKWL.yaml"
ENCODER_DET_CONFIG = REPO_ROOT / "configs_joint" / "cms_Joint_runE_encoderDet.yaml"
ENCODER_DET_SHORT_CONFIG = (
    REPO_ROOT / "configs_joint" / "cms_Joint_runE_encoderDetShort.yaml"
)
DETERMINISTIC_NOISE = {
    "encoder_core": 0.0,
    "encoder_tail": 0.0,
    "decoder_core": 0.0,
    "decoder_tail": 0.0,
}


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


def _arrays_with_val() -> dict[str, dict[str, np.ndarray]]:
    return {
        "jpsi": {
            "x_train": _pairs(32, 3.0969, 1),
            "z_train": _pairs(32, 3.0969, 2),
            "x_val": _pairs(24, 3.0969, 11),
            "z_val": _pairs(24, 3.0969, 12),
        },
        "z": {
            "x_train": _pairs(32, 91.1876, 3),
            "z_train": _pairs(32, 91.1876, 4),
            "x_val": _pairs(24, 91.1876, 13),
            "z_val": _pairs(24, 91.1876, 14),
        },
    }


def _validation_loss_config() -> dict:
    return {
        "kind": "cms_jpsi_doublemuon_loss",
        "num_slices": 4,
        "p": 1,
        "raw_swd": 0.1,
        "marginal_w1": 0.1,
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


def _validation_config() -> dict:
    return {
        "model": _model_config(),
        "region_order": ["jpsi", "z"],
        "loaders": {
            "validation_events": 16,
            "validation_draws": 1,
            "eval_batch_size": 16,
        },
        "checkpoint_selection": {"base_loss_weight": 0.0, "region_reduction": "max"},
        "regions": {
            name: {
                "selection": {
                    "targets": {
                        "cycle_mass_ks": 1.0,
                        "latent_mass_ks": 1.0,
                        "direct_mass_ks": 1.0,
                    }
                }
            }
            for name in ("jpsi", "z")
        },
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


def test_run_d_changes_only_z_prior_identity_and_holdout_claim():
    full = resolve_joint_config(load_config(RUN_C_FULL_CONFIG))
    run_d = resolve_joint_config(load_config(RUN_D_CONFIG))

    assert run_d["run_label"] == "Run D"
    assert run_d["run_name"] == "Run_D"
    assert run_d["comparison"] == {
        "baseline_run": "Run_C_fullScale",
        "controlled_change": "showered_inclusive_ckkwl_0j1j_z_prior",
    }
    assert Path(run_d["regions"]["z"]["paths"]["theory_prior_file"]).name == (
        "cms_dymumu_mg5py8_ckkwl_8tev_inclusive_0j1j_fiducial_70_110_1M.hdf5"
    )
    assert run_d["regions"]["jpsi"] == full["regions"]["jpsi"]
    run_d_z = dict(run_d["regions"]["z"])
    full_z = dict(full["regions"]["z"])
    run_d_z["paths"] = dict(run_d_z["paths"])
    full_z["paths"] = dict(full_z["paths"])
    run_d_z["paths"].pop("theory_prior_file")
    full_z["paths"].pop("theory_prior_file")
    assert run_d_z == full_z
    for key in ("model", "loss", "stages", "loaders", "data_split", "region_weights"):
        if key in full:
            assert run_d[key] == full[key]
    assert run_d["holdout"]["resonance"] == "Upsilon"
    assert run_d["holdout"]["status"] == "excluded_post_unblinding"


def test_run_e_changes_only_epoch_definition():
    run_d = resolve_joint_config(load_config(RUN_D_CONFIG))
    run_e = resolve_joint_config(load_config(RUN_E_CONFIG))

    assert run_e["run_label"] == "Run E"
    assert run_e["run_name"] == "Run_E"
    assert run_e["comparison"] == {
        "baseline_run": "Run_D",
        "controlled_change": "full_training_partitions_once_per_epoch",
    }
    assert run_e["loaders"]["sampling"] == "cycling_without_replacement"
    assert run_e["loaders"]["epoch_definition"] == "full_pass"
    assert run_e["loaders"]["steps_per_epoch"] is None
    for key in ("model", "loss", "stages", "regions", "region_weights"):
        assert run_e[key] == run_d[key]
    assert run_e["holdout"]["status"] == "excluded_post_unblinding"


def test_run_e_constant_noise_arm_shares_one_schedule_for_both_maps():
    """Meeting decision 2026-09-09: stages 2-4 run one shared constant schedule.

    The arm keeps the stage-1 deterministic warmup and removes every per-epoch
    noise ramp. Loss coefficients are inherited unchanged (they were already
    scalars), so the only controlled variable is the noise schedule.
    """
    run_e = resolve_joint_config(load_config(RUN_E_CONFIG))
    arm = resolve_joint_config(load_config(CONST_NOISE_CONFIG))

    assert arm["run_name"] == "Run_E_constNoise"
    for key in ("model", "loss", "regions", "region_weights", "loaders"):
        assert arm[key] == run_e[key]

    stages = {stage["name"]: stage for stage in arm["stages"]}
    reference_stages = {stage["name"]: stage for stage in run_e["stages"]}

    # Stage 1 stays the deterministic warm start.
    warm = stages["runA_stage1_deterministic_identity"]
    for epoch in (1, warm["epochs"]):
        assert _scheduled_value(warm["core_noise_multiplier"], epoch, warm["epochs"]) == 0.0
        assert _scheduled_value(warm["tail_noise_multiplier"], epoch, warm["epochs"]) == 0.0

    stochastic = (
        "runA_stage2_gaussian_response",
        "runA_stage3_stochastic_joint",
        "runA_stage4_tail_polish",
    )
    for name in stochastic:
        stage = stages[name]
        # No per-component override: the shared keys are the only assignment.
        for component in ("encoder", "decoder"):
            for kind in ("core", "tail"):
                assert f"{component}_{kind}_noise_multiplier" not in stage
        # Plain scalars, not {start, end, schedule} ramps.
        assert stage["core_noise_multiplier"] == 1.0
        assert stage["tail_noise_multiplier"] == 0.25
        # Loss coefficients unchanged relative to Run E.
        for key in ("beta", "lamb", "tau", "nu_e", "nu_d", "rho"):
            assert stage.get(key) == reference_stages[name].get(key)

    # Reproduce the trainer's resolution for every epoch: encoder and decoder
    # resolve to the same multiplier, and it does not drift within a stage.
    for name in stochastic:
        stage = stages[name]
        epochs = int(stage["epochs"])
        resolved = set()
        for epoch in range(1, epochs + 1):
            core = _scheduled_value(stage["core_noise_multiplier"], epoch, epochs)
            tail = _scheduled_value(stage["tail_noise_multiplier"], epoch, epochs)
            encoder_core = encoder_tail = decoder_core = decoder_tail = None
            for component, shared in (
                ("encoder", (core, tail)),
                ("decoder", (core, tail)),
            ):
                component_core = (
                    _scheduled_value(
                        stage[f"{component}_core_noise_multiplier"], epoch, epochs
                    )
                    if f"{component}_core_noise_multiplier" in stage
                    else shared[0]
                )
                component_tail = (
                    _scheduled_value(
                        stage[f"{component}_tail_noise_multiplier"], epoch, epochs
                    )
                    if f"{component}_tail_noise_multiplier" in stage
                    else shared[1]
                )
                if component == "encoder":
                    encoder_core, encoder_tail = component_core, component_tail
                else:
                    decoder_core, decoder_tail = component_core, component_tail
            assert encoder_core == decoder_core == core
            assert encoder_tail == decoder_tail == tail
            resolved.add((encoder_core, encoder_tail))
        assert resolved == {(1.0, 0.25)}


def test_load_config_replace_keys_replaces_named_stage_list():
    run_e = resolve_joint_config(load_config(RUN_E_CONFIG))
    run_f = resolve_joint_config(load_config(RUN_F_CONFIG))
    assert [s["name"] for s in run_e["stages"]] == [
        "runA_stage1_deterministic_identity",
        "runA_stage2_gaussian_response",
        "runA_stage3_stochastic_joint",
        "runA_stage4_tail_polish",
    ]
    # replace_keys must replace, not append, the inherited named-stage list.
    assert [s["name"] for s in run_f["stages"]] == [
        "runF_stage1_deterministic_warmup",
        "runF_stage2_stochastic_constant",
    ]


def test_run_f_flat_loss_mechanism_contract():
    run_e = resolve_joint_config(load_config(RUN_E_CONFIG))
    run_f = resolve_joint_config(load_config(RUN_F_CONFIG))

    assert run_f["run_name"] == "Run_F"
    assert run_f["model"]["hidden_dims"] == [256, 256, 256, 256]
    assert run_f["model"]["hidden_dims"] != run_e["model"]["hidden_dims"]
    assert run_f["loaders"]["min_events_per_update"] == 4096
    assert [stage["epochs"] for stage in run_f["stages"]] == [20, 80]

    # The objective is identical in both stages: only the noise level changes.
    for stage in run_f["stages"]:
        assert (stage["beta"], stage["lamb"], stage["tau"]) == (1, 1, 1)
        assert (stage["nu_e"], stage["nu_d"]) == (0, 0)
        assert stage["num_slices"] == 256
        for component in ("encoder", "decoder"):
            for kind in ("core", "tail"):
                assert f"{component}_{kind}_noise_multiplier" not in stage
    warmup, stochastic = run_f["stages"]
    for key in ("beta", "lamb", "tau", "nu_e", "nu_d", "num_slices"):
        assert warmup[key] == stochastic[key]
    assert (warmup["core_noise_multiplier"], warmup["tail_noise_multiplier"]) == (0, 0)
    assert (
        stochastic["core_noise_multiplier"],
        stochastic["tail_noise_multiplier"],
    ) == (1.0, 0.25)
    assert stochastic["lr_schedule"] == "none"

    # Data, priors and region definition are the Run E contract.
    assert run_f["regions"] == run_e["regions"]
    assert run_f["region_weights"] == run_e["region_weights"]


def test_run_f_prior_contract_is_preserved_and_disjoint():
    config = resolve_joint_config(load_config(RUN_F_PRIOR_CONFIG))
    assert config["run_name"] == "Run_F_priorCKKWL"
    assert len(config["stages"]) == 4
    assert config["model"]["hidden_dims"] == [512] * 6
    assert config["loaders"].get("min_events_per_update") is None


def test_expand_indices_pads_to_floor_and_is_reproducible():
    base = np.array([5, 1, 3], dtype=np.int64)
    padded = _expand_indices(base, 20, 10, np.random.default_rng(7))
    assert len(padded) == 10
    assert padded[:3].tolist() == [5, 1, 3]
    assert padded.min() >= 0 and padded.max() < 20
    # min_events <= 0 is the back-compatible no-op.
    assert _expand_indices(base, 20, 0, np.random.default_rng(7)) is base
    # Deterministic for a fixed seed.
    first = _expand_indices(base, 20, 10, np.random.default_rng(7))
    second = _expand_indices(base, 20, 10, np.random.default_rng(7))
    assert first.tolist() == second.tolist()


@unittest.skip(
    "cms_Joint_runE_encoderDet*.yaml retired 2026-09-09: the meeting kept one "
    "shared encoder/decoder noise schedule (cms_Joint_runE_constNoise.yaml)."
)
def test_run_e_encoder_det_arm_zeroes_encoder_noise_only():
    run_e = resolve_joint_config(load_config(RUN_E_CONFIG))
    arm = resolve_joint_config(load_config(ENCODER_DET_CONFIG))

    assert arm["run_name"] == "Run_E_encoderDet"
    assert arm["loaders"]["validation_noise_multipliers"] == DETERMINISTIC_NOISE
    assert arm["final_evaluation"]["noise_multipliers"] == DETERMINISTIC_NOISE
    stages = {stage["name"]: stage for stage in arm["stages"]}
    assert stages["runA_stage1_deterministic_identity"]["enabled"] is False
    for name in (
        "runA_stage2_gaussian_response",
        "runA_stage3_stochastic_joint",
        "runA_stage4_tail_polish",
    ):
        assert stages[name]["encoder_core_noise_multiplier"] == 0.0
        assert stages[name]["encoder_tail_noise_multiplier"] == 0.0
    # The decoder schedule is inherited unchanged.
    assert stages["runA_stage2_gaussian_response"]["core_noise_multiplier"] == {
        "start": 0.1,
        "end": 1,
        "schedule": "cosine",
    }
    assert stages["runA_stage3_stochastic_joint"]["core_noise_multiplier"] == 1
    assert stages["runA_stage3_stochastic_joint"]["tail_noise_multiplier"] == {
        "start": 0,
        "end": 0.25,
        "schedule": "linear",
    }
    assert stages["runA_stage4_tail_polish"]["tail_noise_multiplier"] == {
        "start": 0.25,
        "end": 1,
        "schedule": "cosine",
    }
    # Only the noise assignment changed; the scientific contract is inherited.
    for key in ("model", "loss", "regions", "region_weights"):
        assert arm[key] == run_e[key]


@unittest.skip(
    "cms_Joint_runE_encoderDet*.yaml retired 2026-09-09: the meeting kept one "
    "shared encoder/decoder noise schedule (cms_Joint_runE_constNoise.yaml)."
)
def test_run_e_encoder_det_short_schedule_and_projection_budget():
    arm = resolve_joint_config(load_config(ENCODER_DET_CONFIG))
    short = resolve_joint_config(load_config(ENCODER_DET_SHORT_CONFIG))

    assert short["run_name"] == "Run_E_encoderDet_short"
    assert [stage["epochs"] for stage in short["stages"]] == [60, 100, 120, 80]
    assert sum(stage["epochs"] for stage in short["stages"]) == 360
    # Stage 1 is the warm-start source, so 300 epochs are actually trained.
    assert (
        sum(
            stage["epochs"]
            for stage in short["stages"]
            if stage.get("enabled", True)
        )
        == 300
    )
    assert {stage["num_slices"] for stage in short["stages"]} == {128}
    stages = {stage["name"]: stage for stage in short["stages"]}
    assert stages["runA_stage1_deterministic_identity"]["enabled"] is False
    for name in (
        "runA_stage2_gaussian_response",
        "runA_stage3_stochastic_joint",
        "runA_stage4_tail_polish",
    ):
        assert stages[name]["encoder_core_noise_multiplier"] == 0.0
        assert stages[name]["encoder_tail_noise_multiplier"] == 0.0
    # Decoder schedule inherited unchanged from the full arm.
    assert stages["runA_stage3_stochastic_joint"]["core_noise_multiplier"] == 1
    assert stages["runA_stage4_tail_polish"]["tail_noise_multiplier"] == {
        "start": 0.25,
        "end": 1,
        "schedule": "cosine",
    }
    # Only the epoch/projection budget changed relative to the full arm.
    for key in ("model", "regions", "region_weights", "loaders"):
        assert short[key] == arm[key]


def test_resolve_noise_multipliers_defaults_and_overrides():
    assert resolve_noise_multipliers(None) == DETERMINISTIC_NOISE
    resolved = resolve_noise_multipliers(
        {"core": 1.0, "tail": 0.5, "encoder_core": 0.0}
    )
    assert resolved == {
        "encoder_core": 0.0,
        "encoder_tail": 0.5,
        "decoder_core": 1.0,
        "decoder_tail": 0.5,
    }
    with np.testing.assert_raises(ValueError):
        resolve_noise_multipliers({"decoder_core": -1.0})
    with np.testing.assert_raises(ValueError):
        resolve_noise_multipliers({"core": float("nan")})


def test_evaluate_region_scores_deterministically_and_restores_model_noise():
    arrays = _arrays_with_val()
    model = build_joint_autoencoder(
        _model_config(), arrays, MUON_MASS, [MUON_MASS, MUON_MASS]
    )
    factory = CmsJpsiDoubleMuonLossFactory(
        arrays["jpsi"]["x_train"],
        arrays["jpsi"]["z_train"],
        _validation_loss_config(),
        [MUON_MASS, MUON_MASS],
    )
    kwargs = {
        "daughter_masses": [MUON_MASS, MUON_MASS],
        "device": torch.device("cpu"),
        "batch_size": 16,
        "max_events": 16,
        "decoder_draws": 1,
        "seed": 11,
    }
    loud = {
        "encoder_core": 5.0,
        "encoder_tail": 5.0,
        "decoder_core": 5.0,
        "decoder_tail": 5.0,
    }
    set_noise_multipliers(model, loud)
    from_loud_state = evaluate_region(model, arrays["jpsi"], factory, **kwargs)
    assert current_noise_multipliers(model) == loud
    set_noise_multipliers(model, DETERMINISTIC_NOISE)
    from_clean_state = evaluate_region(model, arrays["jpsi"], factory, **kwargs)
    assert current_noise_multipliers(model) == DETERMINISTIC_NOISE
    # Validation ignores whatever noise the training stage left on the model.
    for key in from_clean_state:
        assert np.isclose(
            from_clean_state[key], from_loud_state[key], rtol=0.0, atol=1e-12
        ), key


def test_validate_joint_records_deterministic_validation_noise():
    arrays = _arrays_with_val()
    model = build_joint_autoencoder(
        _model_config(), arrays, MUON_MASS, [MUON_MASS, MUON_MASS]
    )
    factories = {
        name: CmsJpsiDoubleMuonLossFactory(
            region["x_train"],
            region["z_train"],
            _validation_loss_config(),
            [MUON_MASS, MUON_MASS],
        )
        for name, region in arrays.items()
    }
    loud = {
        "encoder_core": 3.0,
        "encoder_tail": 3.0,
        "decoder_core": 3.0,
        "decoder_tail": 3.0,
    }
    set_noise_multipliers(model, loud)
    _, selection = validate_joint(
        model,
        arrays,
        factories,
        _validation_config(),
        device=torch.device("cpu"),
        seed=5,
    )
    assert selection["validation_noise_multipliers"] == DETERMINISTIC_NOISE
    assert current_noise_multipliers(model) == loud


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


def test_full_pass_batches_visit_every_training_row_once():
    arrays = {
        "jpsi": {
            "x_train": np.zeros((23, 8), dtype=np.float32),
            "z_train": np.zeros((11, 8), dtype=np.float32),
        },
        "z": {
            "x_train": np.zeros((31, 8), dtype=np.float32),
            "z_train": np.zeros((17, 8), dtype=np.float32),
        },
    }
    steps = _full_pass_step_count(arrays, ["jpsi", "z"], batch_size=8)
    assert steps == 4

    first = _full_pass_batches(
        arrays["z"]["x_train"], steps, 1, seed=1701, stream=2
    )
    second = _full_pass_batches(
        arrays["z"]["x_train"], steps, 2, seed=1701, stream=2
    )
    first_flat = np.concatenate(first)
    second_flat = np.concatenate(second)
    assert sorted(first_flat.tolist()) == list(range(31))
    assert sorted(second_flat.tolist()) == list(range(31))
    assert len(np.unique(first_flat)) == 31
    assert max(len(batch) for batch in first) <= 8
    assert not np.array_equal(first_flat, second_flat)


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
    assert result["nonfinite_gradient_skips"] == 0.0
    assert result["optimizer_updates"] == 1.0


class _ZeroValueInfiniteGradientLoss(CmsJpsiDoubleMuonLossFactory):
    """Finite loss with an infinite derivative: the sqrt(0) hazard shape.

    ``sqrt(clamp(0, min=0))`` is 0.0 in the forward pass but its backward is
    infinite -- exactly what a decoded p4 whose float32 pair mass cancels to
    zero used to do inside the decoder's condition.
    """

    def z_prior_loss(self, z_true, z_encoded):
        return torch.sqrt(torch.clamp(z_encoded.sum() * 0.0, min=0.0))


def test_non_finite_gradient_skips_update_without_crashing():
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
        name: _ZeroValueInfiniteGradientLoss(
            region["x_train"], region["z_train"], loss_config, [MUON_MASS, MUON_MASS]
        )
        for name, region in arrays.items()
    }
    config = {
        "region_order": ["jpsi", "z"],
        "region_weights": {"jpsi": 1.0, "z": 1.0},
        "loaders": {"train_batch_size": 8, "steps_per_epoch": 3},
    }
    stage = {
        "name": "poisoned",
        "beta": 1.0,
        "lamb": 1.0,
        "tau": 1.0,
        "nu_e": 0.0,
        "nu_d": 0.0,
        "gradient_clip_norm": 1.0,
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
    original = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
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
    # Every update is dropped, counted, and no parameter is touched; the old
    # behaviour was a RuntimeError from clip_grad_norm_ that ended the run.
    assert result["nonfinite_gradient_skips"] == 3.0
    assert result["optimizer_updates"] == 0.0
    assert result["optimizer_steps_attempted"] == 3.0
    assert np.isfinite(result["loss"])
    for name, parameter in model.named_parameters():
        assert torch.equal(original[name], parameter.detach())


def test_full_pass_training_consumes_unequal_partitions_once():
    arrays = _arrays()
    arrays["jpsi"]["z_train"] = arrays["jpsi"]["z_train"][:12]
    arrays["z"]["x_train"] = arrays["z"]["x_train"][:27]
    arrays["z"]["z_train"] = arrays["z"]["z_train"][:16]
    model = build_joint_autoencoder(
        _model_config(), arrays, MUON_MASS, [MUON_MASS, MUON_MASS]
    )
    loss_config = {
        "kind": "cms_jpsi_doublemuon_loss",
        "num_slices": 4,
        "raw_swd": 0.1,
        "marginal_w1": 0.1,
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
            region["x_train"],
            region["z_train"],
            loss_config,
            [MUON_MASS, MUON_MASS],
        )
        for name, region in arrays.items()
    }
    config = {
        "seed": 1701,
        "region_order": ["jpsi", "z"],
        "region_weights": {"jpsi": 1.0, "z": 1.0},
        "loaders": {
            "train_batch_size": 8,
            "sampling": "cycling_without_replacement",
            "epoch_definition": "full_pass",
        },
    }
    stage = {
        "beta": 1.0,
        "lamb": 1.0,
        "tau": 1.0,
        "nu_e": 0.0,
        "nu_d": 0.0,
        "gradient_clip_norm": 1.0,
    }
    result = train_joint_epoch(
        model,
        torch.optim.Adam(model.parameters(), lr=1.0e-4),
        arrays,
        factories,
        config,
        stage,
        device=torch.device("cpu"),
        seed=99,
        epoch_index=1,
    )
    assert np.isfinite(result["loss"])
    assert result["optimizer_updates"] == 4
    assert result["jpsi_x_events"] == 32
    assert result["jpsi_z_events"] == 12
    assert result["z_x_events"] == 27
    assert result["z_z_events"] == 16
    # Back-compatible default: with no floor, the reported per-update batches
    # are exactly the natural full-pass splits.
    assert result["jpsi_x_batch_per_update"] == 8.0
    assert result["jpsi_z_batch_per_update"] == 3.0
    assert result["z_x_batch_per_update"] == 7.0
    assert result["z_z_batch_per_update"] == 4.0


def test_batch_floor_pads_small_streams_and_is_reported():
    arrays = _arrays()
    arrays["jpsi"]["z_train"] = arrays["jpsi"]["z_train"][:12]
    arrays["z"]["x_train"] = arrays["z"]["x_train"][:27]
    arrays["z"]["z_train"] = arrays["z"]["z_train"][:16]
    model = build_joint_autoencoder(
        _model_config(), arrays, MUON_MASS, [MUON_MASS, MUON_MASS]
    )
    loss_config = _validation_loss_config()
    factories = {
        name: CmsJpsiDoubleMuonLossFactory(
            region["x_train"],
            region["z_train"],
            loss_config,
            [MUON_MASS, MUON_MASS],
        )
        for name, region in arrays.items()
    }
    config = {
        "seed": 1701,
        "region_order": ["jpsi", "z"],
        "region_weights": {"jpsi": 1.0, "z": 1.0},
        "loaders": {
            "train_batch_size": 8,
            "sampling": "cycling_without_replacement",
            "epoch_definition": "full_pass",
            "min_events_per_update": 16,
        },
    }
    stage = {
        "beta": 1.0,
        "lamb": 1.0,
        "tau": 1.0,
        "nu_e": 0.0,
        "nu_d": 0.0,
        "gradient_clip_norm": 1.0,
    }
    result = train_joint_epoch(
        model,
        torch.optim.Adam(model.parameters(), lr=1.0e-4),
        arrays,
        factories,
        config,
        stage,
        device=torch.device("cpu"),
        seed=99,
        epoch_index=1,
    )
    assert np.isfinite(result["loss"])
    # Natural per-update batches are 8/3/7/4; the floor raises every one to 16.
    assert result["jpsi_x_batch_per_update"] == 16.0
    assert result["jpsi_z_batch_per_update"] == 16.0
    assert result["z_x_batch_per_update"] == 16.0
    assert result["z_z_batch_per_update"] == 16.0
    # Unique-event coverage per epoch is unchanged: the floor only pads draws.
    assert result["jpsi_z_events"] == 12
    assert result["z_x_events"] == 27


def test_stage_last_checkpoint_path_is_run_prefixed_per_stage():
    output_dir = REPO_ROOT / "outputs" / "cms_Joint" / "Run_F"
    stage = {"name": "runF_stage2_stochastic_constant"}
    path = _stage_last_path(output_dir, stage, {"run_name": "Run_F"})
    assert path.parent == output_dir
    assert path.name == "last_RunF_stage2_stochastic_constant.pt"
    stage1 = {"name": "runF_stage1_deterministic_warmup"}
    assert _stage_last_path(output_dir, stage1, {"run_name": "Run_F"}).name == (
        "last_RunF_stage1_deterministic_warmup.pt"
    )


def test_run_g_three_stage_contract():
    """Run G = Run F with the stage list and the hidden width changed."""
    run_f = resolve_joint_config(load_config(RUN_F_CONFIG))
    run_g = resolve_joint_config(load_config(RUN_G_CONFIG))

    assert run_g["run_name"] == "Run_G"
    # Run G halves the network width (4x128) and changes the stage list.
    assert run_g["model"]["hidden_dims"] == [128, 128, 128, 128]
    for key, value in run_f["model"].items():
        if key != "hidden_dims":
            assert run_g["model"][key] == value, key
    # Everything else is inherited from Run F unchanged.
    for key in ("loaders", "regions", "region_weights"):
        assert run_g[key] == run_f[key], key

    assert [stage["name"] for stage in run_g["stages"]] == [
        "runG_stage1_deterministic_warmup",
        "runG_stage2_stochastic_core",
        "runG_stage3_stochastic_tail",
    ]
    assert [stage["epochs"] for stage in run_g["stages"]] == [20, 60, 60]
    warmup, core, tail = run_g["stages"]
    assert (warmup["core_noise_multiplier"], warmup["tail_noise_multiplier"]) == (0, 0)
    assert (core["core_noise_multiplier"], core["tail_noise_multiplier"]) == (1.0, 0.25)
    assert (tail["core_noise_multiplier"], tail["tail_noise_multiplier"]) == (1.0, 0.5)

    for stage in run_g["stages"]:
        assert (stage["beta"], stage["lamb"], stage["tau"]) == (1, 1, 1)
        assert (stage["nu_e"], stage["nu_d"]) == (0, 0)
        assert int(stage.get("num_slices", 0)) > 0
        for component in ("encoder", "decoder"):
            for kind in ("core", "tail"):
                assert f"{component}_{kind}_noise_multiplier" not in stage

    # Stage 3 must differ from stage 2 only in name, epochs, num_slices and
    # tail noise.
    for key in (
        "optimizer",
        "lr",
        "lr_schedule",
        "gradient_clip_norm",
        "weight_decay",
        "adam_eps",
        "beta",
        "lamb",
        "tau",
        "nu_e",
        "nu_d",
        "rho",
        "core_noise_multiplier",
        "eval_every",
    ):
        assert tail[key] == core[key], key
    assert tail["tail_noise_multiplier"] != core["tail_noise_multiplier"]


def test_joint_contract_masks_invariant_mass_and_anchors():
    """Every joint config must mask log_pair_mass and use no mass anchor."""
    for config_path in (RUN_A_CONFIG, RUN_E_CONFIG, RUN_F_CONFIG, RUN_G_CONFIG):
        config = resolve_joint_config(load_config(config_path))
        indices = resolve_condition_indices(config["model"].get("condition_features"))
        assert 8 not in indices, config_path
        assert not config.get("allow_mass_anchor", False), config_path
        for region in config["region_order"]:
            merged = dict(config["loss"])
            merged.update(config["regions"][region].get("loss", {}))
            assert float(merged.get("resonance_mass_w1", 0.0) or 0.0) == 0.0
            assert float(merged.get("mass_w1", 0.0) or 0.0) == 0.0
        arrays = _arrays()
        model = build_joint_autoencoder(
            config["model"], arrays, MUON_MASS, [MUON_MASS, MUON_MASS]
        )
        assert 8 not in model.encoder.condition_indices.tolist()
        assert 8 not in model.decoder.condition_indices.tolist()
        # The runtime guards accept these contracts.
        assert_mass_not_conditioned(model, config["run_name"])
        assert_no_mass_anchor(config, config["run_name"])


def test_mass_anchor_and_condition_guards_reject_violations():
    base = {
        "run_name": "synthetic",
        "region_order": ["jpsi"],
        "regions": {"jpsi": {}},
        "loss": {"resonance_mass_w1": 0.0, "mass_w1": 0.0},
    }
    assert_no_mass_anchor(base)  # clean contract is accepted

    for key in ("resonance_mass_w1", "mass_w1"):
        bad = {"run_name": "synthetic", "region_order": ["jpsi"],
               "regions": {"jpsi": {}}, "loss": {key: 1.0}}
        try:
            assert_no_mass_anchor(bad)
            raise AssertionError(f"guard accepted a {key} anchor")
        except RuntimeError:
            pass
        overridden = dict(bad)
        overridden["allow_mass_anchor"] = True
        assert_no_mass_anchor(overridden)  # explicit, declared override

    # A region-level override must be caught too.
    region_level = {
        "run_name": "synthetic",
        "region_order": ["jpsi"],
        "regions": {"jpsi": {"loss": {"mass_w1": 1.0}}},
        "loss": {},
    }
    try:
        assert_no_mass_anchor(region_level)
        raise AssertionError("guard accepted a region-level mass anchor")
    except RuntimeError:
        pass

    # The condition guard rejects index 8 in either map.
    arrays = _arrays()
    model = build_joint_autoencoder(_model_config(), arrays, MUON_MASS, [MUON_MASS, MUON_MASS])
    assert_mass_not_conditioned(model)
    with torch.no_grad():
        model.encoder.condition_indices = torch.tensor([0, 1, 2, 8], dtype=torch.long)
    try:
        assert_mass_not_conditioned(model)
        raise AssertionError("guard accepted index 8 in the encoder mask")
    except RuntimeError:
        pass


def test_run_h_sliced_loss_contract():
    """Run H = 4x256, 20/80/80, num_slices 256, sliced conditional mass term."""
    run_g = resolve_joint_config(load_config(RUN_G_CONFIG))
    run_h = resolve_joint_config(load_config(RUN_H_CONFIG))

    assert run_h["run_name"] == "Run_H"
    assert run_h["model"]["hidden_dims"] == [256, 256, 256, 256]
    assert [stage["name"] for stage in run_h["stages"]] == [
        "runH_stage1_deterministic_warmup",
        "runH_stage2_stochastic_core",
        "runH_stage3_stochastic_tail",
    ]
    assert [stage["epochs"] for stage in run_h["stages"]] == [20, 80, 80]
    assert all(stage["num_slices"] == 256 for stage in run_h["stages"])
    # The 0.5 mass-kinematics coupling moved from the joint SWD to the sliced
    # conditional term, so the two are not stacked.
    assert float(run_h["loss"]["mass_kin_swd"]) == 0.0
    assert float(run_h["loss"]["sliced_mass_w1"]) == 0.5
    assert int(run_h["loss"]["sliced_mass_w1_slices"]) == 4
    assert int(run_h["loss"]["sliced_mass_w1_min_events"]) == 512
    # Everything else inherited from Run G.
    assert run_h["loaders"] == run_g["loaders"]
    assert run_h["regions"] == run_g["regions"]
    assert run_h["region_weights"] == run_g["region_weights"]


def test_sliced_mass_w1_component_and_validation():
    loss_config = {
        "kind": "cms_jpsi_doublemuon_loss",
        "num_slices": 16,
        "p": 1,
        "sliced_mass_w1": 1.0,
        "sliced_mass_w1_slices": 4,
        "sliced_mass_w1_min_events": 200,
        "raw_swd": 0.0,
        "marginal_w1": 0.0,
        "mass_w1": 0.0,
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
    }
    rng = np.random.default_rng(0)
    values = rng.normal(size=(20000, 8)).astype(np.float32) * 0.5
    values[:, 1] += 2.0
    values[:, 5] -= 2.0
    factory = CmsJpsiDoubleMuonLossFactory(values, values, loss_config, [MUON_MASS, MUON_MASS])
    space = factory.x_space
    tensor = torch.as_tensor(values)
    assert float(space.sliced_mass_w1(tensor, tensor)) == 0.0

    # A distortion applied only to the high-pair-pT half must register.
    pt = torch.hypot(tensor[:, 0] + tensor[:, 4], tensor[:, 1] + tensor[:, 5])
    high = pt > torch.median(pt)
    distorted = tensor.clone()
    distorted[high, 0:3] *= 1.03
    distorted[high, 4:7] *= 1.03
    assert float(space.sliced_mass_w1(tensor, distorted)) > 0.0

    # Weight 0 is the back-compatible no-op.
    off = dict(loss_config)
    off["sliced_mass_w1"] = 0.0
    off_factory = CmsJpsiDoubleMuonLossFactory(values, values, off, [MUON_MASS, MUON_MASS])
    assert float(off_factory.x_space.sliced_mass_w1(tensor, distorted)) == 0.0

    # Bad slice settings fail fast.
    for bad in (
        {"sliced_mass_w1_slices": 1},
        {"sliced_mass_w1_slices": 99},
        {"sliced_mass_w1_min_events": 0},
    ):
        broken = dict(loss_config)
        broken.update(bad)
        try:
            CmsJpsiDoubleMuonLossFactory(values, values, broken, [MUON_MASS, MUON_MASS])
            raise AssertionError(f"validation accepted {bad}")
        except ValueError:
            pass


def test_cycle_weight_scale_multiplies_every_enabled_stage():
    """--cycle-weight-scale is a probe knob; disabled stages must not move."""
    from types import SimpleNamespace

    from run_joint import apply_overrides

    config = {
        "stages": [
            {"name": "stage1", "enabled": True, "beta": 1.0},
            {"name": "stage2", "enabled": False, "beta": 2.0},
        ]
    }
    args = SimpleNamespace(
        output_dir=None,
        epochs=None,
        batch_size=None,
        steps_per_epoch=None,
        cycle_weight_scale=5.0,
        smoke=False,
    )
    out = apply_overrides(config, args)
    assert out["stages"][0]["beta"] == 5.0
    assert out["stages"][1]["beta"] == 2.0

    args.cycle_weight_scale = 0.0
    try:
        apply_overrides(config, args)
        raise AssertionError("non-positive cycle weight scale was accepted")
    except ValueError:
        pass


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for function in (
        test_run_a_config_is_two_region_mass_blind_contract,
        test_run_b_changes_only_the_z_prior_and_run_identity,
        test_run_c_contract_has_requested_depth_duration_batch_and_selection,
        test_rms_checkpoint_selection_balances_all_target_metrics,
        test_run_c_fullscale_is_uncapped_and_preserves_run_c_model_contract,
        test_run_d_changes_only_z_prior_identity_and_holdout_claim,
        test_run_e_changes_only_epoch_definition,
        test_run_e_constant_noise_arm_shares_one_schedule_for_both_maps,
        test_load_config_replace_keys_replaces_named_stage_list,
        test_run_f_flat_loss_mechanism_contract,
        test_run_f_prior_contract_is_preserved_and_disjoint,
        test_run_g_three_stage_contract,
        test_run_h_sliced_loss_contract,
        test_sliced_mass_w1_component_and_validation,
        test_joint_contract_masks_invariant_mass_and_anchors,
        test_mass_anchor_and_condition_guards_reject_violations,
        test_expand_indices_pads_to_floor_and_is_reproducible,
        test_run_e_encoder_det_arm_zeroes_encoder_noise_only,
        test_run_e_encoder_det_short_schedule_and_projection_budget,
        test_resolve_noise_multipliers_defaults_and_overrides,
        test_evaluate_region_scores_deterministically_and_restores_model_noise,
        test_validate_joint_records_deterministic_validation_noise,
        test_fullscale_coverage_stream_visits_every_row_before_repeating,
        test_full_pass_batches_visit_every_training_row_once,
        test_joint_model_is_shared_and_preserves_muon_mass_shell,
        test_worst_region_checkpoint_score_cannot_be_hidden_by_other_region,
        test_joint_contract_hash_is_independent_of_cache_hit_state,
        test_one_joint_step_updates_shared_parameters,
        test_non_finite_gradient_skips_update_without_crashing,
        test_full_pass_training_consumes_unequal_partitions_once,
        test_batch_floor_pads_small_streams_and_is_reported,
        test_stage_last_checkpoint_path_is_run_prefixed_per_stage,
        test_cycle_weight_scale_multiplies_every_enabled_stage,
    ):
        suite.addTest(unittest.FunctionTestCase(function))
    return suite


if __name__ == "__main__":
    unittest.main()
