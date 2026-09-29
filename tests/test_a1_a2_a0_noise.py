"""Tests for the non-training response-model changes (A1, A2, A0.3).

A1: the cycle term can be computed without the decoder's injected noise.
A2: optional tail sigma floor and per-map noise overrides.
A0.3: the optional fixed-z noise-budget validation metric.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (
    REPO_ROOT / "scripts",
    REPO_ROOT / "scripts_sota",
    REPO_ROOT / "scripts_joint",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from cms_data import load_config  # noqa: E402
from cylindrical_flow import CylindricalFlowStep  # noqa: E402
from joint_data import resolve_joint_config  # noqa: E402
from joint_metrics import noise_budget_metrics  # noqa: E402
from joint_model import _resolve_noise_overrides  # noqa: E402
from joint_trainer import (  # noqa: E402
    _decode_cycle,
    _load_state,
    _resolve_cycle_decoder_noise,
)


def _step_config(**overrides):
    config = {
        "activation": "SiLU",
        "hidden_dims": [16],
        "core_log_sigma_bias": -4.0,
        "tail_log_sigma_bias": -5.0,
        "student_t_degrees_of_freedom": 4.0,
        "maximum_heavy_noise": 6.0,
        "mean_residual_limits": [0.35, 0.30, 0.25, 0.35, 0.30, 0.25],
        "core_sigma_floors": [0.001] * 6,
        "core_sigma_scales": [0.08, 0.08, 0.06, 0.08, 0.08, 0.06],
        "tail_sigma_scales": [0.12, 0.12, 0.10, 0.12, 0.12, 0.10],
    }
    config.update(overrides)
    return config


class _StubSteps:
    def __init__(self, core: float, tail: float):
        self.core_noise_multiplier = float(core)
        self.tail_noise_multiplier = float(tail)


class _StubMap(nn.Module):
    def __init__(self, core: float, tail: float):
        super().__init__()
        self.steps = [_StubSteps(core, tail)]


class _StubModel(nn.Module):
    """Minimal joint-like model for the noise-policy helpers."""

    def __init__(self, core: float = 1.0, tail: float = 0.5, noise_scale: float = 1.0):
        super().__init__()
        self.encoder = _StubMap(core, tail)
        self.decoder = _StubMap(core, tail)
        self.noise_scale = float(noise_scale)

    def set_component_noise_multipliers(
        self, *, encoder_core, encoder_tail, decoder_core, decoder_tail
    ) -> None:
        for module, (core, tail) in (
            (self.encoder, (encoder_core, encoder_tail)),
            (self.decoder, (decoder_core, decoder_tail)),
        ):
            for step in module.steps:
                step.core_noise_multiplier = float(core)
                step.tail_noise_multiplier = float(tail)

    def decode(self, values: torch.Tensor) -> torch.Tensor:
        scale = self.decoder.steps[0].core_noise_multiplier * self.noise_scale
        return values + scale * torch.randn_like(values)


class TailFloorTests(unittest.TestCase):
    def test_default_has_no_tail_floor_buffer(self):
        step = CylindricalFlowStep(13, [16], nn.SiLU, _step_config())
        self.assertIsNone(step.tail_sigma_floors)
        self.assertNotIn("tail_sigma_floors", step.state_dict())

    def test_configured_tail_floor_is_registered(self):
        floors = [0.002] * 6
        step = CylindricalFlowStep(
            13, [16], nn.SiLU, _step_config(tail_sigma_floors=floors)
        )
        self.assertIn("tail_sigma_floors", step.state_dict())
        self.assertTrue(
            torch.allclose(step.tail_sigma_floors, torch.tensor(floors, dtype=torch.float32))
        )

    def test_negative_floor_rejected(self):
        with self.assertRaises(ValueError):
            CylindricalFlowStep(
                13, [16], nn.SiLU, _step_config(tail_sigma_floors=[-0.1] * 6)
            )

    def test_negative_core_floor_rejected(self):
        with self.assertRaises(ValueError):
            CylindricalFlowStep(13, [16], nn.SiLU, _step_config(core_sigma_floors=[-0.1] * 6))


class PhysicsSigmaFloorTests(unittest.TestCase):
    """A2.3: eta-binned physics floor and the frozen-amplitude mode."""

    SPEC = {
        "eta_bins": [0.0, 1.2, 2.4],
        "logpt_a": [0.01, 0.02],
        "logpt_b": [0.5, 1.0],
        "phi_a": [0.0, 0.0],
        "phi_b": [0.001, 0.002],
        "eta_a": [0.0, 0.0],
        "eta_b": [0.0, 0.0],
        "tail_ratio": 0.0,
        "scale": 1.0,
    }

    def _quiet_step(self, **overrides):
        config = _step_config(**overrides)
        with torch.no_grad():
            step = CylindricalFlowStep(13, [16], nn.SiLU, config)
            # Drive the learned core/tail sigma to ~0 so the floor is visible.
            step.head.bias[6:18].fill_(-20.0)
        return step

    def _sample_std(self, step, coordinates, draws: int = 3000):
        torch.manual_seed(1234)
        condition = torch.zeros(len(coordinates), 13)
        deltas = [
            (step(coordinates, condition) - coordinates).detach() for _ in range(draws)
        ]
        return torch.stack(deltas).std(dim=0)

    POWER_SPEC = {
        "eta_bins": [0.0, 1.2, 2.4],
        "power_c": [0.01, 0.02],
        "power_alpha": [0.5, 0.0],
        "power_pivot_gev": 10.0,
        "tail_ratio": 0.25,
    }

    def test_power_law_spec_validation(self):
        bad_specs = (
            {"eta_bins": [0.0, 1.0], "power_c": [0.01]},
            {"eta_bins": [0.0, 1.0], "power_alpha": [0.5]},
            {
                "eta_bins": [0.0, 1.0],
                "power_c": [0.01],
                "power_alpha": [0.5],
                "logpt_a": [0.01],
                "logpt_b": [0.0],
            },
            {"eta_bins": [0.0, 1.0], "power_c": [-0.01], "power_alpha": [0.5]},
            {"eta_bins": [0.0, 1.0], "power_c": [0.01, 0.02], "power_alpha": [0.5]},
            {"eta_bins": [0.0, 1.0], "power_c": [0.01], "power_alpha": [float("nan")]},
            {
                "eta_bins": [0.0, 1.0],
                "power_c": [0.01],
                "power_alpha": [0.5],
                "power_pivot_gev": 0.0,
            },
        )
        for spec in bad_specs:
            with self.assertRaises(ValueError, msg=str(spec)):
                CylindricalFlowStep(
                    13, [16], nn.SiLU, _step_config(core_sigma_floor_spec=spec)
                )

    def test_power_law_floor_values(self):
        step = CylindricalFlowStep(
            13, [16], nn.SiLU, _step_config(core_sigma_floor_spec=self.POWER_SPEC)
        )
        coordinates = torch.zeros(3, 6)
        coordinates[:, 1] = coordinates[:, 4] = 0.5  # first bin: c=0.01, alpha=0.5
        coordinates[:, 0] = coordinates[:, 3] = math.log(10.0)
        floor = step._sigma_floor(coordinates)
        self.assertAlmostEqual(float(floor[0, 0]), 0.01, places=7)
        coordinates[:, 0] = coordinates[:, 3] = math.log(40.0)
        floor = step._sigma_floor(coordinates)
        self.assertAlmostEqual(float(floor[0, 0]), 0.02, places=7)
        coordinates[:, 1] = coordinates[:, 4] = 1.8  # second bin: c=0.02, alpha=0
        floor = step._sigma_floor(coordinates)
        self.assertAlmostEqual(float(floor[0, 0]), 0.02, places=7)

    def test_power_law_spec_is_not_a_buffer(self):
        plain = CylindricalFlowStep(13, [16], nn.SiLU, _step_config())
        powered = CylindricalFlowStep(
            13, [16], nn.SiLU, _step_config(core_sigma_floor_spec=self.POWER_SPEC)
        )
        powered.load_state_dict(plain.state_dict())

    def test_amplitude_cap_and_pt_clamp(self):
        spec = {
            "eta_bins": [0.0, 2.4],
            "linear_offset": [0.01],
            "linear_slope": [1.0],  # absurd slope; the cap must bound it
            "sigma_cap": 0.05,
            "pt_max_gev": 50.0,
            "tail_ratio": 0.0,
        }
        step = CylindricalFlowStep(
            13, [16], nn.SiLU, _step_config(core_sigma_floor_spec=spec)
        )
        coordinates = torch.zeros(3, 6)
        coordinates[:, 0] = coordinates[:, 3] = float(torch.log(torch.tensor(1000.0)))
        floor = step._sigma_floor(coordinates)
        self.assertLessEqual(float(floor.max()), 0.05 + 1e-9)

    def test_cap_and_pt_max_validation(self):
        for spec in (
            {"eta_bins": [0.0, 1.0], "linear_offset": [0.01], "linear_slope": [0.0], "sigma_cap": 0.0},
            {"eta_bins": [0.0, 1.0], "linear_offset": [0.01], "linear_slope": [0.0], "pt_max_gev": -1.0},
        ):
            with self.assertRaises(ValueError, msg=str(spec)):
                CylindricalFlowStep(
                    13, [16], nn.SiLU, _step_config(core_sigma_floor_spec=spec)
                )

    def test_default_off_and_state_dict_unchanged(self):
        plain = CylindricalFlowStep(13, [16], nn.SiLU, _step_config())
        self.assertIsNone(plain.sigma_floor_spec)
        self.assertFalse(plain.freeze_noise_amplitude)
        with_spec = CylindricalFlowStep(
            13, [16], nn.SiLU, _step_config(core_sigma_floor_spec=self.SPEC)
        )
        # The spec is a plain attribute, not a buffer: no new state-dict keys.
        with_spec.load_state_dict(plain.state_dict())

    def test_spec_validation_rejects_bad_specs(self):
        bad_specs = (
            {"eta_bins": [0.0], "logpt_a": [0.01], "logpt_b": [0.0]},
            {"eta_bins": [0.0, 1.0], "logpt_a": [0.01, 0.02], "logpt_b": [0.0]},
            {"eta_bins": [0.0, 1.0], "logpt_a": [-0.01], "logpt_b": [0.0]},
            {"eta_bins": [1.0, 0.0], "logpt_a": [0.01], "logpt_b": [0.0]},
            {"eta_bins": [0.0, 1.0], "logpt_a": [0.01], "logpt_b": [0.0], "bogus": 1},
            {"eta_bins": [0.0, 1.0], "logpt_a": [0.01], "logpt_b": [0.0], "scale": 0.0},
        )
        for spec in bad_specs:
            with self.assertRaises(ValueError, msg=str(spec)):
                CylindricalFlowStep(
                    13, [16], nn.SiLU, _step_config(core_sigma_floor_spec=spec)
                )

    def test_freeze_without_spec_rejected(self):
        with self.assertRaises(ValueError):
            CylindricalFlowStep(
                13, [16], nn.SiLU, _step_config(freeze_noise_amplitude=True)
            )

    def test_clamp_mode_raises_amplitude_to_the_floor(self):
        step = self._quiet_step(core_sigma_floor_spec=self.SPEC)
        coordinates = torch.zeros(4, 6)
        coordinates[:, 0] = coordinates[:, 3] = float(torch.log(torch.tensor(5.0)))
        coordinates[:, 1] = coordinates[:, 4] = 0.3  # first eta bin
        std = self._sample_std(step, coordinates)
        expected = (0.01**2 + (0.5 / 5.0) ** 2) ** 0.5
        self.assertAlmostEqual(float(std[:, 0].mean()), expected, delta=0.01)
        self.assertAlmostEqual(float(std[:, 3].mean()), expected, delta=0.01)

    def test_eta_bin_lookup_uses_the_second_bin(self):
        step = self._quiet_step(core_sigma_floor_spec=self.SPEC)
        coordinates = torch.zeros(4, 6)
        coordinates[:, 0] = coordinates[:, 3] = float(torch.log(torch.tensor(5.0)))
        coordinates[:, 1] = coordinates[:, 4] = 1.8  # second eta bin
        std = self._sample_std(step, coordinates)
        expected = (0.02**2 + (1.0 / 5.0) ** 2) ** 0.5
        self.assertAlmostEqual(float(std[:, 0].mean()), expected, delta=0.01)

    def test_frozen_mode_matches_floor_and_zeroes_sigma_gradients(self):
        step = self._quiet_step(
            core_sigma_floor_spec=self.SPEC, freeze_noise_amplitude=True
        )
        coordinates = torch.zeros(3, 6)
        coordinates[:, 0] = coordinates[:, 3] = float(torch.log(torch.tensor(5.0)))
        std = self._sample_std(step, coordinates)
        expected = (0.01**2 + (0.5 / 5.0) ** 2) ** 0.5
        self.assertAlmostEqual(float(std[:, 0].mean()), expected, delta=0.01)

        output = step(coordinates, torch.zeros(3, 13))
        output.sum().backward()
        gradient = step.head.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.allclose(gradient[6:], torch.zeros_like(gradient[6:])))


class PerMapOverrideTests(unittest.TestCase):
    BASE = {
        "core_sigma_floors": [0.001] * 6,
        "tail_sigma_scales": [0.12] * 6,
        "hidden_dims": [16],
    }

    def test_default_config_is_returned_unchanged(self):
        config = dict(self.BASE)
        self.assertIs(_resolve_noise_overrides(config, "decoder"), config)

    def test_decoder_override_merges_and_leaves_encoder_alone(self):
        config = dict(self.BASE)
        config["decoder_noise_overrides"] = {"core_sigma_floors": [0.005] * 6}
        decoder = _resolve_noise_overrides(config, "decoder")
        encoder = _resolve_noise_overrides(config, "encoder")
        self.assertEqual(decoder["core_sigma_floors"], [0.005] * 6)
        self.assertEqual(encoder["core_sigma_floors"], [0.001] * 6)
        self.assertEqual(decoder["tail_sigma_scales"], [0.12] * 6)

    def test_unknown_override_key_rejected(self):
        config = dict(self.BASE)
        config["decoder_noise_overrides"] = {"hidden_dims": [32]}
        with self.assertRaises(ValueError):
            _resolve_noise_overrides(config, "decoder")

    def test_non_mapping_override_rejected(self):
        config = dict(self.BASE)
        config["encoder_noise_overrides"] = [0.001] * 6
        with self.assertRaises(ValueError):
            _resolve_noise_overrides(config, "encoder")

    def test_physics_kernel_override_keys_are_allowed(self):
        config = dict(self.BASE)
        spec = {"eta_bins": [0.0, 2.4], "logpt_a": [0.01], "logpt_b": [0.0]}
        config["decoder_noise_overrides"] = {
            "core_sigma_floor_spec": spec,
            "freeze_noise_amplitude": True,
        }
        decoder = _resolve_noise_overrides(config, "decoder")
        self.assertEqual(decoder["core_sigma_floor_spec"], spec)
        self.assertTrue(decoder["freeze_noise_amplitude"])


class CheckpointBackwardCompatibilityTests(unittest.TestCase):
    class _Model(nn.Module):
        def __init__(self, with_floor: bool):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(2))
            if with_floor:
                self.register_buffer("tail_sigma_floors", torch.zeros(6))

    def test_old_state_dict_without_tail_floor_loads(self):
        source = self._Model(with_floor=False)
        target = self._Model(with_floor=True)
        _load_state(target, source.state_dict())

    def test_unexpected_key_still_raises(self):
        source = self._Model(with_floor=False)
        state = source.state_dict()
        state["surprise"] = torch.zeros(1)
        with self.assertRaises(RuntimeError):
            _load_state(self._Model(with_floor=False), state)

    def test_missing_non_floor_key_still_raises(self):
        source = self._Model(with_floor=False)
        state = source.state_dict()
        state.pop("weight")
        with self.assertRaises(RuntimeError):
            _load_state(self._Model(with_floor=False), state)


class CycleNoiseDecouplingTests(unittest.TestCase):
    def test_native_mode_is_stochastic_and_keeps_multipliers(self):
        model = _StubModel()
        torch.manual_seed(1)
        first = _decode_cycle(model, torch.zeros(4, 8), "native")
        torch.manual_seed(2)
        second = _decode_cycle(model, torch.zeros(4, 8), "native")
        self.assertFalse(torch.allclose(first, second))
        self.assertEqual(model.decoder.steps[0].core_noise_multiplier, 1.0)
        self.assertEqual(model.decoder.steps[0].tail_noise_multiplier, 0.5)

    def test_zero_mode_is_deterministic_and_restores_multipliers(self):
        model = _StubModel()
        torch.manual_seed(1)
        first = _decode_cycle(model, torch.zeros(4, 8), "zero")
        torch.manual_seed(2)
        second = _decode_cycle(model, torch.zeros(4, 8), "zero")
        self.assertTrue(torch.allclose(first, second))
        self.assertEqual(model.decoder.steps[0].core_noise_multiplier, 1.0)
        self.assertEqual(model.decoder.steps[0].tail_noise_multiplier, 0.5)
        self.assertEqual(model.encoder.steps[0].core_noise_multiplier, 1.0)

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            _decode_cycle(_StubModel(), torch.zeros(2, 8), "sometimes")

    def test_resolution_precedence(self):
        self.assertEqual(_resolve_cycle_decoder_noise({}, {}), "native")
        self.assertEqual(_resolve_cycle_decoder_noise({"cycle_decoder_noise": "zero"}, {}), "zero")
        self.assertEqual(
            _resolve_cycle_decoder_noise(
                {"cycle_decoder_noise": "native"}, {"cycle_decoder_noise": "zero"}
            ),
            "zero",
        )
        with self.assertRaises(ValueError):
            _resolve_cycle_decoder_noise({"cycle_decoder_noise": "half"}, {})


class NoiseBudgetMetricTests(unittest.TestCase):
    def test_within_width_matches_injected_noise_and_fraction(self):
        model = _StubModel(core=1.0, tail=0.0, noise_scale=0.3)
        z = np.zeros((64, 8), dtype=np.float32)
        metrics = noise_budget_metrics(
            model,
            z,
            daughter_masses=[0.1056583755, 0.1056583755],
            device=torch.device("cpu"),
            batch_size=512,
            draws=64,
            max_events=None,
            seed=3,
            core=1.0,
            tail=0.0,
        )
        # The stub decodes a nonlinear four-vector, so the expected width is
        # measured from the same stub rather than assumed to be the noise scale.
        from physics import invariant_mass_np

        reference = torch.zeros(200, len(z), 8)
        torch.manual_seed(99)
        for index in range(200):
            reference[index] = model.decode(torch.zeros(len(z), 8))
        reference_masses = invariant_mass_np(
            reference.reshape(-1, 8).numpy(),
            daughter_masses=[0.1056583755, 0.1056583755],
            stable=True,
        ).reshape(200, len(z))
        expected = float(np.median(reference_masses.std(axis=0)))
        self.assertEqual(metrics["noise_budget_events"], 64.0)
        self.assertAlmostEqual(
            metrics["noise_budget_within_std_median_gev"], expected, delta=0.02 * expected + 0.01
        )
        self.assertGreater(metrics["noise_budget_noise_variance_fraction"], 0.9)
        self.assertGreater(
            metrics["noise_budget_sigma_only_within_gev"],
            0.8 * metrics["noise_budget_within_std_median_gev"],
        )

    def test_single_draw_rejected(self):
        with self.assertRaises(ValueError):
            noise_budget_metrics(
                _StubModel(),
                np.zeros((4, 8), dtype=np.float32),
                daughter_masses=[0.1056583755, 0.1056583755],
                device=torch.device("cpu"),
                batch_size=64,
                draws=1,
                max_events=None,
                seed=0,
                core=1.0,
                tail=0.0,
            )

    def test_multipliers_restored(self):
        model = _StubModel(core=1.0, tail=0.5)
        noise_budget_metrics(
            model,
            np.zeros((8, 8), dtype=np.float32),
            daughter_masses=[0.1056583755, 0.1056583755],
            device=torch.device("cpu"),
            batch_size=64,
            draws=4,
            max_events=None,
            seed=0,
            core=1.0,
            tail=0.0,
        )
        self.assertEqual(model.decoder.steps[0].core_noise_multiplier, 1.0)
        self.assertEqual(model.decoder.steps[0].tail_noise_multiplier, 0.5)


class ShippedA2ConfigTests(unittest.TestCase):
    """The shipped A2 configs must resolve and wire the fitted kernel.

    These tests fail if the config plumbing silently drops the kernel spec,
    flips the freeze flag, or changes the fitted coefficients, so the two
    pending full runs can be launched from exactly the preflighted state.
    """

    FROZEN = REPO_ROOT / "configs_joint" / "cms_Joint_runH_A2frozen.yaml"
    FLOOR = REPO_ROOT / "configs_joint" / "cms_Joint_runH_A2floor.yaml"
    ETA_BINS = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4]
    OFFSET = [0.00595074, 0.00675931, 0.00802552, 0.00881325, 0.00917026, 0.00880492]
    SLOPE = [0.000390620, 0.000443697, 0.000526813, 0.000578522, 0.000601956, 0.000577975]

    def _decoder_config(self, path: Path) -> dict:
        model = resolve_joint_config(load_config(path))["model"]
        return _resolve_noise_overrides(model, "decoder")

    def _step(self, path: Path) -> CylindricalFlowStep:
        config = _step_config(**self._decoder_config(path))
        return CylindricalFlowStep(13, [16], nn.SiLU, config)

    def _deltas(self, step: CylindricalFlowStep, rows: int) -> torch.Tensor:
        coordinates = torch.zeros(1, 6)
        coordinates[:, 0] = coordinates[:, 3] = math.log(10.0)
        coordinates[:, 1] = coordinates[:, 4] = 0.2
        condition = torch.zeros(1, 13)
        draws = []
        for index in range(rows):
            torch.manual_seed(4000 + index)
            draws.append((step(coordinates, condition) - coordinates).detach())
        return torch.cat(draws, dim=0)

    def test_frozen_config_resolves_with_the_fitted_kernel(self):
        config = self._decoder_config(self.FROZEN)
        spec = config["core_sigma_floor_spec"]
        self.assertEqual(spec["eta_bins"], self.ETA_BINS)
        self.assertEqual(spec["linear_offset"], self.OFFSET)
        self.assertEqual(spec["linear_slope"], self.SLOPE)
        self.assertEqual(float(spec["tail_ratio"]), 0.25)
        self.assertEqual(float(spec["sigma_cap"]), 0.25)
        self.assertEqual(float(spec["pt_max_gev"]), 200.0)
        self.assertTrue(config["freeze_noise_amplitude"])
        # The step infers the linear mode from the shipped coefficients.
        step = self._step(self.FROZEN)
        self.assertEqual(step.sigma_floor_spec["mode"], "linear")
        self.assertAlmostEqual(
            float(step.sigma_floor_spec["linear_offset"][2]), self.OFFSET[2]
        )

    def test_floor_config_differs_only_in_the_freeze_flag(self):
        frozen = self._decoder_config(self.FROZEN)
        floor = self._decoder_config(self.FLOOR)
        self.assertFalse(floor["freeze_noise_amplitude"])
        self.assertEqual(floor["core_sigma_floor_spec"], frozen["core_sigma_floor_spec"])

    def test_shipped_kernel_values_match_the_fitted_coefficients(self):
        step = self._step(self.FROZEN)
        for eta, pt, expected in (
            (0.2, 10.0, 0.00595074 + 0.000390620 * 10.0),
            (2.1, 100.0, 0.00880492 + 0.000577975 * 100.0),
            (3.0, 10.0, 0.00880492 + 0.000577975 * 10.0),
            (2.1, 400.0, 0.00880492 + 0.000577975 * 200.0),  # pt_max clamp
        ):
            coordinates = torch.zeros(2, 6)
            coordinates[:, 0] = coordinates[:, 3] = math.log(pt)
            coordinates[:, 1] = coordinates[:, 4] = eta
            floor = step._sigma_floor(coordinates)
            for column in (0, 3):
                self.assertAlmostEqual(float(floor[0, column]), expected, places=6)
        # The non-log-pT channels default to zero but are clamped, not NaN.
        self.assertTrue(bool(torch.isfinite(step._sigma_floor(coordinates)).all()))

    def test_frozen_amplitude_ignores_the_learned_sigma(self):
        step = self._step(self.FROZEN)
        with torch.no_grad():
            step.head.bias[6:].fill_(10.0)  # a learned sigma far above the kernel
        loud = self._deltas(step, rows=200)
        with torch.no_grad():
            step.head.bias[6:].fill_(-20.0)  # a learned sigma far below the kernel
        quiet = self._deltas(step, rows=200)
        ratio = float(loud[:, 0].std() / quiet[:, 0].std())
        self.assertAlmostEqual(ratio, 1.0, delta=0.05)

    def test_floor_config_lets_the_learned_sigma_grow(self):
        step = self._step(self.FLOOR)
        with torch.no_grad():
            step.head.bias[6:].fill_(-20.0)
        quiet = self._deltas(step, rows=200)
        with torch.no_grad():
            step.head.bias[6:].fill_(10.0)
        loud = self._deltas(step, rows=200)
        self.assertGreater(float(loud[:, 0].std()), 10.0 * float(quiet[:, 0].std()))

    def test_cycle_noise_config_is_a_one_key_change_from_the_frozen_arm(self):
        """A2.4 must differ from A2frozen only in `cycle_decoder_noise`."""
        cycle_path = REPO_ROOT / "configs_joint" / "cms_Joint_runH_A2cycleNoise.yaml"
        frozen = resolve_joint_config(load_config(self.FROZEN))
        cycle = resolve_joint_config(load_config(cycle_path))
        self.assertEqual(cycle["cycle_decoder_noise"], "native")
        self.assertEqual(frozen["cycle_decoder_noise"], "zero")
        for key in ("model", "regions", "stages", "loaders", "prior_components"):
            self.assertEqual(cycle[key], frozen[key], f"{key} changed unexpectedly")
        self.assertEqual(cycle["run_name"], "Run_H_A2cycleNoise")
        self.assertEqual(cycle["comparison"]["baseline_run"], "Run_H_A2frozen")


if __name__ == "__main__":
    unittest.main()
