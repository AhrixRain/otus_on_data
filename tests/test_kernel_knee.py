"""Tests for the knee-augmented linear resolution kernel (I2 / S2', 2026-10-04).

Motivation: outputs/cms_Joint/kernel_joint_calibration/ measures that a pure
offset + slope*pT law cannot satisfy J/psi, Z and Upsilon at once, because the
required per-muon log-pT amplitude is nearly flat between muon pT 4.7 and
13.1 GeV and rises only above that. The knee form adds a third parameter,

    sigma_logpT = scale * (offset + slope * max(0, pT - knee_pt_gev)),

which remains a mass-blind function of the muon kinematics alone.
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

from cylindrical_flow import CylindricalFlowStep, _validate_sigma_floor_spec  # noqa: E402


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


BASE_SPEC = {
    "eta_bins": [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4],
    "linear_offset": [0.008] * 6,
    "linear_slope": [5e-4] * 6,
    "tail_ratio": 0.25,
    "sigma_cap": 0.25,
    "pt_max_gev": 200.0,
}

POINTS_GEV = [0.5, 1.0, 3.0, 7.0, 20.0, 60.0]


def _coordinates(points_gev):
    rows = [
        [math.log(pt), 0.2, 0.0, math.log(pt), 0.2, 0.0] for pt in points_gev
    ]
    return torch.tensor(rows, dtype=torch.float64)


class KernelKneeTests(unittest.TestCase):
    def _sigma(self, spec):
        step = CylindricalFlowStep(
            13, [16], nn.SiLU, _step_config(core_sigma_floor_spec=spec)
        )
        step.eval()
        with torch.no_grad():
            return step._sigma_floor(_coordinates(POINTS_GEV))

    def test_knee_zero_matches_plain_linear_form(self):
        without = self._sigma(dict(BASE_SPEC))
        with_zero = self._sigma({**BASE_SPEC, "knee_pt_gev": 0.0})
        np.testing.assert_allclose(
            with_zero.numpy(), without.numpy(), rtol=1e-12, atol=1e-12
        )

    def test_plateau_below_knee_and_linear_above(self):
        sigmas = self._sigma({**BASE_SPEC, "knee_pt_gev": 10.0}).numpy()
        points = np.asarray(POINTS_GEV)
        expected = 0.008 + 5e-4 * np.clip(points - 10.0, 0.0, None)
        np.testing.assert_allclose(sigmas[:, 0], expected, rtol=1e-6)
        self.assertAlmostEqual(float(sigmas[0, 0]), 0.008, places=9)
        self.assertAlmostEqual(float(sigmas[3, 0]), 0.008, places=9)
        self.assertGreater(float(sigmas[4, 0]), 0.008)

    def test_eta_and_phi_columns_are_knee_independent(self):
        plain = self._sigma(dict(BASE_SPEC)).numpy()
        knee = self._sigma({**BASE_SPEC, "knee_pt_gev": 10.0}).numpy()
        for column in (1, 2, 4, 5):
            np.testing.assert_allclose(
                knee[:, column], plain[:, column], rtol=1e-12, atol=1e-12
            )

    def test_validator_accepts_knee_and_defaults_to_zero(self):
        self.assertEqual(_validate_sigma_floor_spec(dict(BASE_SPEC))["knee_pt_gev"], 0.0)
        normalized = _validate_sigma_floor_spec({**BASE_SPEC, "knee_pt_gev": 12.5})
        self.assertEqual(normalized["knee_pt_gev"], 12.5)

    def test_validator_rejects_negative_and_non_finite_knee(self):
        with self.assertRaises(ValueError):
            _validate_sigma_floor_spec({**BASE_SPEC, "knee_pt_gev": -1.0})
        with self.assertRaises(ValueError):
            _validate_sigma_floor_spec({**BASE_SPEC, "knee_pt_gev": float("nan")})


if __name__ == "__main__":
    unittest.main()
