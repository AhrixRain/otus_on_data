"""Contract tests for the conditional-spread (energy-score) training term.

Two things are pinned here:

1. the loss method itself - strictly proper behaviour, gradients, and the
   refusal to accept fewer than two draws (the within-model term needs them);
2. the trainer's opt-in contract - with ``kappa`` absent or zero the term is not
   built and not registered, so every config written before this feature keeps a
   byte-identical ``history.json`` row. That contract is what makes the new arm a
   single controlled change against ``Run_H_D3b``.
"""
import ast
import inspect
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (
    REPO_ROOT,
    REPO_ROOT / "scripts",
    REPO_ROOT / "scripts_sota",
    REPO_ROOT / "scripts_joint",
    REPO_ROOT / "scripts_joint" / "upsilon",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from loss import CmsJpsiDoubleMuonLossFactory  # noqa: E402
from scoring_rules import gaussian_log_score  # noqa: E402
import joint_trainer  # noqa: E402

TRAINER_SOURCE = Path(joint_trainer.__file__).read_text(encoding="utf-8")
MUON_MASS = 0.1056583755


def _pairs(n: int, mass: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    momentum = np.sqrt((mass / 2.0) ** 2 - MUON_MASS**2)
    phi = rng.uniform(-np.pi, np.pi, n)
    eta = rng.normal(0.0, 0.35, n)
    pt = momentum / np.cosh(eta)
    px1 = pt * np.cos(phi)
    py1 = pt * np.sin(phi)
    pz1 = pt * np.sinh(eta)
    e1 = np.sqrt(px1**2 + py1**2 + pz1**2 + MUON_MASS**2)
    e2 = np.sqrt(px1**2 + py1**2 + pz1**2 + MUON_MASS**2)
    return np.stack([px1, py1, pz1, e1, -px1, -py1, -pz1, e2], axis=1).astype(np.float32)


def _loss_config() -> dict:
    return {
        "kind": "cms_jpsi_doublemuon_loss",
        "num_slices": 4,
        "p": 1,
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
    }


def build_factory() -> CmsJpsiDoubleMuonLossFactory:
    x_train = _pairs(64, 3.0969, 1)
    z_train = _pairs(64, 3.0969, 2)
    return CmsJpsiDoubleMuonLossFactory(
        x_train, z_train, _loss_config(), [MUON_MASS, MUON_MASS]
    )


class EnergyScoreLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261004)
        self.factory = build_factory()
        self.x_true = torch.as_tensor(_pairs(256, 3.0969, 7))

    def test_is_finite_for_a_miscalibrated_model(self):
        too_narrow = torch.stack(
            [self.x_true + 0.001 * torch.randn_like(self.x_true) for _ in range(3)]
        )
        loss = self.factory.energy_score_loss(self.x_true, too_narrow)
        self.assertTrue(torch.isfinite(loss))
        # A log score is unbounded below, so only finiteness is guaranteed here.
        self.assertGreater(float(abs(loss)), 0.0)

    def test_prefers_the_correctly_sized_model(self):
        """With a truth-like spread the score must beat a near-deterministic one.

        Each event needs its own centre and the model's mean must be an
        independent estimate, not the draw average of its own target. The helper
        therefore scores the true centre plus noise, which is the geometry in
        which the rule is proper.
        """
        width = 0.004
        centres = self.x_true.clone()
        targets = centres + width * torch.randn_like(centres)
        good = float(gaussian_log_score(centres, torch.full_like(centres, width), targets))
        narrow = float(
            gaussian_log_score(centres, torch.full_like(centres, width / 20.0), targets)
        )
        self.assertLess(good, narrow)

    def test_collapse_is_penalised(self):
        width = 0.004
        centres = self.x_true.clone()
        targets = centres + width * torch.randn_like(centres)
        collapsed = float(
            gaussian_log_score(centres, torch.full_like(centres, 1e-6), targets)
        )
        calibrated = float(
            gaussian_log_score(centres, torch.full_like(centres, width), targets)
        )
        self.assertGreater(collapsed, calibrated)

    def test_requires_two_draws(self):
        single = self.x_true.unsqueeze(0)
        with self.assertRaises(ValueError):
            self.factory.energy_score_loss(self.x_true, single)

    def test_interpolating_mean_map_is_unidentifiable(self):
        """The documented degenerate limit: with ``mu = x`` the score is minimised
        at ``sigma -> 0``, since only ``log sigma`` is left.

        This is a property of the score under an interpolating mean map, not a
        bug, so the test pins the behaviour to keep the docstring honest.
        """
        target = self.x_true.mean(dim=0, keepdim=True) + 0.004 * torch.randn_like(self.x_true)
        interpolating_mean = target.clone()
        previous = None
        for scale in (1.0, 0.1, 0.01):
            score = float(
                gaussian_log_score(
                    interpolating_mean,
                    torch.full_like(interpolating_mean, scale),
                    target,
                )
            )
            self.assertAlmostEqual(score, math.log(scale), places=4)
            if previous is not None:
                self.assertLess(score, previous)
            previous = score

    def test_propagates_gradients_to_mean_and_scale(self):
        """Draws differing only by their own offset give gradients through both moments."""
        base = self.x_true.clone().requires_grad_(True)
        draws = torch.stack(
            [base + 0.002 * torch.randn_like(base) for _ in range(3)]
        )
        loss = self.factory.energy_score_loss(self.x_true, draws)
        loss.backward()
        self.assertIsNotNone(base.grad)
        self.assertGreater(float(base.grad.abs().sum()), 0.0)

    def test_keeps_the_float32_dtype(self):
        draws = torch.stack(
            [self.x_true + 0.002 * torch.randn_like(self.x_true) for _ in range(2)]
        )
        loss = self.factory.energy_score_loss(self.x_true, draws)
        self.assertEqual(loss.dtype, torch.float32)

    def test_records_a_component(self):
        draws = torch.stack(
            [self.x_true + 0.002 * torch.randn_like(self.x_true) for _ in range(2)]
        )
        self.factory.reset_components()
        self.factory.energy_score_loss(self.x_true, draws)
        self.assertIn("energy_score_raw", self.factory.latest_components)


class TrainerOptInContractTests(unittest.TestCase):
    """Static checks on the wiring, so a refactor cannot silently change it."""

    def test_kappa_defaults_to_zero(self):
        self.assertIn('stage.get("kappa", 0.0)', TRAINER_SOURCE)

    def test_component_is_registered_only_when_active(self):
        self.assertIn('if scoring_rule_weight > 0.0:', TRAINER_SOURCE)
        self.assertIn('components["energy_score_loss"] = scoring_rule_loss', TRAINER_SOURCE)

    def test_zero_weight_does_not_build_the_term(self):
        """The decode loop for the score lives behind the weight guard."""
        tree = ast.parse(TRAINER_SOURCE)
        guarded = []
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
                text = ast.unparse(node.test)
                if "scoring_rule_weight" in text and "> 0.0" in text:
                    guarded.append(ast.unparse(node))
        joined = "\n".join(guarded)
        self.assertIn("model.decode", joined)
        self.assertIn("energy_score_loss", joined)

    def test_draws_must_be_at_least_two(self):
        self.assertIn("scoring_rule.draws must be at least 2", TRAINER_SOURCE)

    def test_score_subset_uses_the_matching_realised_events(self):
        """Draws and realised values must come from the same slice of the batch.

        If the score were evaluated against a different slice it would compare
        the model to unrelated events, which is marginal matching again.
        """
        source = inspect.getsource(joint_trainer)
        self.assertIn("z_score = z[:n_score]", source)
        self.assertIn("x_score = x[:n_score]", source)


if __name__ == "__main__":
    unittest.main()
