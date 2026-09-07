"""Unit tests for the identity/floor reference gauges.

These are pure NumPy statistics, so this module deliberately imports neither
torch nor any model code: it must stay runnable for offline re-scoring of
finished runs.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (REPO_ROOT / "scripts_joint",):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from identity_reference import (  # noqa: E402
    GAUGED_SUFFIXES,
    gauge,
    headroom,
    ks_distance,
    pair_pt,
    reference_block,
    wasserstein_1d,
    width_relative_error,
)


MUON_MASS = 0.1056583745


def _pairs(rng: np.random.Generator, n: int, mass_sigma: float) -> np.ndarray:
    """Toy [n, 8] muon pairs with an exactly controlled pair-mass spread.

    Built in the pair rest frame and boosted, so the invariant mass of every
    event is the value drawn from the Gaussian (to float precision) and the
    pair pT is non-degenerate. Constructing the mass directly is what makes
    "narrow target, wide input" mean what the A/B means by it.
    """
    mass = 3.0969 + mass_sigma * rng.standard_normal(n)
    momentum = np.sqrt(np.clip((mass / 2.0) ** 2 - MUON_MASS**2, 0.0, None))
    cos_theta = rng.uniform(-1.0, 1.0, size=n)
    sin_theta = np.sqrt(np.clip(1.0 - cos_theta**2, 0.0, None))
    phi = rng.uniform(-np.pi, np.pi, size=n)
    rest = np.stack(
        [
            momentum * sin_theta * np.cos(phi),
            momentum * sin_theta * np.sin(phi),
            momentum * cos_theta,
        ],
        axis=1,
    )
    energy = mass / 2.0

    beta_t = rng.uniform(0.1, 0.6, size=n)
    beta_phi = rng.uniform(-np.pi, np.pi, size=n)
    beta = np.stack(
        [
            beta_t * np.cos(beta_phi),
            beta_t * np.sin(beta_phi),
            rng.uniform(-0.5, 0.5, size=n),
        ],
        axis=1,
    )
    beta2 = np.sum(beta * beta, axis=1)
    gamma = 1.0 / np.sqrt(1.0 - beta2)

    values = np.zeros((n, 8), dtype=np.float64)
    for slot, three in ((0, rest), (4, -rest)):
        bp = np.sum(beta * three, axis=1)
        boosted = three + (
            ((gamma - 1.0) * bp / beta2 + gamma * energy)[:, None] * beta
        )
        values[:, slot : slot + 3] = boosted
        values[:, slot + 3] = gamma * (energy + bp)
    return values


def _mass(values: np.ndarray) -> np.ndarray:
    pair = values[:, 0:4] + values[:, 4:8]
    return np.sqrt(
        np.clip(pair[:, 3] ** 2 - np.sum(pair[:, 0:3] ** 2, axis=1), 0.0, None)
    )


def _metrics(prefix: str, real: np.ndarray, fake: np.ndarray) -> dict[str, float]:
    real_mass, fake_mass = _mass(real), _mass(fake)
    return {
        f"{prefix}_mass_ks": ks_distance(real_mass, fake_mass),
        f"{prefix}_mass_w1_gev": wasserstein_1d(real_mass, fake_mass),
        f"{prefix}_mass_width_rel_error": width_relative_error(real_mass, fake_mass),
        f"{prefix}_pair_pt_ks": ks_distance(pair_pt(real), pair_pt(fake)),
    }


class GaugeAlgebraTest(unittest.TestCase):
    def test_identity_is_exactly_one(self):
        self.assertEqual(gauge(0.4, 0.4, 0.02), 1.0)

    def test_floor_is_exactly_zero(self):
        self.assertEqual(gauge(0.02, 0.4, 0.02), 0.0)

    def test_worse_than_identity_exceeds_one(self):
        self.assertGreater(gauge(0.5, 0.4, 0.02), 1.0)

    def test_no_headroom_refuses_to_certify(self):
        # Identity already at the floor: nothing to detect, so nothing passes.
        self.assertEqual(gauge(0.01, 0.02, 0.02), math.inf)
        self.assertEqual(gauge(0.01, 0.01, 0.02), math.inf)

    def test_gauge_never_negative(self):
        self.assertEqual(gauge(0.0, 0.4, 0.02), 0.0)

    def test_headroom_scales_with_separation(self):
        self.assertAlmostEqual(headroom(0.42, 0.02), 20.0)
        self.assertAlmostEqual(headroom(0.04, 0.02), 1.0)


class ReferenceBlockTest(unittest.TestCase):
    """The three cases the A/B showed we cannot distinguish today."""

    def setUp(self):
        rng = np.random.default_rng(20260905)
        # target ("prior"): narrow in mass. input ("data"): wide in mass.
        self.pool = _pairs(rng, 40000, 0.004)
        self.real = self.pool[:4000]
        self.data = _pairs(rng, 4000, 0.030)

    def _block(self, model_output: np.ndarray) -> dict[str, float]:
        return reference_block(
            "latent",
            real_values=self.real,
            identity_values=self.data,
            reference_pool=self.pool[4000:],
            model_metrics=_metrics("latent", self.real, model_output),
            mass_fn=_mass,
            seed=7,
        )

    def test_identity_model_gauges_to_one(self):
        block = self._block(self.data)
        for suffix in GAUGED_SUFFIXES:
            self.assertAlmostEqual(
                block[f"latent_{suffix}_vs_identity"], 1.0, places=9,
                msg=f"identity model must gauge to exactly 1.0 for {suffix}",
            )

    def test_perfect_model_gauges_near_zero(self):
        rng = np.random.default_rng(11)
        perfect = self.pool[4000:][rng.choice(36000, 4000, replace=False)]
        block = self._block(perfect)
        self.assertLess(block["latent_mass_ks_vs_identity"], 0.1)
        self.assertLess(block["latent_mass_w1_gev_vs_identity"], 0.1)

    def test_worse_than_nothing_exceeds_one(self):
        rng = np.random.default_rng(12)
        worse = _pairs(rng, 4000, 0.060)  # broader than the input it was given
        block = self._block(worse)
        self.assertGreater(block["latent_mass_ks_vs_identity"], 1.0)

    def test_headroom_reports_a_discriminating_setup(self):
        block = self._block(self.data)
        self.assertGreater(block["latent_mass_ks_headroom"], 5.0)

    def test_low_headroom_setup_refuses_to_certify(self):
        """Prior and data nearly identical: the no-op is indistinguishable."""
        rng = np.random.default_rng(3)
        pool = _pairs(rng, 40000, 0.030)
        real = pool[:4000]
        near_identical = _pairs(rng, 4000, 0.030)
        block = reference_block(
            "latent",
            real_values=real,
            identity_values=near_identical,
            reference_pool=pool[4000:],
            model_metrics=_metrics("latent", real, near_identical),
            mass_fn=_mass,
            seed=7,
        )
        self.assertLess(block["latent_mass_ks_headroom"], 3.0)

    def test_floor_uses_disjoint_draws_when_the_pool_allows(self):
        block = self._block(self.data)
        self.assertEqual(block["latent_floor_disjoint"], 1.0)
        self.assertEqual(block["latent_floor_scale"], 1.0)
        self.assertEqual(block["latent_floor_events"], 4000.0)

    def test_floor_falls_back_and_flags_the_rescaling(self):
        block = reference_block(
            "latent",
            real_values=self.real,
            identity_values=self.data,
            reference_pool=self.pool[4000:9000],  # only 5000 for n = 4000
            model_metrics=_metrics("latent", self.real, self.data),
            mass_fn=_mass,
            seed=7,
        )
        self.assertEqual(block["latent_floor_disjoint"], 0.0)
        self.assertAlmostEqual(block["latent_floor_scale"], math.sqrt(2500 / 4000))


if __name__ == "__main__":
    unittest.main()
