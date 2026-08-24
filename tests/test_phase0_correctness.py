"""Phase 0 correctness patch tests.

Covers semantic cache invalidation, the shared numerically stable invariant
mass implementations, strict loss-config validation, the unequal-cardinality
resonance W1, the empirical delta_eta normalization, and configurable daughter
masses with old-checkpoint compatibility. All tests use synthetic arrays; no
ROOT/HDF5 source data is required.
"""

from __future__ import annotations

import glob
import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import cms_data  # noqa: E402
from cms_data import (  # noqa: E402
    data_cache_key,
    data_cache_metadata,
    load_and_split_cached,
    load_config,
    resolve_config,
)
from cms_model import (  # noqa: E402
    build_model,
    checkpoint_payload,
    load_model_from_checkpoint,
)
from cms_training import build_loss_factory  # noqa: E402
from loss import (  # noqa: E402
    JPSI_DIMUON_LOSS_KIND,
    build_ee_physics_features,
    sliced_wasserstein,
    validate_loss_config,
)
from physics import (  # noqa: E402
    ELECTRON_MASS_GEV,
    MUON_MASS_GEV,
    invariant_mass_np,
    invariant_mass_torch,
)

try:
    from scipy.stats import wasserstein_distance

    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


MUON_MASSES = (MUON_MASS_GEV, MUON_MASS_GEV)


def make_pairs(n: int = 64, seed: int = 0, masses=MUON_MASSES) -> np.ndarray:
    """Random low-energy pairs with consistent [p4-, p4+] ordering."""
    rng = np.random.default_rng(seed)
    momenta = rng.normal(size=(n, 6)) * 5.0
    e1 = np.sqrt(
        momenta[:, 0] ** 2
        + momenta[:, 1] ** 2
        + momenta[:, 2] ** 2
        + masses[0] ** 2
    )
    e2 = np.sqrt(
        momenta[:, 3] ** 2
        + momenta[:, 4] ** 2
        + momenta[:, 5] ** 2
        + masses[1] ** 2
    )
    return np.concatenate(
        [momenta[:, :3], e1[:, None], momenta[:, 3:], e2[:, None]],
        axis=1,
    )


def make_boosted_pairs(n: int = 256, pt: float = 3500.0, target: float = 3.0969) -> np.ndarray:
    """Nearly collinear multi-TeV muon pairs spanning the J/psi mass scale."""
    rng = np.random.default_rng(42)
    dphi = rng.uniform(-1.0, 1.0, size=n) * (target / pt)
    px1 = np.full(n, pt, dtype=np.float64)
    py1 = np.zeros(n)
    pz1 = np.zeros(n)
    px2 = pt * np.cos(dphi)
    py2 = pt * np.sin(dphi)
    pz2 = np.zeros(n)
    e = np.sqrt(pt**2 + MUON_MASS_GEV**2)
    return np.stack(
        [px1, py1, pz1, np.full(n, e), px2, py2, pz2, np.full(n, e)],
        axis=1,
    )


def float64_reference(pairs: np.ndarray, masses) -> np.ndarray:
    values = pairs.astype(np.float64)
    e1 = np.sqrt(
        values[:, 0] ** 2 + values[:, 1] ** 2 + values[:, 2] ** 2 + masses[0] ** 2
    )
    e2 = np.sqrt(
        values[:, 4] ** 2 + values[:, 5] ** 2 + values[:, 6] ** 2 + masses[1] ** 2
    )
    mass2 = (e1 + e2) ** 2 - (
        (values[:, 0] + values[:, 4]) ** 2
        + (values[:, 1] + values[:, 5]) ** 2
        + (values[:, 2] + values[:, 6]) ** 2
    )
    return np.sqrt(np.maximum(mass2, 0.0))


class TestSharedInvariantMass(unittest.TestCase):
    def test_low_energy_agrees_with_float64_reference(self) -> None:
        pairs = make_pairs(256)
        for masses in (MUON_MASSES, (ELECTRON_MASS_GEV, ELECTRON_MASS_GEV)):
            with self.subTest(masses=masses):
                stable = invariant_mass_np(pairs, daughter_masses=masses)
                reference = float64_reference(pairs, masses)
                np.testing.assert_allclose(stable, reference, rtol=1e-12, atol=1e-12)

    def test_massless_agrees_with_float64_reference(self) -> None:
        pairs = make_pairs(256, masses=(0.0, 0.0))
        stable = invariant_mass_np(pairs, daughter_masses=None)
        reference = float64_reference(pairs, (0.0, 0.0))
        np.testing.assert_allclose(stable, reference, rtol=1e-12, atol=1e-12)

    def test_boosted_pairs_keep_jpsi_scale_mass(self) -> None:
        pairs = make_boosted_pairs()
        mass = invariant_mass_np(pairs, daughter_masses=MUON_MASSES)
        self.assertGreaterEqual(float(mass.min()), 2.0 * MUON_MASS_GEV - 1e-9)
        self.assertLess(float(mass.max()), 3.0970)
        self.assertTrue(np.all(mass > 0.0))

    def test_stable_beats_direct_float32_subtraction(self) -> None:
        pt = 3500.0
        dphi = 2.0 * np.arcsin(3.0969 / (2.0 * pt))
        energy = np.sqrt(pt**2 + MUON_MASS_GEV**2)
        pairs = np.array(
            [
                [pt, 0.0, 0.0, energy, pt * np.cos(dphi), pt * np.sin(dphi), 0.0, energy]
            ],
            dtype=np.float64,
        )
        reference = float(float64_reference(pairs, MUON_MASSES)[0])
        stable = float(invariant_mass_np(pairs, daughter_masses=MUON_MASSES)[0])
        f32 = pairs.astype(np.float32)
        p4 = f32[:, 0:4].astype(np.float64) + f32[:, 4:8].astype(np.float64)
        mass2 = p4[:, 3] ** 2 - (p4[:, 0] ** 2 + p4[:, 1] ** 2 + p4[:, 2] ** 2)
        direct = float(np.sqrt(np.maximum(mass2, 0.0))[0])
        # The float64 direct reference itself carries ~1e-9 cancellation noise
        # at this boost, so 1e-7 is the appropriate agreement bound; the
        # float32 direct subtraction is off by more than 1e-3.
        self.assertLess(abs(stable - reference), 1e-7)
        self.assertGreater(abs(direct - reference), 1e-3)

    def test_torch_matches_numpy_and_has_finite_gradients(self) -> None:
        pairs = make_pairs(128).astype(np.float32)
        np_mass = invariant_mass_np(pairs, daughter_masses=MUON_MASSES)
        tensor = torch.tensor(pairs, requires_grad=True)
        torch_mass = invariant_mass_torch(tensor, daughter_masses=MUON_MASSES)
        self.assertTrue(torch.isfinite(torch_mass).all())
        np.testing.assert_allclose(
            torch_mass.detach().numpy(),
            np_mass,
            rtol=1e-3,
            atol=1e-3,
        )
        torch_mass.sum().backward()
        self.assertIsNotNone(tensor.grad)
        self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_torch_massless_matches_numpy(self) -> None:
        pairs = make_pairs(64, masses=(0.0, 0.0)).astype(np.float32)
        np_mass = invariant_mass_np(pairs, daughter_masses=None)
        torch_mass = invariant_mass_torch(torch.as_tensor(pairs), daughter_masses=None)
        np.testing.assert_allclose(
            torch_mass.detach().numpy(),
            np_mass,
            rtol=1e-3,
            atol=1e-3,
        )

    @unittest.skipUnless(
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available(),
        "MPS not available",
    )
    def test_mps_smoke_without_float64(self) -> None:
        pairs = torch.tensor(
            make_pairs(32).astype(np.float32),
            device="mps",
            requires_grad=True,
        )
        mass = invariant_mass_torch(pairs, daughter_masses=MUON_MASSES)
        self.assertEqual(mass.dtype, torch.float32)
        self.assertTrue(torch.isfinite(mass).all())
        mass.sum().backward()
        self.assertTrue(torch.isfinite(pairs.grad).all())


class TestSemanticCacheInvalidation(unittest.TestCase):
    def make_config(self, tmp: Path) -> dict:
        root_file = tmp / "cms.root"
        prior_file = tmp / "prior.hdf5"
        root_file.write_bytes(b"root")
        prior_file.write_bytes(b"prior")
        return {
            "paths": {
                "output_root": str(tmp / "outputs"),
                "cms_root_file": str(root_file),
                "theory_prior_file": str(prior_file),
            },
            "data": {"channel": "electron"},
            "electron_selection": {
                "electron_pt_min": 20.0,
                "z_mass_min": 70.0,
                "z_mass_max": 110.0,
            },
            "float_type": "float32",
            "seed": 0,
            "data_split": {"train_ratio": 0.8, "val_ratio": 0.1},
        }

    @staticmethod
    def z_like_arrays(n: int = 4) -> dict[str, np.ndarray]:
        """Back-to-back 45 GeV electrons: pair mass 90 GeV, inside [70, 110]."""
        negative = np.zeros((n, 4), dtype=np.float32)
        negative[:, 2] = 45.0
        negative[:, 3] = 45.0
        positive = np.zeros((n, 4), dtype=np.float32)
        positive[:, 2] = -45.0
        positive[:, 3] = 45.0
        rows = np.concatenate([negative, positive], axis=1)
        return {key: rows.copy() for key in cms_data._EXPECTED_CACHE_KEYS}

    def key_for(self, config: dict, num_samples=None) -> str:
        return data_cache_key(data_cache_metadata(config, num_samples))

    def test_sample_cap_changes_key(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config = self.make_config(Path(td))
            self.assertNotEqual(self.key_for(config, None), self.key_for(config, 5))

    def test_selection_changes_key(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config = self.make_config(Path(td))
            first = self.key_for(config)
            config["electron_selection"]["electron_pt_min"] = 25.0
            self.assertNotEqual(first, self.key_for(config))

    def test_pipeline_version_changes_key(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config = self.make_config(Path(td))
            first = self.key_for(config)
            with mock.patch.object(cms_data, "DATA_PIPELINE_CACHE_VERSION", 999):
                self.assertNotEqual(first, self.key_for(config))

    def test_missing_metadata_forces_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config = self.make_config(Path(td))
            with mock.patch.object(
                cms_data,
                "load_and_split",
                return_value=self.z_like_arrays(),
            ) as loader:
                _arrays, info1 = load_and_split_cached(config, None, None, True)
                metadata_path = Path(info1["path"]).with_suffix(".json")
                metadata_path.unlink()
                _arrays, info2 = load_and_split_cached(config, None, None, True)
            self.assertEqual(info1["status"], "miss")
            self.assertEqual(info2["status"], "metadata_invalid")
            self.assertEqual(loader.call_count, 2)

    def test_altered_metadata_forces_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config = self.make_config(Path(td))
            with mock.patch.object(
                cms_data,
                "load_and_split",
                return_value=self.z_like_arrays(),
            ) as loader:
                _arrays, info1 = load_and_split_cached(config, None, None, True)
                metadata_path = Path(info1["path"]).with_suffix(".json")
                metadata_path.write_text(
                    json.dumps({"version": 999}),
                    encoding="utf-8",
                )
                _arrays, info2 = load_and_split_cached(config, None, None, True)
            self.assertEqual(info2["status"], "metadata_invalid")
            self.assertEqual(loader.call_count, 2)

    def test_malformed_array_forces_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config = self.make_config(Path(td))
            with mock.patch.object(
                cms_data,
                "load_and_split",
                return_value=self.z_like_arrays(),
            ) as loader:
                _arrays, info1 = load_and_split_cached(config, None, None, True)
                cache_path = Path(info1["path"])
                np.savez(
                    cache_path.with_name(cache_path.name + ".tmp.npz"),
                    x_train=np.zeros((3, 7), dtype=np.float32),
                )
                cache_path.with_name(cache_path.name + ".tmp.npz").replace(cache_path)
                _arrays, info2 = load_and_split_cached(config, None, None, True)
            self.assertEqual(info2["status"], "content_invalid")
            self.assertEqual(loader.call_count, 2)

    def test_out_of_window_x_forces_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config = self.make_config(Path(td))
            arrays = self.z_like_arrays()
            bad = np.zeros((4, 8), dtype=np.float32)
            for key in ("x_train", "x_val", "x_test"):
                arrays[key] = bad
            with mock.patch.object(cms_data, "load_and_split", return_value=arrays) as loader:
                _arrays, info1 = load_and_split_cached(config, None, None, True)
                _arrays, info2 = load_and_split_cached(config, None, None, True)
            self.assertEqual(info1["status"], "miss")
            self.assertEqual(info2["status"], "content_invalid")
            self.assertEqual(loader.call_count, 2)


class TestLossConfigValidation(unittest.TestCase):
    def test_all_checked_in_configs_validate_and_build(self) -> None:
        x = np.random.default_rng(0).normal(size=(16, 8)).astype(np.float32)
        z = np.random.default_rng(1).normal(size=(16, 8)).astype(np.float32)
        for path in sorted(glob.glob(str(REPO_ROOT / "configs" / "*.yaml"))):
            config = resolve_config(load_config(Path(path)))
            loss_config = config["loss"]
            with self.subTest(config=Path(path).name):
                validate_loss_config(loss_config)
                factory = build_loss_factory(
                    x,
                    z,
                    loss_config,
                    daughter_masses=(config.get("model") or {}).get("daughter_masses"),
                )
                self.assertIsNotNone(factory)

    def test_unknown_top_level_key_fails_with_hint(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            validate_loss_config(
                {"kind": JPSI_DIMUON_LOSS_KIND, "pair_masss_w1": 1.0}
            )
        message = str(ctx.exception)
        self.assertIn("pair_masss_w1", message)
        self.assertIn("pair_mass_w1", message)

    def test_unknown_space_override_fails(self) -> None:
        with self.assertRaises(ValueError):
            validate_loss_config(
                {
                    "kind": JPSI_DIMUON_LOSS_KIND,
                    "space_weights": {"z": {"mass_w1_typo": 1.0}},
                }
            )

    def test_unknown_mass_kin_component_fails(self) -> None:
        with self.assertRaises(ValueError):
            validate_loss_config(
                {
                    "kind": JPSI_DIMUON_LOSS_KIND,
                    "mass_kin_swd_components": {"mlll": 1.0},
                }
            )

    def test_unknown_selection_score_key_fails(self) -> None:
        with self.assertRaises(ValueError):
            validate_loss_config(
                {
                    "kind": JPSI_DIMUON_LOSS_KIND,
                    "selection_score": {"x_simm": 1.0},
                }
            )

    def test_invalid_values_fail_early(self) -> None:
        invalid_configs = [
            {"num_slices": 0},
            {"num_slices": 2.5},
            {"decoder_num_noise_samples": 0},
            {"eps": 0.0},
            {"eps": float("nan")},
            {"tail_frac": 1.5},
            {"resonance_mass_half_width": -1.0},
            {"mass_w1": float("nan")},
            {"mass_kin_swd_components": {"mll": -0.5}},
            {"selection_score": {"x_sim": float("inf")}},
        ]
        for bad in invalid_configs:
            with self.subTest(bad=bad):
                config = {"kind": JPSI_DIMUON_LOSS_KIND, **bad}
                with self.assertRaises(ValueError):
                    validate_loss_config(config)


class TestResonanceMassW1(unittest.TestCase):
    def make_space(self):
        x = make_pairs(32, seed=1).astype(np.float32)
        z = make_pairs(32, seed=2).astype(np.float32)
        return build_loss_factory(
            x,
            z,
            {"kind": JPSI_DIMUON_LOSS_KIND, "num_slices": 4},
            daughter_masses=MUON_MASSES,
        ).x_space

    def test_equal_counts_match_sorted_w1(self) -> None:
        space = self.make_space()
        center = space.resonance_mass_center
        truth = torch.tensor(
            [center - 0.02, center, center + 0.02],
            dtype=torch.float32,
        )
        pred = torch.tensor(
            [center - 0.03, center, center + 0.02],
            dtype=torch.float32,
        )
        actual = space.resonance_mass_w1(truth, pred)
        expected = space.wasserstein_1d_sorted(truth, pred)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    @unittest.skipUnless(HAS_SCIPY, "scipy not available")
    def test_unequal_counts_match_scipy_within_tolerance(self) -> None:
        space = self.make_space()
        center = space.resonance_mass_center
        rng = np.random.default_rng(3)
        truth = torch.as_tensor(
            center + rng.normal(0.0, 0.03, size=50).astype(np.float32)
        )
        pred = torch.as_tensor(
            center + rng.normal(0.0, 0.03, size=30).astype(np.float32)
        )
        truth_window = (truth >= center - space.resonance_mass_half_width) & (
            truth <= center + space.resonance_mass_half_width
        )
        pred_window = (pred >= center - space.resonance_mass_half_width) & (
            pred <= center + space.resonance_mass_half_width
        )
        actual = float(space.resonance_mass_w1(truth, pred))
        expected = float(
            wasserstein_distance(
                truth[truth_window].numpy(),
                pred[pred_window].numpy(),
            )
        )
        self.assertLessEqual(
            abs(actual - expected),
            0.05 * expected + 1e-3,
        )

    def test_symmetry_and_finite_gradients(self) -> None:
        space = self.make_space()
        center = space.resonance_mass_center
        truth = torch.tensor(
            [center - 0.03, center - 0.01, center, center + 0.01, center + 0.03],
            dtype=torch.float32,
        )
        pred = torch.tensor(
            [center - 0.02, center + 0.02],
            dtype=torch.float32,
            requires_grad=True,
        )
        forward = space.resonance_mass_w1(truth, pred)
        reverse = space.resonance_mass_w1(
            pred.detach(),
            truth,
        )
        self.assertTrue(torch.allclose(forward, reverse, atol=1e-6))
        forward.backward()
        self.assertIsNotNone(pred.grad)
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_empty_side_returns_zero(self) -> None:
        space = self.make_space()
        center = space.resonance_mass_center
        truth = torch.tensor([center], dtype=torch.float32)
        pred = torch.tensor([center - 1.0, center + 1.0], dtype=torch.float32)
        self.assertEqual(float(space.resonance_mass_w1(truth, pred)), 0.0)
        self.assertEqual(float(space.resonance_mass_w1(pred, truth)), 0.0)

    @unittest.skipUnless(HAS_SCIPY, "scipy not available")
    def test_general_unequal_w1_uses_both_complete_batches(self) -> None:
        space = self.make_space()
        truth = torch.tensor([0.0, 1.0, 2.0, 8.0, 13.0], dtype=torch.float32)
        pred = torch.tensor([1.0, 3.0], dtype=torch.float32, requires_grad=True)
        actual = space.wasserstein_1d_sorted(truth, pred)
        expected = wasserstein_distance(truth.numpy(), pred.detach().numpy())
        self.assertLessEqual(
            abs(float(actual.detach()) - expected), 0.05 * expected + 1e-3
        )
        actual.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())

    @unittest.skipUnless(HAS_SCIPY, "scipy not available")
    def test_unequal_one_dimensional_swd_uses_empirical_quantiles(self) -> None:
        truth = torch.tensor([[0.0], [1.0], [2.0], [8.0], [13.0]])
        pred = torch.tensor([[1.0], [3.0]], requires_grad=True)
        torch.manual_seed(7)
        actual = sliced_wasserstein(truth, pred, num_slices=1, p=1)
        expected = wasserstein_distance(truth[:, 0].numpy(), pred.detach()[:, 0].numpy())
        self.assertLessEqual(
            abs(float(actual.detach()) - expected), 0.05 * expected + 1e-3
        )
        actual.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())


class TestDeltaEtaNormalization(unittest.TestCase):
    def test_uses_empirical_delta_eta_std(self) -> None:
        x_train = make_pairs(128, seed=5).astype(np.float32)
        z_train = make_pairs(128, seed=6).astype(np.float32)
        factory = build_loss_factory(
            x_train,
            z_train,
            {"kind": JPSI_DIMUON_LOSS_KIND, "num_slices": 4},
            daughter_masses=MUON_MASSES,
        )
        space = factory.x_space
        work = x_train.astype(np.float64)
        pt1 = np.hypot(work[:, 0], work[:, 1])
        pt2 = np.hypot(work[:, 4], work[:, 5])
        eta1 = np.arcsinh(work[:, 2] / np.where(pt1 > 0, pt1, 1.0))
        eta2 = np.arcsinh(work[:, 6] / np.where(pt2 > 0, pt2, 1.0))
        expected = float(np.std(eta1 - eta2))
        self.assertAlmostEqual(space.delta_eta_std, expected, places=5)

        truth = torch.as_tensor(x_train[:32])
        pred = torch.as_tensor(x_train[32:64])
        component = float(space.distribution_components(truth, pred)["delta_eta_w1"])
        truth_deta = build_ee_physics_features(
            truth,
            space.eps,
            space.daughter_masses,
        )["delta_eta"]
        pred_deta = build_ee_physics_features(
            pred,
            space.eps,
            space.daughter_masses,
        )["delta_eta"]
        norm = torch.as_tensor(space.delta_eta_std) + space.eps
        manual = float(
            space.wasserstein_1d_sorted(truth_deta / norm, pred_deta / norm)
        )
        self.assertAlmostEqual(component, manual, places=5)


class TestConfigurableDaughterMasses(unittest.TestCase):
    def model_config(self, masses=None) -> dict:
        config = {
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
        if masses is not None:
            config["model"]["daughter_masses"] = list(masses)
        return config

    def test_legacy_config_stays_massless(self) -> None:
        model = build_model(
            self.model_config(),
            np.zeros(8, dtype=np.float32),
            np.ones(8, dtype=np.float32),
            np.zeros(8, dtype=np.float32),
            np.ones(8, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            np.asarray(model.encoder.inv_masses),
            np.zeros(2, dtype=np.float32),
        )

    def test_muon_mass_output_satisfies_energy_momentum(self) -> None:
        model = build_model(
            self.model_config(MUON_MASSES),
            np.zeros(8, dtype=np.float32),
            np.ones(8, dtype=np.float32),
            np.zeros(8, dtype=np.float32),
            np.ones(8, dtype=np.float32),
        )
        model.eval()
        with torch.no_grad():
            x = model.decode(torch.randn(64, 8))
        for start in (0, 4):
            px, py, pz, energy = x[:, start : start + 4].unbind(dim=1)
            residual = energy**2 - (px**2 + py**2 + pz**2)
            np.testing.assert_allclose(
                residual.numpy(),
                np.full(64, MUON_MASS_GEV**2),
                atol=1e-4,
            )

    def test_massless_output_stays_massless(self) -> None:
        model = build_model(
            self.model_config(),
            np.zeros(8, dtype=np.float32),
            np.ones(8, dtype=np.float32),
            np.zeros(8, dtype=np.float32),
            np.ones(8, dtype=np.float32),
        )
        model.eval()
        with torch.no_grad():
            x = model.decode(torch.randn(64, 8))
        for start in (0, 4):
            px, py, pz, energy = x[:, start : start + 4].unbind(dim=1)
            residual = energy**2 - (px**2 + py**2 + pz**2)
            np.testing.assert_allclose(residual.numpy(), np.zeros(64), atol=1e-4)

    def test_old_checkpoint_keeps_massless_semantics(self) -> None:
        stats = {
            "x_train_mean": np.zeros(8, dtype=np.float32),
            "x_train_std": np.ones(8, dtype=np.float32),
            "z_train_mean": np.zeros(8, dtype=np.float32),
            "z_train_std": np.ones(8, dtype=np.float32),
        }
        legacy_model = build_model(
            self.model_config(),
            stats["x_train_mean"],
            stats["x_train_std"],
            stats["z_train_mean"],
            stats["z_train_std"],
        )
        with tempfile.TemporaryDirectory() as td:
            checkpoint_path = Path(td) / "legacy.pt"
            torch.save(
                checkpoint_payload(
                    legacy_model,
                    self.model_config(),
                    stats,
                    1,
                    None,
                    {"selected": "cpu"},
                ),
                checkpoint_path,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                loaded, _model_config, _stats, _checkpoint = load_model_from_checkpoint(
                    checkpoint_path,
                    config=self.model_config(MUON_MASSES),
                    map_location=torch.device("cpu"),
                )
            self.assertTrue(any("daughter_masses" in str(w.message) for w in caught))
            np.testing.assert_array_equal(
                np.asarray(loaded.encoder.inv_masses),
                np.zeros(2, dtype=np.float32),
            )


class TestSpaceMassConvention(unittest.TestCase):
    def test_z_space_uses_stored_energy_authority(self) -> None:
        """Theory z-priors keep stored-E mass; detector x uses stable-from-p."""
        x = make_pairs(32, seed=11).astype(np.float32)
        # Stored E encodes a 3.0969 pair even though p is degenerate: the
        # direct formula recovers 3.0969 while a p-only reconstruction gives
        # 2*m_mu ~ 0.211 GeV.
        row = [0.0, 0.0, 0.0, 1.54845, 0.0, 0.0, 0.0, 1.54845]
        z = np.tile(np.asarray(row, dtype=np.float32), (32, 1))
        factory = build_loss_factory(
            x,
            z,
            {"kind": JPSI_DIMUON_LOSS_KIND, "num_slices": 4},
            daughter_masses=MUON_MASSES,
        )
        self.assertTrue(factory.z_space.mass_from_energy)
        self.assertFalse(factory.x_space.mass_from_energy)
        self.assertAlmostEqual(float(factory.z_space.mass_mean), 3.0969, places=5)

        named = build_ee_physics_features(
            torch.as_tensor(z),
            factory.z_space.eps,
            MUON_MASSES,
            mass_from_energy=True,
        )
        np.testing.assert_allclose(
            named["m_ee"].detach().numpy(),
            np.full(32, 3.0969),
            rtol=1e-4,
        )


if __name__ == "__main__":
    unittest.main()
