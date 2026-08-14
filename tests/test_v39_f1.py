"""v3.9 F1-restricted data-path correctness tests.

Covers the two E1-driven data changes introduced for the v3.9 pilot:
  - scripts/cms_data.filter_theory_prior (theory_prior_selection)
  - optional muon_pt_max junk-tail guard in the muon loader
All tests use synthetic arrays; no ROOT/HDF5 source data is required.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import cms_data  # noqa: E402
from cms_data import filter_theory_prior, load_config, resolve_config  # noqa: E402


def _row(px1, py1, pz1, e1, px2, py2, pz2, e2):
    return np.array([px1, py1, pz1, e1, px2, py2, pz2, e2], dtype=np.float64)


class TestFilterTheoryPrior(unittest.TestCase):
    def test_no_selection_returns_input_unchanged(self) -> None:
        z = np.zeros((4, 8), dtype=np.float64)
        self.assertIs(filter_theory_prior(z, None), z)
        self.assertIs(filter_theory_prior(z, {}), z)

    def test_bad_shape_rejected(self) -> None:
        with self.assertRaises(ValueError):
            filter_theory_prior(np.zeros((4, 7)), {"muon_pt_min": 1.0})

    def test_zero_survivors_rejected(self) -> None:
        z = np.stack([_row(1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0)] * 4)
        with self.assertRaises(ValueError):
            filter_theory_prior(z, {"muon_pt_min": 50.0})

    def test_pt_and_eta_cuts_applied_per_muon(self) -> None:
        # mu- pT 4, eta 2 (|eta|<2.4 passes); mu+ pT 2.5 (fails pT>3), eta 0.
        z = np.stack([
            _row(4.0, 0.0, 0.0, 4.0, 2.5, 0.0, 0.0, 2.5),  # fails pt
            _row(3.5, 0.0, 3.5 * np.sinh(3.0), 0.0, 3.5, 0.0, 0.0, 0.0),  # bad eta (eta=3)
            _row(3.5, 0.0, 0.0, 3.5, 3.5, 0.0, 0.0, 3.5),  # passes
        ])
        out = filter_theory_prior(
            z, {"muon_pt_min": 3.0, "muon_abs_eta_max": 2.4}
        )
        self.assertEqual(len(out), 1)
        np.testing.assert_array_equal(out[0], z[2])

    def test_mass_window_uses_stored_energy_authority(self) -> None:
        # Same 3-momenta; only the stored energies differ. The z-space mass
        # convention trusts the stored E, so an off-shell E pair must fall
        # outside the window even when the p-based mass is inside.
        z = np.stack([
            _row(1.5, 0.0, 0.0, 1.55, -1.5, 0.0, 0.0, 1.55),   # on-shell ~3.1
            _row(1.5, 0.0, 0.0, 20.0, -1.5, 0.0, 0.0, 20.0),   # E-based m ~40
        ])
        out = filter_theory_prior(
            z, {"mass_min": 3.0369, "mass_max": 3.1569}
        )
        self.assertEqual(len(out), 1)
        np.testing.assert_array_equal(out[0], z[0])

    def test_matches_direct_numpy_reference(self) -> None:
        rng = np.random.default_rng(7)
        z = rng.normal(size=(200, 8))
        z[:, 3] = np.abs(z[:, 3]) + 0.1
        z[:, 7] = np.abs(z[:, 7]) + 0.1
        sel = {"muon_pt_min": 0.5, "muon_abs_eta_max": 1.5,
               "mass_min": 1.0, "mass_max": 5.0}
        out = filter_theory_prior(z, sel)
        # Reference computation (massless daughters, stored-E mass).
        px1, py1, pz1, e1 = z[:, 0], z[:, 1], z[:, 2], z[:, 3]
        px2, py2, pz2, e2 = z[:, 4], z[:, 5], z[:, 6], z[:, 7]
        pt = np.hypot(np.concatenate([px1, px2]), np.concatenate([py1, py2]))
        pabs = np.sqrt(px1**2 + py1**2 + pz1**2)
        keep = (np.hypot(px1, py1) > 0.5) & (np.hypot(px2, py2) > 0.5)
        eta = np.arctanh(np.clip(np.concatenate([pz1, pz2]) / np.concatenate([pabs, np.sqrt(px2**2 + py2**2 + pz2**2)]), -0.999999, 0.999999))
        keep &= (np.abs(eta[:200]) < 1.5) & (np.abs(eta[200:]) < 1.5)
        m2 = (e1 + e2) ** 2 - ((px1 + px2) ** 2 + (py1 + py2) ** 2 + (pz1 + pz2) ** 2)
        mass = np.sqrt(np.maximum(m2, 0.0))
        keep &= (mass > 1.0) & (mass < 5.0)
        self.assertEqual(len(out), int(keep.sum()))
        np.testing.assert_array_equal(out, z[keep])


class TestMuonPtMax(unittest.TestCase):
    def _selection(self, with_max: bool) -> dict:
        sel = {
            "muon_pt_min": 1.0,
            "muon_abs_eta_max": 2.4,
            "jpsi_mass_min": 3.0,
            "jpsi_mass_max": 3.2,
        }
        if with_max:
            sel["muon_pt_max"] = 100.0
        return sel

    def _fake_arrays(self) -> dict:
        import awkward as ak

        # Event 0: back-to-back pT 1.55 pair -> m ~ 3.107 (in window).
        # Event 1: back-to-back pT 4.0 pair -> m ~ 8 (out of window).
        # Event 2: junk pair, pT 100 each, dphi 0.0315 -> m ~ 3.15 (in window,
        #          only kept when the max cut is absent).
        return {
            "nMuon": ak.Array([2, 2, 2]),
            "Muon_pt": ak.Array([[1.55, 1.55], [4.0, 4.0], [100.0, 100.0]]),
            "Muon_eta": ak.Array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            "Muon_phi": ak.Array(
                [[0.0, np.pi], [0.0, np.pi], [0.0, 0.0315]]
            ),
            "Muon_mass": ak.Array([[0.1057, 0.1057]] * 3),
            "Muon_charge": ak.Array([[-1, 1], [-1, 1], [-1, 1]]),
        }

    def _load(self, with_max: bool) -> np.ndarray:
        arrays = self._fake_arrays()

        class FakeEvents:
            def keys(self):
                return set(arrays.keys())

            def iterate(self, branches, step_size=None, library=None):
                yield arrays

        class FakeRoot:
            def __getitem__(self, key):
                return FakeEvents()

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root_path = Path(tmp) / "fake.root"
            root_path.touch()
            with mock.patch("uproot.open", return_value=FakeRoot()):
                return cms_data.load_cms_muon_x_data(
                    root_path, self._selection(with_max)
                )

    def test_junk_pair_removed_by_max_cut(self) -> None:
        rows = self._load(with_max=True)
        self.assertEqual(len(rows), 1)
        pt1 = np.hypot(rows[0, 0], rows[0, 1])
        pt2 = np.hypot(rows[0, 4], rows[0, 5])
        self.assertAlmostEqual(pt1, 1.55, places=3)
        self.assertAlmostEqual(pt2, 1.55, places=3)

    def test_junk_pair_kept_without_max_cut(self) -> None:
        rows = self._load(with_max=False)
        self.assertEqual(len(rows), 2)
        pts = np.sort(
            [np.hypot(r[0], r[1]) for r in rows] + [np.hypot(r[4], r[5]) for r in rows]
        )
        self.assertAlmostEqual(pts[0], 1.55, places=3)
        self.assertAlmostEqual(pts[-1], 100.0, places=3)


class TestV39Config(unittest.TestCase):
    def test_config_resolves_with_prior_selection_and_pt_max(self) -> None:
        cfg = resolve_config(
            load_config(REPO_ROOT / "configs/cms_JpsiDoubleMuons_v3.9_F1_restricted.yaml")
        )
        self.assertEqual(cfg["muon_selection"]["muon_pt_max"], 100.0)
        self.assertEqual(
            cfg["theory_prior_selection"],
            {
                "muon_pt_min": 3.0,
                "muon_abs_eta_max": 2.4,
                "mass_min": 3.0369,
                "mass_max": 3.1569,
            },
        )
        self.assertEqual(
            cfg["muon_selection"]["jpsi_mass_min"],
            cfg["theory_prior_selection"]["mass_min"],
        )
        self.assertEqual(
            cfg["muon_selection"]["jpsi_mass_max"],
            cfg["theory_prior_selection"]["mass_max"],
        )
        # Vanilla two-term objective inherited verbatim from v3.8.
        self.assertTrue(cfg["loss"]["vanilla_swae"])
        self.assertFalse(cfg["loss"]["standardize_raw_matching"])
        self.assertEqual(cfg["stages"][0]["epochs"], 300)

    def test_cache_metadata_records_prior_selection(self) -> None:
        cfg = resolve_config(
            load_config(REPO_ROOT / "configs/cms_JpsiDoubleMuons_v3.9_F1_restricted.yaml")
        )
        meta = cms_data.data_cache_metadata(cfg, None)
        self.assertIn("theory_prior_selection", meta)
        self.assertEqual(meta["theory_prior_selection"]["muon_pt_min"], 3.0)


if __name__ == "__main__":
    unittest.main()
