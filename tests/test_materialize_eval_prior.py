"""Tests for scripts/prior_build/materialize_eval_prior.py.

unittest, not pytest - pytest is not installed in the `cms` env (CLAUDE.md
section 6). The h5py end-to-end cases skip cleanly if h5py is missing.
"""

from __future__ import annotations

# NOTE: this suite runs on Windows in the `cms` conda env. TemporaryDirectory is
# opened with ignore_cleanup_errors=True because Windows refuses to remove a
# directory while any handle in it is still open, and HDF5 can hold one briefly
# after close. The tests assert on content, never on cleanup.

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prior_build.materialize_eval_prior import (  # noqa: E402
    _selection_mask,
    apply_signal_fraction,
    kish_ess,
    main,
    unweight_accept_reject,
)

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None


def make_pair(pt1, eta1, pt2, eta2, phi1=0.0, phi2=np.pi, mass_target=None):
    """One [8] row: mu- then mu+, (px, py, pz, E), massive muons."""
    m = 0.105658

    def vec(pt, eta, phi):
        px, py = pt * np.cos(phi), pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        return np.array([px, py, pz, np.sqrt(px * px + py * py + pz * pz + m * m)])

    a, b = vec(pt1, eta1, phi1), vec(pt2, eta2, phi2)
    return np.concatenate([a, b])


class TestSelectionMask(unittest.TestCase):
    def test_pt_and_eta_cuts(self):
        z = np.stack([
            make_pair(5.0, 0.1, 5.0, -0.1),    # passes
            make_pair(1.0, 0.1, 5.0, -0.1),    # fails pT
            make_pair(5.0, 3.0, 5.0, -0.1),    # fails eta
        ])
        keep = _selection_mask(z, {"muon_pt_min": 3.0, "muon_abs_eta_max": 2.4})
        self.assertEqual(list(keep), [True, False, False])

    def test_mass_window(self):
        z = np.stack([make_pair(5.0, 0.0, 5.0, 0.0), make_pair(50.0, 0.0, 50.0, 0.0)])
        mass = np.sqrt(np.maximum(
            (z[:, 3] + z[:, 7]) ** 2
            - ((z[:, 0] + z[:, 4]) ** 2 + (z[:, 1] + z[:, 5]) ** 2
               + (z[:, 2] + z[:, 6]) ** 2), 0.0))
        keep = _selection_mask(z, {"mass_min": mass[0] - 0.1, "mass_max": mass[0] + 0.1})
        self.assertEqual(list(keep), [True, False])

    def test_rejects_wrong_shape(self):
        with self.assertRaises(ValueError):
            _selection_mask(np.zeros((3, 6)), {"muon_pt_min": 1.0})


class TestWeights(unittest.TestCase):
    def test_kish_ess_uniform_is_n(self):
        self.assertAlmostEqual(kish_ess(np.ones(100)), 100.0)

    def test_kish_ess_degenerate(self):
        w = np.zeros(100)
        w[0] = 1.0
        self.assertAlmostEqual(kish_ess(w), 1.0)

    def test_accept_reject_never_duplicates(self):
        rng = np.random.default_rng(0)
        w = rng.uniform(0.5, 1.0, size=10_000)
        mask = unweight_accept_reject(w, rng)
        self.assertEqual(mask.dtype, np.bool_)
        self.assertEqual(len(mask), len(w))
        # a boolean mask cannot select a row twice - that is the whole point
        self.assertLessEqual(mask.sum(), len(w))

    def test_accept_reject_is_unbiased(self):
        rng = np.random.default_rng(1)
        w = rng.uniform(0.1, 1.0, size=200_000)
        x = rng.normal(size=200_000) + w  # x correlated with w
        mask = unweight_accept_reject(w, rng)
        weighted_mean = float((w * x).sum() / w.sum())
        self.assertAlmostEqual(weighted_mean, float(x[mask].mean()), places=1)

    def test_negative_weights_raise(self):
        with self.assertRaises(ValueError):
            unweight_accept_reject(np.array([1.0, -0.5]), np.random.default_rng(0))


class TestSignalFraction(unittest.TestCase):
    def test_hits_target_without_duplicating(self):
        rng = np.random.default_rng(2)
        cid = np.concatenate([np.zeros(6000, dtype=np.int8),
                              np.full(4000, 3, dtype=np.int8)])
        idx = apply_signal_fraction(cid, {3}, 0.25, rng)
        self.assertEqual(len(set(idx.tolist())), len(idx))
        achieved = float((cid[idx] != 3).mean())
        self.assertAlmostEqual(achieved, 0.25, places=2)

    def test_preserves_inter_signal_ratio(self):
        rng = np.random.default_rng(3)
        cid = np.concatenate([np.zeros(3000, dtype=np.int8),
                              np.ones(1500, dtype=np.int8),
                              np.full(5000, 3, dtype=np.int8)])
        idx = apply_signal_fraction(cid, {3}, 0.2, rng)
        kept = cid[idx]
        ratio = (kept == 0).sum() / max((kept == 1).sum(), 1)
        self.assertAlmostEqual(ratio, 2.0, delta=0.25)

    def test_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            apply_signal_fraction(np.array([0, 3]), {3}, 1.5, np.random.default_rng(0))


@unittest.skipIf(h5py is None, "h5py not installed")
class TestEndToEnd(unittest.TestCase):
    def _write_source(self, path: Path, n=20_000, seed=7):
        rng = np.random.default_rng(seed)
        rows, cids = [], []
        for i in range(n):
            signal = i % 2 == 0
            pt = rng.uniform(2.5, 20.0)
            rows.append(make_pair(pt, rng.uniform(-2.0, 2.0),
                                  pt * rng.uniform(0.8, 1.2), rng.uniform(-2.0, 2.0)))
            cids.append(0 if signal else 3)
        z = np.stack(rows).astype(np.float32)
        with h5py.File(path, "w") as out:
            g = out.create_group("FDL")
            g.create_dataset("zData", data=z)
            g.create_dataset("component_id", data=np.array(cids, dtype=np.int8))
            g.create_dataset("weight", data=rng.uniform(0.6, 1.0, size=n))
            out.attrs["component_id_mapping"] = json.dumps({"sig": 0, "continuum": 3})
            out.attrs["analysis_muon_pt_GeV"] = 3.0
            out.attrs["generation_cuts"] = json.dumps({
                "lepton_pt_min_GeV": 2.0, "lepton_abs_eta_max": 2.6,
                "mass_window_GeV": [0.0, 500.0]})
            out.attrs["analysis_window_legacy_GeV"] = json.dumps([0.5, 400.0])

    def test_runs_and_bakes_selection(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            src, dst = Path(tmp) / "src.hdf5", Path(tmp) / "out.hdf5"
            self._write_source(src)
            rc = main([
                "--in", str(src), "--out", str(dst), "--window", "legacy",
                "--signal-fraction", "0.3", "--min-events", "10",
                "--skip-cms-data-check",
            ])
            self.assertEqual(rc, 0)
            with h5py.File(dst, "r") as handle:
                z = np.asarray(handle["FDL/zData"])
                cid = np.asarray(handle["FDL/component_id"])
                self.assertNotIn("weight", handle["FDL"])  # unweighted away
                self.assertTrue(bool(handle.attrs["selection_baked_in"]))
                self.assertEqual(handle.attrs["unweighting"], "accept_reject")
                self.assertEqual(int(handle.attrs["duplicated_rows"]), 0)
            pt1 = np.hypot(z[:, 0], z[:, 1])
            self.assertTrue((pt1 > 3.0).all(), "muon pT cut not baked in")
            self.assertAlmostEqual(float((cid != 3).mean()), 0.3, places=1)
            self.assertTrue(dst.with_suffix(".json").exists())

    def test_refuses_window_wider_than_generation(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            src, dst = Path(tmp) / "src.hdf5", Path(tmp) / "out.hdf5"
            self._write_source(src)
            with self.assertRaises(SystemExit):
                main([
                    "--in", str(src), "--out", str(dst), "--window", "explicit",
                    "--mass-min", "0.0", "--mass-max", "600.0",
                    "--skip-cms-data-check",
                ])

    def test_refuses_pt_without_margin(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            src, dst = Path(tmp) / "src.hdf5", Path(tmp) / "out.hdf5"
            self._write_source(src)
            with self.assertRaises(SystemExit):
                main([
                    "--in", str(src), "--out", str(dst), "--window", "legacy",
                    "--muon-pt-min", "2.0", "--skip-cms-data-check",
                ])


if __name__ == "__main__":
    unittest.main()
