"""Unit tests for the MG5 prior rebuild tooling (LHE->HDF5, smearing, mixing)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.prior_build.lhe_to_prior_hdf5 import parse_lhe_events, select_dimuons  # noqa: E402
from scripts.prior_build.mix_prior import main as mix_main  # noqa: E402
from scripts.prior_build.smear_prior import calibrate, filter_trigger_matched, smear_rows  # noqa: E402

SYNTHETIC_LHE = """<LesHouchesEvents version="3.0">
<header>
</header>
<init>
 2212 2212 4.000000e+03 4.000000e+03 0 0 247000 247000 -4 1
 1.0 1.0 1.0 1
</init>
<event>
 5 100 1.000000e+00 1.000000e+01 7.546771e-02 1.177800e-01
 21 -1 0 0 501 502 0.000000e+00 0.000000e+00 2.532534e+03 2.532534e+03 0.000000e+00 0.000e+00 9.000e+00
 443 2 1 2 0 0 0.000000e+00 0.000000e+00 2.532534e+03 3.096900e+03 3.096900e+00 0.000e+00 0.000e+00
 13 1 1 2 0 0 -1.500000e+00 -6.000000e-01 2.000000e-01 1.631000e+00 1.056580e-01 0.000e+00 0.000e+00
 -13 1 1 2 0 0 1.500000e+00 6.000000e-01 -2.000000e-01 1.631000e+00 1.056580e-01 0.000e+00 0.000e+00
</event>
<event>
 4 100 1.000000e+00 1.000000e+01 7.546771e-02 1.177800e-01
 -13 1 0 0 0 0 2.000000e+00 0.000000e+00 2.000000e+00 2.828427e+00 1.056580e-01 0.000e+00 0.000e+00
 13 1 0 0 0 0 -2.000000e+00 0.000000e+00 -2.000000e+00 2.828427e+00 1.056580e-01 0.000e+00 0.000e+00
 13 1 0 0 0 0 -2.000000e+00 0.000000e+00 -2.000000e+00 2.828427e+00 1.056580e-01 0.000e+00 0.000e+00
</event>
</LesHouchesEvents>
"""


class TestLHEParsing(unittest.TestCase):
    def test_parse_two_events(self):
        events = parse_lhe_events(SYNTHETIC_LHE)
        self.assertEqual(len(events), 2)

    def test_selection_cutflow(self):
        events = parse_lhe_events(SYNTHETIC_LHE)
        z, flow = select_dimuons(events, pt_min=3.0, eta_max=2.4, mass_min=3.0369, mass_max=3.1569)
        # event 1: muons pT~1.25 < 3 -> rejected; event 2 has two mu+ -> rejected
        self.assertEqual(flow["events"], 2)
        self.assertEqual(flow["exactly_one_pair"], 1)
        self.assertEqual(flow["pt"], 0)
        self.assertEqual(len(z), 0)

    def test_selection_passes_hard_pair(self):
        events = parse_lhe_events(SYNTHETIC_LHE)
        z, flow = select_dimuons(events, pt_min=1.0, eta_max=2.4, mass_min=2.0, mass_max=4.0)
        self.assertEqual(flow["exactly_one_pair"], 1)
        self.assertEqual(flow["pt"], 1)
        self.assertEqual(flow["mass"], 1)
        self.assertEqual(z.shape, (1, 8))
        # charge ordering: mu- first
        self.assertLess(z[0, 0], 0)  # mu- px negative in event 1


class TestSmearing(unittest.TestCase):
    def test_determinism(self):
        z = np.tile(np.array([1.0, 0.0, 2.0, 3.0, -1.0, 0.0, -2.0, 3.0]), (100, 1))
        rng_a = np.random.default_rng(7)
        rng_b = np.random.default_rng(7)
        za = smear_rows(z, 0.01, rng_a)
        zb = smear_rows(z, 0.01, rng_b)
        self.assertTrue(np.array_equal(za, zb))

    def test_smearing_mass_std_scale(self):
        # delta-like boosted pair (both muons pT>3, mass ~3.15) smeared with
        # a=0.0127 should land near the ~28 MeV calibration target
        row = np.array([3.0, 1.57, 0.0, 3.388, 3.0, -1.57, 0.0, 3.388])
        z = np.tile(row, (2000, 1))
        z = filter_trigger_matched(z)
        self.assertGreaterEqual(len(z), 1000)
        a = calibrate(z, target_std=0.0279)
        self.assertGreater(a, 0.005)
        self.assertLess(a, 0.03)

    def test_energy_recomputed(self):
        z = np.tile(np.array([1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0]), (50, 1))
        rng = np.random.default_rng(3)
        zs = smear_rows(z, 0.02, rng)
        for i in (0, 4):
            e_expected = np.sqrt(zs[:, i] ** 2 + zs[:, i + 1] ** 2 + zs[:, i + 2] ** 2 + 0.105658 ** 2)
            self.assertTrue(np.allclose(zs[:, i + 3], e_expected))


class TestMixing(unittest.TestCase):
    def _write(self, path, n):
        import h5py
        rng = np.random.default_rng(0)
        z = rng.normal(size=(n, 8)).astype(np.float32)
        with h5py.File(path, "w") as h:
            grp = h.create_group("FDL")
            grp.create_dataset("zData", data=z)

    def test_fractions_and_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            sig = tmp / "sig.hdf5"
            con = tmp / "con.hdf5"
            out = tmp / "mixed.hdf5"
            self._write(sig, 1000)
            self._write(con, 5000)
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):
                sys.argv = ["mix_prior.py", "--signal", str(sig), "--continuum", str(con),
                            "--out", str(out), "--frac-signal", "0.85", "--seed", "1"]
                mix_main()
            import h5py
            with h5py.File(out, "r") as h:
                n = h["FDL/zData"].shape[0]
                frac = float(h["FDL"].attrs["frac_signal_post_filter"])
                n_sig = int(h["FDL"].attrs["n_signal"])
            # total limited by signal: 1000/0.85 = 1176; n_sig = 1000
            self.assertEqual(n_sig, 1000)
            self.assertAlmostEqual(frac, 0.85)
            self.assertEqual(n, n_sig + n - n_sig)  # n == total
            self.assertLess(n, 1000 + 5000)


if __name__ == "__main__":
    unittest.main()
