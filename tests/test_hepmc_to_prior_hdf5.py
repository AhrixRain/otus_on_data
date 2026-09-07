"""Unit tests for scripts/prior_build/hepmc_to_prior_hdf5.py.

The fixture is a synthetic HepMC2 ASCII file built by hand in this file, not a
generated one, so the tests do not depend on MG5, Pythia or any sample being
present.  It exercises the four cases that matter:

  event 0  final-state QED radiation, so born, bare and dressed all differ
  event 1  no radiation, so all three variants coincide
  event 2  three status-1 muons, which must be rejected on multiplicity
  event 3  a soft muon that the generation selection must remove

Run with:  python -m unittest tests.test_hepmc_to_prior_hdf5 -v
(unittest, not pytest -- pytest is not in the `cms` environment, CLAUDE.md s6.)
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = REPO_ROOT / "scripts" / "prior_build" / "hepmc_to_prior_hdf5.py"
_spec = importlib.util.spec_from_file_location("hepmc_to_prior_hdf5", _MODULE_PATH)
conv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conv)

MU = conv.DEFAULT_MUON_MASS


def on_shell(px, py, pz, mass=MU):
    """Energy that puts a particle exactly on its mass shell."""
    return math.sqrt(px * px + py * py + pz * pz + mass * mass)


# --- the four events, defined once here so the tests can assert against them ---

# event 0: mu- radiates.  born -> (bare, photon); all three variants differ.
BORN_MINUS = (3.00, 0.00, 1.00)
BARE_MINUS = (2.60, 0.00, 0.85)
PHOTON = (0.30, 0.00, 0.10)          # inside dR < 0.1 of BARE_MINUS
DRESSED_MINUS = tuple(a + b for a, b in zip(BARE_MINUS, PHOTON))
PLUS_0 = (-2.80, 0.50, -0.60)

# event 1: no radiation at all.
MINUS_1 = (4.10, 0.30, 2.00)
PLUS_1 = (-3.90, -0.20, -1.10)

# event 3: mu- is too soft for a generation pT cut of 2.0.
MINUS_3 = (1.10, 0.05, 0.40)
PLUS_3 = (-3.10, 0.20, -0.90)


def _p(barcode, pdg, mom, status, end_vtx, mass=MU, energy=None):
    px, py, pz = mom
    e = on_shell(px, py, pz, mass) if energy is None else energy
    return (f"P {barcode} {pdg} {px:.16e} {py:.16e} {pz:.16e} {e:.16e} "
            f"{mass:.16e} {status} 0 0 {end_vtx} 0\n")


def _event_header(number, n_vertices, weight):
    # E num mpi scale aQCD aQED sig_proc sig_vtx n_vtx beam1 beam2
    #   n_random n_weights [weights]
    return (f"E {number} -1 -1.0 -1.0 -1.0 0 0 {n_vertices} 1 2 0 "
            f"3 {weight:.16e} {weight:.16e} {weight:.16e}\n"
            'N 3 "MUR0.5" "Weight" "MUR2.0" \n'
            "U GEV MM\n"
            f"C {1.0:.16e} {0.1:.16e}\n")


def build_fixture(path, gzipped=True):
    """Write the synthetic HepMC2 file.  Weight index of "Weight" is 1."""
    text = ["HepMC::Version 2.06.09\n",
            "HepMC::IO_GenEvent-START_EVENT_LISTING\n"]

    # ---------------- event 0: FSR ----------------
    # -1 : hard vertex, produces the born mu- (ends at -2) and the mu+ (stable)
    # -2 : FSR vertex, born mu- in, bare mu- + photon out
    text.append(_event_header(0, 2, weight=2.5))
    text.append("V -1 0 0 0 0 0 0 2 0\n")
    text.append(_p(1, 13, BORN_MINUS, status=2, end_vtx=-2))
    text.append(_p(2, -13, PLUS_0, status=1, end_vtx=0))
    text.append("V -2 0 0 0 0 0 0 2 0\n")
    text.append(_p(3, 13, BARE_MINUS, status=1, end_vtx=0))
    text.append(_p(4, 22, PHOTON, status=1, end_vtx=0, mass=0.0))

    # ---------------- event 1: no radiation ----------------
    text.append(_event_header(1, 1, weight=4.0))
    text.append("V -1 0 0 0 0 0 0 2 0\n")
    text.append(_p(1, 13, MINUS_1, status=1, end_vtx=0))
    text.append(_p(2, -13, PLUS_1, status=1, end_vtx=0))

    # ---------------- event 2: three stable muons ----------------
    text.append(_event_header(2, 1, weight=1.0))
    text.append("V -1 0 0 0 0 0 0 3 0\n")
    text.append(_p(1, 13, MINUS_1, status=1, end_vtx=0))
    text.append(_p(2, -13, PLUS_1, status=1, end_vtx=0))
    text.append(_p(3, 13, (1.0, 1.0, 1.0), status=1, end_vtx=0))

    # ---------------- event 3: soft mu- ----------------
    text.append(_event_header(3, 1, weight=1.5))
    text.append("V -1 0 0 0 0 0 0 2 0\n")
    text.append(_p(1, 13, MINUS_3, status=1, end_vtx=0))
    text.append(_p(2, -13, PLUS_3, status=1, end_vtx=0))

    text.append("HepMC::IO_GenEvent-END_EVENT_LISTING\n")
    blob = "".join(text)
    if gzipped:
        with gzip.open(path, "wt") as fh:
            fh.write(blob)
    else:
        Path(path).write_text(blob)
    return path


class FixtureTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.hepmc = os.path.join(cls.tmp.name, "events.hepmc.gz")
        build_fixture(cls.hepmc)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()


class TestStreamingParser(FixtureTestCase):
    def test_reads_every_event(self):
        events = list(conv.stream_events(self.hepmc))
        self.assertEqual([e.number for e in events], [0, 1, 2, 3])

    def test_particle_counts_and_graph(self):
        events = list(conv.stream_events(self.hepmc))
        self.assertEqual(len(events[0].particles), 4)
        born = events[0].particles[1]
        bare = events[0].particles[3]
        self.assertEqual(born.status, 2)
        self.assertEqual(born.end_vtx, -2)
        self.assertEqual(bare.prod_vtx, -2, "bare mu- is produced at the FSR vertex")
        self.assertEqual(born.prod_vtx, -1, "born mu- is produced at the hard vertex")

    def test_weights_are_read_by_name_not_position(self):
        # "Weight" is the SECOND of three names, so a converter that took index 0
        # would silently pick the MUR0.5 variation instead of the nominal.
        events = list(conv.stream_events(self.hepmc))
        idx = conv._resolve_weight_index(self.hepmc, "Weight")
        self.assertEqual(idx, 1)
        self.assertAlmostEqual(events[0].weights[idx], 2.5)
        self.assertAlmostEqual(events[1].weights[idx], 4.0)

    def test_plain_and_gzipped_agree(self):
        plain = os.path.join(self.tmp.name, "events.hepmc")
        build_fixture(plain, gzipped=False)
        a = [e.number for e in conv.stream_events(self.hepmc)]
        b = [e.number for e in conv.stream_events(plain)]
        self.assertEqual(a, b)


class TestTruthVariants(FixtureTestCase):
    def _pair(self, event_index, variant):
        events = list(conv.stream_events(self.hepmc))
        pair, reason = conv.extract_pair(events[event_index], variant, 0.1)
        self.assertIsNone(reason)
        return pair

    def test_bare_takes_the_status_one_muon(self):
        (variant, bare) = self._pair(0, "bare")
        np.testing.assert_allclose(variant[0][:3], BARE_MINUS, rtol=0, atol=1e-12)
        self.assertEqual(variant, bare)

    def test_born_walks_back_through_the_fsr_vertex(self):
        (variant, bare) = self._pair(0, "born")
        np.testing.assert_allclose(variant[0][:3], BORN_MINUS, rtol=0, atol=1e-12)
        # and the bare pair is still returned alongside it, unchanged
        np.testing.assert_allclose(bare[0][:3], BARE_MINUS, rtol=0, atol=1e-12)

    def test_born_stops_when_the_parent_is_not_a_muon(self):
        # the mu+ never radiated, so its born and bare four-vectors coincide
        (variant, bare) = self._pair(0, "born")
        np.testing.assert_allclose(variant[1], bare[1], rtol=0, atol=1e-12)

    def test_dressed_recombines_the_photon(self):
        (variant, _) = self._pair(0, "dressed")
        np.testing.assert_allclose(variant[0][:3], DRESSED_MINUS, rtol=0, atol=1e-12)
        expected_e = on_shell(*BARE_MINUS) + on_shell(*PHOTON, mass=0.0)
        self.assertAlmostEqual(variant[0][3], expected_e, places=12)

    def test_dressed_leaves_a_muon_with_no_nearby_photon_alone(self):
        (variant, bare) = self._pair(0, "dressed")
        np.testing.assert_allclose(variant[1], bare[1], rtol=0, atol=1e-12)

    def test_dressing_cone_is_respected(self):
        # with a cone of zero no photon is ever recombined, so dressed == bare
        events = list(conv.stream_events(self.hepmc))
        (variant, bare), _ = conv.extract_pair(events[0], "dressed", 0.0)
        self.assertEqual(variant, bare)

    def test_all_three_variants_coincide_without_radiation(self):
        results = {v: self._pair(1, v)[0] for v in conv.TRUTH_VARIANTS}
        for variant in conv.TRUTH_VARIANTS:
            np.testing.assert_allclose(results[variant][0][:3], MINUS_1,
                                       rtol=0, atol=1e-12)

    def test_the_three_variants_actually_differ_when_there_is_radiation(self):
        # guards against a converter that silently returns `bare` for everything
        got = {v: self._pair(0, v)[0][0][0] for v in conv.TRUTH_VARIANTS}
        self.assertNotAlmostEqual(got["born"], got["bare"])
        self.assertNotAlmostEqual(got["born"], got["dressed"])
        self.assertNotAlmostEqual(got["bare"], got["dressed"])

    def test_bad_muon_multiplicity_is_rejected(self):
        events = list(conv.stream_events(self.hepmc))
        pair, reason = conv.extract_pair(events[2], "bare", 0.1)
        self.assertIsNone(pair)
        self.assertEqual(reason, "bad_stable_muon_multiplicity")


class TestMassConventionGate(unittest.TestCase):
    """Method book section 11 gate 2."""

    def test_on_shell_muons_pass(self):
        z = np.array([[3.0, 0.0, 1.0, on_shell(3.0, 0.0, 1.0),
                       -2.8, 0.5, -0.6, on_shell(-2.8, 0.5, -0.6)]])
        rel = abs(conv.pair_mass_stored(z)[0] - conv.pair_mass_momentum(z, MU)[0])
        self.assertLess(rel / conv.pair_mass_stored(z)[0], 1e-12)

    def test_massless_muons_fail_the_tolerance(self):
        # this is the "restrict card lost MM" failure the gate exists to catch
        z = np.array([[3.0, 0.0, 1.0, on_shell(3.0, 0.0, 1.0, 0.0),
                       -2.8, 0.5, -0.6, on_shell(-2.8, 0.5, -0.6, 0.0)]])
        stored = conv.pair_mass_stored(z)[0]
        momentum = conv.pair_mass_momentum(z, MU)[0]
        self.assertGreater(abs(stored - momentum) / stored, 1e-6)

    def test_converter_aborts_on_massless_muons(self):
        tmp = tempfile.TemporaryDirectory()
        path = os.path.join(tmp.name, "massless.hepmc")
        lines = ["HepMC::Version 2.06.09\n",
                 "HepMC::IO_GenEvent-START_EVENT_LISTING\n",
                 _event_header(0, 1, weight=1.0),
                 "V -1 0 0 0 0 0 0 2 0\n",
                 _p(1, 13, MINUS_1, status=1, end_vtx=0, mass=0.0),
                 _p(2, -13, PLUS_1, status=1, end_vtx=0, mass=0.0),
                 "HepMC::IO_GenEvent-END_EVENT_LISTING\n"]
        Path(path).write_text("".join(lines))
        out = os.path.join(tmp.name, "out.hdf5")
        with self.assertRaises(SystemExit) as ctx:
            conv.main(["--component", f"c:0:{path}", "--truth-variant", "bare",
                       "--out", out])
        self.assertIn("MASS CONVENTION CHECK FAILED", str(ctx.exception))
        tmp.cleanup()

    def test_gate_is_applied_to_bare_muons_so_dressed_does_not_trip_it(self):
        # A dressed muon is off the mass shell by construction.  The gate must
        # still pass, because it is a question about the record, not the output.
        tmp = tempfile.TemporaryDirectory()
        hepmc = build_fixture(os.path.join(tmp.name, "e.hepmc.gz"))
        out = os.path.join(tmp.name, "dressed.hdf5")
        conv.main(["--component", f"c:0:{hepmc}", "--truth-variant", "dressed",
                   "--out", out])
        import h5py
        with h5py.File(out, "r") as fh:
            self.assertEqual(fh["FDL/zData"].shape[1], 8)
        tmp.cleanup()


class TestEndToEnd(FixtureTestCase):
    def test_writes_the_section_10_schema(self):
        import h5py
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "prior.hdf5")
            manifest = os.path.join(tmp, "prior.json")
            conv.main(["--component", f"sig:0:{self.hepmc}",
                       "--component", f"cont:3:{self.hepmc}",
                       "--truth-variant", "bare", "--out", out,
                       "--manifest", manifest])
            with h5py.File(out, "r") as fh:
                z = fh["FDL/zData"][:]
                cid = fh["FDL/component_id"][:]
                w = fh["FDL/weight"][:]
                self.assertEqual(z.dtype, np.float32)
                self.assertEqual(cid.dtype, np.int8)
                self.assertEqual(z.shape, (6, 8))     # 3 good events x 2 components
                self.assertEqual(sorted(set(cid.tolist())), [0, 3])
                for key in ("truth_variant", "columns", "units", "generation_cuts",
                            "component_id_mapping", "component_counts",
                            "daughter_muon_mass_GeV", "components_are_mixed",
                            "reweighting_applied", "artificial_smearing_applied"):
                    self.assertIn(key, fh.attrs, f"missing attribute {key}")
                self.assertEqual(fh.attrs["truth_variant"], "bare")
                self.assertEqual(fh.attrs["components_are_mixed"], "false")
                # weights survive, and by name: 2.5, 4.0, 1.5 (event 2 rejected)
                np.testing.assert_allclose(sorted(set(w.tolist())), [1.5, 2.5, 4.0])
            payload = json.loads(Path(manifest).read_text())
            self.assertIn("sig", payload["per_component"])
            self.assertEqual(payload["per_component"]["sig"]["component_id"], 0)
            self.assertIn("prior_stats", payload["per_component"]["sig"])
            self.assertIn("hdf5_sha256", payload)

    def test_generation_selection_is_applied_and_analysis_selection_is_not(self):
        import h5py
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "prior.hdf5")
            conv.main(["--component", f"c:0:{self.hepmc}",
                       "--truth-variant", "bare", "--out", out,
                       "--gen-pt-min", "2.0"])
            with h5py.File(out, "r") as fh:
                z = fh["FDL/zData"][:]
                # event 3 has a 1.1 GeV muon and must be gone; events 0 and 1 stay
                self.assertEqual(len(z), 2)
                pt = np.hypot(z[:, 0], z[:, 1])
                self.assertTrue((pt > 2.0).all())
                cuts = json.loads(fh.attrs["generation_cuts"])
                self.assertEqual(cuts["lepton_pt_min_GeV"], 2.0)
                # and nothing analysis-like was baked in
                self.assertIn("offline choice", fh.attrs["selection"])

    def test_no_selection_by_default(self):
        import h5py
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "prior.hdf5")
            conv.main(["--component", f"c:0:{self.hepmc}",
                       "--truth-variant", "bare", "--out", out])
            with h5py.File(out, "r") as fh:
                self.assertEqual(len(fh["FDL/zData"][:]), 3)

    def test_cutflow_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "prior.hdf5")
            manifest = os.path.join(tmp, "prior.json")
            conv.main(["--component", f"c:0:{self.hepmc}",
                       "--truth-variant", "bare", "--out", out,
                       "--manifest", manifest, "--gen-pt-min", "2.0"])
            cf = json.loads(Path(manifest).read_text())["per_component"]["c"]["cutflow"]
            self.assertEqual(cf["events_processed"], 4)
            self.assertEqual(cf["events_bad_stable_muon_multiplicity"], 1)
            self.assertEqual(cf["events_rejected_by_generation_selection"], 1)
            self.assertEqual(cf["events_accepted"], 2)


class TestStatsBlock(unittest.TestCase):
    def test_matches_prior_stats_definitions(self):
        rng = np.random.default_rng(0)
        p = rng.normal(0, 5, size=(500, 3))
        q = rng.normal(0, 5, size=(500, 3))
        z = np.column_stack([p, np.sqrt((p ** 2).sum(1) + MU ** 2),
                             q, np.sqrt((q ** 2).sum(1) + MU ** 2)])
        block = conv.stats_block(z)
        # the invariants prior_stats.py asserts on
        self.assertEqual(block["events"], 500)
        for key in ("mass_mean_gev", "mass_std_mev", "mass_median_gev",
                    "mass_iqr_width_mev", "mass_std_stored_energy_mev",
                    "muon_pt_median_gev", "muon_pt_p99_gev",
                    "pair_pt_median_gev", "pair_pt_p99_gev", "abs_eta_median"):
            self.assertIn(key, block)
        # for exactly on-shell muons the two mass conventions must agree
        self.assertAlmostEqual(block["mass_std_mev"],
                               block["mass_std_stored_energy_mev"], places=6)

    def test_weighted_block_appears_only_with_weights(self):
        z = np.array([[3.0, 0, 1, on_shell(3, 0, 1), -3.0, 0, -1, on_shell(-3, 0, -1)]])
        self.assertNotIn("weighted", conv.stats_block(z))
        self.assertIn("weighted", conv.stats_block(z, weights=np.array([2.0])))


if __name__ == "__main__":
    unittest.main()
