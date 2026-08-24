"""Small correctness tests for the CMS upsilon plotting code."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

import plot_upsilon


class UpsilonReconstructionTests(unittest.TestCase):
    def test_invariant_mass_for_back_to_back_muons(self) -> None:
        muon_mass = 0.105658
        target_mass = 10.0
        momentum = math.sqrt((target_mass / 2.0) ** 2 - muon_mass**2)
        mass = plot_upsilon.invariant_mass(
            *(
                np.array([value])
                for value in (target_mass / 2, momentum, 0, 0, target_mass / 2, -momentum, 0, 0)
            )
        )
        self.assertAlmostEqual(float(mass[0]), target_mass, places=10)

    def test_csv_loader_reconstructs_four_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            csv_path = Path(temporary_directory) / "dimuons.csv"
            csv_path.write_text(
                "Run,Event,type1,E1,px1,py1,pz1,Q1,type2,E2,px2,py2,pz2,Q2\n"
                "1,2,G,5.0,4.9988836,0,0,-1,G,5.0,-4.9988836,0,0,1\n",
                encoding="utf-8",
            )
            masses = plot_upsilon.load_csv_masses(csv_path)
        self.assertEqual(len(masses), 1)
        self.assertAlmostEqual(float(masses[0]), 10.0, places=6)

    def test_root_loader_applies_exactly_two_opposite_sign_selection(self) -> None:
        try:
            import awkward as ak
            import uproot
        except ImportError:
            self.skipTest("awkward and uproot are required for the ROOT-input test")

        muon_mass = 0.105658
        target_mass = 9.46
        momentum = math.sqrt((target_mass / 2.0) ** 2 - muon_mass**2)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root_path = Path(temporary_directory) / "dimuons.root"
            with uproot.recreate(root_path) as root_file:
                root_file["Events"] = {
                    "nMuon": np.array([2, 2], dtype=np.uint32),
                    "Muon_pt": ak.Array([[momentum, momentum], [momentum, momentum]]),
                    "Muon_eta": ak.Array([[0.0, 0.0], [0.0, 0.0]]),
                    "Muon_phi": ak.Array([[0.0, math.pi], [0.0, math.pi]]),
                    "Muon_mass": ak.Array([[muon_mass, muon_mass], [muon_mass, muon_mass]]),
                    "Muon_charge": ak.Array([[-1, 1], [1, 1]]),
                }
            masses = plot_upsilon.load_root_masses(root_path, step_size="1 MB")

        self.assertEqual(len(masses), 1)
        self.assertAlmostEqual(float(masses[0]), target_mass, places=5)


if __name__ == "__main__":
    unittest.main()
