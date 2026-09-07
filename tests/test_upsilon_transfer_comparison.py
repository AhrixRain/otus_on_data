"""Statistical behavior of the held-out transfer comparison."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_joint" / "upsilon"))
from compare_transfer_priors import score_cell, select_observables


class TestTransferComparison(unittest.TestCase):
    def test_resolves_improvement_and_degradation_against_same_floor(self):
        rng = np.random.default_rng(19)
        def sample(shift, n=2000):
            return {"mass": rng.normal(10 + shift, 0.1, n),
                    "pair_pt": rng.normal(5 + shift, 0.1, n)}
        real = sample(0, 6000)
        arms = {
            "improved": {"identity": sample(0.5), "decoded": sample(0)},
            "degraded": {"identity": sample(0.5), "decoded": sample(1)},
        }
        result = score_cell(real, arms, 7)
        self.assertEqual(result["events_per_distribution"], 2000)
        a = result["arms"]["improved"]["mass"]["w1"]
        b = result["arms"]["degraded"]["mass"]["w1"]
        self.assertEqual(a["floor"], b["floor"])
        self.assertLess(a["vs_identity"], 0.1)
        self.assertGreater(b["vs_identity"], 1.5)
        self.assertEqual(result, score_cell(real, arms, 7))

    def test_sparse_bin_is_not_scored(self):
        obs = {"mass": np.arange(10.), "pair_pt": np.arange(10.)}
        result = score_cell(obs, {"arm": {"identity": obs, "decoded": obs}}, 3)
        self.assertEqual(result["status"], "insufficient_events")
        self.assertNotIn("arms", result)

    def test_bins_are_half_open_on_each_distributions_own_momentum(self):
        # Back-to-back transverse momenta give masses inside the analysis window.
        z = np.array([[7.5, 0, 0, 7.5, -2.5, 0, 0, 2.5],
                      [5, 0, 0, 5, -5, 0, 0, 5]], dtype=float)
        # Both masses pass; the first event lies exactly on the 5 GeV pT boundary.
        self.assertEqual(len(select_observables(z, 0, 5)["mass"]), 1)
        self.assertEqual(len(select_observables(z, 5, 10)["mass"]), 1)
        self.assertEqual(len(select_observables(z, 10, None)["mass"]), 0)


if __name__ == "__main__":
    unittest.main()
