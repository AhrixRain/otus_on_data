"""Tests for the paired-bench quantile-calibration diagnostic (P2, paired half).

Two layers, matching the deliverable:

1. **pure statistics** - the rank/PIT uniformity property, the exact binomial
   band, central-interval coverage, the bootstrap band and the pull - pinned
   against synthetic posteriors whose calibration state is known by
   construction (a calibrated one, and one whose stated width is too small for
   its own error against the truth);
2. **the CLI** - one cheap end-to-end run on the real ppzee bench and
   checkpoint, asserting the JSON is complete and reproducible under a fixed
   seed. It skips (rather than fails) when the bench or the checkpoint is not on
   this machine.

The synthetic generator is the important part.  A conditional can only be
miscalibrated if the truth is *not* at the centre of its own interval relative
to the stated width, so the generator carries two independent scales: the
posterior-mean error ``error_rms`` and the stated width ``width``.  Calibrated
means ``error_rms == width``; the failure mode measured on ppzee is
``error_rms >> width``.
"""
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

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

from paired_quantile_calibration import (  # noqa: E402
    BAND_SIGMA,
    bootstrap_coverage_band,
    bootstrap_std_band,
    calibration_block,
    calibration_criteria,
    calibration_verdict,
    central_interval_coverage,
    derangement_indices,
    exact_bin_probabilities,
    finite_draw_pull_std,
    json_ready,
    ks_distance_to_normal,
    midrank_pit,
    null_calibration_reference,
    parse_checkpoint_argument,
    pull_statistics,
    pull_values,
    rank_histogram,
    select_subset,
)

CHECKPOINT = REPO_ROOT / "outputs" / "cms_Joint" / "ppzee" / "best_model.pt"
DATASET = (
    REPO_ROOT
    / "outputs"
    / "cms_Joint"
    / "ppzee"
    / "paired_closure"
    / "pairs_test.npz"
)
HAS_FIXTURES = CHECKPOINT.exists() and DATASET.exists()
SCRIPT = REPO_ROOT / "scripts_joint" / "paired_quantile_calibration.py"


def synthetic_posterior(
    events: int,
    draws: int,
    *,
    error_rms: float,
    width: float,
    truth_scale: float = 1.0,
    seed: int = 1234,
) -> tuple[np.ndarray, np.ndarray]:
    """Truth masses and draw masses with a known calibration state.

    ``draw_j = truth + error + N(0, width)`` where ``error ~ N(0, error_rms)``
    is the posterior-mean error shared by the draws of one event.  The pull is
    therefore ``(truth - mean) / std ~ -error / width``: unit width only when
    the stated width equals the actual error.
    """
    rng = np.random.default_rng(seed)
    mass_true = 91.0 + rng.normal(0.0, truth_scale, events)
    error = rng.normal(0.0, error_rms, events)
    noise = rng.normal(0.0, width, (draws, events))
    return mass_true, mass_true[None, :] + error[None, :] + noise


class DerangementTests(unittest.TestCase):
    def test_no_fixed_points_and_valid_permutation(self):
        for n in (2, 3, 7, 50, 1000):
            perm = derangement_indices(n, np.random.default_rng(n))
            self.assertEqual(perm.shape, (n,))
            self.assertEqual(sorted(perm.tolist()), list(range(n)))
            self.assertFalse(np.any(perm == np.arange(n)), msg=f"fixed point at n={n}")

    def test_deterministic_under_a_fixed_seed(self):
        first = derangement_indices(500, np.random.default_rng(7))
        second = derangement_indices(500, np.random.default_rng(7))
        third = derangement_indices(500, np.random.default_rng(8))
        self.assertTrue(np.array_equal(first, second))
        self.assertFalse(np.array_equal(first, third))

    def test_rejects_degenerate_sizes(self):
        for n in (0, 1, -3):
            with self.assertRaises(ValueError):
                derangement_indices(n, np.random.default_rng(1))


class SelectSubsetTests(unittest.TestCase):
    def test_all_events_when_not_subsampling(self):
        for requested in (0, -1, 1000, None):
            subset = select_subset(100, requested, np.random.default_rng(0))
            self.assertEqual(subset.tolist(), list(range(100)))

    def test_subset_is_sorted_unique_and_seeded(self):
        first = select_subset(1000, 100, np.random.default_rng(3))
        second = select_subset(1000, 100, np.random.default_rng(3))
        third = select_subset(1000, 100, np.random.default_rng(4))
        self.assertEqual(len(first), 100)
        self.assertEqual(len(set(first.tolist())), 100)
        self.assertTrue(np.all(np.diff(first) > 0))
        self.assertTrue(np.array_equal(first, second))
        self.assertFalse(np.array_equal(first, third))


class MidrankPitTests(unittest.TestCase):
    def test_rank_and_midpoint_extremes(self):
        truth = np.array([0.0, 10.0, 5.0])
        draws = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 6.0]])
        pit = midrank_pit(truth, draws)
        self.assertEqual(pit["rank"].tolist(), [0, 2, 1])
        expected = [(0 + 0.5) / 3.0, (2 + 0.5) / 3.0, (1 + 0.5) / 3.0]
        np.testing.assert_allclose(pit["u"], expected)

    def test_midpoints_are_strictly_inside_the_unit_interval(self):
        truth = np.array([0.0, 1.0, 2.0, 3.0])
        draws = np.arange(8, dtype=float)[:, None] + np.zeros((1, 4))
        pit = midrank_pit(truth, draws)
        self.assertTrue(np.all(pit["u"] > 0.0))
        self.assertTrue(np.all(pit["u"] < 1.0))

    def test_ties_are_counted_and_do_not_enter_the_rank(self):
        truth = np.array([2.0])
        draws = np.array([[2.0], [2.0], [1.0]])
        pit = midrank_pit(truth, draws)
        self.assertEqual(pit["rank"].tolist(), [1])  # only the 1.0 draw is below
        self.assertEqual(pit["ties"].tolist(), [2])

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            midrank_pit(np.zeros(3), np.zeros((4, 5)))


class ExactBinProbabilityTests(unittest.TestCase):
    def test_probabilities_sum_to_one(self):
        for bins, draws in ((5, 32), (11, 32), (33, 32), (20, 64)):
            probability = exact_bin_probabilities(bins, draws)
            self.assertAlmostEqual(float(probability.sum()), 1.0, places=12)
            self.assertTrue(np.all(probability > 0.0))

    def test_uniform_when_bins_divides_the_rank_count(self):
        probability = exact_bin_probabilities(11, 32)  # 33 ranks / 11 bins
        np.testing.assert_allclose(probability, np.full(11, 3.0 / 33.0))

    def test_all_ranks_separate_bins_are_uniform(self):
        probability = exact_bin_probabilities(33, 32)
        np.testing.assert_allclose(probability, np.full(33, 1.0 / 33.0))

    def test_more_bins_than_ranks_is_rejected(self):
        with self.assertRaises(ValueError):
            exact_bin_probabilities(34, 32)
        with self.assertRaises(ValueError):
            exact_bin_probabilities(0, 32)


class RankHistogramTests(unittest.TestCase):
    def test_flags_a_perfectly_uniform_histogram(self):
        draws = 32
        bins = 11
        atoms = np.arange(draws + 1, dtype=float)
        u = (atoms + 0.5) / (draws + 1)
        values = np.tile(u, 50)  # 1650 events, exactly uniform ranks
        histogram = rank_histogram(values, bins=bins, draws=draws)
        self.assertEqual(sum(histogram["counts"]), values.size)
        np.testing.assert_allclose(
            histogram["fractions"], histogram["expected_probability"], atol=1e-12
        )
        self.assertEqual(histogram["max_abs_deviation_sigma"], 0.0)
        self.assertLess(histogram["chi2"], 1e-9)
        self.assertGreater(histogram["chi2_p_value"], 0.99)

    def test_detects_a_shifted_histogram(self):
        draws = 32
        values = np.concatenate(
            [np.full(500, 0.02), np.full(500, 0.98)]
        )
        histogram = rank_histogram(values, bins=20, draws=draws)
        self.assertGreater(histogram["max_abs_deviation_sigma"], BAND_SIGMA)
        self.assertGreater(histogram["total_variation"], 0.5)
        self.assertIsNotNone(histogram["chi2_p_value"])
        self.assertLess(histogram["chi2_p_value"], 1e-6)

    def test_rejects_values_outside_the_open_unit_interval(self):
        with self.assertRaises(ValueError):
            rank_histogram(np.array([0.0, 0.5]), bins=2, draws=4)
        with self.assertRaises(ValueError):
            rank_histogram(np.array([1.0]), bins=2, draws=4)


class CoverageTests(unittest.TestCase):
    def test_calibrated_posterior_is_close_to_nominal_at_64_draws(self):
        truth, draws = synthetic_posterior(
            20000, 64, error_rms=0.5, width=0.5, seed=5
        )
        low = central_interval_coverage(truth, draws, 0.68)
        high = central_interval_coverage(truth, draws, 0.95)
        self.assertAlmostEqual(low["measured"], 0.68, delta=0.03)
        # 0.95 loses ~2-3 points to the inward bias of sampled quantiles.
        self.assertLess(high["measured"], 0.95 - 0.015)
        self.assertGreater(high["measured"], 0.90)

    def test_the_shortfall_grows_as_draws_shrink(self):
        coarse = central_interval_coverage(
            *synthetic_posterior(20000, 8, error_rms=0.5, width=0.5, seed=5), 0.68
        )
        fine = central_interval_coverage(
            *synthetic_posterior(20000, 64, error_rms=0.5, width=0.5, seed=5), 0.68
        )
        self.assertLess(coarse["measured"], fine["measured"])
        self.assertLess(fine["measured"], 0.72)

    def test_over_confident_posterior_under_covers(self):
        truth, draws = synthetic_posterior(
            20000, 64, error_rms=2.0, width=0.5, seed=5
        )
        low = central_interval_coverage(truth, draws, 0.68)
        high = central_interval_coverage(truth, draws, 0.95)
        self.assertLess(low["measured"], 0.30)
        self.assertLess(high["measured"], 0.45)

    def test_rejects_bad_levels_and_shapes(self):
        truth, draws = synthetic_posterior(50, 8, error_rms=1.0, width=1.0)
        for level in (0.0, 1.0, -0.5, 1.5):
            with self.assertRaises(ValueError):
                central_interval_coverage(truth, draws, level)
        with self.assertRaises(ValueError):
            central_interval_coverage(np.zeros(3), np.zeros((4, 5)), 0.68)


class PullTests(unittest.TestCase):
    def test_calibrated_posterior_has_unit_pull(self):
        truth, draws = synthetic_posterior(
            20000, 64, error_rms=0.5, width=0.5, seed=11
        )
        pull = pull_statistics(truth, draws)
        self.assertAlmostEqual(pull["pull_mean"], 0.0, delta=0.05)
        self.assertAlmostEqual(pull["pull_std"], 1.0, delta=0.05)
        self.assertFalse(pull["degenerate"])
        self.assertEqual(pull["zero_variance_events"], 0)

    def test_over_confident_posterior_has_a_wide_pull(self):
        truth, draws = synthetic_posterior(
            20000, 64, error_rms=2.0, width=0.5, seed=11
        )
        pull = pull_statistics(truth, draws)
        self.assertGreater(pull["pull_std"], 3.5)
        self.assertLess(pull["pull_std"], 4.5)
        self.assertGreater(
            pull["pull_robust_width_1p4826mad"], pull["pull_std"] * 0.8
        )

    def test_zero_variance_posterior_is_flagged_not_nan(self):
        truth = np.array([1.0, 2.0, 3.0])
        draws = np.tile(truth, (4, 1))
        pull = pull_statistics(truth, draws)
        self.assertTrue(pull["degenerate"])
        self.assertEqual(pull["zero_variance_events"], 3)
        self.assertTrue(np.isfinite(pull["pull_std"]))
        self.assertTrue(
            np.all(np.isfinite(pull_values(truth, draws)))
        )

    def test_finite_draw_correction_is_the_expected_size(self):
        truth, draws = synthetic_posterior(
            40000, 16, error_rms=0.5, width=0.5, seed=13
        )
        pull = pull_statistics(truth, draws)
        self.assertAlmostEqual(
            pull["pull_std"], finite_draw_pull_std(16), delta=0.03
        )

    def test_rejects_single_draw(self):
        with self.assertRaises(ValueError):
            pull_values(np.zeros(3), np.zeros((1, 3)))

    def test_ks_statistic_behaves(self):
        rng = np.random.default_rng(2)
        normal = rng.normal(0.0, 1.0, 20000)
        shifted = rng.normal(0.4, 1.0, 20000)
        clean = ks_distance_to_normal(normal)
        dirty = ks_distance_to_normal(shifted)
        self.assertLess(clean["ks_distance"], 0.02)
        self.assertGreater(dirty["ks_distance"], 0.05)
        self.assertLess(dirty["ks_p_value_asymptotic"], 1e-6)
        self.assertGreaterEqual(clean["ks_p_value_asymptotic"], 0.0)


class BootstrapTests(unittest.TestCase):
    def test_band_brackets_the_point_estimate_and_is_seeded(self):
        inside = np.zeros(2000, dtype=bool)
        inside[:1360] = True  # 0.68
        first = bootstrap_coverage_band(inside, boot=200, rng=np.random.default_rng(1))
        second = bootstrap_coverage_band(inside, boot=200, rng=np.random.default_rng(1))
        self.assertEqual(first, second)
        self.assertLessEqual(first["low"], 0.68)
        self.assertGreaterEqual(first["high"], 0.68)
        self.assertLess(first["sd"], 0.02)

    def test_std_band_is_seeded_and_sane(self):
        values = np.random.default_rng(0).normal(0.0, 2.0, 5000)
        first = bootstrap_std_band(values, boot=120, rng=np.random.default_rng(9))
        second = bootstrap_std_band(values, boot=120, rng=np.random.default_rng(9))
        self.assertEqual(first, second)
        self.assertLess(first["low"], 2.0)
        self.assertGreater(first["high"], 2.0)

    def test_rejects_too_few_resamples(self):
        with self.assertRaises(ValueError):
            bootstrap_coverage_band(
                np.ones(10, dtype=bool), boot=1, rng=np.random.default_rng(0)
            )


class FiniteDrawReferenceTests(unittest.TestCase):
    def test_pull_std_reference_is_the_analytic_factor(self):
        # sqrt(1 + 1/D) * sqrt((D-1)/(D-3))
        self.assertAlmostEqual(finite_draw_pull_std(32), 1.0499, places=3)
        self.assertAlmostEqual(finite_draw_pull_std(64), 1.0242, places=3)
        self.assertAlmostEqual(finite_draw_pull_std(16), 1.1072, places=3)
        self.assertTrue(math.isnan(finite_draw_pull_std(3)))
        self.assertTrue(math.isnan(finite_draw_pull_std(2)))

    def test_pull_std_reference_decreases_with_draws(self):
        values = [finite_draw_pull_std(d) for d in (8, 16, 32, 64, 128)]
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertTrue(all(value > 1.0 for value in values))

    def test_null_run_is_calibrated_by_construction(self):
        null = null_calibration_reference(
            6000, 32, levels=(0.68, 0.95), bins=11, boot=40, seed=99
        )
        analytic = finite_draw_pull_std(32)
        self.assertAlmostEqual(null["pull"]["pull_std"], analytic, delta=0.05)
        self.assertAlmostEqual(null["pull"]["pull_mean"], 0.0, delta=0.05)
        # Sampled-quantile intervals are inward-biased, so a *calibrated*
        # conditional still under-covers at this draw count - most at 0.68,
        # where the density is highest. The null is what measures that.
        self.assertLess(null["coverage"]["0.68"]["measured"], 0.68)
        self.assertGreater(null["coverage"]["0.68"]["measured"], 0.56)
        self.assertLess(null["coverage"]["0.95"]["measured"], 0.95)
        self.assertGreater(null["coverage"]["0.95"]["measured"], 0.88)
        self.assertGreater(null["pit"]["chi2_p_value"], 1e-3)

    def test_null_shortfall_shrinks_with_more_draws(self):
        coarse = null_calibration_reference(
            20000, 8, levels=(0.68, 0.95), bins=5, boot=20, seed=3
        )
        fine = null_calibration_reference(
            20000, 64, levels=(0.68, 0.95), bins=5, boot=20, seed=3
        )
        self.assertLess(
            coarse["coverage"]["0.68"]["measured"], fine["coverage"]["0.68"]["measured"]
        )
        self.assertAlmostEqual(
            fine["coverage"]["0.68"]["measured"], 0.68, delta=0.03
        )

    def test_null_is_deterministic_under_a_fixed_seed(self):
        first = null_calibration_reference(
            2000, 8, levels=(0.68, 0.95), bins=5, boot=20, seed=7
        )
        second = null_calibration_reference(
            2000, 8, levels=(0.68, 0.95), bins=5, boot=20, seed=7
        )
        self.assertEqual(
            first["coverage"]["0.68"]["measured"],
            second["coverage"]["0.68"]["measured"],
        )
        self.assertEqual(first["pull"]["pull_std"], second["pull"]["pull_std"])


class CalibrationBlockTests(unittest.TestCase):
    def _block(self, *, error_rms: float, width: float, shuffle: bool = False, seed: int = 21):
        truth, draws = synthetic_posterior(
            8000, 32, error_rms=error_rms, width=width, seed=seed
        )
        if shuffle:
            perm = derangement_indices(truth.size, np.random.default_rng(seed + 1))
            truth = truth[perm]
        return calibration_block(
            truth,
            draws,
            levels=(0.68, 0.95),
            bins=11,
            boot=40,
            rng=np.random.default_rng(seed + 2),
        )

    def _null(self, seed: int = 21):
        return null_calibration_reference(
            8000, 32, levels=(0.68, 0.95), bins=11, boot=40, seed=seed
        )

    def test_calibrated_block_passes_every_criterion(self):
        block = self._block(error_rms=0.5, width=0.5)
        criteria = calibration_criteria(block, self._null())
        for name, criterion in criteria.items():
            self.assertTrue(criterion["pass"], msg=f"{name} failed: {criterion}")
        self.assertEqual(calibration_verdict(criteria, block["pull"]), "calibrated")

    def test_under_confident_block_is_flagged_as_such(self):
        block = self._block(error_rms=0.25, width=0.5)
        criteria = calibration_criteria(block, self._null())
        verdict = calibration_verdict(criteria, block["pull"])
        self.assertIn("not calibrated", verdict)
        self.assertIn("under-confident", verdict)

    def test_over_confident_block_fails_and_says_why(self):
        block = self._block(error_rms=2.0, width=0.5)
        criteria = calibration_criteria(block, self._null())
        verdict = calibration_verdict(criteria, block["pull"])
        self.assertIn("not calibrated", verdict)
        self.assertIn("over-confident", verdict)
        self.assertFalse(criteria["rank_histogram_consistent_with_uniform"]["pass"])
        self.assertFalse(criteria["coverage_68_consistent_with_null"]["pass"])

    def test_coverage_criterion_counts_both_sampling_errors(self):
        block = self._block(error_rms=0.5, width=0.5)
        criterion = calibration_criteria(block, self._null())["coverage_68_consistent_with_null"]
        combined = np.hypot(criterion["se_paired"], criterion["se_null"])
        self.assertAlmostEqual(criterion["se_paired"], criterion["se_null"], places=9)
        self.assertAlmostEqual(
            criterion["z_score_vs_null"],
            (criterion["value"] - criterion["null_value"]) / combined,
            places=9,
        )

    def test_zero_variance_block_is_degenerate_not_calibrated(self):
        truth = np.array([1.0, 2.0, 3.0])
        draws = np.tile(truth, (8, 1))
        block = calibration_block(
            truth, draws, levels=(0.68, 0.95), bins=5, boot=10,
            rng=np.random.default_rng(3),
        )
        criteria = calibration_criteria(block, self._null())
        self.assertIn("degenerate", calibration_verdict(criteria, block["pull"]))

    def test_shuffle_destroys_the_paired_calibration(self):
        paired = self._block(error_rms=0.5, width=0.5)
        shuffled = self._block(error_rms=0.5, width=0.5, shuffle=True)
        self.assertLess(
            shuffled["coverage"]["0.68"]["measured"],
            paired["coverage"]["0.68"]["measured"] - 0.2,
        )
        self.assertGreater(shuffled["pull"]["pull_std"], paired["pull"]["pull_std"])
        self.assertGreater(
            shuffled["pit"]["max_abs_deviation_sigma"],
            paired["pit"]["max_abs_deviation_sigma"],
        )

    def test_block_json_is_serialisable(self):
        block = json_ready(self._block(error_rms=0.5, width=0.5))
        text = json.dumps(block, sort_keys=True)
        self.assertIn("max_abs_deviation_sigma", text)
        self.assertIsInstance(block["coverage"]["0.68"]["bootstrap"]["low"], float)


class HelperTests(unittest.TestCase):
    def test_json_ready_converts_numpy_scalars_and_arrays(self):
        payload = {
            "int": np.int64(3),
            "float": np.float32(0.5),
            "bool": np.bool_(True),
            "array": np.arange(3),
            "nested": {"value": np.float64(1.25)},
        }
        ready = json_ready(payload)
        json.dumps(ready)  # must not raise
        self.assertEqual(ready["int"], 3)
        self.assertIsInstance(ready["float"], float)
        self.assertIs(ready["bool"], True)
        self.assertEqual(ready["array"], [0, 1, 2])

    def test_checkpoint_argument_accepts_path_and_label(self):
        label, path = parse_checkpoint_argument("outputs/cms_Joint/ppzee/best_model.pt")
        self.assertEqual(label, "checkpoint")
        self.assertTrue(path.is_absolute())
        label, path = parse_checkpoint_argument(
            "mylabel=outputs/cms_Joint/ppzee/best_model.pt"
        )
        self.assertEqual(label, "mylabel")
        self.assertTrue(str(path).endswith("best_model.pt"))


@unittest.skipUnless(HAS_FIXTURES, "ppzee bench or checkpoint not present")
class CliSmokeTests(unittest.TestCase):
    """One cheap end-to-end run on the real bench, plus a determinism check."""

    def _run(self, out_dir: Path, events: int, draws: int, seed: int) -> Path:
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--events",
                str(events),
                "--draws",
                str(draws),
                "--seed",
                str(seed),
                "--bins",
                "5",
                "--bootstrap",
                "12",
                "--device",
                "cpu",
                "--no-plot",
                "--output-dir",
                str(out_dir),
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=1800,
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout[-2000:] + result.stderr[-2000:])
        return out_dir / "quantile_calibration.json"

    def test_end_to_end_json_is_complete_and_reproducible(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            json_path = self._run(Path(first), events=64, draws=4, seed=20261004)
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(payload["events_used"], 64)
            self.assertEqual(payload["draws"], 4)
            self.assertEqual(payload["control"]["fixed_points"], 0)
            for key in ("paired", "shuffled", "observed_change", "cross_checks"):
                self.assertIn(key, payload)
            for key in ("0.68", "0.95"):
                self.assertIn(key, payload["paired"]["coverage"])
                self.assertIn("bootstrap", payload["paired"]["coverage"][key])
            self.assertIn("criteria", payload["paired"])
            self.assertIn("verdict", payload["paired"])
            self.assertIn("histogram", payload["paired"]["pull"])
            self.assertEqual(
                sum(payload["paired"]["pit"]["counts"]), payload["events_used"]
            )
            self.assertEqual(
                len(payload["paired"]["pit"]["counts"]), payload["bins"]
            )
            # identical settings, different output directory -> identical numbers
            repeat = json.loads(
                self._run(Path(second), events=64, draws=4, seed=20261004).read_text(
                    encoding="utf-8"
                )
            )
            for volatile in ("created_utc", "runtime_seconds", "artifacts"):
                repeat.pop(volatile, None)
                payload.pop(volatile, None)
            self.assertEqual(payload, repeat)


if __name__ == "__main__":
    unittest.main()
