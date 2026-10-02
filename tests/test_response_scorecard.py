"""Unit tests for the A0.4 response-identification scorecard policy.

These tests are data-free: they exercise the gate bands, the collapse-proof
selection key and the transfer-blind contract on synthetic entries.
"""
import sys
import unittest
from pathlib import Path

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

from response_scorecard import (  # noqa: E402
    DEGENERATE_WIDTH_RATIO,
    MIN_NOISE_VARIANCE_FRACTION,
    REQUIRED_RATIO_BAND,
    SLICE_MAX_RATIO_MAX,
    SLICE_MEDIAN_RATIO_MAX,
    conditional_gate,
    entry_gates,
    response_gate,
    selection_is_transfer_blind,
    selection_key,
    transfer_criteria,
)


def region_block(*, ratio=1.0, noise_share=0.4, sigma_only=0.09):
    return {
        "within_std_over_required": ratio,
        "sigma_noise_only_gev": sigma_only,
        "noise_variance_fraction": noise_share,
        "mean_map_variance_fraction": None if noise_share is None else 1.0 - noise_share,
    }


def slice_block(*, median=1.2, worst=2.0, zero_median=1.1):
    return {
        "zero": {"decode_ratio_median": zero_median, "decode_ratio_max": worst},
        "native": {"decode_ratio_median": median, "decode_ratio_max": worst},
    }


def make_entry(
    label,
    *,
    jpsi_ratio=1.0,
    z_ratio=1.0,
    jpsi_share=0.4,
    z_share=0.3,
    jpsi_sigma=0.09,
    z_sigma=1.2,
    slice_median=1.2,
    slice_max=2.0,
    score=None,
    transfer=None,
):
    entry = {
        "label": label,
        "in_domain_score": score,
        "R": {
            "jpsi": region_block(ratio=jpsi_ratio, noise_share=jpsi_share, sigma_only=jpsi_sigma),
            "z": region_block(ratio=z_ratio, noise_share=z_share, sigma_only=z_sigma),
        },
        "C": {
            "jpsi": slice_block(median=slice_median, worst=slice_max),
            "z": slice_block(median=slice_median, worst=slice_max),
        },
    }
    if transfer is not None:
        entry["T"] = transfer
    entry["gates"] = entry_gates(entry)
    return entry


def transfer_block(best_medians, last_medians):
    def states(values):
        return {
            name: {"median_minus_cms_mev": value}
            for name, value in zip(("upsilon1s", "upsilon2s", "upsilon3s"), values)
        }

    return {"best": states(best_medians), "last": states(last_medians)}


class ResponseGateTests(unittest.TestCase):
    def test_width_band_edges(self):
        low, high = REQUIRED_RATIO_BAND
        for ratio, expected in ((low, True), (high, True), (low - 0.01, False), (high + 0.01, False)):
            gate = response_gate("jpsi", region_block(ratio=ratio))
            self.assertEqual(gate["width_pass"], expected, msg=f"ratio={ratio}")

    def test_degenerate_mean_map_is_flagged_by_width_not_by_share(self):
        # Narrow within-z width: the resolution cannot be in the noise.
        block = region_block(ratio=DEGENERATE_WIDTH_RATIO - 0.01, noise_share=0.9)
        gate = response_gate("jpsi", block)
        self.assertTrue(gate["degenerate"])
        self.assertFalse(gate["pass"])
        self.assertTrue(any("mean map is faking it" in reason for reason in gate["reasons"]))

    def test_live_channel_with_a_small_variance_share_is_not_degenerate(self):
        # Honest narrow priors make the share small even when the channel is live
        # (A2frozen J/psi: ratio 1.09, share 0.016).
        block = region_block(ratio=1.09, noise_share=0.016, sigma_only=0.037)
        gate = response_gate("jpsi", block)
        self.assertFalse(gate["degenerate"])
        self.assertTrue(gate["pass"])

    def test_noise_only_gate_uses_the_region_threshold(self):
        self.assertTrue(response_gate("jpsi", region_block(sigma_only=0.021))["noise_pass"])
        self.assertFalse(response_gate("jpsi", region_block(sigma_only=0.019))["noise_pass"])
        self.assertTrue(response_gate("z", region_block(sigma_only=0.51))["noise_pass"])
        self.assertFalse(response_gate("z", region_block(sigma_only=0.49))["noise_pass"])

    def test_missing_required_resolution_is_degenerate(self):
        gate = response_gate("jpsi", region_block(ratio=None))
        self.assertTrue(gate["degenerate"])
        self.assertFalse(gate["pass"])

    def test_missing_noise_fraction_is_reported_but_not_a_gate(self):
        gate = response_gate("jpsi", region_block(noise_share=None))
        self.assertFalse(gate["degenerate"])
        self.assertTrue(gate["pass"])


class ConditionalGateTests(unittest.TestCase):
    def test_slice_thresholds(self):
        self.assertTrue(conditional_gate("jpsi", slice_block(median=SLICE_MEDIAN_RATIO_MAX))["pass"])
        self.assertFalse(conditional_gate("jpsi", slice_block(median=SLICE_MEDIAN_RATIO_MAX + 0.01))["pass"])
        self.assertTrue(conditional_gate("jpsi", slice_block(median=1.0, worst=SLICE_MAX_RATIO_MAX))["pass"])
        self.assertFalse(conditional_gate("jpsi", slice_block(median=1.0, worst=SLICE_MAX_RATIO_MAX + 0.1))["pass"])


class SelectionKeyTests(unittest.TestCase):
    def test_degenerate_checkpoint_is_rejected_even_with_the_best_marginal(self):
        collapsed = make_entry("warmup", jpsi_ratio=0.15, z_ratio=0.05, score=0.1)
        healthy = make_entry("stage3", jpsi_ratio=1.05, jpsi_share=0.45, z_share=0.4, score=5.0)
        key = selection_key([collapsed, healthy])
        self.assertEqual(key["status"], "identified")
        self.assertEqual(key["selected"], "stage3")
        self.assertIn("warmup", key["rejected"])

    def test_survivors_rank_by_conditional_closure_then_score(self):
        coarse = make_entry("coarse", slice_median=2.5, score=0.5)
        fine = make_entry("fine", slice_median=1.5, score=4.0)
        self.assertEqual(selection_key([coarse, fine])["selected"], "fine")

    def test_all_degenerate_reports_identification_failed(self):
        near = make_entry("near", jpsi_ratio=0.45, z_ratio=0.45)
        far = make_entry("far", jpsi_ratio=0.10, z_ratio=0.10)
        key = selection_key([far, near])
        self.assertEqual(key["status"], "identification_failed")
        self.assertEqual(key["selected"], "near")

    def test_transfer_axis_cannot_change_the_key(self):
        base = make_entry("best_model", score=1.0)
        with_transfer = dict(base)
        with_transfer["T"] = transfer_block([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        self.assertEqual(selection_key([base]), selection_key([with_transfer]))
        self.assertTrue(selection_is_transfer_blind([with_transfer]))

    def test_zero_width_deterministic_checkpoint_does_not_raise(self):
        # A deterministic map has within-z width exactly 0: degenerate, and the
        # fallback ranking must still produce a finite key.
        entry = make_entry("warmup", jpsi_ratio=0.0, z_ratio=0.0)
        self.assertTrue(entry["gates"]["degenerate"])
        self.assertFalse(entry["gates"]["R_pass"])
        self.assertEqual(entry["gates"]["worst_R_log_deviation"], float("inf"))
        self.assertEqual(selection_key([entry])["status"], "identification_failed")

    def test_gates_are_deterministic(self):
        entry = make_entry("best_model")
        self.assertEqual(entry_gates(entry), entry_gates(entry))


class TransferCriteriaTests(unittest.TestCase):
    @staticmethod
    def _add_upsilon(entry, sigma=0.095, share=0.4):
        for state in ("upsilon1s", "upsilon2s", "upsilon3s"):
            entry["R"][state] = region_block(ratio=1.05, noise_share=share, sigma_only=sigma)
        entry["gates"] = entry_gates(entry)
        return entry

    def _entries(self, best_medians, last_medians, *, sigma=0.09, slice_median=1.2, upsilon=True):
        best = make_entry("best_model", score=1.0, slice_median=slice_median, jpsi_sigma=sigma)
        last = make_entry("last_model", score=1.5, slice_median=slice_median, jpsi_sigma=sigma)
        if upsilon:
            self._add_upsilon(best, sigma=max(sigma, 0.0))
            self._add_upsilon(last, sigma=max(sigma, 0.0))
        payload = transfer_block(best_medians, last_medians)
        best["T"] = {"states": payload["best"]}
        last["T"] = {"states": payload["last"]}
        return [best, last]

    def test_criteria_pass_on_a_healthy_arm(self):
        result = transfer_criteria(self._entries([10.0, 5.0, 5.0], [12.0, 6.0, 7.0]))
        self.assertEqual(result["status"], "artifact-measured")
        self.assertTrue(result["criterion_1_medians"]["pass"])
        self.assertTrue(result["criterion_2_noise_carries_width"]["pass"])
        self.assertTrue(result["criterion_4_slice_closure"]["pass"])

    def test_criterion_1_fails_out_of_band_and_on_drift(self):
        out_of_band = transfer_criteria(self._entries([80.0, 0.0, 0.0], [0.0, 0.0, 0.0]))
        self.assertFalse(out_of_band["criterion_1_medians"]["pass"])
        drift = transfer_criteria(self._entries([0.0, 0.0, 0.0], [60.0, 0.0, 0.0]))
        self.assertFalse(drift["criterion_1_medians"]["pass"])

    def test_criterion_2_fails_when_the_noise_channel_is_dead(self):
        result = transfer_criteria(self._entries([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], sigma=0.010))
        self.assertFalse(result["criterion_2_noise_carries_width"]["pass"])

    def test_criterion_2_is_not_measured_without_upsilon_regions(self):
        result = transfer_criteria(
            self._entries([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], upsilon=False)
        )
        self.assertIsNone(result["criterion_2_noise_carries_width"]["pass"])
        self.assertFalse(result["criterion_2_noise_carries_width"]["measured"])

    def test_criterion_3_is_explicitly_not_evaluated(self):
        result = transfer_criteria(self._entries([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]))
        self.assertIsNone(result["criterion_3_in_domain_score"]["pass"])

    def test_missing_last_checkpoint_reports_insufficient(self):
        best = make_entry("best_model")
        best["T"] = transfer_block([0, 0, 0], [0, 0, 0])["best"]
        self.assertEqual(transfer_criteria([best])["status"], "insufficient_entries")


if __name__ == "__main__":
    unittest.main()
