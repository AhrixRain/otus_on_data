"""D3b: the response channel must be present where the decision is made.

The arm is config-only. These tests pin the one-intervention contract against
H_A2frozen, the stage-1 noise change, the scored-map policy, and the inherited
three-stage schedule (a renamed stage would be APPENDED by _deep_merge, not
replaced).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (
    REPO_ROOT / "scripts",
    REPO_ROOT / "scripts_sota",
    REPO_ROOT / "scripts_joint",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from cms_data import load_config  # noqa: E402
from joint_metrics import resolve_noise_multipliers  # noqa: E402


BASELINE_CONFIG = REPO_ROOT / "configs_joint" / "cms_Joint_runH_A2frozen.yaml"
D3B_CONFIG = REPO_ROOT / "configs_joint" / "cms_Joint_runH_D3b.yaml"

STAGE1 = "runH_stage1_deterministic_warmup"
STAGE2 = "runH_stage2_stochastic_core"
STAGE3 = "runH_stage3_stochastic_tail"

# Exactly what this arm is allowed to change. Anything else is a second
# variable and makes the arm uninterpretable.
# The four encoder multipliers are deliberately absent: the arm pins them at
# 0.0, which is the inherited value, so they must NOT show up as a difference.
ALLOWED_DIFFS = {
    "run_label",
    "run_name",
    "comparison.baseline_run",
    "comparison.controlled_change",
    "comparison.not_claimed",
    "loaders.validation_noise_multipliers.decoder_core",
    "loaders.validation_noise_multipliers.decoder_tail",
    "final_evaluation.noise_multipliers.decoder_core",
    "final_evaluation.noise_multipliers.decoder_tail",
    f"stages[{STAGE1}].core_noise_multiplier",
    f"stages[{STAGE1}].tail_noise_multiplier",
}


def _diff(parent, child, path: str = "") -> dict[str, tuple[object, object]]:
    """Return {dotted path: (parent value, child value)} for every difference."""
    out: dict[str, tuple[object, object]] = {}
    if isinstance(parent, dict) and isinstance(child, dict):
        for key in sorted(set(parent) | set(child)):
            here = f"{path}.{key}" if path else key
            if key not in parent or key not in child:
                out[here] = (parent.get(key), child.get(key))
            else:
                out.update(_diff(parent[key], child[key], here))
        return out
    if isinstance(parent, list) and isinstance(child, list):
        named = all(
            isinstance(item, dict) and "name" in item for item in [*parent, *child]
        )
        if named:
            parent_by = {item["name"]: item for item in parent}
            child_by = {item["name"]: item for item in child}
            for name in sorted(set(parent_by) | set(child_by)):
                here = f"{path}[{name}]"
                if name not in parent_by or name not in child_by:
                    out[here] = (parent_by.get(name), child_by.get(name))
                else:
                    out.update(_diff(parent_by[name], child_by[name], here))
            return out
    if parent != child:
        out[path] = (parent, child)
    return out


def _load(path: Path) -> dict:
    config = load_config(path)
    config.pop("_config_path", None)
    return config


class D3bOneInterventionContract(unittest.TestCase):
    def test_diff_against_a2frozen_is_exactly_the_declared_change(self):
        baseline = _load(BASELINE_CONFIG)
        arm = _load(D3B_CONFIG)
        diff = _diff(baseline, arm)
        self.assertEqual(
            sorted(diff),
            sorted(ALLOWED_DIFFS),
            f"unexpected config differences: {sorted(set(diff) - ALLOWED_DIFFS)}",
        )

    def test_stage_list_is_inherited_and_still_three_stages(self):
        baseline = _load(BASELINE_CONFIG)
        arm = _load(D3B_CONFIG)
        # A renamed stage would be appended by _deep_merge, silently making a
        # four-stage schedule with the new name carrying no inherited keys.
        self.assertEqual(
            [stage["name"] for stage in arm["stages"]],
            [stage["name"] for stage in baseline["stages"]],
        )
        self.assertEqual(
            [stage["name"] for stage in arm["stages"]],
            [STAGE1, STAGE2, STAGE3],
        )

    def test_model_and_kernel_are_untouched(self):
        baseline = _load(BASELINE_CONFIG)
        arm = _load(D3B_CONFIG)
        self.assertEqual(arm["model"], baseline["model"])
        self.assertTrue(
            arm["model"]["decoder_noise_overrides"]["freeze_noise_amplitude"]
        )
        # D3b extends the frozen-kernel control, not the refuted z-cycle arm.
        for stage in arm["stages"]:
            self.assertNotIn("zeta", stage)
        self.assertNotIn("z_cycle_noise", arm)


class D3bResponseChannelPresent(unittest.TestCase):
    def test_stage1_carries_the_same_channel_as_stage2(self):
        arm = _load(D3B_CONFIG)
        stages = {stage["name"]: stage for stage in arm["stages"]}
        for key in ("core_noise_multiplier", "tail_noise_multiplier"):
            self.assertEqual(stages[STAGE1][key], stages[STAGE2][key])
            self.assertGreater(float(stages[STAGE1][key]), 0.0)

    def test_scored_map_is_the_response_model_not_the_deterministic_map(self):
        arm = _load(D3B_CONFIG)
        validation = resolve_noise_multipliers(
            arm["loaders"]["validation_noise_multipliers"]
        )
        final = resolve_noise_multipliers(
            arm["final_evaluation"]["noise_multipliers"]
        )
        expected = {
            "encoder_core": 0.0,
            "encoder_tail": 0.0,
            "decoder_core": 1.0,
            "decoder_tail": 0.25,
        }
        self.assertEqual(validation, expected)
        self.assertEqual(final, expected)
        # The decoder channel must be ON in the score: that is the whole point.
        self.assertGreater(validation["decoder_core"], 0.0)
        self.assertGreater(validation["decoder_tail"], 0.0)

    def test_baseline_scored_the_deterministic_map(self):
        # Guards the premise of this arm: if A2frozen ever starts scoring the
        # response channel, D3b stops being a controlled comparison.
        baseline = _load(BASELINE_CONFIG)
        validation = resolve_noise_multipliers(
            baseline["loaders"].get("validation_noise_multipliers")
        )
        self.assertEqual(set(validation.values()), {0.0})


if __name__ == "__main__":
    unittest.main()
