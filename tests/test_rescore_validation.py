#!/usr/bin/env python
"""Contract tests for the read-only validation re-scorer.

These pin the noise-policy resolution only; the validation pass itself is the
trainer's code path and is covered by the joint tests. No data or GPU needed.
"""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "scripts_sota",
                  REPO_ROOT / "scripts_joint", REPO_ROOT / "scripts_joint" / "upsilon"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from rescore_validation import _noise_spec  # noqa: E402


def _args(**kwargs) -> argparse.Namespace:
    defaults = {"noise": "native", "core": None, "tail": None}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class NoiseSpecTests(unittest.TestCase):
    CHECKPOINT = {
        "noise_multipliers": {
            "core": 1.0,
            "tail": 0.5,
            "encoder_core": 1.0,
            "encoder_tail": 0.5,
            "decoder_core": 1.0,
            "decoder_tail": 0.5,
        }
    }

    def test_native_uses_the_checkpoint_recorded_pair(self):
        self.assertEqual(_noise_spec(_args(noise="native"), self.CHECKPOINT),
                         {"core": 1.0, "tail": 0.5})

    def test_zero_is_explicit(self):
        self.assertEqual(_noise_spec(_args(noise="zero"), self.CHECKPOINT),
                         {"core": 0.0, "tail": 0.0})

    def test_explicit_accepts_a_single_side(self):
        spec = _noise_spec(_args(noise="explicit", core=0.7), self.CHECKPOINT)
        self.assertEqual(spec, {"core": 0.7})

    def test_explicit_without_values_is_rejected(self):
        with self.assertRaises(SystemExit):
            _noise_spec(_args(noise="explicit"), self.CHECKPOINT)


if __name__ == "__main__":
    unittest.main()
