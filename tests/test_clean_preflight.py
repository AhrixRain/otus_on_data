"""Tests for the preflight-directory cleanup rule (CLAUDE.md section 3)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts_joint") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts_joint"))

from clean_preflight import (  # noqa: E402
    directory_size_bytes,
    find_preflight_dirs,
    is_preflight_name,
    remove_preflight_dirs,
)


class NameMatchingTests(unittest.TestCase):
    def test_matches_observed_preflight_names(self):
        for name in (
            "Run_H_dryrun",
            "Run_H_smoke",
            "Run_H_cycleNoNoise_dryrun",
            "Run_H_noiseFloor_smoke",
            "unifiedP1_components_dryrun",
            "AB_narrow_split_dryrun",
            "runG0_stable_smoke2",
            "ab_plots_smoke",
        ):
            self.assertTrue(is_preflight_name(name), name)

    def test_does_not_match_real_runs(self):
        for name in (
            "Run_H",
            "Run_H_fix",
            "Run_E_noiseLowCore",
            "runG0_stable_full_20260821",
            "runG0_warmup_probe",
            "noise_budget",
            "swd_variant_probe",
        ):
            self.assertFalse(is_preflight_name(name), name)


class DiscoveryTests(unittest.TestCase):
    def test_finds_nested_and_skips_roots_and_normal_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Run_H_dryrun").mkdir()
            (root / "nested" / "Run_H_smoke").mkdir(parents=True)
            (root / "nested" / "Run_H").mkdir()
            (root / "swd_variant_probe").mkdir()

            found = find_preflight_dirs([root])
            names = sorted(path.name for path in found)
            self.assertEqual(names, ["Run_H_dryrun", "Run_H_smoke"])

    def test_root_itself_is_never_returned(self):
        with tempfile.TemporaryDirectory(prefix="Run_H_smoke_") as tmp:
            (Path(tmp) / "inner").mkdir()
            found = find_preflight_dirs([Path(tmp)])
            self.assertEqual(found, [])

    def test_missing_root_is_ignored(self):
        self.assertEqual(find_preflight_dirs([Path("/does/not/exist/anywhere")]), [])


class RemovalTests(unittest.TestCase):
    def test_remove_apply_deletes_only_preflight_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            preflight = root / "Run_H_dryrun"
            preflight.mkdir()
            (preflight / "config.resolved.json").write_text("{}", encoding="utf-8")
            keep = root / "Run_H"
            keep.mkdir()
            (keep / "history.json").write_text("{}", encoding="utf-8")

            directories = find_preflight_dirs([root])
            self.assertEqual([path.name for path in directories], ["Run_H_dryrun"])
            count, total = remove_preflight_dirs(directories, apply=True)
            self.assertEqual(count, 1)
            self.assertGreater(total, 0)
            self.assertFalse(preflight.exists())
            self.assertTrue(keep.exists())

    def test_list_mode_does_not_delete(self):
        with tempfile.TemporaryDirectory() as tmp:
            preflight = Path(tmp) / "Run_H_smoke"
            preflight.mkdir()
            directories = find_preflight_dirs([Path(tmp)])
            remove_preflight_dirs(directories, apply=False)
            self.assertTrue(preflight.exists())

    def test_directory_size_counts_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "Run_H_dryrun"
            (target / "sub").mkdir(parents=True)
            (target / "a.txt").write_text("12345", encoding="utf-8")
            (target / "sub" / "b.txt").write_text("123", encoding="utf-8")
            self.assertEqual(directory_size_bytes(target), 8)


if __name__ == "__main__":
    unittest.main()
