"""Tests for the paired-benchmark region loader.

The load-bearing ones are the leakage tests: `test_training_arrays_are_not
_paired` and `test_pair_index_is_absent_from_every_training_array`. If the
pairing reaches training, `residual_rms_vs_identity` becomes meaningless and
looks excellent, which is the failure mode that would waste the most time
(memory.md section 7.4, docs/step3_ppzee_closure_prompt.md).

Every test here builds its own synthetic paired HDF5 file, so the suite runs
on a machine that does not have `data/ppzee.hdf5`.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for folder in ("scripts", "scripts_joint", "scripts_sota"):
    path = str(REPO_ROOT / folder)
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    import h5py
except ImportError:  # pragma: no cover - h5py is a hard dependency of the env
    h5py = None

from paired_data import (  # noqa: E402
    MAX_LEAKED_CORRELATION,
    SPLIT_KEYS,
    check_unpaired,
    is_paired_channel,
    leakage_threshold,
    load_paired_region,
    load_paired_source,
    materialize_pairs,
    normalize_channel,
)


def _make_paired_file(path: Path, n: int, seed: int, extra_columns: int = 4) -> None:
    """Write a synthetic FDL/ROL file whose rows really are partners.

    ``ROL[:, 0:8]`` is ``FDL`` plus a small smear, so the per-column
    correlation is high (as it is at 0.976-0.991 in the real file) and the
    leakage guard's positive control has something to fire on. The extra
    columns stand in for the real file's MET block.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 30.0, size=(n, 8))
    z[:, 3] = np.abs(z[:, 3]) + 60.0
    z[:, 7] = np.abs(z[:, 7]) + 60.0
    x_core = z + rng.normal(0.0, 1.0, size=(n, 8))
    x = np.concatenate([x_core, rng.normal(0.0, 5.0, size=(n, extra_columns))], axis=1)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("FDL", data=z)
        handle.create_dataset("ROL", data=x)


def _region_config(train: Path, test: Path, **split) -> dict:
    split_config = {"train_ratio": 0.8, "val_ratio": 0.1}
    split_config.update(split)
    return {
        "paths": {
            "paired_train_file": str(train),
            "paired_test_file": str(test),
        },
        "data": {
            "channel": "ppzee",
            "z_dataset": "FDL",
            "x_dataset": "ROL",
            "z_columns": [0, 8],
            "x_columns": [0, 8],
        },
        "data_split": split_config,
        "seed": 1701,
        "float_type": "float32",
    }


@unittest.skipIf(h5py is None, "h5py not available")
class TestPairedLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.train_path = root / "fake_train.hdf5"
        cls.test_path = root / "fake_test.hdf5"
        _make_paired_file(cls.train_path, 4000, seed=11)
        _make_paired_file(cls.test_path, 900, seed=12)
        cls.config = _region_config(cls.train_path, cls.test_path)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _load(self, config=None):
        return load_paired_region(config or self.config, log=lambda *_: None)

    # --- shapes, dtypes, contract ------------------------------------------

    def test_emits_exactly_the_six_split_arrays(self):
        arrays, _, _ = self._load()
        self.assertEqual(set(arrays), set(SPLIT_KEYS))

    def test_shapes_and_dtypes(self):
        arrays, _, _ = self._load()
        for key, value in arrays.items():
            self.assertEqual(value.ndim, 2, key)
            self.assertEqual(value.shape[1], 8, key)
            self.assertEqual(value.dtype, np.dtype("float32"), key)
            self.assertTrue(np.isfinite(value).all(), key)

    def test_met_columns_are_dropped(self):
        """x is ROL[:, 0:8]; the trailing MET block never reaches the model."""
        arrays, _, _ = self._load()
        with h5py.File(self.train_path, "r") as handle:
            rol = np.asarray(handle["ROL"][:])
        self.assertEqual(rol.shape[1], 12)
        self.assertEqual(arrays["x_train"].shape[1], 8)

    def test_split_sizes_follow_ratios_and_caps(self):
        arrays, _, _ = self._load()
        # 4000 fit rows: 80% train, 10% val. 900 held-out rows, uncapped.
        self.assertEqual(len(arrays["x_train"]), 3200)
        self.assertEqual(len(arrays["z_train"]), 3200)
        self.assertEqual(len(arrays["x_val"]), 400)
        self.assertEqual(len(arrays["x_test"]), 900)

    def test_test_split_takes_the_whole_held_out_file(self):
        """The held-out file is a separate generation, not a ratio of the fit file."""
        arrays, _, _ = self._load()
        with h5py.File(self.test_path, "r") as handle:
            n_held = len(handle["FDL"])
        self.assertEqual(len(arrays["x_test"]), n_held)
        self.assertEqual(len(arrays["z_test"]), n_held)

    def test_test_max_caps_the_held_out_split(self):
        config = _region_config(self.train_path, self.test_path, test_max=250)
        arrays, _, _ = load_paired_region(config, log=lambda *_: None)
        self.assertEqual(len(arrays["x_test"]), 250)

    def test_train_and_val_caps_apply(self):
        config = _region_config(
            self.train_path, self.test_path, train_max=100, val_max=50
        )
        arrays, _, _ = load_paired_region(config, log=lambda *_: None)
        self.assertEqual(len(arrays["x_train"]), 100)
        self.assertEqual(len(arrays["x_val"]), 50)

    # --- the prime directive ------------------------------------------------

    def test_training_arrays_are_not_paired(self):
        """The measured guard, with its positive control.

        On the source file the rows are partners and correlate strongly. After
        the loader's independent permutations that correlation must be gone.
        Both halves are asserted: a guard that cannot fire proves nothing.
        """
        z_src, x_src = load_paired_source(self.train_path)
        source_rho = max(
            abs(np.corrcoef(z_src[:, i], x_src[:, i])[0, 1]) for i in range(8)
        )
        self.assertGreater(source_rho, 0.9, "the fixture is not actually paired")

        arrays, info, _ = self._load()
        x_train, z_train = arrays["x_train"], arrays["z_train"]
        shuffled_rho = max(
            abs(np.corrcoef(x_train[:, i], z_train[:, i])[0, 1]) for i in range(8)
        )
        self.assertLess(shuffled_rho, leakage_threshold(len(x_train)))
        # The gap between the paired source and the shuffled split is the
        # whole point: 0.999 down to a few percent.
        self.assertLess(shuffled_rho, source_rho / 10.0)
        self.assertAlmostEqual(
            info["leakage_guard"]["max_abs_corr_x_train_z_train"],
            shuffled_rho,
            places=6,
        )
        self.assertGreater(info["leakage_guard"]["max_abs_corr_source"], 0.9)

    def test_pair_index_is_absent_from_every_training_array(self):
        arrays, _, pair_index = self._load()
        self.assertEqual(set(arrays), set(SPLIT_KEYS))
        self.assertNotIn("pair_index", arrays)
        # An index array would be integral and one-dimensional; nothing the
        # trainer receives is either.
        for key, value in arrays.items():
            self.assertFalse(np.issubdtype(value.dtype, np.integer), key)
            self.assertEqual(value.ndim, 2, key)
        # And the indices really are recorded somewhere else.
        self.assertEqual(set(pair_index["splits"]), set(SPLIT_KEYS))

    def test_x_and_z_use_different_permutations(self):
        _, _, pair_index = self._load()
        for split in ("train", "val", "test"):
            x_idx = np.asarray(pair_index["splits"][f"x_{split}"]["index"])
            z_idx = np.asarray(pair_index["splits"][f"z_{split}"]["index"])
            n = min(len(x_idx), len(z_idx))
            coincidences = int(np.count_nonzero(x_idx[:n] == z_idx[:n]))
            self.assertLess(coincidences, 50, split)
            self.assertFalse(np.array_equal(x_idx, z_idx), split)

    def test_train_and_test_indices_come_from_different_files(self):
        _, _, pair_index = self._load()
        self.assertEqual(pair_index["sources"]["train"], str(self.train_path))
        self.assertEqual(pair_index["sources"]["test"], str(self.test_path))

    def test_same_file_for_train_and_test_is_refused(self):
        config = _region_config(self.train_path, self.train_path)
        with self.assertRaises(ValueError):
            load_paired_region(config, log=lambda *_: None)

    def test_unpaired_source_is_refused(self):
        """If FDL and ROL are not partners, nothing downstream would be valid."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "unpaired.hdf5"
            rng = np.random.default_rng(3)
            with h5py.File(path, "w") as handle:
                handle.create_dataset("FDL", data=rng.normal(size=(500, 8)))
                handle.create_dataset("ROL", data=rng.normal(size=(500, 12)))
            config = _region_config(path, self.test_path)
            with self.assertRaises(ValueError) as ctx:
                load_paired_region(config, log=lambda *_: None)
            self.assertIn("partners", str(ctx.exception))

    # --- reproducibility ----------------------------------------------------

    def test_split_is_reproducible_under_a_fixed_seed(self):
        first_arrays, _, first_index = self._load()
        second_arrays, _, second_index = self._load()
        for key in SPLIT_KEYS:
            np.testing.assert_array_equal(first_arrays[key], second_arrays[key])
            self.assertEqual(
                first_index["splits"][key]["sha256"],
                second_index["splits"][key]["sha256"],
            )

    def test_a_different_seed_gives_a_different_split(self):
        other = _region_config(self.train_path, self.test_path)
        other["seed"] = 99
        arrays, _, _ = load_paired_region(other, log=lambda *_: None)
        baseline, _, _ = self._load()
        self.assertFalse(
            np.array_equal(arrays["x_train"], baseline["x_train"])
        )

    def test_pair_index_digest_matches_its_own_index(self):
        import hashlib

        _, _, pair_index = self._load()
        for key in SPLIT_KEYS:
            entry = pair_index["splits"][key]
            digest = hashlib.sha256(
                np.asarray(entry["index"], dtype=np.int64).tobytes()
            ).hexdigest()
            self.assertEqual(digest, entry["sha256"], key)
            self.assertEqual(len(entry["index"]), entry["count"], key)

    # --- the pairing can be rebuilt for scoring -----------------------------

    def test_materialize_pairs_recovers_the_partners(self):
        arrays, _, pair_index = self._load()
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "joint_split_manifest.json"
            manifest_path.write_text(
                json.dumps({"regions": {"ppzee": {"pair_index": pair_index}}}),
                encoding="utf-8",
            )
            z_pairs, x_pairs = materialize_pairs(manifest_path, "ppzee", "test")
        # x_pairs must be exactly the x_test the model was shown, and z_pairs
        # must be each of those events' own truth partner.
        np.testing.assert_allclose(
            x_pairs.astype(np.float32), arrays["x_test"], rtol=0, atol=0
        )
        rho = min(abs(np.corrcoef(z_pairs[:, i], x_pairs[:, i])[0, 1]) for i in range(8))
        self.assertGreater(rho, 0.9)

    def test_materialized_pairs_are_not_the_shuffled_z_test(self):
        """z_test is independently shuffled, so it is NOT the partner array."""
        arrays, _, pair_index = self._load()
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "m.json"
            manifest_path.write_text(
                json.dumps({"regions": {"ppzee": {"pair_index": pair_index}}}),
                encoding="utf-8",
            )
            z_pairs, _ = materialize_pairs(manifest_path, "ppzee", "test")
        self.assertFalse(np.array_equal(z_pairs.astype(np.float32), arrays["z_test"]))


class TestLeakageGuard(unittest.TestCase):
    """The guard has to fire on a leak, at every sample size that matters.

    CLAUDE.md section 4: a gate that a no-op passes certifies nothing. The same
    applies to a leakage check that a leak passes.
    """

    def _paired(self, n, seed=5):
        rng = np.random.default_rng(seed)
        z = rng.normal(0.0, 30.0, size=(n, 8))
        x = z + rng.normal(0.0, 1.0, size=(n, 8))
        return x, z

    def test_row_aligned_arrays_are_rejected(self):
        for n in (100, 1000, 20000):
            x, z = self._paired(n)
            with self.subTest(n=n), self.assertRaises(AssertionError) as ctx:
                check_unpaired(x, z)
            self.assertIn("leaked into training", str(ctx.exception))

    def test_partially_leaked_arrays_are_rejected(self):
        """Even a 10% leak -- 90% shuffled, 10% left in place -- must fire."""
        n = 20000
        x, z = self._paired(n)
        rng = np.random.default_rng(7)
        perm = rng.permutation(n)
        keep = perm[: n // 10]
        perm[: n // 10] = keep  # no-op, kept for clarity
        shuffled = z[perm]
        shuffled[: n // 10] = z[: n // 10]  # first 10% put back in place
        with self.assertRaises(AssertionError):
            check_unpaired(x, shuffled)

    def test_independent_arrays_pass_at_every_size(self):
        for n in (100, 1000, 20000, 120000):
            rng = np.random.default_rng(n)
            x, z = self._paired(n)
            with self.subTest(n=n):
                max_leak, threshold = check_unpaired(x[rng.permutation(n)], z)
                self.assertLessEqual(max_leak, threshold)

    def test_threshold_floor_and_band(self):
        # Large n: the fixed floor governs.
        self.assertEqual(leakage_threshold(120000), MAX_LEAKED_CORRELATION)
        # Small n: the finite-sample band governs and is looser.
        self.assertGreater(leakage_threshold(100), MAX_LEAKED_CORRELATION)
        # Monotone non-increasing in n.
        sizes = [10, 100, 1000, 10000, 100000]
        values = [leakage_threshold(n) for n in sizes]
        self.assertEqual(values, sorted(values, reverse=True))


class TestChannelDispatch(unittest.TestCase):
    def test_paired_channels_are_recognised(self):
        for value in ("ppzee", "PPZee", "ppttbar", "paired-benchmark"):
            self.assertTrue(is_paired_channel(value), value)

    def test_muon_channels_are_not_paired(self):
        for value in ("muon", "doublemuon", "mumu", "jpsi_mumu", "electron", "ee"):
            self.assertFalse(is_paired_channel(value), value)

    def test_normalize_channel(self):
        self.assertEqual(normalize_channel(" Paired-Benchmark "), "paired_benchmark")


if __name__ == "__main__":
    unittest.main()
