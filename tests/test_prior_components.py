"""Component-aware unified-prior loading tests (C4).

All fixtures are synthetic HDF5 files written into a temporary directory; the
tests never touch ``data/`` or ``outputs/``.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (
    REPO_ROOT / "scripts",
    REPO_ROOT / "scripts_sota",
    REPO_ROOT / "scripts_joint",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from cms_data import (  # noqa: E402
    filter_theory_prior,
    load_theory_prior_z,
    load_theory_prior_z_components,
    resolve_prior_component_spec,
    theory_prior_keep_mask,
)


def _write_prior(
    path: Path,
    z: np.ndarray,
    *,
    component_id=None,
    weight=None,
    attrs: dict | None = None,
) -> Path:
    with h5py.File(path, "w") as handle:
        group = handle.create_group("FDL")
        group.create_dataset("zData", data=np.asarray(z, dtype=np.float32))
        if component_id is not None:
            group.create_dataset("component_id", data=np.asarray(component_id, dtype=np.int8))
        if weight is not None:
            group.create_dataset("weight", data=np.asarray(weight, dtype=np.float64))
        for key, value in (attrs or {}).items():
            group.attrs[key] = value
    return path


def _random_p4(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 8)).astype(np.float32)


def _spec(**overrides) -> dict:
    base = {
        "enabled": True,
        "weights": {
            "policy": "resample_if_ess_ge_0p5",
            "resample_seed": 1234,
            "stop_if_ess_below": 0.5,
        },
        "signal_fraction": None,
        "signal_fraction_source": "explicit",
    }
    base.update(overrides)
    return base


class ComponentSelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.z = _random_p4(1000, 1)
        self.ids = np.concatenate([np.zeros(600, dtype=np.int8), np.full(400, 3, dtype=np.int8)])
        self.path = _write_prior(
            self.dir / "unified.hdf5",
            self.z,
            component_id=self.ids,
            attrs={
                "component_id_mapping": json.dumps({"jpsi": 0, "continuum": 3}),
                "component_names": json.dumps(["jpsi", "continuum"]),
            },
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_fraction_matching_and_component_selection(self):
        spec = _spec(signal_fraction=0.75, signal_fraction_source="explicit")
        out, report = load_theory_prior_z_components(
            self.path, None, spec, region_name="jpsi", data_root=self.dir
        )
        self.assertEqual(len(out), 1000)
        self.assertAlmostEqual(report["realized_signal_fraction"], 0.75, places=6)
        self.assertEqual(report["output_component_counts"], {0: 750, 3: 250})
        self.assertAlmostEqual(report["target_signal_fraction"], 0.75)
        self.assertEqual(report["raw_component_counts"], {0: 600, 3: 400})
        self.assertEqual(report["selected_component_counts"], {0: 600, 3: 400})

    def test_explicit_component_subset_and_single_component_source(self):
        spec = _spec(
            components=[3],
            signal_fraction=None,
            signal_fraction_source="single_component",
        )
        out, report = load_theory_prior_z_components(
            self.path, None, spec, region_name="jpsi", data_root=self.dir
        )
        self.assertEqual(len(out), 400)
        self.assertEqual(report["output_component_counts"], {3: 400})
        self.assertEqual(report["signal_components"], [])
        self.assertAlmostEqual(report["realized_signal_fraction"], 0.0)

    def test_single_component_source_rejects_multiple_components(self):
        spec = _spec(signal_fraction=None, signal_fraction_source="single_component")
        with self.assertRaisesRegex(ValueError, "exactly one component"):
            load_theory_prior_z_components(
                self.path, None, spec, region_name="jpsi", data_root=self.dir
            )

    def test_resampling_is_reproducible_with_resample_seed(self):
        spec = _spec(signal_fraction=0.6, signal_fraction_source="explicit")
        first, _ = load_theory_prior_z_components(
            self.path, None, spec, region_name="jpsi", data_root=self.dir
        )
        second, _ = load_theory_prior_z_components(
            self.path, None, spec, region_name="jpsi", data_root=self.dir
        )
        np.testing.assert_array_equal(first, second)


class LegacyEffectiveFractionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.z = _random_p4(1000, 2)
        self.ids = np.concatenate([np.zeros(600, dtype=np.int8), np.full(400, 3, dtype=np.int8)])
        self.unified = _write_prior(
            self.dir / "unified.hdf5",
            self.z,
            component_id=self.ids,
            attrs={"component_id_mapping": json.dumps({"jpsi": 0, "continuum": 3})},
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_attributes_define_the_target_fraction(self):
        legacy = _write_prior(
            self.dir / "legacy_attrs.hdf5",
            _random_p4(100, 3),
            attrs={"n_signal": 80, "n_continuum": 20, "frac_signal_post_filter": 0.8},
        )
        spec = _spec(
            signal_fraction=None,
            signal_fraction_source="legacy_prior_effective",
            legacy_prior_file=str(legacy),
        )
        out, report = load_theory_prior_z_components(
            self.unified, None, spec, region_name="jpsi", data_root=self.dir
        )
        self.assertAlmostEqual(report["target_signal_fraction"], 0.8)
        self.assertEqual(report["legacy_prior_measurement"]["measured_from"], "FDL attribute n_signal/n_continuum")
        self.assertEqual(report["output_component_counts"], {0: 800, 3: 200})

    def test_component_id_fallback_when_attributes_absent(self):
        legacy = _write_prior(
            self.dir / "legacy_ids.hdf5",
            _random_p4(100, 4),
            component_id=np.concatenate([np.zeros(30, dtype=np.int8), np.full(70, 3, dtype=np.int8)]),
        )
        spec = _spec(
            signal_fraction=None,
            signal_fraction_source="legacy_prior_effective",
            legacy_prior_file=str(legacy),
        )
        _, report = load_theory_prior_z_components(
            self.unified, None, spec, region_name="jpsi", data_root=self.dir
        )
        self.assertAlmostEqual(report["target_signal_fraction"], 0.3, places=6)
        self.assertEqual(report["legacy_prior_measurement"]["measured_from"], "FDL/component_id == 0")

    def test_inconsistent_legacy_attributes_raise(self):
        legacy = _write_prior(
            self.dir / "legacy_bad.hdf5",
            _random_p4(10, 5),
            attrs={"n_signal": 80, "n_continuum": 20, "frac_signal_post_filter": 0.5},
        )
        spec = _spec(
            signal_fraction=None,
            signal_fraction_source="legacy_prior_effective",
            legacy_prior_file=str(legacy),
        )
        with self.assertRaisesRegex(ValueError, "inconsistent signal fractions"):
            load_theory_prior_z_components(
                self.unified, None, spec, region_name="jpsi", data_root=self.dir
            )


class WeightPolicyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        n = 1000
        # Row identity is carried in the first momentum component so the
        # weighted resampling can be checked row by row.
        self.z = np.zeros((n, 8), dtype=np.float32)
        self.z[:, 0] = np.arange(n)
        self.ids = np.zeros(n, dtype=np.int8)
        self.weight = np.where(np.arange(n) % 2 == 0, 1.0, 3.0)
        self.path = _write_prior(
            self.dir / "weighted.hdf5",
            self.z,
            component_id=self.ids,
            weight=self.weight,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_weights_are_applied_and_ess_reported(self):
        spec = _spec(signal_fraction=None, signal_fraction_source="single_component")
        out, report = load_theory_prior_z_components(
            self.path, None, spec, region_name="z", data_root=self.dir
        )
        expected_ess_over_n = 1.0 / (1.0 + 0.25)  # 50/50 mixture of weights 1 and 3
        self.assertAlmostEqual(report["ess"]["background"]["ess_over_n"], expected_ess_over_n, places=4)
        self.assertGreaterEqual(report["minimum_ess_over_n"], 0.5)
        heavy_source = np.flatnonzero(self.weight == 3.0)
        chosen_heavy = np.isin(out[:, 0].astype(int), heavy_source).sum()
        self.assertAlmostEqual(chosen_heavy / len(out), 0.75, delta=0.05)
        self.assertTrue(report["has_weight_dataset"])

    def test_missing_weight_dataset_defaults_to_ones(self):
        plain = _write_prior(
            self.dir / "plain.hdf5",
            self.z,
            component_id=self.ids,
        )
        spec = _spec(signal_fraction=None, signal_fraction_source="single_component")
        _, report = load_theory_prior_z_components(
            plain, None, spec, region_name="z", data_root=self.dir
        )
        self.assertFalse(report["has_weight_dataset"])
        self.assertAlmostEqual(report["minimum_ess_over_n"], 1.0, places=12)

    def test_ess_below_threshold_raises(self):
        n = 1000
        z = _random_p4(n, 7)
        ids = np.zeros(n, dtype=np.int8)
        weight = np.ones(n)
        weight[0] = 1e6
        path = _write_prior(
            self.dir / "degenerate.hdf5", z, component_id=ids, weight=weight
        )
        spec = _spec(signal_fraction=None, signal_fraction_source="single_component")
        with self.assertRaisesRegex(ValueError, "ESS/N"):
            load_theory_prior_z_components(
                path, None, spec, region_name="z", data_root=self.dir
            )

    def test_threshold_mismatch_raises(self):
        spec = _spec(signal_fraction=None, signal_fraction_source="single_component")
        spec["weights"]["stop_if_ess_below"] = 0.3
        with self.assertRaisesRegex(ValueError, "stop_if_ess_below"):
            load_theory_prior_z_components(
                self.path, None, spec, region_name="z", data_root=self.dir
            )


class ErrorAndDefaultPathTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.z = _random_p4(100, 8)

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_component_id_raises(self):
        path = _write_prior(self.dir / "no_ids.hdf5", self.z)
        spec = _spec(signal_fraction=0.5, signal_fraction_source="explicit")
        with self.assertRaisesRegex(KeyError, "component_id"):
            load_theory_prior_z_components(
                path, None, spec, region_name="jpsi", data_root=self.dir
            )

    def test_weight_length_mismatch_raises(self):
        path = _write_prior(
            self.dir / "bad_weight.hdf5",
            self.z,
            component_id=np.zeros(len(self.z), dtype=np.int8),
            weight=np.ones(len(self.z) - 1),
        )
        spec = _spec(signal_fraction=None, signal_fraction_source="single_component")
        with self.assertRaisesRegex(ValueError, "weight has"):
            load_theory_prior_z_components(
                path, None, spec, region_name="z", data_root=self.dir
            )

    def test_unknown_region_key_raises(self):
        path = _write_prior(
            self.dir / "ids.hdf5", self.z, component_id=np.zeros(len(self.z), dtype=np.int8)
        )
        spec = _spec(signal_fraction=None, signal_fraction_source="single_component")
        spec["mixture_ratio"] = 0.5
        with self.assertRaisesRegex(ValueError, "Unknown/unimplemented"):
            load_theory_prior_z_components(
                path, None, spec, region_name="z", data_root=self.dir
            )

    def test_unknown_policy_and_source_raise(self):
        path = _write_prior(
            self.dir / "ids2.hdf5", self.z, component_id=np.zeros(len(self.z), dtype=np.int8)
        )
        spec = _spec(signal_fraction=None, signal_fraction_source="single_component")
        spec["weights"]["policy"] = "resample_if_ess_ge_0p3"
        with self.assertRaisesRegex(ValueError, "Unimplemented"):
            load_theory_prior_z_components(
                path, None, spec, region_name="z", data_root=self.dir
            )
        spec = _spec(signal_fraction=None, signal_fraction_source="prior_magic")
        with self.assertRaisesRegex(ValueError, "signal_fraction_source"):
            load_theory_prior_z_components(
                path, None, spec, region_name="z", data_root=self.dir
            )

    def test_resolve_spec_absent_or_disabled_is_none(self):
        self.assertIsNone(resolve_prior_component_spec(None, "jpsi"))
        self.assertIsNone(
            resolve_prior_component_spec(
                {"enabled": False, "unvalidated_key": 1}, "jpsi"
            )
        )
        with self.assertRaisesRegex(ValueError, "no 'jpsi' sub-block"):
            resolve_prior_component_spec(
                {"enabled": True, "weights": {}, "z": {}}, "jpsi"
            )
        with self.assertRaisesRegex(ValueError, "Unknown/unimplemented"):
            resolve_prior_component_spec(
                {"enabled": True, "weights": {}, "jpsi": {}, "bogus": 1}, "jpsi"
            )

    def test_default_loader_ignores_components_and_weight(self):
        z9 = np.hstack([self.z, np.zeros((len(self.z), 1), dtype=np.float32)])
        path = _write_prior(
            self.dir / "extra.hdf5",
            z9,
            component_id=np.zeros(len(self.z), dtype=np.int8),
            weight=np.full(len(self.z), 7.0),
        )
        default = load_theory_prior_z(path)
        np.testing.assert_array_equal(default, z9[:, :8])

    def test_keep_mask_matches_filter(self):
        z = np.zeros((2, 8), dtype=np.float32)
        z[0, 0] = 4.0  # mu- pT = 4
        z[0, 4] = 4.0  # mu+ pT = 4
        z[1, 0] = 1.0  # fails pT > 3
        z[1, 4] = 1.0
        selection = {"muon_pt_min": 3}
        mask = theory_prior_keep_mask(z, selection)
        np.testing.assert_array_equal(mask, [True, False])
        filtered = filter_theory_prior(z, selection)
        np.testing.assert_array_equal(filtered, z[mask])
        self.assertIs(filter_theory_prior(z, None), z)


class UnifiedP1ConfigWiringTests(unittest.TestCase):
    def test_config_enables_component_spec_for_each_region(self):
        from cms_data import load_config
        from joint_data import region_data_config, resolve_joint_config

        config = resolve_joint_config(
            load_config(
                REPO_ROOT / "configs_joint" / "cms_Joint_unifiedP1_components.yaml"
            )
        )
        self.assertEqual(config["region_order"], ["jpsi", "z"])
        for region, source in (
            ("jpsi", "legacy_prior_effective"),
            ("z", "single_component"),
        ):
            projected = region_data_config(config, region)
            spec = projected["prior_components"]
            self.assertTrue(spec["enabled"])
            self.assertEqual(spec["signal_fraction_source"], source)
            self.assertEqual(projected["region_name"], region)
            self.assertEqual(spec["weights"]["policy"], "resample_if_ess_ge_0p5")
            self.assertEqual(spec["weights"]["stop_if_ess_below"], 0.5)
            self.assertEqual(spec["weights"]["resample_seed"], 20260906)

    def test_region_without_the_block_keeps_historical_projection(self):
        from cms_data import load_config
        from joint_data import region_data_config, resolve_joint_config

        config = resolve_joint_config(
            load_config(REPO_ROOT / "configs_joint" / "cms_Joint_runE.yaml")
        )
        projected = region_data_config(config, "jpsi")
        self.assertNotIn("prior_components", projected)
        self.assertNotIn("region_name", projected)


if __name__ == "__main__":
    unittest.main()
