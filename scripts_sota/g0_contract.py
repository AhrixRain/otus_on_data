"""Locked data contract and balanced bidirectional diagnostics for Run G0."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cms_data import data_cache_metadata
from physics import invariant_mass_np, validate_daughter_masses

try:
    from .evaluation import c2st_suite, cylindrical_features_np
except ImportError:
    from evaluation import c2st_suite, cylindrical_features_np

try:
    from scipy.stats import energy_distance, ks_2samp, wasserstein_distance

    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False


_SPLIT_KEYS = ("x_train", "x_val", "x_test", "z_train", "z_val", "z_test")


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def array_fingerprint(values: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": digest.hexdigest(),
    }


def build_split_manifest(
    config: dict,
    arrays: dict[str, np.ndarray],
    cache_info: dict,
    *,
    num_samples: int | None,
) -> dict[str, Any]:
    """Build a content-addressed split manifest used by every G comparison."""
    missing = [key for key in _SPLIT_KEYS if key not in arrays]
    if missing:
        raise KeyError(f"Cannot lock split manifest; missing arrays: {missing}")
    metadata = data_cache_metadata(config, num_samples)
    split_fingerprints = {
        key: array_fingerprint(arrays[key]) for key in _SPLIT_KEYS
    }
    contract = {
        "schema_version": 1,
        "purpose": "Run G0 locked unpaired x/z evaluation contract",
        "num_samples": None if num_samples is None else int(num_samples),
        "seed": int(config.get("seed", 0)),
        "data_cache": cache_info,
        "data_contract": metadata,
        "splits": split_fingerprints,
        "locked_test_policy": (
            "Tune on train/validation only. Read test metrics after the candidate "
            "configuration and checkpoint-selection rule are frozen."
        ),
    }
    manifest_copy = dict(contract)
    contract["contract_sha256"] = _json_hash(manifest_copy)
    return contract


def write_split_manifest(manifest: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def assert_contract_compatible(reference: dict, candidate: dict) -> None:
    """Fail if a checkpoint's resolved data semantics differ from G0."""
    reference_meta = data_cache_metadata(reference, None)
    candidate_meta = data_cache_metadata(candidate, None)
    # ``num_samples`` belongs to the CLI evaluation contract, not checkpoints.
    reference_meta.pop("num_samples", None)
    candidate_meta.pop("num_samples", None)
    if reference_meta != candidate_meta:
        raise ValueError(
            "Checkpoint data contract differs from the G0 config. Use the exact "
            "Run E/F-compatible selection and split before comparing models."
        )


def _w1(a: np.ndarray, b: np.ndarray) -> float:
    if _HAS_SCIPY:
        return float(wasserstein_distance(a, b))
    q = np.linspace(0.0, 1.0, 4097)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def _ks(a: np.ndarray, b: np.ndarray) -> float:
    if _HAS_SCIPY:
        return float(ks_2samp(a, b).statistic)
    grid = np.sort(np.concatenate([a, b]))
    return float(
        np.max(
            np.abs(
                np.searchsorted(np.sort(a), grid, side="right") / len(a)
                - np.searchsorted(np.sort(b), grid, side="right") / len(b)
            )
        )
    )


def _energy(a: np.ndarray, b: np.ndarray) -> float:
    if _HAS_SCIPY:
        return float(energy_distance(a, b))
    return float("nan")


def _pair_pt(values: np.ndarray) -> np.ndarray:
    return np.hypot(values[:, 0] + values[:, 4], values[:, 1] + values[:, 5])


def _equal_count(
    a: np.ndarray,
    b: np.ndarray,
    *,
    max_events: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    if max_events is not None:
        n = min(n, int(max_events))
    if n < 100:
        raise ValueError("G0 evaluation requires at least 100 events per distribution")
    rng = np.random.default_rng(seed)
    return (
        np.asarray(a[rng.choice(len(a), n, replace=False)], dtype=np.float32),
        np.asarray(b[rng.choice(len(b), n, replace=False)], dtype=np.float32),
    )


def _bootstrap_primary(
    mass_real: np.ndarray,
    mass_fake: np.ndarray,
    pt_real: np.ndarray,
    pt_fake: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    n = min(len(mass_real), len(mass_fake))
    collected = {"mass_w1": [], "mass_ks": [], "pair_pt_ks": []}
    for _ in range(max(1, int(replicates))):
        ia = rng.integers(0, len(mass_real), n)
        ib = rng.integers(0, len(mass_fake), n)
        collected["mass_w1"].append(_w1(mass_real[ia], mass_fake[ib]))
        collected["mass_ks"].append(_ks(mass_real[ia], mass_fake[ib]))
        collected["pair_pt_ks"].append(_ks(pt_real[ia], pt_fake[ib]))
    return {
        key: {
            "mean": float(np.mean(values)),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
        }
        for key, values in collected.items()
    }


def _projection_w1(real: np.ndarray, fake: np.ndarray, *, directions: int, seed: int) -> dict:
    pooled = np.concatenate([real, fake], axis=0)
    mean = pooled.mean(axis=0, keepdims=True)
    std = np.maximum(pooled.std(axis=0, keepdims=True), 1e-8)
    real = (real - mean) / std
    fake = (fake - mean) / std
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(int(directions), real.shape[1]))
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    values = [_w1(real @ direction, fake @ direction) for direction in vectors]
    return {
        "directions": int(directions),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def distribution_report(
    real: np.ndarray,
    fake: np.ndarray,
    *,
    daughter_masses,
    stable_mass: bool,
    max_events: int | None,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    real, fake = _equal_count(real, fake, max_events=max_events, seed=seed)
    mass_real = invariant_mass_np(real, daughter_masses=daughter_masses, stable=stable_mass)
    mass_fake = invariant_mass_np(fake, daughter_masses=daughter_masses, stable=stable_mass)
    pt_real = _pair_pt(real)
    pt_fake = _pair_pt(fake)
    features_real = cylindrical_features_np(
        real, daughter_masses, stable_mass=stable_mass
    )
    features_fake = cylindrical_features_np(
        fake, daughter_masses, stable_mass=stable_mass
    )
    feature_ks = np.asarray(
        [_ks(features_real[:, i], features_fake[:, i]) for i in range(features_real.shape[1])]
    )
    return {
        "events_per_distribution": int(len(real)),
        "mass": {
            "w1_gev": _w1(mass_real, mass_fake),
            "ks": _ks(mass_real, mass_fake),
            "energy": _energy(mass_real, mass_fake),
            "real_mean_gev": float(mass_real.mean()),
            "fake_mean_gev": float(mass_fake.mean()),
            "real_std_gev": float(mass_real.std()),
            "fake_std_gev": float(mass_fake.std()),
        },
        "pair_pt": {
            "w1_gev": _w1(pt_real, pt_fake),
            "ks": _ks(pt_real, pt_fake),
            "energy": _energy(pt_real, pt_fake),
        },
        "feature_ks": {
            "mean": float(feature_ks.mean()),
            "median": float(np.median(feature_ks)),
            "max": float(feature_ks.max()),
            "per_feature": feature_ks.tolist(),
        },
        "projected_w1": _projection_w1(
            features_real, features_fake, directions=128, seed=seed + 1
        ),
        "bootstrap": _bootstrap_primary(
            mass_real,
            mass_fake,
            pt_real,
            pt_fake,
            replicates=bootstrap_replicates,
            seed=seed + 2,
        ),
    }


@torch.inference_mode()
def transform_batches(function, values: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    outputs = []
    for start in range(0, len(values), int(batch_size)):
        batch = torch.as_tensor(
            np.ascontiguousarray(values[start : start + batch_size]),
            dtype=torch.float32,
            device=device,
        )
        output = function(batch)
        if isinstance(output, (tuple, list)):
            output = output[0]
        outputs.append(output.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def evaluate_bidirectional_model(
    model,
    x_test: np.ndarray,
    z_test: np.ndarray,
    *,
    daughter_masses,
    device: torch.device,
    batch_size: int = 8192,
    max_events: int | None = None,
    bootstrap_replicates: int = 200,
    c2st_max_samples: int = 10000,
    c2st_folds: int = 5,
    c2st_mlp_iterations: int = 250,
    seed: int = 20260821,
) -> dict[str, Any]:
    """Evaluate direct and cycle directions on deterministic equal subsets."""
    masses = validate_daughter_masses(daughter_masses)
    x_equal, z_equal = _equal_count(
        x_test, z_test, max_events=max_events, seed=seed
    )
    model.eval()
    torch.manual_seed(seed)
    x_from_z = transform_batches(model.decode, z_equal, device, batch_size)
    torch.manual_seed(seed + 1)
    z_from_x = transform_batches(model.encode, x_equal, device, batch_size)
    torch.manual_seed(seed + 2)
    x_cycle = transform_batches(model.decode, z_from_x, device, batch_size)
    torch.manual_seed(seed + 3)
    z_cycle = transform_batches(model.encode, x_from_z, device, batch_size)

    reports = {
        "z_to_x": distribution_report(
            x_equal,
            x_from_z,
            daughter_masses=masses,
            stable_mass=True,
            max_events=None,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed + 10,
        ),
        "x_to_z": distribution_report(
            z_equal,
            z_from_x,
            daughter_masses=masses,
            stable_mass=False,
            max_events=None,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed + 20,
        ),
        "x_to_z_to_x": distribution_report(
            x_equal,
            x_cycle,
            daughter_masses=masses,
            stable_mass=True,
            max_events=None,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed + 30,
        ),
        "z_to_x_to_z": distribution_report(
            z_equal,
            z_cycle,
            daughter_masses=masses,
            stable_mass=False,
            max_events=None,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed + 40,
        ),
    }
    reports["z_to_x"]["c2st"] = c2st_suite(
        x_equal,
        x_from_z,
        daughter_masses=masses,
        max_samples=c2st_max_samples,
        folds=c2st_folds,
        seed=seed + 50,
        device=device,
        mlp_iterations=c2st_mlp_iterations,
        stable_mass=True,
    )
    reports["x_to_z"]["c2st"] = c2st_suite(
        z_equal,
        z_from_x,
        daughter_masses=masses,
        max_samples=c2st_max_samples,
        folds=c2st_folds,
        seed=seed + 60,
        device=device,
        mlp_iterations=c2st_mlp_iterations,
        stable_mass=False,
    )
    return {
        "schema_version": 1,
        "equal_count_events": int(len(x_equal)),
        "coverage_note": (
            "No true x/z pairs are used. Direct reports establish marginal closure; "
            "cycle reports do not establish detector-response calibration."
        ),
        "directions": reports,
    }
