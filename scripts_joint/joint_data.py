"""Balanced multi-region data contract for the ``cms_Joint`` series.

Each resonance keeps an independent unpaired x/z split and cache entry.  The
trainer shares model weights, not events or OT endpoint plans, across regions.
This prevents a mixed minibatch from coupling a J/psi event to a Z event.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from cms_data import (
    data_cache_metadata,
    load_and_split_cached,
    resolve_path,
)
from g0_contract import array_fingerprint


_SPLIT_KEYS = ("x_train", "x_val", "x_test", "z_train", "z_val", "z_test")


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_joint_config(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve common and per-region paths without flattening the config."""
    resolved = deepcopy(config)
    resolved.pop("_config_path", None)
    paths = resolved.setdefault("paths", {})
    repo_root = resolve_path(paths.get("repo_root", "."))
    data_root = resolve_path(paths.get("data_root", "data"), repo_root)
    output_root = resolve_path(paths.get("output_root", "outputs/cms_Joint"), repo_root)
    paths["repo_root"] = str(repo_root)
    paths["data_root"] = str(data_root)
    paths["output_root"] = str(output_root)
    if paths.get("cms_root_file") is not None:
        paths["cms_root_file"] = str(resolve_path(paths["cms_root_file"], data_root))

    regions = resolved.get("regions")
    if not isinstance(regions, dict) or len(regions) < 2:
        raise ValueError("cms_Joint requires at least two configured regions")
    order = list(resolved.get("region_order") or regions)
    if set(order) != set(regions) or len(order) != len(regions):
        raise ValueError("region_order must contain every region exactly once")
    resolved["region_order"] = order

    for name in order:
        region = regions[name]
        if str(region.get("data", {}).get("channel", "muon")).lower() not in {
            "muon", "doublemuon", "doublemuons", "mumu", "jpsi_mumu",
        }:
            label = str(resolved.get("run_label", "cms_Joint"))
            raise ValueError(f"{label} region {name!r} must use the common muon channel")
        region_paths = region.setdefault("paths", {})
        cms_source = region_paths.get("cms_root_file", paths.get("cms_root_file"))
        if cms_source is None:
            raise KeyError(f"Region {name!r} has no cms_root_file")
        region_paths["cms_root_file"] = str(resolve_path(cms_source, data_root))
        if region_paths.get("theory_prior_files"):
            region_paths["theory_prior_files"] = [
                str(resolve_path(item, data_root))
                for item in region_paths["theory_prior_files"]
            ]
        elif region_paths.get("theory_prior_file"):
            region_paths["theory_prior_file"] = str(
                resolve_path(region_paths["theory_prior_file"], data_root)
            )
        else:
            raise KeyError(f"Region {name!r} has no theory prior path")
    return resolved


def region_data_config(joint_config: dict[str, Any], region_name: str) -> dict[str, Any]:
    """Project one joint region into the existing single-channel data API."""
    region = joint_config["regions"][region_name]
    common_paths = joint_config["paths"]
    region_paths = region["paths"]
    paths = {
        "repo_root": common_paths["repo_root"],
        "data_root": common_paths["data_root"],
        "output_root": common_paths["output_root"],
        "cms_root_file": region_paths["cms_root_file"],
    }
    for key in ("theory_prior_file", "theory_prior_files", "theory_prior_weights"):
        if key in region_paths:
            paths[key] = deepcopy(region_paths[key])
    seed = int(region.get("seed", int(joint_config.get("seed", 0))))
    return {
        "paths": paths,
        "data": deepcopy(region.get("data", {"channel": "muon"})),
        "data_split": deepcopy(region["data_split"]),
        "muon_selection": deepcopy(region["muon_selection"]),
        "theory_prior_selection": deepcopy(region.get("theory_prior_selection")),
        "seed": seed,
        "float_type": joint_config.get("float_type", "float32"),
        "muon_mass_gev": joint_config.get("muon_mass_gev", 0.1056583755),
        "model": {
            "daughter_masses": deepcopy(
                joint_config.get("model", {}).get(
                    "daughter_masses", [0.1056583755, 0.1056583755]
                )
            )
        },
    }


def load_joint_regions(
    joint_config: dict[str, Any],
    *,
    num_samples: int | None,
    use_cache: bool = True,
    log=print,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict], dict[str, dict]]:
    """Load independent cached splits for every configured resonance."""
    all_arrays: dict[str, dict[str, np.ndarray]] = {}
    cache_info: dict[str, dict] = {}
    region_configs: dict[str, dict] = {}
    cache_root = Path(joint_config["paths"]["output_root"]) / ".region_cache"
    for name in joint_config["region_order"]:
        config = region_data_config(joint_config, name)
        arrays, info = load_and_split_cached(
            config,
            num_samples=num_samples,
            cache_dir=cache_root / name,
            use_cache=use_cache,
            log=lambda message, region=name: log(f"[{region}] {message}"),
        )
        all_arrays[name] = arrays
        cache_info[name] = info
        region_configs[name] = config
    return all_arrays, cache_info, region_configs


def build_joint_split_manifest(
    joint_config: dict[str, Any],
    region_arrays: dict[str, dict[str, np.ndarray]],
    cache_info: dict[str, dict],
    region_configs: dict[str, dict],
    *,
    num_samples: int | None,
) -> dict[str, Any]:
    """Create a content-addressed, two-region locked split contract."""
    regions: dict[str, Any] = {}
    for name in joint_config["region_order"]:
        arrays = region_arrays[name]
        missing = [key for key in _SPLIT_KEYS if key not in arrays]
        if missing:
            raise KeyError(f"Region {name!r} is missing split arrays: {missing}")
        regions[name] = {
            "data_contract": data_cache_metadata(region_configs[name], num_samples),
            "data_cache": cache_info[name],
            "splits": {key: array_fingerprint(arrays[key]) for key in _SPLIT_KEYS},
        }
    contract = {
        "schema_version": 1,
        "purpose": (
            f"cms_Joint {joint_config.get('run_label', 'run')} locked "
            "J/psi + Z dimuon training contract"
        ),
        "region_order": list(joint_config["region_order"]),
        "num_samples": None if num_samples is None else int(num_samples),
        "regions": regions,
        "holdout": deepcopy(joint_config.get("holdout", {})),
        "locked_test_policy": (
            "Tune and select using J/psi and Z train/validation only. The Upsilon "
            "CMS sample is excluded until architecture, checkpoint, prior, cuts, "
            "metrics, and comparison baseline are frozen."
        ),
    }
    # Cache hit/miss/status is useful provenance but is execution state, not
    # data identity. Exclude it from the digest so the same locked arrays have
    # the same contract when a run is resumed from a cache hit.
    identity = deepcopy(contract)
    for region in identity["regions"].values():
        region.pop("data_cache", None)
    contract["contract_sha256"] = _json_hash(identity)
    return contract


def write_joint_split_manifest(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
