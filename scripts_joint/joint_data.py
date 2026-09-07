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
from paired_data import (
    is_paired_channel,
    load_paired_region,
    paired_cache_metadata,
    resolve_paired_region_paths,
)


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
    if not isinstance(regions, dict) or not regions:
        raise ValueError("cms_Joint requires at least one configured region")
    # The joint programme's whole point is >= 2 CMS regions sharing one set of
    # weights, so that requirement stands for the muon channel. A paired
    # benchmark (ppzee, ppttbar) is a single-region development bench for the
    # per-event closure test and is exempt: memory.md section 7 step 3.
    if not all(
        is_paired_channel(region.get("data", {}).get("channel", "muon"))
        for region in regions.values()
    ) and len(regions) < 2:
        raise ValueError("cms_Joint requires at least two configured regions")
    order = list(resolved.get("region_order") or regions)
    if set(order) != set(regions) or len(order) != len(regions):
        raise ValueError("region_order must contain every region exactly once")
    resolved["region_order"] = order

    for name in order:
        region = regions[name]
        channel = region.get("data", {}).get("channel", "muon")
        if is_paired_channel(channel):
            # Paired benchmarks carry their own truth; they have no CMS ROOT
            # file and no MG5 theory prior, so the muon path below does not
            # apply to them.
            resolve_paired_region_paths(region.setdefault("paths", {}), data_root, name)
            continue
        if str(channel).lower() not in {
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
    if is_paired_channel(region.get("data", {}).get("channel", "muon")):
        # A paired benchmark has no ROOT file, no theory prior and no muon
        # selection; its two HDF5 files are already the selected sample.
        return {
            "paths": {
                "repo_root": common_paths["repo_root"],
                "data_root": common_paths["data_root"],
                "output_root": common_paths["output_root"],
                "paired_train_file": region_paths["paired_train_file"],
                "paired_test_file": region_paths["paired_test_file"],
            },
            "data": deepcopy(region.get("data", {})),
            "data_split": deepcopy(region["data_split"]),
            "seed": int(region.get("seed", int(joint_config.get("seed", 0)))),
            "float_type": joint_config.get("float_type", "float32"),
            "model": {
                "daughter_masses": deepcopy(
                    joint_config.get("model", {}).get(
                        "daughter_masses", [0.1056583755, 0.1056583755]
                    )
                )
            },
        }
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
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, dict],
    dict[str, dict],
    dict[str, dict],
]:
    """Load independent cached splits for every configured resonance.

    Returns ``(arrays, cache_info, region_configs, pair_indices)``.
    ``pair_indices`` is empty for CMS muon regions -- the open data has no
    truth partner to withhold -- and carries the withheld pairing for paired
    benchmark regions. It goes into the split manifest and NEVER into the
    trainer; see ``scripts_joint/paired_data.py``.
    """
    all_arrays: dict[str, dict[str, np.ndarray]] = {}
    cache_info: dict[str, dict] = {}
    region_configs: dict[str, dict] = {}
    pair_indices: dict[str, dict] = {}
    cache_root = Path(joint_config["paths"]["output_root"]) / ".region_cache"
    for name in joint_config["region_order"]:
        config = region_data_config(joint_config, name)
        region_log = lambda message, region=name: log(f"[{region}] {message}")
        if is_paired_channel(
            joint_config["regions"][name].get("data", {}).get("channel", "muon")
        ):
            arrays, info, pair_index = load_paired_region(
                config, num_samples=num_samples, log=region_log
            )
            pair_indices[name] = pair_index
        else:
            arrays, info = load_and_split_cached(
                config,
                num_samples=num_samples,
                cache_dir=cache_root / name,
                use_cache=use_cache,
                log=region_log,
            )
        all_arrays[name] = arrays
        cache_info[name] = info
        region_configs[name] = config
    return all_arrays, cache_info, region_configs, pair_indices


def build_joint_split_manifest(
    joint_config: dict[str, Any],
    region_arrays: dict[str, dict[str, np.ndarray]],
    cache_info: dict[str, dict],
    region_configs: dict[str, dict],
    *,
    num_samples: int | None,
    pair_indices: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """Create a content-addressed, two-region locked split contract.

    ``pair_indices`` is the withheld event pairing for paired benchmark
    regions. It is recorded here, in the provenance document, precisely
    because this is the one place it can be audited without being anywhere
    near the training loop. A region with no entry is unchanged, so CMS muon
    manifests are byte-identical to those written before this argument existed.
    """
    pair_indices = pair_indices or {}
    regions: dict[str, Any] = {}
    for name in joint_config["region_order"]:
        arrays = region_arrays[name]
        missing = [key for key in _SPLIT_KEYS if key not in arrays]
        if missing:
            raise KeyError(f"Region {name!r} is missing split arrays: {missing}")
        paired = is_paired_channel(
            joint_config["regions"][name].get("data", {}).get("channel", "muon")
        )
        regions[name] = {
            "data_contract": (
                paired_cache_metadata(region_configs[name], num_samples)
                if paired
                else data_cache_metadata(region_configs[name], num_samples)
            ),
            "data_cache": cache_info[name],
            "splits": {key: array_fingerprint(arrays[key]) for key in _SPLIT_KEYS},
        }
        if name in pair_indices:
            regions[name]["pair_index"] = pair_indices[name]
    # The two strings below are the J/psi + Z contract's own wording, kept
    # byte-identical so every existing CMS manifest and contract hash is
    # unchanged. A paired benchmark is a different experiment and says so.
    if pair_indices:
        purpose = (
            f"cms_Joint {joint_config.get('run_label', 'run')} locked paired "
            "benchmark training contract (trained unpaired, scored per event)"
        )
        locked_test_policy = (
            "Train and select on the train/validation splits only. The event "
            "pairing recorded under regions.<name>.pair_index is withheld from "
            "training and is used only by scripts_joint/paired_closure.py at "
            "scoring time."
        )
    else:
        purpose = (
            f"cms_Joint {joint_config.get('run_label', 'run')} locked "
            "J/psi + Z dimuon training contract"
        )
        locked_test_policy = (
            "Tune and select using J/psi and Z train/validation only. The Upsilon "
            "CMS sample is excluded until architecture, checkpoint, prior, cuts, "
            "metrics, and comparison baseline are frozen."
        )
    contract = {
        "schema_version": 1,
        "purpose": purpose,
        "region_order": list(joint_config["region_order"]),
        "num_samples": None if num_samples is None else int(num_samples),
        "regions": regions,
        "holdout": deepcopy(joint_config.get("holdout", {})),
        "locked_test_policy": locked_test_policy,
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
