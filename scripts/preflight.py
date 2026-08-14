#!/usr/bin/env python
"""Fail-fast preflight validation for a configured CMS dilepton workflow.

Validates and reports, without creating any production run directory:

  - both input files exist, with size and full-content SHA-256 fingerprints;
  - the CMS ROOT tree names, total event count, and required branches;
  - the MG5 HDF5 dataset key, shape, dtype, and finite-value checks;
  - the complete selected/split arrays (train/validation/test for both
    domains), including whether any configured data cap is active;
  - the resolved loader policy, batches per epoch, and the number of events
    repeated because of unequal domain sizes;
  - the selected compute device and documented host/accelerator memory
    estimates.

This script reuses the same keyed on-disk cache as ``train.py`` (it can write
a new cache entry under ``<output_root>/.plot_cache``), but it never creates
or overwrites a run directory and never writes checkpoints.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cms_data import (
    inspect_cms_root,
    inspect_theory_prior,
    load_and_split_cached,
    load_config,
    resolve_config,
)
from cms_training import build_loaders
from device_utils import device_report, select_device


def estimate_memory(
    arrays: dict[str, np.ndarray],
    loader_info: dict[str, Any],
) -> dict[str, Any]:
    """Documented approximations of host and accelerator memory needs."""
    host_array_bytes = int(sum(arr.nbytes for arr in arrays.values()))
    train_batch_size = int(loader_info["train_batch_size"])
    num_slices = int(loader_info.get("num_slices", 1000))
    # Two raw float32 [B, 8] batches plus the two sorted [B, num_slices]
    # projection matrices kept during a training step.
    batch_bytes = train_batch_size * 8 * 4 * 2
    projection_bytes = train_batch_size * num_slices * 4 * 2
    estimate = {
        "host_split_arrays_bytes": host_array_bytes,
        "host_split_arrays_mib": round(host_array_bytes / (1024**2), 1),
        "per_train_step_accelerator_bytes": int(batch_bytes + projection_bytes),
        "per_train_step_accelerator_mib": round(
            (batch_bytes + projection_bytes) / (1024**2), 1
        ),
        "note": (
            "The accelerator figure is the raw data/projection working set only; "
            "model parameters, gradients, optimizer state, and framework overhead "
            "are additional. The host figure is the cached split arrays only; "
            "loading, caching, and Python overhead are additional."
        ),
    }
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        estimate["cuda_memory"] = {
            "total_bytes": int(total),
            "free_bytes": int(free),
        }
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        try:
            estimate["mps_memory"] = {
                "current_allocated_bytes": int(torch.mps.current_allocated_memory()),
                "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
            }
        except Exception:
            pass
    return estimate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu.")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Optional explicit smoke/debug cap for the selection/split pass.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for the machine-readable JSON report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = resolve_config(load_config(args.config))
    paths = config["paths"]
    data_config = config.get("data", {})
    channel = str(data_config.get("channel", "electron"))

    print("Preflight for:", args.config.resolve())
    print("Data root:", paths["data_root"])
    print("CMS ROOT file:", paths["cms_root_file"])
    print("MG5 HDF5 prior:", paths["theory_prior_file"])

    problems: list[str] = []

    cms_report = inspect_cms_root(
        paths["cms_root_file"],
        config.get("muon_selection", {}),
        channel=channel,
    )
    prior_report = inspect_theory_prior(paths["theory_prior_file"])
    if cms_report.get("error"):
        problems.append(f"CMS ROOT: {cms_report['error']}")
    if prior_report.get("error"):
        problems.append(f"MG5 HDF5: {prior_report['error']}")

    device = select_device(args.device)
    report = device_report(device)

    split_report: dict[str, Any] = {}
    loader_report: dict[str, Any] = {}
    memory_report: dict[str, Any] = {}
    try:
        arrays, cache_info = load_and_split_cached(
            config,
            num_samples=args.num_samples,
            cache_dir=None,
            use_cache=True,
        )
        split_report = {
            "split_counts": {key: int(len(value)) for key, value in arrays.items()},
            "cms_selected_total": int(
                sum(len(arrays[key]) for key in ("x_train", "x_val", "x_test"))
            ),
            "mg5_selected_total": int(
                sum(len(arrays[key]) for key in ("z_train", "z_val", "z_test"))
            ),
            "data_caps": {
                key: config.get("data_split", {}).get(key)
                for key in ("train_max", "val_max", "test_max")
            },
            "data_cache": cache_info,
        }
        split_report["any_cap_active"] = any(
            value is not None for value in split_report["data_caps"].values()
        )
        if args.num_samples is not None:
            split_report["num_samples_override"] = int(args.num_samples)

        train_loaders, eval_loaders, loader_info = build_loaders(
            config,
            arrays,
            batch_size_override=None,
            device=device,
        )
        stage = next(
            (stage for stage in config["stages"] if stage.get("enabled", True)),
            {},
        )
        loader_info["num_slices"] = int(stage.get("num_slices", 1000))
        loader_info["n_train_x"] = int(len(arrays["x_train"]))
        loader_info["n_train_z"] = int(len(arrays["z_train"]))
        loader_info["n_val_x"] = int(len(arrays["x_val"]))
        loader_info["n_val_z"] = int(len(arrays["z_val"]))
        loader_report = dict(loader_info)
        memory_report = estimate_memory(arrays, loader_info)
        del train_loaders, eval_loaders
    except Exception as exc:
        problems.append(f"data/split/loader validation: {type(exc).__name__}: {exc}")

    if cms_report.get("missing_branches"):
        problems.append(
            "missing ROOT branches: " + ", ".join(cms_report["missing_branches"])
        )

    output = {
        "config": str(args.config.resolve()),
        "resolved_run_name": config.get("run_name"),
        "loss": config.get("loss", {}),
        "samplers": {
            "train_sampler": config.get("loaders", {}).get("train_sampler"),
            "eval_sampler": config.get("loaders", {}).get("eval_sampler"),
        },
        "cms_root": cms_report,
        "theory_prior": prior_report,
        "data": split_report,
        "loaders": loader_report,
        "memory": memory_report,
        "device_report": report,
        "problems": problems,
        "ok": not problems,
    }

    print("--- CMS ROOT ---")
    print(json.dumps(cms_report, indent=2, sort_keys=True))
    print("--- MG5 HDF5 ---")
    print(json.dumps(prior_report, indent=2, sort_keys=True))
    print("--- Data / splits ---")
    print(json.dumps(split_report, indent=2, sort_keys=True))
    print("--- Loaders ---")
    print(json.dumps(loader_report, indent=2, sort_keys=True))
    print("--- Memory ---")
    print(json.dumps(memory_report, indent=2, sort_keys=True))
    print("--- Device ---")
    print(json.dumps(report, indent=2, sort_keys=True))

    if args.json_output is not None:
        json_path = args.json_output.expanduser().resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("Wrote JSON report:", json_path)

    if problems:
        print("PREFLIGHT FAILED:")
        for problem in problems:
            print("  -", problem)
        return 1
    print("PREFLIGHT PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
