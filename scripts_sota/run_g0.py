#!/usr/bin/env python
"""Run G0: lock the unpaired split and compare checkpoints bidirectionally.

This command never trains or modifies a checkpoint.  It produces one split
manifest plus equal-count, bootstrap-aware reports for every named checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "scripts_sota"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from cms_data import load_and_split_cached, load_config, resolve_config  # noqa: E402
from device_utils import device_report, select_device  # noqa: E402
from g0_contract import (  # noqa: E402
    assert_contract_compatible,
    build_split_manifest,
    evaluate_bidirectional_model,
    write_split_manifest,
)
from physics import daughter_masses_from_config  # noqa: E402
from run_e import build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Repeat for Run E, Run F, and candidate checkpoints.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--bootstrap-replicates", type=int, default=200)
    parser.add_argument("--c2st-max-samples", type=int, default=10000)
    parser.add_argument("--c2st-folds", type=int, default=5)
    parser.add_argument("--c2st-mlp-iterations", type=int, default=250)
    return parser.parse_args()


def _checkpoint_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Checkpoint must be LABEL=PATH, got {value!r}")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label or not all(character.isalnum() or character in "-_" for character in label):
        raise ValueError(f"Invalid checkpoint label {label!r}")
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return label, path


def _headline(report: dict) -> dict:
    directions = report["directions"]
    return {
        name: {
            "mass_w1_gev": values["mass"]["w1_gev"],
            "mass_ks": values["mass"]["ks"],
            "pair_pt_ks": values["pair_pt"]["ks"],
            "feature_ks_max": values["feature_ks"]["max"],
            "projected_w1_mean": values["projected_w1"]["mean"],
            "c2st_mlp_auc": (values.get("c2st") or {}).get("mlp", {}).get("c2st_auc_mean"),
            "c2st_mlp_detectability": (values.get("c2st") or {}).get("mlp", {}).get("detectability_auc"),
        }
        for name, values in directions.items()
    }


def main() -> int:
    args = parse_args()
    device = select_device(args.device)
    config = resolve_config(load_config(args.config))
    output_dir = args.output_dir or (
        Path(config["paths"]["output_root"]) / "g0_locked_evaluation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays, cache_info = load_and_split_cached(
        config,
        num_samples=args.num_samples,
        cache_dir=None,
        use_cache=True,
    )
    manifest = build_split_manifest(
        config, arrays, cache_info, num_samples=args.num_samples
    )
    write_split_manifest(manifest, output_dir / "g0_split_manifest.json")

    comparison = {
        "contract_sha256": manifest["contract_sha256"],
        "device": device_report(device),
        "checkpoints": {},
    }
    for label, checkpoint_path in map(_checkpoint_spec, args.checkpoint):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_config = checkpoint.get("config")
        if not isinstance(checkpoint_config, dict):
            raise ValueError(f"Checkpoint has no resolved config: {checkpoint_path}")
        assert_contract_compatible(config, checkpoint_config)
        model = build_model(checkpoint_config, arrays).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        report = evaluate_bidirectional_model(
            model,
            arrays["x_test"],
            arrays["z_test"],
            daughter_masses=daughter_masses_from_config(checkpoint_config),
            device=device,
            batch_size=args.batch_size,
            max_events=args.max_events,
            bootstrap_replicates=args.bootstrap_replicates,
            c2st_max_samples=args.c2st_max_samples,
            c2st_folds=args.c2st_folds,
            c2st_mlp_iterations=args.c2st_mlp_iterations,
            seed=int(config.get("seed", 0)) + 20260821,
        )
        report.update(
            {
                "label": label,
                "checkpoint": str(checkpoint_path),
                "checkpoint_epoch": checkpoint.get("epoch"),
                "contract_sha256": manifest["contract_sha256"],
            }
        )
        report_path = output_dir / f"g0_evaluation_{label}.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        comparison["checkpoints"][label] = {
            "checkpoint": str(checkpoint_path),
            "report": str(report_path),
            "headline": _headline(report),
        }

    comparison_path = output_dir / "g0_comparison.json"
    comparison_path.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))
    print("G0 manifest:", output_dir / "g0_split_manifest.json")
    print("G0 comparison:", comparison_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
