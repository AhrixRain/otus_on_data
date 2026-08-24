#!/usr/bin/env python
"""Generic entry point for the locked ``cms_Joint`` run series."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (
    REPO_ROOT,
    REPO_ROOT / "scripts",
    REPO_ROOT / "scripts_sota",
    REPO_ROOT / "scripts_joint",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from cms_data import load_config, save_resolved_config  # noqa: E402
from device_utils import device_report, select_device  # noqa: E402
from g0_contract import evaluate_bidirectional_model  # noqa: E402
from loss import CmsJpsiDoubleMuonLossFactory  # noqa: E402
from joint_data import (  # noqa: E402
    build_joint_split_manifest,
    load_joint_regions,
    resolve_joint_config,
    write_joint_split_manifest,
)
from joint_model import build_joint_autoencoder  # noqa: E402
from joint_trainer import (  # noqa: E402
    _full_pass_step_count,
    restore_joint_checkpoint,
    run_joint_training,
)


def parse_args(default_config: Path | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one shared dimuon response model on independent mass regions."
    )
    parser.add_argument("--config", type=Path, default=default_config, required=default_config is None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Override the output root."
    )
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    return parser.parse_args()


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = deepcopy(config)
    if args.output_dir is not None:
        config.setdefault("paths", {})["output_root"] = str(args.output_dir)
    if args.epochs is not None:
        for stage in config["stages"]:
            if stage.get("enabled", True):
                stage["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        config.setdefault("loaders", {})["train_batch_size"] = int(args.batch_size)
        config.setdefault("loaders", {})["eval_batch_size"] = int(args.batch_size)
    if args.steps_per_epoch is not None:
        loaders = config.setdefault("loaders", {})
        loaders["steps_per_epoch"] = int(args.steps_per_epoch)
        loaders["epoch_definition"] = "fixed_steps"
    if args.smoke:
        model = config.setdefault("model", {})
        model["hidden_dims"] = [64, 64]
        model["flow_steps"] = 1
        for stage in config["stages"]:
            if stage.get("enabled", True):
                stage["epochs"] = 1
                stage["num_slices"] = 16
                stage["eval_every"] = 1
        loaders = config.setdefault("loaders", {})
        loaders["train_batch_size"] = 128
        loaders["eval_batch_size"] = 256
        loaders["steps_per_epoch"] = 2
        loaders["validation_events"] = 256
        loaders["validation_draws"] = 1
        config.setdefault("checkpoint_selection", {})["hard_gates"] = False
        final = config.setdefault("final_evaluation", {})
        final.update(
            {
                "batch_size": 256,
                "max_events": 128,
                "bootstrap_replicates": 5,
                "c2st_max_samples": 128,
                "c2st_folds": 2,
                "c2st_mlp_iterations": 10,
            }
        )
    return config


def configure_cuda_memory_limit(config: dict[str, Any], device: torch.device):
    if device.type != "cuda" or config.get("cuda_memory_limit_gb") is None:
        return None
    limit_gb = float(config["cuda_memory_limit_gb"])
    if limit_gb <= 0.0:
        raise ValueError("cuda_memory_limit_gb must be positive")
    index = device.index if device.index is not None else torch.cuda.current_device()
    total = int(torch.cuda.get_device_properties(index).total_memory)
    fraction = min(1.0, int(limit_gb * 1024**3) / total)
    torch.cuda.set_per_process_memory_fraction(fraction, device=index)
    return {
        "requested_gb": limit_gb,
        "device_total_gb": total / 1024**3,
        "allocator_fraction": fraction,
    }


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def build_loss_factories(config, region_arrays):
    masses = config["model"].get("daughter_masses")
    factories = {}
    for name in config["region_order"]:
        loss_config = _merge(config["loss"], config["regions"][name].get("loss", {}))
        factories[name] = CmsJpsiDoubleMuonLossFactory(
            region_arrays[name]["x_train"],
            region_arrays[name]["z_train"],
            loss_config,
            daughter_masses=masses,
        )
    return factories


def _write_provenance(output_dir: Path, args, config, cache_info, memory_limit) -> None:
    payload = {
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "openmp_environment": {
            "KMP_DUPLICATE_LIB_OK": os.environ.get("KMP_DUPLICATE_LIB_OK"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "config_path": str(Path(args.config).expanduser().resolve()),
        "run_label": config.get("run_label"),
        "series": config.get("series"),
        "data_cache": cache_info,
        "cuda_memory_limit": memory_limit,
    }
    path = output_dir / "provenance.json"
    if path.exists() and bool(args.resume or args.resume_checkpoint is not None):
        try:
            original = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            original = {"unreadable_original_provenance": True}
        original.setdefault("resume_invocations", []).append(payload)
        payload = original
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main(default_config: Path | None = None) -> int:
    args = parse_args(default_config)
    raw_config = load_config(args.config)
    config = resolve_joint_config(apply_overrides(raw_config, args))
    device = select_device(args.device or config.get("device", "auto"))
    memory_limit = configure_cuda_memory_limit(config, device)
    np.random.seed(int(config.get("seed", 0)))
    torch.manual_seed(int(config.get("seed", 0)))

    default_name = str(config.get("run_name", "cms_Joint"))
    run_label = str(config.get("run_label", default_name))
    if args.run_name:
        run_name = args.run_name
    elif args.smoke:
        run_name = default_name + "_smoke"
    elif args.dry_run:
        run_name = default_name + "_dryrun"
    else:
        run_name = default_name
    output_dir = Path(config["paths"]["output_root"]) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    continuing = bool(args.resume or args.resume_checkpoint is not None)
    occupied = [
        path
        for path in (output_dir / "history.json", output_dir / "best_model.pt")
        if path.exists()
    ]
    if occupied and not continuing and not args.dry_run:
        raise FileExistsError(
            f"Run directory already contains training artifacts: {occupied[0]}. "
            "Use --resume or choose a new --run-name."
        )

    effective_num_samples = args.num_samples
    if effective_num_samples is None and (args.smoke or args.dry_run):
        effective_num_samples = 1000
    print(f"cms_Joint {config.get('run_label', run_name)} on {device}")
    print(f"Output: {output_dir}")
    region_arrays, cache_info, region_configs = load_joint_regions(
        config,
        num_samples=effective_num_samples,
        use_cache=not args.no_cache,
    )
    for name in config["region_order"]:
        shapes = {key: list(value.shape) for key, value in region_arrays[name].items()}
        print(f"[{name}] {json.dumps(shapes, sort_keys=True)}")
    epoch_definition = str(
        config.get("loaders", {}).get("epoch_definition", "fixed_steps")
    ).lower()
    if epoch_definition == "full_pass":
        steps = _full_pass_step_count(
            region_arrays,
            list(config["region_order"]),
            int(config["loaders"]["train_batch_size"]),
        )
        epochs = sum(
            int(stage["epochs"])
            for stage in config["stages"]
            if stage.get("enabled", True)
        )
        print(
            f"Full-pass epoch: {steps} optimizer updates; "
            f"configured schedule: {epochs} epochs / {steps * epochs} updates"
        )

    manifest = build_joint_split_manifest(
        config,
        region_arrays,
        cache_info,
        region_configs,
        num_samples=effective_num_samples,
    )
    write_joint_split_manifest(manifest, output_dir / "joint_split_manifest.json")
    config["_joint_contract_sha256"] = manifest["contract_sha256"]
    save_resolved_config(config, output_dir / "config.resolved.json")
    report = device_report(device)
    (output_dir / "device_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_provenance(output_dir, args, config, cache_info, memory_limit)

    model = build_joint_autoencoder(
        config["model"],
        region_arrays,
        float(config.get("muon_mass_gev", 0.1056583755)),
        config["model"].get("daughter_masses"),
    ).to(device)
    loss_factories = build_loss_factories(config, region_arrays)
    print(f"Shared model parameters: {sum(p.numel() for p in model.parameters()):,}")
    if 8 in model.encoder.condition_indices.tolist():
        raise RuntimeError(
            f"{run_label} contract violation: parent mass is a condition feature"
        )

    resume = None
    checkpoint_path = None
    if args.resume_checkpoint is not None:
        checkpoint_path = args.resume_checkpoint.expanduser().resolve()
    elif args.resume:
        checkpoint_path = output_dir / "last_model.pt"
    if checkpoint_path is not None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
        resume = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if resume.get("joint_contract_sha256") != manifest["contract_sha256"]:
            raise RuntimeError("Resume checkpoint does not match the locked joint data contract")
        restore_joint_checkpoint(model, resume)
        print(f"Resuming from {checkpoint_path}")

    if args.dry_run:
        print(f"{run_label} dry-run passed; Upsilon was not opened.")
        return 0

    _, best = run_joint_training(
        model,
        config,
        region_arrays,
        loss_factories,
        device,
        output_dir,
        resume=resume,
    )
    if args.skip_evaluation:
        print(f"Training complete: {output_dir / 'best_model.pt'}")
        return 0

    final_cfg = config.get("final_evaluation", {})
    final_report: dict[str, Any] = {
        "schema_version": 1,
        "series": config.get("series"),
        "run_label": config.get("run_label"),
        "joint_contract_sha256": manifest["contract_sha256"],
        "selected_checkpoint": {
            "global_epoch": best.get("global_epoch"),
            "stage": (best.get("stage") or {}).get("name"),
            "stage_epoch": best.get("stage_epoch"),
            "joint_selection": best.get("joint_selection"),
            "global_gate_fallback": bool(best.get("global_gate_fallback", False)),
        },
        "holdout": config.get("holdout"),
        "regions": {},
    }
    for offset, name in enumerate(config["region_order"]):
        final_report["regions"][name] = evaluate_bidirectional_model(
            model,
            region_arrays[name]["x_test"],
            region_arrays[name]["z_test"],
            daughter_masses=config["model"].get("daughter_masses"),
            device=device,
            batch_size=int(final_cfg.get("batch_size", 2048)),
            max_events=final_cfg.get("max_events"),
            bootstrap_replicates=int(final_cfg.get("bootstrap_replicates", 200)),
            c2st_max_samples=int(final_cfg.get("c2st_max_samples", 5000)),
            c2st_folds=int(final_cfg.get("c2st_folds", 5)),
            c2st_mlp_iterations=int(final_cfg.get("c2st_mlp_iterations", 250)),
            seed=int(final_cfg.get("seed", 20260821)) + 1000 * offset,
        )
    report_path = output_dir / "joint_evaluation.json"
    report_path.write_text(
        json.dumps(final_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"{run_label} complete: {output_dir / 'best_model.pt'}")
    print(f"Evaluation: {report_path}")
    if config.get("holdout", {}).get("status") == "locked_zero_shot":
        print("Upsilon remains unopened and locked for zero-shot testing.")
    else:
        print(
            "Upsilon was not opened by this run and remained excluded from "
            "training, tuning, and checkpoint selection."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
