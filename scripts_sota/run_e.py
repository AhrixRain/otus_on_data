#!/usr/bin/env python
"""Run E entry point.

Builds the cylindrical-flow model and the Tier A loss factory, trains with
mass-gated checkpoint selection, then writes C2ST / stochasticity / coverage
diagnostics for the selected checkpoint.

Typical use::

    python scripts_sota/run_e.py \
      --config configs_sota/cms_Jpsi_ptj5_runE_tierA.yaml \
      --device auto --run-name run_e_probe
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "scripts_sota"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from cms_data import load_and_split_cached, load_config, resolve_config, save_resolved_config  # noqa: E402
from device_utils import device_report, select_device  # noqa: E402
from physics import daughter_masses_from_config  # noqa: E402
from cylindrical_flow import build_cylindrical_flow_autoencoder  # noqa: E402
from flow_matching import build_flow_matching_autoencoder  # noqa: E402
from reciprocal_bridge import build_reciprocal_bridge_autoencoder  # noqa: E402
from evaluation import (  # noqa: E402
    c2st_score,
    cycle_coverage,
    fixed_z_stochasticity,
    pseudo_pair_coverage,
    write_evaluation_report,
)
from g0_contract import (  # noqa: E402
    build_split_manifest,
    evaluate_bidirectional_model,
    write_split_manifest,
)
from sota_loss import SotaLossFactory  # noqa: E402
from trainer import run_sota_training  # noqa: E402


def configure_cuda_memory_limit(config: dict, device: torch.device) -> dict | None:
    """Apply an optional hard PyTorch allocator cap before CUDA tensors exist."""
    if device.type != "cuda" or config.get("cuda_memory_limit_gb") is None:
        return None
    limit_gb = float(config["cuda_memory_limit_gb"])
    if not 0.0 < limit_gb:
        raise ValueError("cuda_memory_limit_gb must be positive")
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    total_bytes = int(torch.cuda.get_device_properties(device_index).total_memory)
    requested_bytes = int(limit_gb * 1024**3)
    fraction = min(1.0, requested_bytes / total_bytes)
    torch.cuda.set_per_process_memory_fraction(fraction, device=device_index)
    return {
        "requested_gb": limit_gb,
        "device_total_gb": total_bytes / 1024**3,
        "allocator_fraction": fraction,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E: Tier A neural-OT cylindrical-flow OTUS.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="One tiny epoch with reduced model/slices.")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from <output_root>/<run-name>/last_model.pt.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Resume from an explicit checkpoint, e.g. a stage-best .pt file.",
    )
    return parser.parse_args()


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    if args.output_dir is not None:
        config.setdefault("paths", {})["output_root"] = str(args.output_dir)
    if args.epochs is not None:
        for stage in config["stages"]:
            if stage.get("enabled", True):
                stage["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        config.setdefault("loaders", {})["train_batch_size"] = int(args.batch_size)
        config.setdefault("loaders", {})["eval_batch_size"] = int(args.batch_size)
    if args.smoke:
        model = config.setdefault("model", {})
        model["hidden_dims"] = [64, 64]
        model["flow_steps"] = 1
        model["integration_steps"] = 2
        model["ot_max_batch"] = min(128, int(model.get("ot_max_batch", 128)))
        for stage in config["stages"]:
            if stage.get("enabled", True):
                stage["epochs"] = 1
                stage["num_slices"] = 16
                stage["eval_every"] = 1
        config.setdefault("loaders", {})["train_batch_size"] = 256
        config.setdefault("loaders", {})["eval_batch_size"] = 256
        sota = config.setdefault("loss", {}).setdefault("sota", {})
        sota.setdefault("sinkhorn", {})["max_batch"] = 256
        sota.setdefault("max_swd", {})["num_directions"] = 8
        config.setdefault("sota_selection_gates", {})["max_validation_events"] = 1024
        config.setdefault("sota_selection_gates", {})["enabled"] = False
        g0 = config.setdefault("g0_contract", {})
        if bool(g0.get("enabled", False)):
            g0["max_events"] = 512
            g0["bootstrap_replicates"] = 10
            g0["c2st_max_samples"] = 512
            g0["c2st_folds"] = 2
            g0["c2st_mlp_iterations"] = 20
    return config


def build_model(config, arrays):
    model_cfg = config["model"]
    masses = daughter_masses_from_config(config)
    class_name = model_cfg.get("class", "cylindrical_flow")
    if class_name == "flow_matching":
        return build_flow_matching_autoencoder(
            model_cfg,
            arrays["x_train"],
            arrays["z_train"],
            float(config.get("muon_mass_gev", 0.1056583755)),
            masses,
        )
    if class_name == "reciprocal_bridge":
        return build_reciprocal_bridge_autoencoder(
            model_cfg,
            arrays["x_train"],
            arrays["z_train"],
            float(config.get("muon_mass_gev", 0.1056583755)),
            masses,
        )
    return build_cylindrical_flow_autoencoder(
        model_cfg,
        arrays["x_train"],
        arrays["z_train"],
        float(config.get("muon_mass_gev", 0.1056583755)),
        masses,
    )


def main() -> int:
    args = parse_args()
    config = resolve_config(load_config(args.config))
    config = apply_overrides(config, args)
    device = select_device(args.device)
    memory_limit = configure_cuda_memory_limit(config, device)
    report = device_report(device)
    print("Device:", device)
    if memory_limit is not None:
        print("CUDA memory limit:", json.dumps(memory_limit, sort_keys=True))
    print("Resolved config path:", config.get("_config_path"))

    run_name = args.run_name or config.get("run_name") or "run_e"
    output_dir = Path(config["paths"]["output_root"]) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    resume = None
    if args.resume_checkpoint is not None:
        checkpoint_path = args.resume_checkpoint.expanduser().resolve()
    elif args.resume:
        checkpoint_path = output_dir / "last_model.pt"
    else:
        checkpoint_path = None
    if checkpoint_path is not None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
        resume = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        print("Resuming from:", checkpoint_path)
        print("Checkpoint epoch:", resume.get("epoch"), "stage:", (resume.get("stage") or {}).get("name"))
        # Keep the CLI config (it carries the repaired max-SW settings).  The
        # model/data sections must still match the checkpoint; any mismatch is
        # caught by load_state_dict / build_model below.

    arrays, cache_info = load_and_split_cached(
        config,
        num_samples=args.num_samples,
        cache_dir=None,
        use_cache=True,
    )
    print("Data cache:", json.dumps(cache_info, sort_keys=True))
    for key in ("x_train", "x_val", "x_test", "z_train", "z_val", "z_test"):
        print(f"  {key}: {arrays[key].shape}")

    g0_config = config.get("g0_contract", {})
    if bool(g0_config.get("enabled", False)):
        manifest = build_split_manifest(
            config, arrays, cache_info, num_samples=args.num_samples
        )
        write_split_manifest(manifest, output_dir / "g0_split_manifest.json")
        config["_g0_contract_sha256"] = manifest["contract_sha256"]
        print("G0 contract SHA-256:", manifest["contract_sha256"])

    model = build_model(config, arrays).to(device)
    if resume is not None:
        model.load_state_dict(resume["model_state_dict"])
    loss_factory = SotaLossFactory(
        arrays["x_train"],
        arrays["z_train"],
        config["loss"],
        daughter_masses=daughter_masses_from_config(config),
    )
    if resume is not None and "sota_loss_state" in resume:
        checkpoint_sota = (resume.get("config") or {}).get("loss", {}).get("sota", {}).get("max_swd", {})
        current_sota = config.get("loss", {}).get("sota", {}).get("max_swd", {})
        same_max_swd = all(
            checkpoint_sota.get(key) == current_sota.get(key)
            for key in ("p", "lr", "num_directions", "direction_grad_clip", "update_every", "distance_clamp")
        )
        if same_max_swd:
            try:
                loss_factory.load_state_dict(resume["sota_loss_state"])
                print("Restored SOTA max-SW direction state.")
            except Exception:
                print("Could not restore SOTA loss state; continuing with fresh directions.")
        else:
            print("max-SW settings changed since the checkpoint; starting with fresh slicing directions.")
    print("Model parameters:", sum(p.numel() for p in model.parameters()))
    print("SOTA loss:", {
        "sinkhorn": loss_factory.sinkhorn_enabled,
        "sinkhorn_weight": loss_factory.sinkhorn_weight,
        "monge_gap": loss_factory.monge_enabled,
        "max_swd": loss_factory.max_swd_enabled,
        "max_swd_weight": loss_factory.max_swd_weight,
    })

    save_resolved_config(config, output_dir / "config.resolved.json")
    (output_dir / "device_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.dry_run:
        print("Dry run passed:", output_dir)
        return 0

    history, best = run_sota_training(
        model,
        config,
        arrays,
        loss_factory,
        device,
        output_dir,
        resume=resume,
    )
    print("Selected checkpoint epoch:", best["epoch"], "stage:", best.get("stage", {}).get("name"))
    print("Selected gates:", best.get("validation_gates"))

    if args.skip_evaluation:
        return 0

    model.load_state_dict(best["model_state_dict"])
    model.to(device)
    model.eval()
    masses = daughter_masses_from_config(config)
    decoded_parts = []
    for start in range(0, len(arrays["z_test"]), 8192):
        batch = torch.as_tensor(arrays["z_test"][start : start + 8192], dtype=torch.float32, device=device)
        decoded_parts.append(model.decode(batch).detach().cpu().numpy())
    z_decoded = np.concatenate(decoded_parts, axis=0)
    c2st = c2st_score(
        arrays["x_test"],
        z_decoded,
        daughter_masses=masses,
        device=device,
    )
    stochasticity = fixed_z_stochasticity(
        model,
        arrays["z_test"],
        daughter_masses=masses,
        device=device,
    )
    cycle = cycle_coverage(model, arrays["x_test"], daughter_masses=masses, device=device)
    pseudo = pseudo_pair_coverage(
        model,
        arrays["z_test"],
        arrays["x_test"],
        daughter_masses=masses,
        device=device,
    )
    evaluation = {
        "checkpoint_epoch": best["epoch"],
        "c2st": c2st,
        "fixed_z_stochasticity": stochasticity,
        "cycle_coverage": cycle,
        "pseudo_pair_coverage": pseudo,
    }
    write_evaluation_report(evaluation, output_dir / "sota_evaluation.json")
    if bool(g0_config.get("enabled", False)):
        g0_evaluation = evaluate_bidirectional_model(
            model,
            arrays["x_test"],
            arrays["z_test"],
            daughter_masses=masses,
            device=device,
            batch_size=int(g0_config.get("batch_size", 8192)),
            max_events=g0_config.get("max_events"),
            bootstrap_replicates=int(g0_config.get("bootstrap_replicates", 200)),
            c2st_max_samples=int(g0_config.get("c2st_max_samples", 10000)),
            c2st_folds=int(g0_config.get("c2st_folds", 5)),
            c2st_mlp_iterations=int(g0_config.get("c2st_mlp_iterations", 250)),
            seed=int(g0_config.get("seed", 20260821)),
        )
        g0_evaluation.update(
            {
                "checkpoint_epoch": best["epoch"],
                "contract_sha256": manifest["contract_sha256"],
            }
        )
        write_evaluation_report(g0_evaluation, output_dir / "g0_evaluation.json")
    print(json.dumps(evaluation, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
