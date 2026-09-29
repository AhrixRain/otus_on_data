#!/usr/bin/env python
"""Generic entry point for the locked ``cms_Joint`` run series."""


from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
import json
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
from identity_reference import equal_subset, reference_block  # noqa: E402
from physics import invariant_mass_np  # noqa: E402
from joint_model import build_joint_autoencoder  # noqa: E402
from joint_metrics import (  # noqa: E402
    current_noise_multipliers,
    resolve_noise_multipliers,
    set_noise_multipliers,
)
from joint_trainer import (  # noqa: E402
    _full_pass_step_count,
    restore_joint_checkpoint,
    run_joint_training,
)


CONFIG_DIR = REPO_ROOT / "configs_joint"


def available_runs() -> dict[str, Path]:
    """Map a short run id to its config.

    ``cms_Joint_runE.yaml`` is addressable as ``E``; any other
    ``cms_Joint_<id>.yaml`` (the A/B diagnostics, for instance) is addressable
    as ``<id>``.
    """
    runs = {}
    for path in sorted(CONFIG_DIR.glob("cms_Joint_*.yaml")):
        stem = path.stem[len("cms_Joint_"):]
        key = stem[len("run"):] if stem.startswith("run") and len(stem) > 3 else stem
        runs[key] = path
    return runs


def resolve_run(name: str) -> Path:
    runs = available_runs()
    for key, path in runs.items():
        if key.lower() == name.lower():
            return path
    raise SystemExit(
        f"Unknown run {name!r}. Available: {', '.join(sorted(runs)) or '(none)'}"
    )


def parse_args(default_config: Path | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one shared dimuon response model on independent mass regions."
    )
    parser.add_argument(
        "--run",
        default=None,
        help="Short run id resolved against configs_joint/cms_Joint_run<ID>.yaml "
        f"(available: {', '.join(sorted(available_runs())) or 'none'}).",
    )
    parser.add_argument("--config", type=Path, default=default_config)
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
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Skip training and re-score an existing checkpoint (default "
        "<run-name>/best_model.pt, or --resume-checkpoint) with the run's "
        "final-evaluation protocol at the configured/overridden noise level. "
        "Writes a labeled report and never overwrites joint_evaluation.json.",
    )
    parser.add_argument(
        "--evaluation-label",
        default=None,
        help="Suffix for the evaluate-only report: joint_evaluation_<label>.json.",
    )
    parser.add_argument(
        "--eval-noise-core",
        type=float,
        default=None,
        help="Override the final-evaluation shared core noise multiplier.",
    )
    parser.add_argument(
        "--eval-noise-tail",
        type=float,
        default=None,
        help="Override the final-evaluation shared tail noise multiplier.",
    )
    parser.add_argument(
        "--cycle-weight-scale",
        type=float,
        default=None,
        help="Multiply the per-stage cycle weight (beta) for every enabled "
        "stage; a probe knob for the per-event term.",
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Load weights from a checkpoint and start from the first enabled stage "
        "without resume history (warm start).",
    )
    args = parser.parse_args()
    if args.run is not None:
        if args.config is not None and default_config is None:
            parser.error("--run and --config are mutually exclusive")
        args.config = resolve_run(args.run)
    if args.config is None:
        parser.error("one of --run or --config is required")
    return args


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
    if args.cycle_weight_scale is not None:
        # Probe knob for the per-event term: multiply every enabled stage's
        # cycle weight (beta) without editing the shipped config.
        scale = float(args.cycle_weight_scale)
        if scale <= 0.0:
            raise ValueError("--cycle-weight-scale must be positive")
        for stage in config["stages"]:
            if stage.get("enabled", True):
                stage["beta"] = float(stage.get("beta", 1.0)) * scale
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


def configure_matmul_precision(config: dict[str, Any], device: torch.device):
    """Opt in to TF32 matmuls, and record exactly what was enabled.

    Why this is a flag and not a default: TF32 changes the mantissa of every
    matmul, so a run with it on is not bit-comparable with the runs already in
    the record. Everything through 2026-09-04 (Runs A-F, both A/B arms, the
    split-schedule rerun) was trained in strict float32 and stays that way. Turn
    this on for NEW experiments, where the ~1.3-2x speedup on a six-layer 512
    MLP is worth far more than mantissa bits the physics never sees.

    It is safe for this pipeline specifically because TF32 applies only to
    matmul and convolution internals. The parts of the code where float32
    precision was a real problem -- the cancellation-free invariant mass in
    scripts/physics.py, and the sorts inside the sliced-Wasserstein terms --
    are elementwise and comparison operations, which TF32 does not touch.
    """
    settings = config.get("performance") or {}
    enable = bool(settings.get("tf32", False))
    applied = {"tf32_requested": enable, "tf32_applied": False}
    if not enable or device.type != "cuda":
        return applied
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    applied["tf32_applied"] = True
    print("TF32 matmul enabled (performance.tf32); this run is not bit-comparable "
          "with the strict-float32 runs already in the record.")
    return applied


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


def assert_mass_not_conditioned(model, run_label: str = "<run>") -> None:
    """Hard contract: the explicit dimuon invariant mass is never a model input.

    Index 8 of the 14 cylindrical physics features is ``log_pair_mass``. Both the
    encoder and the decoder must mask it out. The check is per component because
    a checkpoint can in principle carry different masks.
    """
    for component_name in ("encoder", "decoder"):
        indices = getattr(model, component_name).condition_indices.tolist()
        if 8 in indices:
            raise RuntimeError(
                f"{run_label} contract violation: invariant mass (feature index 8) "
                f"is a {component_name} condition input"
            )


def assert_no_mass_anchor(config, run_label: str = "<run>") -> None:
    """Hard contract: no invariant-mass anchor term in any region.

    ``resonance_mass_w1`` is the physical-GeV windowed mass anchor and
    ``mass_w1`` is its standardized duplicate. A deliberate, declared
    mass-anchor experiment can override with ``allow_mass_anchor: true`` at the
    top level of its config. The distributed mass terms (``pair_mass_w1``,
    ``mass_kin_swd``, ``physics_swd``, the cycle) are not anchors and are left
    untouched.
    """
    if bool(config.get("allow_mass_anchor", False)):
        return
    for region in config.get("region_order", []):
        merged = dict(config.get("loss", {}))
        merged.update(config.get("regions", {}).get(region, {}).get("loss", {}))
        for key in ("resonance_mass_w1", "mass_w1"):
            value = float(merged.get(key, 0.0) or 0.0)
            if value != 0.0:
                raise RuntimeError(
                    f"{run_label} contract violation: {key}={value} in region "
                    f"{region!r}; the invariant mass must not be used as an anchor. "
                    "Set allow_mass_anchor: true only for a declared mass-anchor "
                    "experiment."
                )


def _schedule_end_value(spec: Any, default: float) -> float:
    """End-of-stage value of a scalar or ``{start,end,schedule}`` multiplier."""
    if spec is None:
        return float(default)
    if isinstance(spec, dict):
        return float(spec.get("end", spec.get("start", default)))
    return float(spec)


def assert_mean_map_anchor_contract(config, run_label: str = "<run>") -> None:
    """Hard contract for the frozen mean-map anchor, mirroring the mass guard.

    The invariant-mass contract forbids the physical mass from being an anchor.
    This is its complement: the deterministic map captured in the declared
    noiseless reference stage is frozen, and later stochastic stages may not
    move it. The guard enforces four things:

    1. the anchor is opt-in and declared at the top level of the config;
    2. at least one enabled stage actually uses it (a weight above zero);
    3. the declared reference stage exists and is enabled;
    4. **the model stays stochastic in both directions**: every anchored stage
       keeps nonzero encoder *and* decoder core/tail noise. The anchor may not
       be used to switch the model off -- that is the whole point of keeping
       the encoder noise (audit decision, 2026-09-11).
    """
    anchor = config.get("mean_map_anchor") or {}
    anchored = [
        stage
        for stage in config.get("stages", [])
        if stage.get("enabled", True)
        and _schedule_end_value(stage.get("mean_map_anchor_weight"), 0.0) > 0.0
    ]
    if not anchored:
        return
    if not anchor.get("enabled", False):
        raise RuntimeError(
            f"{run_label} contract violation: a stage sets mean_map_anchor_weight "
            "but mean_map_anchor.enabled is not true. Declare the anchor at the "
            "top level of the config."
        )
    enabled_names = {
        str(stage.get("name"))
        for stage in config.get("stages", [])
        if stage.get("enabled", True)
    }
    reference_stage = str(anchor.get("reference_stage", ""))
    if reference_stage not in enabled_names:
        raise RuntimeError(
            f"{run_label} contract violation: mean_map_anchor.reference_stage="
            f"{reference_stage!r} is not an enabled stage ({sorted(enabled_names)})."
        )
    if int(anchor.get("events_per_region", 0)) < 2:
        raise RuntimeError(
            f"{run_label} contract violation: mean_map_anchor.events_per_region "
            "must be at least 2."
        )
    for stage in anchored:
        name = str(stage.get("name"))
        for component in ("encoder", "decoder"):
            multipliers = []
            for kind, default in (("core", 1.0), ("tail", 0.0)):
                override = stage.get(f"{component}_{kind}_noise_multiplier")
                shared = stage.get(f"{kind}_noise_multiplier")
                multipliers.append(
                    _schedule_end_value(
                        override if override is not None else shared, default
                    )
                )
            if all(value <= 0.0 for value in multipliers):
                raise RuntimeError(
                    f"{run_label} contract violation: stage {name!r} anchors the "
                    f"mean map but switches the {component} noise off. The joint "
                    "model must stay stochastic in both directions; the anchor "
                    "must not be used to freeze it."
                )


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


def _evaluate_bidirectional_with_noise(
    model,
    *args,
    noise_multipliers=None,
    **kwargs,
):
    """Score the final report under an explicit noise policy, then restore.

    The final evaluation must not inherit the selected checkpoint's training
    noise: that would report a different map than the deterministic one used
    for selection. Defaults to the deterministic map; override with
    ``final_evaluation.noise_multipliers``.
    """
    previous = current_noise_multipliers(model)
    set_noise_multipliers(model, resolve_noise_multipliers(noise_multipliers))
    try:
        return evaluate_bidirectional_model(model, *args, **kwargs)
    finally:
        set_noise_multipliers(model, previous)


def _write_provenance(output_dir: Path, args, config, cache_info, memory_limit,
                      precision=None) -> None:
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
        "matmul_precision": precision,
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
    precision = configure_matmul_precision(config, device)
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
    evaluate_only = bool(args.evaluate_only)
    continuing = bool(
        args.resume or args.resume_checkpoint is not None or evaluate_only
    )
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
    region_arrays, cache_info, region_configs, pair_indices = load_joint_regions(
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
        pair_indices=pair_indices,
    )
    if not evaluate_only:
        # Evaluate-only is read-only on the run directory apart from its own
        # labeled report: do not rewrite provenance/manifest/config metadata.
        write_joint_split_manifest(manifest, output_dir / "joint_split_manifest.json")
        config["_joint_contract_sha256"] = manifest["contract_sha256"]
        save_resolved_config(config, output_dir / "config.resolved.json")
        report = device_report(device)
        (output_dir / "device_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _write_provenance(output_dir, args, config, cache_info, memory_limit,
                          precision=precision)

    model = build_joint_autoencoder(
        config["model"],
        region_arrays,
        float(config.get("muon_mass_gev", 0.1056583755)),
        config["model"].get("daughter_masses"),
    ).to(device)
    loss_factories = build_loss_factories(config, region_arrays)
    print(f"Shared model parameters: {sum(p.numel() for p in model.parameters()):,}")
    # Hard contracts for the joint series: the explicit invariant mass is never
    # a model input, and it is never used as an anchor term.
    assert_mass_not_conditioned(model, run_label)
    assert_no_mass_anchor(config, run_label)
    # Complementary contract: later stochastic stages may not move the frozen
    # deterministic mean map, and may not switch the bidirectional noise off.
    assert_mean_map_anchor_contract(config, run_label)

    if args.init_checkpoint is not None:
        init_path = args.init_checkpoint.expanduser().resolve()
        if not init_path.exists():
            raise FileNotFoundError(f"Init checkpoint not found: {init_path}")
        init_ckpt = torch.load(init_path, map_location="cpu", weights_only=False)
        restore_joint_checkpoint(model, init_ckpt)
        print(f"Initialized from {init_path}")

    resume = None
    checkpoint_path = None
    if args.resume_checkpoint is not None:
        checkpoint_path = args.resume_checkpoint.expanduser().resolve()
    elif args.resume:
        checkpoint_path = output_dir / "last_model.pt"
    if evaluate_only and checkpoint_path is None:
        checkpoint_path = output_dir / "best_model.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"--evaluate-only needs a checkpoint: {checkpoint_path} not found"
            )
    if checkpoint_path is not None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
        resume = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if resume.get("joint_contract_sha256") != manifest["contract_sha256"]:
            raise RuntimeError("Resume checkpoint does not match the locked joint data contract")
        restore_joint_checkpoint(model, resume)
        print(f"Resuming from {checkpoint_path}")

    if args.dry_run and not evaluate_only:
        print(f"{run_label} dry-run passed; Upsilon was not opened.")
        return 0

    if evaluate_only:
        print(f"Evaluate-only: scoring {checkpoint_path}")
        best = {
            "global_epoch": resume.get("global_epoch"),
            "stage": resume.get("stage"),
            "stage_epoch": resume.get("stage_epoch"),
            "joint_selection": resume.get("joint_selection"),
            "noise_multipliers": resume.get("noise_multipliers"),
        }
    else:
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
    final_spec = dict(final_cfg.get("noise_multipliers") or {})
    if args.eval_noise_core is not None or args.eval_noise_tail is not None:
        # The resolved config may carry explicit encoder_*/decoder_* keys;
        # they would override a shared core/tail value, so drop them first.
        core = final_spec.get("core", 0.0)
        tail = final_spec.get("tail", 0.0)
        if args.eval_noise_core is not None:
            core = float(args.eval_noise_core)
        if args.eval_noise_tail is not None:
            tail = float(args.eval_noise_tail)
        for key in ("encoder_core", "encoder_tail", "decoder_core", "decoder_tail"):
            final_spec.pop(key, None)
        final_spec["core"] = core
        final_spec["tail"] = tail
    final_noise = resolve_noise_multipliers(final_spec)
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
            "checkpoint_noise_multipliers": best.get("noise_multipliers"),
        },
        "evaluation_noise_multipliers": final_noise,
        "holdout": config.get("holdout"),
        "regions": {},
    }
    masses = config["model"].get("daughter_masses")
    for offset, name in enumerate(config["region_order"]):
        eval_seed = int(final_cfg.get("seed", 20260821)) + 1000 * offset
        region_report = _evaluate_bidirectional_with_noise(
            model,
            region_arrays[name]["x_test"],
            region_arrays[name]["z_test"],
            daughter_masses=masses,
            device=device,
            batch_size=int(final_cfg.get("batch_size", 2048)),
            max_events=final_cfg.get("max_events"),
            bootstrap_replicates=int(final_cfg.get("bootstrap_replicates", 200)),
            c2st_max_samples=int(final_cfg.get("c2st_max_samples", 5000)),
            c2st_folds=int(final_cfg.get("c2st_folds", 5)),
            c2st_mlp_iterations=int(final_cfg.get("c2st_mlp_iterations", 250)),
            seed=eval_seed,
            noise_multipliers=final_noise,
        )
        # What the same metrics read for the identity map and for pure
        # finite-sample noise, on the same event set. Without these two lines a
        # reader cannot tell a model that unfolds from one that looks good
        # because the prior already looks like the data (memory.md 7.2).
        # evaluate_bidirectional_model draws its equal-count pair with
        # ``eval_seed``, so reusing that seed reproduces the scored events.
        x_equal, z_equal = equal_subset(
            region_arrays[name]["x_test"],
            region_arrays[name]["z_test"],
            max_events=final_cfg.get("max_events"),
            seed=eval_seed,
        )
        directions = region_report["directions"]
        model_metrics = {
            "latent_mass_ks": directions["x_to_z"]["mass"]["ks"],
            "latent_mass_w1_gev": directions["x_to_z"]["mass"]["w1_gev"],
            "latent_mass_width_rel_error": abs(
                directions["x_to_z"]["mass"]["fake_std_gev"]
                - directions["x_to_z"]["mass"]["real_std_gev"]
            )
            / max(directions["x_to_z"]["mass"]["real_std_gev"], 1.0e-12),
            "latent_pair_pt_ks": directions["x_to_z"]["pair_pt"]["ks"],
            "direct_mass_ks": directions["z_to_x"]["mass"]["ks"],
            "direct_mass_w1_gev": directions["z_to_x"]["mass"]["w1_gev"],
            "direct_mass_width_rel_error": abs(
                directions["z_to_x"]["mass"]["fake_std_gev"]
                - directions["z_to_x"]["mass"]["real_std_gev"]
            )
            / max(directions["z_to_x"]["mass"]["real_std_gev"], 1.0e-12),
            "direct_pair_pt_ks": directions["z_to_x"]["pair_pt"]["ks"],
        }
        identity_block = reference_block(
            "latent",
            real_values=z_equal,
            identity_values=x_equal,
            reference_pool=region_arrays[name]["z_test"],
            model_metrics=model_metrics,
            mass_fn=lambda values: invariant_mass_np(
                values, daughter_masses=masses, stable=False
            ),
            seed=eval_seed + 500,
        )
        identity_block.update(
            reference_block(
                "direct",
                real_values=x_equal,
                identity_values=z_equal,
                reference_pool=region_arrays[name]["x_test"],
                model_metrics=model_metrics,
                mass_fn=lambda values: invariant_mass_np(
                    values, daughter_masses=masses, stable=True
                ),
                seed=eval_seed + 600,
            )
        )
        region_report["identity_reference"] = identity_block
        final_report["regions"][name] = region_report
    if args.evaluation_label:
        report_path = output_dir / f"joint_evaluation_{args.evaluation_label}.json"
    elif evaluate_only:
        report_path = output_dir / "joint_evaluation_reeval.json"
    else:
        report_path = output_dir / "joint_evaluation.json"
    if evaluate_only and report_path.exists():
        raise FileExistsError(
            f"evaluate-only report already exists: {report_path}; "
            "choose another --evaluation-label"
        )
    report_path.write_text(
        json.dumps(final_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if evaluate_only:
        print(f"{run_label} evaluate-only ({checkpoint_path})")
    else:
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
