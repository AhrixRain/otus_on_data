"""Balanced multi-region trainer for the ``cms_Joint`` run series."""

from __future__ import annotations

import json
import math
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cms_training import make_stage_loss_config
try:
    from .joint_metrics import evaluate_region, score_joint_metrics
except ImportError:  # scripts_joint/ added directly to sys.path
    from joint_metrics import evaluate_region, score_joint_metrics


def _first(value):
    return value[0] if isinstance(value, (tuple, list)) else value


def _as_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def _scheduled_value(spec: Any, local_epoch: int, epochs: int) -> float:
    if not isinstance(spec, dict):
        return float(spec)
    start = float(spec["start"])
    end = float(spec.get("end", start))
    t = 0.0 if epochs <= 1 else float(local_epoch - 1) / float(epochs - 1)
    schedule = str(spec.get("schedule", "linear")).lower()
    if schedule == "linear":
        return start + (end - start) * t
    if schedule == "cosine":
        return end + (start - end) * (1.0 + math.cos(math.pi * t)) / 2.0
    if schedule == "constant":
        return start
    raise ValueError(f"Unknown schedule {schedule!r}")


def _region_weights(config: dict[str, Any]) -> dict[str, float]:
    raw = config.get("region_weights", {})
    values = {
        name: float(raw.get(name, 1.0)) for name in config["region_order"]
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in values.values()):
        raise ValueError("Every joint region weight must be finite and positive")
    total = sum(values.values())
    return {name: value / total for name, value in values.items()}


def _sample_batch(
    values: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    device: torch.device,
) -> torch.Tensor:
    replace = len(values) < batch_size
    indices = rng.choice(len(values), size=batch_size, replace=replace)
    return torch.as_tensor(
        np.ascontiguousarray(values[indices]), dtype=torch.float32, device=device
    )


def _coverage_indices(
    length: int,
    start: int,
    count: int,
    *,
    seed: int,
    stream: int,
) -> np.ndarray:
    """Return a deterministic no-repeat stream over successive full cycles.

    The selected arrays are already shuffled by the locked data split. Each
    cycle applies an additional seeded affine permutation, which is bijective
    because its multiplier is chosen coprime to the array length. This avoids
    materializing multi-million-row permutations every epoch while ensuring
    every row is visited once before that stream repeats.
    """
    length = int(length)
    start = int(start)
    count = int(count)
    if length <= 0 or start < 0 or count <= 0:
        raise ValueError("coverage stream requires length/count > 0 and start >= 0")
    positions = np.arange(start, start + count, dtype=np.int64)
    cycles = positions // length
    offsets = positions % length
    indices = np.empty(count, dtype=np.int64)
    for cycle in np.unique(cycles):
        rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), int(stream), int(cycle)])
        )
        if length == 1:
            multiplier = 1
            shift = 0
        else:
            multiplier = int(rng.integers(1, length))
            while math.gcd(multiplier, length) != 1:
                multiplier = 1 if multiplier + 1 >= length else multiplier + 1
            shift = int(rng.integers(0, length))
        mask = cycles == cycle
        indices[mask] = (multiplier * offsets[mask] + shift) % length
    return indices


def _coverage_batches(
    values: np.ndarray,
    batch_size: int,
    steps: int,
    epoch_index: int,
    *,
    seed: int,
    stream: int,
) -> np.ndarray:
    draws_per_epoch = int(batch_size) * int(steps)
    start = (int(epoch_index) - 1) * draws_per_epoch
    return _coverage_indices(
        len(values), start, draws_per_epoch, seed=seed, stream=stream
    ).reshape(int(steps), int(batch_size))


def _build_optimizer(parameters, stage: dict[str, Any]):
    name = str(stage.get("optimizer", "adamw")).lower()
    kwargs = {
        "lr": float(stage["lr"]),
        "betas": tuple(float(v) for v in stage.get("adam_betas", (0.9, 0.999))),
        "eps": float(stage.get("adam_eps", 1.0e-8)),
        "weight_decay": float(stage.get("weight_decay", 0.0)),
    }
    if name == "adam":
        return torch.optim.Adam(parameters, **kwargs)
    if name == "adamw":
        return torch.optim.AdamW(parameters, **kwargs)
    raise ValueError(f"Unknown optimizer {name!r}")


def _build_scheduler(optimizer, stage: dict[str, Any], start_epoch: int):
    name = str(stage.get("lr_schedule", "none")).lower()
    epochs = max(1, int(stage["epochs"]))
    warmup = min(max(0, int(stage.get("lr_warmup_epochs", 0))), max(0, epochs - 1))
    if name == "none":
        return None
    if name != "cosine":
        raise ValueError(f"Unknown lr_schedule {name!r}")
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs - warmup),
        eta_min=float(stage["lr"]) * float(stage.get("lr_min_factor", 0.05)),
    )
    if warmup:
        linear = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=float(stage.get("lr_warmup_start_factor", 0.2)),
            end_factor=1.0,
            total_iters=warmup,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [linear, cosine], milestones=[warmup]
        )
    else:
        scheduler = cosine
    if start_epoch > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(start_epoch - 1):
                scheduler.step()
    return scheduler


def _set_trainable(model, stage: dict[str, Any]) -> None:
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(not bool(stage.get("freeze_encoder", False)))
    for parameter in model.decoder.parameters():
        parameter.requires_grad_(not bool(stage.get("freeze_decoder", False)))


def train_joint_epoch(
    model,
    optimizer,
    region_arrays: dict[str, dict[str, np.ndarray]],
    loss_factories: dict[str, Any],
    config: dict[str, Any],
    stage: dict[str, Any],
    *,
    device: torch.device,
    seed: int,
    epoch_index: int | None = None,
) -> dict[str, float]:
    """Accumulate one independent loss per region before each shared update."""
    model.train()
    loaders = config.get("loaders", {})
    batch_size = int(loaders.get("train_batch_size", 1024))
    steps = int(loaders.get("steps_per_epoch", 100))
    weights = _region_weights(config)
    rng = np.random.default_rng(seed)
    sums: dict[str, float] = {"loss": 0.0, "grad_norm": 0.0}

    sampling = str(loaders.get("sampling", "random")).lower()
    if sampling not in {"random", "cycling_without_replacement"}:
        raise ValueError(
            "loaders.sampling must be 'random' or 'cycling_without_replacement'"
        )
    coverage: dict[tuple[str, str], np.ndarray] = {}
    if sampling == "cycling_without_replacement":
        if epoch_index is None or int(epoch_index) < 1:
            raise ValueError(
                "cycling_without_replacement requires a positive epoch_index"
            )
        base_seed = int(config.get("seed", 0))
        for region_index, name in enumerate(config["region_order"]):
            arrays = region_arrays[name]
            for domain_index, key in enumerate(("x_train", "z_train")):
                coverage[(name, key)] = _coverage_batches(
                    arrays[key],
                    batch_size,
                    steps,
                    int(epoch_index),
                    seed=base_seed,
                    stream=2 * region_index + domain_index,
                )

    for step_index in range(steps):
        optimizer.zero_grad(set_to_none=True)
        step_total = 0.0
        for name in config["region_order"]:
            arrays = region_arrays[name]
            if sampling == "cycling_without_replacement":
                x = torch.as_tensor(
                    np.ascontiguousarray(
                        arrays["x_train"][coverage[(name, "x_train")][step_index]]
                    ),
                    dtype=torch.float32,
                    device=device,
                )
                z = torch.as_tensor(
                    np.ascontiguousarray(
                        arrays["z_train"][coverage[(name, "z_train")][step_index]]
                    ),
                    dtype=torch.float32,
                    device=device,
                )
            else:
                x = _sample_batch(arrays["x_train"], batch_size, rng, device)
                z = _sample_batch(arrays["z_train"], batch_size, rng, device)
            factory = loss_factories[name]
            factory.reset_components()

            z_encoded = _first(model.encode(x))
            x_reco = _first(model.decode(z_encoded))
            x_loss = factory.x_reco_loss(x, x_reco)
            z_loss = factory.z_prior_loss(z, z_encoded)
            direct_loss = x.new_tensor(0.0)
            decoder_anchor = x.new_tensor(0.0)
            if stage["tau"] > 0.0 or stage["nu_d"] > 0.0:
                x_from_z = _first(model.decode(z))
                direct_loss = factory.x_sim_loss(x, x_from_z)
                if stage["nu_d"] > 0.0:
                    decoder_anchor = factory.decoder_anchor_loss(z, x_from_z)
            encoder_anchor = (
                factory.encoder_anchor_loss(z_encoded, x)
                if stage["nu_e"] > 0.0
                else x.new_tensor(0.0)
            )
            region_loss = (
                stage["beta"] * x_loss
                + stage["lamb"] * z_loss
                + stage["tau"] * direct_loss
                + stage["nu_e"] * encoder_anchor
                + stage["nu_d"] * decoder_anchor
            )
            weighted_loss = weights[name] * region_loss
            if not torch.isfinite(weighted_loss):
                raise FloatingPointError(f"Non-finite joint loss in region {name}")
            # Backward immediately: the J/psi graph is released before the Z
            # graph is built, keeping memory near one-region peak usage.
            weighted_loss.backward()
            step_total += _as_float(weighted_loss)
            for key, value in {
                "loss": region_loss,
                "x_loss": x_loss,
                "z_loss": z_loss,
                "direct_loss": direct_loss,
                "encoder_anchor": encoder_anchor,
                "decoder_anchor": decoder_anchor,
            }.items():
                sums[f"{name}_{key}"] = sums.get(f"{name}_{key}", 0.0) + _as_float(value)

        parameters = [
            p for p in model.parameters() if p.requires_grad and p.grad is not None
        ]
        clip = float(stage.get("gradient_clip_norm", 0.0))
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            max_norm=clip if clip > 0.0 else math.inf,
            error_if_nonfinite=True,
        )
        optimizer.step()
        sums["loss"] += step_total
        sums["grad_norm"] += _as_float(grad_norm)
    return {key: value / max(1, steps) for key, value in sums.items()}


def validate_joint(
    model,
    region_arrays: dict[str, dict[str, np.ndarray]],
    loss_factories: dict[str, Any],
    config: dict[str, Any],
    *,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    loaders = config.get("loaders", {})
    metrics = {}
    for offset, name in enumerate(config["region_order"]):
        metrics[name] = evaluate_region(
            model,
            region_arrays[name],
            loss_factories[name],
            daughter_masses=config["model"].get("daughter_masses"),
            device=device,
            batch_size=int(loaders.get("eval_batch_size", 2048)),
            max_events=loaders.get("validation_events", 8192),
            decoder_draws=int(loaders.get("validation_draws", 2)),
            seed=seed + 1000 * offset,
        )
    return metrics, score_joint_metrics(metrics, config)


def _checkpoint(
    model,
    optimizer,
    config,
    stage,
    global_epoch,
    local_epoch,
    metrics,
    selection,
) -> dict[str, Any]:
    first_step = model.encoder.steps[0]
    return {
        "schema_version": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "stage": dict(stage),
        "global_epoch": int(global_epoch),
        "stage_epoch": int(local_epoch),
        "region_validation": metrics,
        "joint_selection": selection,
        "joint_contract_sha256": config.get("_joint_contract_sha256"),
        "noise_multipliers": {
            "core": float(first_step.core_noise_multiplier),
            "tail": float(first_step.tail_noise_multiplier),
        },
    }


def restore_joint_checkpoint(model, checkpoint: dict[str, Any]) -> None:
    model.load_state_dict(checkpoint["model_state_dict"])
    noise = checkpoint.get("noise_multipliers") or {}
    model.set_noise_multipliers(
        float(noise.get("core", 1.0)), float(noise.get("tail", 1.0))
    )


def run_joint_training(
    model,
    config: dict[str, Any],
    region_arrays: dict[str, dict[str, np.ndarray]],
    loss_factories: dict[str, Any],
    device: torch.device,
    output_dir: Path,
    *,
    resume: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run all stages, restoring each stage's worst-region-best checkpoint."""
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.json"
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
        except Exception:
            history = []
    else:
        history = []

    resume_stage = str((resume or {}).get("stage", {}).get("name", ""))
    resume_local = int((resume or {}).get("stage_epoch", 0))
    resume_optimizer = (resume or {}).get("optimizer_state_dict")
    global_epoch = int((resume or {}).get("global_epoch", 0))
    if resume:
        history = [row for row in history if int(row.get("global_epoch", 0)) <= global_epoch]

    global_best_score = math.inf
    global_best_path = output_dir / "best_model.pt"
    for row in history:
        selection = row.get("joint_selection") or {}
        if selection.get("all_gates_passed"):
            global_best_score = min(
                global_best_score, float(selection.get("selection_score", math.inf))
            )

    reached_resume_stage = not bool(resume_stage)
    last_payload = None
    for stage in config["stages"]:
        if not stage.get("enabled", True):
            continue
        name = str(stage["name"])
        if resume_stage and not reached_resume_stage:
            if name != resume_stage:
                continue
            reached_resume_stage = True
        start_epoch = resume_local + 1 if name == resume_stage else 1
        if start_epoch > int(stage["epochs"]):
            resume_stage = ""
            continue

        _set_trainable(model, stage)
        parameters = [p for p in model.parameters() if p.requires_grad]
        if not parameters:
            raise ValueError(f"Stage {name!r} has no trainable parameters")
        optimizer = _build_optimizer(parameters, stage)
        if name == resume_stage and resume_optimizer:
            try:
                optimizer.load_state_dict(resume_optimizer)
            except (ValueError, RuntimeError):
                pass
        scheduler = _build_scheduler(optimizer, stage, start_epoch)
        stage_best_score = math.inf
        stage_best_path = output_dir / f"best_{name}.pt"

        for local_epoch in range(start_epoch, int(stage["epochs"]) + 1):
            global_epoch += 1
            started = time.time()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            stage_resolved = dict(stage)
            stage_resolved.update(
                make_stage_loss_config(stage, local_epoch, int(stage["epochs"]))
            )
            core = _scheduled_value(
                stage.get("core_noise_multiplier", 1.0),
                local_epoch,
                int(stage["epochs"]),
            )
            tail = _scheduled_value(
                stage.get("tail_noise_multiplier", 0.0),
                local_epoch,
                int(stage["epochs"]),
            )
            model.set_noise_multipliers(core, tail)
            for factory in loss_factories.values():
                factory.set_num_slices(int(stage_resolved["num_slices"]))
            train = train_joint_epoch(
                model,
                optimizer,
                region_arrays,
                loss_factories,
                config,
                stage_resolved,
                device=device,
                seed=int(config.get("seed", 0)) + 10000 * global_epoch,
                epoch_index=global_epoch,
            )
            if scheduler is not None:
                scheduler.step()
            row: dict[str, Any] = {
                "global_epoch": global_epoch,
                "stage": name,
                "stage_epoch": local_epoch,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "core_noise_multiplier": core,
                "tail_noise_multiplier": tail,
                "seconds": time.time() - started,
                "train": train,
            }
            should_eval = (
                local_epoch == 1
                or local_epoch == int(stage["epochs"])
                or local_epoch % int(stage.get("eval_every", 5)) == 0
            )
            if should_eval:
                metrics, selection = validate_joint(
                    model,
                    region_arrays,
                    loss_factories,
                    config,
                    device=device,
                    seed=int(config.get("seed", 0)) + global_epoch,
                )
                row["region_validation"] = metrics
                row["joint_selection"] = selection
                payload = _checkpoint(
                    model,
                    optimizer,
                    config,
                    stage,
                    global_epoch,
                    local_epoch,
                    metrics,
                    selection,
                )
                last_payload = payload
                score = float(selection["selection_score"])
                if score < stage_best_score:
                    stage_best_score = score
                    torch.save(payload, stage_best_path)
                hard = bool(config.get("checkpoint_selection", {}).get("hard_gates", True))
                eligible = selection["all_gates_passed"] or not hard
                if eligible and score < global_best_score:
                    global_best_score = score
                    torch.save(payload, global_best_path)
                torch.save(payload, output_dir / "last_model.pt")
                print(
                    f"[{name} {local_epoch}/{stage['epochs']}] "
                    f"joint={score:.4g} worst={selection['worst_region']} "
                    f"gates={selection['all_gates_passed']}"
                    + (
                        f" peak_cuda={torch.cuda.max_memory_reserved(device) / 1024**3:.2f}GiB"
                        if device.type == "cuda"
                        else ""
                    )
                )
            if device.type == "cuda":
                row["peak_cuda_allocated_gb"] = (
                    torch.cuda.max_memory_allocated(device) / 1024**3
                )
                row["peak_cuda_reserved_gb"] = (
                    torch.cuda.max_memory_reserved(device) / 1024**3
                )
            history.append(row)
            history_path.write_text(
                json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

        if not stage_best_path.exists():
            raise RuntimeError(f"Stage {name!r} produced no validation checkpoint")
        stage_best = torch.load(stage_best_path, map_location="cpu", weights_only=False)
        restore_joint_checkpoint(model, stage_best)
        if bool(config.get("require_stage_gate_pass", False)) and not stage_best[
            "joint_selection"
        ]["all_gates_passed"]:
            raise RuntimeError(f"Stage {name!r} never passed all joint gates")
        resume_stage = ""
        resume_local = 0
        resume_optimizer = None

    if not global_best_path.exists():
        if last_payload is None:
            raise RuntimeError("No joint training checkpoint was produced")
        last_payload["global_gate_fallback"] = True
        torch.save(last_payload, global_best_path)
        run_label = str(config.get("run_label", "cms_Joint run"))
        (output_dir / "global_gate_fallback.txt").write_text(
            f"No checkpoint passed every {run_label} hard gate; best_model.pt is "
            f"the final validated checkpoint and is not an accepted {run_label} result.\n",
            encoding="utf-8",
        )
    best = torch.load(global_best_path, map_location="cpu", weights_only=False)
    restore_joint_checkpoint(model, best)
    return history, best
