"""Balanced multi-region trainer for the ``cms_Joint`` run series."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cms_training import make_stage_loss_config
try:
    from .joint_metrics import (
        current_noise_multipliers,
        evaluate_region,
        noise_budget_metrics,
        resolve_noise_multipliers,
        score_joint_metrics,
        set_noise_multipliers,
    )
except ImportError:  # scripts_joint/ added directly to sys.path
    from joint_metrics import (
        current_noise_multipliers,
        evaluate_region,
        noise_budget_metrics,
        resolve_noise_multipliers,
        score_joint_metrics,
        set_noise_multipliers,
    )
try:
    from .joint_anchor import MeanMapAnchor
except ImportError:  # scripts_joint/ added directly to sys.path
    from joint_anchor import MeanMapAnchor
try:
    from .joint_train_utils import (
        _as_float,
        _build_optimizer,
        _build_scheduler,
        _coverage_batches,
        _coverage_indices,
        _expand_indices,
        _first,
        _full_pass_batches,
        _full_pass_step_count,
        _region_weights,
        _sample_batch,
        _scheduled_value,
        _set_trainable,
    )
except ImportError:  # scripts_joint/ added directly to sys.path
    from joint_train_utils import (
        _as_float,
        _build_optimizer,
        _build_scheduler,
        _coverage_batches,
        _coverage_indices,
        _expand_indices,
        _first,
        _full_pass_batches,
        _full_pass_step_count,
        _region_weights,
        _sample_batch,
        _scheduled_value,
        _set_trainable,
    )


def _new_ema_state(model) -> dict[str, torch.Tensor]:
    """Full state_dict shadow for per-step weight averaging."""
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _update_ema_state(model, ema_state: dict[str, torch.Tensor], decay: float) -> None:
    """One per-step EMA update. Float tensors average; integer buffers copy."""
    decay = float(decay)
    if not 0.0 <= decay < 1.0:
        raise ValueError("ema_decay must be in [0, 1)")
    with torch.no_grad():
        for key, value in model.state_dict().items():
            shadow = ema_state.get(key)
            if shadow is None:
                ema_state[key] = value.detach().clone()
            elif value.is_floating_point():
                shadow.mul_(decay).add_(value.detach(), alpha=1.0 - decay)
            else:
                shadow.copy_(value)


def _load_state(model, state: dict[str, torch.Tensor]) -> None:
    """Load weights, tolerating only the optional ``tail_sigma_floors`` buffer.

    ``tail_sigma_floors`` was added on 2026-09-23 and is absent from every
    checkpoint written before then. A config that enables the floor therefore
    has to load the older weights and keep the floor from the config; any other
    missing or unexpected key is still a hard error.
    """
    incompatible = model.load_state_dict(state, strict=False)
    allowed_missing = {
        key
        for key in incompatible.missing_keys
        if key.split(".")[-1] == "tail_sigma_floors"
    }
    failed_missing = [
        key for key in incompatible.missing_keys if key not in allowed_missing
    ]
    if failed_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "state dict mismatch: missing "
            f"{failed_missing} unexpected {list(incompatible.unexpected_keys)}"
        )


_CYCLE_DECODER_NOISE_CHOICES = ("native", "zero")


def _resolve_cycle_decoder_noise(config: dict[str, Any], stage: dict[str, Any]) -> str:
    """Read the A1 switch: does the cycle see the decoder's noise or not?

    ``native`` (default) is the historical behaviour. ``zero`` computes
    ``x_reco`` with the decoder's injected noise switched off, so the
    reconstruction term constrains the deterministic map only and stops
    penalising the stochastic channel (the A1 hypothesis in
    docs/project_tree.md).
    """
    value = stage.get("cycle_decoder_noise", config.get("cycle_decoder_noise", "native"))
    mode = str(value).lower()
    if mode not in _CYCLE_DECODER_NOISE_CHOICES:
        raise ValueError(
            f"cycle_decoder_noise must be one of {_CYCLE_DECODER_NOISE_CHOICES}, "
            f"got {value!r}"
        )
    return mode


def _decode_cycle(model, z_encoded: torch.Tensor, mode: str) -> torch.Tensor:
    """Decode the encoded batch for the cycle term under an explicit noise policy."""
    if mode == "native":
        return _first(model.decode(z_encoded))
    if mode == "zero":
        previous = current_noise_multipliers(model)
        zeroed = dict(previous)
        zeroed["decoder_core"] = 0.0
        zeroed["decoder_tail"] = 0.0
        set_noise_multipliers(model, zeroed)
        try:
            return _first(model.decode(z_encoded))
        finally:
            set_noise_multipliers(model, previous)
    raise ValueError(f"unknown cycle decoder noise mode {mode!r}")


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
    mean_map_anchor: MeanMapAnchor | None = None,
    ema_state: dict[str, torch.Tensor] | None = None,
    ema_decay: float = 0.0,
) -> dict[str, float]:
    """Accumulate one independent loss per region before each shared update."""
    model.train()
    cycle_noise_mode = _resolve_cycle_decoder_noise(config, stage)
    loaders = config.get("loaders", {})
    batch_size = int(loaders.get("train_batch_size", 1024))
    # Mechanism fix for the flat-loss diagnosis: pad every stream whose
    # per-update batch falls below this floor. Zero keeps the historical
    # behaviour exactly (back-compatible default).
    min_events_per_update = int(loaders.get("min_events_per_update", 0) or 0)
    if min_events_per_update < 0:
        raise ValueError("loaders.min_events_per_update must be non-negative")
    epoch_definition = str(loaders.get("epoch_definition", "fixed_steps")).lower()
    if epoch_definition not in {"fixed_steps", "full_pass"}:
        raise ValueError(
            "loaders.epoch_definition must be 'fixed_steps' or 'full_pass'"
        )
    if epoch_definition == "full_pass":
        steps = _full_pass_step_count(
            region_arrays, list(config["region_order"]), batch_size
        )
    else:
        steps = int(loaders.get("steps_per_epoch", 100))
        if steps < 1:
            raise ValueError("loaders.steps_per_epoch must be positive")
    weights = _region_weights(config)
    rng = np.random.default_rng(seed)
    sums: dict[str, float] = {"loss": 0.0, "grad_norm": 0.0}
    # Updates dropped because a gradient was non-finite. Reported per epoch in
    # the returned train dict (and therefore in history.json) so the count is
    # auditable rather than silently swallowed.
    nonfinite_skips = 0
    applied_steps = 0

    sampling = str(loaders.get("sampling", "random")).lower()
    if sampling not in {"random", "cycling_without_replacement"}:
        raise ValueError(
            "loaders.sampling must be 'random' or 'cycling_without_replacement'"
        )
    coverage: dict[tuple[str, str], Any] = {}
    if sampling == "cycling_without_replacement":
        if epoch_index is None or int(epoch_index) < 1:
            raise ValueError(
                "cycling_without_replacement requires a positive epoch_index"
            )
        base_seed = int(config.get("seed", 0))
        for region_index, name in enumerate(config["region_order"]):
            arrays = region_arrays[name]
            for domain_index, key in enumerate(("x_train", "z_train")):
                if epoch_definition == "full_pass":
                    coverage[(name, key)] = _full_pass_batches(
                        arrays[key],
                        steps,
                        int(epoch_index),
                        seed=base_seed,
                        stream=2 * region_index + domain_index,
                    )
                else:
                    coverage[(name, key)] = _coverage_batches(
                        arrays[key],
                        batch_size,
                        steps,
                        int(epoch_index),
                        seed=base_seed,
                        stream=2 * region_index + domain_index,
                    )
    elif epoch_definition == "full_pass":
        raise ValueError(
            "loaders.epoch_definition='full_pass' requires "
            "loaders.sampling='cycling_without_replacement'"
        )

    for step_index in range(steps):
        optimizer.zero_grad(set_to_none=True)
        step_total = 0.0
        for name in config["region_order"]:
            arrays = region_arrays[name]
            if sampling == "cycling_without_replacement":
                x_index = _expand_indices(
                    coverage[(name, "x_train")][step_index],
                    len(arrays["x_train"]),
                    min_events_per_update,
                    rng,
                )
                z_index = _expand_indices(
                    coverage[(name, "z_train")][step_index],
                    len(arrays["z_train"]),
                    min_events_per_update,
                    rng,
                )
                x = torch.as_tensor(
                    np.ascontiguousarray(arrays["x_train"][x_index]),
                    dtype=torch.float32,
                    device=device,
                )
                z = torch.as_tensor(
                    np.ascontiguousarray(arrays["z_train"][z_index]),
                    dtype=torch.float32,
                    device=device,
                )
            else:
                effective_batch = max(batch_size, min_events_per_update)
                x = _sample_batch(arrays["x_train"], effective_batch, rng, device)
                z = _sample_batch(arrays["z_train"], effective_batch, rng, device)
            factory = loss_factories[name]
            factory.reset_components()

            z_encoded = _first(model.encode(x))
            x_reco = _decode_cycle(model, z_encoded, cycle_noise_mode)
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

        # Frozen mean-map anchor. The invariant-mass guard forbids a physical
        # mass anchor; this is the complementary contract: later stochastic
        # stages may not move the deterministic map captured from the declared
        # noiseless reference stage. The core/tail noise stays on and trainable.
        anchor_weight = float(stage.get("mean_map_anchor_weight", 0.0) or 0.0)
        if anchor_weight > 0.0:
            if mean_map_anchor is None:
                raise RuntimeError(
                    f"stage {stage.get('name')!r} sets mean_map_anchor_weight="
                    f"{anchor_weight} but no mean-map anchor was constructed; "
                    "declare mean_map_anchor in the config"
                )
            anchor_loss, anchor_terms = mean_map_anchor.loss(model)
            if not torch.isfinite(anchor_loss):
                raise FloatingPointError("Non-finite mean-map anchor loss")
            (anchor_weight * anchor_loss).backward()
            anchor_scalar = _as_float(anchor_loss)
            sums["mean_map_anchor"] = sums.get("mean_map_anchor", 0.0) + anchor_scalar
            step_total += anchor_weight * anchor_scalar
            for key, value in anchor_terms.items():
                sums[f"mean_map_anchor_{key}"] = sums.get(
                    f"mean_map_anchor_{key}", 0.0
                ) + float(value)

        parameters = [
            p for p in model.parameters() if p.requires_grad and p.grad is not None
        ]
        clip = float(stage.get("gradient_clip_norm", 0.0))
        grad_norm = None
        if not parameters:
            # Historical no-op: ``optimizer.step()`` with nothing to update.
            applied_steps += 1
        else:
            # Two distinct hazards are dropped rather than allowed to end a
            # multi-hour run:
            #   1. a non-finite entry, e.g. from a decoded p4 that cancels to an
            #      exactly zero pair mass (SqrtBackward0 = inf), and
            #   2. every entry finite but so large that the float32 total norm
            #      overflows, which `clip_grad_norm_` also reports as
            #      "non-finite".
            # Scaling either with `error_if_nonfinite=False` would write NaN
            # into the weights, so the update is skipped and counted instead.
            # The count lands in history.json as ``nonfinite_gradient_skips``
            # (and ``optimizer_updates`` records how many updates were actually
            # applied), so a systematic recurrence is visible, not hidden.
            grads_usable = all(
                bool(torch.isfinite(parameter.grad).all()) for parameter in parameters
            )
            if grads_usable:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters,
                    max_norm=clip if clip > 0.0 else math.inf,
                    error_if_nonfinite=False,
                )
                grads_usable = bool(torch.isfinite(grad_norm))
            if grads_usable:
                optimizer.step()
                applied_steps += 1
                if ema_state is not None and ema_decay > 0.0:
                    _update_ema_state(model, ema_state, ema_decay)
            else:
                nonfinite_skips += 1
                optimizer.zero_grad(set_to_none=True)
                grad_norm = None
                print(
                    f"[warn] {stage.get('name', '?')} epoch "
                    f"{epoch_index if epoch_index is not None else '?'} step "
                    f"{step_index + 1}/{steps}: non-finite gradient; optimizer "
                    f"update skipped (skips this epoch: {nonfinite_skips})",
                    flush=True,
                )
        sums["loss"] += step_total
        if grad_norm is not None:
            sums["grad_norm"] += _as_float(grad_norm)
    divisor = max(1, applied_steps)
    result = {
        key: value / divisor if key == "grad_norm" else value / max(1, steps)
        for key, value in sums.items()
    }
    result["optimizer_updates"] = float(applied_steps)
    result["optimizer_steps_attempted"] = float(steps)
    result["nonfinite_gradient_skips"] = float(nonfinite_skips)
    for name in config["region_order"]:
        if epoch_definition == "full_pass":
            result[f"{name}_x_events"] = float(len(region_arrays[name]["x_train"]))
            result[f"{name}_z_events"] = float(len(region_arrays[name]["z_train"]))
        else:
            events = float(batch_size * steps)
            result[f"{name}_x_events"] = events
            result[f"{name}_z_events"] = events
        # Effective per-update cardinalities actually fed to the loss. When
        # min_events_per_update is active the small stream reports the padded
        # size, so the training log states the floor that applies to each term.
        if sampling == "cycling_without_replacement":
            x_per_update = max(
                len(coverage[(name, "x_train")][0]), min_events_per_update
            )
            z_per_update = max(
                len(coverage[(name, "z_train")][0]), min_events_per_update
            )
        else:
            x_per_update = z_per_update = max(batch_size, min_events_per_update)
        result[f"{name}_x_batch_per_update"] = float(x_per_update)
        result[f"{name}_z_batch_per_update"] = float(z_per_update)
    return result


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
    # Validation is deterministic by default. Scoring at the stage's training
    # noise level makes stage scores incomparable and hides deterministic-map
    # degeneration; override with loaders.validation_noise_multipliers only for
    # an explicit noisy diagnostic.
    validation_noise = resolve_noise_multipliers(
        loaders.get("validation_noise_multipliers")
    )
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
            noise_multipliers=validation_noise,
        )

    # A0.3: optional fixed-z noise budget. Diagnostic only; it adds keys to the
    # region metrics but never to the configured selection targets. The
    # evaluate_region calls restore the model multipliers, so the stage policy
    # read here is the live training policy.
    budget_cfg = loaders.get("validation_noise_budget") or {}
    if budget_cfg.get("enabled"):
        stage_noise = current_noise_multipliers(model)
        core = float(budget_cfg.get("core_multiplier", stage_noise["decoder_core"]))
        tail = float(budget_cfg.get("tail_multiplier", stage_noise["decoder_tail"]))
        for offset, name in enumerate(config["region_order"]):
            metrics[name].update(
                noise_budget_metrics(
                    model,
                    region_arrays[name]["z_val"],
                    daughter_masses=config["model"].get("daughter_masses"),
                    device=device,
                    batch_size=int(loaders.get("eval_batch_size", 2048)),
                    draws=int(budget_cfg.get("draws", 8)),
                    max_events=budget_cfg.get("max_events", 1024),
                    seed=seed + 2000 * offset,
                    core=core,
                    tail=tail,
                )
            )
        selection_note = {
            "enabled": True,
            "decoder_core": core,
            "decoder_tail": tail,
            "draws": int(budget_cfg.get("draws", 8)),
            "events_cap": budget_cfg.get("max_events", 1024),
        }
    else:
        selection_note = {"enabled": False}

    selection = score_joint_metrics(metrics, config)
    selection["validation_noise_multipliers"] = validation_noise
    selection["validation_noise_budget"] = selection_note
    return metrics, selection


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
            # "core"/"tail" stay for backward compatibility with checkpoints
            # and readers that predate the encoder/decoder split; they carry
            # the ENCODER values, which is what ``first_step`` has always been.
            "core": float(first_step.core_noise_multiplier),
            "tail": float(first_step.tail_noise_multiplier),
            "encoder_core": float(model.encoder.steps[0].core_noise_multiplier),
            "encoder_tail": float(model.encoder.steps[0].tail_noise_multiplier),
            "decoder_core": float(model.decoder.steps[0].core_noise_multiplier),
            "decoder_tail": float(model.decoder.steps[0].tail_noise_multiplier),
        },
    }


def restore_joint_checkpoint(model, checkpoint: dict[str, Any]) -> None:
    # _load_state tolerates the optional tail_sigma_floors buffer added
    # 2026-09-23, so pre-change checkpoints still initialize a model whose
    # config enables the floor.
    _load_state(model, checkpoint["model_state_dict"])
    noise = checkpoint.get("noise_multipliers") or {}
    shared_core = float(noise.get("core", 1.0))
    shared_tail = float(noise.get("tail", 1.0))
    # Checkpoints written before the encoder/decoder split carry only the
    # shared pair; restoring them must reproduce the old behaviour exactly.
    model.set_component_noise_multipliers(
        encoder_core=float(noise.get("encoder_core", shared_core)),
        encoder_tail=float(noise.get("encoder_tail", shared_tail)),
        decoder_core=float(noise.get("decoder_core", shared_core)),
        decoder_tail=float(noise.get("decoder_tail", shared_tail)),
    )



def _stage_run_prefix(config, output_dir: Path) -> str:
    """Return a human-friendly run prefix for stage-best checkpoint files.

    Example: Run_E -> RunE, Run_A -> RunA, Run_C_fullScale -> RunCfullScale.
    """
    raw = str(config.get("run_name") or output_dir.name)
    return raw.replace("_", "")


def _stage_short_name(stage) -> str:
    """Strip a leading runA_/runB_/... prefix from a stage name."""
    name = str(stage["name"])
    return name.split("_", 1)[1] if name.startswith("run") and "_" in name else name


def _stage_best_path(output_dir: Path, stage, config) -> Path:
    """Return the run-prefixed stage-best path, with legacy fallback.

    New runs write best_RunE_stage1_*.pt instead of the inherited
    best_runA_stage1_*.pt. When an old run directory still contains only the
    legacy filename, that legacy file is used for compatibility.
    """
    new_path = output_dir / (
        f"best_{_stage_run_prefix(config, output_dir)}_{_stage_short_name(stage)}.pt"
    )
    legacy_path = output_dir / f"best_{stage['name']}.pt"
    if new_path.exists():
        return new_path
    if legacy_path.exists():
        return legacy_path
    return new_path


def _stage_last_path(output_dir: Path, stage, config) -> Path:
    """Return the run-prefixed per-stage LAST-checkpoint path.

    ``last_model.pt`` is the last evaluated epoch of the whole run, i.e. the
    last stage's last epoch. It therefore cannot answer "what did stage 2 look
    like when it ended?" after a later stage has overwritten it. This path keeps
    one such file per stage: last_<Run>_<stage>.pt, overwritten at every
    validation so it always holds the latest evaluated epoch of that stage.

    No legacy fallback: the file is new (2026-09-10) and absent for old runs.
    """
    return output_dir / (
        f"last_{_stage_run_prefix(config, output_dir)}_{_stage_short_name(stage)}.pt"
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
    anchor_config = config.get("mean_map_anchor") or {}
    mean_map_anchor: MeanMapAnchor | None = None
    anchor_reference_stage = ""
    if anchor_config.get("enabled", False):
        anchor_reference_stage = str(anchor_config.get("reference_stage", ""))
        mean_map_anchor = MeanMapAnchor(
            events_per_region=int(anchor_config.get("events_per_region", 256))
        )
        mean_map_anchor.build_inputs(region_arrays, config["region_order"], device)
        print(
            f"[mean-map-anchor] armed: reference_stage={anchor_reference_stage!r}, "
            f"events_per_region={mean_map_anchor.events_per_region}"
        )
        # Resume-past-the-reference-stage: the reference weights are already on
        # disk, so capture from that checkpoint instead of waiting for a stage
        # that will not run again.
        reference_stage_dict = next(
            (
                stage
                for stage in config["stages"]
                if str(stage.get("name")) == anchor_reference_stage
            ),
            None,
        )
        if reference_stage_dict is not None:
            reference_path = _stage_best_path(
                output_dir, reference_stage_dict, config
            )
            if reference_path.exists():
                live_state = {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                }
                live_noise = current_noise_multipliers(model)
                restore_joint_checkpoint(
                    model,
                    torch.load(reference_path, map_location="cpu", weights_only=False),
                )
                info = mean_map_anchor.capture(model, anchor_reference_stage)
                model.load_state_dict(live_state, strict=True)
                set_noise_multipliers(model, live_noise)
                print(
                    f"[mean-map-anchor] captured reference from existing "
                    f"checkpoint {reference_path.name}: {info}"
                )
    ema_decay = float((config.get("loaders") or {}).get("ema_decay", 0.0) or 0.0)
    ema_state = _new_ema_state(model) if ema_decay > 0.0 else None
    if ema_state is not None:
        print(f"[ema] per-step weight EMA enabled, decay={ema_decay}")
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
    # State carried from the previous stage's restored best weights (not the
    # last epoch) so a warm-started stage starts from a matched (theta, moments).
    previous_optimizer_state = resume_optimizer
    global_epoch = int((resume or {}).get("global_epoch", 0))
    if resume:
        history = [row for row in history if int(row.get("global_epoch", 0)) <= global_epoch]

    global_best_score = math.inf
    global_best_path = output_dir / "best_model.pt"
    # Only trust a historical score when the checkpoint it refers to is still on
    # disk; otherwise the first improvement of this run must be free to write it.
    if global_best_path.exists():
        for row in history:
            selection = row.get("joint_selection") or {}
            if not selection:
                continue
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
            # This stage finished before the interruption. The non-resume path
            # restores the stage-best weights here, so the resume path must too.
            completed_best = _stage_best_path(output_dir, stage, config)
            if completed_best.exists():
                restore_joint_checkpoint(
                    model,
                    torch.load(completed_best, map_location="cpu", weights_only=False),
                )
                print(f"Resume: restored {name} stage-best weights before continuing")
            resume_stage = ""
            resume_local = 0
            resume_optimizer = None
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
        elif (
            bool(stage.get("carry_optimizer_state", False))
            and previous_optimizer_state
        ):
            try:
                # Carry the AdamW moments only. ``load_state_dict`` would also
                # restore the previous stage's param_groups (lr, weight_decay),
                # silently overriding this stage's schedule, so re-apply the
                # fresh stage hyperparameters afterwards.
                fresh_groups = [dict(group) for group in optimizer.param_groups]
                optimizer.load_state_dict(previous_optimizer_state)
                for group, fresh in zip(optimizer.param_groups, fresh_groups):
                    group.update(
                        {key: value for key, value in fresh.items() if key != "params"}
                    )
                print(
                    f"Stage {name!r}: carried optimizer state from the previous stage"
                )
            except (ValueError, RuntimeError) as error:
                print(
                    f"Stage {name!r}: could not carry optimizer state ({error}); "
                    "starting from a fresh optimizer"
                )
        scheduler = _build_scheduler(optimizer, stage, start_epoch)
        stage_best_score = math.inf
        stage_best_path = _stage_best_path(output_dir, stage, config)
        if stage_best_path.exists() and start_epoch > 1:
            # Resuming inside a stage: recover the score the saved stage-best
            # actually represents, otherwise any later epoch overwrites it.
            for row in history:
                if str(row.get("stage", "")) != name:
                    continue
                selection = row.get("joint_selection") or {}
                if not selection:
                    continue
                stage_best_score = min(
                    stage_best_score, float(selection.get("selection_score", math.inf))
                )
            if math.isfinite(stage_best_score):
                print(
                    f"Resume: {name} stage-best score restored as {stage_best_score:.6g}"
                )

        for local_epoch in range(start_epoch, int(stage["epochs"]) + 1):
            global_epoch += 1
            started = time.time()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            stage_resolved = dict(stage)
            stage_resolved.update(
                make_stage_loss_config(stage, local_epoch, int(stage["epochs"]))
            )
            # Shared defaults, then optional per-component overrides. A stage
            # that names only ``core_noise_multiplier`` / ``tail_noise_``
            # behaves exactly as it did before the split, so every existing
            # config and every completed run is reproducible unchanged.
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
            component_noise = {}
            for component in ("encoder", "decoder"):
                for kind, shared in (("core", core), ("tail", tail)):
                    key = f"{component}_{kind}_noise_multiplier"
                    component_noise[f"{component}_{kind}"] = (
                        _scheduled_value(
                            stage[key], local_epoch, int(stage["epochs"])
                        )
                        if key in stage
                        else shared
                    )
            model.set_component_noise_multipliers(**component_noise)
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
                mean_map_anchor=mean_map_anchor,
                ema_state=ema_state,
                ema_decay=ema_decay,
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
                "encoder_core_noise_multiplier": component_noise["encoder_core"],
                "encoder_tail_noise_multiplier": component_noise["encoder_tail"],
                "decoder_core_noise_multiplier": component_noise["decoder_core"],
                "decoder_tail_noise_multiplier": component_noise["decoder_tail"],
                "num_slices": int(stage_resolved["num_slices"]),
                "seconds": time.time() - started,
                "train": train,
            }
            should_eval = (
                local_epoch == 1
                or local_epoch == int(stage["epochs"])
                or local_epoch % int(stage.get("eval_every", 5)) == 0
            )
            train_loss = float(train["loss"])
            eval_loss = None
            if should_eval:
                metrics, selection = validate_joint(
                    model,
                    region_arrays,
                    loss_factories,
                    config,
                    device=device,
                    seed=int(config.get("seed", 0)) + global_epoch,
                )
                checkpoint_state = "raw"
                if ema_state is not None:
                    # Score the live weights and their EMA; keep the better one.
                    # The live weights are restored either way so training
                    # continues from the un-averaged iterate.
                    raw_state = {
                        key: value.detach().clone()
                        for key, value in model.state_dict().items()
                    }
                    _load_state(model, ema_state)
                    ema_metrics, ema_selection = validate_joint(
                        model,
                        region_arrays,
                        loss_factories,
                        config,
                        device=device,
                        seed=int(config.get("seed", 0)) + global_epoch,
                    )
                    _load_state(model, raw_state)
                    ema_score = float(ema_selection["selection_score"])
                    raw_score = float(selection["selection_score"])
                    row["ema_selection_score"] = ema_score
                    if ema_score < raw_score:
                        metrics, selection = ema_metrics, ema_selection
                        checkpoint_state = "ema"
                row["region_validation"] = metrics
                row["joint_selection"] = selection
                validation_noise = selection.get("validation_noise_multipliers") or {}
                row["validation_noise_multipliers"] = validation_noise
                weights = _region_weights(config)
                eval_loss = float(sum(
                    weights[name] * float(metrics[name].get("base_loss", 0.0))
                    for name in config["region_order"]
                ))
                row["eval_loss"] = eval_loss
                row["checkpoint_state"] = checkpoint_state
                if checkpoint_state == "ema":
                    live_state = {
                        key: value.detach().clone()
                        for key, value in model.state_dict().items()
                    }
                    _load_state(model, ema_state)
                    payload = _checkpoint(
                        model, optimizer, config, stage, global_epoch, local_epoch,
                        metrics, selection,
                    )
                    _load_state(model, live_state)
                else:
                    payload = _checkpoint(
                        model, optimizer, config, stage, global_epoch, local_epoch,
                        metrics, selection,
                    )
                last_payload = payload
                score = float(selection["selection_score"])
                if not stage_best_path.exists() or score < stage_best_score:
                    stage_best_score = score
                    torch.save(payload, stage_best_path)
                if not global_best_path.exists() or score < global_best_score:
                    global_best_score = score
                    torch.save(payload, global_best_path)
                # Global last (last stage's last eval) and per-stage last. The
                # per-stage file answers "what did this stage look like when it
                # ended?" without a later stage overwriting it.
                torch.save(payload, output_dir / "last_model.pt")
                torch.save(payload, _stage_last_path(output_dir, stage, config))
                print(
                    f"[{name} {local_epoch}/{stage['epochs']}] "
                    f"train_loss={train_loss:.6g} eval_loss={eval_loss:.6g} "
                    f"score={score:.4g} worst={selection['worst_region']} "
                    f"val_noise=enc{validation_noise.get('encoder_core', 0.0):.3g}"
                    f"/{validation_noise.get('encoder_tail', 0.0):.3g}"
                    f",dec{validation_noise.get('decoder_core', 0.0):.3g}"
                    f"/{validation_noise.get('decoder_tail', 0.0):.3g}"
                    + (
                        f" peak_cuda={torch.cuda.max_memory_reserved(device) / 1024**3:.2f}GiB"
                        if device.type == "cuda"
                        else ""
                    )
                )
            else:
                print(
                    f"[{name} {local_epoch}/{stage['epochs']}] "
                    f"train_loss={train_loss:.6g}"
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
        # Carry the optimizer state that belongs to the restored stage-best
        # weights, so `carry_optimizer_state: true` warm-starts the next stage.
        previous_optimizer_state = stage_best.get("optimizer_state_dict")
        if ema_state is not None:
            ema_state = _new_ema_state(model)
        if (
            mean_map_anchor is not None
            and name == anchor_reference_stage
            and not mean_map_anchor.ready
        ):
            info = mean_map_anchor.capture(model, name)
            print(
                f"[mean-map-anchor] captured reference from stage {name!r}: {info}"
            )
        resume_stage = ""
        resume_local = 0
        resume_optimizer = None

    if not global_best_path.exists():
        if last_payload is None:
            raise RuntimeError("No joint training checkpoint was produced")
        torch.save(last_payload, global_best_path)
    best = torch.load(global_best_path, map_location="cpu", weights_only=False)
    restore_joint_checkpoint(model, best)
    return history, best
