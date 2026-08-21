"""Small Run E training loop.

The loop reuses the current project's data loaders and stage-coefficient
resolution, but adds the Tier A training protocol:

  * per-stage ``core_noise_multiplier`` / ``tail_noise_multiplier``,
  * optional gradient clipping,
  * optional per-stage cosine LR schedule,
  * hard mass gates in checkpoint selection,
  * restoration of each stage's best checkpoint before the next stage.
"""

from __future__ import annotations

import json
import math
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from scripts_sota.selection import GateConfig, checkpoint_selection_score, evaluate_mass_gates
    from scripts.cms_training import build_loaders, make_stage_loss_config
except ImportError:  # package directories added directly to sys.path
    from selection import GateConfig, checkpoint_selection_score, evaluate_mass_gates
    from cms_training import build_loaders, make_stage_loss_config


def _first_tensor(value):
    if isinstance(value, (tuple, list)):
        return value[0]
    return value


def _float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def _set_trainable(model, encoder: bool, decoder: bool) -> None:
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(encoder)
    for parameter in model.decoder.parameters():
        parameter.requires_grad_(decoder)


def _train_epoch(
    model,
    optimizer,
    x_loader,
    z_loader,
    stage,
    loss_factory,
    device,
):
    model.train()
    totals = {
        "loss": 0.0,
        "x_loss": 0.0,
        "z_loss": 0.0,
        "alt_x_loss": 0.0,
        "flow_loss": 0.0,
        "grad_norm": 0.0,
    }
    n_batches = 0
    clip = float(stage.get("gradient_clip_norm", 0.0))
    for x, z in zip(x_loader, z_loader):
        x = x.to(device)
        z = z.to(device)
        if hasattr(loss_factory, "reset_components"):
            loss_factory.reset_components()
        optimizer.zero_grad(set_to_none=True)

        flow_loss = x.new_tensor(0.0)
        if hasattr(model, "flow_matching_loss"):
            flow_loss = model.flow_matching_loss(x, z)
            if not torch.isfinite(flow_loss):
                raise FloatingPointError("Non-finite flow-matching loss")

        z_encoded = _first_tensor(model.encode(x))
        x_reco = _first_tensor(model.decode(z_encoded))
        x_loss = loss_factory.x_reco_loss(x, x_reco)
        z_loss = loss_factory.z_prior_loss(z, z_encoded)

        encoder_anchor = (
            loss_factory.encoder_anchor_loss(z_encoded, x)
            if stage["nu_e"] > 0
            else x.new_tensor(0.0)
        )
        decoder_anchor = x.new_tensor(0.0)
        alt_x_loss = x.new_tensor(0.0)
        needs_direct_decoder = (
            bool(getattr(loss_factory, "vanilla_swae", False))
            or stage["tau"] > 0
            or stage["rho"] > 0
            or stage["nu_d"] > 0
        )
        if needs_direct_decoder:
            x_from_z = _first_tensor(model.decode(z))
            alt_x_loss = loss_factory.x_sim_loss(x, x_from_z)
            decoder_anchor = (
                loss_factory.decoder_anchor_loss(z, x_from_z[: len(z)])
                if stage["nu_d"] > 0
                else x.new_tensor(0.0)
            )

        total = (
            stage["beta"] * x_loss
            + stage["lamb"] * z_loss
            + stage["tau"] * alt_x_loss
            + stage["nu_e"] * encoder_anchor
            + stage["nu_d"] * decoder_anchor
            + float(stage.get("flow_weight", 0.0)) * flow_loss
        )
        if not torch.isfinite(total):
            raise FloatingPointError("Non-finite Run E training loss")
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad and p.grad is not None],
            max_norm=clip,
        ) if clip > 0 else torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad and p.grad is not None],
            max_norm=math.inf,
        )
        optimizer.step()

        totals["loss"] += _float(total)
        totals["x_loss"] += _float(x_loss)
        totals["z_loss"] += _float(z_loss)
        totals["alt_x_loss"] += _float(alt_x_loss)
        totals["flow_loss"] += _float(flow_loss)
        totals["grad_norm"] += _float(grad_norm)
        for key, value in getattr(loss_factory, "latest_components", {}).items():
            if isinstance(value, torch.Tensor):
                totals.setdefault(key, 0.0)
                totals[key] += _float(value)
        for key, value in getattr(model, "latest_bridge_components", {}).items():
            totals.setdefault(key, 0.0)
            totals[key] += float(value)
        n_batches += 1
    return {key: value / max(1, n_batches) for key, value in totals.items()}


@torch.inference_mode()
def _evaluate_losses(model, x_loader, z_loader, loss_factory, device):
    model.eval()
    totals = {"x_loss": 0.0, "z_loss": 0.0, "alt_x_loss": 0.0, "score": 0.0}
    n = 0
    for x, z in zip(x_loader, z_loader):
        x = x.to(device)
        z = z.to(device)
        z_encoded = _first_tensor(model.encode(x))
        x_reco = _first_tensor(model.decode(z_encoded))
        x_loss = loss_factory.x_reco_loss(x, x_reco)
        z_loss = loss_factory.z_prior_loss(z, z_encoded)
        alt_x_loss = loss_factory.x_sim_loss(x, _first_tensor(model.decode(z)))
        score = loss_factory.validation_score(
            {"x_loss": x_loss, "z_loss": z_loss, "alt_x_loss": alt_x_loss, "cycle_loss": x_loss}
        )
        totals["x_loss"] += _float(x_loss)
        totals["z_loss"] += _float(z_loss)
        totals["alt_x_loss"] += _float(alt_x_loss)
        totals["score"] += _float(score)
        n += 1
    return {key: value / max(1, n) for key, value in totals.items()}


def _checkpoint_payload(model, optimizer, config, stats, epoch, stage, losses, gates, loss_factory=None):
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "stats": {key: value.tolist() for key, value in stats.items()},
        "epoch": int(epoch),
        "stage": stage,
        "validation_losses": losses,
        "validation_gates": gates,
    }
    if loss_factory is not None and hasattr(loss_factory, "state_dict"):
        payload["sota_loss_state"] = loss_factory.state_dict()
    return payload


def run_sota_training(
    model,
    config,
    arrays,
    loss_factory,
    device,
    output_dir: Path,
    callbacks: dict[str, Any] | None = None,
    resume: dict[str, Any] | None = None,
):
    """Run all enabled stages with mass-gated checkpoint selection.

    ``resume`` is the raw checkpoint dictionary from ``last_model.pt`` (or
    any trainer checkpoint).  Training skips completed epochs before the
    checkpoint's stage/epoch, restores model weights and optimizer state, and
    appends to the existing ``history.json``.  Loader/RNG streams are rebuilt,
    so continuation is statistically equivalent but not bitwise identical.
    """
    loader_cfg = config.get("loaders", {})
    eval_batch_size = int(loader_cfg.get("eval_batch_size", 8192))
    train_loaders, eval_loaders, loader_info = build_loaders(
        config, arrays, None, device
    )
    x_mean = np.mean(arrays["x_train"], axis=0).astype(np.float32)
    x_std = np.std(arrays["x_train"], axis=0).astype(np.float32)
    z_mean = np.mean(arrays["z_train"], axis=0).astype(np.float32)
    z_std = np.std(arrays["z_train"], axis=0).astype(np.float32)
    stats = {
        "x_train_mean": x_mean,
        "x_train_std": x_std,
        "z_train_mean": z_mean,
        "z_train_std": z_std,
    }

    history_path = output_dir / "history.json"
    history = []
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
        except Exception:
            history = []
    resume_stage_name = None
    resume_epoch = None
    resume_optimizer_state = None
    if resume:
        resume_stage_name = str((resume.get("stage") or {}).get("name"))
        resume_epoch = int(resume.get("epoch", 0))
        resume_optimizer_state = resume.get("optimizer_state_dict")
        # history may contain rows trained after the last saved checkpoint
        # (rows 16/17 in the interrupted full run).  Those rows do not have
        # matching weights, so truncate history to the checkpoint epoch.
        history = [
            row
            for row in history
            if int(row.get("global_epoch", 0)) <= resume_epoch
        ]
    global_epoch = resume_epoch if resume_epoch is not None else max(
        [int(row["global_epoch"]) for row in history] or [0]
    )

    global_best_score = math.inf
    for row in history:
        value = row.get("validation_gate_score")
        if isinstance(value, (int, float)) and math.isfinite(value):
            global_best_score = min(global_best_score, float(value))
    completed_epochs_before = 0
    for stage in config["stages"]:
        if not stage.get("enabled", True):
            continue
        stage_name = str(stage["name"])
        stage_epochs = int(stage["epochs"])
        start_local_epoch = 1
        restore_optimizer = False
        if resume_stage_name is not None and stage_name == resume_stage_name:
            resume_local = int(resume_epoch) - completed_epochs_before
            if resume_local >= stage_epochs:
                completed_epochs_before += stage_epochs
                continue  # this stage was already completed
            start_local_epoch = max(1, resume_local + 1)
            restore_optimizer = resume_optimizer_state is not None
        elif resume_stage_name is not None:
            # A stage after the checkpoint stage starts fresh at epoch 1.
            pass
        completed_epochs_before += stage_epochs
        _set_trainable(
            model,
            encoder=not bool(stage.get("freeze_encoder", False)),
            decoder=not bool(stage.get("freeze_decoder", False)),
        )
        if hasattr(model, "set_noise_multipliers"):
            model.set_noise_multipliers(
                float(stage.get("core_noise_multiplier", 1.0)),
                float(stage.get("tail_noise_multiplier", 0.0)),
            )
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=float(stage["lr"]))
        if restore_optimizer and resume_optimizer_state:
            try:
                optimizer.load_state_dict(resume_optimizer_state)
            except Exception:
                # A shape/config mismatch should not prevent resuming from the
                # saved model weights with a fresh optimizer.
                pass
        scheduler = None
        if stage.get("lr_schedule", "none") == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, int(stage["epochs"]))
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for _ in range(start_local_epoch - 1):
                    scheduler.step()
        elif stage.get("lr_decay", False):
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda epoch: 1 / (1 + 0.1 * epoch)
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for _ in range(start_local_epoch - 1):
                    scheduler.step()

        stage_rows = [row for row in history if row.get("stage") == stage_name]
        stage_finite_scores = [
            float(row["validation_gate_score"])
            for row in stage_rows
            if isinstance(row.get("validation_gate_score"), (int, float))
            and math.isfinite(row["validation_gate_score"])
        ]
        stage_best_score = min(stage_finite_scores) if stage_finite_scores else math.inf
        stage_best_path = output_dir / f"best_{stage_name}.pt"
        for local_epoch in range(start_local_epoch, stage_epochs + 1):
            global_epoch += 1
            started = time.time()
            stage_cfg = make_stage_loss_config(stage, local_epoch, int(stage["epochs"]))
            train = _train_epoch(
                model, optimizer, train_loaders[0], train_loaders[1],
                stage_cfg, loss_factory, device,
            )
            if scheduler is not None:
                scheduler.step()

            should_eval = (
                local_epoch == 1
                or local_epoch == int(stage["epochs"])
                or local_epoch % int(stage.get("eval_every", 2)) == 0
            )
            row = {
                "global_epoch": global_epoch,
                "stage": stage_name,
                "stage_epoch": local_epoch,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "seconds": time.time() - started,
                **{f"train_{k}": v for k, v in train.items()},
            }
            if should_eval:
                losses = _evaluate_losses(model, eval_loaders[0], eval_loaders[1], loss_factory, device)
                gates, passed = evaluate_mass_gates(
                    model,
                    arrays["x_val"],
                    arrays["z_val"],
                    daughter_masses=config.get("model", {}).get("daughter_masses"),
                    gate_config=GateConfig.from_config(config),
                    device=device,
                    batch_size=eval_batch_size,
                    decoder_draws=int(loader_cfg.get("validation_draws", 1)),
                )
                score = checkpoint_selection_score(
                    losses["score"], gates, passed, config
                )
                row.update({f"validation_{k}": v for k, v in losses.items()})
                row["validation_gate_score"] = score
                row["validation_gates"] = gates
                row["validation_gate_passed"] = passed

                is_best = score < global_best_score
                if score < stage_best_score:
                    stage_best_score = score
                    torch.save(
                        _checkpoint_payload(model, optimizer, config, stats, global_epoch, stage, losses, gates, loss_factory),
                        stage_best_path,
                    )
                if is_best:
                    global_best_score = score
                    torch.save(
                        _checkpoint_payload(model, optimizer, config, stats, global_epoch, stage, losses, gates, loss_factory),
                        output_dir / "best_model.pt",
                    )
                torch.save(
                    _checkpoint_payload(model, optimizer, config, stats, global_epoch, stage, losses, gates, loss_factory),
                    output_dir / "last_model.pt",
                )
                if callbacks and "eval" in callbacks:
                    callbacks["eval"](row)

            history.append(row)
            (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        if not stage_best_path.exists():
            # Every mass gate can fail during early exploration.  Keep the
            # latest checkpoint as the stage fallback so training can continue,
            # but record the failure explicitly.
            fallback = _checkpoint_payload(model, optimizer, config, stats, global_epoch, stage, losses, gates, loss_factory)
            torch.save(fallback, stage_best_path)
            fallback_note = (
                f"{stage_name} ended at epoch {global_epoch} without passing the mass gates\n"
            )
            with (output_dir / "stage_gate_fallback.txt").open("a", encoding="utf-8") as handle:
                handle.write(fallback_note)
            if bool(config.get("require_stage_gate_pass", False)):
                raise RuntimeError(
                    f"Stopping: {stage_name} never passed the mass gates and "
                    "require_stage_gate_pass=true."
                )
        best = torch.load(stage_best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(best["model_state_dict"])

    best_path = output_dir / "best_model.pt"
    if not best_path.exists():
        torch.save(
            _checkpoint_payload(model, optimizer, config, stats, global_epoch, stage, losses, gates, loss_factory),
            best_path,
        )
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    return history, best
