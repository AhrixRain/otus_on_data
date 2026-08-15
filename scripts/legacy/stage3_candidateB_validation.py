#!/usr/bin/env python
"""Resume the existing Stage-3 decoder refinement from a Stage-2 checkpoint.

This is a thin validation wrapper for the OTUS-on-CMS J/psi Candidate B run:

    .venv/bin/python scripts/stage3_candidateB_validation.py \
        --config configs/archive/cms_JpsiDoubleMuons_encoder_candidateB.yaml \
        --checkpoint outputs/cms_JpsiDoubleMuons/archive/encoder_diag_candidateB/checkpoint_stage2_joint_transport.pt \
        --output-dir outputs/cms_JpsiDoubleMuons/archive/stage3_candidateB_validation/training \
        --num-samples 50000

It does not change the Stage-3 schedule or the loss. It only starts the model
from the saved Stage-2 weights, keeps the encoder frozen exactly as the
existing Stage-3 config does, and adds diagnostic bookkeeping:

  * checkpoints at the requested Stage-3 evaluation points (+20/+50/+100),
  * the same encoder/cycle metric trajectory as the Stage-1/2 diagnostic,
  * decoder gradient norms and the reconstruction-vs-z->x cosine so Stage-3
    optimization conflicts are visible.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cms_data import (
    array_stats,
    load_and_split_cached,
    load_config,
    resolve_config,
    save_resolved_config,
)
from cms_model import checkpoint_payload, load_model_from_checkpoint
from cms_training import (
    HistoryLogger,
    build_loaders,
    build_loss_factory,
    first_tensor,
    flat_grad_vector,
    grad_cosine,
    grad_norm,
    train_all_stages,
)
from device_utils import device_report, select_device
from encoder_diagnostics import make_training_callback, write_metric_row


STAGE3_NAME = "stage3_decoder_response_mass_protected"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume Stage 3 from a saved Stage-2 checkpoint (Candidate B validation)."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Stage-2 checkpoint (.pt).")
    parser.add_argument("--output-dir", type=Path, required=True, help="Training/checkpoint output dir.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=100, help="Stage-3 epoch count (default 100).")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed.")
    parser.add_argument("--start-epoch", type=int, default=100, help="Global epoch of the Stage-2 checkpoint.")
    parser.add_argument(
        "--checkpoint-epochs",
        default="20,50,100",
        help="Comma-separated Stage-3 local epochs at which to save named checkpoints.",
    )
    parser.add_argument(
        "--progress-log-steps",
        type=int,
        default=1,
        help="Print every N training steps (log progress mode).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one tiny Stage-3 epoch with reduced slices (no scientific value).",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def clean_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def write_status(status_path: Path, status: dict[str, Any]) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = status_path.with_name(f"{status_path.name}.tmp")
    tmp.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(status_path)


def combine_grad_lists(grads_a, grads_b):
    if grads_a is None and grads_b is None:
        return None
    if grads_a is None:
        return grads_b
    if grads_b is None:
        return grads_a
    combined = []
    for a, b in zip(grads_a, grads_b):
        if a is not None and b is not None:
            combined.append(a + b)
        elif a is not None:
            combined.append(a)
        elif b is not None:
            combined.append(b)
        else:
            combined.append(None)
    return combined


def decoder_gradient_metrics(
    model: torch.nn.Module,
    loss_factory: Any,
    stage_config: dict[str, Any],
    x_sample: np.ndarray,
    z_sample: np.ndarray,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Measure the Stage-3 decoder gradient families on the fixed diagnostic sample.

    The losses are computed exactly as in ``train_standard_epoch`` (same order,
    same stage weights, same noise-sample handling). The encoder is frozen by
    the existing Stage-3 config, so encoder-family columns are zero by
    construction; the decoder families expose any reco-vs-z->x conflict.
    """
    params = [p for p in model.decoder.parameters() if p.requires_grad]
    if not params:
        return {}
    model.train()
    set_seed(seed)
    x = torch.as_tensor(np.ascontiguousarray(x_sample, dtype=np.float32), dtype=torch.float32, device=device)
    z = torch.as_tensor(np.ascontiguousarray(z_sample, dtype=np.float32), dtype=torch.float32, device=device)

    z_tilde = first_tensor(model.encode(x))
    z_loss = (
        loss_factory.z_prior_loss(z, z_tilde)
        if float(stage_config.get("lamb", 0.0)) > 0.0
        else x.new_tensor(0.0)
    )
    x_tilde = first_tensor(model.decode(z_tilde))
    x_loss = (
        loss_factory.x_reco_loss(x, x_tilde)
        if float(stage_config.get("beta", 0.0)) > 0.0
        else x.new_tensor(0.0)
    )
    decoder_samples = max(1, int(getattr(loss_factory, "decoder_num_noise_samples", 1)))
    tau = float(stage_config.get("tau", 0.0))
    if decoder_samples > 1 and tau > 0.0:
        model_x = torch.cat([first_tensor(model.decode(z)) for _ in range(decoder_samples)], dim=0)
        x_for_distribution = x.repeat((decoder_samples, 1))
    else:
        model_x = first_tensor(model.decode(z))
        x_for_distribution = x
    alt_x_loss = (
        loss_factory.x_sim_loss(x_for_distribution, model_x)
        if tau > 0.0
        else x.new_tensor(0.0)
    )

    grads: dict[str, Any] = {}
    if float(stage_config.get("lamb", 0.0)) > 0.0 and z_loss.requires_grad:
        grads["latent"] = list(
            torch.autograd.grad(
                float(stage_config["lamb"]) * z_loss,
                params,
                retain_graph=True,
                allow_unused=True,
            )
        )
    else:
        grads["latent"] = None
    if float(stage_config.get("beta", 0.0)) > 0.0 and x_loss.requires_grad:
        grads["reco"] = list(
            torch.autograd.grad(
                float(stage_config["beta"]) * x_loss,
                params,
                retain_graph=True,
                allow_unused=True,
            )
        )
    else:
        grads["reco"] = None
    if tau > 0.0 and alt_x_loss.requires_grad:
        grads["alt_x"] = list(
            torch.autograd.grad(
                tau * alt_x_loss,
                params,
                retain_graph=True,
                allow_unused=True,
            )
        )
    else:
        grads["alt_x"] = None
    grads["combined"] = combine_grad_lists(
        combine_grad_lists(grads["latent"], grads["reco"]),
        grads["alt_x"],
    )

    norm_latent = grad_norm(grads["latent"])
    norm_reco = grad_norm(grads["reco"])
    norm_alt = grad_norm(grads["alt_x"])
    out: dict[str, Any] = {
        "grad_norm_decoder_latent": norm_latent,
        "grad_norm_decoder_reco": norm_reco,
        "grad_norm_decoder_alt_x": norm_alt,
        "grad_norm_decoder_total": grad_norm(grads["combined"]),
        "grad_cosine_decoder_latent_reco": grad_cosine(grads["latent"], grads["reco"]),
        "grad_cosine_decoder_reco_altx": grad_cosine(grads["reco"], grads["alt_x"]),
        "grad_ratio_decoder_latent_reco": (
            norm_latent / norm_reco if norm_reco and norm_reco > 0.0 else None
        ),
        "grad_ratio_decoder_altx_reco": (
            norm_alt / norm_reco if norm_reco and norm_reco > 0.0 else None
        ),
    }
    return out


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    import copy

    config = copy.deepcopy(config)
    if args.seed is not None:
        config["seed"] = int(args.seed)
    stage3 = None
    for stage in config["stages"]:
        if stage["name"] == STAGE3_NAME:
            stage3 = stage
        stage["enabled"] = stage["name"] == STAGE3_NAME
    if stage3 is None:
        raise ValueError(f"Config has no stage named {STAGE3_NAME!r}.")
    stage3["epochs"] = int(args.epochs)
    if args.smoke_test:
        stage3["epochs"] = 1
        stage3["num_slices"] = min(int(stage3.get("num_slices", 1500)), 10)
        stage3["log_freq"] = 1
        config.setdefault("loaders", {})["train_batch_size"] = min(
            int(config["loaders"].get("train_batch_size", 8192)),
            512,
        )
        config.setdefault("loaders", {})["eval_batch_size"] = min(
            int(config["loaders"].get("eval_batch_size", 60000)),
            512,
        )
    return config


def main() -> None:
    args = parse_args()
    config = resolve_config(apply_cli_overrides(load_config(args.config), args))
    config["device"] = args.device
    config["run_name"] = args.output_dir.name
    config["num_samples"] = args.num_samples
    config["smoke_test"] = bool(args.smoke_test)

    seed = int(config.get("seed", 0))
    set_seed(seed)
    device = select_device(args.device)
    report = device_report(device)
    run_dir = args.output_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "last_model.pt").exists() and not args.smoke_test:
        raise FileExistsError(f"Refusing to overwrite existing checkpoint dir: {run_dir}")

    print("Using device:", device)
    print("Output directory:", run_dir)
    print("Resuming from:", args.checkpoint.expanduser().resolve())

    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model, model_config, stats, loaded_ckpt = load_model_from_checkpoint(
        checkpoint_path,
        config=config,
        map_location=device,
    )
    model.to(device)
    loaded_epoch = int(loaded_ckpt.get("epoch", args.start_epoch))
    start_epoch = max(int(args.start_epoch), loaded_epoch)
    print(f"Stage-2 checkpoint epoch: {loaded_epoch}; continuing from global epoch {start_epoch}")

    arrays, cache_info = load_and_split_cached(
        config,
        num_samples=args.num_samples,
        cache_dir=None,
        use_cache=True,
    )
    print("Data cache:", json.dumps(cache_info, sort_keys=True))
    for key, value in arrays.items():
        print(f"{key}: shape={value.shape}, dtype={value.dtype}")

    x_train_mean, x_train_std = array_stats(arrays["x_train"])
    z_train_mean, z_train_std = array_stats(arrays["z_train"])
    stats = {
        "x_train_mean": x_train_mean,
        "x_train_std": x_train_std,
        "z_train_mean": z_train_mean,
        "z_train_std": z_train_std,
    }
    loss_factory = build_loss_factory(
        arrays["x_train"],
        arrays["z_train"],
        config.get("loss", {}),
        daughter_masses=(config.get("model") or {}).get("daughter_masses"),
    )
    train_loaders, eval_loaders, loader_info = build_loaders(
        config,
        arrays,
        batch_size_override=None,
        device=device,
    )
    config["loader_info"] = loader_info
    print("Loader info:", json.dumps(loader_info, sort_keys=True))

    eval_batch_size = int(loader_info.get("eval_batch_size", 8192))
    diag_x_eval = arrays["x_test"][: min(len(arrays["x_test"]), eval_batch_size)]
    diag_z_eval = arrays["z_test"][: min(len(arrays["z_test"]), eval_batch_size)]
    base_callback = make_training_callback(
        run_dir,
        diag_x_eval,
        diag_z_eval,
        device,
        seed=seed,
        batch_size=eval_batch_size,
    )
    decoder_grad_path = run_dir / "encoder_alignment_diagnostic" / "decoder_gradient_trajectory.csv"
    diagnostic_freq = int(config.get("evaluation", {}).get("diagnostic_freq", 10))

    def diagnostics_callback(model_: torch.nn.Module, global_epoch: int, stage: str, loss_factory_: Any) -> None:
        effective_epoch = global_epoch + start_epoch
        base_callback(model_, effective_epoch, stage, loss_factory_)
        stage3 = next(s for s in config["stages"] if s["name"] == STAGE3_NAME)
        grad_row = decoder_gradient_metrics(
            model_,
            loss_factory_,
            stage3,
            diag_x_eval,
            diag_z_eval,
            device,
            seed=seed + 10 + global_epoch,
        )
        row = {"global_epoch": effective_epoch, "stage": stage}
        row.update(grad_row)
        write_metric_row(decoder_grad_path, row)
        print(
            f"[stage3_grads] epoch {effective_epoch:04d} | "
            f"decoder_reco={grad_row.get('grad_norm_decoder_reco'):.4g} "
            f"decoder_alt_x={grad_row.get('grad_norm_decoder_alt_x'):.4g} "
            f"cos(reco,alt_x)={grad_row.get('grad_cosine_decoder_reco_altx')}",
            flush=True,
        )

    save_resolved_config(config, run_dir / "config.resolved.json")
    logger = HistoryLogger(run_dir / "train_log.csv")
    status_path = run_dir / "status.json"
    checkpoint_epochs = {int(value) for value in args.checkpoint_epochs.split(",") if value.strip()}

    def save_checkpoint(
        epoch: int,
        best_eval_loss: float | None,
        is_best: bool,
        eval_losses: dict[str, Any] | None = None,
    ) -> None:
        payload = checkpoint_payload(model, config, stats, epoch, best_eval_loss, report)
        torch.save(payload, run_dir / "last_model.pt")
        if is_best:
            shutil.copy2(run_dir / "last_model.pt", run_dir / "best_model.pt")

    def save_stage_checkpoint(stage_name: str, epoch: int, eval_loss: float | None) -> None:
        checkpoint_path = run_dir / f"checkpoint_{stage_name}.pt"
        payload = checkpoint_payload(model, config, stats, epoch, eval_loss, report)
        torch.save(payload, checkpoint_path)
        print(
            f"Saved stage-boundary checkpoint: {checkpoint_path} (global epoch {epoch})",
            flush=True,
        )

    def save_eval_point(local_epoch: int, global_epoch: int, eval_loss: float | None) -> None:
        if local_epoch not in checkpoint_epochs:
            return
        checkpoint_path = run_dir / f"checkpoint_stage3_epoch{local_epoch}.pt"
        payload = checkpoint_payload(model, config, stats, global_epoch, eval_loss, report)
        torch.save(payload, checkpoint_path)
        print(
            f"Saved Stage-3 evaluation checkpoint: {checkpoint_path} "
            f"(global epoch {global_epoch})",
            flush=True,
        )

    last_status_write = time.time()
    latest_status: dict[str, Any] = {}

    def progress_callback(progress: dict[str, Any]) -> None:
        nonlocal last_status_write, latest_status
        status = {
            "run_name": run_dir.name,
            "event": progress.get("event", "train_step"),
            "stage": progress["stage"],
            "epoch": int(progress["epoch"]),
            "epochs_in_stage": int(progress["epochs_in_stage"]),
            "global_epoch": int(progress["global_epoch"]) + start_epoch,
            "total_epochs": int(progress["total_epochs"]) + start_epoch,
            "step": int(progress["step"]),
            "steps_in_epoch": int(progress["steps_in_epoch"]),
            "global_step": int(progress["global_step"]),
            "total_steps": int(progress["total_steps"]),
            "percent": round(float(progress["percent"]), 1),
            "latest_train_loss": clean_float(progress["train_loss"]),
            "latest_eval_loss": clean_float(progress["eval_loss"]),
            "best_eval_loss": clean_float(progress["best_eval_loss"]),
            "device": str(device),
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        if progress["event"] == "epoch_end":
            save_eval_point(
                int(progress["epoch"]),
                int(progress["global_epoch"]) + start_epoch,
                clean_float(progress["eval_loss"]),
            )
        latest_status = status
        now = time.time()
        force_write = bool(progress.get("evaluated")) or progress["event"] == "epoch_end"
        if force_write or (now - last_status_write) >= 5.0:
            write_status(status_path, status)
            last_status_write = now
        if progress["event"] != "train_step" or (
            progress["global_step"] % max(1, args.progress_log_steps) == 0
        ):
            print(
                "[progress] "
                f"event={status['event']} "
                f"stage={status['stage']} "
                f"epoch={status['epoch']}/{status['epochs_in_stage']} "
                f"global={status['global_epoch']}/{status['total_epochs']} "
                f"train_loss={status['latest_train_loss']:.4e} "
                f"eval_loss={status['latest_eval_loss']} "
                f"best_eval={status['best_eval_loss']}",
                flush=True,
            )

    def wrap_save_callback(
        epoch: int,
        eval_loss: float | None,
        is_best: bool,
        eval_losses: dict[str, Any] | None = None,
    ) -> None:
        save_checkpoint(epoch + start_epoch, eval_loss, is_best, eval_losses)

    def wrap_stage_checkpoint(stage_name: str, epoch: int, eval_loss: float | None) -> None:
        save_stage_checkpoint(stage_name, epoch + start_epoch, eval_loss)

    history, best_eval_loss, final_epoch = train_all_stages(
        model,
        config,
        train_loaders,
        eval_loaders,
        loss_factory,
        device,
        logger,
        wrap_save_callback,
        progress_callback=progress_callback,
        stage_checkpoint_callback=wrap_stage_checkpoint,
        diagnostics_callback=diagnostics_callback,
        diagnostic_every=diagnostic_freq,
    )
    if final_epoch == 0:
        save_checkpoint(start_epoch, None, is_best=True)

    write_status(
        status_path,
        {
            **latest_status,
            "event": "training_complete",
            "best_eval_loss": best_eval_loss,
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        },
    )

    history_json = {
        key: (
            [start_epoch + value for value in values]
            if key == "epoch"
            else values
        )
        for key, values in history.items()
    }
    (run_dir / "history.json").write_text(json.dumps(history_json, indent=2), encoding="utf-8")

    diag_dir = run_dir / "encoder_alignment_diagnostic"
    diag_dir.mkdir(parents=True, exist_ok=True)
    loss_fields = [
        "epoch",
        "stage",
        "train_loss",
        "train_x_loss",
        "train_z_loss",
        "train_alt_x_loss",
        "train_x_constraint_loss",
    ]
    for key in sorted(history.keys()):
        if key.startswith("train_") and key not in loss_fields:
            loss_fields.append(key)
    n_history_rows = len(history.get("epoch", []))
    with (diag_dir / "loss_trajectory.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=loss_fields)
        writer.writeheader()
        for idx in range(n_history_rows):
            row = {
                key: (
                    history[key][idx] + start_epoch
                    if key == "epoch"
                    else history[key][idx]
                )
                for key in loss_fields
            }
            writer.writerow(row)

    grad_fields = [
        "epoch",
        "stage",
        "grad_norm_encoder_latent",
        "grad_norm_encoder_reco",
        "grad_norm_encoder_anchor",
        "grad_norm_encoder_latent_weighted",
        "grad_norm_encoder_reco_weighted",
        "grad_norm_encoder_anchor_weighted",
        "grad_norm_encoder_total",
        "grad_cosine_latent_reco",
    ]
    grad_rows: list[dict[str, Any]] = []
    for idx in range(n_history_rows):
        row = {key: history.get(key, [None] * n_history_rows)[idx] for key in grad_fields}
        row["epoch"] = row["epoch"] + start_epoch
        grad_rows.append(row)
    decoder_rows = list(csv.DictReader(decoder_grad_path.open(newline="", encoding="utf-8")))
    decoder_by_epoch = {int(row["global_epoch"]): row for row in decoder_rows if row.get("global_epoch")}
    extra_fields: list[str] = []
    for row in decoder_rows:
        for key in row:
            if key not in grad_fields and key not in extra_fields:
                extra_fields.append(key)
    for row in grad_rows:
        decoder_row = decoder_by_epoch.get(int(row["epoch"]))
        if decoder_row:
            for key in extra_fields:
                row[key] = decoder_row.get(key, "")
    all_fields = grad_fields + extra_fields
    with (diag_dir / "gradient_trajectory.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(grad_rows)

    # Copy the same fixed-sample trajectories to the validation package root.
    package_root = run_dir.parent
    for name in ("metric_trajectory.csv", "loss_trajectory.csv", "gradient_trajectory.csv"):
        src = diag_dir / name
        if src.exists():
            shutil.copy2(src, package_root / name)

    if not (run_dir / "best_model.pt").exists():
        shutil.copy2(run_dir / "last_model.pt", run_dir / "best_model.pt")
    print("Training complete.")
    print("Best eval loss:", best_eval_loss)
    print("Saved outputs in:", run_dir)


if __name__ == "__main__":
    main()
