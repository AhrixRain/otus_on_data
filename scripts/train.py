from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
import time
from copy import deepcopy
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
from cms_model import build_model, checkpoint_payload
from cms_training import HistoryLogger, build_loaders, build_loss_factory, train_all_stages
from device_utils import device_report, select_device
from encoder_diagnostics import make_training_callback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a configured CMS dilepton OTUS workflow.")
    parser.add_argument("--config", type=Path, required=True, help="YAML/JSON config path.")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu.")
    parser.add_argument("--run-name", default=None, help="Run id under the configured output root.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override config output root.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for each enabled stage.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override train batch size.")
    parser.add_argument("--num-samples", type=int, default=None, help="Limit selected CMS and MG5 rows.")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed.")
    parser.add_argument("--dry-run", action="store_true", help="Load data, build model/loaders, then exit.")
    parser.add_argument(
        "--progress",
        choices=("auto", "bar", "log", "none"),
        default="auto",
        help="Progress reporting mode. auto uses a tqdm bar only for an interactive terminal.",
    )
    parser.add_argument(
        "--progress-log-steps",
        type=int,
        default=1,
        help="In log progress mode, print every N training steps.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one tiny first-stage epoch with reduced slices.",
    )
    return parser.parse_args()


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    config = deepcopy(config)
    if args.seed is not None:
        config["seed"] = int(args.seed)
    if args.output_dir is not None:
        config.setdefault("paths", {})["output_root"] = str(args.output_dir)
    if args.epochs is not None:
        for stage in config["stages"]:
            if stage.get("enabled", True):
                stage["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        config.setdefault("loaders", {})["train_batch_size"] = int(args.batch_size)
    if args.smoke_test:
        first_enabled_seen = False
        for stage in config["stages"]:
            if not stage.get("enabled", True):
                continue
            if first_enabled_seen:
                stage["enabled"] = False
                continue
            first_enabled_seen = True
            stage["epochs"] = 1
            stage["num_slices"] = min(int(stage.get("num_slices", 1000)), 10)
            stage["log_freq"] = 1
        config.setdefault("loaders", {})["train_batch_size"] = min(
            int(config["loaders"].get("train_batch_size", 512)),
            512,
        )
        config.setdefault("loaders", {})["eval_batch_size"] = min(
            int(config["loaders"].get("eval_batch_size", 512)),
            512,
        )
    return config


def make_run_dir(config: dict, run_name: str | None, dry_run: bool) -> Path:
    output_root = Path(config["paths"]["output_root"])
    run_id = run_name or time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    if not dry_run and run_dir.exists() and any(
        (run_dir / name).exists() for name in ("best_model.pt", "last_model.pt")
    ):
        raise FileExistsError(
            f"Refusing to overwrite existing checkpoint directory: {run_dir}. "
            "Use a new --run-name or --output-dir."
        )
    return run_dir


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def format_loss(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4e}"


class ProgressReporter:
    def __init__(
        self,
        run_dir: Path,
        run_name: str,
        device: torch.device,
        progress_mode: str,
        log_every_steps: int,
    ):
        self.run_dir = run_dir
        self.run_name = run_name
        self.device = str(device)
        self.requested_mode = progress_mode
        self.mode = self._resolve_mode(progress_mode)
        self.log_every_steps = max(1, int(log_every_steps))
        self.status_path = run_dir / "status.json"
        self.latest_train_loss: float | None = None
        self.latest_eval_loss: float | None = None
        self.best_eval_loss: float | None = None
        self.status_write_interval = 5.0
        self._last_status_write = time.time()
        self._latest_status: dict[str, Any] | None = None
        self._bar = None
        self._bar_step = 0

    @staticmethod
    def _resolve_mode(progress_mode: str) -> str:
        if progress_mode == "auto":
            return "bar" if sys.stdout.isatty() else "log"
        return progress_mode

    def __enter__(self) -> "ProgressReporter":
        if self.mode == "bar":
            try:
                from tqdm import tqdm
            except ImportError as exc:
                if self.requested_mode != "bar":
                    self.mode = "log"
                    return self
                raise RuntimeError(
                    "--progress bar requires tqdm. Install it with `python -m pip install tqdm`."
                ) from exc
            self._bar_factory = tqdm
        return self

    def start(self, total_steps: int) -> None:
        if self.mode == "bar" and self._bar is None:
            self._bar = self._bar_factory(
                total=total_steps,
                unit="step",
                dynamic_ncols=True,
                file=sys.stdout,
            )

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
        if self._latest_status is not None:
            self.write_status(self._latest_status)

    def update(self, progress: dict[str, Any]) -> None:
        total_steps = int(progress["total_steps"])
        if total_steps and self._bar is None:
            self.start(total_steps)

        self.latest_train_loss = clean_float(progress["train_loss"])
        if progress["eval_loss"] is not None:
            self.latest_eval_loss = clean_float(progress["eval_loss"])
        self.best_eval_loss = clean_float(progress["best_eval_loss"])

        status = {
            "run_name": self.run_name,
            "event": progress.get("event", "train_step"),
            "stage": progress["stage"],
            "epoch": int(progress["epoch"]),
            "epochs_in_stage": int(progress["epochs_in_stage"]),
            "global_epoch": int(progress["global_epoch"]),
            "total_epochs": int(progress["total_epochs"]),
            "step": int(progress["step"]),
            "steps_in_epoch": int(progress["steps_in_epoch"]),
            "global_step": int(progress["global_step"]),
            "total_steps": total_steps,
            "percent": round(float(progress["percent"]), 1),
            "latest_train_loss": self.latest_train_loss,
            "latest_eval_loss": self.latest_eval_loss,
            "best_eval_loss": self.best_eval_loss,
            "device": self.device,
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "progress_mode": self.mode,
            "evaluated_this_epoch": bool(progress["evaluated"]),
        }
        self._latest_status = status
        now = time.time()
        force_write = bool(progress.get("evaluated")) or progress["event"] == "epoch_end"
        if force_write or (now - self._last_status_write) >= self.status_write_interval:
            self.write_status(status)
            self._last_status_write = now
        self.report(status)

    def write_status(self, status: dict[str, Any]) -> None:
        tmp_path = self.status_path.with_name(f"{self.status_path.name}.tmp")
        tmp_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp_path.replace(self.status_path)

    def report(self, status: dict[str, Any]) -> None:
        if self.mode == "none":
            return
        if self.mode == "bar":
            if self._bar is None:
                self.start(int(status["total_steps"]))
            delta = int(status["global_step"]) - self._bar_step
            if delta > 0:
                self._bar.update(delta)
                self._bar_step = int(status["global_step"])
            self._bar.set_description(str(status["stage"]))
            self._bar.set_postfix(
                epoch=f"{status['epoch']}/{status['epochs_in_stage']}",
                step=f"{status['step']}/{status['steps_in_epoch']}",
                train_loss=format_loss(status["latest_train_loss"]),
                eval_loss=format_loss(status["latest_eval_loss"]),
                best_eval=format_loss(status["best_eval_loss"]),
            )
            return

        if status["event"] == "train_step":
            should_log_step = (
                status["global_step"] % self.log_every_steps == 0
                or status["step"] == status["steps_in_epoch"]
            )
            if not should_log_step:
                return
        elif not status["evaluated_this_epoch"]:
            return

        epoch_width = len(str(status["epochs_in_stage"]))
        step_width = len(str(status["steps_in_epoch"]))
        global_width = len(str(status["total_epochs"]))
        global_step_width = len(str(status["total_steps"]))
        print(
            "[progress] "
            f"event={status['event']} "
            f"stage={status['stage']} "
            f"epoch={status['epoch']:0{epoch_width}d}/{status['epochs_in_stage']} "
            f"global={status['global_epoch']:0{global_width}d}/{status['total_epochs']} "
            f"step={status['step']:0{step_width}d}/{status['steps_in_epoch']} "
            f"global_step={status['global_step']:0{global_step_width}d}/{status['total_steps']} "
            f"pct={status['percent']:.1f} "
            f"train_loss={format_loss(status['latest_train_loss'])} "
            f"eval_loss={format_loss(status['latest_eval_loss'])} "
            f"best_eval={format_loss(status['best_eval_loss'])}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    raw_config = load_config(args.config)
    config = resolve_config(apply_cli_overrides(raw_config, args))
    config["device"] = args.device
    config["run_name"] = args.run_name
    config["num_samples"] = args.num_samples
    config["smoke_test"] = bool(args.smoke_test)

    seed = int(config.get("seed", 0))
    set_seed(seed)
    device = select_device(args.device)
    report = device_report(device)
    run_dir = make_run_dir(config, args.run_name, args.dry_run)
    config["run_name"] = run_dir.name

    print("Using device:", device)
    print("Device report:", json.dumps(report, sort_keys=True))
    print("Output directory:", run_dir)
    print("PYTORCH_ENABLE_MPS_FALLBACK:", os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"))

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

    model = build_model(config, x_train_mean, x_train_std, z_train_mean, z_train_std)
    model.to(device)
    loss_factory = build_loss_factory(arrays["x_train"], arrays["z_train"], config.get("loss", {}))
    train_loaders, eval_loaders, loader_info = build_loaders(
        config,
        arrays,
        batch_size_override=args.batch_size,
        device=device,
    )
    config["loader_info"] = loader_info
    print("Loader info:", json.dumps(loader_info, sort_keys=True))

    eval_batch_size = int(loader_info.get("eval_batch_size", 8192))
    diag_x_eval = arrays["x_test"][: min(len(arrays["x_test"]), eval_batch_size)]
    diag_z_eval = arrays["z_test"][: min(len(arrays["z_test"]), eval_batch_size)]
    diagnostics_callback = make_training_callback(
        run_dir,
        diag_x_eval,
        diag_z_eval,
        device,
        seed=int(config.get("seed", 0)),
        batch_size=eval_batch_size,
    )
    diagnostic_freq = int(config.get("evaluation", {}).get("diagnostic_freq", 10))

    if args.dry_run:
        print("Dry run complete. No files written.")
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, run_dir / "config.resolved.json")
    logger = HistoryLogger(run_dir / "train_log.csv")
    progress_reporter = ProgressReporter(
        run_dir,
        run_dir.name,
        device,
        args.progress,
        args.progress_log_steps,
    )

    def save_checkpoint(epoch: int, best_eval_loss: float | None, is_best: bool) -> None:
        payload = checkpoint_payload(model, config, stats, epoch, best_eval_loss, report)
        torch.save(payload, run_dir / "last_model.pt")
        if is_best:
            shutil.copy2(run_dir / "last_model.pt", run_dir / "best_model.pt")

    def save_stage_checkpoint(stage_name: str, epoch: int, eval_loss: float | None) -> None:
        """Keep a named checkpoint at every stage boundary (diagnostic support).

        Training math/schedule are unchanged; this only preserves the model at
        the end of each stage (e.g. epoch 100 after Stage 2) so later stages can
        be compared against it (see scripts/stage_diagnostic.py).
        """
        checkpoint_path = run_dir / f"checkpoint_{stage_name}.pt"
        payload = checkpoint_payload(model, config, stats, epoch, eval_loss, report)
        torch.save(payload, checkpoint_path)
        print(
            f"Saved stage-boundary checkpoint: {checkpoint_path} "
            f"(global epoch {epoch}, eval loss {eval_loss})",
            flush=True,
        )

    with progress_reporter:
        progress_reporter.start(
            sum(int(stage["epochs"]) for stage in config["stages"] if stage.get("enabled", True))
            * min(len(train_loaders[0]), len(train_loaders[1]))
        )
        history, best_eval_loss, final_epoch = train_all_stages(
            model,
            config,
            train_loaders,
            eval_loaders,
            loss_factory,
            device,
            logger,
            save_checkpoint,
            progress_reporter.update,
            stage_checkpoint_callback=save_stage_checkpoint,
            diagnostics_callback=diagnostics_callback,
            diagnostic_every=diagnostic_freq,
        )
    if final_epoch == 0:
        save_checkpoint(0, None, is_best=True)

    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
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
            row = {key: history.get(key, [None] * n_history_rows)[idx] for key in loss_fields}
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
    with (diag_dir / "gradient_trajectory.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=grad_fields)
        writer.writeheader()
        for idx in range(n_history_rows):
            row = {key: history.get(key, [None] * n_history_rows)[idx] for key in grad_fields}
            writer.writerow(row)
    if not (run_dir / "best_model.pt").exists():
        shutil.copy2(run_dir / "last_model.pt", run_dir / "best_model.pt")
    print("Training complete.")
    print("Best eval loss:", best_eval_loss)
    print("Saved outputs in:", run_dir)


if __name__ == "__main__":
    main()
