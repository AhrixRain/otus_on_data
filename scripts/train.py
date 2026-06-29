from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from cms_data import array_stats, load_and_split, load_config, resolve_config, save_resolved_config
from cms_model import build_model, checkpoint_payload
from cms_training import HistoryLogger, ZLossFactory, build_loaders, train_all_stages
from device_utils import device_report, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CMS DoubleElectron OTUS on a portable device.")
    parser.add_argument("--config", type=Path, required=True, help="YAML/JSON config path.")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu.")
    parser.add_argument("--run-name", default=None, help="Run id under outputs/cms_doubleelectron.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override config output root.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for each enabled stage.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override train batch size.")
    parser.add_argument("--num-samples", type=int, default=None, help="Limit selected CMS and MG5 rows.")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed.")
    parser.add_argument("--dry-run", action="store_true", help="Load data, build model/loaders, then exit.")
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

    print("Using device:", device)
    print("Device report:", json.dumps(report, sort_keys=True))
    print("Output directory:", run_dir)
    print("PYTORCH_ENABLE_MPS_FALLBACK:", os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"))

    arrays = load_and_split(config, num_samples=args.num_samples)
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
    loss_factory = ZLossFactory(arrays["x_train"], arrays["z_train"], config.get("loss", {}))
    train_loaders, eval_loaders, loader_info = build_loaders(
        config,
        arrays,
        batch_size_override=args.batch_size,
        device=device,
    )
    config["loader_info"] = loader_info
    print("Loader info:", json.dumps(loader_info, sort_keys=True))

    if args.dry_run:
        print("Dry run complete. No files written.")
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, run_dir / "config.resolved.json")
    logger = HistoryLogger(run_dir / "train_log.csv")

    def save_checkpoint(epoch: int, best_eval_loss: float | None, is_best: bool) -> None:
        payload = checkpoint_payload(model, config, stats, epoch, best_eval_loss, report)
        torch.save(payload, run_dir / "last_model.pt")
        if is_best:
            shutil.copy2(run_dir / "last_model.pt", run_dir / "best_model.pt")

    history, best_eval_loss, final_epoch = train_all_stages(
        model,
        config,
        train_loaders,
        eval_loaders,
        loss_factory,
        device,
        logger,
        save_checkpoint,
    )
    if final_epoch == 0:
        save_checkpoint(0, None, is_best=True)

    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if not (run_dir / "best_model.pt").exists():
        shutil.copy2(run_dir / "last_model.pt", run_dir / "best_model.pt")
    print("Training complete.")
    print("Best eval loss:", best_eval_loss)
    print("Saved outputs in:", run_dir)


if __name__ == "__main__":
    main()
