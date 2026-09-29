"""Debug harness: reproduce the non-finite gradient crash of Run_H_A2frozen.

Reads (never writes) the artefacts of ``outputs/cms_Joint/Run_H_A2frozen`` and
re-runs stage 2 from ``last_model.pt`` with instrumentation:

* per-parameter gradient finiteness check at the clip call, and
* a report of the parameters carrying the largest gradient magnitude.

Usage:
    python debug_nan/repro_a2frozen.py --epochs 6 [--anomaly-from 2]
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO = Path(__file__).resolve().parents[1]
for _directory in (REPO, REPO / "scripts", REPO / "scripts_sota", REPO / "scripts_joint"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from cms_data import load_config  # noqa: E402
from cms_training import make_stage_loss_config  # noqa: E402
from device_utils import select_device  # noqa: E402
from joint_data import load_joint_regions, resolve_joint_config  # noqa: E402
from joint_model import build_joint_autoencoder  # noqa: E402
from joint_train_utils import (  # noqa: E402
    _build_optimizer,
    _build_scheduler,
    _scheduled_value,
    _set_trainable,
)
from joint_trainer import restore_joint_checkpoint, train_joint_epoch  # noqa: E402
from run_joint import build_loss_factories  # noqa: E402

RUN_DIR = REPO / "outputs" / "cms_Joint" / "Run_H_A2frozen"
CONFIG_PATH = REPO / "configs_joint" / "cms_Joint_runH_A2frozen.yaml"
STAGE_NAME = "runH_stage2_stochastic_core"

STATE: dict = {"model": None, "call": 0, "report_every_epoch": True, "last_epoch": None}


def _grad_report(model) -> list[tuple[float, str, int]]:
    rows = []
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            grad = parameter.grad
            if grad is None:
                continue
            rows.append((float(grad.detach().abs().max()), name, grad.numel()))
    rows.sort(reverse=True)
    return rows


def _install_clip_probe() -> None:
    """Wrap clip_grad_norm_ with a per-parameter finiteness probe."""
    original = torch.nn.utils.clip_grad_norm_

    def probe(parameters, max_norm, error_if_nonfinite=False, foreach=None):
        parameters = [p for p in parameters if p is not None and p.grad is not None]
        STATE["call"] += 1
        bad = []
        with torch.no_grad():
            for name, parameter in STATE["model"].named_parameters():
                grad = parameter.grad
                if grad is None:
                    continue
                finite = torch.isfinite(grad)
                if bool(finite.all()):
                    continue
                finite_values = grad[finite]
                bad.append(
                    {
                        "name": name,
                        "nan": int(torch.isnan(grad).sum()),
                        "inf": int(torch.isinf(grad).sum()),
                        "numel": grad.numel(),
                        "max_finite_abs": float(finite_values.abs().max())
                        if finite_values.numel()
                        else float("nan"),
                        "param_max_abs": float(parameter.detach().abs().max()),
                    }
                )
        if bad:
            print(f"\n!!! non-finite gradient at clip call {STATE['call']}")
            for row in bad:
                print("   ", row)
            top = _grad_report(STATE["model"])[:12]
            print("    largest |grad| entries:")
            for value, name, _numel in top:
                print(f"      {value: .6e}  {name}")
            raise SystemExit(3)
        if STATE["report_every_epoch"]:
            STATE["report_every_epoch"] = False
            top = _grad_report(STATE["model"])[:12]
            print(f"    [probe] top |grad| at first clip of epoch (call {STATE['call']}):")
            for value, name, _numel in top:
                print(f"      {value: .6e}  {name}")
        return original(
            parameters,
            max_norm=max_norm,
            error_if_nonfinite=error_if_nonfinite,
            foreach=foreach,
        )

    torch.nn.utils.clip_grad_norm_ = probe


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--anomaly-from", type=int, default=0,
                        help="enable autograd anomaly detection from this local epoch")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default=str(RUN_DIR / "last_model.pt"))
    args = parser.parse_args()

    config = resolve_joint_config(load_config(CONFIG_PATH))
    device = select_device(args.device)
    print(f"device={device} torch={torch.__version__}")

    region_arrays, cache_info, region_configs, pair_indices = load_joint_regions(
        config, num_samples=None, use_cache=True, log=lambda *a: None
    )
    for name in config["region_order"]:
        shapes = {key: list(value.shape) for key, value in region_arrays[name].items()}
        print(f"[data] {name} {shapes}")

    model = build_joint_autoencoder(
        config["model"],
        region_arrays,
        float(config.get("muon_mass_gev", 0.1056583755)),
        config["model"].get("daughter_masses"),
    ).to(device)
    loss_factories = build_loss_factories(config, region_arrays)
    STATE["model"] = model

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    restore_joint_checkpoint(model, checkpoint)
    start_local = int(checkpoint["stage_epoch"]) + 1
    start_global = int(checkpoint["global_epoch"])
    print(
        f"[resume] {checkpoint['stage']['name']} local={checkpoint['stage_epoch']} "
        f"global={start_global} -> start local={start_local}"
    )

    stage = next(s for s in config["stages"] if s["name"] == STAGE_NAME)
    _set_trainable(model, stage)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = _build_optimizer(parameters, stage)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler = _build_scheduler(optimizer, stage, start_local)
    print(f"[stage] {STAGE_NAME} lr={optimizer.param_groups[0]['lr']} optim={stage['optimizer']}")

    _install_clip_probe()
    global_epoch = start_global
    for local_epoch in range(start_local, start_local + args.epochs):
        global_epoch += 1
        if local_epoch > int(stage["epochs"]):
            print("stage finished")
            break
        if args.anomaly_from and local_epoch >= args.anomaly_from:
            torch.autograd.set_detect_anomaly(True)
        stage_resolved = dict(stage)
        stage_resolved.update(make_stage_loss_config(stage, local_epoch, int(stage["epochs"])))
        core = _scheduled_value(stage.get("core_noise_multiplier", 1.0), local_epoch, int(stage["epochs"]))
        tail = _scheduled_value(stage.get("tail_noise_multiplier", 0.0), local_epoch, int(stage["epochs"]))
        component_noise = {}
        for component in ("encoder", "decoder"):
            for kind, shared in (("core", core), ("tail", tail)):
                key = f"{component}_{kind}_noise_multiplier"
                component_noise[f"{component}_{kind}"] = (
                    _scheduled_value(stage[key], local_epoch, int(stage["epochs"]))
                    if key in stage
                    else shared
                )
        model.set_component_noise_multipliers(**component_noise)
        for factory in loss_factories.values():
            factory.set_num_slices(int(stage_resolved["num_slices"]))
        STATE["report_every_epoch"] = True
        started = time.time()
        try:
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
                mean_map_anchor=None,
                ema_state=None,
                ema_decay=0.0,
            )
        except SystemExit:
            print(f"[crash] local epoch {local_epoch} (global {global_epoch})")
            raise
        if scheduler is not None:
            scheduler.step()
        print(
            f"[epoch] local={local_epoch} global={global_epoch} "
            f"loss={train['loss']:.6g} grad_norm={train['grad_norm']:.6g} "
            f"sec={time.time() - started:.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
