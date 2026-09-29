"""Stress probe: amplify the model noise until a backward pass turns non-finite.

Runs real ``train_joint_epoch`` updates on a subsample of the training data with
the stage-2 loss factory and the epoch-50 checkpoint, with autograd anomaly
detection ON, so the offending forward op is named instead of a bare
``clip_grad_norm_`` failure.
"""
from __future__ import annotations

import argparse
import os
import sys
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
from joint_data import load_joint_regions, resolve_joint_config  # noqa: E402
from joint_model import build_joint_autoencoder  # noqa: E402
from joint_train_utils import _build_optimizer, _scheduled_value, _set_trainable  # noqa: E402
from joint_trainer import restore_joint_checkpoint, train_joint_epoch  # noqa: E402
from run_joint import build_loss_factories  # noqa: E402

RUN_DIR = REPO / "outputs" / "cms_Joint" / "Run_H_A2frozen"
CONFIG_PATH = REPO / "configs_joint" / "cms_Joint_runH_A2frozen.yaml"
STAGE_NAME = "runH_stage2_stochastic_core"

FAILURE: dict = {}


def install_clip_probe(model) -> None:
    original = torch.nn.utils.clip_grad_norm_

    def probe(parameters, max_norm, error_if_nonfinite=False, foreach=None):
        bad = []
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    continue
                if not bool(torch.isfinite(parameter.grad).all()):
                    g = parameter.grad
                    bad.append(
                        (name, int(torch.isnan(g).sum()), int(torch.isinf(g).sum()),
                         float(g[torch.isfinite(g)].abs().max())
                         if bool(torch.isfinite(g).any()) else float("nan"))
                    )
        if bad:
            FAILURE["bad"] = bad
            print("!!! non-finite gradient:", bad)
            raise RuntimeError("non-finite gradient")
        return original(parameters, max_norm=max_norm,
                        error_if_nonfinite=error_if_nonfinite, foreach=foreach)

    torch.nn.utils.clip_grad_norm_ = probe


def subsample(region_arrays, limit: int) -> dict:
    small = {}
    for name, arrays in region_arrays.items():
        small[name] = {}
        for key, values in arrays.items():
            small[name][key] = values[:limit] if values.shape[0] > limit else values
    return small


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factors", default="1,3,10,30,100,300")
    parser.add_argument("--events", type=int, default=120000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-anomaly", action="store_true",
                        help="disable autograd anomaly detection (production-like)")
    args = parser.parse_args()

    config = resolve_joint_config(load_config(CONFIG_PATH))
    device = torch.device(args.device)
    region_arrays, _, _, _ = load_joint_regions(
        config, num_samples=None, use_cache=True, log=lambda *a: None
    )
    small = subsample(region_arrays, args.events)
    loss_factories = build_loss_factories(config, region_arrays)
    stage = next(s for s in config["stages"] if s["name"] == STAGE_NAME)
    checkpoint = torch.load(RUN_DIR / "last_model.pt", map_location="cpu", weights_only=False)

    for factor in [float(v) for v in args.factors.split(",")]:
        model = build_joint_autoencoder(
            config["model"], region_arrays,
            float(config.get("muon_mass_gev", 0.1056583755)),
            config["model"].get("daughter_masses"),
        ).to(device)
        restore_joint_checkpoint(model, checkpoint)
        original_setter = model.set_component_noise_multipliers

        def scaled_setter(*, _f=factor, _o=original_setter, **kwargs):
            _o(**{key: float(value) * _f for key, value in kwargs.items()})

        model.set_component_noise_multipliers = scaled_setter
        _set_trainable(model, stage)
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = _build_optimizer(parameters, stage)
        install_clip_probe(model)
        if not args.no_anomaly:
            torch.autograd.set_detect_anomaly(True)
        local_epoch = 31
        stage_resolved = dict(stage)
        stage_resolved.update(make_stage_loss_config(stage, local_epoch, int(stage["epochs"])))
        for factory in loss_factories.values():
            factory.set_num_slices(int(stage_resolved["num_slices"]))
        print(f"\n=== noise factor {factor} ===", flush=True)
        try:
            train = train_joint_epoch(
                model, optimizer, small, loss_factories, config, stage_resolved,
                device=device, seed=int(config.get("seed", 0)) + 10000 * 54,
                epoch_index=54,
            )
            print(f"factor {factor}: completed, loss={train['loss']:.6g} "
                  f"grad_norm={train['grad_norm']:.6g} "
                  f"skips={train.get('nonfinite_gradient_skips')}", flush=True)
        except RuntimeError as error:
            print(f"factor {factor}: FAILED with {type(error).__name__}: {error}", flush=True)
            if "grad" in str(error):
                return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
