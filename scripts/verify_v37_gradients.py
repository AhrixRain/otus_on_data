#!/usr/bin/env python
"""Gradient-path audit for the v3.7 vanilla OTUS/SWAE objective.

Builds the configured model and loss factory on a small data sample and
verifies, for one mini-batch, that:

  reconstruction loss  -> encoder grad nonzero, decoder grad nonzero
  latent SW loss       -> encoder grad nonzero, decoder grad zero/not involved
  total vanilla loss   -> encoder grad nonzero, decoder grad nonzero

The audit never modifies training behavior and writes only its JSON report
(no checkpoints).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cms_data import array_stats, load_and_split_cached, load_config, resolve_config
from cms_model import build_model
from cms_training import build_loss_factory, first_tensor
from device_utils import select_device


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        torch.mps.manual_seed(seed)


def grads_for(
    loss: torch.Tensor,
    params: list[torch.nn.Parameter],
) -> list[torch.Tensor | None]:
    return list(
        torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    )


def norm_of(grads: list[torch.Tensor | None]) -> float:
    parts = [
        grad.detach().reshape(-1)
        for grad in grads
        if grad is not None and grad.numel() > 0
    ]
    if not parts:
        return 0.0
    return float(torch.cat(parts).norm())


def named_norms(grads: list[torch.Tensor | None], names: list[str]) -> dict[str, float | None]:
    return {
        name: (None if grad is None else float(grad.detach().norm()))
        for name, grad in zip(names, grads)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu.")
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--slices", type=int, default=None, help="Override SW slices.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("v37_gradient_audit.json"),
        help="JSON report path (default: ./v37_gradient_audit.json).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = resolve_config(load_config(args.config))
    loss_config = config.get("loss", {})
    if not bool(loss_config.get("vanilla_v3_7", False)):
        print("ERROR: --config does not enable vanilla_v3_7.", file=sys.stderr)
        return 2

    seed = int(config.get("seed", 0))
    set_seed(seed)
    device = select_device(args.device)
    num_samples = None if args.num_samples <= 0 else args.num_samples
    arrays, cache_info = load_and_split_cached(
        config,
        num_samples=num_samples,
        cache_dir=None,
        use_cache=True,
    )
    print("Data cache:", json.dumps(cache_info, sort_keys=True))

    x_train_mean, x_train_std = array_stats(arrays["x_train"])
    z_train_mean, z_train_std = array_stats(arrays["z_train"])
    model = build_model(
        config,
        x_train_mean,
        x_train_std,
        z_train_mean,
        z_train_std,
    )
    model.to(device)
    model.train()
    loss_factory = build_loss_factory(
        arrays["x_train"],
        arrays["z_train"],
        loss_config,
    )
    stage = next(
        (stage for stage in config["stages"] if stage.get("enabled", True)),
        None,
    )
    if stage is None:
        print("ERROR: no enabled stage in config.", file=sys.stderr)
        return 2
    num_slices = int(args.slices or stage.get("num_slices", 1000))
    loss_factory.set_num_slices(num_slices)
    lamb = float(stage.get("lamb", 1.0))

    batch_size = min(int(args.batch_size), len(arrays["x_test"]), len(arrays["z_test"]))
    x_batch = torch.as_tensor(
        arrays["x_test"][:batch_size],
        dtype=torch.float32,
        device=device,
    )
    z_batch = torch.as_tensor(
        arrays["z_test"][:batch_size],
        dtype=torch.float32,
        device=device,
    )

    set_seed(seed + 1000)
    z_hat = first_tensor(model.encode(x_batch))
    x_hat = first_tensor(model.decode(z_hat))
    reco_loss = loss_factory.x_reco_loss(x_batch, x_hat)
    set_seed(seed + 1001)
    sw_loss = loss_factory.z_prior_loss(z_batch, z_hat)
    total_loss = reco_loss + lamb * sw_loss

    encoder_params = [param for param in model.encoder.parameters() if param.requires_grad]
    decoder_params = [param for param in model.decoder.parameters() if param.requires_grad]
    all_params = [*encoder_params, *decoder_params]
    encoder_names = [f"encoder.{i}:{name}" for i, name in enumerate(encoder_params)]
    decoder_names = [f"decoder.{i}:{name}" for i, name in enumerate(decoder_params)]

    reco_grads = grads_for(reco_loss, all_params)
    sw_grads = grads_for(sw_loss, all_params)
    total_grads = grads_for(total_loss, all_params)

    reco_encoder = norm_of(reco_grads[: len(encoder_params)])
    reco_decoder = norm_of(reco_grads[len(encoder_params):])
    sw_encoder = norm_of(sw_grads[: len(encoder_params)])
    sw_decoder = norm_of(sw_grads[len(encoder_params):])
    total_encoder = norm_of(total_grads[: len(encoder_params)])
    total_decoder = norm_of(total_grads[len(encoder_params):])

    losses_finite = all(
        torch.isfinite(value).item()
        for value in (reco_loss, sw_loss, total_loss)
    )
    checks = {
        "reco_encoder_nonzero": reco_encoder > 0.0,
        "reco_decoder_nonzero": reco_decoder > 0.0,
        "sw_encoder_nonzero": sw_encoder > 0.0,
        "sw_decoder_zero": sw_decoder == 0.0,
        "total_encoder_nonzero": total_encoder > 0.0,
        "total_decoder_nonzero": total_decoder > 0.0,
        "losses_finite": bool(losses_finite),
    }

    def representative_norms() -> dict[str, float | None]:
        candidates: list[tuple[str, torch.nn.Parameter]] = []
        for prefix, module in (("encoder", model.encoder), ("decoder", model.decoder)):
            for name, param in module.named_parameters():
                if "weight" in name and "output_nn" in name:
                    candidates.append((f"{prefix}.{name}", param))
                    break
        names = [name for name, _ in candidates]
        params = [param for _, param in candidates]
        grads = grads_for(total_loss, params)
        return named_norms(grads, names)

    report = {
        "config": str(args.config.resolve()),
        "device": str(device),
        "seed": seed,
        "lambda": lamb,
        "num_slices": num_slices,
        "batch_size": batch_size,
        "n_train_x": int(len(arrays["x_train"])),
        "n_train_z": int(len(arrays["z_train"])),
        "losses": {
            "reconstruction": float(reco_loss.detach().cpu()),
            "sw": float(sw_loss.detach().cpu()),
            "total": float(total_loss.detach().cpu()),
        },
        "grad_norms": {
            "reco_encoder": reco_encoder,
            "reco_decoder": reco_decoder,
            "sw_encoder": sw_encoder,
            "sw_decoder": sw_decoder,
            "total_encoder": total_encoder,
            "total_decoder": total_decoder,
        },
        "representative_parameter_grad_norms": representative_norms(),
        "checks": checks,
    }

    print(json.dumps(report, indent=2, sort_keys=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Wrote gradient audit:", args.output.resolve())

    failed = [key for key, ok in checks.items() if not ok]
    if failed:
        print("FAILED CHECKS:", ", ".join(failed), file=sys.stderr)
        return 1
    print("All v3.7 gradient-path checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
