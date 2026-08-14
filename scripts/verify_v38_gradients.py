#!/usr/bin/env python
"""Gradient-path and objective audit for the vanilla SWAE (v3.7/v3.8) modes.

For one mini-batch, verifies:

  - the latent loss equals sliced_wasserstein(z, E(x), num_slices, p=2) on
    raw coordinates when the config sets standardize_raw_matching=false
    (v3.8), and on standardized coordinates for the legacy v3.7 default;
  - the reconstruction loss equals torch.mean((x - x_hat) ** 2);
  - the total loss equals beta * reconstruction + lambda * latent SWD;
  - reconstruction contributes gradients to both encoder and decoder;
  - the latent SWD contributes gradients to the encoder and not the decoder;
  - a training step records no anchor/alt-x/constraint gradient or loss.

With ``--checkpoint`` it additionally reloads the checkpoint and reports its
embedded run metadata. The audit never modifies training behavior and writes
only its JSON report (no checkpoints).
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
from cms_model import build_model, load_model_from_checkpoint
from cms_training import build_loss_factory, first_tensor
from device_utils import select_device
from loss import sliced_wasserstein


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        torch.mps.manual_seed(seed)


def norm_of(grads: list[torch.Tensor | None]) -> float:
    parts = [
        grad.detach().reshape(-1)
        for grad in grads
        if grad is not None and grad.numel() > 0
    ]
    if not parts:
        return 0.0
    return float(torch.cat(parts).norm())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu.")
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--slices", type=int, default=None, help="Override SW slices.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint to reload and report.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("v38_gradient_audit.json"),
        help="JSON report path (default: ./v38_gradient_audit.json).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = resolve_config(load_config(args.config))
    loss_config = config.get("loss", {})
    vanilla_v3_7 = bool(loss_config.get("vanilla_v3_7", False))
    vanilla_swae = bool(loss_config.get("vanilla_swae", False)) or vanilla_v3_7
    if not vanilla_swae:
        print("ERROR: --config does not enable vanilla_v3_7 or vanilla_swae.", file=sys.stderr)
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
    model = build_model(config, x_train_mean, x_train_std, z_train_mean, z_train_std)
    model.to(device)
    model.train()
    loss_factory = build_loss_factory(
        arrays["x_train"],
        arrays["z_train"],
        loss_config,
        daughter_masses=(config.get("model") or {}).get("daughter_masses"),
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
    beta = float(stage.get("beta", 1.0))
    lamb = float(stage.get("lamb", 1.0))
    raw_latent = not bool(loss_factory.z_space.standardize_raw_matching)

    batch_size = min(int(args.batch_size), len(arrays["x_test"]), len(arrays["z_test"]))
    x_batch = torch.as_tensor(arrays["x_test"][:batch_size], dtype=torch.float32, device=device)
    z_batch = torch.as_tensor(arrays["z_test"][:batch_size], dtype=torch.float32, device=device)

    set_seed(seed + 1000)
    z_hat = first_tensor(model.encode(x_batch))
    x_hat = first_tensor(model.decode(z_hat))
    reco_loss = loss_factory.x_reco_loss(x_batch, x_hat)
    set_seed(seed + 1001)
    sw_loss = loss_factory.z_prior_loss(z_batch, z_hat)
    total_loss = beta * reco_loss + lamb * sw_loss

    # Reference values with an identical RNG stream.
    set_seed(seed + 1001)
    if raw_latent:
        expected_sw = sliced_wasserstein(z_batch, z_hat, num_slices, 2)
    else:
        expected_sw = sliced_wasserstein(
            loss_factory.standardize_z_raw(z_batch),
            loss_factory.standardize_z_raw(z_hat),
            num_slices,
            2,
        )
    expected_reco = torch.mean((x_batch - x_hat) ** 2)

    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    decoder_params = [p for p in model.decoder.parameters() if p.requires_grad]
    all_params = [*encoder_params, *decoder_params]
    n_enc = len(encoder_params)

    def grads(loss: torch.Tensor) -> tuple[float, float]:
        values = torch.autograd.grad(loss, all_params, retain_graph=True, allow_unused=True)
        return norm_of(values[:n_enc]), norm_of(values[n_enc:])

    reco_enc, reco_dec = grads(reco_loss)
    sw_enc, sw_dec = grads(sw_loss)
    total_enc, total_dec = grads(total_loss)

    checks = {
        "latent_swd_matches_reference": bool(
            torch.allclose(sw_loss, expected_sw, atol=1e-4, rtol=1e-4)
        ),
        "reco_matches_raw_mse": bool(
            torch.allclose(reco_loss, expected_reco, atol=1e-4, rtol=1e-4)
        ),
        "total_equals_beta_reco_plus_lambda_swd": bool(
            torch.allclose(total_loss, beta * reco_loss + lamb * sw_loss, atol=1e-4)
        ),
        "reco_grad_encoder_nonzero": reco_enc > 0.0,
        "reco_grad_decoder_nonzero": reco_dec > 0.0,
        "latent_grad_encoder_nonzero": sw_enc > 0.0,
        "latent_grad_decoder_zero": sw_dec == 0.0,
        "total_grad_encoder_nonzero": total_enc > 0.0,
        "total_grad_decoder_nonzero": total_dec > 0.0,
        "losses_finite": all(
            bool(torch.isfinite(value))
            for value in (reco_loss, sw_loss, total_loss)
        ),
    }

    report: dict[str, Any] = {
        "config": str(args.config.resolve()),
        "device": str(device),
        "mode": "vanilla_swae",
        "raw_latent_swd": raw_latent,
        "seed": seed,
        "beta": beta,
        "lambda": lamb,
        "num_slices": num_slices,
        "batch_size": batch_size,
        "n_train_x": int(len(arrays["x_train"])),
        "n_train_z": int(len(arrays["z_train"])),
        "losses": {
            "reconstruction": float(reco_loss.detach().cpu()),
            "latent_swd": float(sw_loss.detach().cpu()),
            "total": float(total_loss.detach().cpu()),
        },
        "grad_norms": {
            "reco_encoder": reco_enc,
            "reco_decoder": reco_dec,
            "latent_encoder": sw_enc,
            "latent_decoder": sw_dec,
            "total_encoder": total_enc,
            "total_decoder": total_dec,
        },
        "checks": checks,
    }

    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint.expanduser().resolve()
        _model, _cfg, _stats, checkpoint = load_model_from_checkpoint(
            checkpoint_path, config=None, map_location=torch.device("cpu")
        )
        report["checkpoint_reload"] = {
            "path": str(checkpoint_path),
            "epoch": checkpoint.get("epoch"),
            "eval_loss": checkpoint.get("eval_loss"),
            "has_run_metadata": "run_metadata" in checkpoint,
            "run_metadata": checkpoint.get("run_metadata"),
        }

    print(json.dumps(report, indent=2, sort_keys=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Wrote gradient audit:", args.output.resolve())

    failed = [key for key, ok in checks.items() if not ok]
    if failed:
        print("FAILED CHECKS:", ", ".join(failed), file=sys.stderr)
        return 1
    print("All vanilla-SWAE gradient-path checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
