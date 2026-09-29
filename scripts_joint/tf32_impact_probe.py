#!/usr/bin/env python
"""Measure the numerical impact of TF32 on the joint model, loss and gradients.

Builds the Run G model (4x128) from its config with real condition statistics,
then runs the same forward/loss/backward pass with TF32 off and on and reports
the relative differences. Also reports a pure fp32-vs-TF32 matmul error and a
matmul timing ratio, so the accuracy and the speed claims are separated.

This is a numerical probe, not a training run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (
    REPO_ROOT / "scripts",
    REPO_ROOT / "scripts_sota",
    REPO_ROOT / "scripts_joint",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from cms_data import load_config  # noqa: E402
from joint_data import resolve_joint_config  # noqa: E402
from joint_model import build_joint_autoencoder  # noqa: E402
from run_joint import build_loss_factories  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs_joint" / "cms_Joint_runG.yaml")
    parser.add_argument("--events", type=int, default=8192)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "cms_Joint" / "tf32_probe")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU (TF32 is a no-op there).")
    return parser.parse_args()


def set_tf32(enabled: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = bool(enabled)
    torch.backends.cudnn.allow_tf32 = bool(enabled)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if enabled else "highest")


def grad_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().pow(2).sum())
    return total**0.5


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if args.cpu else "cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    config = resolve_joint_config(load_config(args.config))

    # Real condition statistics from a subsample of the region caches.
    cache_root = Path(config["paths"]["cache_root"])
    region_arrays = {}
    for region in config["region_order"]:
        candidates = sorted(
            (cache_root / region).glob("selected_split_*.npz"),
            key=lambda path: path.stat().st_size,
        )
        with np.load(candidates[-1], allow_pickle=True) as data:
            region_arrays[region] = {
                "x_train": data["x_train"][:200000],
                "z_train": data["z_train"][:80000],
            }
    jpsi = region_arrays["jpsi"]

    torch.manual_seed(20260910)
    model = build_joint_autoencoder(
        config["model"],
        region_arrays,
        float(config.get("muon_mass_gev", 0.1056583755)),
        config["model"].get("daughter_masses"),
    ).to(device)
    factories = build_loss_factories(config, region_arrays)
    factory = factories["jpsi"]
    factory.set_num_slices(256)

    x = torch.as_tensor(jpsi["x_train"][: args.events], dtype=torch.float32, device=device)
    z = torch.as_tensor(jpsi["z_train"][: args.events], dtype=torch.float32, device=device)

    def probe(enabled: bool) -> dict:
        set_tf32(enabled)
        model.set_noise_multipliers(0.0, 0.0)
        model.eval()
        with torch.no_grad():
            torch.manual_seed(11)
            encoded = model.encode(x)
            direct = model.decode(z)
            factory.reset_components()
            torch.manual_seed(7)
            l_direct = float(factory.x_sim_loss(x, direct))
            torch.manual_seed(7)
            l_latent = float(factory.z_prior_loss(z, encoded))
            torch.manual_seed(7)
            l_cycle = float(factory.x_reco_loss(x, model.decode(encoded)))
        model.train()
        model.zero_grad(set_to_none=True)
        torch.manual_seed(7)
        encoded = model.encode(x)
        direct = model.decode(z)
        cycle = model.decode(encoded)
        loss = (
            factory.x_reco_loss(x, cycle)
            + factory.z_prior_loss(z, encoded)
            + factory.x_sim_loss(x, direct)
        )
        loss.backward()
        result = {
            "encoded": encoded.detach().cpu().numpy(),
            "direct": direct.detach().cpu().numpy(),
            "l_direct": l_direct,
            "l_latent": l_latent,
            "l_cycle": l_cycle,
            "total_loss": float(loss.detach().cpu()),
            "encoder_grad_norm": grad_norm(model.encoder.parameters()),
            "decoder_grad_norm": grad_norm(model.decoder.parameters()),
        }
        model.zero_grad(set_to_none=True)
        return result

    if device.type != "cuda":
        report = {"device": "cpu", "note": "TF32 is a CUDA-only setting; both arms are fp32."}
    fp32 = probe(False)
    tf32 = probe(True)

    def rel_scalar(a: float, b: float) -> float:
        return abs(a - b) / max(abs(a), abs(b), 1e-30)

    def rel_tensor(a: np.ndarray, b: np.ndarray) -> dict:
        scale = max(float(np.abs(a).max()), 1e-30)
        return {
            "max_abs_diff": float(np.abs(a - b).max()),
            "max_rel_diff_vs_scale": float(np.abs(a - b).max() / scale),
            "rms_rel_diff_vs_scale": float(np.sqrt(((a - b) ** 2).mean()) / scale),
        }

    # Pure matmul error and timing at a representative shape.
    a = torch.randn(16384, 256, device=device)
    b = torch.randn(256, 256, device=device)
    set_tf32(False)
    ref = (a @ b).clone()
    set_tf32(True)
    approx = a @ b
    matmul_rel = float((approx - ref).abs().max() / max(ref.abs().max(), 1e-30))
    timings = {}
    for enabled in (False, True):
        set_tf32(enabled)
        for _ in range(3):
            _ = a @ b
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(50):
            _ = a @ b
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings["tf32" if enabled else "fp32"] = time.perf_counter() - start

    report = {
        "device": str(device),
        "config": str(args.config),
        "hidden_dims": config["model"]["hidden_dims"],
        "events": args.events,
        "matmul_shape": [16384, 256, 256],
        "pure_matmul_max_rel_error": matmul_rel,
        "matmul_seconds_50": timings,
        "matmul_speedup": (
            timings["fp32"] / timings["tf32"] if timings.get("tf32") else None
        ),
        "encoded": rel_tensor(fp32["encoded"], tf32["encoded"]),
        "direct": rel_tensor(fp32["direct"], tf32["direct"]),
        "losses": {
            "l_direct": {"fp32": fp32["l_direct"], "tf32": tf32["l_direct"],
                         "relative_difference": rel_scalar(fp32["l_direct"], tf32["l_direct"])},
            "l_latent": {"fp32": fp32["l_latent"], "tf32": tf32["l_latent"],
                         "relative_difference": rel_scalar(fp32["l_latent"], tf32["l_latent"])},
            "l_cycle": {"fp32": fp32["l_cycle"], "tf32": tf32["l_cycle"],
                        "relative_difference": rel_scalar(fp32["l_cycle"], tf32["l_cycle"])},
            "total_loss": {"fp32": fp32["total_loss"], "tf32": tf32["total_loss"],
                           "relative_difference": rel_scalar(fp32["total_loss"], tf32["total_loss"])},
        },
        "gradient_norms": {
            "encoder": {"fp32": fp32["encoder_grad_norm"], "tf32": tf32["encoder_grad_norm"],
                        "relative_difference": rel_scalar(fp32["encoder_grad_norm"], tf32["encoder_grad_norm"])},
            "decoder": {"fp32": fp32["decoder_grad_norm"], "tf32": tf32["decoder_grad_norm"],
                        "relative_difference": rel_scalar(fp32["decoder_grad_norm"], tf32["decoder_grad_norm"])},
        },
    }
    (args.output_dir / "tf32_impact.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
