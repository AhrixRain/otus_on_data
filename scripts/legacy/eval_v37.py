#!/usr/bin/env python
"""Three-path evaluation for the vanilla OTUS/SWAE experiments (v3.7/v3.8).

For each requested checkpoint, evaluates on the fixed held-out test sample:

  A. encoder-only:      z_hat = E(x_test)           vs. z_test MG5 prior
  B. cycle:             x_cycle = D(E(x_test))      vs. x_test CMS
  C. generator:         x_gen   = D(z_test)         vs. x_test CMS

All stochastic draws and SW projections use fixed seeds, so every checkpoint
(including every lambda run) is evaluated on exactly the same
events/projections. Training is never started by this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from typing import Any

import numpy as np
import torch

from cms_data import load_and_split_cached, load_config, resolve_config
from cms_model import load_model_from_checkpoint
from cms_training import build_loss_factory, first_tensor
from device_utils import device_report, select_device
from loss import build_ee_physics_features
from metrics import invariant_mass

try:
    from scipy.stats import ks_2samp

    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


COMPONENT_LABELS = [
    "mu_minus_px",
    "mu_minus_py",
    "mu_minus_pz",
    "mu_minus_E",
    "mu_plus_px",
    "mu_plus_py",
    "mu_plus_pz",
    "mu_plus_E",
]
PHYSICS_LABELS = [
    "pt_m",
    "pt_p",
    "eta_m",
    "eta_p",
    "m_ll",
    "pt_ll",
    "y_ll",
    "cos_dphi",
    "sin_dphi",
    "pair_px",
    "pair_py",
    "pair_pz",
    "dpx",
    "dpy",
    "dpz",
]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        torch.mps.manual_seed(seed)


def finite(a: np.ndarray) -> np.ndarray:
    values = np.asarray(a).reshape(-1)
    return values[np.isfinite(values)]


def maybe_ks(a: np.ndarray, b: np.ndarray) -> float | None:
    if not HAS_SCIPY:
        return None
    a = finite(a)
    b = finite(b)
    if len(a) == 0 or len(b) == 0:
        return None
    return float(ks_2samp(a, b).statistic)


def pair_pt(pairs: np.ndarray) -> np.ndarray:
    px = pairs[:, 0] + pairs[:, 4]
    py = pairs[:, 1] + pairs[:, 5]
    return np.sqrt(px**2 + py**2)


def ks_dict(truth: np.ndarray, pred: np.ndarray, labels: list[str]) -> dict[str, float | None]:
    return {
        label: maybe_ks(truth[:, j], pred[:, j])
        for j, label in enumerate(labels)
    }


def ks_summary(values: dict[str, float | None]) -> tuple[float | None, float | None]:
    present = [value for value in values.values() if value is not None]
    if not present:
        return None, None
    return float(np.mean(present)), float(np.max(present))


def encode_batches(
    model: torch.nn.Module,
    x: np.ndarray,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    model.eval()
    set_seed(seed)
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.as_tensor(
                x[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            outputs.append(first_tensor(model.encode(batch)).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def decode_batches(
    model: torch.nn.Module,
    z: np.ndarray,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    model.eval()
    set_seed(seed)
    outputs = []
    with torch.no_grad():
        for start in range(0, len(z), batch_size):
            batch = torch.as_tensor(
                z[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            outputs.append(first_tensor(model.decode(batch)).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def physics_features_np(
    values: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    outputs = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.as_tensor(
                values[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            outputs.append(
                build_ee_physics_features(batch)["physics_features"]
                .detach()
                .cpu()
                .numpy()
            )
    return np.concatenate(outputs, axis=0)


def latent_sw(
    factory: Any,
    z_true: np.ndarray,
    z_encoded: np.ndarray,
    device: torch.device,
    num_slices: int,
    seed: int,
) -> float:
    factory.set_num_slices(num_slices)
    set_seed(seed)
    with torch.no_grad():
        loss = factory.z_prior_loss(
            torch.as_tensor(z_true, dtype=torch.float32, device=device),
            torch.as_tensor(z_encoded, dtype=torch.float32, device=device),
        )
    return float(loss.detach().cpu())


def standardized_mse_np(
    factory: Any,
    truth: np.ndarray,
    pred: np.ndarray,
) -> float:
    mean = np.asarray(factory.x_space.raw_mean, dtype=np.float64)
    std = np.asarray(factory.x_space.raw_std, dtype=np.float64)
    eps = float(factory.x_space.eps)
    a = (truth.astype(np.float64) - mean) / (std + eps)
    b = (pred.astype(np.float64) - mean) / (std + eps)
    return float(np.mean((a - b) ** 2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint .pt file, or a run directory to evaluate all v3.7 checkpoints.",
    )
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--sw-batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def evaluate_checkpoint(
    checkpoint_path: Path,
    config: dict[str, Any],
    arrays: dict[str, np.ndarray],
    device: torch.device,
    *,
    seed: int,
    batch_size: int,
    sw_batch_size: int,
    num_slices: int,
    sw_eval_seed: int,
    output_dir: Path,
    mode: str,
) -> dict[str, Any]:
    model, model_config, _stats, checkpoint = load_model_from_checkpoint(
        checkpoint_path,
        config=None,
        map_location=torch.device("cpu"),
    )
    model.to(device)

    x_eval = arrays["x_test"]
    z_eval = arrays["z_test"]
    z_hat = encode_batches(model, x_eval, device, batch_size, seed + 0)
    x_cycle = decode_batches(model, z_hat, device, batch_size, seed + 1)
    x_gen = decode_batches(model, z_eval, device, batch_size, seed + 2)

    checkpoint_stages = model_config.get("stages") or [{}]
    checkpoint_lambda = float(checkpoint_stages[0].get("lamb", 1.0))
    checkpoint_slices = int(checkpoint_stages[0].get("num_slices", num_slices))
    loss_factory = build_loss_factory(
        arrays["x_train"],
        arrays["z_train"],
        model_config.get("loss", {}),
        daughter_masses=(model_config.get("model") or {}).get("daughter_masses"),
    )
    sw_n = min(sw_batch_size, len(x_eval), len(z_eval))
    z_sw = latent_sw(
        loss_factory,
        z_eval[:sw_n],
        z_hat[:sw_n],
        device,
        checkpoint_slices,
        sw_eval_seed,
    )

    z_physics = physics_features_np(z_eval, device, batch_size)
    zhat_physics = physics_features_np(z_hat, device, batch_size)
    x_physics = physics_features_np(x_eval, device, batch_size)
    xcycle_physics = physics_features_np(x_cycle, device, batch_size)
    xgen_physics = physics_features_np(x_gen, device, batch_size)

    z_component_ks = ks_dict(z_eval, z_hat, COMPONENT_LABELS)
    z_physics_ks = ks_dict(z_physics, zhat_physics, PHYSICS_LABELS)
    cycle_component_ks = ks_dict(x_eval, x_cycle, COMPONENT_LABELS)
    cycle_physics_ks = ks_dict(x_physics, xcycle_physics, PHYSICS_LABELS)
    gen_component_ks = ks_dict(x_eval, x_gen, COMPONENT_LABELS)
    gen_physics_ks = ks_dict(x_physics, xgen_physics, PHYSICS_LABELS)

    z_mean_ks, z_max_ks = ks_summary(z_component_ks)
    z_phys_mean_ks, z_phys_max_ks = ks_summary(z_physics_ks)
    cycle_mean_ks, cycle_max_ks = ks_summary(cycle_component_ks)
    cycle_phys_mean_ks, cycle_phys_max_ks = ks_summary(cycle_physics_ks)
    gen_mean_ks, gen_max_ks = ks_summary(gen_component_ks)
    gen_phys_mean_ks, gen_phys_max_ks = ks_summary(gen_physics_ks)

    cycle_metrics = {
        "reco_mse_raw": float(np.mean((x_eval - x_cycle) ** 2)),
        "reco_mse_standardized": standardized_mse_np(loss_factory, x_eval, x_cycle),
        "mass_ks": maybe_ks(invariant_mass(x_eval), invariant_mass(x_cycle)),
        "pt_ks": maybe_ks(pair_pt(x_eval), pair_pt(x_cycle)),
        "mean_ks_8d": cycle_mean_ks,
        "max_ks_8d": cycle_max_ks,
        "component_ks": cycle_component_ks,
        "mean_ks_physics": cycle_phys_mean_ks,
        "max_ks_physics": cycle_phys_max_ks,
        "physics_ks": cycle_physics_ks,
    }
    summary = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_kind": checkpoint_path.stem,
        "epoch": checkpoint.get("epoch"),
        "lambda": checkpoint_lambda,
        "seed": int(seed),
        "num_slices": int(checkpoint_slices),
        "sw_batch_size": int(sw_n),
        "transform_batch_size": int(batch_size),
        "n_x_eval": int(len(x_eval)),
        "n_z_eval": int(len(z_eval)),
        "scipy_ks": bool(HAS_SCIPY),
        "encoder_only": {
            "z_sw": z_sw,
            "z_mean_ks": z_mean_ks,
            "z_max_ks": z_max_ks,
            "z_component_ks": z_component_ks,
            "z_mass_ks": maybe_ks(
                invariant_mass(z_eval, stable=False),
                invariant_mass(z_hat, stable=False),
            ),
            "z_pt_ks": maybe_ks(pair_pt(z_eval), pair_pt(z_hat)),
            "z_physics_mean_ks": z_phys_mean_ks,
            "z_physics_max_ks": z_phys_max_ks,
            "z_physics_ks": z_physics_ks,
        },
        # "cycle" is retained for v3.7 output compatibility; v3.8 outputs use
        # the "reconstruction" name for the same x -> z -> x path.
        "cycle": cycle_metrics,
        "reconstruction": cycle_metrics,
        "generator": {
            "mass_ks": maybe_ks(invariant_mass(x_eval), invariant_mass(x_gen)),
            "pt_ks": maybe_ks(pair_pt(x_eval), pair_pt(x_gen)),
            "mean_ks_8d": gen_mean_ks,
            "max_ks_8d": gen_max_ks,
            "component_ks": gen_component_ks,
            "mean_ks_physics": gen_phys_mean_ks,
            "max_ks_physics": gen_phys_max_ks,
            "physics_ks": gen_physics_ks,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{mode}_eval_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("Wrote:", summary_path.resolve())
    return summary


def main() -> int:
    args = parse_args()
    config = resolve_config(load_config(args.config))
    seed = int(config.get("seed", 0) if args.seed is None else args.seed)
    device = select_device(args.device)
    report = device_report(device)
    arrays, cache_info = load_and_split_cached(
        config,
        num_samples=args.num_samples,
        cache_dir=None,
        use_cache=True,
    )
    print("Data cache:", json.dumps(cache_info, sort_keys=True))
    if args.num_samples is not None:
        arrays = {
            key: value[: args.num_samples]
            for key, value in arrays.items()
            if key in {"x_train", "z_train", "x_val", "z_val", "x_test", "z_test"}
        }

    checkpoint_arg = args.checkpoint.expanduser().resolve()
    if checkpoint_arg.is_dir():
        candidates = [
            "best_model.pt",
            "best_combined.pt",
            "best_z_prior.pt",
            "best_cycle.pt",
            "best_reconstruction.pt",
            "checkpoint_final.pt",
            "last_model.pt",
        ]
        checkpoint_paths = [
            checkpoint_arg / name for name in candidates if (checkpoint_arg / name).exists()
        ]
        if not checkpoint_paths:
            print("ERROR: no vanilla-SWAE checkpoints found in", checkpoint_arg, file=sys.stderr)
            return 2
    else:
        checkpoint_paths = [checkpoint_arg]

    loss_config = config.get("loss", {})
    num_slices = int(config["stages"][0].get("num_slices", 1000))
    eval_config = config.get("evaluation", {})
    batch_size = int(
        args.batch_size
        or config.get("loaders", {}).get("eval_batch_size", 20000)
    )
    sw_batch_size = int(
        args.sw_batch_size
        or eval_config.get("sw_batch_size")
        or eval_config.get("v37_sw_batch_size", batch_size)
    )
    sw_eval_seed = int(eval_config.get("sw_eval_seed", seed + 3))

    vanilla_v3_7 = bool(loss_config.get("vanilla_v3_7", False))
    vanilla_swae = bool(loss_config.get("vanilla_swae", False)) or vanilla_v3_7
    if not vanilla_swae:
        print(
            "ERROR: --config does not enable vanilla_v3_7 or vanilla_swae.",
            file=sys.stderr,
        )
        return 2
    mode = "v37" if vanilla_v3_7 else "v38"

    for checkpoint_path in checkpoint_paths:
        if args.output_dir is not None:
            out_dir = args.output_dir / f"{mode}_eval" / checkpoint_path.stem
        else:
            out_dir = checkpoint_path.parent / f"{mode}_eval" / checkpoint_path.stem
        evaluate_checkpoint(
            checkpoint_path,
            config,
            arrays,
            device,
            seed=seed,
            batch_size=batch_size,
            sw_batch_size=sw_batch_size,
            num_slices=num_slices,
            sw_eval_seed=sw_eval_seed,
            output_dir=out_dir,
            mode=mode,
        )

    print("Using device:", device)
    print("Device report:", json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
