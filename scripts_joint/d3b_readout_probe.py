#!/usr/bin/env python
"""D3b read-out probe: is the mean map carrying the J/psi resolution?

Read-only diagnostic.  It answers one question with numbers taken from the
locked honest-prior splits and existing checkpoints:

    At the model's own operating point (native noise, frozen kernel), how much
    of the decoded J/psi mass width is contributed by the deterministic mean
    map ``Var_z[E(x|z)]``, and how much by the response channel
    ``E_z[Var(x|z)]`` -- and how does the mean-map share compare with a map
    that carries no resolution at all (a mean map pinned to the prior width)?

Motivation (memory.md Session 64/65).  ``var = E_z[Var(x|z)] + Var_z[E(x|z)]``
is exact.  A checkpoint whose mean map has absorbed the detector resolution
shows a large ``across`` term at *zero* noise (the zero-noise decode is the
deterministic map), and the total native width then overshoots because the
frozen kernel is added on top.  The clean signature is

    across_std(zero) / prior_width  ~ 1   -> the mean map carries no resolution
    across_std(zero) / prior_width  >> 1  -> the mean map absorbed it

Nothing is trained.  Nothing under ``data/``, ``outputs/`` or any checkpoint is
modified: the resolved config and the locked region cache are read, checkpoints
are loaded read-only, and every artifact is written to a new ``--output-dir``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

MASS_INDEX = 8


def find_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "scripts_joint").is_dir() and (candidate / "scripts_sota").is_dir():
            return candidate
    raise RuntimeError("Could not locate the OTUS repository root")


REPO_ROOT = find_repo_root()
for directory in (
    REPO_ROOT / "scripts",
    REPO_ROOT / "scripts_sota",
    REPO_ROOT / "scripts_joint",
    REPO_ROOT / "scripts_joint" / "upsilon",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from decode_prior import load_frozen_model  # noqa: E402
from joint_data import load_joint_regions, resolve_joint_config  # noqa: E402


def mass8(values: np.ndarray) -> np.ndarray:
    """Stable dimuon invariant mass from an (N, 8) [p-, E-, p+, E+] array."""
    values = np.asarray(values, dtype=np.float64)
    energy = values[:, 3] + values[:, 7]
    momentum = values[:, :3] + values[:, 4:7]
    return np.sqrt(np.maximum(energy * energy - np.sum(momentum * momentum, axis=1), 0.0))


def mass8_torch(values: torch.Tensor) -> torch.Tensor:
    energy = values[:, 3] + values[:, 7]
    momentum = values[:, :3] + values[:, 4:7]
    return torch.sqrt(
        torch.clamp(energy * energy - torch.sum(momentum * momentum, dim=1), min=0.0)
    )


def robust_half_width(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    lower = np.quantile(values, 0.16, axis=axis)
    upper = np.quantile(values, 0.84, axis=axis)
    return (upper - lower) / 2.0


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def decode_mass_matrix(
    model,
    z: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    core: float,
    tail: float,
    draws: int,
    seed: int,
) -> np.ndarray:
    """Decode each fixed z ``draws`` times; return an (N, draws) mass matrix."""
    model.decoder.set_noise_multipliers(core, tail)
    torch.manual_seed(int(seed))
    n_events = len(z)
    indices = np.repeat(np.arange(n_events), draws)
    flat = np.empty(len(indices), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            block = indices[start : start + batch_size]
            values = torch.as_tensor(
                np.ascontiguousarray(z[block]), dtype=torch.float32, device=device
            )
            decoded = model.decode(values)
            flat[start : start + len(block)] = (
                mass8_torch(decoded).detach().cpu().numpy()
            )
    return flat.reshape(n_events, draws)


def matrix_stats(matrix: np.ndarray) -> dict:
    """Within-z (noise) and across-z (mean map) statistics of a decode matrix."""
    n_events, draws = matrix.shape
    within = matrix.std(axis=1, ddof=0)
    conditional_mean = matrix.mean(axis=1)
    within_var = float(np.mean(within**2))
    across_var = float(np.var(conditional_mean, ddof=0))
    total_var = float(np.var(matrix.ravel(), ddof=0))
    deterministic = bool(
        draws == 1 or np.array_equal(matrix, np.repeat(matrix[:, :1], draws, axis=1))
    )
    return {
        "n_events": int(n_events),
        "draws": int(draws),
        "deterministic_map": deterministic,
        "ensemble_mean_gev": float(np.mean(matrix)),
        "ensemble_std_gev": float(np.sqrt(total_var)),
        "ensemble_robust_half_width_gev": float(robust_half_width(matrix.ravel())),
        "across_std_gev": float(np.sqrt(across_var)),
        "across_robust_half_width_gev": float(robust_half_width(conditional_mean)),
        "within_std_median_gev": float(np.median(within)),
        "within_robust_median_gev": float(
            np.median(robust_half_width(matrix, axis=1))
        ),
        "within_variance_mean_gev2": within_var,
        "noise_variance_fraction": (
            float(within_var / total_var) if total_var > 0 else None
        ),
    }


def parse_checkpoint(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise SystemExit(f"--checkpoint expects LABEL=RELATIVE_PATH, got {raw!r}")
    label, relative = raw.split("=", 1)
    return label, (REPO_ROOT / relative)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="LABEL=RELATIVE_PATH",
        help="checkpoint to probe (repeatable); the first one supplies the config",
    )
    parser.add_argument("--region", default="jpsi")
    parser.add_argument("--events", type=int, default=12000)
    parser.add_argument("--draws", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "cms_Joint" / "d3b_readout_probe",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"

    def log(message: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    device = torch.device(args.device)
    checkpoints = [parse_checkpoint(raw) for raw in args.checkpoint]
    for label, path in checkpoints:
        if not path.exists():
            raise FileNotFoundError(f"checkpoint {label} not found: {path}")

    log(f"D3b read-out probe on {device}; region {args.region!r}; output {output_dir}")

    # ---- locked splits and the reference widths ------------------------------
    _, _, config = load_frozen_model(checkpoints[0][1], torch.device("cpu"))
    resolved = resolve_joint_config(config)
    arrays, cache_info, region_configs, _ = load_joint_regions(
        resolved, num_samples=None, use_cache=True, log=lambda message: None
    )
    region = args.region
    if region not in arrays:
        raise SystemExit(f"region {region!r} not in the locked split: {sorted(arrays)}")
    z_test = np.asarray(arrays[region]["z_test"], dtype=np.float32)
    x_test = np.asarray(arrays[region]["x_test"], dtype=np.float32)
    region_config = region_configs[region]
    selection = region_config.get("theory_prior_selection") or {}
    log(
        f"locked split {region}: z_test {len(z_test)}, x_test {len(x_test)}; "
        f"prior window [{selection.get('mass_min')}, {selection.get('mass_max')}] GeV"
    )

    rng = np.random.default_rng(args.seed)
    count = min(int(args.events), len(z_test))
    chosen = np.sort(rng.choice(len(z_test), size=count, replace=False))
    z_fixed = np.ascontiguousarray(z_test[chosen])

    prior_mass = mass8(z_fixed)
    cms_count = min(200_000, len(x_test))
    cms_mass = mass8(np.ascontiguousarray(x_test[:cms_count]))
    reference = {
        "region": region,
        "n_prior": int(len(prior_mass)),
        "n_cms": int(len(cms_mass)),
        "prior_window_gev": [
            selection.get("mass_min"),
            selection.get("mass_max"),
        ],
        "prior_mass_std_gev": float(np.std(prior_mass)),
        "prior_mass_robust_half_width_gev": float(robust_half_width(prior_mass)),
        "cms_mass_std_gev": float(np.std(cms_mass)),
        "cms_mass_robust_half_width_gev": float(robust_half_width(cms_mass)),
        "required_quadrature_std_gev": float(
            np.sqrt(max(np.var(cms_mass) - np.var(prior_mass), 0.0))
        ),
        "cache_dir": cache_info[region].get("cache_dir")
        if isinstance(cache_info[region], dict)
        else None,
    }
    log(
        f"prior mass std {reference['prior_mass_std_gev']:.5g} GeV (robust "
        f"{reference['prior_mass_robust_half_width_gev']:.5g}); CMS x mass std "
        f"{reference['cms_mass_std_gev']:.5g} GeV (robust "
        f"{reference['cms_mass_robust_half_width_gev']:.5g})"
    )

    # ---- checkpoint loop ------------------------------------------------------
    results: dict[str, dict] = {}
    for index, (label, path) in enumerate(checkpoints):
        model, checkpoint, _ = load_frozen_model(path, device)
        model.eval()
        recorded = checkpoint.get("noise_multipliers") or {}
        shared_core = float(recorded.get("core", 1.0))
        shared_tail = float(recorded.get("tail", 1.0))
        native_core = float(recorded.get("decoder_core", shared_core))
        native_tail = float(recorded.get("decoder_tail", shared_tail))
        stage = checkpoint.get("stage") or {}

        zero_matrix = decode_mass_matrix(
            model,
            z_fixed,
            batch_size=args.batch_size,
            device=device,
            core=0.0,
            tail=0.0,
            draws=2,
            seed=args.seed + 7 + index,
        )
        # The zero-noise map is deterministic; keep one column and report the
        # across-event spread, which is exactly Var_z[E(x|z)].
        zero_masses = zero_matrix[:, 0]
        native_matrix = decode_mass_matrix(
            model,
            z_fixed,
            batch_size=args.batch_size,
            device=device,
            core=native_core,
            tail=native_tail,
            draws=args.draws,
            seed=args.seed + 100 + index,
        )
        zero_stats = matrix_stats(zero_matrix)
        native_stats = matrix_stats(native_matrix)
        # The noise channel's own contribution in mass: the RMS of the per-event
        # within-z standard deviation, i.e. sqrt(E_z[Var(x|z)]).
        kernel_only_std = float(
            np.sqrt(max(native_stats["within_variance_mean_gev2"], 0.0))
        )
        sigma_noise_only = float(
            np.sqrt(
                max(
                    native_stats["ensemble_std_gev"] ** 2
                    - zero_stats["ensemble_std_gev"] ** 2,
                    0.0,
                )
            )
        )
        entry = {
            "path": str(path),
            "global_epoch": checkpoint.get("global_epoch"),
            "stage_name": stage.get("name") if isinstance(stage, dict) else None,
            "recorded_noise_multipliers": json_ready(recorded),
            "native_decoder_multipliers": {"core": native_core, "tail": native_tail},
            "zero_noise": {
                "ensemble_std_gev": float(np.std(zero_masses)),
                "ensemble_robust_half_width_gev": float(
                    robust_half_width(zero_masses)
                ),
                "deterministic_map": zero_stats["deterministic_map"],
                "mean_map_spread_over_prior": float(
                    np.std(zero_masses) / max(reference["prior_mass_std_gev"], 1e-12)
                ),
                "mean_map_spread_over_cms": float(
                    np.std(zero_masses) / max(reference["cms_mass_std_gev"], 1e-12)
                ),
            },
            "native_noise": {
                "ensemble_std_gev": native_stats["ensemble_std_gev"],
                "ensemble_robust_half_width_gev": native_stats[
                    "ensemble_robust_half_width_gev"
                ],
                "across_std_gev": native_stats["across_std_gev"],
                "within_std_median_gev": native_stats["within_std_median_gev"],
                "within_robust_median_gev": native_stats["within_robust_median_gev"],
                "noise_variance_fraction": native_stats["noise_variance_fraction"],
                "sigma_noise_only_gev": sigma_noise_only,
                "total_over_cms": float(
                    native_stats["ensemble_std_gev"]
                    / max(reference["cms_mass_std_gev"], 1e-12)
                ),
            },
            "kernel_only_std_gev": kernel_only_std,
            "quadrature_check_total_gev": float(
                np.sqrt(
                    native_stats["within_variance_mean_gev2"]
                    + native_stats["across_std_gev"] ** 2
                )
            ),
        }
        results[label] = entry
        log(
            f"[{index + 1}/{len(checkpoints)}] {label} ep {entry['global_epoch']}: "
            f"zero-noise mean-map spread {entry['zero_noise']['ensemble_std_gev']:.5g} GeV "
            f"({entry['zero_noise']['mean_map_spread_over_prior']:.2f}x prior, "
            f"{entry['zero_noise']['mean_map_spread_over_cms']:.2f}x CMS); "
            f"native total {entry['native_noise']['ensemble_std_gev']:.5g} GeV "
            f"({entry['native_noise']['total_over_cms']:.2f}x CMS), "
            f"noise var fraction {entry['native_noise']['noise_variance_fraction']:.3g}"
        )
        del model

    payload = {
        "schema_version": 1,
        "diagnostic": "D3b read-out probe (mean-map vs response-channel share)",
        "status": "completed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.time() - started,
        "device": str(device),
        "events": args.events,
        "draws": args.draws,
        "seed": args.seed,
        "reference": reference,
        "checkpoints": results,
    }
    json_path = output_dir / "d3b_readout.json"
    json_path.write_text(json.dumps(json_ready(payload), indent=2) + "\n", encoding="utf-8")
    log(f"wrote {json_path}")
    log(f"done in {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
