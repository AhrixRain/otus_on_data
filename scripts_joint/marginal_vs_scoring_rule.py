#!/usr/bin/env python
"""Marginal gauges vs a proper scoring rule, on our own checkpoint.

Read-only.  Tests the claim that a marginal-matching objective cannot see where
the detector resolution lives, while a strictly proper scoring rule can.

Protocol
--------
One checkpoint, two decode conditions that differ ONLY in the response channel:

    zero    decoder multipliers (0, 0)      -> the deterministic mean map
    native  the checkpoint's multipliers     -> the mean map plus the channel

For the same fixed truth events we measure, against the locked CMS x sample:

  * marginal gauges: mass W1 [GeV], mass KS, pair-pT KS, decoded mass std;
  * scores that see the conditional: the energy distance between the decoded
    ensemble and the data sample, and the kernel score with a Gaussian kernel
    whose bandwidth is set from the data sample (both strictly proper up to the
    additive constant that does not depend on the model).

The prediction being tested: the marginal gauges move little between the two
conditions (both decodes have almost the same x marginal), while the energy
distance / kernel score move substantially in favour of the channel that
actually has a conditional spread.

Nothing is trained; no checkpoint, data file or existing output is modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def find_repo_root() -> Path:
    for candidate in HERE.parents:
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


def pair_pt(values: np.ndarray) -> np.ndarray:
    momentum = values[:, :3] + values[:, 4:7]
    return np.sqrt(np.sum(momentum * momentum, axis=1))


def wasserstein_1d(real: np.ndarray, fake: np.ndarray) -> float:
    grid = np.sort(np.concatenate([real, fake]))
    cdf_real = np.searchsorted(np.sort(real), grid, side="right") / len(real)
    cdf_fake = np.searchsorted(np.sort(fake), grid, side="right") / len(fake)
    return float(np.trapz(np.abs(cdf_real - cdf_fake), grid))


def ks_distance(real: np.ndarray, fake: np.ndarray) -> float:
    grid = np.sort(np.concatenate([real, fake]))
    cdf_real = np.searchsorted(np.sort(real), grid, side="right") / len(real)
    cdf_fake = np.searchsorted(np.sort(fake), grid, side="right") / len(fake)
    return float(np.max(np.abs(cdf_real - cdf_fake)))


def _cross_l1(draws: np.ndarray, target: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """Mean |X - Y| between each draw's sample and the target, chunked over Y."""
    n_draws, n = draws.shape
    total = np.zeros(n_draws, dtype=np.float64)
    for start in range(0, len(target), chunk):
        block = target[start : start + chunk]
        diff = draws[:, :, None] - block[None, None, :]        # (m, n, c)
        total += np.abs(diff).mean(axis=(1, 2))
    return total / max(np.ceil(len(target) / chunk), 1.0)


def _cross_gaussian(draws: np.ndarray, target: np.ndarray, h2: float,
                    chunk: int = 4096) -> np.ndarray:
    """Mean exp(-(X-Y)^2 / 2h^2) between each draw's sample and the target."""
    n_draws, n = draws.shape
    total = np.zeros(n_draws, dtype=np.float64)
    for start in range(0, len(target), chunk):
        block = target[start : start + chunk]
        diff = draws[:, :, None] - block[None, None, :]
        total += np.exp(-(diff**2) / h2).mean(axis=(1, 2))
    return total / max(np.ceil(len(target) / chunk), 1.0)


def energy_distance(draws: np.ndarray, target: np.ndarray, chunk: int = 4096) -> dict:
    """Energy distance between the decoded ensemble and the target sample.

    ``draws`` is (m, n) with m independent samples of n values each, so the
    per-draw statistic is an unbiased estimate of 2E|X-Y| - E|X-X'| - E|Y-Y'|.
    """
    n = draws.shape[1]
    within_x = np.abs(draws[:, :, None] - draws[:, None, :]).mean(axis=(1, 2))
    cross = _cross_l1(draws, target, chunk=chunk)
    reference = target[: min(len(target), 8000)]
    within_y = float(
        np.abs(reference[:, None] - reference[None, :]).mean()
    )
    per_draw = 2.0 * cross - within_x - within_y
    return {
        "mean": float(np.mean(per_draw)),
        "std": float(np.std(per_draw, ddof=0)),
        "draws": int(draws.shape[0]),
        "n_target": int(len(target)),
        "n_model_per_draw": int(n),
        "within_model_term": float(np.mean(within_x)),
        "within_target_term": within_y,
    }


def kernel_score(draws: np.ndarray, target: np.ndarray, *, bandwidth: float,
                 chunk: int = 4096) -> dict:
    """Gaussian-kernel score, unbiased in the within-model term.

    S = E[k(X~,X~')] - 2 E[k(X~, x)], k = exp(-(a-b)^2 / (2 h^2)).
    Computed per draw and averaged; the target-target term is a constant and
    cancels in every comparison we report.
    """
    h2 = 2.0 * bandwidth**2
    m, n = draws.shape
    within = np.exp(-((draws[:, :, None] - draws[:, None, :]) ** 2) / h2).sum(axis=(1, 2))
    within = within / (n * (n - 1))
    cross = _cross_gaussian(draws, target, h2, chunk=chunk)
    per_draw = within - 2.0 * cross
    return {
        "mean": float(np.mean(per_draw)),
        "std": float(np.std(per_draw, ddof=0)),
        "bandwidth": float(bandwidth),
        "draws": int(m),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="joint checkpoint")
    parser.add_argument("--regions", default="jpsi,z")
    parser.add_argument("--events", type=int, default=3000, help="fixed truth events per region")
    parser.add_argument("--draws", type=int, default=10)
    parser.add_argument("--target-events", type=int, default=60000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--shuffle-target",
        action="store_true",
        help="also score a negative control: the decoded ensemble of each truth "
        "event, shuffled across truth events. This preserves the model's "
        "marginal distribution exactly (it is a permutation of the same "
        "values) while destroying the z -> x map, so a gauge that still "
        "improves on it is not measuring the map at all.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/cms_Joint/d3b_readout_probe/identity_vs_conditional",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    model, checkpoint, config = load_frozen_model(args.checkpoint, device)
    model.eval()
    resolved = resolve_joint_config(config)
    arrays, _, _, _ = load_joint_regions(
        resolved, num_samples=None, use_cache=True, log=lambda message: None
    )

    recorded = checkpoint.get("noise_multipliers") or {}
    shared_core = float(recorded.get("core", 1.0))
    shared_tail = float(recorded.get("tail", 1.0))
    native = (
        float(recorded.get("decoder_core", shared_core)),
        float(recorded.get("decoder_tail", shared_tail)),
    )
    conditions = [("zero", 0.0, 0.0), ("native", native[0], native[1])]

    rng = np.random.default_rng(args.seed)
    payload = {
        "schema_version": 1,
        "diagnostic": "marginal gauges vs proper scoring rules on one checkpoint",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "global_epoch": checkpoint.get("global_epoch"),
        "checkpoint_noise_multipliers": recorded,
        "native_decoder_multipliers": {"core": native[0], "tail": native[1]},
        "events_per_region": args.events,
        "draws": args.draws,
        "seed": args.seed,
        "regions": {},
    }

    for region in [name.strip() for name in args.regions.split(",") if name.strip()]:
        z_test = np.asarray(arrays[region]["z_test"], dtype=np.float32)
        x_test = np.asarray(arrays[region]["x_test"], dtype=np.float32)

        n_truth = min(int(args.events), len(z_test))
        z_fixed = np.ascontiguousarray(
            z_test[np.sort(rng.choice(len(z_test), size=n_truth, replace=False))]
        )
        n_target = min(int(args.target_events), len(x_test))
        x_target_raw = np.ascontiguousarray(
            x_test[np.sort(rng.choice(len(x_test), size=n_target, replace=False))]
        )
        mass_target = mass8(x_target_raw)
        scale = float(np.std(mass_target))
        bandwidth = float(np.median(np.abs(np.diff(np.sort(mass_target[:20000]))))) * 1.0
        bandwidth = max(bandwidth, 1e-4)

        entry = {
            "n_truth": int(n_truth),
            "n_target": int(n_target),
            "mass_target_std_gev": scale,
            "kernel_bandwidth_gev": bandwidth,
            "identity_marginal": {
                "mass_w1_gev": wasserstein_1d(mass_target, mass8(z_fixed)),
                "mass_ks": ks_distance(mass_target, mass8(z_fixed)),
                "pair_pt_ks": ks_distance(pair_pt(x_target_raw), pair_pt(z_fixed)),
                "model_mass_std_gev": float(np.std(mass8(z_fixed))),
            },
            "conditions": {},
        }

        per_event_by_condition: dict[str, np.ndarray] = {}

        for label, core, tail in conditions:
            model.decoder.set_noise_multipliers(core, tail)
            per_event = np.empty((args.draws, n_truth), dtype=np.float64)
            with torch.no_grad():
                for draw in range(args.draws):
                    torch.manual_seed(args.seed + 1000 * draw + (0 if label == "zero" else 7))
                    out = np.empty(n_truth, dtype=np.float64)
                    for start in range(0, n_truth, args.batch_size):
                        block = slice(start, min(start + args.batch_size, n_truth))
                        tensor = torch.as_tensor(
                            np.ascontiguousarray(z_fixed[block]),
                            dtype=torch.float32,
                            device=device,
                        )
                        out[block] = (
                            mass8_torch(model.decode(tensor)).detach().cpu().numpy()
                        )
                    per_event[draw] = out
            per_event_by_condition[label] = per_event
            flat = per_event.ravel()
            per_event_std = per_event.std(axis=0, ddof=0)
            entry["conditions"][label] = {
                "multipliers": {"core": core, "tail": tail},
                "decoded_mass_std_gev": float(np.std(flat)),
                "decoded_mass_robust_half_width_gev": float(
                    (np.quantile(flat, 0.84) - np.quantile(flat, 0.16)) / 2.0
                ),
                "within_event_std_median_gev": float(np.median(per_event_std)),
                "marginal": {
                    "mass_w1_gev": wasserstein_1d(mass_target, flat),
                    "mass_ks": ks_distance(mass_target, flat),
                    "pair_pt_ks": None,
                    "model_mass_std_gev": float(np.std(flat)),
                },
                "energy_distance": energy_distance(per_event, mass_target),
                "kernel_score": kernel_score(per_event, mass_target, bandwidth=bandwidth),
            }
            stats = entry["conditions"][label]
            print(
                f"[{region}/{label}] mass W1 {stats['marginal']['mass_w1_gev']:.5f} GeV, "
                f"mass KS {stats['marginal']['mass_ks']:.5f}, "
                f"std {stats['decoded_mass_std_gev']:.5f} GeV | "
                f"energy distance {stats['energy_distance']['mean']:+.5f} "
                f"(+-{stats['energy_distance']['std']:.5f}) | "
                f"kernel score {stats['kernel_score']['mean']:+.4f}"
            )

        zero = entry["conditions"]["zero"]
        native_entry = entry["conditions"]["native"]
        entry["comparison"] = {
            "marginal_mass_w1_change": native_entry["marginal"]["mass_w1_gev"]
            - zero["marginal"]["mass_w1_gev"],
            "marginal_mass_ks_change": native_entry["marginal"]["mass_ks"]
            - zero["marginal"]["mass_ks"],
            "decoded_std_change_gev": native_entry["decoded_mass_std_gev"]
            - zero["decoded_mass_std_gev"],
            "energy_distance_change": native_entry["energy_distance"]["mean"]
            - zero["energy_distance"]["mean"],
            "energy_distance_sigma": (
                abs(native_entry["energy_distance"]["mean"] - zero["energy_distance"]["mean"])
                / max(
                    np.hypot(
                        native_entry["energy_distance"]["std"],
                        zero["energy_distance"]["std"],
                    ),
                    1e-12,
                )
            ),
            "kernel_score_change": native_entry["kernel_score"]["mean"]
            - zero["kernel_score"]["mean"],
            "note": (
                "A marginal gauge that does not move while the energy distance / "
                "kernel score moves is the direct measurement that marginal "
                "metrics cannot see the conditional spread."
            ),
        }

        if args.shuffle_target:
            # Negative control. Permuting the decoded ensemble across truth
            # events leaves the pooled marginal EXACTLY unchanged (it is the
            # same multiset of values) but destroys any z -> x map.
            control = {}
            for label in ("zero", "native"):
                stats = entry["conditions"][label]
                permuted = np.empty_like(per_event_by_condition[label])
                for draw in range(args.draws):
                    permuted[draw] = rng.permutation(per_event_by_condition[label][draw])
                flat = permuted.ravel()
                control[label] = {
                    "decoded_mass_std_gev": float(np.std(flat)),
                    "within_event_std_median_gev": float(
                        np.median(permuted.std(axis=0, ddof=0))
                    ),
                    "marginal": {
                        "mass_w1_gev": wasserstein_1d(mass_target, flat),
                        "mass_ks": ks_distance(mass_target, flat),
                        "model_mass_std_gev": float(np.std(flat)),
                    },
                    "energy_distance": energy_distance(permuted, mass_target),
                    "kernel_score": kernel_score(
                        permuted, mass_target, bandwidth=bandwidth
                    ),
                }
                print(
                    f"[{region}/{label}/shuffled] mass KS "
                    f"{control[label]['marginal']['mass_ks']:.5f}, "
                    f"energy distance {control[label]['energy_distance']['mean']:+.5f}, "
                    f"kernel score {control[label]['kernel_score']['mean']:+.4f}"
                )
            entry["shuffled_control"] = control
            entry["shuffled_comparison"] = {
                "marginal_mass_ks_gap_removed_by_shuffle": (
                    control["native"]["marginal"]["mass_ks"]
                    - entry["conditions"]["native"]["marginal"]["mass_ks"]
                ),
                "energy_distance_gap_removed_by_shuffle": (
                    control["native"]["energy_distance"]["mean"]
                    - entry["conditions"]["native"]["energy_distance"]["mean"]
                ),
                "kernel_score_gap_removed_by_shuffle": (
                    control["native"]["kernel_score"]["mean"]
                    - entry["conditions"]["native"]["kernel_score"]["mean"]
                ),
            }

        payload["regions"][region] = entry

    json_path = output_dir / "identity_vs_conditional.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
