#!/usr/bin/env python
"""Measure the irreducible finite-sample floor of the joint distributional losses.

Why this exists. The 2026-09-10 flat-loss audit showed that the dominant joint
loss terms are W1/sliced-Wasserstein distances between two finite samples, so
their minimum achievable value is not zero: it is the distance between two
independent draws of the same target distribution. At Run E's real per-update
cardinalities that floor is ~0.34-0.37 for J/psi (the prior stream had only 417
events per update against 16,858 CMS events), so ~75% of the logged train loss
could not descend no matter what the model did.

This probe reports, for one config:

* the natural per-update batch size of every (region, stream),
* the effective size after ``loaders.min_events_per_update``,
* the expected ``x_sim_loss`` / ``z_prior_loss`` between two independent draws
  of the same target at those cardinalities.

Run it before a training run to know how much of the loss is learnable.

Examples::

    python scripts_joint/loss_floor_probe.py --config configs_joint/cms_Joint_runE.yaml
    python scripts_joint/loss_floor_probe.py --run F --output outputs/cms_Joint/loss_floor_probe/F.json
"""

from __future__ import annotations

import argparse
import json
import sys
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
from joint_data import load_joint_regions, resolve_joint_config  # noqa: E402
from joint_train_utils import _full_pass_batches, _full_pass_step_count  # noqa: E402
from run_joint import build_loss_factories  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", default=None, help="Short run id, e.g. E or F.")
    group.add_argument("--config", type=Path, default=None)
    parser.add_argument("--reps", type=int, default=8, help="Independent floor draws.")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def resolve_config_path(args: argparse.Namespace) -> Path:
    if args.config is not None:
        return args.config
    candidates = {
        path.stem[len("cms_Joint_run") :]: path
        for path in (REPO_ROOT / "configs_joint").glob("cms_Joint_run*.yaml")
    }
    if args.run not in candidates:
        raise SystemExit(f"Unknown run {args.run!r}; available: {sorted(candidates)}")
    return candidates[args.run]


def _disjoint(a: np.ndarray, n_first: int, n_second: int, rng) -> tuple[torch.Tensor, torch.Tensor]:
    idx = rng.choice(len(a), size=n_first + n_second, replace=False)
    tensor = torch.as_tensor(np.ascontiguousarray(a[idx]), dtype=torch.float32)
    return tensor[:n_first], tensor[n_first:]


def _floor(
    factory,
    kind: str,
    x_pool: np.ndarray,
    z_pool: np.ndarray,
    n_first: int,
    n_second: int,
    reps: int,
    seed: int,
):
    """Expected loss between two independent draws of the same target.

    ``kind='x'`` draws both sides from the CMS x distribution and scores
    ``x_sim_loss``; ``kind='z'`` draws both sides from the prior z distribution
    and scores ``z_prior_loss``.
    """
    rng = np.random.default_rng(seed)
    pool = x_pool if kind == "x" else z_pool
    values = []
    for _ in range(reps):
        first, second = _disjoint(pool, n_first, n_second, rng)
        with torch.no_grad():
            if kind == "x":
                values.append(float(factory.x_sim_loss(first, second)))
            else:
                values.append(float(factory.z_prior_loss(first, second)))
    return float(np.mean(values)), float(np.std(values))


def main() -> int:
    args = parse_args()
    config_path = resolve_config_path(args)
    config = resolve_joint_config(load_config(config_path))
    region_arrays, *_ = load_joint_regions(config, num_samples=None, use_cache=True)

    loaders = config["loaders"]
    batch_size = int(loaders.get("train_batch_size", 1024))
    min_events = int(loaders.get("min_events_per_update", 0) or 0)
    epoch_definition = str(loaders.get("epoch_definition", "fixed_steps")).lower()
    sampling = str(loaders.get("sampling", "random")).lower()

    factories = build_loss_factories(config, region_arrays)

    # Floor draws come from the pooled train+val splits of the target
    # distribution, sampled disjointly so the two sides are independent.
    pools = {
        region: {
            "x": np.concatenate(
                [region_arrays[region]["x_train"], region_arrays[region]["x_val"]], axis=0
            ),
            "z": np.concatenate(
                [region_arrays[region]["z_train"], region_arrays[region]["z_val"]], axis=0
            ),
        }
        for region in config["region_order"]
    }

    if sampling == "cycling_without_replacement" and epoch_definition == "full_pass":
        steps = _full_pass_step_count(region_arrays, list(config["region_order"]), batch_size)
    else:
        steps = None

    report: dict = {
        "config": str(config_path),
        "run_name": config.get("run_name"),
        "batch_size": batch_size,
        "min_events_per_update": min_events,
        "epoch_definition": epoch_definition,
        "sampling": sampling,
        "optimizer_updates_per_epoch": steps,
        "reps": int(args.reps),
        "regions": {},
    }

    total_floor = 0.0
    total_weight = 0.0
    region_weights = config.get("region_weights", {})
    weight_total = sum(float(region_weights.get(name, 1.0)) for name in config["region_order"])

    print(f"{config.get('run_label', config.get('run_name'))}: {config_path}")
    print(f"  min_events_per_update={min_events}  updates/epoch={steps}")
    print(f"  {'region':6s} {'stream':4s} {'natural':>8s} {'effective':>9s} "
          f"{'L_floor':>9s} {'sd':>7s}")
    for region in config["region_order"]:
        arrays = region_arrays[region]
        if steps is None:
            natural_x = natural_z = batch_size
        else:
            natural_x = len(_full_pass_batches(arrays["x_train"], steps, 1, seed=0, stream=0)[0])
            natural_z = len(_full_pass_batches(arrays["z_train"], steps, 1, seed=0, stream=1)[0])
        effective_x = max(natural_x, min_events)
        effective_z = max(natural_z, min_events)

        factory = factories[region]
        factory.set_num_slices(int(config["stages"][0].get("num_slices", 256)))

        # L_x compares x_true (effective_x) with D(z) built from effective_z
        # prior events, so its floor is two independent x-draws at those sizes.
        # L_z is the mirror image on the prior distribution.
        lx, lx_sd = _floor(
            factory,
            "x",
            pools[region]["x"],
            pools[region]["z"],
            effective_x,
            effective_z,
            args.reps,
            args.seed + 1,
        )
        lz, lz_sd = _floor(
            factory,
            "z",
            pools[region]["x"],
            pools[region]["z"],
            effective_z,
            effective_x,
            args.reps,
            args.seed + 2,
        )
        report["regions"][region] = {
            "x_natural_per_update": int(natural_x),
            "z_natural_per_update": int(natural_z),
            "x_effective_per_update": int(effective_x),
            "z_effective_per_update": int(effective_z),
            "L_x_floor": lx,
            "L_x_floor_sd": lx_sd,
            "L_z_floor": lz,
            "L_z_floor_sd": lz_sd,
        }
        weight = float(region_weights.get(region, 1.0)) / weight_total
        total_floor += weight * (lx + lz)
        total_weight += weight
        print(f"  {region:6s} {'x':4s} {natural_x:8d} {effective_x:9d} {lx:9.4f} {lx_sd:7.4f}")
        print(f"  {region:6s} {'z':4s} {natural_z:8d} {effective_z:9d} {lz:9.4f} {lz_sd:7.4f}")

    report["weighted_sum_of_distributional_floors"] = total_floor
    print(f"\n  weighted (region-average) distributional floor: {total_floor:.4f} "
          "(lamb=tau=1 reference; scale by the stage coefficients)")

    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"  wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
