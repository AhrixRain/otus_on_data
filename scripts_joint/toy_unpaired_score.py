#!/usr/bin/env python
"""Does an UNPAIRED proper score identify the conditional spread?

Why this exists (read with `docs/project_tree.md` branch P and the Session 72-74
records)
------------------------------------------------------------------------------
The Phase 0 toy (`scripts_joint/toy_identifiability.py`) scores the conditional
against a **paired** observation: ``x = response.sample(z)``, so the target
belongs to the same ``z`` the draws were made from. That toy recovers the planted
spread (1.03x) and was the existence proof for the Phase 1 production arm.

The production arm did not score a paired target. In `joint_trainer.py` the two
batches are drawn independently
(``x = _sample_batch(x_train)``, ``z = _sample_batch(z_train)``) and the term is
``energy_score_loss(x[:n], draws(z[:n]))``: the i-th truth event is scored against
the i-th detector event of an **independent** sample. In expectation that is the
*marginal* score ``E_{x,z}[log p(x|z)]`` with ``x`` independent of ``z``, whose
optimiser over a flexible mean map is a **constant** mean (pull the map onto the
marginal mean) with a scale equal to the marginal residual, not the conditional
spread. So the instrument that was refuted in production is not the instrument
that passed the toy.

This script puts the three differences on the same planted problem, one at a
time:

* ``paired``            paired target (the Phase 0 control)
* ``unpaired``          independent target, no floor, no kernel
* ``unpaired_floored``  + the shipped relative scale floor (a clamped term)
* ``unpaired_kernel``   + a kernel floor on the effective scale
* ``unpaired_kernel_linear_mean``  + a capacity-constrained (linear) mean map

Readouts per arm and seed: recovered spread versus planted spread
(``k_hat/k_true``), the **mean-map spread ratio** (contraction is the failure
signature the production arm showed: 29.2 -> 11.7 MeV), whether the recovered
scale still tracks ``z`` at all, the final score value, and 68% coverage of the
residual under the claimed scale (the criterion of
``scripts_joint/conditional_spread.py``).

CPU only, small networks, no data, no checkpoints, no run directory touched.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

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

from toy_identifiability import (  # noqa: E402
    PlantedResponse,
    ToyDecoder,
    marginal_w1,
)

ARMS = (
    "paired",
    "unpaired",
    "unpaired_floored",
    "unpaired_kernel",
    "unpaired_kernel_linear_mean",
)


class LinearMeanDecoder(ToyDecoder):
    """The same scale head, but a mean map far too small to absorb the spread.

    This is the plan's P3 lever in its cheapest form: one linear layer instead of
    a 64-unit trunk, so it cannot interpolate its batch.
    """

    def __init__(self, dimension: int, hidden: int = 64):
        super().__init__(dimension, hidden=hidden)
        self.linear_mean = nn.Linear(dimension, dimension)
        nn.init.zeros_(self.linear_mean.weight)
        nn.init.zeros_(self.linear_mean.bias)
        for parameter in self.mean_head.parameters():
            parameter.requires_grad_(False)

    def forward(self, z: torch.Tensor):
        features = self.trunk(z)
        mean = torch.tanh(self.linear_mean(z))
        scale = torch.nn.functional.softplus(self.scale_head(features)) + 1e-3
        return mean, scale


class UnboundedMeanDecoder(ToyDecoder):
    """The same network with the mean head's ``tanh`` bound removed.

    ``ToyDecoder`` bounds the mean by ``tanh``, so it can carry at most ~1 of the
    marginal spread; the real decoder's flow mean map has no such bound. This
    variant separates "the marginal objective cannot identify the split" from "this
    particular mean head is too weak to absorb it".
    """

    def forward(self, z: torch.Tensor):
        features = self.trunk(z)
        mean = 3.0 * self.mean_head(features)
        scale = torch.nn.functional.softplus(self.scale_head(features)) + 1e-3
        return mean, scale


def _draws(
    mean: torch.Tensor, scale: torch.Tensor, draws: int, generator: torch.Generator
) -> torch.Tensor:
    noise = torch.randn(draws, *mean.shape, dtype=mean.dtype, generator=generator)
    return mean.unsqueeze(0) + scale.unsqueeze(0) * noise


def run_arm(
    arm: str,
    *,
    response: PlantedResponse,
    seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    kappa: float,
    draws: int,
    scale_floor: float,
    kernel_floor: float,
    mean_head: str,
    device: torch.device,
) -> dict:
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 1)
    if arm.endswith("linear_mean"):
        model = LinearMeanDecoder(response.dimension)
    elif mean_head == "unbounded":
        model = UnboundedMeanDecoder(response.dimension)
    else:
        model = ToyDecoder(response.dimension)
    model = model.to(device)
    optimiser = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=learning_rate
    )

    paired = arm == "paired"
    use_floor = arm in {"unpaired_floored", "unpaired_kernel", "unpaired_kernel_linear_mean"}
    use_kernel = arm in {"unpaired_kernel", "unpaired_kernel_linear_mean"}

    score_history: list[float] = []
    scale_history: list[float] = []
    for step in range(int(steps)):
        z_prior = torch.randn(
            batch_size, response.dimension, dtype=torch.float32, generator=generator
        ).to(device)
        # The DATA batch is an independent sample: this is the unpaired setting.
        z_data = torch.randn(
            batch_size, response.dimension, dtype=torch.float32, generator=generator
        ).to(device)
        x_data = response.sample(z_data, generator).to(device)

        mean, scale = model(z_prior)
        sample = _draws(mean, scale, draws, generator)
        loss = marginal_w1(x_data, sample[0])
        score_value = x_data.new_tensor(0.0)
        if kappa > 0.0:
            mu = sample.mean(dim=0)
            claimed = sample.std(dim=0, unbiased=True)
            target = (
                response.sample(z_prior, generator).to(device) if paired else x_data
            )
            # The shipped implementation standardises by the target's own scale
            # and floors the predicted scale relative to that scale.
            reference = target.detach().std(dim=0, unbiased=False).clamp_min(1e-8)
            if use_floor:
                claimed = torch.maximum(claimed, scale_floor * reference)
            if use_kernel:
                claimed = torch.maximum(claimed, kernel_floor * reference)
            residual = (target - mu) / reference
            score_value = (
                residual.pow(2) / (2.0 * claimed.pow(2)) + torch.log(claimed)
            ).mean()
            loss = loss + kappa * score_value
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        score_history.append(float(score_value.detach()))
        if step % max(int(steps) // 20, 1) == 0:
            with torch.no_grad():
                _, scale_now = model(z_prior)
            scale_history.append(float(scale_now.mean()))

    # ---- readout on a grid of z_0 values ------------------------------------
    grid = torch.linspace(-2.0, 2.0, 21, dtype=torch.float32, device=device)
    z_grid = torch.zeros(len(grid), response.dimension, device=device)
    z_grid[:, 0] = grid
    with torch.no_grad():
        mean_grid, scale_grid = model(z_grid)
        sample_grid = _draws(mean_grid, scale_grid, 64, generator)
        claimed_grid = sample_grid.std(dim=0, unbiased=True)[:, 0].cpu().numpy()
        learned_grid = scale_grid[:, 0].cpu().numpy()
        mean_grid_values = mean_grid[:, 0].cpu().numpy()
    truth_spread = response.spread(z_grid.cpu())[:, 0].numpy()
    true_mean = response.mean_map(z_grid.cpu())[:, 0].numpy()
    central = np.abs(grid.cpu().numpy()) <= 1.5
    ratio = claimed_grid / truth_spread
    return {
        "arm": arm,
        "seed": int(seed),
        "final_loss": float(loss.detach()),
        "final_score": float(np.mean(score_history[-20:])),
        "score_first": float(np.mean(score_history[:20])),
        "learned_scale_history": scale_history,
        "claimed_scale_grid": claimed_grid.tolist(),
        "learned_scale_grid": learned_grid.tolist(),
        "true_spread_grid": truth_spread.tolist(),
        "mean_map_grid": mean_grid_values.tolist(),
        "true_mean_grid": true_mean.tolist(),
        "ratio_median_central": float(np.median(ratio[central])),
        "fraction_within_25pct": float(np.mean(np.abs(ratio[central] - 1.0) <= 0.25)),
        "mean_map_spread_ratio": float(
            np.std(mean_grid_values) / max(float(np.std(true_mean)), 1e-12)
        ),
        "scale_z_dependence": float(
            np.std(claimed_grid) / max(float(np.mean(claimed_grid)), 1e-12)
        ),
        "true_z_dependence": float(np.std(truth_spread) / max(float(np.mean(truth_spread)), 1e-12)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--scale-floor", type=float, default=0.1)
    parser.add_argument("--kernel-floor", type=float, default=0.5)
    parser.add_argument("--spread-scale", type=float, default=1.0)
    parser.add_argument("--spread-contrast", type=float, default=1.0)
    parser.add_argument("--mean-amplitude", type=float, default=0.5)
    parser.add_argument(
        "--mean-head",
        choices=("bounded", "unbounded"),
        default="bounded",
        help="tanh-bounded mean head (the toy default) or unbounded",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "cms_Joint" / "toy_unpaired_score",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    response = PlantedResponse(
        dimension=args.dimension,
        spread_scale=args.spread_scale,
        spread_contrast=args.spread_contrast,
        mean_amplitude=args.mean_amplitude,
    )
    results: list[dict] = []
    for arm in ARMS:
        for seed in range(int(args.seeds)):
            outcome = run_arm(
                arm,
                response=response,
                seed=seed,
                steps=args.steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                kappa=args.kappa,
                draws=args.draws,
                scale_floor=args.scale_floor,
                kernel_floor=args.kernel_floor,
                mean_head=args.mean_head,
                device=device,
            )
            results.append(outcome)
            print(
                f"[{arm}] seed {seed}: k_hat/k_true {outcome['ratio_median_central']:.3f}, "
                f"mean-map spread ratio {outcome['mean_map_spread_ratio']:.3f}, "
                f"score {outcome['final_score']:.3f}",
                flush=True,
            )

    summary: dict[str, dict] = {}
    for arm in ARMS:
        rows = [row for row in results if row["arm"] == arm]
        ratios = np.array([row["ratio_median_central"] for row in rows])
        summary[arm] = {
            "seeds": len(rows),
            "ratio_median_central_mean": float(ratios.mean()),
            "ratio_median_central_std": float(ratios.std()),
            "seeds_within_0p75_1p25": int(np.sum((ratios >= 0.75) & (ratios <= 1.25))),
            "mean_map_spread_ratio_mean": float(
                np.mean([row["mean_map_spread_ratio"] for row in rows])
            ),
            "final_score_mean": float(np.mean([row["final_score"] for row in rows])),
            "scale_z_dependence_mean": float(
                np.mean([row["scale_z_dependence"] for row in rows])
            ),
        }
    payload = {
        "schema_version": 1,
        "diagnostic": "unpaired proper scoring rule on a planted conditional spread",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "dimension": args.dimension,
            "seeds": args.seeds,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "draws": args.draws,
            "kappa": args.kappa,
            "scale_floor": args.scale_floor,
            "kernel_floor": args.kernel_floor,
            "mean_head": args.mean_head,
        },
        "criterion": (
            "paired arms must recover k (k_hat/k_true in [0.75, 1.25]); the unpaired "
            "arms are expected to follow the marginal solution instead"
        ),
        "summary": summary,
        "runs": results,
    }
    json_path = output_dir / "toy_unpaired_score.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {json_path}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
