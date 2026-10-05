#!/usr/bin/env python
"""Toy identifiability study: can the conditional spread be learned at all?

The question this answers, in hours instead of GPU-days: given a known
``x = M(z) + k(z) * eps``, does a marginal-matching objective recover ``k``,
and does adding the energy score recover it? Everything here is small enough to
run on a CPU.

Design
------
* ``z ~ N(0, 1)`` in ``dimension`` dimensions (1 by default).
* planted mean ``M(z)``: a fixed smooth non-linear map.
* planted spread ``k(z)``: ``k0 * exp(k1 * tanh(z_0))``, so the true conditional
  spread varies across the domain and the recovery can be checked pointwise.
* data: ``x = M(z) + k(z) * eps`` with ``eps ~ N(0, 1)``.
* model: a small MLP reading ``z`` and emitting a mean and a positive scale per
  dimension; it has exactly the architecture family the real decoder uses
  (bounded mean + learned positive scale), reduced to its essentials.

Arms (identical except for the item under test)
-----------------------------------------------
``free_mean_marginal``  the small flexible network learns the mean map and the
                        scale, trained by paired reconstruction MSE plus a
                        marginal W1 to the data - the shape of the objective the
                        real project trains with
``free_mean_energy``    the same, plus ``kappa * gaussian_log_score(mu, sigma, x)``
``fixed_mean_energy``   the mean map is fixed to the planted truth and only the
                        scale is learned, with the energy score term

The third arm is the reference. It separates two hypotheses that are usually
conflated: "the loss cannot see the spread" and "the mean map makes the spread
unidentifiable". Measured so far: the marginal loss alone collapses the scale to
0.31x truth; the log score recovers it (1.04x) even with a flexible mean map,
because this network is far too small to interpolate 512 events; and a fixed,
correctly specified mean recovers it too (1.01x). The degeneracy therefore
depends on the mean map's capacity *relative to the number of events*, not on the
loss alone - and the real model, with 830k parameters against 3.0M and 4.2M
training events, may or may not be in the degenerate regime. That is why the
production arm's first checkpoint is a decision point rather than a formality:
measure whether the channel amplitude rises or collapses before spending the
schedule. ``scripts_joint/single_observation_limits.py`` measures the degenerate
limit directly (a mean that hits its own observation).

Pre-declared readout
--------------------
For each arm and seed: the recovered spread profile ``k_hat(z)`` on a grid,
reported as the median ratio ``k_hat / k_true`` over the central domain and the
fraction of the grid within +/-25%. Success: the arms carrying the log score
recover the spread (median ratio in [0.75, 1.25] on at least 4 of 5 seeds) while
the marginal-only arm collapses toward the floor.

Nothing here touches the joint model, the data or any run directory. Outputs go
to a new ``--output-dir``.
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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

from scoring_rules import gaussian_log_score  # noqa: E402


# ---------------------------------------------------------------------------
# the planted problem
# ---------------------------------------------------------------------------

class PlantedResponse:
    """``x = M(z) + k(z) * eps`` with a known, non-constant spread."""

    def __init__(
        self,
        *,
        dimension: int,
        spread_scale: float = 1.0,
        spread_contrast: float = 1.0,
        mean_amplitude: float = 0.5,
    ):
        self.dimension = int(dimension)
        self.spread_scale = float(spread_scale)
        self.spread_contrast = float(spread_contrast)
        self.mean_amplitude = float(mean_amplitude)

    def mean_map(self, z: torch.Tensor) -> torch.Tensor:
        first = z[:, :1]
        base = self.mean_amplitude * torch.sin(first)
        if self.dimension == 1:
            return base
        rest = 0.3 * self.mean_amplitude * torch.tanh(z[:, 1:])
        return torch.cat([base, rest], dim=1)

    def spread(self, z: torch.Tensor) -> torch.Tensor:
        profile = self.spread_scale * torch.exp(
            self.spread_contrast * torch.tanh(z[:, :1])
        )
        if self.dimension == 1:
            return profile
        return profile.expand(-1, self.dimension) * 0.7

    def sample(self, z: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        noise = torch.randn(z.shape, dtype=z.dtype, generator=generator)
        return self.mean_map(z) + self.spread(z) * noise


# ---------------------------------------------------------------------------
# the model: bounded mean + learned positive scale, the real decoder's essence
# ---------------------------------------------------------------------------

class ToyDecoder(nn.Module):
    def __init__(self, dimension: int, hidden: int = 64):
        super().__init__()
        self.dimension = int(dimension)
        self.trunk = nn.Sequential(
            nn.Linear(dimension, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.mean_head = nn.Linear(hidden, dimension)
        self.scale_head = nn.Linear(hidden, dimension)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.scale_head.weight)
        nn.init.constant_(self.scale_head.bias, -1.0)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(z)
        mean = torch.tanh(self.mean_head(features))
        scale = torch.nn.functional.softplus(self.scale_head(features)) + 1e-3
        return mean, scale

    def sample(
        self, z: torch.Tensor, draws: int, generator: torch.Generator
    ) -> torch.Tensor:
        mean, scale = self(z)
        noise = torch.randn(
            draws, *z.shape, dtype=z.dtype, generator=generator
        )
        return mean.unsqueeze(0) + scale.unsqueeze(0) * noise


# ---------------------------------------------------------------------------
# objectives
# ---------------------------------------------------------------------------

def marginal_w1(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    """Mean over dimensions of the 1-D Wasserstein distance between samples."""
    total = 0.0
    for column in range(real.shape[1]):
        a = torch.sort(real[:, column]).values
        b = torch.sort(fake[:, column]).values
        step = 1.0 / max(len(a), len(b))
        grid = torch.linspace(0.0, 1.0, max(len(a), len(b)), dtype=real.dtype)
        total = total + torch.trapezoid(
            torch.abs(
                torch.quantile(a, grid, interpolation="linear")
                - torch.quantile(b, grid, interpolation="linear")
            ),
            grid,
        )
    return total / real.shape[1]


def run_arm(
    arm: str,
    *,
    response: PlantedResponse,
    seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    kappa: float,
    device: torch.device,
) -> dict:
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 1)
    model = ToyDecoder(response.dimension).to(device)
    fixed_mean = arm == "fixed_mean_energy"
    if fixed_mean:
        # Freeze the mean head so the only free parameters are the scales.
        for parameter in model.mean_head.parameters():
            parameter.requires_grad_(False)
        trainable = [p for p in model.parameters() if p.requires_grad]
    else:
        trainable = list(model.parameters())
    optimiser = torch.optim.Adam(trainable, lr=learning_rate)
    history: list[float] = []

    for step in range(int(steps)):
        z = torch.randn(batch_size, response.dimension, dtype=torch.float32, generator=generator)
        z = z.to(device)
        x = response.sample(z, generator).to(device)
        if fixed_mean:
            mean = response.mean_map(z)
        else:
            mean, _ = model(z)
        loss = (mean - x).pow(2).mean()
        if arm != "energy_only":
            loss = loss + marginal_w1(x, mean)
        if arm in {"free_mean_energy", "fixed_mean_energy", "energy_only"}:
            # Strictly proper score of the conditional: mean from the map, scale
            # from the learned head. No draws needed.
            _, scale = model(z)
            loss = loss + kappa * gaussian_log_score(mean, scale, x)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        if step % max(int(steps) // 20, 1) == 0 or step == int(steps) - 1:
            history.append(float(loss.detach()))

    # Recovered spread profile on a grid of z_0 values, by Monte Carlo.
    grid = torch.linspace(-2.0, 2.0, 21, dtype=torch.float32, device=device)
    z_grid = torch.zeros(len(grid), response.dimension, device=device)
    z_grid[:, 0] = grid
    with torch.no_grad():
        _, scale = model(z_grid)
        recovered = scale[:, 0].cpu().numpy()
    truth = response.spread(z_grid.cpu())[:, 0].numpy()
    ratio = recovered / truth
    central = np.abs(grid.cpu().numpy()) <= 1.5
    return {
        "arm": arm,
        "seed": int(seed),
        "final_loss": history[-1],
        "recovered_spread": recovered.tolist(),
        "true_spread": truth.tolist(),
        "ratio_median_central": float(np.median(ratio[central])),
        "ratio_median_all": float(np.median(ratio)),
        "fraction_within_25pct": float(np.mean(np.abs(ratio - 1.0) <= 0.25)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--spread-scale", type=float, default=1.0)
    parser.add_argument("--spread-contrast", type=float, default=1.0)
    parser.add_argument("--mean-amplitude", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "cms_Joint" / "toy_identifiability",
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

    arms = ("free_mean_marginal", "free_mean_energy", "fixed_mean_energy")
    results: list[dict] = []
    for arm in arms:
        for seed in range(int(args.seeds)):
            record = run_arm(
                arm,
                response=response,
                seed=1000 + seed,
                steps=args.steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                kappa=args.kappa,
                device=device,
            )
            results.append(record)
            print(
                f"[{arm:12s} seed {seed}] median ratio (central) "
                f"{record['ratio_median_central']:+.3f} | within 25%: "
                f"{record['fraction_within_25pct']:.2f}",
                flush=True,
            )

    summary: dict[str, dict] = {}
    for arm in arms:
        rows = [r for r in results if r["arm"] == arm]
        ratios = np.array([r["ratio_median_central"] for r in rows])
        summary[arm] = {
            "seeds": len(rows),
            "ratio_median_central_mean": float(ratios.mean()),
            "ratio_median_central_std": float(ratios.std(ddof=0)),
            "seeds_within_0p75_1p25": int(np.sum((ratios >= 0.75) & (ratios <= 1.25))),
            "seeds_below_0p20": int(np.sum(ratios < 0.20)),
        }

    payload = {
        "schema_version": 1,
        "diagnostic": "toy identifiability of the conditional spread",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "dimension": args.dimension,
            "seeds": args.seeds,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "kappa": args.kappa,
            "spread_scale": args.spread_scale,
            "spread_contrast": args.spread_contrast,
            "mean_amplitude": args.mean_amplitude,
        },
        "criterion": {
            "positive": ">=4/5 seeds with median k_hat/k_true in [0.75, 1.25]",
            "negative": ">=4/5 seeds with median k_hat/k_true < 0.20",
        },
        "summary": summary,
        "runs": results,
    }
    json_path = output_dir / "toy_identifiability.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # ---- figure --------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    colors = {
        "free_mean_marginal": "#c44e52",
        "free_mean_energy": "#4c72b0",
        "fixed_mean_energy": "#55a868",
    }
    labels = {
        "free_mean_marginal": "flexible mean, marginal loss",
        "free_mean_energy": "flexible mean, + energy score",
        "fixed_mean_energy": "fixed mean, + energy score",
    }
    grid = results[0]["true_spread"]
    x_axis = np.linspace(-2.0, 2.0, len(grid))
    axes[0].plot(x_axis, grid, color="black", lw=2.2, label="planted $k(z)$")
    for arm in arms:
        rows = [r for r in results if r["arm"] == arm]
        curves = np.array([r["recovered_spread"] for r in rows])
        axes[0].plot(
            x_axis, curves.mean(axis=0), color=colors[arm], lw=1.8, label=labels[arm]
        )
        axes[0].fill_between(
            x_axis,
            curves.mean(axis=0) - curves.std(axis=0),
            curves.mean(axis=0) + curves.std(axis=0),
            color=colors[arm],
            alpha=0.15,
        )
    axes[0].set_xlabel(r"$z_0$")
    axes[0].set_ylabel(r"conditional spread $k(z)$")
    axes[0].set_title("Recovered versus planted spread")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.2)

    for arm in arms:
        ratios = [r["ratio_median_central"] for r in results if r["arm"] == arm]
        axes[1].scatter(
            [arm] * len(ratios), ratios, color=colors[arm], s=42, zorder=3
        )
    axes[1].axhline(1.0, color="black", ls="--", lw=1.0)
    axes[1].axhspan(0.75, 1.25, color="black", alpha=0.08)
    axes[1].axhline(0.2, color="#c44e52", ls=":", lw=1.2)
    axes[1].set_ylabel(r"median $\hat{k}/k$ over the central domain")
    axes[1].set_title("Per-seed recovery")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    png = output_dir / "toy_identifiability.png"
    fig.savefig(png, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {json_path}")
    print(f"wrote {png}")
    for arm, stats in summary.items():
        print(
            f"  {arm:12s} median ratio {stats['ratio_median_central_mean']:+.3f} "
            f"+- {stats['ratio_median_central_std']:.3f} | in band: "
            f"{stats['seeds_within_0p75_1p25']}/{stats['seeds']} | collapsed: "
            f"{stats['seeds_below_0p20']}/{stats['seeds']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
