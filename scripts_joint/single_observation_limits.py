#!/usr/bin/env python
"""Is the conditional spread learnable at all? The observation-count question.

This is the smallest experiment that decides whether a scoring-rule training term
can work in our setting, and it needs no data and no GPU.

The setup is the real one, stripped to one coordinate. Events have a truth-level
value ``z`` and a detector-level value ``x = M(z) + sigma * eps``. A model predicts
``mu(z)`` and ``sigma_hat(z)`` and is scored by a strictly proper rule on the
realised ``x``.

Two regimes are compared:

* **one observation per event** - what an unpaired training batch gives us: each
  ``z`` is seen once, with one ``x``. This is also what a paired simulation gives.
* **repeated observations per event** - two independent detector outcomes for the
  same ``z``, which is what the eINN-style quantile-calibration protocol needs and
  what our ppzee benchmark can supply.

For each regime the table reports the mean score at several ``sigma_hat``, with a
mean map rich enough to interpolate its own event. The outcome is the point of
the experiment: in the one-observation regime every proper rule is minimised at
``sigma_hat -> 0`` no matter how much capacity the mean map has, so no term of
this kind can make the spread learnable from unpaired data. Only repeated
observations break the degeneracy.

Nothing here trains or writes outside ``--output-dir``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

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

from scoring_rules import gaussian_log_score, median_bandwidth  # noqa: E402


def crps_gaussian(mean: torch.Tensor, scale: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """CRPS for a Gaussian, closed form; lower is better, finite at scale -> 0."""
    z = (target - mean) / scale
    normal = torch.distributions.Normal(
        torch.zeros((), dtype=mean.dtype), torch.ones((), dtype=mean.dtype)
    )
    phi = torch.exp(normal.log_prob(z))
    Phi = normal.cdf(z)
    return (scale * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / np.sqrt(np.pi))).mean()


def scale_scan(
    *,
    events: int,
    observations: int,
    truth_scale: float,
    seed: int,
    mean_is_free: bool,
) -> dict:
    """Mean score of each rule across ``sigma_hat``, for one regime.

    ``mean_is_free`` uses a mean map that can interpolate its own event
    (``mu = x_1 + noise``), which is the worst case for identification; otherwise
    the mean is the truth-level mean, i.e. a correctly specified mean map.
    """
    generator = torch.Generator().manual_seed(seed)
    z = torch.randn(events, 1, dtype=torch.float64, generator=generator)
    repeats = torch.stack(
        [torch.randn(events, 1, dtype=torch.float64, generator=generator) for _ in range(observations)],
        dim=0,
    )
    x = repeats  # the realisation(s) of this event
    if mean_is_free:
        # A map with one effective parameter per event can hit the first
        # observation exactly; this is the flexible-map limit.
        mean = x[0].clone()
    else:
        mean = torch.zeros_like(z)

    rows = []
    for scale_hat in (0.01, 0.1, 0.5, 1.0, 2.0, 4.0):
        scale = torch.full_like(mean, float(scale_hat))
        # Score against the realised first observation, as the project would.
        log_score = float(gaussian_log_score(mean, scale, x[0]))
        crps = float(crps_gaussian(mean, scale, x[0]))
        # Energy score needs draws of the *model*: build them from the mean and
        # the predicted scale, then score them by the realised value.
        draws = mean.unsqueeze(0) + scale.unsqueeze(0) * torch.randn(
            2, events, 1, dtype=torch.float64, generator=generator
        )
        energy = float(
            (2.0 * (draws[0] - x[0]).abs() - (draws[0] - draws[1]).abs()).mean()
        )
        rows.append(
            {
                "sigma_hat": float(scale_hat),
                "gaussian_log_score": log_score,
                "crps": crps,
                "energy_score": energy,
            }
        )
    return {
        "events": int(events),
        "observations_per_event": int(observations),
        "truth_scale": float(truth_scale),
        "mean_map": "interpolating" if mean_is_free else "correct",
        "scan": rows,
        "median_bandwidth_of_truth": float(median_bandwidth(x[0])),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=20000)
    parser.add_argument("--truth-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "cms_Joint" / "single_observation_limits",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = {
        "one_observation_interpolating_mean": dict(observations=1, mean_is_free=True),
        "one_observation_correct_mean": dict(observations=1, mean_is_free=False),
        "repeated_observations_interpolating_mean": dict(observations=2, mean_is_free=True),
        "repeated_observations_correct_mean": dict(observations=2, mean_is_free=False),
    }
    payload = {
        "schema_version": 1,
        "diagnostic": "conditional-spread identifiability versus observation count",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "truth_scale": args.truth_scale,
        "events": args.events,
        "results": {},
    }
    for label, kwargs in cases.items():
        result = scale_scan(
            events=args.events, truth_scale=args.truth_scale, seed=args.seed, **kwargs
        )
        payload["results"][label] = result
        best = {
            rule: min(result["scan"], key=lambda row: row[rule])["sigma_hat"]
            for rule in ("gaussian_log_score", "crps", "energy_score")
        }
        print(f"[{label}] argmin sigma_hat -> {best}", flush=True)

    json_path = output_dir / "single_observation_limits.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # ---- figure ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.3), sharex=True)
    rules = (
        ("gaussian_log_score", "Gaussian log score"),
        ("crps", "CRPS (bounded)"),
        ("energy_score", "energy score (2 model draws)"),
    )
    styles = {
        "one_observation_interpolating_mean": ("#c44e52", "-", "1 obs/event, flexible mean"),
        "one_observation_correct_mean": ("#c44e52", "--", "1 obs/event, correct mean"),
        "repeated_observations_interpolating_mean": ("#4c72b0", "-", "2 obs/event, flexible mean"),
        "repeated_observations_correct_mean": ("#4c72b0", "--", "2 obs/event, correct mean"),
    }
    for axis, (rule, title) in zip(axes, rules):
        for label, (color, style, legend) in styles.items():
            rows = payload["results"][label]["scan"]
            axis.plot(
                [r["sigma_hat"] for r in rows],
                [r[rule] for r in rows],
                color=color,
                ls=style,
                lw=1.8,
                marker="o",
                ms=3.5,
                label=legend,
            )
        axis.axvline(args.truth_scale, color="black", ls=":", lw=1.0)
        axis.set_xscale("log")
        axis.set_xlabel(r"predicted $\hat{\sigma}$")
        axis.set_title(title)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("mean score (lower is better)")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "A conditional-spread score cannot be minimised at the truth from one observation per event"
    )
    fig.tight_layout()
    png = output_dir / "single_observation_limits.png"
    fig.savefig(png, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {json_path}")
    print(f"wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
