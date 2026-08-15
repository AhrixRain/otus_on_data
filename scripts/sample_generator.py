#!/usr/bin/env python
"""Generator (z -> x) quality check with well-populated sampling.

The standard eval.py simulation comparison decodes z_test once per event;
for the v3.9 pilot z_test is only 600 events, so the pred mass histogram is
statistically thin. This script instead decodes the full filtered prior
(5,995 events for v3.9) K times each with fresh decoder noise, and compares
the resulting mass distribution against x_test using the same residual
metrics as eval.py (W1/KS/shape stats).

Usage:
  python scripts/sample_generator.py \
      --config configs/archive/cms_JpsiDoubleMuons_v3.9_F1_restricted.yaml \
      --checkpoint outputs/.../best_model.pt --device mps --draws 20 \
      --out /tmp/gen_check
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--draws", type=int, default=20)
    parser.add_argument("--num-z", type=int, default=None,
                        help="Cap on prior events decoded (default: all).")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import torch

    from scripts.cms_data import (
        filter_theory_prior,
        load_and_split_cached,
        load_config,
        load_theory_prior_z,
        resolve_config,
    )
    from scripts.cms_model import load_model_from_checkpoint
    from scripts.eval import decode_in_batches, write_mass_comparison

    config = resolve_config(load_config(args.config))
    device = torch.device(
        "mps" if args.device == "auto" and torch.backends.mps.is_available() else args.device
    )
    model, model_config, _stats, checkpoint = load_model_from_checkpoint(
        args.checkpoint, config, map_location=torch.device("cpu")
    )
    model.to(device)
    print("checkpoint epoch:", checkpoint.get("epoch"), "| eval loss:", checkpoint.get("eval_loss"))

    z_prior = filter_theory_prior(
        load_theory_prior_z(Path(model_config["paths"]["theory_prior_file"])),
        model_config.get("theory_prior_selection"),
    )
    if args.num_z:
        z_prior = z_prior[: args.num_z]
    print("prior events:", len(z_prior), "| draws:", args.draws)

    # Decode all draws of one event in one batch: [N, K, 8] -> [N*K, 8].
    z_batch = np.repeat(z_prior, args.draws, axis=0)
    rng = np.random.default_rng(12345)
    z_batch = z_batch[rng.permutation(len(z_batch))]
    decoded = decode_in_batches(model, z_batch, device, batch_size=8192)

    arrays, cache_info = load_and_split_cached(
        model_config, num_samples=None, cache_dir=None, use_cache=True
    )
    x_test = arrays["x_test"]
    print("x_test:", len(x_test), "| cache:", cache_info.get("status"))

    out_dir = args.out or (args.checkpoint.parent / "generator_check")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics, metric_arrays, truth_mass, pred_mass = write_mass_comparison(
        out_dir,
        "simulation",
        x_test,
        decoded,
        bins=90,
        mass_range=(2.6, 3.5),
        min_truth_count=20,
        labels=("x_test CMS", f"D(prior, K={args.draws})"),
        write_outputs=True,
        daughter_masses=model_config["model"].get("daughter_masses"),
        stable=True,
    )
    metrics["checkpoint"] = str(args.checkpoint)
    metrics["num_prior_events"] = int(len(z_prior))
    metrics["num_draws"] = int(args.draws)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print("W1 %.4f | KS %.4f" % (metrics["w1_distance"], metrics["ks_statistic"]))
    print("truth mean/std %.4f/%.4f | pred mean/std %.4f/%.4f" % (
        truth_mass.mean(), truth_mass.std(), pred_mass.mean(), pred_mass.std()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
