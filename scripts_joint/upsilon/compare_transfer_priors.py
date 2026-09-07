#!/usr/bin/env python
"""Compare frozen Upsilon transfers with identity/floor references by pair pT.

Bins are fixed analysis choices, applied to each distribution's OWN pair pT.
All prior arms and their identity maps use the same event count and CMS draw
within a bin. Floors average 20 disjoint CMS split comparisons at that count.
These are conditional marginal tests, not per-event response closure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate_z_to_x import (
    DEFAULT_CMS, file_sha256, load_cms_p4, load_decoded,
    observable_metric, physics_observables,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from identity_reference import gauge

PT_BINS = [(0, 5), (5, 10), (10, 20), (20, 40), (40, None)]


def select_observables(values, low=0, high=None):
    obs = physics_observables(values)
    keep = (obs["mass"] >= 8.5) & (obs["mass"] <= 11.5)
    keep &= obs["pair_pt"] >= low
    if high is not None:
        keep &= obs["pair_pt"] < high
    return {k: v[keep] for k, v in obs.items() if k in ("mass", "pair_pt")}


def score_cell(cms, arms, seed, min_events=100):
    counts = {name: {kind: len(obs["mass"]) for kind, obs in arm.items()}
              for name, arm in arms.items()}
    n = min([len(cms["mass"]) // 2] +
            [count for arm in counts.values() for count in arm.values()])
    result = {"cms_events": len(cms["mass"]), "available_events": counts,
              "events_per_distribution": n}
    if n < min_events:
        return {**result, "status": "insufficient_events"}
    rng = np.random.default_rng(seed)
    reference_idx = rng.choice(len(cms["mass"]), n, replace=False)
    floors = {key: {metric: [] for metric in ("ks", "w1")} for key in cms}
    for _ in range(20):
        idx = rng.choice(len(cms["mass"]), 2 * n, replace=False)
        for key, values in cms.items():
            metrics = observable_metric(values[idx[:n]], values[idx[n:]])
            for metric in ("ks", "w1"):
                floors[key][metric].append(metrics[metric])
    result.update(status="scored", arms={})
    for name, arm in arms.items():
        indices = {kind: rng.choice(len(obs["mass"]), n, replace=False)
                   for kind, obs in arm.items()}
        scores = {}
        for key, values in cms.items():
            reference = values[reference_idx]
            baseline = observable_metric(reference, arm["identity"][key][indices["identity"]])
            model = observable_metric(reference, arm["decoded"][key][indices["decoded"]])
            scores[key] = {}
            for metric in ("ks", "w1"):
                floor = float(np.mean(floors[key][metric]))
                gauged = gauge(model[metric], baseline[metric], floor)
                scores[key][metric] = {
                    "model": model[metric], "identity": baseline[metric], "floor": floor,
                    "vs_identity": float(gauged) if np.isfinite(gauged) else None,
                    "resolvable": baseline[metric] > floor,
                }
        result["arms"][name] = scores
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-decoded", type=Path, required=True)
    parser.add_argument("--unified-decoded", type=Path, required=True)
    parser.add_argument("--cms", type=Path, default=DEFAULT_CMS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "transfer_comparison.json"
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(report_path)
    cms = load_cms_p4(args.cms)
    arms, sources = {}, {}
    for name, path in [("legacy", args.legacy_decoded), ("unified", args.unified_decoded)]:
        z, x, _, _, provenance = load_decoded(path)
        arms[name] = {"identity": z, "decoded": x}
        sources[name] = {"path": str(path.resolve()), "sha256": file_sha256(path),
                         "provenance": provenance}
    hashes = {source["provenance"].get("checkpoint_sha256") for source in sources.values()}
    if len(hashes) != 1 or None in hashes:
        raise ValueError("Prior arms must use the same frozen checkpoint")
    report = {
        "evaluation_scope": "post_unblinding_heldout_transfer",
        "sources": sources, "cms_sha256": file_sha256(args.cms), "seed": args.seed,
        "method": __doc__, "mass_window_gev": [8.5, 11.5],
        "mass_convention": "stable invariant mass with physical muon masses, as evaluate_z_to_x.py",
        "interpretation": "Conditional marginal comparison; no response inversion claim. "
            "vs_identity=1 is no improvement, 0 is the estimated finite-sample floor; "
            "null means the identity is already at or below the floor. "
            "Mixture is declared from the old prior, not fitted to these results.",
        "cells": [],
    }
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for index, (ax, (low, high)) in enumerate(zip(axes.flat, [(0, None)] + PT_BINS)):
        real = select_observables(cms, low, high)
        selected = {name: {kind: select_observables(v, low, high) for kind, v in arm.items()}
                    for name, arm in arms.items()}
        cell = score_cell(real, selected, args.seed + index)
        cell["pair_pt_bin_gev"] = [low, high]
        cell["label"] = "inclusive" if index == 0 else f"{low} to {high or 'infinity'} GeV"
        report["cells"].append(cell)
        bins = np.linspace(8.5, 11.5, 76)
        if len(real["mass"]):
            ax.hist(real["mass"], bins=bins, density=True, histtype="step",
                    color="black", label="CMS", linewidth=1.5)
        for name, color in [("legacy", "#0072B2"), ("unified", "#D55E00")]:
            for kind, style in [("identity", "--"), ("decoded", "-")]:
                values = selected[name][kind]["mass"]
                if len(values):
                    ax.hist(values, bins=bins, density=True, histtype="step", color=color,
                            linestyle=style, label=f"{name} {kind}", linewidth=1.1)
        count_label = (f"scored n={cell['events_per_distribution']:,}"
                       if cell["status"] == "scored" else "too few events to score")
        ax.set(title=f"{cell['label']} | {count_label}",
               xlabel="Dimuon mass [GeV]", ylabel="Density [1/GeV]")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("unifiedP1 frozen fallback checkpoint: held-out Upsilon transfer", fontsize=15)
    fig.savefig(args.output_dir / "transfer_comparison.png", dpi=180)
    plt.close(fig)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(report_path)
    print(json.dumps(report["cells"][0], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
