#!/usr/bin/env python
"""Per-event closure figure (RQ3): residual vs identity and posterior calibration.

Read-only. Reads the retained ppzee artifacts and writes one two-panel PNG:
  (a) residual_rms_vs_identity for identity, upstream OTUS, this work and the oracle;
  (b) pull standard deviation and 1-sigma coverage for the three retained checkpoints
      against their nominal values.

Usage:
    python scripts_joint/plot_per_event_closure.py [--output PaperPlots/images/per_event_closure.png]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
PULL_JSON = REPO_ROOT / "outputs" / "cms_Joint" / "ppzee" / "pull_coverage" / "pull_coverage.json"
CLOSURE_JSON = REPO_ROOT / "outputs" / "cms_Joint" / "ppzee" / "paired_closure.json"

IDENTITY = 1.0
ORACLE = 0.8099          # affine oracle fitted on the pairing (ppzee_error_decomposition)
UPSTREAM = 3.3304        # upstream OTUS encoder, same events
NOMINAL_PULL_STD = 1.0
NOMINAL_COVERAGE = 0.6827


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def residual_rows() -> list[tuple[str, float]]:
    rows = [("identity (hand back x)", IDENTITY), ("upstream OTUS", UPSTREAM)]
    if PULL_JSON.exists():
        payload = load_json(PULL_JSON)
        for label, block in payload.get("checkpoints", {}).items():
            native = (block.get("settings") or {}).get("native") or {}
            value = (native.get("per_event_mass") or {}).get("residual_rms_vs_identity")
            if value is None:
                value = native.get("residual_rms_vs_identity")
            if value is not None:
                pretty = {"stage1_deterministic": "this work, stage 1",
                          "stage2_gaussian": "this work, stage 2",
                          "last": "this work, last"}.get(label, label)
                rows.append((pretty, float(value)))
    rows.append(("oracle (fitted on pairs)", ORACLE))
    return rows


def calibration_rows() -> list[tuple[str, float, float]]:
    rows = []
    if PULL_JSON.exists():
        payload = load_json(PULL_JSON)
        for label, block in payload.get("checkpoints", {}).items():
            native = (block.get("settings") or {}).get("native") or {}
            posterior = native.get("posterior") or native
            pull = posterior.get("pull_std")
            coverage = posterior.get("coverage_1sigma")
            if pull is None or coverage is None:
                continue
            pretty = {"stage1_deterministic": "stage 1 (det.)",
                      "stage2_gaussian": "stage 2",
                      "last": "last"}.get(label, label)
            rows.append((pretty, float(pull), float(coverage)))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "PaperPlots" / "images" / "per_event_closure.png")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    rows = residual_rows()
    labels = [row[0] for row in rows]
    values = [row[1] for row in rows]
    colors = ["#888888", "#b83280"] + ["#2b6cb0"] * (len(rows) - 3) + ["#2f855a"]
    axes[0].barh(labels[::-1], values[::-1], color=colors[::-1])
    axes[0].axvline(IDENTITY, color="black", lw=1.0, ls=":")
    axes[0].axvline(ORACLE, color="tab:green", lw=1.0, ls="--")
    axes[0].set_xlabel("residual rms / identity rms")
    axes[0].set_title("Per-event closure (paired benchmark)")
    for index, value in enumerate(values[::-1]):
        axes[0].text(value + 0.05, index, f"{value:.3f}", va="center", fontsize=8)

    calib = calibration_rows()
    if calib:
        names = [row[0] for row in calib]
        pulls = np.array([row[1] for row in calib], dtype=float)
        coverage = np.array([row[2] for row in calib], dtype=float)
        x = np.arange(len(names))
        pulls_plot = np.where(pulls > 100.0, 100.0, pulls)
        axes[1].bar(x - 0.2, pulls_plot, 0.4, label="pull std", color="#dd6b20")
        axes[1].bar(x + 0.2, coverage * 100.0, 0.4, label="1-sigma coverage [%]", color="#2b6cb0")
        axes[1].axhline(NOMINAL_PULL_STD, color="#dd6b20", ls=":", lw=1.0)
        axes[1].axhline(NOMINAL_COVERAGE * 100.0, color="#2b6cb0", ls=":", lw=1.0)
        axes[1].set_yscale("log")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(names)
        axes[1].set_ylabel("value (log scale; pull std clipped at 100)")
        axes[1].set_title("Posterior calibration (dotted = nominal)")
        for index, (pull, cov) in enumerate(zip(pulls, coverage)):
            note = "degenerate" if pull > 100.0 else f"{pull:.1f}"
            axes[1].text(index - 0.2, pulls_plot[index] * 1.15, note, ha="center", fontsize=8)
            axes[1].text(index + 0.2, max(cov * 100.0, 0.05) * 1.15, f"{cov * 100:.1f}", ha="center", fontsize=8)
        axes[1].legend(frameon=False, fontsize=8)
    else:
        axes[1].text(0.5, 0.5, "no calibration payload", ha="center", va="center")
    figure.tight_layout()
    figure.savefig(args.output, dpi=140, bbox_inches="tight")
    print("wrote", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
