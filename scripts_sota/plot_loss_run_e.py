#!/usr/bin/env python
"""Plot Run E history.json loss curves while training continues."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def series(rows, key):
    x, y = [], []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)):
            x.append(int(row["global_epoch"]))
            y.append(float(value))
    return x, y


def main() -> int:
    args = parse_args()
    history = json.loads((args.run_dir / "history.json").read_text(encoding="utf-8"))
    out = args.output or (args.run_dir / "run_e_loss_curve.png")

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    axes[0, 0].plot(*series(history, "train_loss"), label="train total", linewidth=1.2)
    axes[0, 0].plot(*series(history, "validation_gate_score"), ".", label="val gated score", markersize=4)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("global epoch")
    axes[0, 0].set_ylabel("loss")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].set_title("total / gated validation score")

    axes[0, 1].plot(*series(history, "train_x_loss"), label="x reco")
    axes[0, 1].plot(*series(history, "train_z_loss"), label="z prior")
    axes[0, 1].plot(*series(history, "train_alt_x_loss"), label="x sim (D(z))")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("global epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].set_title("base loss components")

    axes[1, 0].plot(*series(history, "train_x_sota_sinkhorn_weighted"), label="x Sinkhorn weighted")
    axes[1, 0].plot(*series(history, "train_z_sota_sinkhorn_weighted"), label="z Sinkhorn weighted")
    axes[1, 0].plot(*series(history, "train_x_sota_sinkhorn_raw"), label="x Sinkhorn raw", alpha=0.45)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("global epoch")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].set_title("Sinkhorn")

    axes[1, 1].plot(*series(history, "train_x_sota_max_swd_weighted"), label="x max-SW weighted")
    axes[1, 1].plot(*series(history, "train_z_sota_max_swd_weighted"), label="z max-SW weighted")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("global epoch")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].set_title("max-SW")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("Wrote", out, "rows", len(history))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
