#!/usr/bin/env python
"""Compare loss curves across every J/psi run that has a train_log.csv.

Reads all ``outputs/cms_JpsiDoubleMuons/<run>/train_log.csv`` files and draws
two log-scale panels: training loss and evaluation loss versus global epoch.
Runs with at most two logged epochs are drawn as points (smoke/partial runs);
longer runs are drawn as lines. A run log that only contains a header is
skipped. The script is read-only with respect to the run directories and
writes a single PNG.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = REPO_ROOT / "outputs" / "cms_JpsiDoubleMuons"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help=f"Directory of run folders (default: {DEFAULT_RUNS_ROOT}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: <runs-root>/all_runs_loss.png).",
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Use linear y-axes instead of logarithmic ones.",
    )
    return parser.parse_args()


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if parsed <= 0:
        return None
    return parsed


def load_run(log_path: Path) -> dict[str, list] | None:
    try:
        with log_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return None
    if not rows:
        return None
    required = {"epoch", "train_loss"}
    if not required.issubset(rows[0]):
        return None
    epochs: list[int] = []
    train: list[float] = []
    eval_: list[float] = []
    for row in rows:
        train_value = parse_float(row.get("train_loss"))
        if train_value is None:
            continue
        epochs.append(int(row["epoch"]))
        train.append(train_value)
        eval_value = parse_float(row.get("eval_loss"))
        if eval_value is not None:
            eval_.append((int(row["epoch"]), eval_value))
    if not train:
        return None
    return {
        "name": log_path.parent.name,
        "epochs": epochs,
        "train": train,
        "eval_epochs": [epoch for epoch, _ in eval_],
        "eval": [value for _, value in eval_],
        "n_rows": len(train),
    }


def collect_runs(runs_root: Path) -> list[dict[str, list]]:
    runs: list[dict[str, list]] = []
    for log_path in sorted(runs_root.glob("*/train_log.csv")):
        run = load_run(log_path)
        if run is not None:
            runs.append(run)
    return runs


def draw_panel(
    ax: plt.Axes,
    runs: list[dict[str, list]],
    *,
    column: str,
    ylabel: str,
    colors: list,
) -> None:
    for index, run in enumerate(runs):
        color = colors[index % len(colors)]
        if column == "train":
            x = run["epochs"]
            y = run["train"]
        else:
            x = run["eval_epochs"]
            y = run["eval"]
        if not y:
            continue
        if run["n_rows"] <= 2:
            ax.plot(x, y, marker="o", linestyle="", color=color, markersize=5, label=run["name"])
        else:
            marker = "o" if run["n_rows"] < 8 else None
            ax.plot(
                x,
                y,
                marker=marker,
                markersize=3.5,
                linewidth=1.6,
                color=color,
                label=run["name"],
            )
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.7)


def main() -> int:
    args = parse_args()
    runs_root = args.runs_root.expanduser().resolve()
    if not runs_root.is_dir():
        raise FileNotFoundError(f"Runs root not found: {runs_root}")
    runs = collect_runs(runs_root)
    if not runs:
        raise ValueError(f"No runs with readable train_log.csv under {runs_root}")

    output_path = args.output or (runs_root / "all_runs_loss.png")
    output_path = output_path.expanduser().resolve()

    cmap = plt.get_cmap("tab20")
    colors = [cmap(index) for index in range(20)] if len(runs) <= 20 else [
        plt.get_cmap("viridis")(index / max(1, len(runs) - 1)) for index in range(len(runs))
    ]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(13, 9.5),
        sharex=True,
        constrained_layout=True,
    )
    draw_panel(axes[0], runs, column="train", ylabel="Training loss", colors=colors)
    draw_panel(axes[1], runs, column="eval", ylabel="Evaluation loss", colors=colors)
    for ax in axes:
        if not args.linear:
            ax.set_yscale("log")
    axes[1].set_xlabel("Global epoch")
    axes[0].set_title("J/psi runs: loss comparison (all runs under outputs/cms_JpsiDoubleMuons)")
    fig.legend(
        handles=axes[0].lines,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        fontsize=7.5,
        frameon=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path} ({len(runs)} runs).")
    for run in runs:
        suffix = " [partial/smoke]" if run["n_rows"] <= 2 else ""
        print(f"  {run['name']}: {run['n_rows']} logged epoch(s){suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
