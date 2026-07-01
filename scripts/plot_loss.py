from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOSS_COLUMNS = [
    "train_loss",
    "eval_loss",
    "train_x_loss",
    "eval_x_loss",
    "train_z_loss",
    "eval_z_loss",
    "train_alt_x_loss",
    "eval_alt_x_loss",
    "train_x_constraint_loss",
]

TOTAL_COLUMNS = ["train_loss", "eval_loss"]
COMPONENT_COLUMNS = [column for column in LOSS_COLUMNS if column not in TOTAL_COLUMNS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot CMS DoubleElectron training losses from train_log.csv.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path, help="Run directory containing train_log.csv.")
    source.add_argument("--log", type=Path, help="Path to a train_log.csv file.")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path.")
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Use a linear y-axis instead of the default logarithmic y-axis.",
    )
    parser.add_argument(
        "--components",
        action="store_true",
        help="Add component loss curves below the total train/eval loss plot.",
    )
    parser.add_argument("--title", default=None, help="Optional plot title.")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.run_dir is not None:
        run_dir = args.run_dir
        log_path = run_dir / "train_log.csv"
        output_path = args.output or (run_dir / "loss_curve.png")
    else:
        log_path = args.log
        output_path = args.output or log_path.with_name("loss_curve.png")
    return log_path, output_path


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


def load_rows(log_path: Path) -> list[dict[str, str]]:
    if not log_path.exists():
        raise FileNotFoundError(f"Training log not found: {log_path}")
    with log_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Training log has no data rows: {log_path}")
    required = {"epoch", "stage", "train_loss", "eval_loss"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Training log is missing required columns: {', '.join(sorted(missing))}")
    return rows


def series(rows: list[dict[str, str]], column: str) -> tuple[list[int], list[float]]:
    epochs: list[int] = []
    values: list[float] = []
    for row in rows:
        value = parse_float(row.get(column))
        if value is None:
            continue
        epochs.append(int(row["epoch"]))
        values.append(value)
    return epochs, values


def stage_boundaries(rows: list[dict[str, str]]) -> list[tuple[int, str]]:
    boundaries: list[tuple[int, str]] = []
    previous_stage: str | None = None
    for row in rows:
        stage = row.get("stage", "")
        if stage and stage != previous_stage:
            boundaries.append((int(row["epoch"]), stage))
            previous_stage = stage
    return boundaries


def draw_stage_markers(ax: plt.Axes, rows: list[dict[str, str]]) -> None:
    ymin, ymax = ax.get_ylim()
    for epoch, stage in stage_boundaries(rows):
        ax.axvline(epoch, color="0.72", linewidth=0.9, linestyle="--", zorder=0)
        ax.text(
            epoch,
            ymax,
            stage,
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
            color="0.35",
        )
    ax.set_ylim(ymin, ymax)


def plot_columns(ax: plt.Axes, rows: list[dict[str, str]], columns: list[str]) -> bool:
    plotted = False
    for column in columns:
        epochs, values = series(rows, column)
        if not values:
            continue
        marker = "o" if len(values) < 3 else None
        ax.plot(epochs, values, marker=marker, linewidth=1.8, markersize=4, label=column)
        plotted = True
    return plotted


def make_plot(
    rows: list[dict[str, str]],
    output_path: Path,
    *,
    log_scale: bool,
    components: bool,
    title: str | None,
) -> None:
    nrows = 2 if components else 1
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=(10, 5.2 if not components else 8.0),
        sharex=True,
        constrained_layout=True,
    )
    if nrows == 1:
        axes = [axes]
    else:
        axes = list(axes)

    total_ax = axes[0]
    plot_columns(total_ax, rows, TOTAL_COLUMNS)
    total_ax.set_ylabel("Loss")
    total_ax.set_title(title or "Training Loss Curve")
    total_ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.7)
    if log_scale:
        total_ax.set_yscale("log")
    total_ax.legend(loc="best", frameon=False)
    draw_stage_markers(total_ax, rows)

    if components:
        component_ax = axes[1]
        if plot_columns(component_ax, rows, COMPONENT_COLUMNS):
            component_ax.legend(loc="best", frameon=False, ncol=2, fontsize=8)
        component_ax.set_ylabel("Component loss")
        component_ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.7)
        if log_scale:
            component_ax.set_yscale("log")
        draw_stage_markers(component_ax, rows)

    axes[-1].set_xlabel("Global epoch")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    log_path, output_path = resolve_paths(args)
    rows = load_rows(log_path)
    make_plot(
        rows,
        output_path,
        log_scale=not args.linear,
        components=args.components,
        title=args.title,
    )
    print(f"Wrote {output_path} from {log_path} ({len(rows)} logged epochs).")


if __name__ == "__main__":
    main()
