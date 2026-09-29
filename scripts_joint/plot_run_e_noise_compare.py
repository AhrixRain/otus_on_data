#!/usr/bin/env python
"""Compare Run E and the Run E noise-control runs.

The two noise-control runs (noiseLowCore and noiseLowBoth) resumed from the
Run_E stage-1 best checkpoint and therefore start at Run_E stage 2.  Two views
are produced:

* the full Run_E history, with the noise-control runs shifted onto Run_E's
  global-epoch axis so the stage boundaries line up;
* an apples-to-apples stage-2 -> stage-3 -> stage-4 view, with Run_E truncated
  to the same stages and stage-local epochs concatenated for every run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_E = REPO_ROOT / "outputs" / "cms_Joint" / "Run_E"
DEFAULT_RUN_CORE = DEFAULT_RUN_E / "Run_E_noiseLowCore"
DEFAULT_RUN_BOTH = DEFAULT_RUN_E / "Run_E_noiseLowBoth"

COLORS = {
    "Run_E": "#2b6cb0",
    "Run_E_noiseLowCore": "#dd6b20",
    "Run_E_noiseLowBoth": "#2f855a",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-e", type=Path, default=DEFAULT_RUN_E)
    parser.add_argument("--run-core", type=Path, default=DEFAULT_RUN_CORE)
    parser.add_argument("--run-both", type=Path, default=DEFAULT_RUN_BOTH)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to <run-e>/loss_curve_compare_noise_controls.png.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=10,
        help="Trailing moving-average window for the total train loss (1 disables).",
    )
    return parser.parse_args()


def _history_path(value: Path) -> Path:
    value = value.expanduser().resolve()
    return value / "history.json" if value.is_dir() else value


def load_history(value: Path) -> list[dict]:
    path = _history_path(value)
    if not path.exists():
        raise FileNotFoundError(f"Training history not found: {path}")
    history = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(history, list) or not history:
        raise ValueError(f"Training history is empty or invalid: {path}")
    return history


def load_region_weights(run_dir: Path) -> dict[str, float]:
    config_path = run_dir.expanduser().resolve() / "config.resolved.json"
    if not config_path.exists():
        return {}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        str(name): float(value)
        for name, value in (config.get("region_weights") or {}).items()
    }


def _train_series(history: list[dict], *, offset: int) -> tuple[np.ndarray, np.ndarray]:
    epochs: list[int] = []
    values: list[float] = []
    for row in history:
        train = row.get("train") or {}
        value = train.get("loss")
        if isinstance(value, (int, float)) and np.isfinite(value):
            epochs.append(int(row["global_epoch"]) + offset)
            values.append(float(value))
    return np.asarray(epochs, dtype=int), np.asarray(values, dtype=float)


def _validation_series(
    history: list[dict],
    *,
    offset: int,
    region_weights: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return validation loss, using the stored eval_loss or rebuilding it.

    Run_E predates the top-level ``eval_loss`` field, but its per-region
    ``base_loss`` values are stored in ``region_validation``.  The trainer's
    eval loss is a region-weight-normalised sum of those base losses, so the
    same quantity can be reconstructed exactly for every run here.
    """

    weights = region_weights or {}
    epochs: list[int] = []
    values: list[float] = []
    for row in history:
        stored = row.get("eval_loss")
        if isinstance(stored, (int, float)) and np.isfinite(stored):
            value = float(stored)
        else:
            region_validation = row.get("region_validation") or {}
            total = 0.0
            weight_sum = 0.0
            for name, metrics in region_validation.items():
                if not isinstance(metrics, dict) or "base_loss" not in metrics:
                    continue
                weight = float(weights.get(name, 1.0))
                total += weight * float(metrics["base_loss"])
                weight_sum += weight
            if weight_sum == 0.0:
                continue
            value = total / weight_sum
        epochs.append(int(row["global_epoch"]) + offset)
        values.append(value)
    return np.asarray(epochs, dtype=int), np.asarray(values, dtype=float)


def _smoothed(
    epochs: np.ndarray, values: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray]:
    window = min(max(int(window), 1), len(values))
    if window <= 1 or len(values) < window:
        return epochs, values
    kernel = np.ones(window, dtype=float) / window
    return epochs[window - 1 :], np.convolve(values, kernel, mode="valid")


def _stage_spans(history: list[dict]) -> list[tuple[int, int, str]]:
    starts: list[tuple[int, str]] = []
    previous: str | None = None
    for row in history:
        stage = str(row["stage"])
        if stage != previous:
            starts.append((int(row["global_epoch"]), stage))
            previous = stage
    final_epoch = int(history[-1]["global_epoch"])
    spans: list[tuple[int, int, str]] = []
    for index, (start, stage) in enumerate(starts):
        end = starts[index + 1][0] - 1 if index + 1 < len(starts) else final_epoch
        spans.append((start, end, stage))
    return spans


def _draw_stage_guides(ax: plt.Axes, history: list[dict]) -> None:
    for index, (start, end, stage) in enumerate(_stage_spans(history)):
        ax.axvspan(
            start,
            end,
            color="#718096",
            alpha=0.04 if index % 2 == 0 else 0.09,
            zorder=0,
        )
        if index:
            ax.axvline(start - 0.5, color="#718096", linewidth=0.8, alpha=0.5, zorder=1)
        ax.text(
            (start + end) / 2,
            0.98,
            stage.replace("runA_", "").replace("_", " "),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            color="#4a5568",
        )


def _plot_train(
    ax: plt.Axes,
    history: list[dict],
    *,
    label: str,
    color: str,
    offset: int,
    smooth: int,
) -> None:
    epochs, values = _train_series(history, offset=offset)
    ax.plot(epochs, values, color=color, alpha=0.16, linewidth=0.8)
    smooth_epochs, smooth_values = _smoothed(epochs, values, smooth)
    ax.plot(
        smooth_epochs,
        smooth_values,
        color=color,
        linewidth=2.3,
        label=f"{label} (moving average {max(smooth, 1)})",
    )
    ax.set_ylabel("Total training loss")
    ax.set_yscale("log")


def _plot_validation(
    ax: plt.Axes,
    history: list[dict],
    *,
    label: str,
    color: str,
    offset: int,
    region_weights: dict[str, float],
) -> None:
    epochs, values = _validation_series(
        history, offset=offset, region_weights=region_weights
    )
    ax.plot(
        epochs,
        values,
        color=color,
        marker="o",
        markersize=3.2,
        linewidth=1.7,
        label=label,
    )
    ax.set_ylabel("Validation loss")
    ax.set_yscale("log")


def _stage_sequence_series(
    history: list[dict],
    key: str,
    *,
    region_weights: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a series against concatenated stage-local epoch.

    The first stage in *history* is placed at x=1; subsequent stages continue
    after the previous stage's last epoch.
    """

    stage_offsets: dict[str, int] = {}
    next_offset = 0
    epochs: list[int] = []
    values: list[float] = []
    for row in history:
        stage = str(row["stage"])
        if stage not in stage_offsets:
            stage_offsets[stage] = next_offset
            next_offset += int(max(r["stage_epoch"] for r in history if r["stage"] == stage))
        x = stage_offsets[stage] + int(row["stage_epoch"])
        if key == "train.loss":
            value = (row.get("train") or {}).get("loss")
        elif key == "validation_loss":
            value = None
            stored = row.get("eval_loss")
            if isinstance(stored, (int, float)) and np.isfinite(stored):
                value = float(stored)
            else:
                region_validation = row.get("region_validation") or {}
                total = 0.0
                weight_sum = 0.0
                weights = region_weights or {}
                for name, metrics in region_validation.items():
                    if not isinstance(metrics, dict) or "base_loss" not in metrics:
                        continue
                    weight = float(weights.get(name, 1.0))
                    total += weight * float(metrics["base_loss"])
                    weight_sum += weight
                if weight_sum:
                    value = total / weight_sum
        else:
            raise ValueError(f"Unknown stage-sequence key: {key!r}")
        if isinstance(value, (int, float)) and np.isfinite(value):
            epochs.append(x)
            values.append(float(value))
    return np.asarray(epochs, dtype=int), np.asarray(values, dtype=float)


def _stage_sequence_spans(history: list[dict]) -> list[tuple[int, int, str]]:
    stages: list[str] = []
    for row in history:
        stage = str(row["stage"])
        if stage not in stages:
            stages.append(stage)
    spans: list[tuple[int, int, str]] = []
    start = 1
    for stage in stages:
        count = int(max(row["stage_epoch"] for row in history if row["stage"] == stage))
        spans.append((start, start + count - 1, stage))
        start += count
    return spans


def make_full_figure(
    run_e_history: list[dict],
    low_runs: list[dict],
    output: Path,
    *,
    run_e_weights: dict[str, float],
    smooth: int,
) -> None:
    """Full-history view with low-noise runs shifted onto Run_E global epochs."""

    first_stage = str(low_runs[0]["history"][0]["stage"])
    run_e_stage_start = next(
        (
            int(row["global_epoch"])
            for row in run_e_history
            if str(row["stage"]) == first_stage
        ),
        None,
    )
    if run_e_stage_start is None:
        raise ValueError(
            f"Run_E does not contain the first low-noise stage {first_stage!r}"
        )
    offset = run_e_stage_start - int(low_runs[0]["history"][0]["global_epoch"])

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13, 9),
        sharex=True,
        constrained_layout=True,
    )

    _plot_train(
        axes[0],
        run_e_history,
        label="Run_E",
        color=COLORS["Run_E"],
        offset=0,
        smooth=smooth,
    )
    _plot_validation(
        axes[1],
        run_e_history,
        label="Run_E validation",
        color=COLORS["Run_E"],
        offset=0,
        region_weights=run_e_weights,
    )

    for item in low_runs:
        _plot_train(
            axes[0],
            item["history"],
            label=item["label"],
            color=item["color"],
            offset=offset,
            smooth=smooth,
        )
        _plot_validation(
            axes[1],
            item["history"],
            label=f"{item['label']} validation",
            color=item["color"],
            offset=offset,
            region_weights=item["weights"],
        )

    for ax in axes:
        _draw_stage_guides(ax, run_e_history)
        ax.axvline(
            run_e_stage_start - 0.5,
            color="#c05621",
            linewidth=1.4,
            linestyle="--",
            zorder=2,
        )
        ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.6)

    axes[0].annotate(
        "Noise-control runs start here\n"
        "(stage 2, resumed from Run_E stage-1 best checkpoint)",
        xy=(run_e_stage_start - 0.5, 0.70),
        xycoords=("data", "axes fraction"),
        xytext=(run_e_stage_start + 8, 0.84),
        textcoords=("data", "axes fraction"),
        fontsize=9,
        color="#9c4221",
        arrowprops={"arrowstyle": "->", "color": "#c05621", "lw": 1.0},
    )
    axes[0].set_title("Run E noise-control comparison: total training loss", fontsize=13)
    axes[0].legend(frameon=False, loc="upper right")
    axes[1].set_title(
        "Validation loss (region-weighted base loss; lower is better)",
        fontsize=12,
    )
    axes[1].legend(frameon=False, loc="upper right")
    axes[1].set_xlabel(
        "Run E global epoch "
        f"(noise-control runs shifted by +{offset} so stage 2 starts at epoch "
        f"{run_e_stage_start})"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


def make_stage2plus_figure(
    run_e_history: list[dict],
    low_runs: list[dict],
    output: Path,
    *,
    run_e_weights: dict[str, float],
    smooth: int,
) -> None:
    """Apples-to-apples stages 2-4 view with concatenated stage-local epochs."""

    run_e_stage2 = [
        row
        for row in run_e_history
        if str(row["stage"]) != "runA_stage1_deterministic_identity"
    ]
    if not run_e_stage2:
        raise ValueError("Run_E has no stages after stage 1")

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13, 9),
        sharex=True,
        constrained_layout=True,
    )

    train_x, train_y = _stage_sequence_series(run_e_stage2, "train.loss")
    axes[0].plot(train_x, train_y, color=COLORS["Run_E"], alpha=0.16, linewidth=0.8)
    smooth_x, smooth_y = _smoothed(train_x, train_y, smooth)
    axes[0].plot(
        smooth_x,
        smooth_y,
        color=COLORS["Run_E"],
        linewidth=2.3,
        label=f"Run_E (moving average {max(smooth, 1)})",
    )
    val_x, val_y = _stage_sequence_series(
        run_e_stage2, "validation_loss", region_weights=run_e_weights
    )
    axes[1].plot(
        val_x,
        val_y,
        color=COLORS["Run_E"],
        marker="o",
        markersize=3.2,
        linewidth=1.7,
        label="Run_E validation",
    )

    for item in low_runs:
        train_x, train_y = _stage_sequence_series(item["history"], "train.loss")
        axes[0].plot(train_x, train_y, color=item["color"], alpha=0.16, linewidth=0.8)
        smooth_x, smooth_y = _smoothed(train_x, train_y, smooth)
        axes[0].plot(
            smooth_x,
            smooth_y,
            color=item["color"],
            linewidth=2.3,
            label=f"{item['label']} (moving average {max(smooth, 1)})",
        )
        val_x, val_y = _stage_sequence_series(
            item["history"], "validation_loss", region_weights=item["weights"]
        )
        axes[1].plot(
            val_x,
            val_y,
            color=item["color"],
            marker="o",
            markersize=3.2,
            linewidth=1.7,
            label=f"{item['label']} validation",
        )

    spans = _stage_sequence_spans(low_runs[0]["history"])
    for ax in axes:
        for index, (start, end, stage) in enumerate(spans):
            ax.axvspan(
                start,
                end,
                color="#718096",
                alpha=0.04 if index % 2 == 0 else 0.09,
                zorder=0,
            )
            if index:
                ax.axvline(start - 0.5, color="#718096", linewidth=0.8, alpha=0.5, zorder=1)
            ax.text(
                (start + end) / 2,
                0.98,
                stage.replace("runA_", "").replace("_", " "),
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8,
                color="#4a5568",
            )
        ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.6)
        ax.set_yscale("log")

    axes[0].set_ylabel("Total training loss")
    axes[0].set_title(
        "Run E noise-control comparison from stage 2 onward "
        "(Run_E truncated to the same stages; low-noise runs started at stage 2)",
        fontsize=12,
    )
    axes[0].legend(frameon=False, loc="upper right")
    axes[1].set_ylabel("Validation loss")
    axes[1].set_title(
        "Validation loss (region-weighted base loss; lower is better)",
        fontsize=12,
    )
    axes[1].legend(frameon=False, loc="upper right")
    axes[1].set_xlabel(
        "Epoch within the stage-2 -> stage-3 -> stage-4 sequence "
        "(stage-local epochs concatenated)"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


def main() -> int:
    args = parse_args()
    run_e_dir = args.run_e.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else run_e_dir / "loss_curve_compare_noise_controls.png"
    )

    run_e_history = load_history(run_e_dir)
    run_e_weights = load_region_weights(run_e_dir)
    low_runs = []
    for label, source, color in (
        ("Run_E_noiseLowCore", args.run_core, COLORS["Run_E_noiseLowCore"]),
        ("Run_E_noiseLowBoth", args.run_both, COLORS["Run_E_noiseLowBoth"]),
    ):
        source = source.expanduser().resolve()
        low_runs.append(
            {
                "label": label,
                "history": load_history(source),
                "weights": load_region_weights(source),
                "color": color,
            }
        )

    make_full_figure(
        run_e_history,
        low_runs,
        output,
        run_e_weights=run_e_weights,
        smooth=args.smooth,
    )
    make_stage2plus_figure(
        run_e_history,
        low_runs,
        output.with_name(f"{output.stem}_stage2plus{output.suffix}"),
        run_e_weights=run_e_weights,
        smooth=args.smooth,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
