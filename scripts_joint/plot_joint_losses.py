#!/usr/bin/env python
"""Plot separate training-loss figures for cms_Joint runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_A = REPO_ROOT / "outputs" / "cms_Joint" / "Run_A"
DEFAULT_RUN_B = REPO_ROOT / "outputs" / "cms_Joint" / "Run_B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-a",
        type=Path,
        default=DEFAULT_RUN_A,
        help="Run A directory or history.json path.",
    )
    parser.add_argument(
        "--run-b",
        type=Path,
        default=DEFAULT_RUN_B,
        help="Run B directory or history.json path.",
    )
    parser.add_argument(
        "--output-a",
        type=Path,
        default=None,
        help="Run A output image; defaults to <run-a>/loss_curve.png.",
    )
    parser.add_argument(
        "--output-b",
        type=Path,
        default=None,
        help="Run B output image; defaults to <run-b>/loss_curve.png.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=10,
        help="Trailing moving-average window; use 1 to disable smoothing.",
    )
    parser.add_argument(
        "--linear-y", action="store_true", help="Use a linear rather than logarithmic y-axis."
    )
    return parser.parse_args()


def _history_path(value: Path) -> Path:
    value = value.expanduser().resolve()
    return value / "history.json" if value.is_dir() else value


def load_history(value: Path) -> list[dict]:
    path = _history_path(value)
    if not path.exists():
        raise FileNotFoundError(
            f"Training history not found: {path}\n"
            "Finish that run first, or pass its directory with --run-a/--run-b."
        )
    history = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(history, list) or not history:
        raise ValueError(f"Training history is empty or invalid: {path}")
    required = {"global_epoch", "stage", "train"}
    if any(not required.issubset(row) for row in history):
        raise ValueError(f"Training history has an unsupported schema: {path}")
    return history


def _series(history: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    epochs = np.asarray([row["global_epoch"] for row in history], dtype=int)
    values = np.asarray([row["train"][key] for row in history], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite values found in train.{key}")
    return epochs, values


def _smoothed(
    epochs: np.ndarray, values: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray]:
    window = min(max(int(window), 1), len(values))
    if window == 1:
        return epochs, values
    kernel = np.ones(window, dtype=float) / window
    return epochs[window - 1 :], np.convolve(values, kernel, mode="valid")


def _stage_guides(ax, history: list[dict]) -> None:
    starts: list[tuple[int, str]] = []
    previous = None
    for row in history:
        stage = str(row["stage"])
        if stage != previous:
            starts.append((int(row["global_epoch"]), stage))
            previous = stage
    final_epoch = int(history[-1]["global_epoch"])
    for index, (start, _) in enumerate(starts):
        end = starts[index + 1][0] - 1 if index + 1 < len(starts) else final_epoch
        ax.axvspan(start, end, color="#718096", alpha=0.04 if index % 2 == 0 else 0.09)
        if index:
            ax.axvline(start - 0.5, color="#718096", linewidth=0.8, alpha=0.5)
        ax.text(
            (start + end) / 2,
            0.98,
            f"Stage {index + 1}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            color="#4a5568",
        )


def plot_history(
    history: list[dict],
    run_label: str,
    output: Path,
    *,
    smooth: int = 10,
    log_y: bool = True,
) -> Path:
    colors = {"total": "#2b6cb0", "jpsi": "#2f855a", "z": "#dd6b20"}
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True, constrained_layout=True)

    epochs, loss = _series(history, "loss")
    axes[0].plot(
        epochs, loss, color=colors["total"], alpha=0.2, linewidth=0.8, label="Raw"
    )
    smooth_epochs, smooth_loss = _smoothed(epochs, loss, smooth)
    axes[0].plot(
        smooth_epochs,
        smooth_loss,
        color=colors["total"],
        linewidth=2.2,
        label="Moving average",
    )

    for region in ("jpsi", "z"):
        region_epochs, region_loss = _series(history, f"{region}_loss")
        axes[1].plot(
            region_epochs,
            region_loss,
            color=colors[region],
            alpha=0.15,
            linewidth=0.7,
        )
        region_epochs, region_loss = _smoothed(region_epochs, region_loss, smooth)
        axes[1].plot(
            region_epochs,
            region_loss,
            color=colors[region],
            linewidth=2.0,
            label=region.upper() if region == "z" else "J/psi",
        )

    for ax in axes:
        _stage_guides(ax, history)
        ax.grid(alpha=0.2)
        ax.set_ylabel("Training loss")
        if log_y:
            ax.set_yscale("log")
        ax.legend(frameon=False, ncol=2)
    axes[0].set_title(
        f"cms_Joint {run_label} training loss (moving average: {max(smooth, 1)} epochs)"
    )
    axes[1].set_xlabel("Global epoch")
    axes[1].legend(frameon=False, ncol=2)

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> int:
    args = parse_args()
    if args.smooth < 1:
        raise ValueError("--smooth must be at least 1")
    outputs = []
    for label, source, requested_output in (
        ("Run A", args.run_a, args.output_a),
        ("Run B", args.run_b, args.output_b),
    ):
        history_path = _history_path(source)
        output = (
            requested_output
            if requested_output is not None
            else history_path.parent / "loss_curve.png"
        )
        outputs.append(
            plot_history(
                load_history(source),
                label,
                output,
                smooth=args.smooth,
                log_y=not args.linear_y,
            )
        )
    for output in outputs:
        print(f"Wrote loss curve: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
