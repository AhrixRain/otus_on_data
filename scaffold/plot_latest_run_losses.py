#!/usr/bin/env python
"""Plot train and eval loss curves for the latest cms_Joint runs.

Ad-hoc analysis helper (scaffold/). Reads only: <run>/history.json and
<run>/config.resolved.json. Writes only into the --output-dir it is given.

Usage:
    python scaffold/plot_latest_run_losses.py
    python scaffold/plot_latest_run_losses.py --in-progress Run_H_D3zcycle
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
JOINT_ROOT = REPO_ROOT / "outputs" / "cms_Joint"

# (run directory, short label) - ordered oldest -> newest by history mtime.
RUN_TABLE = [
    ("Run_H", "H · pT-sliced (baseline)"),
    ("Run_H_cycleNoNoise", "H_cycleNoNoise · cycle"),
    ("Run_H_anchor", "H_anchor · frozen mean map"),
    ("Run_H_A2frozen", "A2frozen · frozen kernel"),
    ("Run_H_A2cycleNoise", "A2.4 · cycle noise native"),
    ("Run_H_A2floor", "A2floor · floor + learned"),
    ("Run_H_A2cycW5native", "A2cycW5native · D3 probe β=5"),
    ("Run_H_D3zcycle", "D3zcycle · z-space cycle"),
]

RUN_COLORS = [
    "#718096",
    "#2b6cb0",
    "#2c7a7b",
    "#2f855a",
    "#b7791f",
    "#c05621",
    "#9b2c2c",
    "#6b46c1",
]
TRAIN_COLOR = "#2b6cb0"
EVAL_COLOR = "#dd6b20"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Run directory names to include; defaults to the latest campaign set.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory; defaults to outputs/cms_Joint/loss_curves_<date>.",
    )
    p.add_argument("--smooth", type=int, default=10, help="Trailing moving-average window.")
    p.add_argument(
        "--in-progress",
        nargs="*",
        default=[],
        help="Run dirs currently training (marked in titles, not treated as complete).",
    )
    return p.parse_args()


def history_path(run_dir: Path) -> Path:
    return run_dir / "history.json"


def load_history(run_dir: Path) -> list[dict]:
    path = history_path(run_dir)
    history = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(history, list) or not history:
        raise ValueError(f"empty or invalid history: {path}")
    required = {"global_epoch", "stage", "train"}
    if any(not required.issubset(row) for row in history):
        raise ValueError(f"unsupported history schema: {path}")
    return history


def planned_epochs(run_dir: Path) -> int | None:
    cfg = run_dir / "config.resolved.json"
    if not cfg.exists():
        return None
    stages = json.loads(cfg.read_text(encoding="utf-8")).get("stages", [])
    enabled = [s for s in stages if s.get("enabled", True)]
    return sum(int(s.get("epochs", 0)) for s in enabled)


def prior_info(run_dir: Path) -> dict:
    """Prior identity, so eval-loss levels are not compared across priors."""
    cfg = run_dir / "config.resolved.json"
    if not cfg.exists():
        return {"tag": "prior unknown", "jpsi_prior": None}
    resolved = json.loads(cfg.read_text(encoding="utf-8"))
    components = resolved.get("prior_components") or {}
    honest = bool(components.get("enabled"))
    jpsi = resolved.get("regions", {}).get("jpsi", {}).get("paths", {}).get("theory_prior_file")
    return {
        "tag": "honest component prior" if honest else "legacy prior",
        "jpsi_prior": Path(jpsi).name if jpsi else None,
    }


def stage_spans(history: list[dict]) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    previous = None
    for row in history:
        name = str(row["stage"])
        epoch = int(row["global_epoch"])
        if name != previous:
            spans.append([epoch, epoch, name])  # type: ignore[arg-type]
            previous = name
        else:
            spans[-1][1] = epoch
    spans[-1][1] = int(history[-1]["global_epoch"])
    return [tuple(s) for s in spans]  # type: ignore[return-value]


def _series(history: list[dict], key: str, *, top_level: bool = False) -> tuple[np.ndarray, np.ndarray]:
    epochs, values = [], []
    for row in history:
        source = row if top_level else row["train"]
        value = source.get(key)
        if value is None:
            continue
        value = float(value)
        if not np.isfinite(value):
            continue
        epochs.append(int(row["global_epoch"]))
        values.append(value)
    return np.asarray(epochs, dtype=int), np.asarray(values, dtype=float)


def _smoothed(epochs: np.ndarray, values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    window = min(max(int(window), 1), len(values))
    if window <= 1:
        return epochs, values
    kernel = np.ones(window, dtype=float) / window
    return epochs[window - 1 :], np.convolve(values, kernel, mode="valid")


def draw_stages(ax, spans, *, label_style: str = "short") -> None:
    for index, (start, end, _name) in enumerate(spans):
        ax.axvspan(start - 0.5, end + 0.5, color="#718096", alpha=0.04 if index % 2 == 0 else 0.09)
        if index:
            ax.axvline(start - 0.5, color="#718096", linewidth=0.8, alpha=0.5)
        text = f"S{index + 1}" if label_style == "short" else f"Stage {index + 1}"
        ax.text(
            (start + end) / 2,
            0.97,
            text,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7.5,
            color="#4a5568",
        )


def load_runs(names: list[str], in_progress: set[str]) -> list[dict]:
    runs = []
    for name in names:
        run_dir = JOINT_ROOT / name
        history = load_history(run_dir)
        spans = stage_spans(history)
        last_ge = int(history[-1]["global_epoch"])
        planned = planned_epochs(run_dir)
        if name in in_progress:
            status = "in progress"
        elif planned is None:
            status = "unknown plan"
        elif last_ge >= planned:
            status = "complete"
        else:
            status = "partial"
        runs.append(
            {
                "name": name,
                "dir": run_dir,
                "history": history,
                "spans": spans,
                "last_ge": last_ge,
                "planned": planned,
                "status": status,
                "mtime": history_path(run_dir).stat().st_mtime,
                "prior": prior_info(run_dir),
            }
        )
    return runs


def plot_grid(runs: list[dict], output: Path, smooth: int) -> Path:
    ncols = 4
    nrows = int(np.ceil(len(runs) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.6 * ncols, 3.4 * nrows), constrained_layout=True
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, run in zip(axes, runs):
        history = run["history"]
        epochs, loss = _series(history, "loss")
        ax.plot(epochs, loss, color=TRAIN_COLOR, alpha=0.18, linewidth=0.8)
        se, sv = _smoothed(epochs, loss, smooth)
        ax.plot(se, sv, color=TRAIN_COLOR, linewidth=2.0, label=f"train (avg {min(smooth, len(loss))})")
        ee, ev = _series(history, "eval_loss", top_level=True)
        if len(ee):
            ax.plot(
                ee,
                ev,
                color=EVAL_COLOR,
                linewidth=1.5,
                marker="o",
                markersize=3.5,
                label="eval",
            )
        draw_stages(ax, run["spans"])
        ax.set_yscale("log")
        ax.grid(alpha=0.2)
        plan = f"/{run['planned']}" if run["planned"] else ""
        ax.set_title(
            f"{run['name']}\n{run['last_ge']}{plan} epochs · {run['status']}\n{run['prior']['tag']}",
            fontsize=9,
        )
        ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    for ax in axes[len(runs) :]:
        ax.axis("off")
    fig.suptitle("cms_Joint latest runs — train vs eval loss", fontsize=13)
    fig.supxlabel("Global epoch", fontsize=10)
    fig.supylabel("Loss (log scale)", fontsize=10)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_overlay(
    runs: list[dict], output: Path, smooth: int, *, key: str, top_level: bool, title: str, ylabel: str
) -> Path:
    fig, ax = plt.subplots(figsize=(11, 6.0), constrained_layout=True)
    for color, run in zip(RUN_COLORS, runs):
        epochs, values = _series(run["history"], key, top_level=top_level)
        if not len(epochs):
            continue
        style = dict(linewidth=1.4, marker="o", markersize=3.5) if top_level else dict(linewidth=1.8)
        ax.plot(epochs, values, color=color, label=f"{run['name']} · {run['prior']['tag']}", **style)
    ax.set_yscale("log")
    ax.grid(alpha=0.2)
    ax.set_xlabel("Global epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8.5, ncol=2)
    tags = {run["prior"]["tag"] for run in runs}
    if len(tags) > 1:
        note = (
            "Caution: two different J/psi theory priors are mixed here "
            "(legacy single prior vs honest component prior). "
            "Absolute loss levels are not comparable across that boundary."
        )
        fig.text(0.5, -0.02, note, ha="center", va="top", fontsize=8.5, color="#9b2c2c")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> int:
    args = parse_args()
    if args.smooth < 1:
        raise ValueError("--smooth must be >= 1")
    names = args.runs if args.runs else [name for name, _ in RUN_TABLE]
    labels = dict(RUN_TABLE)
    runs = load_runs(names, set(args.in_progress))
    runs.sort(key=lambda r: r["mtime"])

    output_dir = args.output_dir or JOINT_ROOT / f"loss_curves_{time.strftime('%Y-%m-%d')}"

    written = []
    written.append(plot_grid(runs, output_dir / "loss_curves_latest_runs.png", args.smooth))
    written.append(
        plot_overlay(
            runs,
            output_dir / "loss_curves_latest_runs_train_overlay.png",
            args.smooth,
            key="loss",
            top_level=False,
            title="cms_Joint latest runs — train total loss (raw epoch values)",
            ylabel="Train loss",
        )
    )
    written.append(
        plot_overlay(
            runs,
            output_dir / "loss_curves_latest_runs_eval_overlay.png",
            args.smooth,
            key="eval_loss",
            top_level=True,
            title="cms_Joint latest runs — held-out eval loss (every eval_every epochs)",
            ylabel="Eval loss",
        )
    )

    for region, label in (("jpsi", "J/ψ"), ("z", "Z")):
        written.append(
            plot_overlay(
                runs,
                output_dir / f"loss_curves_latest_runs_{region}_overlay.png",
                args.smooth,
                key=f"{region}_loss",
                top_level=False,
                title=f"cms_Joint latest runs — {label} region train loss",
                ylabel=f"{label} train loss",
            )
        )

    manifest = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "smooth_window": args.smooth,
        "in_progress": sorted(args.in_progress),
        "runs": [
            {
                "run": r["name"],
                "label": labels.get(r["name"], r["name"]),
                "history": str(history_path(r["dir"]).relative_to(REPO_ROOT)),
                "history_mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["mtime"])),
                "rows": len(r["history"]),
                "last_global_epoch": r["last_ge"],
                "planned_epochs": r["planned"],
                "status": r["status"],
                "prior": r["prior"],
                "stages": [s[2] for s in r["spans"]],
                "first_train_loss": float(r["history"][0]["train"]["loss"]),
                "last_train_loss": float(r["history"][-1]["train"]["loss"]),
                "first_eval_loss": next(
                    (float(x["eval_loss"]) for x in r["history"] if x.get("eval_loss") is not None),
                    None,
                ),
                "last_eval_loss": next(
                    (
                        float(x["eval_loss"])
                        for x in reversed(r["history"])
                        if x.get("eval_loss") is not None
                    ),
                    None,
                ),
            }
            for r in runs
        ],
        "figures": [str(p.relative_to(REPO_ROOT)) for p in written],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    for path in written:
        print(f"Wrote {path}")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
