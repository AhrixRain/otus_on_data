#!/usr/bin/env python
"""Compare checkpoints with identical evaluation/plot settings.

The standard use case is the OTUS-on-CMS J/psi pipeline:

    python scripts/stage_diagnostic.py \
        --config configs/cms_JpsiDoubleMuons_mps.yaml \
        --checkpoint stage2_end=outputs/cms_JpsiDoubleMuons/Jpsi_v3.5/checkpoint_stage2_joint_transport.pt \
        --checkpoint best_stage3=outputs/cms_JpsiDoubleMuons/Jpsi_v3.5/best_model.pt \
        --output-dir outputs/cms_JpsiDoubleMuons/Jpsi_v3.5/stage_diagnostic

For every checkpoint the script runs the *same* eval + plot pipeline (scripts/
eval.py and scripts/plot.py) with identical data/plotting arguments, then
aggregates the per-checkpoint summaries into:

    <output_dir>/summary.json   (machine-readable, all metrics + deltas)
    <output_dir>/summary.md     (human-readable comparison + explicit answers)

The summary answers two questions depending on how the checkpoints are labeled:

  * stage2_end vs best/final stage3 -> "Did Stage 3 damage x -> z -> x?"
  * v3.5 vs v3.6A (no explicit mass) -> "Did removing explicit invariant-mass
    supervision improve x -> z / x -> z -> x without destroying z -> x?"

Run with the same .venv environment used for training/eval, e.g.
`.venv/bin/python scripts/stage_diagnostic.py ...`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from cms_data import load_config, resolve_config  # noqa: E402

try:
    from scipy.stats import ks_2samp, wasserstein_distance

    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


def finite_values(a: np.ndarray) -> np.ndarray:
    values = np.asarray(a).reshape(-1)
    return values[np.isfinite(values)]


def maybe_ks(a: np.ndarray, b: np.ndarray) -> float | None:
    if not HAS_SCIPY:
        return None
    if len(finite_values(a)) == 0 or len(finite_values(b)) == 0:
        return None
    return float(ks_2samp(finite_values(a), finite_values(b)).statistic)


def maybe_w1(a: np.ndarray, b: np.ndarray) -> float | None:
    if not HAS_SCIPY:
        return None
    if len(finite_values(a)) == 0 or len(finite_values(b)) == 0:
        return None
    return float(wasserstein_distance(finite_values(a), finite_values(b)))


def mean_std(a: np.ndarray) -> tuple[float, float]:
    values = finite_values(a)
    if len(values) == 0:
        return float("nan"), float("nan")
    return float(values.mean()), float(values.std())


def parse_checkpoint_arg(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        return label.strip(), Path(path)
    path = Path(value)
    return path.parent.name, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run identical eval/plot pipelines for multiple checkpoints and compare."
    )
    parser.add_argument("--config", type=Path, required=True, help="YAML/JSON config path.")
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Checkpoint to evaluate. Repeatable. Use label=path or a bare path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Diagnostic output directory (default: <checkpoint parent>/stage_diagnostic).",
    )
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu.")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=200000,
        help="Selected-row cap passed identically to eval and plot (pre-split).",
    )
    parser.add_argument("--split", choices=("all", "val", "test"), default="test")
    parser.add_argument("--max-x-events", type=int, default=200000)
    parser.add_argument("--max-z-events", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=None, help="Plot subsampling seed.")
    parser.add_argument("--threads", default="auto", help="CPU thread count for plot.py.")
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Reuse existing per-checkpoint summaries instead of rerunning eval/plot.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun eval/plot even when per-checkpoint summaries already exist.",
    )
    return parser.parse_args()


def checkpoint_dirs(output_dir: Path, label: str) -> tuple[Path, Path]:
    return output_dir / label / "plots", output_dir / label / "eval"


def run_pipeline(
    args: argparse.Namespace,
    config: dict[str, Any],
    label: str,
    checkpoint: Path,
) -> None:
    plots_dir, eval_dir = checkpoint_dirs(args.output_dir, label)
    config_path = args.config.expanduser().resolve()
    checkpoint = checkpoint.expanduser().resolve()
    common = [
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint),
        "--device",
        args.device,
    ]

    plot_ready = (plots_dir / "paperstyle_summary.json").exists()
    eval_ready = (eval_dir / "metrics.json").exists()
    if args.skip_plots and plot_ready and eval_ready:
        print(f"[stage_diagnostic] Reusing existing outputs for {label}.")
        return
    if not args.force and plot_ready and eval_ready:
        print(
            f"[stage_diagnostic] Outputs already exist for {label}; "
            "use --force to rerun or --skip-plots to reuse."
        )
        return

    plot_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "plot.py"),
        *common,
        "--output-dir",
        str(plots_dir),
        "--num-samples",
        str(args.num_samples),
        "--split",
        args.split,
        "--max-x-events",
        str(args.max_x_events),
        "--max-z-events",
        str(args.max_z_events),
        "--seed",
        str(config.get("seed", 0) if args.seed is None else args.seed),
        "--threads",
        str(args.threads),
        "--progress-log-steps",
        "1",
    ]
    eval_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "eval.py"),
        *common,
        "--output-dir",
        str(eval_dir),
        "--num-samples",
        str(args.num_samples),
        "--seed",
        str(config.get("seed", 0) if args.seed is None else args.seed),
    ]
    print(f"[stage_diagnostic] Plot pipeline for {label} ({checkpoint})")
    subprocess.run(plot_cmd, check=True)
    print(f"[stage_diagnostic] Eval pipeline for {label} ({checkpoint})")
    subprocess.run(eval_cmd, check=True)


def load_npz_stats(plots_dir: Path) -> dict[str, Any]:
    """Component-level KS/W1/mean/std computed from the saved model outputs."""
    npz_path = plots_dir / "paperstyle_loaded_model_outputs.npz"
    if not npz_path.exists():
        return {}
    with np.load(npz_path, allow_pickle=False) as data:
        x_plot = data["x_plot"]
        x_reco = data["x_reco"]
        x_from_z = data["x_from_z"]
        z_plot = data["z_plot"]
        z_encoded = data["z_encoded"]
        m_x = data["m_x"]
        m_x_reco = data["m_x_reco"]
        m_x_from_z = data["m_x_from_z"]

    stats: dict[str, Any] = {
        "mass": {
            "cycle": {
                "ks": maybe_ks(m_x, m_x_reco),
                "w1": maybe_w1(m_x, m_x_reco),
                "truth_mean": float(np.mean(finite_values(m_x))),
                "truth_std": float(np.std(finite_values(m_x))),
                "pred_mean": float(np.mean(finite_values(m_x_reco))),
                "pred_std": float(np.std(finite_values(m_x_reco))),
            },
            "z_to_x": {
                "ks": maybe_ks(m_x, m_x_from_z),
                "w1": maybe_w1(m_x, m_x_from_z),
                "truth_mean": float(np.mean(finite_values(m_x))),
                "truth_std": float(np.std(finite_values(m_x))),
                "pred_mean": float(np.mean(finite_values(m_x_from_z))),
                "pred_std": float(np.std(finite_values(m_x_from_z))),
            },
        },
        "x_components": {},
        "z_components": {},
    }
    for j in range(8):
        stats["x_components"][f"x_{j:02d}"] = {
            "cycle": {
                "ks": maybe_ks(x_plot[:, j], x_reco[:, j]),
                "w1": maybe_w1(x_plot[:, j], x_reco[:, j]),
                "truth_mean": mean_std(x_plot[:, j])[0],
                "truth_std": mean_std(x_plot[:, j])[1],
                "pred_mean": mean_std(x_reco[:, j])[0],
                "pred_std": mean_std(x_reco[:, j])[1],
            },
            "z_to_x": {
                "ks": maybe_ks(x_plot[:, j], x_from_z[:, j]),
                "w1": maybe_w1(x_plot[:, j], x_from_z[:, j]),
                "pred_mean": mean_std(x_from_z[:, j])[0],
                "pred_std": mean_std(x_from_z[:, j])[1],
            },
        }
        stats["z_components"][f"z_{j:02d}"] = {
            "ks": maybe_ks(z_plot[:, j], z_encoded[:, j]),
            "w1": maybe_w1(z_plot[:, j], z_encoded[:, j]),
            "truth_mean": mean_std(z_plot[:, j])[0],
            "truth_std": mean_std(z_plot[:, j])[1],
            "pred_mean": mean_std(z_encoded[:, j])[0],
            "pred_std": mean_std(z_encoded[:, j])[1],
        }
    return stats


def load_checkpoint_summary(plots_dir: Path, eval_dir: Path) -> dict[str, Any]:
    summary_path = plots_dir / "paperstyle_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing plot summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    eval_path = eval_dir / "metrics.json"
    if eval_path.exists():
        summary["eval_metrics"] = json.loads(eval_path.read_text(encoding="utf-8"))
    summary["extra_component_stats"] = load_npz_stats(plots_dir)
    return summary


def numeric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def delta(value_a: Any, value_b: Any) -> float | None:
    a = numeric(value_a)
    b = numeric(value_b)
    if a is None or b is None:
        return None
    return b - a


def pct_change(value_a: Any, value_b: Any) -> float | None:
    a = numeric(value_a)
    b = numeric(value_b)
    if a is None or b is None or a == 0.0:
        return None
    return 100.0 * (b - a) / abs(a)


def pick_metrics(summary: dict[str, Any]) -> dict[str, float | None]:
    mass = summary.get("mass", {})
    pt = summary.get("pt", {})
    xspace_res = summary.get("xspace_mass_residual", {})
    observables = summary.get("observable_residuals", {})
    z_comps = summary.get("zspace_component_residuals", {})
    extra = summary.get("extra_component_stats", {})
    extra_mass = extra.get("mass", {})
    cycle = extra_mass.get("cycle", {})
    z_to_x = extra_mass.get("z_to_x", {})

    def obs_ks(name: str) -> float | None:
        item = observables.get(name, {})
        return numeric(item.get("ks_statistic"))

    def obs_w1(name: str) -> float | None:
        item = observables.get(name, {})
        return numeric(item.get("w1_distance"))

    def z_comp_metric(field: str) -> float | None:
        values = [
            numeric(item.get(field))
            for item in z_comps.values()
            if numeric(item.get(field)) is not None
        ]
        if not values:
            return None
        return float(np.mean(values))

    def z_comp_extra(field: str, sub: str = "ks") -> float | None:
        values = [
            numeric(item.get(sub))
            for item in extra.get("z_components", {}).values()
            if numeric(item.get(sub)) is not None
        ]
        if not values:
            return None
        return float(np.mean(values))

    return {
        "cycle_mass_std": numeric(mass.get("x_reco_std")),
        "cycle_mass_ks": numeric(cycle.get("ks")),
        "cycle_mass_w1": numeric(cycle.get("w1")),
        "cycle_pt_ks": numeric(pt.get("x_vs_reco_ks_statistic")),
        "z_to_x_mass_std": numeric(mass.get("x_from_z_std")),
        "z_to_x_mass_ks": numeric(z_to_x.get("ks")),
        "z_to_x_mass_w1": numeric(z_to_x.get("w1")),
        "z_to_x_pt_ks": numeric(pt.get("x_vs_zx_ks_statistic")),
        "zspace_mass_ks": numeric(mass.get("zspace_ks_statistic")),
        "xspace_mass_cycle_ks": numeric(xspace_res.get("x_to_z_to_x", {}).get("ks_statistic")),
        "xspace_mass_cycle_w1": numeric(xspace_res.get("x_to_z_to_x", {}).get("w1_distance")),
        "xspace_mass_zx_ks": numeric(xspace_res.get("z_to_x", {}).get("ks_statistic")),
        "xspace_mass_zx_w1": numeric(xspace_res.get("z_to_x", {}).get("w1_distance")),
        "m_obs_ks_zx": obs_ks("m_mumu"),
        "pt_obs_ks_zx": obs_ks("pT_mumu"),
        "zspace_mass_obs_ks": obs_ks("zspace_mass"),
        "z_component_ks_mean": z_comp_metric("ks_statistic"),
        "z_component_w1_mean": z_comp_metric("w1_distance"),
        "z_component_ks_mean_npz": z_comp_extra("z_components", "ks"),
        "z_component_w1_mean_npz": z_comp_extra("z_components", "w1"),
    }


def build_comparison(checkpoints: list[dict[str, Any]]) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    baseline = checkpoints[0]
    baseline_metrics = baseline["metrics"]
    for other in checkpoints[1:]:
        pair = {
            "baseline_label": baseline["label"],
            "comparison_label": other["label"],
            "deltas": {},
            "pct_changes": {},
        }
        for key in baseline_metrics:
            pair["deltas"][key] = delta(baseline_metrics[key], other["metrics"][key])
            pair["pct_changes"][key] = pct_change(baseline_metrics[key], other["metrics"][key])
        comparison[f"{baseline['label']}__vs__{other['label']}"] = pair
    return comparison


def stage3_verdict(pair: dict[str, Any]) -> dict[str, Any]:
    deltas = pair["deltas"]
    changes = pair["pct_changes"]
    cycle_std_change = changes.get("cycle_mass_std")
    cycle_ks_delta = deltas.get("cycle_mass_ks")
    cycle_pt_ks_delta = deltas.get("cycle_pt_ks")
    zx_mass_ks_delta = deltas.get("z_to_x_mass_ks")
    zx_pt_ks_delta = deltas.get("z_to_x_pt_ks")

    cycle_worse = (
        (cycle_std_change is not None and cycle_std_change <= -20.0)
        or (cycle_ks_delta is not None and cycle_ks_delta >= 0.05)
        or (cycle_pt_ks_delta is not None and cycle_pt_ks_delta >= 0.05)
    )
    decoder_fine = not (
        (zx_mass_ks_delta is not None and zx_mass_ks_delta >= 0.05)
        or (zx_pt_ks_delta is not None and zx_pt_ks_delta >= 0.05)
    )
    verdict = bool(cycle_worse and decoder_fine)
    return {
        "stage3_damages_cycle": verdict,
        "criteria": {
            "cycle_mass_std_pct_change": cycle_std_change,
            "cycle_mass_ks_delta": cycle_ks_delta,
            "cycle_pt_ks_delta": cycle_pt_ks_delta,
            "decoder_zx_mass_ks_delta": zx_mass_ks_delta,
            "decoder_zx_pt_ks_delta": zx_pt_ks_delta,
            "rule": (
                "cycle worse if cycle std shrinks >=20%, cycle mass KS rises >=0.05, "
                "or cycle pT KS rises >=0.05; decoder fine if z->x mass/pT KS do not "
                "rise >=0.05. Verdict requires cycle worse AND decoder fine."
            ),
        },
    }


def ablation_verdict(pair: dict[str, Any]) -> dict[str, Any]:
    deltas = pair["deltas"]
    z_ks_delta = deltas.get("zspace_mass_ks")
    z_comp_delta = deltas.get("z_component_ks_mean")
    cycle_ks_delta = deltas.get("cycle_mass_ks")
    cycle_pt_delta = deltas.get("cycle_pt_ks")
    zx_mass_ks_delta = deltas.get("z_to_x_mass_ks")
    zx_pt_delta = deltas.get("z_to_x_pt_ks")

    z_improved = not (
        (z_ks_delta is not None and z_ks_delta >= 0.05)
        or (z_comp_delta is not None and z_comp_delta >= 0.05)
    )
    cycle_improved = not (
        (cycle_ks_delta is not None and cycle_ks_delta >= 0.05)
        and (cycle_pt_delta is not None and cycle_pt_delta >= 0.05)
    )
    decoder_preserved = not (
        (zx_mass_ks_delta is not None and zx_mass_ks_delta >= 0.1)
        or (zx_pt_delta is not None and zx_pt_delta >= 0.1)
    )
    recommend_b = bool(cycle_improved and not z_improved and decoder_preserved)
    return {
        "recommend_v3_6B": recommend_b,
        "criteria": {
            "zspace_mass_ks_delta": z_ks_delta,
            "z_component_ks_mean_delta": z_comp_delta,
            "cycle_mass_ks_delta": cycle_ks_delta,
            "cycle_pt_ks_delta": cycle_pt_delta,
            "z_to_x_mass_ks_delta": zx_mass_ks_delta,
            "z_to_x_pt_ks_delta": zx_pt_delta,
            "rule": (
                "recommend v3.6B only if mass removal improved the cycle, did not "
                "improve z-space closure, and preserved z->x (KS deltas < 0.1)."
            ),
        },
    }


def format_value(value: Any, width: int = 10) -> str:
    if value is None:
        return " " * width
    if isinstance(value, float):
        return f"{value:>{width}.4f}"
    return f"{value:>{width}}"


def write_summary_md(
    args: argparse.Namespace,
    checkpoints: list[dict[str, Any]],
    comparison: dict[str, Any],
    sections: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append("# Stage / ablation diagnostic summary")
    lines.append("")
    lines.append(f"- Config: `{args.config}`")
    lines.append(f"- Split: `{args.split}`, num-samples: {args.num_samples}, "
                 f"max x/z events: {args.max_x_events}/{args.max_z_events}")
    lines.append(f"- Seed: {args.seed if args.seed is not None else 'config seed'}")
    lines.append("")
    lines.append("## Checkpoints")
    lines.append("")
    lines.append("| label | checkpoint | epoch | eval loss |")
    lines.append("|---|---|---|---|")
    for item in checkpoints:
        lines.append(
            f"| {item['label']} | `{item['checkpoint']}` | "
            f"{item['summary'].get('checkpoint_epoch', 'n/a')} | "
            f"{format_value(item['summary'].get('checkpoint_eval_loss'))} |"
        )
    lines.append("")

    key_rows = [
        ("cycle_mass_std", "x->z->x invariant-mass std [GeV]"),
        ("cycle_mass_ks", "x->z->x invariant-mass KS"),
        ("cycle_mass_w1", "x->z->x invariant-mass W1"),
        ("cycle_pt_ks", "x->z->x dilepton pT KS"),
        ("z_to_x_mass_std", "z->x invariant-mass std [GeV]"),
        ("z_to_x_mass_ks", "z->x invariant-mass KS"),
        ("z_to_x_pt_ks", "z->x dilepton pT KS"),
        ("zspace_mass_ks", "z-space mass KS (MG5 z vs x->z)"),
        ("z_component_ks_mean", "mean 8D z-component KS"),
        ("z_component_w1_mean", "mean 8D z-component W1"),
    ]
    for pair_name, pair in comparison.items():
        baseline = checkpoints[0]["metrics"]
        other = next(
            item["metrics"] for item in checkpoints[1:]
            if f"{checkpoints[0]['label']}__vs__{item['label']}" == pair_name
        )
        lines.append(f"## Comparison: {pair_name}")
        lines.append("")
        lines.append("| metric | baseline | comparison | delta | % change |")
        lines.append("|---|---|---|---|---|")
        for key, label in key_rows:
            if key not in baseline:
                continue
            lines.append(
                f"| {label} | {format_value(baseline[key])} | "
                f"{format_value(other[key])} | "
                f"{format_value(pair['deltas'].get(key))} | "
                f"{format_value(pair['pct_changes'].get(key))} |"
            )
        lines.append("")

    for section in sections:
        lines.append(f"## {section['title']}")
        lines.append("")
        for line in section["lines"]:
            lines.append(line)
        lines.append("")

    text = "\n".join(lines).rstrip() + "\n"
    return text


def main() -> None:
    args = parse_args()
    config = resolve_config(load_config(args.config))
    if args.output_dir is None:
        common_parent = Path(args.checkpoint[0].split("=", 1)[-1]).expanduser().resolve().parent
        args.output_dir = common_parent / "stage_diagnostic"
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    entries = [parse_checkpoint_arg(value) for value in args.checkpoint]
    checkpoints: list[dict[str, Any]] = []
    for label, checkpoint in entries:
        if not checkpoint.expanduser().resolve().exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        run_pipeline(args, config, label, checkpoint)
        plots_dir, eval_dir = checkpoint_dirs(args.output_dir, label)
        summary = load_checkpoint_summary(plots_dir, eval_dir)
        checkpoints.append(
            {
                "label": label,
                "checkpoint": str(checkpoint.expanduser().resolve()),
                "summary": summary,
                "metrics": pick_metrics(summary),
            }
        )

    comparison = build_comparison(checkpoints)
    sections: list[dict[str, Any]] = []
    verdicts: dict[str, Any] = {}
    labels = [item["label"].lower() for item in checkpoints]

    if len(checkpoints) >= 2:
        pair = comparison[f"{checkpoints[0]['label']}__vs__{checkpoints[1]['label']}"]
        stage3_pair = (
            any("stage2" in label for label in labels)
            and any(("stage3" in label or "best" in label or "final" in label) for label in labels)
        )
        ablation_pair = any("v3.5" in label for label in labels) and any(
            "v3.6" in label or "no_explicit_mass" in label for label in labels
        )
        if stage3_pair:
            verdict = stage3_verdict(pair)
            verdicts["stage3_effect"] = verdict
            answer = "YES" if verdict["stage3_damages_cycle"] else "NO"
            sections.append(
                {
                    "title": "Stage-3 effect (stage2_end -> best/final stage3)",
                    "lines": [
                        f"Q: Does x -> z -> x closure become substantially worse after "
                        f"entering decoder-only Stage 3 while z -> x improves?",
                        f"A: {answer}",
                        f"  cycle mass std % change: {verdict['criteria']['cycle_mass_std_pct_change']}",
                        f"  cycle mass KS delta: {verdict['criteria']['cycle_mass_ks_delta']}",
                        f"  cycle pT KS delta: {verdict['criteria']['cycle_pt_ks_delta']}",
                        f"  z->x mass KS delta: {verdict['criteria']['decoder_zx_mass_ks_delta']}",
                        f"  z->x pT KS delta: {verdict['criteria']['decoder_zx_pt_ks_delta']}",
                        f"  heuristic rule: {verdict['criteria']['rule']}",
                    ],
                }
            )
        if ablation_pair:
            verdict = ablation_verdict(pair)
            verdicts["v3_6A_effect"] = verdict
            recommend = "YES" if verdict["recommend_v3_6B"] else "NO"
            sections.append(
                {
                    "title": "Mass-removal ablation effect (v3.5 -> v3.6A)",
                    "lines": [
                        "Q1: Did removing explicit invariant-mass losses improve x -> z?",
                        f"    z-space mass KS delta: {verdict['criteria']['zspace_mass_ks_delta']}",
                        f"    mean 8D z-component KS delta: {verdict['criteria']['z_component_ks_mean_delta']}",
                        "Q2: Did it improve x -> z -> x?",
                        f"    cycle mass KS delta: {verdict['criteria']['cycle_mass_ks_delta']}",
                        f"    cycle pT KS delta: {verdict['criteria']['cycle_pt_ks_delta']}",
                        "Q3: How much did z -> x change?",
                        f"    z->x mass KS delta: {verdict['criteria']['z_to_x_mass_ks_delta']}",
                        f"    z->x pT KS delta: {verdict['criteria']['z_to_x_pt_ks_delta']}",
                        "Q4: Did the 8 individual z components improve?",
                        f"    mean 8D z-component KS delta: {verdict['criteria']['z_component_ks_mean_delta']}",
                        "Q5: Recommendation for v3.6B:",
                        f"    {recommend}",
                        f"    heuristic rule: {verdict['criteria']['rule']}",
                    ],
                }
            )

    summary = {
        "config": str(args.config.expanduser().resolve()),
        "output_dir": str(args.output_dir),
        "settings": {
            "split": args.split,
            "num_samples": args.num_samples,
            "max_x_events": args.max_x_events,
            "max_z_events": args.max_z_events,
            "seed": args.seed,
        },
        "checkpoints": [
            {
                "label": item["label"],
                "checkpoint": item["checkpoint"],
                "metrics": item["metrics"],
            }
            for item in checkpoints
        ],
        "comparison": comparison,
        "verdicts": verdicts,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md = write_summary_md(args, checkpoints, comparison, sections)
    (args.output_dir / "summary.md").write_text(md, encoding="utf-8")
    print("Wrote:", args.output_dir / "summary.json")
    print("Wrote:", args.output_dir / "summary.md")


if __name__ == "__main__":
    main()
