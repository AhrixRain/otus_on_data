#!/usr/bin/env python
"""Side-by-side comparison of the Run H drift-control arms.

Builds three tables, all artifact-measured from the run directories:

1. in-domain  - per-stage selection-score / eval-loss / train-loss summary and
                the global best / last epochs, read from ``history.json``;
2. Upsilon    - per-state decoded mass shape from a frozen, read-only decode of
                the Upsilon prior: median, std, skew, q16-median and the shift
                against the CMS three-peak fit;
3. conditional - the per-pair-pT slice ratios (decode W1 / floor) from
                ``slice_resolved_diagnostic.py``.

Everything is discovery-driven, so the same command compares ``Run_H``,
``Run_H_anchor`` and ``Run_H_fix`` (or any later arms) without edits.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import torch


def find_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "scripts_joint").is_dir() and (candidate / "scripts_sota").is_dir():
            return candidate
    raise RuntimeError("Could not locate the OTUS repository root")


REPO_ROOT = find_repo_root()
for directory in (
    REPO_ROOT / "scripts",
    REPO_ROOT / "scripts_sota",
    REPO_ROOT / "scripts_joint",
    REPO_ROOT / "scripts_joint" / "upsilon",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from decode_prior import load_frozen_model  # noqa: E402


STATE_COMPONENTS = ("upsilon1s", "upsilon2s", "upsilon3s")
COMPONENT_MAP = {"upsilon1s": 0, "upsilon2s": 1, "upsilon3s": 2}
CMS_FIT_MASS_GEV = {
    "upsilon1s": 9.445066420298902,
    "upsilon2s": 10.016273935221268,
    "upsilon3s": 10.34168938611508,
}
NOISE_COMPONENTS = ("encoder_core", "encoder_tail", "decoder_core", "decoder_tail")


def invariant_mass(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    energy = values[:, 3] + values[:, 7]
    momentum = values[:, :3] + values[:, 4:7]
    return np.sqrt(np.maximum(energy * energy - np.sum(momentum * momentum, axis=1), 0.0))


def load_prior(prior: Path, max_events: int | None, seed: int):
    with h5py.File(prior, "r") as source:
        z = np.asarray(source["FDL/zData"][:], dtype=np.float32)
        component_id = np.asarray(source["FDL/component_id"][:])
    if max_events is not None and max_events < len(z):
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(z), size=int(max_events), replace=False)
        chosen.sort()
        z = z[chosen]
        component_id = component_id[chosen]
    return z, component_id


def native_noise(checkpoint: dict) -> dict[str, float]:
    stored = checkpoint.get("noise_multipliers") or {}
    return {
        "encoder_core": float(stored.get("encoder_core", stored.get("core", 1.0))),
        "encoder_tail": float(stored.get("encoder_tail", stored.get("tail", 0.0))),
        "decoder_core": float(stored.get("decoder_core", stored.get("core", 1.0))),
        "decoder_tail": float(stored.get("decoder_tail", stored.get("tail", 0.0))),
    }


@torch.inference_mode()
def decode_prior(model, z: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    outputs = []
    for start in range(0, len(z), batch_size):
        block = torch.as_tensor(
            np.ascontiguousarray(z[start : start + batch_size]),
            dtype=torch.float32,
            device=device,
        )
        outputs.append(model.decode(block).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def shape_table(x: np.ndarray, component_id: np.ndarray) -> dict[str, dict[str, float]]:
    masses = invariant_mass(x)
    table: dict[str, dict[str, float]] = {}
    for name, identifier in COMPONENT_MAP.items():
        sample = masses[component_id == identifier]
        mean = float(np.mean(sample))
        std = float(np.std(sample))
        median = float(np.median(sample))
        q16 = float(np.quantile(sample, 0.16))
        q84 = float(np.quantile(sample, 0.84))
        q025 = float(np.quantile(sample, 0.025))
        q975 = float(np.quantile(sample, 0.975))
        table[name] = {
            "events": int(len(sample)),
            "mean_gev": mean,
            "median_gev": median,
            "std_gev": std,
            "skew": float(np.mean(((sample - mean) / std) ** 3)) if std > 0 else 0.0,
            "q16_gev": q16,
            "q84_gev": q84,
            "q025_gev": q025,
            "q975_gev": q975,
            "median_minus_cms_gev": median - CMS_FIT_MASS_GEV[name],
            "q16_minus_median_gev": q16 - median,
            "mean_minus_median_gev": mean - median,
        }
    return table


def in_domain_summary(run_dir: Path) -> dict:
    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    stages: "OrderedDict[str, list]" = OrderedDict()
    for row in history:
        stages.setdefault(str(row["stage"]), []).append(row)
    stage_rows = []
    for name, rows in stages.items():
        evals = [row for row in rows if row.get("eval_loss") is not None]
        scores = [float(row["joint_selection"]["selection_score"]) for row in evals]
        best = min(evals, key=lambda row: row["joint_selection"]["selection_score"]) if evals else None
        train = [float(row["train"]["loss"]) for row in rows]
        stage_rows.append(
            {
                "stage": name,
                "epochs": len(rows),
                "evals": len(evals),
                "score_mean": float(np.mean(scores)) if scores else None,
                "score_median": float(np.median(scores)) if scores else None,
                "score_min": float(np.min(scores)) if scores else None,
                "score_max": float(np.max(scores)) if scores else None,
                "best_epoch": int(best["global_epoch"]) if best else None,
                "best_score": float(best["joint_selection"]["selection_score"]) if best else None,
                "eval_loss_mean": float(np.mean([row["eval_loss"] for row in evals])) if evals else None,
                "eval_loss_min": float(np.min([row["eval_loss"] for row in evals])) if evals else None,
                "train_loss_first5": float(np.mean(train[:5])) if train else None,
                "train_loss_last5": float(np.mean(train[-5:])) if train else None,
                "anchor_loss_first": next(
                    (float(row["train"]["mean_map_anchor"]) for row in rows
                     if row["train"].get("mean_map_anchor") is not None), None
                ),
                "anchor_loss_last": next(
                    (float(row["train"]["mean_map_anchor"]) for row in reversed(rows)
                     if row["train"].get("mean_map_anchor") is not None), None
                ),
            }
        )
    last_epoch = int(history[-1]["global_epoch"]) if history else None
    best_epoch = None
    best_score = None
    for row in history:
        selection = row.get("joint_selection") or {}
        if not selection:
            continue
        score = float(selection["selection_score"])
        if best_score is None or score < best_score:
            best_score = score
            best_epoch = int(row["global_epoch"])
    return {
        "stages": stage_rows,
        "global_best_epoch": best_epoch,
        "global_best_score": best_score,
        "last_epoch": last_epoch,
    }


def checkpoint_list(run_dir: Path) -> list[tuple[str, Path, int | None]]:
    entries: list[tuple[str, Path, int | None]] = []
    candidates: list[Path] = []
    for name in ("best_model.pt", "last_model.pt"):
        if (run_dir / name).exists():
            candidates.append(run_dir / name)
    for pattern in ("best_*.pt", "last_*.pt"):
        for path in sorted(run_dir.glob(pattern)):
            if path.name in {"best_model.pt", "last_model.pt"}:
                continue
            candidates.append(path)
    for path in candidates:
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            epoch = checkpoint.get("global_epoch")
            stage = checkpoint.get("stage")
            stage_name = stage.get("name") if isinstance(stage, dict) else stage
            label = f"{stage_name or '?'}@ep{epoch}"
        except Exception:
            label = path.stem
        entries.append((label, path, None))
    return entries


def run_slice_diagnostic(script: Path, checkpoint: Path, output_dir: Path, device: str) -> dict | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint", str(checkpoint),
            "--output-dir", str(output_dir),
            "--device", device,
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  slice diagnostic failed for {checkpoint.name}: {result.stderr.strip()[-300:]}")
        return None
    payload = json.loads((output_dir / "slice_diagnostic.json").read_text(encoding="utf-8"))
    ratios: dict[str, list[float]] = {}
    for region, block in payload.get("regions", {}).items():
        ratios[region] = [
            float(slice_row["decode_mass_w1"] / slice_row["decode_mass_w1_floor"])
            for slice_row in block["slices"]
        ]
    return {"ratios": ratios, "epoch": payload.get("checkpoint_global_epoch")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=["Run_H", "Run_H_anchor", "Run_H_fix"])
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs" / "cms_Joint")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--prior",
        type=Path,
        default=REPO_ROOT / "data" / "upsilon_prior_continuumReweighted.hdf5",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--max-events", type=int, default=400000)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--skip-upsilon", action="store_true")
    parser.add_argument("--skip-slices", action="store_true")
    parser.add_argument(
        "--slice-script",
        type=Path,
        default=REPO_ROOT / "scripts_joint" / "slice_resolved_diagnostic.py",
    )
    args = parser.parse_args()

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else args.output_root / "run_comparison"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    payload: dict[str, dict] = {"runs": {}, "events_decoded": None}
    prior_z = prior_component = None
    if not args.skip_upsilon:
        prior_z, prior_component = load_prior(args.prior, args.max_events, args.seed)
        payload["events_decoded"] = int(len(prior_z))

    for run_name in args.runs:
        run_dir = args.output_root / run_name
        if not (run_dir / "history.json").exists():
            print(f"skip missing run {run_dir}")
            continue
        print(f"=== {run_name}")
        entry: dict = {"in_domain": in_domain_summary(run_dir), "checkpoints": {}}
        for label, path, _ in checkpoint_list(run_dir):
            checkpoint_payload: dict = {"label": label, "path": str(path)}
            try:
                model, checkpoint, _ = load_frozen_model(path, device)
            except Exception as error:
                print(f"  {path.name}: load failed ({error})")
                continue
            model.eval()
            noise = native_noise(checkpoint)
            checkpoint_payload["global_epoch"] = checkpoint.get("global_epoch")
            checkpoint_payload["noise"] = noise
            if not args.skip_upsilon:
                model.set_component_noise_multipliers(**noise)
                decoded = decode_prior(model, prior_z, args.batch_size, device)
                checkpoint_payload["upsilon_shape"] = shape_table(decoded, prior_component)
                one = checkpoint_payload["upsilon_shape"]["upsilon1s"]
                print(
                    "  %-38s 1S median-CMS %+6.1f MeV  std %.3f  skew %+.2f"
                    % (label, one["median_minus_cms_gev"] * 1000.0, one["std_gev"], one["skew"])
                )
            if not args.skip_slices:
                slice_dir = output_dir / "slices" / run_name / path.stem
                checkpoint_payload["slice_ratios"] = run_slice_diagnostic(
                    args.slice_script, path, slice_dir, args.device
                )
            entry["checkpoints"][path.stem] = checkpoint_payload
        payload["runs"][run_name] = entry

    (output_dir / "run_comparison.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8"
    )
    write_report(output_dir / "COMPARISON.md", payload)
    print(f"wrote {output_dir / 'COMPARISON.md'}")
    return 0


def _fmt(value, spec="%.3f"):
    if value is None:
        return "-"
    try:
        return spec % value
    except (TypeError, ValueError):
        return str(value)


def write_report(path: Path, payload: dict) -> None:
    lines = ["# Drift-control arm comparison", ""]
    if payload.get("events_decoded"):
        lines.append(f"- Upsilon events decoded per checkpoint: {payload['events_decoded']}")
        lines.append("")

    lines.append("## In-domain (history.json)")
    lines.append("")
    lines.append("| run | stage | epochs | evals | score mean | score median | score min | best ep | best score | eval_loss mean | train first5 -> last5 | anchor first -> last |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for run_name, entry in payload["runs"].items():
        for stage in entry["in_domain"]["stages"]:
            lines.append(
                "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s -> %s | %s -> %s |"
                % (
                    run_name,
                    stage["stage"],
                    stage["epochs"],
                    stage["evals"],
                    _fmt(stage["score_mean"]),
                    _fmt(stage["score_median"]),
                    _fmt(stage["score_min"]),
                    stage["best_epoch"],
                    _fmt(stage["best_score"]),
                    _fmt(stage["eval_loss_mean"]),
                    _fmt(stage["train_loss_first5"], "%.4f"),
                    _fmt(stage["train_loss_last5"], "%.4f"),
                    _fmt(stage["anchor_loss_first"], "%.3e"),
                    _fmt(stage["anchor_loss_last"], "%.3e"),
                )
            )
        lines.append(
            "| %s | **global** | - | - | - | - | - | best ep %s | %s | - | last ep %s | - |"
            % (
                run_name,
                entry["in_domain"]["global_best_epoch"],
                _fmt(entry["in_domain"]["global_best_score"]),
                entry["in_domain"]["last_epoch"],
            )
        )
    lines.append("")

    lines.append("## Upsilon decoded mass shape (frozen prior, native noise)")
    lines.append("")
    lines.append("| run | checkpoint | ep | 1S med-CMS [MeV] | 1S std | 1S skew | 1S q16-med [MeV] | 2S med-CMS [MeV] | 3S med-CMS [MeV] |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for run_name, entry in payload["runs"].items():
        for checkpoint in entry["checkpoints"].values():
            shape = checkpoint.get("upsilon_shape")
            if not shape:
                continue
            one, two, three = (shape[name] for name in STATE_COMPONENTS)
            lines.append(
                "| %s | %s | %s | %+.0f | %.3f | %+.2f | %+.0f | %+.0f | %+.0f |"
                % (
                    run_name,
                    checkpoint["label"],
                    checkpoint.get("global_epoch"),
                    one["median_minus_cms_gev"] * 1000.0,
                    one["std_gev"],
                    one["skew"],
                    one["q16_minus_median_gev"] * 1000.0,
                    two["median_minus_cms_gev"] * 1000.0,
                    three["median_minus_cms_gev"] * 1000.0,
                )
            )
    lines.append("")

    lines.append("## Conditional slice ratios (decode mass W1 / floor, noise 0)")
    lines.append("")
    lines.append("| run | checkpoint | J/psi s0..s3 | Z s0..s3 |")
    lines.append("|---|---|---|---|")
    for run_name, entry in payload["runs"].items():
        for checkpoint in entry["checkpoints"].values():
            slices = checkpoint.get("slice_ratios")
            if not slices:
                continue
            jpsi = " / ".join(f"{value:.2f}" for value in slices["ratios"].get("jpsi", []))
            z = " / ".join(f"{value:.2f}" for value in slices["ratios"].get("z", []))
            lines.append(
                "| %s | %s | %s | %s |"
                % (run_name, checkpoint["label"], jpsi or "-", z or "-")
            )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
