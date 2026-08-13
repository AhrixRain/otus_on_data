#!/usr/bin/env python
"""Post-training comparison of the five v3.7 lambda runs.

Reads the per-checkpoint `v37_eval_summary.json` files produced by
scripts/eval_v37.py and emits:

  * v37_lambda_comparison.json   (machine-readable rows)
  * v37_lambda_comparison.md     (human-readable Markdown table)

All rows use the same fixed test sample and the same SW projections because
every run was evaluated by eval_v37.py with the shared v3.7 seed/slice setup.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


CHECKPOINT_CHOICES = ("best_model", "best_z_prior", "best_cycle", "last_model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        type=Path,
        required=True,
        help="v3.7 run directories (e.g. outputs/cms_JpsiDoubleMuons/v3.7_lambda0p1 ...).",
    )
    parser.add_argument(
        "--checkpoint",
        choices=CHECKPOINT_CHOICES,
        default="best_model",
        help="Which checkpoint family to compare (default: best_model = combined).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write the comparison files (default: parent of the first run dir).",
    )
    return parser.parse_args()


def num(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    if abs(value) >= 1e3 or (value != 0.0 and abs(value) < 1e-3):
        return f"{value:.3e}"
    return f"{value:.4f}"


def main() -> int:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for run_dir in args.run_dirs:
        summary_path = run_dir / "v37_eval" / args.checkpoint / "v37_eval_summary.json"
        if not summary_path.exists():
            missing.append(str(run_dir))
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        encoder = summary.get("encoder_only", {})
        cycle = summary.get("cycle", {})
        generator = summary.get("generator", {})
        rows.append(
            {
                "run_dir": str(run_dir),
                "lambda": num(summary.get("lambda")),
                "best_epoch": summary.get("epoch"),
                "z_sw": num(encoder.get("z_sw")),
                "z_mean_ks": num(encoder.get("z_mean_ks")),
                "z_max_ks": num(encoder.get("z_max_ks")),
                "z_mass_ks": num(encoder.get("z_mass_ks")),
                "cycle_mass_ks": num(cycle.get("mass_ks")),
                "cycle_pt_ks": num(cycle.get("pt_ks")),
                "cycle_reconstruction": num(cycle.get("reco_mse_raw")),
                "z_to_x_mass_ks": num(generator.get("mass_ks")),
                "z_to_x_pt_ks": num(generator.get("pt_ks")),
            }
        )
    rows.sort(key=lambda row: (row["lambda"] is None, row["lambda"] or float("inf")))

    if args.output_dir is None:
        output_dir = args.run_dirs[0].resolve().parent if args.run_dirs else Path.cwd()
    else:
        output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "v37_lambda_comparison.json"
    md_path = output_dir / "v37_lambda_comparison.md"
    json_path.write_text(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "rows": rows,
                "missing_runs": missing,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    lines = [
        f"# v3.7 lambda comparison (`{args.checkpoint}` checkpoints)",
        "",
        "Metrics are computed by `scripts/eval_v37.py` on the fixed held-out",
        "test sample with shared seeds and SW projections. Do not rank solely",
        "by generated mass: the useful tradeoff is latent alignment vs cycle",
        "quality vs generator quality together.",
        "",
        "| lambda | best epoch | z SW | z mean KS | z max KS | z mass KS |"
        " cycle mass KS | cycle pT KS | cycle reconstruction | z->x mass KS |"
        " z->x pT KS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {lam} | {epoch} | {z_sw} | {z_mean_ks} | {z_max_ks} |"
            " {z_mass_ks} | {cycle_mass_ks} | {cycle_pt_ks} |"
            " {cycle_recon} | {gen_mass_ks} | {gen_pt_ks} |".format(
                lam=fmt(row["lambda"]),
                epoch=("n/a" if row["best_epoch"] is None else int(row["best_epoch"])),
                z_sw=fmt(row["z_sw"]),
                z_mean_ks=fmt(row["z_mean_ks"]),
                z_max_ks=fmt(row["z_max_ks"]),
                z_mass_ks=fmt(row["z_mass_ks"]),
                cycle_mass_ks=fmt(row["cycle_mass_ks"]),
                cycle_pt_ks=fmt(row["cycle_pt_ks"]),
                cycle_recon=fmt(row["cycle_reconstruction"]),
                gen_mass_ks=fmt(row["z_to_x_mass_ks"]),
                gen_pt_ks=fmt(row["z_to_x_pt_ks"]),
            )
        )
    if missing:
        lines.extend(["", "Missing run summaries (not evaluated yet):"])
        lines.extend(f"- `{path}`" for path in missing)
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print("Wrote:", json_path.resolve())
    print("Wrote:", md_path.resolve())
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
