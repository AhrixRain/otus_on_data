#!/usr/bin/env python
"""A0.3 selection study: does native-noise selection pick a different checkpoint?

Read-only. Every run in the record selects its checkpoint at *zero* noise
(``resolve_noise_multipliers(None)``), which structurally penalises a model
whose resolution lives in the decoder's noise channel. A2.3/A2.2 (frozen and
floored kernels) are exactly such models, and A2floor's selected checkpoint is
a documented case of the zero-noise pick being a poor transfer checkpoint
(memory.md Session 54, A0.3).

This study re-runs the trainer's own validation path (via
``scripts_joint/rescore_validation.py``) on every *retained* checkpoint of the
response arms, at zero noise and at each checkpoint's native multipliers, and
asks: **within an arm, does the native-noise argmin differ from the zero-noise
argmin?**

Scope and honest limitations
----------------------------
* Only retained checkpoints are in the pool: per-stage ``best_*`` and
  ``last_*`` files plus ``best_model``/``last_model``. The true epoch-wise
  selection ran over all epochs; a native-noise selection over the full
  history could pick an epoch that was never retained. So a *null* result
  ("same pick") is weaker than it looks, while a *non-null* result ("different
  pick") is a genuine finding: the retained pool already contains a
  native-preferred checkpoint that zero-noise selection discarded.
* Scores are comparable **within** an arm only. Arms with different priors
  (Run E's legacy smeared J/psi prior, for instance) validate on different
  splits, so cross-arm score comparison is meaningless here.
* Nothing is trained. Only new files under the study directory are written.

Usage
-----
python scripts_joint/a03_selection_study.py                # run/skip + aggregate
python scripts_joint/a03_selection_study.py --aggregate-only
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = REPO_ROOT / "outputs" / "cms_Joint" / "a03_selection_study"

# Response arms that keep retained checkpoints. Order is the report order.
RUNS = [
    "Run_E",
    "Run_F",
    "Run_G_tf32",
    "Run_H",
    "Run_H_anchor",
    "Run_H_fix",
    "Run_H_cycleNoNoise",
    "Run_H_A2frozen",
    "Run_H_A2floor",
]

# Arms whose resolved config no longer resolves on this machine. Run E's config
# predates the documented move of the legacy priors into data/legacy/, and
# rescore_validation.py exposes no prior/config override, so it cannot be
# re-scored without a tool change. Recorded rather than silently skipped.
NOT_RESOLVABLE = {
    "Run_E": (
        "not re-scorable: `config.resolved.json` points at "
        "`data/cms_jpsi_mumu_mg5_8tev_mixed_ptj5.hdf5`, which was moved to "
        "`data/legacy/` (see `data/legacy/README.md`); the tool exposes no "
        "prior-path override"
    ),
}


def checkpoint_paths(run: str) -> list[Path]:
    run_dir = REPO_ROOT / "outputs" / "cms_Joint" / run
    return sorted(p for p in run_dir.glob("*.pt") if p.is_file())


def report_path(run: str, checkpoint: Path) -> Path:
    return STUDY_DIR / f"{run}__{checkpoint.stem}.json"


def run_one(run: str, checkpoint: Path, force: bool) -> Path:
    out = report_path(run, checkpoint)
    if out.exists() and not force:
        return out
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts_joint" / "rescore_validation.py"),
        "--checkpoint",
        str(checkpoint),
        "--noise",
        "native",
        "--compare-zero",
        "--device",
        "cpu",
        "--out",
        str(out),
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  !! FAILED {run}/{checkpoint.name}")
        print((proc.stderr or proc.stdout or "").strip()[-600:])
        return out
    return out


def collect() -> list[dict]:
    rows: list[dict] = []
    for run in RUNS:
        for path in checkpoint_paths(run):
            rep = report_path(run, path)
            if not rep.exists():
                continue
            d = json.loads(rep.read_text(encoding="utf-8"))
            zero = d["runs"].get("zero") or {}
            native = d["runs"].get("native") or {}
            mult = d.get("checkpoint_noise_multipliers") or {}
            rows.append(
                {
                    "run": run,
                    "checkpoint": path.name,
                    "global_epoch": d.get("checkpoint_global_epoch"),
                    "stage": d.get("checkpoint_stage"),
                    "core_mult": mult.get("decoder_core", mult.get("core")),
                    "tail_mult": mult.get("decoder_tail", mult.get("tail")),
                    "zero_score": (zero.get("selection") or {}).get(
                        "raw_worst_region_score"
                    ),
                    "native_score": (native.get("selection") or {}).get(
                        "raw_worst_region_score"
                    ),
                }
            )
    return rows


def write_csv(rows: list[dict]) -> Path:
    out = STUDY_DIR / "a03_selection_summary.csv"
    fields = [
        "run",
        "checkpoint",
        "global_epoch",
        "stage",
        "core_mult",
        "tail_mult",
        "zero_score",
        "native_score",
    ]
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return out


def write_report(rows: list[dict]) -> Path:
    out = STUDY_DIR / "REPORT.md"
    lines = [
        "# A0.3 selection study — zero-noise vs native-noise checkpoint choice",
        "",
        "*Read-only. Generated by `scripts_joint/a03_selection_study.py`.*",
        "Scores are `raw_worst_region_score` from the trainer's own validation",
        "path; lower is better. Comparable within an arm only.",
        "",
        "| run | retained ckpts | zero-noise pick (ep) | zero score | native-noise pick (ep) | native score | same pick? |",
        "|---|---|---|---|---|---|---|",
    ]
    for run in RUNS:
        sub = [r for r in rows if r["run"] == run and r["zero_score"] is not None]
        if not sub:
            note = NOT_RESOLVABLE.get(run, "no retained checkpoints scored")
            lines.append(f"| `{run}` | 0 | — | — | — | — | {note} |")
            continue
        z = min(sub, key=lambda r: r["zero_score"])
        n = min(sub, key=lambda r: r["native_score"])
        same = (z["checkpoint"] == n["checkpoint"]) or (
            z["global_epoch"] == n["global_epoch"]
        )
        lines.append(
            "| `{run}` | {k} | {zc} (g{zg}) | {zs:.3f} | {nc} (g{ng}) | {ns:.3f} | {same} |".format(
                run=run,
                k=len(sub),
                zc=z["checkpoint"].replace("_" + run, ""),
                zg=z["global_epoch"],
                zs=z["zero_score"],
                nc=n["checkpoint"].replace("_" + run, ""),
                ng=n["global_epoch"],
                ns=n["native_score"],
                same="**same**" if same else "**DIFFERENT**",
            )
        )

    lines += ["", "## Per-checkpoint rows", ""]
    for run in RUNS:
        sub = [r for r in rows if r["run"] == run]
        if not sub:
            continue
        lines += [
            f"### `{run}`",
            "",
            "| checkpoint | global ep | stage | dec (core/tail) | zero score | native score |",
            "|---|---|---|---|---|---|",
        ]
        for r in sorted(sub, key=lambda r: (r["global_epoch"] or 0)):
            zs = f"{r['zero_score']:.3f}" if r["zero_score"] is not None else "—"
            ns = f"{r['native_score']:.3f}" if r["native_score"] is not None else "—"
            lines.append(
                f"| `{r['checkpoint']}` | {r['global_epoch']} | {r['stage']} | "
                f"{r['core_mult']}/{r['tail_mult']} | {zs} | {ns} |"
            )
        lines.append("")

    lines += [
        "## Limitations",
        "",
        "* The pool is *retained* checkpoints only. The real selection ran over",
        "  every epoch; a native-noise selection could prefer an epoch that was",
        "  never retained, so a null result is weaker than it appears. A",
        "  non-null result is a genuine finding: the native-preferred checkpoint",
        "  was already on disk and zero-noise selection discarded it.",
        "* Cross-arm score comparison is invalid (different priors/splits).",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit-checkpoints", type=int, default=None)
    args = ap.parse_args()

    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    if not args.aggregate_only:
        for run in RUNS:
            paths = checkpoint_paths(run)
            if args.limit_checkpoints:
                paths = paths[: args.limit_checkpoints]
            print(f"{run}: {len(paths)} checkpoints")
            for path in paths:
                existed = report_path(run, path).exists()
                run_one(run, path, args.force)
                if not existed:
                    print(f"  scored {path.name}")

    rows = collect()
    write_csv(rows)
    rep = write_report(rows)
    print(f"\n{len(rows)} checkpoint rows -> {rep}")
    for run in RUNS:
        sub = [r for r in rows if r["run"] == run and r["zero_score"] is not None]
        if not sub:
            continue
        z = min(sub, key=lambda r: r["zero_score"])
        n = min(sub, key=lambda r: r["native_score"])
        flag = "same" if z["global_epoch"] == n["global_epoch"] else "DIFFERENT"
        print(
            f"  {run:22s} zero g{z['global_epoch']:<4} native g{n['global_epoch']:<4} {flag}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
