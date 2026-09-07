#!/usr/bin/env python
"""Audit a finished paired-benchmark run for pairing leakage.

docs/step3_ppzee_closure_prompt.md section 5: a `residual_rms_vs_identity`
below 1.0 must be treated as leakage until three things come back clean.
This script runs all three against a written run directory, so the answer is
measured rather than argued, and so the audit exists whatever the number turns
out to be. It reads only; it trains nothing and loads no model.

    python scripts_joint/paired_leakage_audit.py --run-dir outputs/cms_Joint/ppzee

Check 1 -- the pairing never entered training. The six split arrays are
    reconstructed from the manifest's own pair index and the source files, and
    the per-column correlation between `x_train` and `z_train` is measured
    against what the same statistic reads on the unshuffled source. A leak
    reads ~0.98; an honest split reads at the finite-sample floor.

Check 2 -- no duplicated rows. Every split array is hashed row by row and the
    train, val and test sets are intersected pairwise, on x and on z.

Check 3 -- the scorer was not handed the truth it is scoring against. The
    prediction archive written by `score_paired_closure.py` is compared with
    the truth it was scored against: if `z_pred` were the truth, or a copy of
    it, the per-event residual would be identically zero. The distance from
    both `z_true` and `x_input` is reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
for directory in (
    REPO_ROOT / "scripts",
    REPO_ROOT / "scripts_sota",
    REPO_ROOT / "scripts_joint",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from paired_data import (  # noqa: E402
    SPLIT_KEYS,
    _per_column_correlation,
    leakage_threshold,
    load_paired_source,
)


def _row_hashes(values: np.ndarray) -> set[bytes]:
    block = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return {hashlib.blake2b(row.tobytes(), digest_size=16).digest() for row in block}


def audit(run_dir: Path, region: str | None) -> dict:
    manifest = json.loads((run_dir / "joint_split_manifest.json").read_text(encoding="utf-8"))
    if region is None:
        paired = [n for n, b in manifest["regions"].items() if "pair_index" in b]
        if len(paired) != 1:
            raise SystemExit(f"Pass --region; paired regions are {paired}")
        region = paired[0]
    block = manifest["regions"][region]["pair_index"]
    reader = block["reader"]

    sources: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split, path in block["sources"].items():
        if path not in {str(p) for p in sources}:
            sources.setdefault(path, load_paired_source(Path(path), **reader))

    arrays: dict[str, np.ndarray] = {}
    for key in SPLIT_KEYS:
        split = key.split("_", 1)[1]
        z_all, x_all = sources[block["sources"][split]]
        index = np.asarray(block["splits"][key]["index"], dtype=np.int64)
        arrays[key] = (z_all if key.startswith("z_") else x_all)[index]

    report: dict = {"region": region, "run_dir": str(run_dir)}

    # --- check 0: the pair index describes the arrays the trainer received ---
    # Without this, every check below could be auditing arrays the run never
    # saw. The manifest independently records a content fingerprint of each
    # split array as the trainer received it; the arrays reconstructed from the
    # pair index must reproduce those fingerprints exactly.
    from g0_contract import array_fingerprint  # noqa: E402

    recorded = manifest["regions"][region]["splits"]
    dtype = manifest["regions"][region]["data_contract"].get("float_type", "float32")
    fingerprints = {}
    for key in SPLIT_KEYS:
        rebuilt = array_fingerprint(arrays[key].astype(np.dtype(dtype), copy=False))
        fingerprints[key] = {
            "recorded": recorded[key],
            "rebuilt": rebuilt,
            "matches": rebuilt == recorded[key],
        }
    report["check_0_pair_index_describes_the_training_arrays"] = {
        "splits": fingerprints,
        "passed": all(entry["matches"] for entry in fingerprints.values()),
        "note": (
            "The manifest's split fingerprints are written from the arrays the "
            "trainer was handed. Rebuilding them from the pair index and the "
            "source files must reproduce them exactly, or the audit below is "
            "auditing the wrong arrays."
        ),
    }

    # --- check 1: the pairing never entered training ------------------------
    z_src, x_src = sources[block["sources"]["train"]]
    source_corr = _per_column_correlation(z_src, x_src)
    checks = {}
    for split in ("train", "val", "test"):
        corr = _per_column_correlation(arrays[f"x_{split}"], arrays[f"z_{split}"])
        n = min(len(arrays[f"x_{split}"]), len(arrays[f"z_{split}"]))
        checks[split] = {
            "events": int(n),
            "max_abs_corr": float(np.max(np.abs(corr))),
            "threshold": float(leakage_threshold(n)),
            "per_column": corr.tolist(),
            "passed": bool(np.max(np.abs(corr)) <= leakage_threshold(n)),
        }
    report["check_1_pairing_not_in_training"] = {
        "source_max_abs_corr": float(np.max(np.abs(source_corr))),
        "source_per_column": source_corr.tolist(),
        "splits": checks,
        "passed": all(entry["passed"] for entry in checks.values()),
        "note": (
            "The source column is the positive control: on the unshuffled "
            "file the rows ARE partners. A guard that cannot fire proves "
            "nothing (CLAUDE.md section 4)."
        ),
    }

    # --- check 2: no duplicated rows ---------------------------------------
    duplicates = {}
    for side in ("x", "z"):
        hashes = {split: _row_hashes(arrays[f"{side}_{split}"]) for split in
                  ("train", "val", "test")}
        for split, values in hashes.items():
            duplicates[f"{side}_{split}_unique_rows"] = len(values)
            duplicates[f"{side}_{split}_total_rows"] = int(len(arrays[f"{side}_{split}"]))
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            duplicates[f"{side}_{left}_vs_{right}_shared_rows"] = len(
                hashes[left] & hashes[right]
            )
    report["check_2_no_duplicated_rows"] = {
        "counts": duplicates,
        "passed": all(
            value == 0 for key, value in duplicates.items() if key.endswith("shared_rows")
        )
        and all(
            duplicates[f"{side}_{split}_unique_rows"]
            == duplicates[f"{side}_{split}_total_rows"]
            for side in ("x", "z")
            for split in ("train", "val", "test")
        ),
    }

    # --- check 3: the scorer was not handed the truth -----------------------
    closure_dir = run_dir / "paired_closure"
    scored = {}
    for split in ("train", "val", "test"):
        pairs_path = closure_dir / f"pairs_{split}.npz"
        pred_path = closure_dir / f"pred_{split}.npz"
        if not (pairs_path.exists() and pred_path.exists()):
            continue
        pairs = np.load(pairs_path)
        pred = np.load(pred_path)
        z_true = np.asarray(pairs["z"], dtype=np.float64)
        x_input = np.asarray(pairs["x"], dtype=np.float64)
        z_pred = np.asarray(pred["z_pred"], dtype=np.float64)
        rms_to_truth = float(np.sqrt(np.mean((z_pred - z_true) ** 2)))
        rms_to_input = float(np.sqrt(np.mean((z_pred - x_input) ** 2)))
        # The x the scorer used must be exactly the x the model was shown.
        matches_split = bool(
            len(x_input) == len(arrays[f"x_{split}"])
            and np.allclose(x_input, arrays[f"x_{split}"], rtol=0, atol=1e-6)
        )
        scored[split] = {
            "events": int(len(z_true)),
            "rms_pred_minus_truth": rms_to_truth,
            "rms_pred_minus_input": rms_to_input,
            "pred_is_a_copy_of_truth": bool(rms_to_truth < 1.0e-9),
            "pred_is_a_copy_of_input": bool(rms_to_input < 1.0e-9),
            "scored_x_matches_the_split_x": matches_split,
            "passed": bool(rms_to_truth > 1.0e-9 and matches_split),
        }
    report["check_3_scorer_not_given_the_truth"] = {
        "splits": scored,
        "passed": all(entry["passed"] for entry in scored.values()) if scored else None,
        "note": (
            "None means score_paired_closure.py has not been run yet. "
            "rms_pred_minus_truth of exactly zero would mean the scorer was "
            "handed its own answer."
        ),
    }

    verdicts = [
        report["check_0_pair_index_describes_the_training_arrays"]["passed"],
        report["check_1_pairing_not_in_training"]["passed"],
        report["check_2_no_duplicated_rows"]["passed"],
        report["check_3_scorer_not_given_the_truth"]["passed"],
    ]
    report["all_checks_passed"] = all(v for v in verdicts if v is not None) and (
        None not in verdicts
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--region", default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    report = audit(run_dir, args.region)

    c0 = report["check_0_pair_index_describes_the_training_arrays"]
    c1 = report["check_1_pairing_not_in_training"]
    print(f"\n=== leakage audit: {run_dir.name} / {report['region']} ===")
    print("check 0 -- the pair index describes the arrays the trainer received")
    for key, entry in sorted(c0["splits"].items()):
        print(f"  {key:<8} manifest sha256 reproduced: {entry['matches']}")
    print(f"  verdict: {'PASS' if c0['passed'] else 'FAIL'}")
    print("check 1 -- the pairing never entered training")
    print(f"  unshuffled source, max |rho|      {c1['source_max_abs_corr']:.4f}"
          "   (the positive control)")
    for split, entry in c1["splits"].items():
        print(f"  {split:<5} n={entry['events']:<7} max |rho| {entry['max_abs_corr']:.4f}"
              f" <= {entry['threshold']:.4f}   {'PASS' if entry['passed'] else 'FAIL'}")
    c2 = report["check_2_no_duplicated_rows"]
    print("check 2 -- no duplicated rows")
    for key, value in sorted(c2["counts"].items()):
        if key.endswith("shared_rows"):
            print(f"  {key:<34} {value}")
    print(f"  verdict: {'PASS' if c2['passed'] else 'FAIL'}")
    c3 = report["check_3_scorer_not_given_the_truth"]
    print("check 3 -- the scorer was not handed the truth")
    if not c3["splits"]:
        print("  not run yet (no paired_closure/pred_*.npz)")
    for split, entry in c3["splits"].items():
        print(f"  {split:<5} rms(pred - truth) {entry['rms_pred_minus_truth']:.4f}"
              f"   rms(pred - input) {entry['rms_pred_minus_input']:.4f}"
              f"   x matches split: {entry['scored_x_matches_the_split_x']}"
              f"   {'PASS' if entry['passed'] else 'FAIL'}")
    print(f"\nALL CHECKS PASSED: {report['all_checks_passed']}")

    out_path = args.out or (run_dir / "paired_leakage_audit.json")
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
