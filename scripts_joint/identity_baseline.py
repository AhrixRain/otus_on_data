#!/usr/bin/env python
"""Score a finished joint run against the identity map and the noise floor.

Motivation
----------
memory.md section 7.2: on the smeared J/psi prior the map ``z~ = x`` -- no
model at all -- passes every strict latent target. Any gate number from a run
that predates the reference metrics is therefore uninterpretable on its own.
This tool reconstructs those references for a completed run WITHOUT the model
and without a GPU: the identity map's output is the model's input, and the
finite-sample floor comes from two draws of the target distribution, so both
are functions of the cached data alone.

It reproduces the run's exact sampling geometry rather than approximating it:
the per-epoch validation seed (``config.seed + global_epoch``, offset by
``1000 * region_index``) and the final-evaluation seed are recomputed from the
resolved config, so the identity and floor numbers are the ones the patched
``joint_metrics`` would have written had it existed when the run started.

Usage
-----
    python scripts_joint/identity_baseline.py --run-dir outputs/cms_Joint/AB_narrow
    python scripts_joint/identity_baseline.py --run-dir <dir> --out <file.json>

Nothing is written unless ``--out`` is given.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (REPO_ROOT / "scripts", REPO_ROOT / "scripts_joint"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from identity_reference import (  # noqa: E402
    GAUGED_SUFFIXES,
    equal_subset,
    gauge,
    reference_block,
)
from physics import invariant_mass_np  # noqa: E402


def _mass_fn(masses, *, stable: bool):
    def inner(values: np.ndarray) -> np.ndarray:
        return invariant_mass_np(values, daughter_masses=masses, stable=stable)

    return inner


def _load_cache(run_dir: Path, region: str, provenance: dict[str, Any]) -> dict:
    """Locate the region cache this run actually used.

    ``provenance.json`` records an absolute Windows path; that is not portable,
    so the cache key is used to rebuild the path relative to the run directory,
    which is where ``.region_cache`` always lives for the series.
    """
    entry = (provenance.get("data_cache") or {}).get(region) or {}
    key = entry.get("key")
    if not key:
        raise SystemExit(f"provenance.json has no cache key for region {region!r}")
    path = run_dir.parent / ".region_cache" / region / f"selected_split_{key}.npz"
    if not path.exists():
        recorded = entry.get("path")
        if recorded and Path(recorded).exists():
            path = Path(recorded)
        else:
            raise SystemExit(f"Region cache not found for {region!r}: {path}")
    return np.load(path)


def _model_metrics_from_row(row: dict, region: str) -> dict[str, float]:
    return dict((row.get("region_validation") or {}).get(region) or {})


def region_report(
    run_dir: Path,
    config: dict[str, Any],
    provenance: dict[str, Any],
    history: list[dict],
    evaluation: dict[str, Any] | None,
    region: str,
    offset: int,
) -> dict[str, Any]:
    cache = _load_cache(run_dir, region, provenance)
    masses = config["model"].get("daughter_masses")
    latent_mass = _mass_fn(masses, stable=False)
    direct_mass = _mass_fn(masses, stable=True)

    loaders = config.get("loaders", {})
    validation_events = loaders.get("validation_events", 8192)
    draws = max(1, int(loaders.get("validation_draws", 2)))
    base_seed = int(config.get("seed", 0))

    selection = config["regions"][region].get("selection", {})
    gates = selection.get("gates", {})
    targets = selection.get("targets", {})

    epochs: list[dict[str, Any]] = []
    for row in history:
        if "region_validation" not in row:
            continue
        metrics = _model_metrics_from_row(row, region)
        if not metrics:
            continue
        region_seed = base_seed + int(row["global_epoch"]) + 1000 * offset
        x, z = equal_subset(
            cache["x_val"],
            cache["z_val"],
            max_events=validation_events,
            seed=region_seed,
        )
        block = reference_block(
            "latent",
            real_values=z,
            identity_values=x,
            reference_pool=cache["z_val"],
            model_metrics=metrics,
            mass_fn=latent_mass,
            seed=region_seed + 500,
        )
        block.update(
            reference_block(
                "direct",
                real_values=np.tile(x, (draws, 1)),
                identity_values=np.tile(z, (draws, 1)),
                reference_pool=cache["x_val"],
                model_metrics=metrics,
                mass_fn=direct_mass,
                seed=region_seed + 600,
            )
        )
        epochs.append(
            {
                "global_epoch": int(row["global_epoch"]),
                "stage": row.get("stage"),
                "core_noise_multiplier": row.get("core_noise_multiplier"),
                "gate_passed": bool(
                    ((row.get("joint_selection") or {}).get("regions") or {})
                    .get(region, {})
                    .get("gate_passed", False)
                ),
                "model": {
                    key: metrics[key]
                    for key in metrics
                    if any(key.endswith(suffix) for suffix in GAUGED_SUFFIXES)
                },
                "reference": block,
            }
        )

    # Final-evaluation geometry, to sit next to joint_evaluation.json.
    final_cfg = config.get("final_evaluation", {})
    eval_seed = int(final_cfg.get("seed", 20260821)) + 1000 * offset
    x_t, z_t = equal_subset(
        cache["x_test"],
        cache["z_test"],
        max_events=final_cfg.get("max_events"),
        # evaluate_bidirectional_model draws the equal-count pair ONCE with the
        # outer seed and then hands those arrays to distribution_report, whose
        # own reseed is a permutation of the same events. Matching the outer
        # seed therefore reproduces the exact event set that was scored.
        seed=eval_seed,
    )
    model_x_to_z = {}
    if evaluation:
        directions = (
            ((evaluation.get("regions") or {}).get(region) or {}).get("directions") or {}
        )
        mass_block = (directions.get("x_to_z") or {}).get("mass") or {}
        model_x_to_z = {
            "latent_mass_ks": mass_block.get("ks"),
            "latent_mass_w1_gev": mass_block.get("w1_gev"),
        }
    test_block = reference_block(
        "latent",
        real_values=z_t,
        identity_values=x_t,
        reference_pool=cache["z_test"],
        model_metrics={
            "latent_mass_ks": model_x_to_z.get("latent_mass_ks", float("nan")),
            "latent_mass_w1_gev": model_x_to_z.get("latent_mass_w1_gev", float("nan")),
            "latent_mass_width_rel_error": float("nan"),
            "latent_pair_pt_ks": float("nan"),
        },
        mass_fn=latent_mass,
        seed=eval_seed + 500,
    )
    return {
        "gates": gates,
        "targets": targets,
        "epochs": epochs,
        "final_evaluation": {
            "events_per_distribution": int(len(z_t)),
            "model": model_x_to_z,
            "reference": test_block,
        },
    }


def _fmt(value) -> str:
    if value is None:
        return "     -"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value):
        return "   inf"
    return f"{value:6.4f}" if abs(value) < 100 else f"{value:6.1f}"


def print_report(region: str, report: dict[str, Any]) -> None:
    gates = report["gates"]
    targets = report["targets"]
    print(f"\n=== region {region} ===")
    final = report["final_evaluation"]
    ref = final["reference"]
    print(
        f"final evaluation, x->z, n={final['events_per_distribution']} "
        f"(floor from {'disjoint draws' if ref['latent_floor_disjoint'] else 'rescaled halves'})"
    )
    header = f"{'metric':30s}{'model':>8}{'identity':>10}{'floor':>8}{'vs_id':>8}{'headroom':>10}{'gate':>8}{'target':>8}"
    print(header)
    print("-" * len(header))
    for suffix in ("mass_ks", "mass_w1_gev"):
        key = f"latent_{suffix}"
        model = final["model"].get(key)
        identity = ref[f"latent_identity_{suffix}"]
        floor = ref[f"latent_floor_{suffix}"]
        vs = gauge(model, identity, floor) if model is not None else None
        print(
            f"{key:30s}{_fmt(model):>8}{_fmt(identity):>10}{_fmt(floor):>8}"
            f"{_fmt(vs):>8}{_fmt(ref[f'latent_{suffix}_headroom']):>10}"
            f"{_fmt(gates.get(key)):>8}{_fmt(targets.get(key)):>8}"
        )
        if identity <= float(gates.get(key, np.inf)):
            print(
                f"    NOTE: the identity map passes this gate "
                f"({_fmt(identity)} <= {_fmt(gates.get(key))}). "
                "The gate cannot certify unfolding in this region."
            )

    if report["epochs"]:
        print("\nper-validation latent gauge (1.0 = no better than the input):")
        print(f"{'epoch':>6}{'stage':>38}{'ks':>8}{'ks_vs_id':>10}{'w1_vs_id':>10}{'gate':>7}")
        for row in report["epochs"]:
            block = row["reference"]
            model = row["model"]
            print(
                f"{row['global_epoch']:6d}{str(row['stage'])[:36]:>38}"
                f"{_fmt(model.get('latent_mass_ks')):>8}"
                f"{_fmt(block['latent_mass_ks_vs_identity']):>10}"
                f"{_fmt(block['latent_mass_w1_gev_vs_identity']):>10}"
                f"{str(row['gate_passed']):>7}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write the full JSON report here. Omit to print only.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    config = json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    history_path = run_dir / "history.json"
    history = (
        json.loads(history_path.read_text(encoding="utf-8"))
        if history_path.exists()
        else []
    )
    evaluation_path = run_dir / "joint_evaluation.json"
    evaluation = (
        json.loads(evaluation_path.read_text(encoding="utf-8"))
        if evaluation_path.exists()
        else None
    )

    report = {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "run_label": config.get("run_label"),
        "regions": {},
    }
    for offset, region in enumerate(config["region_order"]):
        report["regions"][region] = region_report(
            run_dir, config, provenance, history, evaluation, region, offset
        )
        if not args.quiet:
            print_report(region, report["regions"][region])

    if args.out is not None:
        args.out.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
