#!/usr/bin/env python
"""Re-score a finished joint checkpoint with the trainer's own validation path.

Read-only. The training-time ``joint_selection`` score is computed at zero
noise by default, which structurally penalises a model whose resolution lives
in the decoder's noise channel (A0.3 in ``docs/project_tree.md``). This tool
re-runs ``joint_trainer.validate_joint`` on the run's locked validation splits
with an explicit noise policy, so the A2 arms can be compared at their
operating point and the G-A "in-domain score" item can be read directly.

Nothing is trained and no existing artifact is modified; a new JSON report is
written only when ``--out`` is given.

Usage
-----
python scripts_joint/rescore_validation.py \
    --checkpoint outputs/cms_Joint/Run_H_A2frozen/best_model.pt \
    --noise native --compare-zero \
    --out outputs/cms_Joint/Run_H_A2frozen/validation_native_best.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (
    REPO_ROOT,
    REPO_ROOT / "scripts",
    REPO_ROOT / "scripts_sota",
    REPO_ROOT / "scripts_joint",
    REPO_ROOT / "scripts_joint" / "upsilon",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import torch  # noqa: E402

from decode_prior import load_frozen_model  # noqa: E402
from joint_data import load_joint_regions  # noqa: E402
from joint_trainer import validate_joint  # noqa: E402
from run_joint import build_loss_factories  # noqa: E402


def _noise_spec(args, checkpoint: dict) -> dict[str, float]:
    if args.noise == "zero":
        return {"core": 0.0, "tail": 0.0}
    if args.noise == "native":
        recorded = checkpoint.get("noise_multipliers") or {}
        return {
            "core": float(recorded.get("core", recorded.get("decoder_core", 1.0))),
            "tail": float(recorded.get("tail", recorded.get("decoder_tail", 0.0))),
        }
    spec: dict[str, float] = {}
    if args.core is not None:
        spec["core"] = float(args.core)
    if args.tail is not None:
        spec["tail"] = float(args.tail)
    if not spec:
        raise SystemExit("--noise explicit needs --core and/or --tail")
    return spec


def _summary(selection: dict) -> str:
    regions = selection.get("regions", {})
    parts = []
    for name, block in regions.items():
        parts.append(
            f"{name}: score={block.get('score', float('nan')):.3f} "
            f"worst={block.get('worst_metric')}"
        )
    return (
        f"worst_region={selection.get('worst_region')} "
        f"score={selection.get('selection_score', float('nan')):.5f} | "
        + " | ".join(parts)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--noise",
        choices=("native", "zero", "explicit"),
        default="native",
        help="noise policy for the validation pass (default: the checkpoint's "
        "recorded multipliers)",
    )
    parser.add_argument("--core", type=float, default=None)
    parser.add_argument("--tail", type=float, default=None)
    parser.add_argument(
        "--compare-zero",
        action="store_true",
        help="also score the deterministic map (0/0) for reference",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    model, checkpoint, config = load_frozen_model(args.checkpoint, device)
    model.eval()

    region_arrays, cache_info, _, _ = load_joint_regions(
        config, num_samples=None, use_cache=True
    )
    factories = build_loss_factories(config, region_arrays)

    seed = int(config.get("seed", 0)) + int(checkpoint.get("global_epoch", 0) or 0)
    payload: dict = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "checkpoint_global_epoch": checkpoint.get("global_epoch"),
        "checkpoint_stage": (checkpoint.get("stage") or {}).get("name"),
        "checkpoint_noise_multipliers": checkpoint.get("noise_multipliers"),
        "validation_seed": seed,
        "validation_events": (config.get("loaders") or {}).get("validation_events"),
        "runs": {},
    }

    policies = [(_noise_spec(args, checkpoint), args.noise)]
    if args.compare_zero and args.noise != "zero":
        policies.append(({"core": 0.0, "tail": 0.0}, "zero"))

    for spec, label in policies:
        config.setdefault("loaders", {})["validation_noise_multipliers"] = spec
        metrics, selection = validate_joint(
            model,
            region_arrays,
            factories,
            config,
            device=device,
            seed=seed,
        )
        payload["runs"][label] = {
            "noise_multipliers": spec,
            "selection": selection,
            "region_metrics": metrics,
        }
        print(f"[{label}] {_summary(selection)}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
