#!/usr/bin/env python
"""Score a finished paired-benchmark run against its withheld truth partners.

This is the driver that closes memory.md section 7 step 3. It does three
things and nothing else:

1. reads the run's ``joint_split_manifest.json``, pulls the withheld
   ``pair_index`` for one split, and re-reads the source HDF5 to rebuild the
   ``(z, x)`` partners in the exact order the model was shown ``x``;
2. runs the frozen encoder on that ``x``, producing ``z_pred``;
3. hands both to ``scripts_joint/paired_closure.py`` and writes the report.

The pairing is touched for the first time HERE, after training is over. It is
never in any array the trainer receives -- see the prime directive in
``scripts_joint/paired_data.py``.

The npz files are written out as well, so the report can be reproduced by the
documented CLI without this script:

    python scripts_joint/paired_closure.py \\
        --pairs-npz <run>/paired_closure/pairs_test.npz \\
        --pred-npz  <run>/paired_closure/pred_test.npz

Usage:

    python scripts_joint/score_paired_closure.py --run-dir outputs/cms_Joint/ppzee
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
for directory in (
    REPO_ROOT / "scripts",
    REPO_ROOT / "scripts_sota",
    REPO_ROOT / "scripts_joint",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from joint_model import JointDimuonAutoencoder  # noqa: E402
from joint_trainer import restore_joint_checkpoint  # noqa: E402
from paired_closure import paired_closure, print_report  # noqa: E402
from paired_data import check_unpaired, materialize_pairs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Defaults to <run-dir>/best_model.pt. A run with "
        "global_gate_fallback.txt has an unaccepted best_model.pt; score "
        "last_model.pt as well when comparing arms.",
    )
    parser.add_argument("--region", default=None, help="Defaults to the only paired region.")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument(
        "--draws",
        type=int,
        default=0,
        help="Posterior draws of the encoder. 0 (default) scores the single "
        "deterministic pass. Only meaningful when the checkpoint carries a "
        "non-zero encoder noise multiplier.",
    )
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def _buffer(state, key):
    if key not in state:
        raise KeyError(f"Checkpoint is missing {key!r}")
    return state[key].detach().cpu().numpy()


def load_frozen_model(checkpoint_path: Path, device: torch.device):
    """Rebuild the model from its own checkpoint, exactly as decode_prior does."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict")
    config = checkpoint.get("config")
    if not isinstance(state, dict) or not isinstance(config, dict):
        raise ValueError("Checkpoint lacks model_state_dict or its resolved config")
    indices = tuple(int(v) for v in _buffer(state, "encoder.condition_indices").tolist())
    if 8 in indices:
        raise ValueError("Frozen joint contract violation: explicit pair mass is conditioned")
    model_config = config["model"]
    model = JointDimuonAutoencoder(
        (_buffer(state, "encoder.condition_mean"), _buffer(state, "encoder.condition_std")),
        (_buffer(state, "decoder.condition_mean"), _buffer(state, "decoder.condition_std")),
        indices,
        model_config,
        float(config.get("muon_mass_gev", 0.1056583755)),
        model_config.get("daughter_masses"),
    ).to(device)
    restore_joint_checkpoint(model, checkpoint)
    model.eval()
    return model, checkpoint, config


@torch.no_grad()
def encode_all(model, values, *, device, batch_size, seed):
    torch.manual_seed(seed)
    out = []
    for start in range(0, len(values), batch_size):
        block = torch.as_tensor(
            values[start : start + batch_size], dtype=torch.float32, device=device
        )
        encoded = model.encode(block)
        if isinstance(encoded, (tuple, list)):
            encoded = encoded[0]
        out.append(encoded.detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    manifest_path = run_dir / "joint_split_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No split manifest in {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    region = args.region
    if region is None:
        paired = [
            name for name, block in manifest["regions"].items() if "pair_index" in block
        ]
        if len(paired) != 1:
            raise SystemExit(
                f"Pass --region; {run_dir.name} has paired regions {paired}"
            )
        region = paired[0]
    if "pair_index" not in manifest["regions"].get(region, {}):
        raise SystemExit(
            f"Region {region!r} carries no withheld pairing. Only paired "
            "benchmark regions (ppzee, ppttbar) can be scored per event; the "
            "CMS open data has no truth partner."
        )

    checkpoint_path = args.checkpoint or (run_dir / "best_model.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    device = choose_device(args.device)
    model, checkpoint, config = load_frozen_model(checkpoint_path, device)

    z_true, x_input = materialize_pairs(manifest_path, region, args.split)
    print(f"{region} {args.split}: {len(z_true)} withheld pairs from "
          f"{manifest['regions'][region]['pair_index']['sources'][args.split]}")

    # The pairs must BE pairs -- the same positive control the loader runs, now
    # in the opposite direction. If this reads low, the scorer is being handed
    # mismatched rows and every number below is meaningless.
    try:
        check_unpaired(x_input, z_true)
    except AssertionError:
        pass  # correlated, as required
    else:
        raise SystemExit(
            "The materialized pairs are NOT correlated event by event. The "
            "pair index or the source file is wrong; refusing to report a "
            "closure number."
        )

    z_pred = encode_all(
        model,
        x_input.astype(np.float32),
        device=device,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    draws = None
    if args.draws > 0:
        draws = np.stack([
            encode_all(
                model,
                x_input.astype(np.float32),
                device=device,
                batch_size=args.batch_size,
                seed=args.seed + 1000 * (index + 1),
            )
            for index in range(int(args.draws))
        ])

    out_dir = run_dir / "paired_closure"
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = out_dir / f"pairs_{args.split}.npz"
    pred_path = out_dir / f"pred_{args.split}.npz"
    np.savez_compressed(pairs_path, z=z_true, x=x_input)
    if draws is None:
        np.savez_compressed(pred_path, z_pred=z_pred)
    else:
        np.savez_compressed(pred_path, z_pred=z_pred, z_pred_draws=draws)
    print(f"wrote {pairs_path}\nwrote {pred_path}")

    report = paired_closure(z_true, x_input, z_pred, z_pred_draws=draws)
    print_report(f"{run_dir.name} encoder E(x) vs truth partner", report)

    # DID THE MAP MOVE AT ALL?
    #
    # Our encoder is a residual flow initialised near the identity (the
    # residual head starts at 1e-4 and passes through a bounded tanh --
    # memory.md section 7.6 reason 2). An untrained one therefore scores
    # residual_rms_vs_identity ~ 1.0 for the trivial reason that E(x) ~ x, not
    # because it inverted anything. artifact-measured on the two-epoch smoke:
    # 1.0439. The upstream encoder is a plain MLP with no such prior and scores
    # 3.33. So the ratio alone cannot be read without this number beside it:
    #
    #   displacement_vs_resolution = rms(E(x) - x) / rms(x - z_true)
    #
    #   ~ 0   the map is the identity in disguise; the ratio is meaningless
    #   ~ 1   the map moved events by about one detector resolution, which is
    #         the scale it would have to move them to invert the response
    #   >> 1  the map moved them much further than the resolution
    #
    # CLAUDE.md section 4: never quote a gate number without its reference.
    def _mass(values):
        pair = np.asarray(values, dtype=np.float64)
        p = pair[:, 0:4] + pair[:, 4:8]
        return np.sqrt(np.clip(p[:, 3] ** 2 - np.sum(p[:, 0:3] ** 2, axis=1), 0.0, None))

    resolution_rms = float(report["per_event_mass"]["identity_rms"])
    mass_displacement = _mass(z_pred) - _mass(x_input)
    displacement = {
        "mass_displacement_rms_gev": float(np.sqrt(np.mean(mass_displacement**2))),
        "displacement_vs_resolution": (
            float(np.sqrt(np.mean(mass_displacement**2)) / resolution_rms)
            if resolution_rms > 0
            else float("inf")
        ),
        "fourvector_displacement_rms_per_column": np.sqrt(
            np.mean((np.asarray(z_pred, dtype=np.float64) - x_input) ** 2, axis=0)
        ).tolist(),
        "pred_vs_input_mass_correlation": float(
            np.corrcoef(_mass(z_pred), _mass(x_input))[0, 1]
        ),
        "meaning": (
            "rms(mass(E(x)) - mass(x)) over rms(mass(x) - mass(z_true)). Near "
            "zero means the encoder is the identity in disguise and "
            "residual_rms_vs_identity carries no claim."
        ),
    }
    report["displacement_from_input"] = displacement
    print("did the map move at all?")
    print(f"  rms(mass(E(x)) - mass(x))          {displacement['mass_displacement_rms_gev']:.4f} GeV")
    print(f"  displacement_vs_resolution         {displacement['displacement_vs_resolution']:.4f}"
          "   (~0 = identity in disguise)")
    print(f"  corr(mass(E(x)), mass(x))          {displacement['pred_vs_input_mass_correlation']:.4f}")

    noise = checkpoint.get("noise_multipliers") or {}
    payload = {
        "model": report,
        "provenance": {
            "run_dir": str(run_dir),
            "checkpoint": str(checkpoint_path),
            "checkpoint_global_epoch": checkpoint.get("global_epoch"),
            "checkpoint_stage": (checkpoint.get("stage") or {}).get("name"),
            "global_gate_fallback": bool(checkpoint.get("global_gate_fallback", False)),
            "noise_multipliers": noise,
            "region": region,
            "split": args.split,
            "events": int(len(z_true)),
            "encoder_draws": int(args.draws),
            "joint_contract_sha256": manifest.get("contract_sha256"),
            "run_label": config.get("run_label"),
            "pairs_npz": str(pairs_path),
            "pred_npz": str(pred_path),
        },
        "reference": {
            "identity_map": 1.0,
            "identity_map_meaning": (
                "residual_rms_vs_identity for handing back the detector-level "
                "event unchanged. Below 1.0 means resolution was inverted "
                "event by event; at or above 1.0 means it was not."
            ),
            "upstream_otus_encoder": 3.3304,
            "upstream_reference": (
                "artifact-measured 2026-09-07 from "
                "experiments/ppzee/otus_results-dataset=ppzee_test.npz, which "
                "is data/ppzee_test.hdf5 row for row to float32; memory.md "
                "section 7.4 recorded 3.33."
            ),
        },
    }
    out_path = args.out or (out_dir / f"paired_closure_{args.split}.json")
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    ratio = report["per_event_mass"].get("residual_rms_vs_identity")
    print("\n--- the three-way comparison ---")
    print(f"  identity (hand back x)      1.0000")
    print(f"  this model                  {ratio:.4f}"
          f"   (moved {displacement['displacement_vs_resolution']:.3f} resolutions)")
    print(f"  upstream OTUS encoder       3.3304")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
