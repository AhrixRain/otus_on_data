#!/usr/bin/env python
"""I6: decompose the Upsilon zero-noise mean-map spread along truth pair-pT.

At the D3b best checkpoint the decoded Upsilon(1S) width at zero noise is
~214 MeV against a ~110 MeV prior and an 84 MeV detector reference, i.e. the
mean map alone is 2.5x the detector resolution. This audit asks WHERE that
spread lives: bin the truth sample by pair-pT and measure the local spread of
the deterministic map response (decoded - truth) inside each bin.

If the local response spread grows with truth pair-pT, the map is
extrapolating a J/psi-calibrated, pT-driven response into a region the training
regions do not populate, and the fix is a pT-resolved constraint on the map.
If it is flat, the spread is a global scale error and a different fix applies.

Read-only: the checkpoint is loaded read-only, the prior is read, and every
artifact goes to a new --output-dir.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for _directory in (
    REPO_ROOT / "scripts",
    REPO_ROOT / "scripts_sota",
    REPO_ROOT / "scripts_joint",
    REPO_ROOT / "scripts_joint" / "upsilon",
):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from decode_prior import load_frozen_model  # noqa: E402
from fixed_z_noise_budget import decode_mass_matrix, load_fixed_z, mass8  # noqa: E402


def pair_pt(p4: np.ndarray) -> np.ndarray:
    values = np.asarray(p4, dtype=np.float64)
    px = values[:, 0] + values[:, 4]
    py = values[:, 1] + values[:, 5]
    return np.hypot(px, py)


def robust(values: np.ndarray) -> float:
    return float((np.quantile(values, 0.84) - np.quantile(values, 0.16)) / 2.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "outputs" / "cms_Joint" / "Run_H_D3b" / "best_model.pt",
    )
    parser.add_argument(
        "--prior", type=Path, default=REPO_ROOT / "data" / "upsilon_prior_continuumReweighted.hdf5"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--events", type=int, default=20000)
    parser.add_argument("--bins", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    model, checkpoint, _config = load_frozen_model(args.checkpoint, device)
    model.eval()

    payload_states = {}
    for index, state in enumerate(("upsilon1s", "upsilon2s", "upsilon3s")):
        z, info = load_fixed_z(
            args.prior, events=args.events, seed=args.seed + index, component_id=index
        )
        truth_mass = mass8(z)
        pt = pair_pt(z)
        matrix, _ = decode_mass_matrix(
            model,
            z,
            batch_size=8192,
            device=device,
            core=0.0,
            tail=0.0,
            draws=1,
            seed=args.seed + 100 + index,
        )
        decoded_mass = matrix[:, 0] if matrix.ndim == 2 else matrix
        residual = decoded_mass - truth_mass

        edges = np.quantile(pt, np.linspace(0.0, 1.0, args.bins + 1))
        edges[-1] = edges[-1] + 1e-9
        rows = []
        for low, high in zip(edges[:-1], edges[1:]):
            mask = (pt >= low) & (pt < high)
            if mask.sum() < 50:
                continue
            rows.append(
                {
                    "pair_pt_low_gev": float(low),
                    "pair_pt_high_gev": float(high),
                    "n": int(mask.sum()),
                    "residual_mean_mev": float(np.mean(residual[mask]) * 1000.0),
                    "residual_std_mev": float(np.std(residual[mask]) * 1000.0),
                    "residual_robust_mev": robust(residual[mask]) * 1000.0,
                    "decoded_std_mev": float(np.std(decoded_mass[mask]) * 1000.0),
                    "truth_std_mev": float(np.std(truth_mass[mask]) * 1000.0),
                }
            )
        payload_states[state] = {
            "component_id": index,
            "available": info.get("available"),
            "sampled": int(len(z)),
            "zero_noise_decoded_std_mev": float(np.std(decoded_mass) * 1000.0),
            "zero_noise_decoded_robust_mev": robust(decoded_mass) * 1000.0,
            "truth_std_mev": float(np.std(truth_mass) * 1000.0),
            "truth_robust_mev": robust(truth_mass) * 1000.0,
            "residual_mean_mev": float(np.mean(residual) * 1000.0),
            "residual_std_mev": float(np.std(residual) * 1000.0),
            "pt_bins": rows,
        }
        print(
            f"{state}: decoded(std) {np.std(decoded_mass)*1000:.1f} MeV, truth(std) "
            f"{np.std(truth_mass)*1000:.1f} MeV, residual std {np.std(residual)*1000:.1f} MeV"
        )
        for row in rows:
            print(
                f"   pair-pT {row['pair_pt_low_gev']:.2f}-{row['pair_pt_high_gev']:.2f}: "
                f"n={row['n']:>5d} residual std {row['residual_std_mev']:7.1f} MeV "
                f"mean {row['residual_mean_mev']:7.1f} MeV"
            )

    payload = {
        "schema_version": 1,
        "diagnostic": "I6 Upsilon zero-noise mean-map spread audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_global_epoch": checkpoint.get("global_epoch"),
        "prior": str(args.prior),
        "noise": {"decoder_core": 0.0, "decoder_tail": 0.0},
        "states": payload_states,
    }
    (output_dir / "upsilon_map_spread.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {output_dir / 'upsilon_map_spread.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
