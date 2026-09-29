#!/usr/bin/env python
"""Slice-resolved (conditional) diagnostic for the joint loss, on Run F.

Answers, with the Run F data contract and the Run F best_model (noise 0):

1. In-distribution (J/psi, Z): is the mass response conditionally wrong at fixed
   pair-pT, or is the per-slice mismatch already at the finite-sample floor?
   A sliced loss can only help where the mismatch clearly exceeds the floor.
2. What is the per-slice finite-sample floor at the real per-update
   cardinalities, for 2/3/4/5 slices?  That sets the floor budget.

Slices are equal-count by the *truth* pair-pT (quartiles), so each slice has
comparable statistics.  Units are the standardized 1-D mass W1 used by the loss.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (
    REPO_ROOT / "scripts",
    REPO_ROOT / "scripts_sota",
    REPO_ROOT / "scripts_joint",
    REPO_ROOT / "scripts_joint" / "upsilon",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from decode_prior import load_frozen_model  # noqa: E402
from joint_data import load_joint_regions  # noqa: E402
from run_joint import build_loss_factories  # noqa: E402

CHECKPOINT = REPO_ROOT / "outputs" / "cms_Joint" / "Run_F" / "best_model.pt"
N_SLICES_FOR_FLOOR = (1, 2, 3, 4, 5)


def pair_pt(values: torch.Tensor) -> torch.Tensor:
    return torch.hypot(values[:, 0] + values[:, 4], values[:, 1] + values[:, 5])


def w1_standardized(space, a: torch.Tensor, b: torch.Tensor) -> float:
    ma = space.standardize_mass(space.invariant_mass(a))
    mb = space.standardize_mass(space.invariant_mass(b))
    return float(space.wasserstein_1d_sorted(ma, mb))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "cms_Joint" / "slice_diagnostic")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT,
        help="Frozen checkpoint to score (default: Run F best_model.pt).",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-samples", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument(
        "--noise-core",
        type=float,
        default=0.0,
        help="Core noise multiplier for the decode/encode pass (default 0 = "
        "deterministic map; use the checkpoint's native value for the "
        "generative readout).",
    )
    parser.add_argument(
        "--noise-tail",
        type=float,
        default=0.0,
        help="Tail noise multiplier for the decode/encode pass.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    model, checkpoint, config = load_frozen_model(args.checkpoint, device)
    model.set_noise_multipliers(args.noise_core, args.noise_tail)
    model.eval()

    region_arrays, cache_info, _, _ = load_joint_regions(
        config, num_samples=None, use_cache=True
    )
    factories = build_loss_factories(config, region_arrays)

    def sample(values: np.ndarray, n: int, rng) -> torch.Tensor:
        if len(values) > n:
            values = values[rng.choice(len(values), size=n, replace=False)]
        return torch.as_tensor(np.ascontiguousarray(values), dtype=torch.float32, device=device)

    rng = np.random.default_rng(args.seed)
    report: dict = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_global_epoch": checkpoint.get("global_epoch"),
        "n_samples": args.n_samples,
        "noise_multipliers": {"core": args.noise_core, "tail": args.noise_tail},
        "regions": {},
        "floor_budget": {},
    }

    for region in ("jpsi", "z"):
        factory = factories[region]
        ed, ez = factory.x_space, factory.z_space
        arrays = region_arrays[region]
        x = torch.cat([sample(arrays["x_train"], args.n_samples, rng), sample(arrays["x_val"], args.n_samples, rng)])
        z = torch.cat([sample(arrays["z_train"], args.n_samples, rng), sample(arrays["z_val"], args.n_samples, rng)])
        with torch.no_grad():
            decoded = model.decode(z)
            encoded = model.encode(x)
            if isinstance(encoded, (tuple, list)):
                encoded = encoded[0]
        x_pt, z_pt, dec_pt, enc_pt = pair_pt(x), pair_pt(z), pair_pt(decoded), pair_pt(encoded)

        edges_x = np.percentile(x_pt.cpu().numpy(), [0, 25, 50, 75, 100])
        edges_z = np.percentile(z_pt.cpu().numpy(), [0, 25, 50, 75, 100])
        edges_x[-1] += 1e-6
        edges_z[-1] += 1e-6

        rows = []
        for i in range(len(edges_x) - 1):
            lo_x, hi_x = float(edges_x[i]), float(edges_x[i + 1])
            lo_z, hi_z = float(edges_z[i]), float(edges_z[i + 1])
            sx = (x_pt >= lo_x) & (x_pt < hi_x)
            sd = (dec_pt >= lo_x) & (dec_pt < hi_x)
            sz = (z_pt >= lo_z) & (z_pt < hi_z)
            se = (enc_pt >= lo_z) & (enc_pt < hi_z)
            floor_dec = floor_enc = None
            if int(sx.sum()) > 50:
                half = int(sx.sum()) // 2
                xs = x[sx]
                floor_dec = w1_standardized(ed, xs[:half], xs[half : 2 * half])
            if int(sz.sum()) > 50:
                half = int(sz.sum()) // 2
                zs = z[sz]
                floor_enc = w1_standardized(ez, zs[:half], zs[half : 2 * half])
            rows.append(
                {
                    "slice": i,
                    "x_pair_pt_lo": lo_x,
                    "x_pair_pt_hi": hi_x,
                    "z_pair_pt_lo": lo_z,
                    "z_pair_pt_hi": hi_z,
                    "n_decode_true": int(sx.sum()),
                    "n_decode_pred": int(sd.sum()),
                    "n_encode_true": int(sz.sum()),
                    "n_encode_pred": int(se.sum()),
                    "decode_mass_w1": float(w1_standardized(ed, x[sx], decoded[sd])) if sx.sum() > 50 and sd.sum() > 50 else None,
                    "decode_mass_w1_floor": floor_dec,
                    "encode_mass_w1": float(w1_standardized(ez, z[sz], encoded[se])) if sz.sum() > 50 and se.sum() > 50 else None,
                    "encode_mass_w1_floor": floor_enc,
                }
            )
        report["regions"][region] = {"slices": rows}

    # ---- per-slice floor versus slice count, real per-update cardinality ----
    # The W1 floor is set by the smaller side; for the decoder that is the z
    # stream (J/psi 4096, Z 4651 after min_events_per_update), for the encoder
    # the z truth stream. Equal-count slices of the x pool are used as the proxy.
    for region, effective in (("jpsi", 4096), ("z", 4651)):
        factory = factories[region]
        ed = factory.x_space
        arrays = region_arrays[region]
        pool = np.concatenate(
            [arrays["x_train"][:200000], arrays["x_val"][:200000]], axis=0
        )
        budget = {}
        for n_slices in N_SLICES_FOR_FLOOR:
            n_per = max(effective // n_slices, 32)
            floors = []
            for _ in range(8):
                idx = rng.choice(len(pool), size=2 * n_per, replace=False)
                a = torch.as_tensor(pool[idx[:n_per]], dtype=torch.float32, device=device)
                b = torch.as_tensor(pool[idx[n_per:]], dtype=torch.float32, device=device)
                floors.append(w1_standardized(ed, a, b))
            budget[int(n_slices)] = {
                "events_per_slice": int(n_per),
                "mean_slice_floor": float(np.mean(floors)),
                "sd": float(np.std(floors)),
            }
        report["floor_budget"][region] = budget

    (args.output_dir / "slice_diagnostic.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        f"# Slice-resolved diagnostic (noise {args.noise_core:g}/{args.noise_tail:g})",
        "",
        f"Checkpoint: `{report['checkpoint']}` (global epoch {report['checkpoint_global_epoch']}).",
        "",
    ]
    for region, block in report["regions"].items():
        lines += [
            f"## {region} (equal-count truth pair-pT quartiles)",
            "",
            "| slice | decode mass W1 | decode floor | ratio | encode mass W1 | encode floor | ratio |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in block["slices"]:
            def ratio(value, floor):
                return "-" if value is None or not floor else f"{value / floor:.2f}"
            lines.append(
                "| {slice} | {d} | {df} | {dr} | {e} | {ef} | {er} |".format(
                    slice=row["slice"],
                    d="-" if row["decode_mass_w1"] is None else f"{row['decode_mass_w1']:.4f}",
                    df="-" if not row["decode_mass_w1_floor"] else f"{row['decode_mass_w1_floor']:.4f}",
                    dr=ratio(row["decode_mass_w1"], row["decode_mass_w1_floor"]),
                    e="-" if row["encode_mass_w1"] is None else f"{row['encode_mass_w1']:.4f}",
                    ef="-" if not row["encode_mass_w1_floor"] else f"{row['encode_mass_w1_floor']:.4f}",
                    er=ratio(row["encode_mass_w1"], row["encode_mass_w1_floor"]),
                )
            )
        lines.append("")
    lines += [
        "## Per-slice floor vs slice count (real per-update cardinality)",
        "",
        "| region | slices | events/slice | mean slice floor | sd |",
        "|---|---|---|---|---|",
    ]
    for region, block in report["floor_budget"].items():
        for n_slices, value in block.items():
            lines.append(
                f"| {region} | {n_slices} | {value['events_per_slice']} | "
                f"{value['mean_slice_floor']:.4f} | {value['sd']:.4f} |"
            )
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
