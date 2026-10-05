#!/usr/bin/env python
"""I2: joint two-region kernel calibration (read-only, decode-probe based).

Question this answers
---------------------
Does ONE amplitude law

    sigma_logpT = shape(eta) * (offset + slope * pT)

exist that puts BOTH training regions on their adopted quadrature targets, and
what does that law then predict for the Upsilon family it never saw?

The shipped kernel was fitted by the same machinery but against the three
DOCUMENTED targets {jpsi 28.1 MeV, upsilon1s 84 MeV, z 2.502 GeV}; the delivered
ratios were 1.21 / 0.86 / 0.83, i.e. a single law missed all three. This script
asks the sharper, two-parameter-exact question and reports the Upsilon
*prediction* rather than a fit residual.

Estimator conventions (docs/calibration_target_2026-10-04.md)
-------------------------------------------------------------
The probe returns both a robust and a std within-z width, so both P4 readings
are evaluated against a matching estimator:

    robust (reading F, the A0.4 R-gate denominator): jpsi 30.02 MeV, z 2.538 GeV
    std    (reading E, the adopted target):          jpsi 23.81 MeV, z 2.884 GeV

Upsilon has no quadrature target (negative variance); our own fit gives
84.42 MeV for the 1S and is used only as a reference for the prediction.

Nothing is trained. Nothing under data/, outputs/ or any checkpoint is
modified: checkpoints are loaded read-only and every artifact goes to a new
--output-dir.
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
from fixed_z_noise_budget import (  # noqa: E402
    decode_mass_matrix,
    load_fixed_z,
    resolve_prior_path,
)
from kernel_decode_probe import (  # noqa: E402
    COMPONENT_MAP,
    UPSILON_STATES,
    within_metrics,
)
from kernel_powerlaw_fit import build_kernel_spec, median_muon_pt  # noqa: E402
from cylindrical_flow import _validate_sigma_floor_spec  # noqa: E402

DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "outputs"
    / "cms_Joint"
    / "Run_H_cycleNoNoise"
    / "best_RunHcycleNoNoise_stage2_stochastic_core.pt"
)
DEFAULT_BASE_SPEC = (
    REPO_ROOT
    / "outputs"
    / "cms_Joint"
    / "Run_H_cycleNoNoise"
    / "kernel_fit_linear"
    / "kernel_spec_fitted.json"
)

CONVENTIONS = {
    "robust": {
        "metric": "robust",
        "label": "F (robust half-width; the A0.4 R-gate denominator)",
        "targets": {"jpsi": 0.03002, "z": 2.538},
    },
    "std": {
        "metric": "std",
        "label": "E (std; the adopted P4 target)",
        "targets": {"jpsi": 0.02381, "z": 2.884},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--base-spec", type=Path, default=DEFAULT_BASE_SPEC)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--events", type=int, default=256)
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--design-p1", default="0.004,0.011")
    parser.add_argument("--design-p2", type=float, default=6e-4)
    parser.add_argument("--upsilon-reference", type=float, default=0.08442)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda"
        if (args.device == "auto" and torch.cuda.is_available())
        else ("cpu" if args.device == "auto" else args.device)
    )

    base = json.loads(args.base_spec.read_text(encoding="utf-8"))
    validated_base = _validate_sigma_floor_spec(base)
    # The shipped linear spec stores shape(eta) * p, so the mean-normalised
    # offset array IS the eta shape this spec was built from.
    offset_array = np.asarray(validated_base["linear_offset"], dtype=float)
    slope_array = np.asarray(validated_base["linear_slope"], dtype=float)
    shape = (offset_array / offset_array.mean()).tolist()
    shipped_p1 = float(offset_array.mean())
    shipped_p2 = float(slope_array.mean())
    if not np.allclose(slope_array, shipped_p2 * np.asarray(shape), rtol=1e-6, atol=1e-12):
        print("warning: shipped slope array is not proportional to the offset shape")

    model, checkpoint, config = load_frozen_model(args.checkpoint, device)
    model.eval()
    print(
        f"checkpoint {args.checkpoint.name} global_epoch {checkpoint.get('global_epoch')} "
        f"device {device}"
    )
    print(f"shipped kernel: offset={shipped_p1:.6g} slope={shipped_p2:.6g}")

    fixed_z: dict[str, np.ndarray] = {}
    for offset, region in enumerate(("jpsi", "z")):
        path, _, _ = resolve_prior_path(config, region)
        fixed_z[region], _ = load_fixed_z(path, events=args.events, seed=args.seed + offset)
    upsilon_path = REPO_ROOT / "data" / "upsilon_prior_continuumReweighted.hdf5"
    for index, state in enumerate(UPSILON_STATES):
        fixed_z[state], _ = load_fixed_z(
            upsilon_path,
            events=args.events,
            seed=args.seed + 10 + index,
            component_id=COMPONENT_MAP[state],
        )
    print("fixed z: " + ", ".join(f"{name}={len(z)}" for name, z in fixed_z.items()))

    def measure(p1: float, p2: float) -> dict[str, dict[str, float]]:
        spec = _validate_sigma_floor_spec(build_kernel_spec("linear", shape, p1, p2, base))
        for step in model.decoder.steps:
            step.sigma_floor_spec = spec
            step.freeze_noise_amplitude = True
        out: dict[str, dict[str, float]] = {}
        for index, (name, z) in enumerate(fixed_z.items()):
            matrix, _ = decode_mass_matrix(
                model,
                z,
                batch_size=args.batch_size,
                device=device,
                core=1.0,
                tail=0.0,
                draws=args.draws,
                seed=args.seed + 100 * index,
            )
            metrics = within_metrics(matrix)
            out[name] = {
                "std": float(metrics["within_std_median_gev"]),
                "robust": float(metrics["within_robust_median_gev"]),
            }
        return out

    p1_low, p1_high = (float(v) for v in args.design_p1.split(","))
    design = {
        "p1": [p1_low, p1_high],
        "p2": args.design_p2,
        "w_low": measure(p1_low, 0.0),
        "w_high": measure(p1_high, 0.0),
        "w_second": measure(p1_low, args.design_p2),
        "shipped": measure(shipped_p1, shipped_p2),
    }
    print("design points measured")

    results: dict[str, dict] = {}
    for name, convention in CONVENTIONS.items():
        metric = convention["metric"]
        targets = convention["targets"]
        rows = []
        coefficients: dict[str, list[float]] = {}
        for region in ("jpsi", "z"):
            c_first = (
                design["w_high"][region][metric] - design["w_low"][region][metric]
            ) / (p1_high - p1_low)
            c_second = (
                design["w_second"][region][metric] - design["w_low"][region][metric]
            ) / args.design_p2
            coefficients[region] = [c_first, c_second]
            rows.append([c_first, c_second])
        matrix = np.asarray(rows, dtype=float)
        rhs = np.asarray([targets[r] for r in ("jpsi", "z")], dtype=float)
        try:
            solution = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError:
            solution = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
        p1_fit = max(float(solution[0]), 0.0)
        p2_fit = max(float(solution[1]), 0.0)

        validation = measure(p1_fit, p2_fit)
        per_region = {}
        for region in ("jpsi", "z", *UPSILON_STATES):
            achieved = validation[region][metric]
            if region in targets:
                target = targets[region]
                per_region[region] = {
                    "achieved_gev": achieved,
                    "target_gev": target,
                    "ratio": achieved / target,
                    "kind": "training-region constraint",
                }
            else:
                per_region[region] = {
                    "achieved_gev": achieved,
                    "target_gev": args.upsilon_reference,
                    "ratio": achieved / args.upsilon_reference,
                    "kind": "PREDICTION (region never trained)",
                }
        shipped_ratios = {
            region: design["shipped"][region][metric] / targets[region]
            for region in targets
        }
        results[name] = {
            "convention": convention["label"],
            "metric": metric,
            "targets_gev": targets,
            "coefficients": coefficients,
            "fit": {"offset_p1": p1_fit, "slope_p2": p2_fit},
            "validation": per_region,
            "shipped_ratios_same_metric": shipped_ratios,
            "shipped_widths_same_metric": {
                region: design["shipped"][region][metric] for region in fixed_z
            },
        }
        print(
            f"[{name}] offset={p1_fit:.6g} slope={p2_fit:.6g} | "
            + " ".join(
                f"{r}={per_region[r]['achieved_gev']:.4g}({per_region[r]['ratio']:.3f}x)"
                for r in ("jpsi", "z", "upsilon1s")
            )
        )

    payload = {
        "schema_version": 1,
        "diagnostic": "I2 joint two-region kernel calibration",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_global_epoch": checkpoint.get("global_epoch"),
        "base_spec": str(args.base_spec),
        "shipped_kernel": {"offset_p1": shipped_p1, "slope_p2": shipped_p2},
        "design": {
            "p1": [p1_low, p1_high],
            "p2": args.design_p2,
            "design_widths": design["w_low"],
            "widths_p1_high": design["w_high"],
            "widths_p2": design["w_second"],
        },
        "median_muon_pt_gev": {name: median_muon_pt(z) for name, z in fixed_z.items()},
        "conventions": results,
        "device": str(device),
        "events": args.events,
        "draws": args.draws,
    }
    (output_dir / "joint_calibration.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# I2 - joint two-region kernel calibration",
        "",
        f"*{payload['created_utc']} - read-only decode probe, no training. Checkpoint "
        f"`{Path(args.checkpoint).name}` (epoch {checkpoint.get('global_epoch')}), "
        f"{args.events} fixed z per region, {args.draws} draws, device {device}.*",
        "",
        "Question: does ONE law `shape(eta) * (offset + slope*pT)` put BOTH training",
        "regions on their adopted targets, and what does it predict for Upsilon?",
        "",
        "| convention | fitted offset | fitted slope | jpsi | z | U(1S) prediction | U(2S) | U(3S) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, entry in results.items():
        v = entry["validation"]
        lines.append(
            f"| {name} | {entry['fit']['offset_p1']:.5g} | {entry['fit']['slope_p2']:.5g} | "
            f"{v['jpsi']['achieved_gev']:.4g} ({v['jpsi']['ratio']:.3f}x) | "
            f"{v['z']['achieved_gev']:.4g} ({v['z']['ratio']:.3f}x) | "
            f"{v['upsilon1s']['achieved_gev']:.4g} ({v['upsilon1s']['ratio']:.3f}x) | "
            f"{v['upsilon2s']['achieved_gev']:.4g} ({v['upsilon2s']['ratio']:.3f}x) | "
            f"{v['upsilon3s']['achieved_gev']:.4g} ({v['upsilon3s']['ratio']:.3f}x) |"
        )
    lines += [
        "",
        "## Reading",
        "",
        "- jpsi and z are the two constraints the fit solves for, so their ratios are",
        "  the fit residual (they should be ~1.000 if the local linear model is exact).",
        "- Upsilon is a genuine prediction: the region is never trained, and its",
        "  reference is our own fit 84.42 MeV (no quadrature target exists there).",
        f"- Shipped kernel offset={shipped_p1:.5g} slope={shipped_p2:.5g}; its ratios",
        "  under the same metric and targets are in `joint_calibration.json` under",
        "  `conventions.<name>.shipped_ratios_same_metric`.",
        "",
        "## Median muon pT per region [GeV]",
        "",
        "| region | median muon pT |",
        "|---|---|",
    ]
    for region, value in payload["median_muon_pt_gev"].items():
        lines.append(f"| {region} | {value:.4g} |")
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output_dir / 'joint_calibration.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
