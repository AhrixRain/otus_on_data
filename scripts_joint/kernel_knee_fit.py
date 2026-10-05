#!/usr/bin/env python
"""S2' fit: knee-augmented linear resolution kernel (read-only decode probe).

I2 (outputs/cms_Joint/kernel_joint_calibration/) measured that a pure
offset + slope*pT law cannot satisfy J/psi, Z and Upsilon simultaneously: with
the two training regions solved exactly, Upsilon comes out at 0.367x its
reference, because its muons are softer (median 4.7 GeV versus 13.1 GeV at
J/psi) yet its required per-muon log-pT amplitude is essentially the same.

This script fits the three-parameter knee law

    sigma_logpT = scale * (offset + slope * max(0, pT - knee_pt_gev))

over a knee grid. For each knee the two remaining parameters are solved exactly
for the two TRAINING regions on the robust (reading F) denominators
J/psi 30.02 MeV and Z 2.538 GeV; the Upsilon family is then a prediction against
our own Upsilon(1S) fit 84.42 MeV.

Nothing is trained; checkpoints are read-only and every artifact goes to a new
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
from kernel_decode_probe import COMPONENT_MAP, UPSILON_STATES, within_metrics  # noqa: E402
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
    parser.add_argument("--knees", default="0,8,12,16")
    parser.add_argument("--jpsi-target", type=float, default=0.03002)
    parser.add_argument("--z-target", type=float, default=2.538)
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
    offset_array = np.asarray(validated_base["linear_offset"], dtype=float)
    shape = (offset_array / offset_array.mean()).tolist()

    model, checkpoint, config = load_frozen_model(args.checkpoint, device)
    model.eval()
    print(
        f"checkpoint {args.checkpoint.name} global_epoch {checkpoint.get('global_epoch')} "
        f"device {device}"
    )

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

    def measure(p1: float, p2: float, knee: float) -> dict[str, dict[str, float]]:
        raw_spec = build_kernel_spec("linear", shape, p1, p2, base)
        if knee > 0.0:
            raw_spec["knee_pt_gev"] = float(knee)
        spec = _validate_sigma_floor_spec(raw_spec)
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
    knees = [float(v) for v in args.knees.split(",")]
    trials: dict[str, dict] = {}
    for knee in knees:
        w_low = measure(p1_low, 0.0, knee)
        w_high = measure(p1_high, 0.0, knee)
        w_second = measure(p1_low, args.design_p2, knee)
        rows, targets = [], []
        for region, target in (("jpsi", args.jpsi_target), ("z", args.z_target)):
            c_first = (w_high[region]["robust"] - w_low[region]["robust"]) / (
                p1_high - p1_low
            )
            c_second = (w_second[region]["robust"] - w_low[region]["robust"]) / args.design_p2
            rows.append([c_first, c_second])
            targets.append(target)
        try:
            solution = np.linalg.solve(np.asarray(rows, dtype=float), np.asarray(targets))
        except np.linalg.LinAlgError:
            solution = np.linalg.lstsq(np.asarray(rows, dtype=float), np.asarray(targets), rcond=None)[0]
        p1_fit = max(float(solution[0]), 0.0)
        p2_fit = max(float(solution[1]), 0.0)
        validation = measure(p1_fit, p2_fit, knee)
        ratios = {}
        for region in ("jpsi", "z", *UPSILON_STATES):
            reference = (
                args.jpsi_target
                if region == "jpsi"
                else args.z_target
                if region == "z"
                else args.upsilon_reference
            )
            ratios[region] = {
                "robust_gev": validation[region]["robust"],
                "std_gev": validation[region]["std"],
                "reference_gev": reference,
                "ratio_robust": validation[region]["robust"] / reference,
                "ratio_std_vs_E": validation[region]["std"]
                / (
                    0.02381
                    if region == "jpsi"
                    else 2.884
                    if region == "z"
                    else args.upsilon_reference
                ),
                "kind": "constraint" if region in ("jpsi", "z") else "prediction",
            }
        score = abs(np.log(ratios["upsilon1s"]["ratio_robust"])) + 0.5 * (
            abs(np.log(ratios["jpsi"]["ratio_robust"]))
            + abs(np.log(ratios["z"]["ratio_robust"]))
        )
        trials[f"knee_{knee:g}"] = {
            "knee_pt_gev": knee,
            "fit": {"offset_p1": p1_fit, "slope_p2": p2_fit},
            "design": {"w_low": w_low, "w_high": w_high, "w_second": w_second},
            "validation": ratios,
            "score": float(score),
        }
        print(
            f"knee={knee:>5.1f} offset={p1_fit:.6g} slope={p2_fit:.6g} | "
            f"jpsi {ratios['jpsi']['ratio_robust']:.3f}x  z {ratios['z']['ratio_robust']:.3f}x  "
            f"U1S {ratios['upsilon1s']['ratio_robust']:.3f}x  "
            f"U2S {ratios['upsilon2s']['ratio_robust']:.3f}x  "
            f"U3S {ratios['upsilon3s']['ratio_robust']:.3f}x | score {score:.3f}"
        )

    best_name = min(trials, key=lambda name: trials[name]["score"])
    best = trials[best_name]
    best_spec_raw = build_kernel_spec(
        "linear", shape, best["fit"]["offset_p1"], best["fit"]["slope_p2"], base
    )
    if best["knee_pt_gev"] > 0.0:
        best_spec_raw["knee_pt_gev"] = float(best["knee_pt_gev"])
    best_spec = _validate_sigma_floor_spec(best_spec_raw)
    best_spec["provenance"] = {
        "method": "S2' knee-augmented linear fit via frozen-kernel decode probe",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_spec": str(args.base_spec),
        "checkpoint": str(args.checkpoint),
        "targets_robust_gev": {
            "jpsi": args.jpsi_target,
            "z": args.z_target,
            "upsilon1s_reference_unused_in_fit": args.upsilon_reference,
        },
        "knee_grid": knees,
        "selected_knee_pt_gev": best["knee_pt_gev"],
        "constraints": "J/psi and Z only; Upsilon is a prediction and was never fitted",
    }
    (output_dir / "kernel_spec_knee.json").write_text(
        json.dumps(best_spec, indent=2) + "\n", encoding="utf-8"
    )

    payload = {
        "schema_version": 1,
        "diagnostic": "S2' knee-augmented kernel fit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_global_epoch": checkpoint.get("global_epoch"),
        "base_spec": str(args.base_spec),
        "targets_robust_gev": {"jpsi": args.jpsi_target, "z": args.z_target},
        "upsilon_reference_gev": args.upsilon_reference,
        "design": {"p1": [p1_low, p1_high], "p2": args.design_p2},
        "knee_grid": knees,
        "median_muon_pt_gev": {name: median_muon_pt(z) for name, z in fixed_z.items()},
        "trials": trials,
        "selected": best_name,
        "selected_spec": best_spec,
        "device": str(device),
        "events": args.events,
        "draws": args.draws,
    }
    (output_dir / "knee_fit.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# S2' - knee-augmented kernel fit",
        "",
        f"*{payload['created_utc']} - read-only decode probe, no training. Checkpoint "
        f"`{Path(args.checkpoint).name}` (epoch {checkpoint.get('global_epoch')}), "
        f"{args.events} fixed z per region, {args.draws} draws, device {device}.*",
        "",
        "The two TRAINING regions (J/psi, Z) are solved exactly on the robust (F)",
        "denominators; Upsilon is a prediction.",
        "",
        "| knee [GeV] | offset | slope | J/psi | Z | U(1S) | U(2S) | U(3S) | score |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, trial in trials.items():
        v = trial["validation"]
        lines.append(
            f"| {trial['knee_pt_gev']:g} | {trial['fit']['offset_p1']:.5g} | "
            f"{trial['fit']['slope_p2']:.5g} | {v['jpsi']['ratio_robust']:.3f}x | "
            f"{v['z']['ratio_robust']:.3f}x | {v['upsilon1s']['ratio_robust']:.3f}x | "
            f"{v['upsilon2s']['ratio_robust']:.3f}x | {v['upsilon3s']['ratio_robust']:.3f}x | "
            f"{trial['score']:.3f} |"
        )
    lines += [
        "",
        f"Selected: `{best_name}` (offset {best['fit']['offset_p1']:.6g}, slope "
        f"{best['fit']['slope_p2']:.6g}, knee {best['knee_pt_gev']:g} GeV).",
        "",
        "## Median muon pT per region [GeV]",
        "",
        "| region | median muon pT |",
        "|---|---|",
    ]
    for region, value in payload["median_muon_pt_gev"].items():
        lines.append(f"| {region} | {value:.4g} |")
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"selected {best_name}; wrote {output_dir / 'kernel_spec_knee.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
