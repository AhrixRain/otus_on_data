#!/usr/bin/env python
"""Fit the A2.3 resolution kernel to the model's effective response.

Read-only. Uses the A2.0 eta shape and fits a per-eta amplitude law so that the
*model's achieved* fixed-z mass width matches the physical targets:

* ``--basis power_law``: ``sigma = shape(eta) * C * (pT / pivot)**alpha``
* ``--basis linear``:    ``sigma = shape(eta) * (C + D * pT)``

The response is measured with the decode probe at three design points (two
amplitudes of the first parameter, one of the second), fitted with a local
linear model, solved by relative least squares, and validated with a fresh
decode at the solution.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from datetime import datetime, timezone
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
from fixed_z_noise_budget import (  # noqa: E402
    decode_mass_matrix,
    load_fixed_z,
    resolve_prior_path,
)
from kernel_decode_probe import (  # noqa: E402
    COMPONENT_MAP,
    TARGETS,
    UPSILON_STATES,
    within_metrics,
)
from cylindrical_flow import _validate_sigma_floor_spec  # noqa: E402


def eta_shape(spec: dict) -> list[float]:
    """Normalised eta shape of the A2.0 amplitude curve (power_c or logpt_a)."""
    key = "power_c" if "power_c" in spec else "logpt_a"
    values = np.asarray(spec[key], dtype=float)
    if np.all(values <= 0.0):
        raise ValueError(f"base spec has an all-zero {key} shape")
    return (values / values.mean()).tolist()


def build_kernel_spec(basis: str, shape: list[float], p1: float, p2: float, base: dict) -> dict:
    if p1 < 0.0 or p2 < 0.0:
        raise ValueError("kernel parameters must be non-negative")
    spec = copy.deepcopy(base)
    spec.pop("scale", None)
    spec.pop("mode", None)
    if basis == "power_law":
        spec["power_c"] = [float(p1 * value) for value in shape]
        spec["power_alpha"] = [float(p2)] * len(shape)
        spec.pop("linear_offset", None)
        spec.pop("linear_slope", None)
    elif basis == "linear":
        spec["linear_offset"] = [float(p1 * value) for value in shape]
        spec["linear_slope"] = [float(p2 * value) for value in shape]
        spec.pop("power_c", None)
        spec.pop("power_alpha", None)
        spec.pop("power_pivot_gev", None)
    else:
        raise ValueError(f"unknown basis {basis!r}")
    return spec


def solve_relative(rows: list[list[float]], targets: list[float]) -> tuple[float, float]:
    matrix = np.asarray(rows, dtype=float)
    rhs = np.ones(len(targets), dtype=float)
    solution, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
    return max(float(solution[0]), 0.0), max(float(solution[1]), 0.0)


def median_muon_pt(p4: np.ndarray) -> float:
    values = np.asarray(p4, dtype=float)
    return float(
        np.median(
            np.concatenate(
                [
                    np.hypot(values[:, 0], values[:, 1]),
                    np.hypot(values[:, 4], values[:, 5]),
                ]
            )
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-spec", type=Path, required=True)
    parser.add_argument("--basis", choices=("power_law", "linear"), default="linear")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "cms_Joint"
        / "Run_H_cycleNoNoise"
        / "best_RunHcycleNoNoise_stage2_stochastic_core.pt",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--events", type=int, default=256)
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--design-p1", default="0.005,0.010")
    parser.add_argument("--design-p2", type=float, default=None)
    parser.add_argument(
        "--jpsi-target",
        type=float,
        default=0.0281,
        help="physical J/psi resolution [GeV]; 0.012183 is the smeared-prior additional target",
    )
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
    design_p2 = args.design_p2
    if design_p2 is None:
        design_p2 = 1.0 if args.basis == "power_law" else 4e-4

    base = json.loads(args.base_spec.read_text(encoding="utf-8"))
    shape = eta_shape(base)
    base = _validate_sigma_floor_spec(base)
    model, checkpoint, config = load_frozen_model(args.checkpoint, device)
    model.eval()

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

    def measure(p1: float, p2: float) -> dict[str, float]:
        spec = _validate_sigma_floor_spec(
            build_kernel_spec(args.basis, shape, p1, p2, base)
        )
        for step in model.decoder.steps:
            step.sigma_floor_spec = spec
            step.freeze_noise_amplitude = True
        widths = {}
        for index, (name, z) in enumerate(fixed_z.items()):
            matrix, _ = decode_mass_matrix(
                model,
                z,
                batch_size=args.batch_size,
                device=device,
                core=1.0,
                tail=0.0,
                draws=args.draws,
                seed=args.seed + 100 * index + int(1e5 * p1 + 1e3 * p2),
            )
            widths[name] = within_metrics(matrix)["within_robust_median_gev"]
        return widths

    p1_values = [float(value) for value in args.design_p1.split(",")]
    if len(p1_values) != 2:
        raise SystemExit("--design-p1 needs exactly two values")
    w_low = measure(p1_values[0], 0.0)
    w_high = measure(p1_values[1], 0.0)
    w_second = measure(p1_values[0], design_p2)

    targets = {name: TARGETS[name] for name in ("jpsi", "upsilon1s", "z")}
    targets["jpsi"] = float(args.jpsi_target)
    rows = []
    slopes: dict[str, tuple[float, float]] = {}
    for name, target in targets.items():
        c_first = (w_high[name] - w_low[name]) / (p1_values[1] - p1_values[0])
        c_second = (w_second[name] - w_low[name]) / design_p2
        slopes[name] = (c_first, c_second)
        rows.append([c_first / target, c_second / target])
    p1_fit, p2_fit = solve_relative(rows, list(targets.values()))

    validation = measure(p1_fit, p2_fit)
    fitted = _validate_sigma_floor_spec(build_kernel_spec(args.basis, shape, p1_fit, p2_fit, base))
    fitted["provenance"] = {
        "method": f"model-effective {args.basis} fit via decode probe (frozen mode)",
        "base_spec": str(args.base_spec),
        "checkpoint": str(args.checkpoint),
        "fitted_p1": p1_fit,
        "fitted_p2": p2_fit,
        "targets_robust_gev": targets,
    }
    names = list(TARGETS)
    payload = {
        "schema_version": 1,
        "diagnostic": "A2.3 model-effective kernel fit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "basis": args.basis,
        "checkpoint": str(args.checkpoint),
        "checkpoint_global_epoch": checkpoint.get("global_epoch"),
        "base_spec": str(args.base_spec),
        "design": {
            "p1": p1_values,
            "p2": design_p2,
            "widths_p1_low": w_low,
            "widths_p1_high": w_high,
            "widths_p2": w_second,
        },
        "coefficients": {name: list(value) for name, value in slopes.items()},
        "fit": {
            "p1": p1_fit,
            "p2": p2_fit,
            "targets": targets,
            "median_muon_pt_gev": {name: median_muon_pt(fixed_z[name]) for name in targets},
        },
        "validation": {
            name: {
                "achieved_robust_gev": validation[name],
                "target_robust_gev": targets.get(name, TARGETS[name]),
                "ratio": validation[name] / targets.get(name, TARGETS[name]),
            }
            for name in names
        },
        "fitted_spec": fitted,
    }
    (output_dir / "kernel_spec_fitted.json").write_text(
        json.dumps(fitted, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "fit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    p1_name, p2_name = ("C", "alpha") if args.basis == "power_law" else ("offset", "slope")
    lines = [
        "# A2.3 model-effective kernel fit",
        "",
        f"*{payload['created_utc']} - read-only inference, no training. Basis "
        f"`{args.basis}`; checkpoint `{Path(args.checkpoint).name}` "
        f"(epoch {checkpoint.get('global_epoch')}).*",
        "",
        "## Fitted kernel",
        "",
        f"- `{p1_name} = {p1_fit:.5g}`, `{p2_name} = {p2_fit:.5g}`",
        "",
        "## Validation at the fitted point (scale 1)",
        "",
        "| region | achieved robust [GeV] | target robust [GeV] | ratio |",
        "|---|---|---|---|",
    ]
    for name in names:
        entry = payload["validation"][name]
        lines.append(
            f"| {name} | {entry['achieved_robust_gev']:.4g} | "
            f"{entry['target_robust_gev']:.4g} | {entry['ratio']:.3f} |"
        )
    lines += [
        "",
        "Targets: J/psi physical 0.0281 GeV (`--jpsi-target 0.012183` for the "
        "smeared-prior additional target), Upsilon(1S) 0.084, Z 2.50188.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"basis={args.basis} fitted {p1_name}={p1_fit:.5g} {p2_name}={p2_fit:.5g}")
    for name in names:
        entry = payload["validation"][name]
        print(f"  {name}: {entry['achieved_robust_gev']:.4g} / {entry['target_robust_gev']:.4g} = {entry['ratio']:.3f}")
    print(f"wrote {output_dir / 'kernel_spec_fitted.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
