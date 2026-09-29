#!/usr/bin/env python
"""Calibrate the A2.3 kernel to the *model's* effective sigma->mass response.

The A2.0 analytic conversion (``sigma_logpT = sqrt(2) * sigma_m / m``) is only a
first-order guide: the model injects noise in two re-conditioned cylindrical
steps, so the mass width it produces per unit sigma differs from the analytic
value and depends on kinematics. This read-only script measures that response
with the decode probe on a small design grid, fits the linear model

    width_region(A, B) = cA_region * A + cB_region * B

for a kernel ``sigma_logpT(pT, eta) = shape(eta) * (A + B * pT)``, solves the
2-parameter least-squares problem against the physical targets, and writes a
revised kernel spec plus a validation measurement at the fitted point.

Nothing is trained; new artifacts only.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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


def shape_from_spec(spec: dict) -> list[float]:
    """Normalised eta shape of the A2.0 log-pT amplitude curve."""
    values = np.asarray(spec["logpt_a"], dtype=float)
    if np.all(values <= 0.0):
        raise ValueError("base spec has an all-zero logpt_a shape")
    return (values / values.mean()).tolist()


def build_spec(shape: list[float], a: float, b: float, base: dict) -> dict:
    """Kernel spec with sigma_logpT(pT) = shape(eta) * (a + b * pT)."""
    if a < 0.0 or b < 0.0:
        raise ValueError("a and b must be non-negative")
    spec = copy.deepcopy(base)
    spec["logpt_a"] = [float(a * value) for value in shape]
    spec["logpt_b"] = [float(b * value) for value in shape]
    spec.pop("scale", None)
    return spec


def solve_effective_scale(
    coefficients: dict[str, tuple[float, float]],
    targets: dict[str, float],
) -> tuple[float, float]:
    """Least-squares (A, B) for width_r = cA_r * A + cB_r * B = target_r."""
    rows = []
    values = []
    for region, target in targets.items():
        c_a, c_b = coefficients[region]
        rows.append([c_a / target, c_b / target])
        values.append(1.0)
    solution, *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(values), rcond=None)
    a, b = (float(solution[0]), float(solution[1]))
    return max(a, 0.0), max(b, 0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-spec", type=Path, required=True)
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
    parser.add_argument(
        "--design-a",
        default="0.004,0.012",
        help="two A values, both measured at B=0",
    )
    parser.add_argument(
        "--design-b",
        default="0.0001,0.0003",
        help="two B values, both measured at the first A",
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

    base_spec = json.loads(args.base_spec.read_text(encoding="utf-8"))
    shape = shape_from_spec(base_spec)
    base_spec = _validate_sigma_floor_spec(base_spec)
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
    print(f"fixed z: " + ", ".join(f"{name}={len(z)}" for name, z in fixed_z.items()))

    def measure(a: float, b: float) -> dict[str, float]:
        spec = _validate_sigma_floor_spec(build_spec(shape, a, b, base_spec))
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
                seed=args.seed + 100 * index + int(1e6 * (a + b)),
            )
            widths[name] = within_metrics(matrix)["within_robust_median_gev"]
        return widths

    design_a = [float(value) for value in args.design_a.split(",")]
    design_b = [float(value) for value in args.design_b.split(",")]
    if len(design_a) != 2 or len(design_b) != 2:
        raise SystemExit("--design-a and --design-b each need exactly two values")

    measurements: list[dict] = []
    for a in design_a:
        widths = measure(a, 0.0)
        measurements.append({"a": a, "b": 0.0, "widths": widths})
        print(f"A={a:g}, B=0: " + ", ".join(f"{k} {v:.4g}" for k, v in widths.items()))
    for b in design_b:
        widths = measure(design_a[0], b)
        measurements.append({"a": design_a[0], "b": b, "widths": widths})
        print(f"A={design_a[0]:g}, B={b:g}: " + ", ".join(f"{k} {v:.4g}" for k, v in widths.items()))

    coefficients: dict[str, tuple[float, float]] = {}
    for name in fixed_z:
        w_a0 = next(m for m in measurements if m["a"] == design_a[0] and m["b"] == 0.0)["widths"][name]
        w_a1 = next(m for m in measurements if m["a"] == design_a[1] and m["b"] == 0.0)["widths"][name]
        w_b0 = w_a0
        w_b1 = next(m for m in measurements if m["a"] == design_a[0] and m["b"] == design_b[1])["widths"][name]
        c_a = (w_a1 - w_a0) / (design_a[1] - design_a[0])
        c_b = (w_b1 - w_b0) / (design_b[1] - design_b[0])
        coefficients[name] = (c_a, c_b)
        print(f"{name}: cA={c_a:.6g} GeV per unit A, cB={c_b:.6g} GeV per unit B")

    fit_targets = {
        "jpsi": TARGETS["jpsi"],
        "upsilon1s": TARGETS["upsilon1s"],
        "z": TARGETS["z"],
    }
    a_fit, b_fit = solve_effective_scale(
        {name: coefficients[name] for name in fit_targets}, fit_targets
    )
    print(f"fitted effective kernel: A={a_fit:.6g}, B={b_fit:.6g}")

    fitted_spec = _validate_sigma_floor_spec(build_spec(shape, a_fit, b_fit, base_spec))
    fitted_spec["provenance"] = {
        "method": "effective model-response calibration (decode probe, frozen mode)",
        "base_spec": str(args.base_spec),
        "checkpoint": str(args.checkpoint),
        "fitted_a": a_fit,
        "fitted_b": b_fit,
        "targets_robust_gev": fit_targets,
        "shape_from": "base spec logpt_a normalised to its mean",
    }
    validation = measure(a_fit, b_fit)
    recommendations = {}
    for name, width in validation.items():
        target = TARGETS[name]
        recommendations[name] = {
            "achieved_robust_gev": width,
            "target_robust_gev": target,
            "ratio": width / target,
        }

    payload = {
        "schema_version": 1,
        "diagnostic": "A2.3 effective kernel calibration",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_global_epoch": checkpoint.get("global_epoch"),
        "base_spec": str(args.base_spec),
        "design": measurements,
        "coefficients": {k: list(v) for k, v in coefficients.items()},
        "fit": {"a": a_fit, "b": b_fit, "targets": fit_targets},
        "validation": recommendations,
        "fitted_spec": fitted_spec,
    }
    (output_dir / "kernel_spec_model_effective.json").write_text(
        json.dumps(fitted_spec, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "calibration.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# A2.3 effective kernel calibration",
        "",
        f"*{payload['created_utc']} - read-only inference, no training. Checkpoint "
        f"`{Path(args.checkpoint).name}` (epoch {checkpoint.get('global_epoch')}), "
        f"{args.events} fixed z, {args.draws} draws.*",
        "",
        "## Model response coefficients",
        "",
        "| region | cA [GeV per unit A] | cB [GeV per unit B] |",
        "|---|---|---|",
    ]
    for name, (c_a, c_b) in coefficients.items():
        lines.append(f"| {name} | {c_a:.5g} | {c_b:.5g} |")
    lines += [
        "",
        f"## Fitted kernel: sigma_logpT(pT, eta) = shape(eta) * (A + B * pT)",
        "",
        f"- **A = {a_fit:.5g}**, **B = {b_fit:.5g}** (per GeV of muon pT)",
        "",
        "## Validation at the fitted point (scale 1)",
        "",
        "| region | achieved robust [GeV] | target robust [GeV] | ratio |",
        "|---|---|---|---|",
    ]
    for name, entry in recommendations.items():
        lines.append(
            f"| {name} | {entry['achieved_robust_gev']:.4g} | "
            f"{entry['target_robust_gev']:.4g} | {entry['ratio']:.3f} |"
        )
    lines.append("")
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    names = list(fit_targets)
    ratio = [recommendations[name]["ratio"] for name in names]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(names, ratio, color="#2b6cb0")
    axis.axhline(1.0, color="black", linestyle=":")
    axis.set_ylabel("achieved / target")
    axis.set_title("A2.3 effective kernel validation (scale 1)")
    figure.tight_layout()
    figure.savefig(output_dir / "kernel_validation.png", dpi=150)
    plt.close(figure)

    print(f"wrote {output_dir / 'kernel_spec_model_effective.json'}")
    print(f"wrote {output_dir / 'calibration.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
