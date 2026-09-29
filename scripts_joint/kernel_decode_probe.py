#!/usr/bin/env python
"""Decode-only calibration probe for the A2.3 physics resolution kernel.

Read-only. Injects a kernel spec (with a swept overall ``scale``) into the
decoder steps of a *trained* checkpoint, decodes a fixed truth sample, and
measures the achieved within-z mass width at fixed z against the target
detector resolution. It answers the pre-training question "does this kernel
actually produce the resolution we measured?" and returns the scale that does.

Nothing is trained; the checkpoint's mean map is untouched, only the noise
amplitude policy changes at inference. New artifacts only.
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
    robust_half_width,
)
from cylindrical_flow import _validate_sigma_floor_spec  # noqa: E402


UPSILON_STATES = ("upsilon1s", "upsilon2s", "upsilon3s")
COMPONENT_MAP = {"upsilon1s": 0, "upsilon2s": 1, "upsilon3s": 2}

# Target *additional* resolved smearing (robust) per region/state, from A0.
TARGETS = {
    "jpsi": 0.012183,
    "z": 2.50188,
    "upsilon1s": 0.084,
    "upsilon2s": 0.084,
    "upsilon3s": 0.084,
}


def apply_scale(spec: dict, scale: float) -> dict:
    scaled = copy.deepcopy(spec)
    scaled["scale"] = float(scale)
    return scaled


def within_metrics(matrix: np.ndarray) -> dict:
    """Within-z and variance-decomposition statistics for (N, draws) masses."""
    within_std = matrix.std(axis=1)
    within_robust = (
        np.quantile(matrix, 0.84, axis=1) - np.quantile(matrix, 0.16, axis=1)
    ) / 2.0
    within_variance = within_std**2
    total_variance = float(matrix.var())
    return {
        "ensemble_std_gev": float(matrix.std()),
        "within_std_median_gev": float(np.median(within_std)),
        "within_robust_median_gev": float(np.median(within_robust)),
        "sigma_only_within_gev": float(np.sqrt(max(within_variance.mean(), 0.0))),
        "noise_variance_fraction": (
            float(within_variance.mean() / total_variance) if total_variance > 0 else 0.0
        ),
    }


def recommend_scale(points: list[tuple[float, float]], target: float) -> float:
    """Smallest sampled scale whose width reaches ``target`` (else best match).

    ``points`` is a sorted list of (scale, width). Linear interpolation between
    the bracketing samples gives the crossing; when the target is never reached
    the closest sampled width wins.
    """
    if not points:
        raise ValueError("points is empty")
    ordered = sorted(points)
    for (scale_a, width_a), (scale_b, width_b) in zip(ordered, ordered[1:]):
        if (width_a - target) * (width_b - target) <= 0 and width_b != width_a:
            fraction = (target - width_a) / (width_b - width_a)
            return float(scale_a + fraction * (scale_b - scale_a))
        if width_a >= target:
            return float(scale_a)
    if ordered[0][1] >= target:
        return float(ordered[0][0])
    return float(min(ordered, key=lambda item: abs(item[1] - target))[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "cms_Joint"
        / "Run_H_cycleNoNoise"
        / "best_RunHcycleNoNoise_stage2_stochastic_core.pt",
    )
    parser.add_argument("--spec", type=Path, required=True, help="kernel spec JSON")
    parser.add_argument("--mode", choices=("frozen", "floor"), default="frozen")
    parser.add_argument("--scales", default="0.5,1.0,1.5,2.0,3.0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--events", type=int, default=256)
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260923)
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
    spec_raw = json.loads(args.spec.read_text(encoding="utf-8"))
    base_spec = _validate_sigma_floor_spec(spec_raw)
    scales = [float(value) for value in args.scales.split(",") if value.strip()]

    model, checkpoint, config = load_frozen_model(args.checkpoint, device)
    model.eval()
    native = checkpoint.get("noise_multipliers") or {}
    print(
        f"checkpoint {args.checkpoint.name} global_epoch {checkpoint.get('global_epoch')} "
        f"native decoder {native.get('decoder_core', native.get('core'))}/"
        f"{native.get('decoder_tail', native.get('tail'))}"
    )

    # Fixed z samples: J/psi and Z from the checkpoint's own prior paths,
    # Upsilon from the continuum-reweighted evaluation prior.
    fixed_z: dict[str, np.ndarray] = {}
    for offset, region in enumerate(("jpsi", "z")):
        path, _, _ = resolve_prior_path(config, region)
        z, _ = load_fixed_z(
            path, events=args.events, seed=args.seed + offset
        )
        fixed_z[region] = z
        print(f"{region}: {len(z)} fixed z from {path.name}")
    upsilon_path = REPO_ROOT / "data" / "upsilon_prior_continuumReweighted.hdf5"
    for index, state in enumerate(UPSILON_STATES):
        z, _ = load_fixed_z(
            upsilon_path,
            events=args.events,
            seed=args.seed + 10 + index,
            component_id=COMPONENT_MAP[state],
        )
        fixed_z[state] = z
        print(f"{state}: {len(z)} fixed z")

    # Zero-noise reference: the mean map alone.
    zero_metrics: dict[str, dict] = {}
    model.decoder.set_noise_multipliers(0.0, 0.0)
    for name, z in fixed_z.items():
        matrix, _ = decode_mass_matrix(
            model,
            z,
            batch_size=args.batch_size,
            device=device,
            core=0.0,
            tail=0.0,
            draws=max(2, args.draws),
            seed=args.seed + 500,
        )
        zero_metrics[name] = within_metrics(matrix)

    results: dict[str, dict] = {}
    for scale in scales:
        spec = apply_scale(base_spec, scale)
        for step in model.decoder.steps:
            step.sigma_floor_spec = _validate_sigma_floor_spec(spec)
            step.freeze_noise_amplitude = args.mode == "frozen"
        scale_entry: dict[str, dict] = {}
        for index, (name, z) in enumerate(fixed_z.items()):
            matrix, _ = decode_mass_matrix(
                model,
                z,
                batch_size=args.batch_size,
                device=device,
                core=1.0,
                tail=0.0,
                draws=args.draws,
                seed=args.seed + 100 * index + int(scale * 100),
            )
            metrics = within_metrics(matrix)
            metrics["target_robust_gev"] = TARGETS[name]
            metrics["ratio_to_target"] = metrics["within_robust_median_gev"] / TARGETS[name]
            metrics["zero_noise_ensemble_std_gev"] = zero_metrics[name]["ensemble_std_gev"]
            scale_entry[name] = metrics
        results[f"{scale:g}"] = scale_entry
        summary = ", ".join(
            f"{name} {scale_entry[name]['within_robust_median_gev'] * 1000:.1f} MeV"
            for name in ("upsilon1s", "jpsi", "z")
        )
        print(f"scale {scale:g}: {summary}")

    recommendations = {}
    for name in fixed_z:
        points = [
            (float(scale), results[f"{scale:g}"][name]["within_robust_median_gev"])
            for scale in scales
        ]
        recommendations[name] = {
            "target_robust_gev": TARGETS[name],
            "recommended_scale": recommend_scale(points, TARGETS[name]),
            "points": [{"scale": s, "width": w} for s, w in points],
        }

    payload = {
        "schema_version": 1,
        "diagnostic": "A2.3 decode-only kernel calibration probe",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_global_epoch": checkpoint.get("global_epoch"),
        "mode": args.mode,
        "spec": str(args.spec),
        "scales": scales,
        "events": args.events,
        "draws": args.draws,
        "device": str(device),
        "targets_robust_gev": TARGETS,
        "zero_noise": zero_metrics,
        "results": results,
        "recommendations": recommendations,
    }
    (output_dir / "kernel_decode_probe.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# A2.3 decode-only kernel calibration probe",
        "",
        f"*{payload['created_utc']} - read-only inference, no training. "
        f"Mode `{args.mode}`, {args.events} fixed z, {args.draws} draws per scale.*",
        "",
        "Achieved within-z robust mass width [GeV] versus the kernel scale:",
        "",
        "| scale | " + " | ".join(fixed_z) + " |",
        "|---" * (len(fixed_z) + 1) + "|",
    ]
    for scale in scales:
        row = [f"{scale:g}"]
        for name in fixed_z:
            value = results[f"{scale:g}"][name]["within_robust_median_gev"]
            row.append(f"{value:.4g}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("Targets (A0, robust): " + ", ".join(
        f"{name} {TARGETS[name]:.4g} GeV" for name in fixed_z
    ))
    lines.append("")
    lines.append("## Recommended scales to hit the target")
    lines.append("")
    for name, entry in recommendations.items():
        lines.append(
            f"- `{name}`: recommended scale **{entry['recommended_scale']:.3f}** "
            f"(target {entry['target_robust_gev']:.4g} GeV) from points "
            + ", ".join(f"({p['scale']:g}, {p['width']:.4g})" for p in entry["points"])
        )
    lines.append("")
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    groups = (
        (axes[0], ("upsilon1s", "upsilon2s", "upsilon3s"), "Upsilon"),
        (axes[1], ("jpsi",), "J/psi"),
        (axes[2], ("z",), "Z"),
    )
    for axis, names, title in groups:
        for name in names:
            widths = [results[f"{scale:g}"][name]["within_robust_median_gev"] for scale in scales]
            axis.plot(scales, widths, marker="o", label=name)
            axis.axhline(TARGETS[name], linestyle=":", color="black", linewidth=0.8)
        axis.set_xlabel("kernel scale")
        axis.set_ylabel("within-z robust width [GeV]")
        axis.set_title(title)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "kernel_decode_probe.png", dpi=150)
    figure.savefig(output_dir / "kernel_decode_probe.pdf")
    plt.close(figure)

    print(f"wrote {output_dir / 'kernel_decode_probe.json'}")
    print(f"wrote {output_dir / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
