#!/usr/bin/env python
"""Per-state Upsilon peak medians vs the CMS fit, for arbitrary joint checkpoints.

Read-only. This is the A1 / D2 acceptance readout that the Run-H-specific
`runH_tail_audit.py` produces only for Run H's own checkpoint names: decode a
fixed Upsilon subset at each checkpoint's native multipliers and report the
1S/2S/3S median (and mode) minus the CMS fit, the decoded widths and the
window efficiency.

Nothing is trained and no existing artifact is modified; new files only.
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
for directory in (
    REPO_ROOT / "scripts",
    REPO_ROOT / "scripts_sota",
    REPO_ROOT / "scripts_joint",
    REPO_ROOT / "scripts_joint" / "upsilon",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from decode_prior import load_frozen_model  # noqa: E402
from runH_tail_audit import CMS_FIT_MASS_GEV, load_z, mass8, state_moments  # noqa: E402

STATE_COMPONENTS = ("upsilon1s", "upsilon2s", "upsilon3s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="LABEL=RELATIVE_PATH",
        help="checkpoint to decode (repeatable)",
    )
    parser.add_argument(
        "--prior",
        type=Path,
        default=REPO_ROOT / "data" / "upsilon_prior_continuumReweighted.hdf5",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--max-events", type=int, default=400000)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--mass-min", type=float, default=8.5)
    parser.add_argument("--mass-max", type=float, default=11.5)
    parser.add_argument("--mass-bin-width", type=float, default=0.02)
    return parser.parse_args()


def decode_native(model, z: np.ndarray, *, batch_size: int, device: torch.device) -> np.ndarray:
    outputs = []
    with torch.no_grad():
        for start in range(0, len(z), batch_size):
            values = torch.as_tensor(
                np.ascontiguousarray(z[start : start + batch_size]),
                dtype=torch.float32,
                device=device,
            )
            outputs.append(model.decode(values).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def compute_moments(x: np.ndarray, component_id: np.ndarray, args) -> dict:
    moments = state_moments(
        x, component_id, args.mass_min, args.mass_max, args.mass_bin_width
    )
    for value in moments.values():
        value["median_minus_cms_gev"] = float(
            value["median_gev"] - value["cms_fit_mass_gev"]
        )
    return moments


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        ("cpu" if args.device == "auto" else args.device)
    )

    checkpoints: list[tuple[str, Path]] = []
    for raw in args.checkpoint:
        if "=" not in raw:
            raise SystemExit(f"--checkpoint expects LABEL=PATH, got {raw!r}")
        label, relative = raw.split("=", 1)
        path = (REPO_ROOT / relative).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        checkpoints.append((label, path))

    z, component_id = load_z(args.prior, args.max_events, args.seed)
    prior_moments = compute_moments(z, component_id, args)
    print(f"prior modes: 1S {prior_moments['upsilon1s']['mode_bin_center_gev']:.3f} "
          f"2S {prior_moments['upsilon2s']['mode_bin_center_gev']:.3f} "
          f"3S {prior_moments['upsilon3s']['mode_bin_center_gev']:.3f}")

    results: dict[str, dict] = {}
    for label, path in checkpoints:
        model, checkpoint, _ = load_frozen_model(path, device)
        model.eval()
        noise = checkpoint.get("noise_multipliers") or {}
        shared_core = float(noise.get("core", 1.0))
        shared_tail = float(noise.get("tail", 1.0))
        core = float(noise.get("decoder_core", shared_core))
        tail = float(noise.get("decoder_tail", shared_tail))
        model.decoder.set_noise_multipliers(core, tail)
        decoded = decode_native(model, z, batch_size=args.batch_size, device=device)
        results[label] = {
            "path": str(path),
            "global_epoch": checkpoint.get("global_epoch"),
            "native_decoder_multipliers": {"core": core, "tail": tail},
            "state_moments": compute_moments(decoded, component_id, args),
        }
        medians = " / ".join(
            f"{results[label]['state_moments'][state]['median_minus_cms_gev'] * 1000:+.0f}"
            for state in STATE_COMPONENTS
        )
        print(f"{label}: median-CMS [MeV] 1S/2S/3S = {medians}")

    drift = {}
    for state in STATE_COMPONENTS:
        spreads = {
            f"{a}_minus_{b}": float(
                results[a]["state_moments"][state]["median_gev"]
                - results[b]["state_moments"][state]["median_gev"]
            )
            for i, a in enumerate(results)
            for b in list(results)[:i]
        }
        drift[state] = spreads

    payload = {
        "schema_version": 1,
        "diagnostic": "Upsilon per-state peak medians vs CMS fit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "prior": str(args.prior),
        "max_events": args.max_events,
        "seed": args.seed,
        "device": str(device),
        "cms_fit_mass_gev": CMS_FIT_MASS_GEV,
        "prior_state_moments": prior_moments,
        "checkpoints": results,
        "pairwise_median_drift_gev": drift,
    }
    (output_dir / "upsilon_peak_medians.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Upsilon per-state peak medians vs the CMS fit",
        "",
        f"*{payload['created_utc']} - read-only inference. Prior `{args.prior.name}`, "
        f"{len(z)} events, seed {args.seed}, device `{device}`.*",
        "",
        "| checkpoint | native (core, tail) | 1S median-CMS [MeV] | 2S [MeV] | 3S [MeV] | "
        "1S mode-CMS [MeV] | 1S std [MeV] | 1S robust [MeV] |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label, entry in results.items():
        m = entry["state_moments"]
        native = entry["native_decoder_multipliers"]
        lines.append(
            f"| {label} | ({native['core']}, {native['tail']}) | "
            f"{m['upsilon1s']['median_minus_cms_gev'] * 1000:+.1f} | "
            f"{m['upsilon2s']['median_minus_cms_gev'] * 1000:+.1f} | "
            f"{m['upsilon3s']['median_minus_cms_gev'] * 1000:+.1f} | "
            f"{m['upsilon1s']['mode_minus_cms_gev'] * 1000:+.1f} | "
            f"{m['upsilon1s']['std_gev'] * 1000:.1f} | "
            f"{(m['upsilon1s']['q84_gev'] - m['upsilon1s']['q16_gev']) * 500:.1f} |"
        )
    lines.append("")
    lines.append("## Pairwise median drift [MeV]")
    lines.append("")
    for state, spreads in drift.items():
        body = ", ".join(f"{name}: {value * 1000:+.1f}" for name, value in spreads.items())
        lines.append(f"- `{state}`: {body}")
    lines.append("")
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output_dir / 'upsilon_peak_medians.json'}")
    print(f"wrote {output_dir / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
