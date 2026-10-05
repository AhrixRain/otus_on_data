#!/usr/bin/env python
"""Upsilon noise-channel test: decode the same events at zero and native noise.

Read-only.  Answers the D3b transfer question directly: is the decoded Upsilon
peak shape (position, width, low-mass shoulder) carried by the response
channel, or is it already in the deterministic mean map?

For every configured noise condition the script decodes the *same* subset of
each signal component of a continuum-reweighted Upsilon prior, then reports
per-state moments against the prior and against the CMS three-peak fit:

    zero      decoder multipliers (0, 0)      -> the deterministic mean map
    native    the checkpoint's own multipliers -> the response channel on

If the 1S width and the low-mass shoulder are unchanged between the two, the
mean map carries them; if they shrink when the channel is switched off, the
channel is manufacturing them.

Nothing is trained; no checkpoint, data file or existing output is modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import h5py  # noqa: E402

HERE = Path(__file__).resolve().parent


def find_repo_root() -> Path:
    for candidate in HERE.parents:
        if (candidate / "scripts_joint").is_dir() and (candidate / "scripts_sota").is_dir():
            return candidate
    raise RuntimeError("Could not locate the OTUS repository root")


REPO_ROOT = find_repo_root()
for directory in (
    REPO_ROOT / "scripts",
    REPO_ROOT / "scripts_sota",
    REPO_ROOT / "scripts_joint",
    REPO_ROOT / "scripts_joint" / "upsilon",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from decode_prior import load_frozen_model  # noqa: E402

STATES = (("upsilon1s", 0), ("upsilon2s", 1), ("upsilon3s", 2))
CMS_FIT_MASS_GEV = {"upsilon1s": 9.445066420298902, "upsilon2s": 10.01627393522132,
                    "upsilon3s": 10.34168938611508}
CMS_SIGMA_GEV = 0.0844
MASS_WINDOW = (8.5, 11.5)
PLOT_WINDOW = (8.6, 10.9)


def mass8(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    energy = values[:, 3] + values[:, 7]
    momentum = values[:, :3] + values[:, 4:7]
    return np.sqrt(np.maximum(energy * energy - np.sum(momentum * momentum, axis=1), 0.0))


def mass8_torch(values: torch.Tensor) -> torch.Tensor:
    energy = values[:, 3] + values[:, 7]
    momentum = values[:, :3] + values[:, 4:7]
    return torch.sqrt(
        torch.clamp(energy * energy - torch.sum(momentum * momentum, dim=1), min=0.0)
    )


def robust_half_width(values: np.ndarray) -> float:
    return float((np.quantile(values, 0.84) - np.quantile(values, 0.16)) / 2.0)


def decode(model, z: np.ndarray, *, core: float, tail: float, batch_size: int,
           device: torch.device, seed: int) -> np.ndarray:
    model.decoder.set_noise_multipliers(core, tail)
    torch.manual_seed(int(seed))
    masses = np.empty(len(z), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(z), batch_size):
            block = slice(start, min(start + batch_size, len(z)))
            values = torch.as_tensor(
                np.ascontiguousarray(z[block]), dtype=torch.float32, device=device
            )
            decoded = model.decode(values)
            masses[block] = mass8_torch(decoded).detach().cpu().numpy()
    return masses


def state_report(decoded: np.ndarray, prior: np.ndarray, state: str) -> dict:
    cms = CMS_FIT_MASS_GEV[state]
    lower, upper = cms - 0.3, cms + 0.3
    in_peak = decoded[(decoded > lower) & (decoded < upper)]
    report = {
        "events": int(len(decoded)),
        "decoded_median_gev": float(np.median(decoded)),
        "decoded_std_gev": float(np.std(decoded)),
        "decoded_robust_half_width_gev": robust_half_width(decoded),
        "decoded_q16_gev": float(np.quantile(decoded, 0.16)),
        "decoded_q84_gev": float(np.quantile(decoded, 0.84)),
        "median_minus_cms_gev": float(np.median(decoded) - cms),
        "std_over_cms_resolution": float(np.std(decoded) / CMS_SIGMA_GEV),
        "peak_window_std_gev": float(np.std(in_peak)) if in_peak.size > 10 else None,
        "peak_window_robust_half_width_gev": robust_half_width(in_peak)
        if in_peak.size > 10
        else None,
        "low_mass_shoulder_fraction": float(np.mean(decoded < cms - 0.06)),
        "prior_median_gev": float(np.median(prior)),
        "prior_std_gev": float(np.std(prior)),
        "added_width_quadrature_gev": float(
            np.sqrt(max(np.std(decoded) ** 2 - np.std(prior) ** 2, 0.0))
        ),
        "window_efficiency": float(np.mean((decoded > MASS_WINDOW[0]) & (decoded < MASS_WINDOW[1]))),
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--prior",
        type=Path,
        default=REPO_ROOT / "data/upsilon_prior_continuumReweighted.hdf5",
    )
    parser.add_argument("--prior-dataset", default="FDL/zData")
    parser.add_argument("--component-dataset", default="FDL/component_id")
    parser.add_argument("--max-events-per-state", type=int, default=150000)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/cms_Joint/Run_H_D3b/upsilon_noise_test",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    model, checkpoint, _ = load_frozen_model(args.checkpoint, device)
    model.eval()
    recorded = checkpoint.get("noise_multipliers") or {}
    shared_core = float(recorded.get("core", 1.0))
    shared_tail = float(recorded.get("tail", 1.0))
    native = (
        float(recorded.get("decoder_core", shared_core)),
        float(recorded.get("decoder_tail", shared_tail)),
    )
    conditions = [("zero", 0.0, 0.0), ("native", native[0], native[1])]

    with h5py.File(args.prior, "r") as source:
        z_all = np.asarray(source[args.prior_dataset][:], dtype=np.float32)
        component = np.asarray(source[args.component_dataset][:])

    rng = np.random.default_rng(args.seed)
    subsets: dict[str, np.ndarray] = {}
    for state, component_id in STATES:
        index = np.flatnonzero(component == component_id)
        if index.size == 0:
            raise SystemExit(f"component {component_id} empty in {args.prior}")
        count = min(int(args.max_events_per_state), index.size)
        chosen = np.sort(rng.choice(index, size=count, replace=False))
        subsets[state] = np.ascontiguousarray(z_all[chosen])

    payload = {
        "schema_version": 1,
        "diagnostic": "Upsilon response-channel test (zero vs native noise)",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "global_epoch": checkpoint.get("global_epoch"),
        "recorded_noise_multipliers": recorded,
        "native_decoder_multipliers": {"core": native[0], "tail": native[1]},
        "prior": str(args.prior),
        "max_events_per_state": args.max_events_per_state,
        "seed": args.seed,
        "cms_fit_mass_gev": CMS_FIT_MASS_GEV,
        "cms_resolution_gev": CMS_SIGMA_GEV,
        "conditions": {},
    }

    spectra: dict[tuple[str, str], np.ndarray] = {}
    for label, core, tail in conditions:
        entry = {"multipliers": {"core": core, "tail": tail}, "states": {}}
        for state, _ in STATES:
            z = subsets[state]
            prior_masses = mass8(z)
            decoded = decode(
                model,
                z,
                core=core,
                tail=tail,
                batch_size=args.batch_size,
                device=device,
                seed=args.seed + (0 if label == "zero" else 101),
            )
            entry["states"][state] = state_report(decoded, prior_masses, state)
            spectra[(label, state)] = decoded
            print(
                f"[{label}] {state}: median-CMS "
                f"{entry['states'][state]['median_minus_cms_gev'] * 1000:+.1f} MeV, "
                f"std {entry['states'][state]['decoded_std_gev'] * 1000:.1f} MeV, "
                f"shoulder {entry['states'][state]['low_mass_shoulder_fraction']:.3f}"
            )
        payload["conditions"][label] = entry

    json_path = output_dir / "upsilon_noise_test.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # ---- overlay figure --------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True)
    edges = np.linspace(PLOT_WINDOW[0], PLOT_WINDOW[1], 161)
    colors = {"zero": "#4c72b0", "native": "#c44e52"}
    for ax, (state, label) in zip(axes, (("upsilon1s", "Y(1S)"), ("upsilon2s", "Y(2S)"),
                                         ("upsilon3s", "Y(3S)"))):
        prior_masses = mass8(subsets[state])
        counts, _ = np.histogram(prior_masses, bins=edges)
        ax.step(0.5 * (edges[:-1] + edges[1:]), counts, where="mid",
                color="0.55", ls="--", lw=1.4, label="prior z")
        for condition in ("zero", "native"):
            decoded = spectra[(condition, state)]
            counts, _ = np.histogram(decoded, bins=edges)
            ax.step(0.5 * (edges[:-1] + edges[1:]), counts, where="mid",
                    color=colors[condition], lw=1.9,
                    label=f"decoded, {condition} noise")
        cms_mass = CMS_FIT_MASS_GEV[state]
        ax.axvline(cms_mass, color="black", ls=":", lw=1.0)
        ax.set_title(f"{label}  (CMS fit {cms_mass:.4f} GeV)")
        ax.set_xlabel(r"$m_{\mu\mu}$ [GeV]")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Events / 14.4 MeV")
    axes[0].legend(fontsize=8.5, loc="upper right")
    fig.suptitle(
        "Upsilon peak shape at zero vs native noise (same events) - "
        f"{args.checkpoint.parent.name} ep {checkpoint.get('global_epoch')}"
    )
    fig.tight_layout()
    png = output_dir / "upsilon_zero_vs_native_noise.png"
    pdf = output_dir / "upsilon_zero_vs_native_noise.pdf"
    fig.savefig(png, dpi=160, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    # ---- markdown summary ------------------------------------------------
    lines = [
        "# Upsilon response-channel test - zero vs native noise",
        "",
        f"checkpoint: `{args.checkpoint}` (global epoch {checkpoint.get('global_epoch')}), "
        f"native decoder multipliers {native}",
        f"prior: `{args.prior}`, {args.max_events_per_state} events per state, "
        f"the same events in both conditions.",
        "",
        "| state | condition | median $-$ CMS [MeV] | std [MeV] | std / 84.4 | "
        "peak-window std [MeV] | q16 $-$ CMS [MeV] | added width [MeV] |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for state, label in (("upsilon1s", "1S"), ("upsilon2s", "2S"), ("upsilon3s", "3S")):
        for condition in ("zero", "native"):
            s = payload["conditions"][condition]["states"][state]
            lines.append(
                f"| {label} | {condition} | {s['median_minus_cms_gev'] * 1000:+.1f} | "
                f"{s['decoded_std_gev'] * 1000:.1f} | {s['std_over_cms_resolution']:.2f} | "
                f"{(s['peak_window_std_gev'] or float('nan')) * 1000:.1f} | "
                f"{(s['decoded_q16_gev'] - CMS_FIT_MASS_GEV[state]) * 1000:+.1f} | "
                f"{s['added_width_quadrature_gev'] * 1000:.1f} |"
            )
    lines += [
        "",
        "Prior (before decoding), std [MeV]: "
        + ", ".join(
            f"{label} {np.std(mass8(subsets[state])) * 1000:.1f}"
            for state, label in (("upsilon1s", "1S"), ("upsilon2s", "2S"), ("upsilon3s", "3S"))
        ),
        "",
        f"Figure: `{png.name}` / `{pdf.name}`",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
