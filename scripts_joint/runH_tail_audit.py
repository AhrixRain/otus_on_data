#!/usr/bin/env python
"""Inference-time audit of the Run H core/tail noise schedule.

Two questions this answers, with the trained Run H checkpoints read-only:

1. Schedule verification: what core/tail multipliers did every epoch of every
   stage actually run, and were encoder and decoder shared?  Read from
   ``history.json``.

2. Peak-shift mechanism: the decoded Upsilon peaks move down by ~100 MeV when
   the tail multiplier is raised from 0.25 (stage 2) to 0.5 (stage 3).  Is that
   the *sampled* heavy-tail noise (overflow), or is it baked into the learned
   deterministic mean map?

   Method: decode one fixed Upsilon subset with a grid of (core, tail)
   inference multipliers, on several checkpoints.  In particular

   * stage-3 checkpoint at tail 0 / 0.25 / 0.5 / 1.0   -> is the shift sampled?
   * stage-2 checkpoint at tail 0.5                     -> does forcing tail
                                                          noise onto a healthy
                                                          checkpoint move it?
   * every checkpoint at (0, 0)                         -> the mean-map drift
                                                          across stages.

   We also record the learned per-step ``core_sigma`` / ``tail_sigma`` values
   on the same events, so "the model cannot learn the tail" is measured rather
   than asserted.

Outputs JSON + REPORT.md + PNG under ``<output-dir>``.  Nothing is modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def find_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
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


STATE_COMPONENTS = ("upsilon1s", "upsilon2s", "upsilon3s")
COMPONENT_MAP = {"upsilon1s": 0, "upsilon2s": 1, "upsilon3s": 2, "continuum": 3}
CMS_FIT_MASS_GEV = {
    "upsilon1s": 9.445066420298902,
    "upsilon2s": 10.016273935221268,
    "upsilon3s": 10.34168938611508,
}


def mass8(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    energy = values[:, 3] + values[:, 7]
    momentum = values[:, :3] + values[:, 4:7]
    return np.sqrt(np.maximum(energy * energy - np.sum(momentum * momentum, axis=1), 0.0))


def state_moments(
    x: np.ndarray,
    component_id: np.ndarray,
    mass_min: float,
    mass_max: float,
    bin_width: float,
) -> dict[str, dict[str, float | int]]:
    values = mass8(x)
    edges = np.arange(mass_min, mass_max + 0.5 * bin_width, bin_width)
    centers = 0.5 * (edges[:-1] + edges[1:])
    out: dict[str, dict[str, float | int]] = {}
    for name in STATE_COMPONENTS:
        selected = component_id == COMPONENT_MAP[name]
        sample = values[selected]
        counts, _ = np.histogram(sample, bins=edges)
        out[name] = {
            "events": int(np.sum(selected)),
            "mean_gev": float(np.mean(sample)),
            "std_gev": float(np.std(sample)),
            "median_gev": float(np.median(sample)),
            "q16_gev": float(np.quantile(sample, 0.16)),
            "q84_gev": float(np.quantile(sample, 0.84)),
            "mode_bin_center_gev": float(centers[np.argmax(counts)]),
            "window_efficiency": float(np.mean((sample >= mass_min) & (sample <= mass_max))),
            "cms_fit_mass_gev": CMS_FIT_MASS_GEV[name],
            "mode_minus_cms_gev": float(centers[np.argmax(counts)] - CMS_FIT_MASS_GEV[name]),
        }
    return out


class SigmaRecorder:
    """Forward hooks on each decoder head recording learned core/tail sigma."""

    def __init__(self, model):
        self.records: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
        self.handles = []
        for index, step in enumerate(model.decoder.steps):
            self.handles.append(step.head.register_forward_hook(self._make(index, step)))

    def _make(self, index, step):
        def hook(module, inputs, output):  # noqa: ANN001
            _, core_raw, tail_raw = output.split(6, dim=1)
            core_sigma = step.core_sigma_floors + F.softplus(core_raw) * step.core_sigma_scales
            tail_sigma = F.softplus(tail_raw) * step.tail_sigma_scales
            self.records.setdefault(index, []).append(
                (core_sigma.detach().cpu().numpy(), tail_sigma.detach().cpu().numpy())
            )

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for index, chunks in self.records.items():
            core = np.concatenate([chunk[0] for chunk in chunks], axis=0)
            tail = np.concatenate([chunk[1] for chunk in chunks], axis=0)
            out[f"step{index}"] = {
                "core_sigma_median_all_coords": float(np.median(core)),
                "core_sigma_mean_all_coords": float(np.mean(core)),
                "core_sigma_p99_all_coords": float(np.quantile(core, 0.99)),
                "tail_sigma_median_all_coords": float(np.median(tail)),
                "tail_sigma_mean_all_coords": float(np.mean(tail)),
                "tail_sigma_p99_all_coords": float(np.quantile(tail, 0.99)),
                # log-pT coordinates only (index 0 and 3 of the 6-D block)
                "core_sigma_logpt_median": float(np.median(core[:, [0, 3]])),
                "tail_sigma_logpt_median": float(np.median(tail[:, [0, 3]])),
            }
        return out


@torch.no_grad()
def decode_subset(model, z: np.ndarray, batch_size: int, device: torch.device, core: float, tail: float):
    model.decoder.set_noise_multipliers(core, tail)
    recorder = SigmaRecorder(model)
    outputs = []
    for start in range(0, len(z), batch_size):
        block = torch.as_tensor(z[start : start + batch_size], device=device, dtype=torch.float32)
        outputs.append(model.decode(block).cpu().numpy())
    recorder.close()
    return np.concatenate(outputs, axis=0), recorder.summary()


def load_z(prior_path: Path, max_events: int | None, seed: int):
    with h5py.File(prior_path, "r") as source:
        z = np.asarray(source["FDL/zData"][:], dtype=np.float32)
        component_id = np.asarray(source["FDL/component_id"][:])
    if max_events is not None and max_events < len(z):
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(z), size=int(max_events), replace=False)
        chosen.sort()
        z = z[chosen]
        component_id = component_id[chosen]
    return z, component_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=REPO_ROOT / "outputs" / "cms_Joint" / "Run_H")
    parser.add_argument(
        "--prior",
        type=Path,
        default=REPO_ROOT / "data" / "upsilon_prior_continuumReweighted.hdf5",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--max-events", type=int, default=300000)
    parser.add_argument("--full-events", action="store_true", help="decode all 1M prior events")
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--mass-min", type=float, default=8.5)
    parser.add_argument("--mass-max", type=float, default=11.5)
    parser.add_argument("--mass-bin-width", type=float, default=0.02)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_dir / "tail_audit"
    )
    if not (run_dir / "history.json").exists():
        raise FileNotFoundError(f"history.json not found under {run_dir}")
    if not args.prior.exists():
        raise FileNotFoundError(f"prior not found: {args.prior}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    max_events = None if args.full_events else args.max_events
    z, component_id = load_z(args.prior, max_events, args.seed)
    print(f"Decoding {len(z)} events on {device}")
    prior_z_moments = state_moments(
        z, component_id, args.mass_min, args.mass_max, args.mass_bin_width
    )
    print("prior z modes: 1S %.3f 2S %.3f 3S %.3f"
          % (prior_z_moments["upsilon1s"]["mode_bin_center_gev"],
             prior_z_moments["upsilon2s"]["mode_bin_center_gev"],
             prior_z_moments["upsilon3s"]["mode_bin_center_gev"]))

    # ---- 1. schedule verification from history --------------------------------
    history = json.loads((run_dir / "history.json").read_text())
    schedule: "OrderedDict[str, dict]" = OrderedDict()
    for row in history:
        entry = schedule.setdefault(
            row["stage"],
            {
                "epochs": 0,
                "core_values": set(),
                "tail_values": set(),
                "encoder_equals_decoder": True,
                "encoder_core_values": set(),
                "decoder_core_values": set(),
                "encoder_tail_values": set(),
                "decoder_tail_values": set(),
            },
        )
        entry["epochs"] += 1
        entry["core_values"].add(round(float(row["core_noise_multiplier"]), 6))
        entry["tail_values"].add(round(float(row["tail_noise_multiplier"]), 6))
        entry["encoder_core_values"].add(round(float(row["encoder_core_noise_multiplier"]), 6))
        entry["decoder_core_values"].add(round(float(row["decoder_core_noise_multiplier"]), 6))
        entry["encoder_tail_values"].add(round(float(row["encoder_tail_noise_multiplier"]), 6))
        entry["decoder_tail_values"].add(round(float(row["decoder_tail_noise_multiplier"]), 6))
        entry["encoder_equals_decoder"] &= (
            row["encoder_core_noise_multiplier"] == row["decoder_core_noise_multiplier"]
            and row["encoder_tail_noise_multiplier"] == row["decoder_tail_noise_multiplier"]
        )
    schedule_json = {
        name: {
            "epochs": entry["epochs"],
            "core": sorted(entry["core_values"]),
            "tail": sorted(entry["tail_values"]),
            "encoder_core": sorted(entry["encoder_core_values"]),
            "decoder_core": sorted(entry["decoder_core_values"]),
            "encoder_tail": sorted(entry["encoder_tail_values"]),
            "decoder_tail": sorted(entry["decoder_tail_values"]),
            "encoder_equals_decoder": bool(entry["encoder_equals_decoder"]),
        }
        for name, entry in schedule.items()
    }

    # ---- 2. checkpoint / inference-noise grid --------------------------------
    checkpoints = OrderedDict(
        [
            ("best_stage1", "best_RunH_stage1_deterministic_warmup.pt"),
            ("best_stage2", "best_RunH_stage2_stochastic_core.pt"),
            ("last_stage2", "last_RunH_stage2_stochastic_core.pt"),
            ("best_stage3", "best_RunH_stage3_stochastic_tail.pt"),
            ("last_stage3", "last_RunH_stage3_stochastic_tail.pt"),
        ]
    )
    grids = {
        "best_stage1": [(0.0, 0.0)],
        "best_stage2": [(0.0, 0.0), (1.0, 0.25), (1.0, 0.5)],
        "last_stage2": [(0.0, 0.0), (1.0, 0.0), (1.0, 0.25), (1.0, 0.5)],
        "best_stage3": [(0.0, 0.0), (1.0, 0.0), (1.0, 0.5)],
        "last_stage3": [(0.0, 0.0), (1.0, 0.0), (1.0, 0.25), (1.0, 0.5), (1.0, 1.0), (2.0, 0.5)],
    }

    results: dict[str, dict] = {}
    for label, filename in checkpoints.items():
        path = run_dir / filename
        if not path.exists():
            print(f"skip missing checkpoint {path}")
            continue
        model, checkpoint, _ = load_frozen_model(path, device)
        model.eval()
        native = checkpoint.get("noise_multipliers", {})
        print(f"=== {label} ({filename}, global_epoch {checkpoint.get('global_epoch')}) native={native}")
        per_checkpoint: dict[str, dict] = {
            "checkpoint": str(path),
            "global_epoch": checkpoint.get("global_epoch"),
            "stage": checkpoint.get("stage", {}).get("name") if isinstance(checkpoint.get("stage"), dict) else checkpoint.get("stage"),
            "native_noise_multipliers": native,
            "configs": {},
        }
        seen = set()
        for core, tail in grids.get(label, []):
            key = f"core{core:g}_tail{tail:g}"
            if key in seen:
                continue
            seen.add(key)
            torch.manual_seed(args.seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(args.seed)
            decoded, sigma_summary = decode_subset(model, z, args.batch_size, device, core, tail)
            moments = state_moments(
                decoded, component_id, args.mass_min, args.mass_max, args.mass_bin_width
            )
            per_checkpoint["configs"][key] = {
                "core": core,
                "tail": tail,
                "state_moments": moments,
                "learned_sigma": sigma_summary,
            }
            print(
                "   %-14s 1S mode %.3f (CMS %+.3f)  2S %.3f (%+.3f)  3S %.3f (%+.3f)"
                % (
                    key,
                    moments["upsilon1s"]["mode_bin_center_gev"],
                    moments["upsilon1s"]["mode_minus_cms_gev"],
                    moments["upsilon2s"]["mode_bin_center_gev"],
                    moments["upsilon2s"]["mode_minus_cms_gev"],
                    moments["upsilon3s"]["mode_bin_center_gev"],
                    moments["upsilon3s"]["mode_minus_cms_gev"],
                )
            )
        results[label] = per_checkpoint

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_dir": str(run_dir),
        "prior": str(args.prior),
        "events_decoded": int(len(z)),
        "seed": args.seed,
        "mass_range_gev": [args.mass_min, args.mass_max],
        "mass_bin_width_gev": args.mass_bin_width,
        "prior_z_moments": prior_z_moments,
        "schedule_from_history": schedule_json,
        "checkpoints": results,
    }
    (output_dir / "tail_audit.json").write_text(json.dumps(payload, indent=1, sort_keys=True))
    write_report(output_dir / "REPORT.md", payload)
    write_plot(output_dir / "peak_shift_vs_tail.png", payload)
    print(f"wrote {output_dir / 'tail_audit.json'}")
    return 0


def _caption(label: str, entry: dict) -> str:
    return f"{label} ({entry.get('stage')}, ep {entry.get('global_epoch')})"


def write_report(path: Path, payload: dict) -> None:
    lines = ["# Run H core/tail audit", ""]
    lines.append(f"- events decoded: {payload['events_decoded']}")
    lines.append(f"- prior: `{payload['prior']}`")
    lines.append(f"- mass range: {payload['mass_range_gev']} GeV, bin {payload['mass_bin_width_gev']} GeV")
    lines.append("")
    lines.append("## Verified stage schedule (history.json)")
    lines.append("")
    lines.append("| stage | epochs | core | tail | encoder=decoder |")
    lines.append("|---|---|---|---|---|")
    for name, entry in payload["schedule_from_history"].items():
        lines.append(
            f"| {name} | {entry['epochs']} | {entry['core']} | {entry['tail']} | {entry['encoder_equals_decoder']} |"
        )
    lines.append("")
    zp = payload["prior_z_moments"]
    lines.append(
        "Prior (truth-z) modes: 1S %.4f, 2S %.4f, 3S %.4f GeV; CMS fit: %.4f / %.4f / %.4f GeV."
        % (
            zp["upsilon1s"]["mode_bin_center_gev"],
            zp["upsilon2s"]["mode_bin_center_gev"],
            zp["upsilon3s"]["mode_bin_center_gev"],
            zp["upsilon1s"]["cms_fit_mass_gev"],
            zp["upsilon2s"]["cms_fit_mass_gev"],
            zp["upsilon3s"]["cms_fit_mass_gev"],
        )
    )
    lines.append("")
    lines.append("## Decoded Upsilon peak position vs inference noise")
    lines.append("")
    lines.append(
        "`mode - CMS` is the histogram-mode bin centre of the decoded per-state mass minus the "
        "CMS three-peak fit mass (GeV). Negative = peak moved down."
    )
    lines.append("")
    lines.append("| checkpoint | core | tail | 1S mode-CMS | 2S mode-CMS | 3S mode-CMS | 1S mean | 1S std |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for label, entry in payload["checkpoints"].items():
        for key, conf in entry["configs"].items():
            m = conf["state_moments"]
            lines.append(
                "| %s | %g | %g | %+.4f | %+.4f | %+.4f | %.4f | %.4f |"
                % (
                    _caption(label, entry),
                    conf["core"],
                    conf["tail"],
                    m["upsilon1s"]["mode_minus_cms_gev"],
                    m["upsilon2s"]["mode_minus_cms_gev"],
                    m["upsilon3s"]["mode_minus_cms_gev"],
                    m["upsilon1s"]["mean_gev"],
                    m["upsilon1s"]["std_gev"],
                )
            )
    lines.append("")
    lines.append("## Learned noise scales on the Upsilon prior (log-pT medians)")
    lines.append("")
    lines.append("| checkpoint | step | core_sigma (log pT) | tail_sigma (log pT) |")
    lines.append("|---|---|---|---|")
    for label, entry in payload["checkpoints"].items():
        first = next(iter(entry["configs"].values()))
        for step, values in first["learned_sigma"].items():
            lines.append(
                "| %s | %s | %.6f | %.6f |"
                % (
                    _caption(label, entry),
                    step,
                    values["core_sigma_logpt_median"],
                    values["tail_sigma_logpt_median"],
                )
            )
    path.write_text("\n".join(lines) + "\n")


def write_plot(path: Path, payload: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    states = STATE_COMPONENTS
    for ax, state in zip(axes, states):
        for label, entry in payload["checkpoints"].items():
            configs = sorted(entry["configs"].values(), key=lambda c: (c["core"], c["tail"]))
            xs = [c["tail"] for c in configs]
            ys = [c["state_moments"][state]["mode_minus_cms_gev"] * 1000.0 for c in configs]
            ax.plot(xs, ys, marker="o", label=_caption(label, entry))
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_title(state)
        ax.set_xlabel("tail noise multiplier")
    axes[0].set_ylabel("decoded mode - CMS fit [MeV]")
    axes[0].legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
