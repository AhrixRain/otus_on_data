#!/usr/bin/env python
"""S5 systematics study: does every headline claim survive the obvious variations?

Read-only. Three axes are quantified for the Upsilon and in-domain claims:

  prior       - the same frozen checkpoint evaluated on two Upsilon evaluation
                priors (raw 1M vs sideband-derived continuum-reweighted), plus the
                required-resolution shift between the legacy hand-smeared J/psi
                prior and the honest component-aware prior.
  checkpoint  - the spread across the retained checkpoints of one arm.
  seed        - the same checkpoint re-decoded at three subsample seeds.
  statistical - a bootstrap of the per-state median/width from the decoded sample.

Each headline claim is then reported as value plus the spread contributed by each
axis, with a robust/fragile verdict against the claim own tolerance. Nothing is
trained; only frozen checkpoints, existing decoded files and fixed-size re-decodes
are used.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
for _directory in (
    REPO_ROOT,
    REPO_ROOT / "scripts",
    REPO_ROOT / "scripts_sota",
    REPO_ROOT / "scripts_joint",
    REPO_ROOT / "scripts_joint" / "upsilon",
):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from decode_prior import load_frozen_model  # noqa: E402
from fixed_z_noise_budget import compute_data_reference, json_ready  # noqa: E402
from runH_tail_audit import CMS_FIT_MASS_GEV, decode_subset, load_z, mass8  # noqa: E402


RAW_UPSILON_PRIOR = (
    REPO_ROOT / "data" / "cms_upsilon_mumu_mg5py8_ckkwl_8tev_inclusive_0j1j_fiducial_8p5_11p5_1M.hdf5"
)
REWEIGHTED_UPSILON_PRIOR = REPO_ROOT / "data" / "upsilon_prior_continuumReweighted.hdf5"
CMS_CSV = REPO_ROOT / "experiments" / "cms_upsilon" / "data" / "Ymumu.csv"

STATES = ("upsilon1s", "upsilon2s", "upsilon3s")
COMPONENT = {"upsilon1s": 0, "upsilon2s": 1, "upsilon3s": 2, "continuum": 3}
MASS_WINDOW = (8.5, 11.5)
REGION = (8.5, 9.25)
RATIO_WINDOW = (8.5, 11.2)
BIN_WIDTH = 0.02

TOLERANCES = {
    "median_mev": 30.0,
    "drift_mev": 40.0,
    "width_mev": 30.0,
    "mass_w1_gev": 0.05,
    "r_ratio": 0.225,
    "slice_ratio": 3.0,
}


def spread(values: list) -> dict:
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None, "half_range": None}
    array = np.asarray(values, dtype=float)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
        "half_range": float(0.5 * (array.max() - array.min())),
    }


def combine_systematics(*spreads: dict) -> dict:
    half_ranges = [float(s["half_range"]) for s in spreads if s and s.get("half_range") is not None]
    stds = [float(s["std"]) for s in spreads if s and s.get("std") is not None]
    return {
        "half_range_max": max(half_ranges) if half_ranges else None,
        "quadrature_std": math.sqrt(sum(value * value for value in stds)) if stds else None,
        "axes": [{"n": s.get("n"), "std": s.get("std"), "half_range": s.get("half_range")} for s in spreads],
    }


def claim_verdict(value, central: float, tolerance: float, total: dict) -> dict:
    if value is None or total.get("half_range_max") is None:
        return {"status": "not_measured", "margin": None, "tolerance": tolerance}
    margin = abs(float(value) - float(central))
    budget = margin + float(total["half_range_max"])
    return {
        "status": "robust" if budget <= tolerance else "fragile",
        "margin": margin,
        "budget": budget,
        "tolerance": tolerance,
        "headroom": tolerance - budget,
    }


def bootstrap_median(values: np.ndarray, *, draws: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"median": None, "std": None, "q16": None, "q84": None}
    picks = rng.integers(0, values.size, size=(int(draws), values.size))
    medians = np.median(values[picks], axis=1)
    return {
        "median": float(np.median(values)),
        "std": float(medians.std(ddof=1)),
        "q16": float(np.quantile(medians, 0.16)),
        "q84": float(np.quantile(medians, 0.84)),
    }


def decoded_mass(path: Path):
    import h5py

    with h5py.File(path, "r") as handle:
        x = np.asarray(handle["FDL/xData"][:], dtype=np.float64)
        component = np.asarray(handle["FDL/component_id"][:])
        z = np.asarray(handle["FDL/zData"][:], dtype=np.float64) if "FDL/zData" in handle else None
    return mass8(x), component, (None if z is None else mass8(z))


def cms_mass() -> np.ndarray:
    table = np.genfromtxt(CMS_CSV, delimiter=",", names=True, encoding="utf-8")
    energy = table["E1"] + table["E2"]
    px = table["px1"] + table["px2"]
    py = table["py1"] + table["py2"]
    pz = table["pz1"] + table["pz2"]
    return np.sqrt(np.maximum(energy**2 - px**2 - py**2 - pz**2, 0.0))


def state_stats(masses: np.ndarray, component: np.ndarray, *, bootstrap_draws: int, seed: int) -> dict:
    out = {}
    for state in STATES:
        sample = masses[component == COMPONENT[state]]
        if sample.size == 0:
            out[state] = {"events": 0}
            continue
        boot = bootstrap_median(sample, draws=bootstrap_draws, seed=seed)
        out[state] = {
            "events": int(sample.size),
            "median_gev": float(np.median(sample)),
            "median_minus_cms_mev": float((np.median(sample) - CMS_FIT_MASS_GEV[state]) * 1000.0),
            "median_bootstrap_std_mev": None if boot["std"] is None else float(boot["std"] * 1000.0),
            "std_gev": float(np.std(sample)),
            "robust_gev": float(0.5 * (np.quantile(sample, 0.8413) - np.quantile(sample, 0.1587))),
            "window_efficiency": float(np.mean((sample >= MASS_WINDOW[0]) & (sample <= MASS_WINDOW[1]))),
        }
    return out


def region_ratio(masses: np.ndarray, cms: np.ndarray) -> float:
    def fraction(values: np.ndarray) -> float:
        in_window = (values >= RATIO_WINDOW[0]) & (values < RATIO_WINDOW[1])
        in_region = (values >= REGION[0]) & (values < REGION[1])
        return float(in_region.sum() / max(int(in_window.sum()), 1))

    return fraction(masses) / fraction(cms)


def prior_axis(entries, *, bootstrap_draws: int, seed: int) -> dict:
    cms = cms_mass()
    rows = []
    for label, decoded_path, prior_path in entries:
        if not decoded_path.exists():
            rows.append({"label": label, "decoded": str(decoded_path), "status": "missing"})
            continue
        masses, component, z_mass = decoded_mass(decoded_path)
        row = {
            "label": label,
            "decoded": str(decoded_path),
            "prior": None if prior_path is None else str(prior_path),
            "status": "measured",
            "states": state_stats(masses, component, bootstrap_draws=bootstrap_draws, seed=seed),
            "ratio_8p50_9p25_over_cms": region_ratio(masses, cms),
            "inclusive_mean_gev": float(np.mean(masses)),
            "inclusive_std_gev": float(np.std(masses)),
        }
        if z_mass is not None:
            row["truth_inclusive_std_gev"] = float(np.std(z_mass))
        rows.append(row)
    return {"rows": rows}


def reference_shift(legacy_checkpoint: Path, honest_checkpoint: Path) -> dict:
    out = {}
    for label, path in (("legacy_prior", legacy_checkpoint), ("honest_prior", honest_checkpoint)):
        if not path.exists():
            out[label] = {"status": "missing", "checkpoint": str(path)}
            continue
        reference = compute_data_reference(path, log=lambda message: None)
        region = (reference.get("regions") or {}).get("jpsi") or {}
        out[label] = {
            "status": reference.get("status"),
            "checkpoint": str(path),
            "jpsi_required_robust_gev": region.get("required_robust_gev"),
            "jpsi_required_std_gev": region.get("required_std_gev"),
            "jpsi_prior_robust_gev": region.get("prior_z_robust_gev"),
            "jpsi_cms_robust_gev": region.get("cms_x_robust_gev"),
        }
    legacy = out.get("legacy_prior", {}).get("jpsi_required_robust_gev")
    honest = out.get("honest_prior", {}).get("jpsi_required_robust_gev")
    if legacy and honest:
        out["ratio_honest_over_legacy"] = float(honest / legacy)
    return out


def seed_axis(
    checkpoint: Path,
    prior: Path,
    *,
    events: int,
    seeds,
    device: torch.device,
    bootstrap_draws: int,
    log=print,
) -> dict:
    rows = []
    for seed in seeds:
        model, checkpoint_payload, _config = load_frozen_model(checkpoint, device)
        model.eval()
        recorded = checkpoint_payload.get("noise_multipliers") or {}
        core = float(recorded.get("decoder_core", recorded.get("core", 1.0)))
        tail = float(recorded.get("decoder_tail", recorded.get("tail", 1.0)))
        z, component = load_z(prior, events, seed)
        decoded, _sigma = decode_subset(model, z, 16384, device, core, tail)
        masses = mass8(decoded)
        rows.append(
            {
                "seed": int(seed),
                "events": int(len(z)),
                "native_multipliers": {"core": core, "tail": tail},
                "states": state_stats(masses, component, bootstrap_draws=bootstrap_draws, seed=seed),
                "inclusive_std_gev": float(np.std(masses)),
            }
        )
        log("[seed] %s: 1S median-CMS %+.1f MeV" % (seed, rows[-1]["states"]["upsilon1s"]["median_minus_cms_mev"]))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {"rows": rows}


def scorecard_axis(paths) -> dict:
    rows = []
    for path in paths:
        if not path.exists():
            rows.append({"path": str(path), "status": "missing"})
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        entry = payload["checkpoints"][payload["selection"]["selected"]]
        gates = entry["gates"]
        rows.append(
            {
                "path": str(path),
                "status": "measured",
                "seed": payload.get("seed"),
                "selected": payload["selection"]["selected"],
                "jpsi_ratio": (gates["R"].get("jpsi") or {}).get("within_std_over_required"),
                "z_ratio": (gates["R"].get("z") or {}).get("within_std_over_required"),
                "jpsi_slice_median": (gates["C"].get("jpsi") or {}).get("decode_ratio_median_native"),
                "z_slice_median": (gates["C"].get("z") or {}).get("decode_ratio_median_native"),
            }
        )
    return {"rows": rows}


def build_claims(payload: dict) -> list:
    claims = []
    prior_rows = [row for row in payload["axes"]["prior"]["rows"] if row.get("status") == "measured"]
    checkpoint_rows = [row for row in payload["axes"]["checkpoint"]["rows"] if row.get("status") == "measured"]
    seed_rows = [row for row in payload["axes"]["seed"]["rows"] if row.get("status", "measured") == "measured"]
    scorecard_rows = [row for row in payload["axes"]["scorecard"]["rows"] if row.get("status") == "measured"]

    def medians(rows, state):
        return [row["states"][state]["median_minus_cms_mev"] for row in rows if row.get("states", {}).get(state)]

    def widths(rows, state="upsilon1s"):
        return [row["states"][state]["std_gev"] * 1000.0 for row in rows if row.get("states", {}).get(state)]

    for state in STATES:
        axis_prior = spread(medians(prior_rows, state))
        axis_checkpoint = spread(medians(checkpoint_rows, state))
        axis_seed = spread(medians(seed_rows, state))
        total = combine_systematics(axis_prior, axis_checkpoint, axis_seed)
        claims.append(
            {
                "claim": state + " median - CMS [MeV]",
                "value": axis_prior["mean"],
                "central": 0.0,
                "tolerance": TOLERANCES["median_mev"],
                "axes": {"prior": axis_prior, "checkpoint": axis_checkpoint, "seed": axis_seed},
                "total": total,
                "verdict": claim_verdict(axis_prior["mean"], 0.0, TOLERANCES["median_mev"], total),
            }
        )

    axis_prior_width = spread(widths(prior_rows))
    axis_checkpoint_width = spread(widths(checkpoint_rows))
    axis_seed_width = spread(widths(seed_rows))
    total_width = combine_systematics(axis_prior_width, axis_checkpoint_width, axis_seed_width)
    claims.append(
        {
            "claim": "upsilon1s decoded width vs the 84 MeV resolution [MeV]",
            "value": axis_prior_width["mean"],
            "central": 84.4,
            "tolerance": TOLERANCES["width_mev"],
            "axes": {"prior": axis_prior_width, "checkpoint": axis_checkpoint_width, "seed": axis_seed_width},
            "total": total_width,
            "verdict": claim_verdict(axis_prior_width["mean"], 84.4, TOLERANCES["width_mev"], total_width),
        }
    )

    if scorecard_rows:
        for key, label, tolerance in (
            ("jpsi_ratio", "in-domain jpsi within/required", TOLERANCES["r_ratio"]),
            ("z_ratio", "in-domain z within/required", TOLERANCES["r_ratio"]),
            ("jpsi_slice_median", "jpsi conditional slice ratio (native)", TOLERANCES["slice_ratio"]),
        ):
            axis = spread([row.get(key) for row in scorecard_rows])
            total = combine_systematics(axis)
            claims.append(
                {
                    "claim": label + " [seed axis]",
                    "value": axis["mean"],
                    "central": 1.0,
                    "tolerance": tolerance,
                    "axes": {"seed": axis},
                    "total": total,
                    "verdict": claim_verdict(axis["mean"], 1.0, tolerance, total),
                }
            )
    return claims


def write_report(path: Path, payload: dict) -> None:
    claims = payload["claims"]
    lines = [
        "# S5 systematics study",
        "",
        "*%s - read-only. Checkpoint %s, %s Upsilon events per seed, seeds %s.*"
        % (payload["created_utc"], payload["checkpoint"], payload["events"], payload["seeds"]),
        "",
        "## Claim summary",
        "",
        "| claim | value | prior spread | checkpoint spread | seed spread | total half-range | tolerance | verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]

    def fmt(value, digits=3):
        if value is None:
            return "n/a"
        return format(float(value), "." + str(digits) + "g")

    for claim in claims:
        axes = claim["axes"]
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                claim["claim"],
                fmt(claim["value"]),
                fmt((axes.get("prior") or {}).get("half_range")),
                fmt((axes.get("checkpoint") or {}).get("half_range")),
                fmt((axes.get("seed") or {}).get("half_range")),
                fmt(claim["total"].get("half_range_max")),
                fmt(claim["tolerance"]),
                claim["verdict"]["status"],
            )
        )
    reference = payload.get("reference_shift") or {}
    lines += [
        "",
        "## Required-resolution shift (the denominator of the in-domain claim)",
        "",
        "| prior | jpsi required robust [MeV] | prior robust [MeV] | CMS robust [MeV] |",
        "|---|---|---|---|",
    ]
    for key in ("legacy_prior", "honest_prior"):
        block = reference.get(key) or {}
        lines.append(
            "| %s | %s | %s | %s |"
            % (
                key,
                fmt(None if block.get("jpsi_required_robust_gev") is None else block["jpsi_required_robust_gev"] * 1000),
                fmt(None if block.get("jpsi_prior_robust_gev") is None else block["jpsi_prior_robust_gev"] * 1000),
                fmt(None if block.get("jpsi_cms_robust_gev") is None else block["jpsi_cms_robust_gev"] * 1000),
            )
        )
    if reference.get("ratio_honest_over_legacy"):
        lines += [
            "",
            "The honest prior demands **%.2fx** the additional J/psi smearing the legacy hand-smeared prior"
            % reference["ratio_honest_over_legacy"],
            "demands: any in-domain ratio quoted without its prior is undefined.",
        ]
    lines += [
        "",
        "## Prior axis detail",
        "",
        "| label | prior | 1S median-CMS [MeV] | 2S | 3S | 1S std [MeV] | 8.5-9.25 / CMS |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in payload["axes"]["prior"]["rows"]:
        if row.get("status") != "measured":
            lines.append("| %s | - | missing | | | | |" % row["label"])
            continue
        states = row["states"]
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s |"
            % (
                row["label"],
                Path(row["prior"]).name if row.get("prior") else "n/a",
                fmt(states["upsilon1s"]["median_minus_cms_mev"], 4),
                fmt(states["upsilon2s"]["median_minus_cms_mev"], 4),
                fmt(states["upsilon3s"]["median_minus_cms_mev"], 4),
                fmt(states["upsilon1s"]["std_gev"] * 1000, 4),
                fmt(row.get("ratio_8p50_9p25_over_cms"), 3),
            )
        )
    lines += [
        "",
        "## Statistical (bootstrap) detail",
        "",
        "| label | 1S median bootstrap std [MeV] | events |",
        "|---|---|---|",
    ]
    for row in payload["axes"]["prior"]["rows"]:
        if row.get("status") != "measured":
            continue
        state = row["states"]["upsilon1s"]
        lines.append("| %s | %s | %s |" % (row["label"], fmt(state.get("median_bootstrap_std_mev")), state.get("events")))
    lines += ["", "## Verdicts", ""]
    for claim in claims:
        verdict = claim["verdict"]
        lines.append(
            "- **%s**: %s (budget %s vs tolerance %s)"
            % (claim["claim"], verdict["status"], fmt(verdict.get("budget")), fmt(claim["tolerance"]))
        )
    lines += [
        "",
        "## Method notes",
        "",
        "- The prior axis uses existing decoded files, so the two priors differ only in the",
        "  truth reference and the sample; the checkpoint and noise multipliers are identical.",
        "- The seed axis re-decodes the same checkpoint with a different subsample seed, so it",
        "  measures evaluation noise, not training noise. A training-seed study needs new runs.",
        "- The bootstrap resamples the decoded events; it is the statistical floor of the",
        "  measurement, not a systematic.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plot(path: Path, payload: dict) -> None:
    prior_rows = [row for row in payload["axes"]["prior"]["rows"] if row.get("status") == "measured"]
    seed_rows = [row for row in payload["axes"]["seed"]["rows"] if row.get("status", "measured") == "measured"]
    if not prior_rows and not seed_rows:
        return
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    if prior_rows:
        labels = [row["label"] for row in prior_rows]
        x = np.arange(len(labels))
        width = 0.8 / len(STATES)
        for index, state in enumerate(STATES):
            values = [row["states"][state]["median_minus_cms_mev"] for row in prior_rows]
            axes[0].bar(x + index * width, values, width, label=state)
        axes[0].axhspan(-TOLERANCES["median_mev"], TOLERANCES["median_mev"], color="tab:green", alpha=0.15)
        axes[0].set_xticks(x + 0.4 - width / 2)
        axes[0].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        axes[0].set_ylabel("median - CMS [MeV]")
        axes[0].set_title("Prior axis")
        axes[0].legend(frameon=False, fontsize=8)
    if seed_rows:
        seeds = [row["seed"] for row in seed_rows]
        x = np.arange(len(seeds))
        width = 0.8 / len(STATES)
        for index, state in enumerate(STATES):
            values = [row["states"][state]["median_minus_cms_mev"] for row in seed_rows]
            axes[1].bar(x + index * width, values, width, label=state)
        axes[1].axhspan(-TOLERANCES["median_mev"], TOLERANCES["median_mev"], color="tab:green", alpha=0.15)
        axes[1].set_xticks(x + 0.4 - width / 2)
        axes[1].set_xticklabels([str(seed) for seed in seeds])
        axes[1].set_xlabel("subsample seed")
        axes[1].set_title("Seed axis")
    figure.tight_layout()
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prior-entry", action="append", default=None, metavar="LABEL=DECODED_PATH")
    parser.add_argument("--checkpoint-entry", action="append", default=None, metavar="LABEL=DECODED_PATH")
    parser.add_argument("--scorecard", action="append", type=Path, default=None)
    parser.add_argument("--legacy-checkpoint", type=Path, default=None)
    parser.add_argument("--honest-checkpoint", type=Path, default=None)
    parser.add_argument("--seed-axis-prior", type=Path, default=REWEIGHTED_UPSILON_PRIOR)
    parser.add_argument("--seeds", default="20260822,20260823,20260824")
    parser.add_argument("--events", type=int, default=400000)
    parser.add_argument("--bootstrap-draws", type=int, default=200)
    parser.add_argument("--skip-seed-axis", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = (args.output_dir or (REPO_ROOT / "outputs" / "cms_Joint" / "systematics")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device)
    )
    seeds = [int(value) for value in str(args.seeds).split(",") if value.strip()]

    def parse_entries(specs, default_prior=None):
        entries = []
        for spec in specs or []:
            label, _, raw = spec.partition("=")
            path = Path(raw)
            if not path.is_absolute():
                path = (REPO_ROOT / path).resolve()
            entries.append((label, path, default_prior))
        return entries

    prior_entries = parse_entries(args.prior_entry)
    checkpoint_entries = parse_entries(args.checkpoint_entry) or prior_entries
    legacy = args.legacy_checkpoint or (run_dir / "best_model.pt")
    honest = args.honest_checkpoint or (run_dir / "best_model.pt")

    payload = {
        "schema_version": 1,
        "study": "S5 systematics",
        "status": "completed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "checkpoint": str(args.checkpoint),
        "events": int(args.events),
        "seeds": seeds,
        "tolerances": TOLERANCES,
        "axes": {
            "prior": prior_axis(prior_entries, bootstrap_draws=args.bootstrap_draws, seed=seeds[0] if seeds else 0),
            "checkpoint": prior_axis(checkpoint_entries, bootstrap_draws=args.bootstrap_draws, seed=seeds[0] if seeds else 0),
            "seed": {"rows": []},
            "scorecard": scorecard_axis([path for path in (args.scorecard or [])]),
        },
        "reference_shift": reference_shift(legacy, honest),
    }
    if not args.skip_seed_axis:
        payload["axes"]["seed"] = seed_axis(
            args.checkpoint.expanduser().resolve(),
            args.seed_axis_prior,
            events=args.events,
            seeds=seeds,
            device=device,
            bootstrap_draws=args.bootstrap_draws,
        )
    payload["claims"] = build_claims(payload)
    payload["runtime_seconds"] = time.time() - started

    (output_dir / "systematics.json").write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(output_dir / "REPORT.md", payload)
    make_plot(output_dir / "systematics.png", payload)
    rows = ["claim,value,tolerance,prior_half_range,checkpoint_half_range,seed_half_range,verdict"]
    for claim in payload["claims"]:
        axes = claim["axes"]
        rows.append(
            ",".join(
                str(value)
                for value in (
                    claim["claim"],
                    claim["value"],
                    claim["tolerance"],
                    (axes.get("prior") or {}).get("half_range"),
                    (axes.get("checkpoint") or {}).get("half_range"),
                    (axes.get("seed") or {}).get("half_range"),
                    claim["verdict"]["status"],
                )
            )
        )
    (output_dir / "systematics.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))
    print("wrote %s" % (output_dir / "systematics.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
