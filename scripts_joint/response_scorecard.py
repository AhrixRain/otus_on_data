#!/usr/bin/env python
"""Response-identification scorecard for joint runs (plan-tree leaf A0.4).

The trainer selection score is a *marginal* gauge evaluated at zero noise; it
cannot see whether the detector resolution is carried by the learned noise or
faked by the deterministic mean map, and A0.3 showed that scoring at native
noise instead selects the deterministic warmup in 8 of 8 arms. This module
measures the split directly and turns it into a collapse-proof selection key.

Axes
----
R  response identification (fixed-z noise budget): within-z width vs the required
   resolution, sigma_noise_only, and the exact variance split
   Var(x) = E_z[Var(x|z)] + Var_z[E(x|z)] (noise share vs mean-map share).
C  conditional closure: per pair-pT slice mass W1 / finite-sample floor, at zero
   and at the checkpoint native noise.
T  transfer / OOD (opt-in, post-unblinding): Upsilon per-state medians vs the CMS
   fit. Axis T is report-only and never allowed to influence selection; that is
   enforced by selection_is_transfer_blind.

Every number carries its reference where one exists (identity, floor) and the
checkpoint provenance (sha256, epoch, native multipliers, config).

Read-only: opens checkpoints and priors, writes only into --output-dir.
"""

from __future__ import annotations

import argparse
import copy
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
from fixed_z_noise_budget import (  # noqa: E402
    COMPONENT_MAP,
    DOCUMENTED_REFERENCES,
    FALLBACK_REFERENCES,
    UPSILON_STATES,
    compute_data_reference,
    decode_mass_matrix,
    deterministic_statistics,
    draw_statistics,
    json_ready,
    load_fixed_z,
    resolve_prior_path,
    sha256_file,
)
from joint_data import load_joint_regions  # noqa: E402
from runH_tail_audit import CMS_FIT_MASS_GEV, decode_subset, load_z, state_moments  # noqa: E402
from run_joint import build_loss_factories  # noqa: E402
from slice_resolved_diagnostic import pair_pt, w1_standardized  # noqa: E402


SCHEMA_VERSION = 1
UPSILON_PRIOR = REPO_ROOT / "data" / "upsilon_prior_continuumReweighted.hdf5"

# --- gate bands (documented, single place to change) -----------------------
REQUIRED_RATIO_BAND = (0.8, 1.25)
NOISE_ONLY_MIN_GEV = {
    "jpsi": 0.020,
    "z": 0.50,
    "upsilon1s": 0.050,
    "upsilon2s": 0.050,
    "upsilon3s": 0.050,
}
# Degeneracy is a *width* criterion, not a share criterion: the noise variance
# share depends on how wide the region prior is relative to the required
# resolution (the honest narrow J/psi prior makes the share small even when the
# channel is live), while "the within-z width is far below what the detector
# needs" is exactly the F1 mean-map signature and is portable across priors.
DEGENERATE_WIDTH_RATIO = 0.5
MIN_NOISE_VARIANCE_FRACTION = 0.10
SLICE_MEDIAN_RATIO_MAX = 3.0
SLICE_MAX_RATIO_MAX = 10.0
DEFAULT_CHECKPOINTS = ("best_model.pt", "last_model.pt")


# ---------------------------------------------------------------------------
# provenance helpers
# ---------------------------------------------------------------------------

def native_decoder_multipliers(checkpoint: dict) -> tuple[float, float]:
    recorded = checkpoint.get("noise_multipliers") or {}
    shared_core = float(recorded.get("core", 1.0))
    shared_tail = float(recorded.get("tail", 1.0))
    return (
        float(recorded.get("decoder_core", shared_core)),
        float(recorded.get("decoder_tail", shared_tail)),
    )


def in_domain_score(checkpoint: dict) -> float | None:
    selection = checkpoint.get("joint_selection") or {}
    value = selection.get("selection_score")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def resolve_checkpoints(run_dir: Path, specs: list[str] | None, all_checkpoints: bool) -> list[tuple[str, Path]]:
    if specs:
        out = []
        for spec in specs:
            label, _, raw = spec.partition("=")
            path = Path(raw) if raw else Path(label)
            if not path.is_absolute():
                candidate = run_dir / path
                path = candidate if candidate.exists() else path
            out.append((label if raw else path.stem, path))
        return out
    if all_checkpoints:
        files = sorted(run_dir.glob("*.pt"))
        if not files:
            raise FileNotFoundError(f"no checkpoints in {run_dir}")
        return [(path.stem, path) for path in files]
    out = []
    for name in DEFAULT_CHECKPOINTS:
        path = run_dir / name
        if path.exists():
            out.append((path.stem, path))
    if not out:
        raise FileNotFoundError(
            f"no default checkpoints in {run_dir}; pass --checkpoint LABEL=PATH"
        )
    return out


# ---------------------------------------------------------------------------
# Axis R - response identification
# ---------------------------------------------------------------------------

def _mass8(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    energy = values[:, 3] + values[:, 7]
    momentum = values[:, :3] + values[:, 4:7]
    return np.sqrt(np.maximum(energy * energy - np.sum(momentum * momentum, axis=1), 0.0))


def _robust_half_width(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    return float(0.5 * (np.quantile(values, 0.8413) - np.quantile(values, 0.1587)))


def fixed_z_samples(
    config: dict,
    *,
    regions: tuple[str, ...],
    events: int,
    seed: int,
    include_upsilon: bool,
    log=print,
) -> dict[str, dict]:
    samples: dict[str, dict] = {}
    for offset, region in enumerate(("jpsi", "z")):
        if region not in regions:
            continue
        path, used, fallback = resolve_prior_path(config, region)
        z, info = load_fixed_z(path, events=events, seed=seed + offset)
        samples[region] = {
            "prior_path_configured": config["regions"][region]["paths"]["theory_prior_file"],
            "prior_path_used": used,
            "prior_fallback_used": bool(fallback),
            "component_id": info.get("component_id"),
            "available": info.get("available"),
            "z": z,
            "prior_mass_std_gev": float(np.std(_mass8(z))),
            "prior_mass_robust_gev": float(_robust_half_width(_mass8(z))),
        }
        log(f"[data] {region}: {len(z)} fixed z from {used} (fallback={fallback})")
    if include_upsilon:
        if not UPSILON_PRIOR.exists():
            raise FileNotFoundError(f"Upsilon prior not found: {UPSILON_PRIOR}")
        for state in UPSILON_STATES:
            z, info = load_fixed_z(
                UPSILON_PRIOR,
                events=events,
                seed=seed + 10 + COMPONENT_MAP[state],
                component_id=COMPONENT_MAP[state],
            )
            samples[state] = {
                "prior_path_configured": str(UPSILON_PRIOR),
                "prior_path_used": str(UPSILON_PRIOR),
                "prior_fallback_used": False,
                "component_id": info.get("component_id"),
                "available": info.get("available"),
                "z": z,
                "prior_mass_std_gev": float(np.std(_mass8(z))),
                "prior_mass_robust_gev": float(_robust_half_width(_mass8(z))),
            }
            log(f"[data] {state}: {len(z)} fixed z from {UPSILON_PRIOR}")
    return samples


def region_reference(reference: dict, region: str) -> dict:
    """Required resolution for a region, from the data-driven estimate or docs."""
    entry = (reference.get("regions") or {}).get(region) or {}
    documented = DOCUMENTED_REFERENCES.get(region, {})
    if entry.get("required_robust_gev"):
        return {
            "required_robust_gev": float(entry["required_robust_gev"]),
            "required_std_gev": entry.get("required_std_gev"),
            "source": f"data-driven ({reference.get('status', 'unknown')})",
        }
    fallback = FALLBACK_REFERENCES.get(region, {})
    if fallback.get("required_robust_gev") and not entry:
        return {
            "required_robust_gev": float(fallback["required_robust_gev"]),
            "required_std_gev": fallback.get("required_std_gev"),
            "source": "fallback table",
        }
    value = documented.get("value_gev")
    return {
        "required_robust_gev": None if value is None else float(value),
        "required_std_gev": None if value is None else float(value),
        "source": documented.get("source", "documented"),
    }


def axis_response(
    model,
    checkpoint: dict,
    samples: dict[str, dict],
    reference: dict,
    *,
    device: torch.device,
    draws: int,
    batch_size: int,
    seed: int,
    index: int,
    log=print,
) -> dict[str, dict]:
    native_core, native_tail = native_decoder_multipliers(checkpoint)
    out: dict[str, dict] = {}
    for region, payload in samples.items():
        z = payload["z"]
        zero_matrix, _ = decode_mass_matrix(
            model, z, batch_size=batch_size, device=device,
            core=0.0, tail=0.0, draws=draws, seed=seed + 700 + index,
        )
        zero_identical = bool(
            np.array_equal(zero_matrix, np.repeat(zero_matrix[:, :1], zero_matrix.shape[1], axis=1))
        )
        native_matrix, sigma_summary = decode_mass_matrix(
            model, z, batch_size=batch_size, device=device,
            core=native_core, tail=native_tail, draws=draws, seed=seed + 100 + index,
            record_sigma=True,
        )
        zero_stats = deterministic_statistics(zero_matrix[:, 0])
        native_stats = draw_statistics(native_matrix)
        ref = region_reference(reference, region)
        within_robust = float(native_stats["within_robust_median_gev"])
        required = ref.get("required_robust_gev")
        ratio = within_robust / required if required else float("nan")
        sigma_only = math.sqrt(max(float(native_stats["within_variance_mean_gev2"]), 0.0))
        noise_fraction = native_stats.get("noise_variance_fraction")
        out[region] = {
            "status": "artifact-measured",
            "prior_path_used": payload["prior_path_used"],
            "prior_fallback_used": payload["prior_fallback_used"],
            "prior_mass_std_gev": payload["prior_mass_std_gev"],
            "prior_mass_robust_gev": payload["prior_mass_robust_gev"],
            "required_robust_gev": required,
            "required_std_gev": ref.get("required_std_gev"),
            "reference_source": ref.get("source"),
            "zero_noise_draws_identical": zero_identical,
            "zero_noise_std_gev": float(zero_stats["ensemble_std_gev"]),
            "zero_noise_robust_gev": float(zero_stats["ensemble_robust_half_width_gev"]),
            "native_ensemble_mean_gev": float(native_stats["ensemble_mean_gev"]),
            "native_ensemble_std_gev": float(native_stats["ensemble_std_gev"]),
            "within_std_median_gev": float(native_stats["within_std_median_gev"]),
            "within_robust_median_gev": within_robust,
            "within_std_over_required": ratio,
            "sigma_noise_only_gev": sigma_only,
            "noise_variance_fraction": noise_fraction,
            "mean_map_variance_fraction": native_stats.get("mean_map_variance_fraction"),
            "conditional_mean_std_gev": float(native_stats["conditional_mean_std_gev"]),
            "learned_sigma_native": sigma_summary,
            "full_stats": native_stats,
        }
        log(
            f"    R {region}: within-z robust {within_robust * 1000:.1f} MeV "
            f"vs required {'n/a' if not required else format(required * 1000, '.1f')} MeV "
            f"(ratio {ratio:.3f}), sigma_noise_only {sigma_only * 1000:.1f} MeV, "
            f"noise share {noise_fraction if noise_fraction is None else round(noise_fraction, 4)}"
        )
    return out


# ---------------------------------------------------------------------------
# Axis C - conditional closure
# ---------------------------------------------------------------------------

def _conditional_slices(
    space_x,
    space_z,
    x: torch.Tensor,
    z: torch.Tensor,
    decoded: torch.Tensor,
    encoded: torch.Tensor,
    *,
    slices: int = 4,
    min_events: int = 50,
) -> dict:
    x_pt, z_pt, dec_pt, enc_pt = pair_pt(x), pair_pt(z), pair_pt(decoded), pair_pt(encoded)
    edges_x = np.percentile(x_pt.cpu().numpy(), np.linspace(0, 100, slices + 1))
    edges_z = np.percentile(z_pt.cpu().numpy(), np.linspace(0, 100, slices + 1))
    edges_x[-1] += 1e-6
    edges_z[-1] += 1e-6
    rows = []
    for i in range(slices):
        lo_x, hi_x = float(edges_x[i]), float(edges_x[i + 1])
        lo_z, hi_z = float(edges_z[i]), float(edges_z[i + 1])
        sx = (x_pt >= lo_x) & (x_pt < hi_x)
        sd = (dec_pt >= lo_x) & (dec_pt < hi_x)
        sz = (z_pt >= lo_z) & (z_pt < hi_z)
        se = (enc_pt >= lo_z) & (enc_pt < hi_z)
        floor_dec = floor_enc = None
        if int(sx.sum()) > min_events:
            half = int(sx.sum()) // 2
            xs = x[sx]
            floor_dec = w1_standardized(space_x, xs[:half], xs[half : 2 * half])
        if int(sz.sum()) > min_events:
            half = int(sz.sum()) // 2
            zs = z[sz]
            floor_enc = w1_standardized(space_z, zs[:half], zs[half : 2 * half])
        rows.append(
            {
                "slice": i,
                "x_pair_pt_lo": lo_x,
                "x_pair_pt_hi": hi_x,
                "n_decode_true": int(sx.sum()),
                "n_decode_pred": int(sd.sum()),
                "decode_mass_w1": float(w1_standardized(space_x, x[sx], decoded[sd]))
                if int(sx.sum()) > min_events and int(sd.sum()) > min_events
                else None,
                "decode_mass_w1_floor": floor_dec,
                "encode_mass_w1": float(w1_standardized(space_z, z[sz], encoded[se]))
                if int(sz.sum()) > min_events and int(se.sum()) > min_events
                else None,
                "encode_mass_w1_floor": floor_enc,
            }
        )
    ratios = [
        row["decode_mass_w1"] / row["decode_mass_w1_floor"]
        for row in rows
        if row["decode_mass_w1"] and row["decode_mass_w1_floor"]
    ]
    return {
        "slices": rows,
        "decode_ratio_median": float(np.median(ratios)) if ratios else None,
        "decode_ratio_max": float(np.max(ratios)) if ratios else None,
        "n_scored_slices": len(ratios),
    }


def axis_conditional(
    model,
    region_arrays: dict,
    factories: dict,
    *,
    device: torch.device,
    samples: int,
    seed: int,
    native: tuple[float, float],
    log=print,
) -> dict[str, dict]:
    rng = np.random.default_rng(seed)
    out: dict[str, dict] = {}
    for region in ("jpsi", "z"):
        if region not in region_arrays:
            continue
        factory = factories[region]
        arrays = region_arrays[region]

        def take(values: np.ndarray) -> torch.Tensor:
            pool = values
            if len(pool) > samples:
                pool = pool[rng.choice(len(pool), size=samples, replace=False)]
            return torch.as_tensor(np.ascontiguousarray(pool), dtype=torch.float32, device=device)

        x = torch.cat([take(arrays["x_train"]), take(arrays["x_val"])])
        z = torch.cat([take(arrays["z_train"]), take(arrays["z_val"])])
        block: dict[str, dict] = {}
        for label, (core, tail) in (("zero", (0.0, 0.0)), ("native", native)):
            model.set_noise_multipliers(core, tail)
            with torch.no_grad():
                decoded = model.decode(z)
                encoded = model.encode(x)
                if isinstance(encoded, (tuple, list)):
                    encoded = encoded[0]
            block[label] = _conditional_slices(
                factory.x_space, factory.z_space, x, z, decoded, encoded
            )
            block[label]["multipliers"] = {"core": core, "tail": tail}
            log(
                f"    C {region} ({label}): decode slice ratio median "
                f"{block[label]['decode_ratio_median']} max {block[label]['decode_ratio_max']}"
            )
        out[region] = block
    return out


# ---------------------------------------------------------------------------
# Axis T - transfer / OOD (report-only)
# ---------------------------------------------------------------------------

def axis_transfer(
    model,
    checkpoint: dict,
    *,
    device: torch.device,
    prior: Path,
    max_events: int,
    seed: int,
    mass_min: float,
    mass_max: float,
    bin_width: float,
    batch_size: int,
    log=print,
) -> dict:
    z, component_id = load_z(prior, max_events, seed)
    native_core, native_tail = native_decoder_multipliers(checkpoint)
    decoded, sigma_summary = decode_subset(model, z, batch_size, device, native_core, native_tail)
    moments = state_moments(decoded, component_id, mass_min, mass_max, bin_width)
    out = {
        "status": "artifact-measured",
        "prior": str(prior),
        "events": int(len(z)),
        "native_multipliers": {"core": native_core, "tail": native_tail},
        "states": {},
        "learned_sigma_native": sigma_summary,
    }
    for state, values in moments.items():
        cms = CMS_FIT_MASS_GEV[state]
        out["states"][state] = {
            "events": values["events"],
            "median_gev": values["median_gev"],
            "median_minus_cms_gev": float(values["median_gev"] - cms),
            "median_minus_cms_mev": float((values["median_gev"] - cms) * 1000.0),
            "std_gev": values["std_gev"],
            "q16_gev": values["q16_gev"],
            "q84_gev": values["q84_gev"],
            "cms_fit_mass_gev": cms,
        }
        log(
            f"    T {state}: median-CMS {out['states'][state]['median_minus_cms_mev']:+.1f} MeV, "
            f"std {values['std_gev'] * 1000:.1f} MeV"
        )
    return out


# ---------------------------------------------------------------------------
# gates and the collapse-proof selection key
# ---------------------------------------------------------------------------

def response_gate(region: str, block: dict) -> dict:
    ratio = block.get("within_std_over_required")
    noise_fraction = block.get("noise_variance_fraction")
    sigma_only = block.get("sigma_noise_only_gev")
    minimum_noise = NOISE_ONLY_MIN_GEV.get(region)
    width_pass = bool(
        ratio is not None
        and math.isfinite(ratio)
        and REQUIRED_RATIO_BAND[0] <= ratio <= REQUIRED_RATIO_BAND[1]
    )
    noise_pass = bool(
        sigma_only is not None
        and minimum_noise is not None
        and math.isfinite(sigma_only)
        and sigma_only >= minimum_noise
    )
    degenerate = bool(
        ratio is None
        or not math.isfinite(float(ratio))
        or float(ratio) < DEGENERATE_WIDTH_RATIO
    )
    reasons = []
    if not width_pass:
        reasons.append(f"{region}: within/required {ratio} outside {REQUIRED_RATIO_BAND}")
    if not noise_pass:
        reasons.append(f"{region}: sigma_noise_only {sigma_only} GeV below gate {minimum_noise} GeV")
    if degenerate:
        reasons.append(
            f"{region}: within/required {ratio} below {DEGENERATE_WIDTH_RATIO} - the noise "
            f"channel cannot be carrying the resolution (noise share "
            f"{noise_fraction if noise_fraction is None else round(float(noise_fraction), 4)}); "
            "the deterministic mean map is faking it"
        )
    return {
        "region": region,
        "within_std_over_required": ratio,
        "sigma_noise_only_gev": sigma_only,
        "noise_variance_fraction": noise_fraction,
        "mean_map_variance_fraction": block.get("mean_map_variance_fraction"),
        "width_pass": width_pass,
        "noise_pass": noise_pass,
        "degenerate": degenerate,
        "pass": bool(width_pass and noise_pass and not degenerate),
        "reasons": reasons,
    }


def conditional_gate(region: str, block: dict) -> dict:
    native = block.get("native") or {}
    median = native.get("decode_ratio_median")
    worst = native.get("decode_ratio_max")
    median_pass = bool(median is not None and math.isfinite(median) and median <= SLICE_MEDIAN_RATIO_MAX)
    max_pass = bool(worst is not None and math.isfinite(worst) and worst <= SLICE_MAX_RATIO_MAX)
    reasons = []
    if not median_pass:
        reasons.append(f"{region}: native slice ratio median {median} > {SLICE_MEDIAN_RATIO_MAX}")
    if not max_pass:
        reasons.append(f"{region}: native slice ratio max {worst} > {SLICE_MAX_RATIO_MAX}")
    return {
        "region": region,
        "decode_ratio_median_native": median,
        "decode_ratio_max_native": worst,
        "decode_ratio_median_zero": (block.get("zero") or {}).get("decode_ratio_median"),
        "median_pass": median_pass,
        "max_pass": max_pass,
        "pass": bool(median_pass and max_pass),
        "reasons": reasons,
    }


def entry_gates(entry: dict) -> dict:
    response = {region: response_gate(region, block) for region, block in (entry.get("R") or {}).items()}
    conditional = {region: conditional_gate(region, block) for region, block in (entry.get("C") or {}).items()}
    r_pass = bool(response) and all(item["pass"] for item in response.values())
    c_pass = (bool(conditional) and all(item["pass"] for item in conditional.values())) if conditional else None
    degenerate = any(item["degenerate"] for item in response.values())
    ratios = [
        item["within_std_over_required"]
        for item in response.values()
        if item["within_std_over_required"] is not None
        and math.isfinite(float(item["within_std_over_required"]))
    ]
    # A deterministic checkpoint has within-z width exactly 0; log(0) is not a
    # number, and "infinitely far from the required width" is the right answer.
    deviations = [abs(math.log(value)) if value > 0 else float("inf") for value in ratios]
    worst_deviation = max(deviations, default=float("inf"))
    median_ratios = [
        item["decode_ratio_median_native"]
        for item in conditional.values()
        if item["decode_ratio_median_native"] is not None
    ]
    return {
        "R": response,
        "C": conditional,
        "R_pass": r_pass,
        "C_pass": c_pass,
        "degenerate": degenerate,
        "identified": bool(r_pass and not degenerate),
        "worst_R_log_deviation": worst_deviation,
        "worst_C_median_ratio": max(median_ratios) if median_ratios else float("inf"),
        "reasons": [
            reason
            for item in list(response.values()) + list(conditional.values())
            for reason in item["reasons"]
        ],
    }


def _selection_view(entry: dict) -> dict:
    """Transfer-blind view of an entry (Axis T must never reach the key)."""
    gates = entry.get("gates") or entry_gates(entry)
    return {
        "label": entry.get("label"),
        "identified": gates["identified"],
        "degenerate": gates["degenerate"],
        "R_pass": gates["R_pass"],
        "C_pass": gates["C_pass"],
        "worst_R_log_deviation": gates["worst_R_log_deviation"],
        "worst_C_median_ratio": gates["worst_C_median_ratio"],
        "in_domain_score": entry.get("in_domain_score"),
    }


def selection_key(entries: list[dict]) -> dict:
    """Tiered, collapse-proof ranking.

    Tier 0 rejects every checkpoint whose response gate fails or that is flagged
    degenerate. Tier 1 ranks the survivors by conditional (per-slice) closure,
    then by the worst-region in-domain score. If nothing survives, the key
    reports identification_failed and returns the least-bad response gate - it
    never silently crowns a marginal winner.
    """
    views = [_selection_view(entry) for entry in entries]
    survivors = [view for view in views if view["identified"]]
    inf = float("inf")

    def rank_key(view: dict):
        score = view["in_domain_score"]
        return (
            view["worst_C_median_ratio"] if math.isfinite(view["worst_C_median_ratio"]) else inf,
            score if score is not None else inf,
            view["label"] or "",
        )

    if survivors:
        ranked = sorted(survivors, key=rank_key)
        status = "identified"
        rejected = [view["label"] for view in views if not view["identified"]]
    else:
        ranked = sorted(views, key=lambda view: (view["worst_R_log_deviation"], view["label"] or ""))
        status = "identification_failed"
        rejected = []
    return {
        "status": status,
        "selected": ranked[0]["label"] if ranked else None,
        "ranking": [view["label"] for view in ranked],
        "rejected": rejected,
        "criteria": {
            "reject_if": "R gate fails or degenerate",
            "then_rank_by": "worst conditional slice ratio, then in-domain score",
            "fallback": "least-bad R gate, status=identification_failed",
            "transfer_ignored": True,
        },
    }


def selection_is_transfer_blind(entries: list[dict]) -> bool:
    """Contract check: Axis T must not change the selection key."""
    stripped = []
    for entry in entries:
        clone = {key: value for key, value in entry.items() if key != "T"}
        stripped.append(clone)
    return selection_key(entries) == selection_key(stripped)


def transfer_criteria(entries: list[dict]) -> dict:
    """The four plan-tree D2 criteria, evaluated at arm level when possible."""
    by_label = {entry.get("label"): entry for entry in entries}

    def pick(*names):
        for name in names:
            for label, entry in by_label.items():
                if label and name in label:
                    return entry
        return None

    best = pick("best_model", "stage3_best", "stage2_best")
    last = pick("last_model")
    if best is None or last is None:
        return {"status": "insufficient_entries"}

    def medians(entry):
        states = ((entry.get("T") or {}).get("states")) or {}
        return [states.get(name, {}).get("median_minus_cms_mev") for name in UPSILON_STATES]

    best_medians, last_medians = medians(best), medians(last)
    in_band = all(value is not None and abs(value) <= 30.0 for value in best_medians + last_medians)
    drift = [None if (a is None or b is None) else abs(a - b) for a, b in zip(best_medians, last_medians)]
    drift_ok = all(value is not None and value < 40.0 for value in drift)
    sigma_only = [
        (entry.get("R") or {}).get(state, {}).get("sigma_noise_only_gev")
        for entry in (best, last)
        for state in UPSILON_STATES
    ]
    sigma_measured = all(value is not None for value in sigma_only)
    sigma_ok = bool(sigma_measured and all(value >= 0.050 for value in sigma_only))
    slices = [(entry.get("gates") or entry_gates(entry))["worst_C_median_ratio"] for entry in (best, last)]
    slices_ok = all(math.isfinite(value) and value <= SLICE_MEDIAN_RATIO_MAX for value in slices)
    scores = [entry.get("in_domain_score") for entry in (best, last)]
    return {
        "status": "artifact-measured",
        "best": best.get("label"),
        "last": last.get("label"),
        "criterion_1_medians": {
            "pass": bool(in_band and drift_ok),
            "best_medians_mev": best_medians,
            "last_medians_mev": last_medians,
            "drift_mev": drift,
            "band_mev": 30.0,
            "drift_limit_mev": 40.0,
        },
        "criterion_2_noise_carries_width": {
            "pass": sigma_ok if sigma_measured else None,
            "measured": sigma_measured,
            "sigma_noise_only_gev": sigma_only,
            "threshold_gev": 0.050,
            "note": None if sigma_measured else "requires --include-upsilon (Axis R on Upsilon states)",
        },
        "criterion_3_in_domain_score": {
            "pass": None,
            "scores": scores,
            "legacy_reference": 0.839,
            "note": "honest-prior arms are not comparable to the legacy 0.839; coupled to D3",
        },
        "criterion_4_slice_closure": {
            "pass": bool(slices_ok),
            "worst_median_ratio": slices,
            "threshold": SLICE_MEDIAN_RATIO_MAX,
        },
    }


# ---------------------------------------------------------------------------
# data loading (with a legacy-prior fallback)
# ---------------------------------------------------------------------------

def load_regions_with_fallback(config: dict, log=print):
    """Load the cached region splits; if a configured prior has moved, retry
    with the resolved path (data/legacy/...). Nothing on disk is modified."""
    try:
        arrays, cache_info, region_configs, pair_indices = load_joint_regions(
            config, num_samples=None, use_cache=True, log=lambda message: None
        )
        return arrays, cache_info, region_configs, pair_indices, config, False
    except FileNotFoundError as error:
        log(f"[data] primary load failed ({error}); retrying with resolved prior paths")
        patched = copy.deepcopy(config)
        changed = False
        for region in ("jpsi", "z"):
            try:
                path, used, fallback = resolve_prior_path(config, region)
            except FileNotFoundError:
                continue
            if fallback:
                patched["regions"][region]["paths"]["theory_prior_file"] = str(path)
                changed = True
        if not changed:
            raise
        arrays, cache_info, region_configs, pair_indices = load_joint_regions(
            patched, num_samples=None, use_cache=True, log=lambda message: None
        )
        return arrays, cache_info, region_configs, pair_indices, patched, True


# ---------------------------------------------------------------------------
# report writers
# ---------------------------------------------------------------------------

def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return "inf"
    return f"{value:.{digits}g}"


def write_report(path: Path, payload: dict) -> None:
    checkpoints = payload["checkpoints"]
    lines = [
        "# Response-identification scorecard",
        "",
        f"*{payload['created_utc']} - plan-tree leaf A0.4. Device {payload['device']}, "
        f"{payload['events_per_region']} fixed z, {payload['draws']} draws, "
        f"{payload['slice_samples']} slice events, seed {payload['seed']}.*",
        "",
        f"Selection key: **{payload['selection']['status']}** -> {payload['selection']['selected']}",
        "",
        "## Axis R - response identification",
        "",
        "| checkpoint | region | within-z robust [MeV] | required [MeV] | ratio | "
        "sigma_noise_only [MeV] | noise share | mean-map share | gate |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for label, entry in checkpoints.items():
        gates = entry["gates"]["R"]
        for region, block in entry["R"].items():
            gate = gates[region]
            required = block.get("required_robust_gev")
            lines.append(
                f"| {label} | {region} | {_fmt(block['within_robust_median_gev'] * 1000, 4)} | "
                f"{'n/a' if not required else _fmt(required * 1000, 4)} | "
                f"{_fmt(gate['within_std_over_required'], 3)} | "
                f"{_fmt(block['sigma_noise_only_gev'] * 1000, 4)} | "
                f"{_fmt(gate['noise_variance_fraction'], 3)} | "
                f"{_fmt(gate['mean_map_variance_fraction'], 3)} | "
                f"{'PASS' if gate['pass'] else 'FAIL'} |"
            )
    lines += [
        "",
        "## Axis C - conditional closure (decode, equal-count pair-pT slices)",
        "",
        "| checkpoint | region | ratio median (zero) | ratio median (native) | ratio max (native) | gate |",
        "|---|---|---|---|---|---|",
    ]
    for label, entry in checkpoints.items():
        gates = entry["gates"]["C"]
        for region, block in (entry.get("C") or {}).items():
            gate = gates[region]
            lines.append(
                f"| {label} | {region} | {_fmt(gate['decode_ratio_median_zero'], 3)} | "
                f"{_fmt(gate['decode_ratio_median_native'], 3)} | "
                f"{_fmt(gate['decode_ratio_max_native'], 3)} | "
                f"{'PASS' if gate['pass'] else 'FAIL'} |"
            )
    transfer = payload.get("transfer_criteria") or {}
    if transfer.get("status") == "artifact-measured":
        lines += [
            "",
            "## Axis T - Upsilon transfer (report-only, never selects)",
            "",
            "| criterion | pass | detail |",
            "|---|---|---|",
            f"| 1 medians within +/-30 MeV, drift < 40 MeV | {transfer['criterion_1_medians']['pass']} | "
            f"best {transfer['criterion_1_medians']['best_medians_mev']}, "
            f"last {transfer['criterion_1_medians']['last_medians_mev']} |",
            f"| 2 sigma_noise_only >= 50 MeV | {transfer['criterion_2_noise_carries_width']['pass']} | "
            f"{[_fmt(v, 3) for v in transfer['criterion_2_noise_carries_width']['sigma_noise_only_gev']]} |",
            f"| 3 in-domain score (D3-coupled) | n/a | "
            f"{transfer['criterion_3_in_domain_score']['scores']} |",
            f"| 4 slice closure | {transfer['criterion_4_slice_closure']['pass']} | "
            f"worst median ratio {_fmt(transfer['criterion_4_slice_closure']['worst_median_ratio'], 3)} |",
        ]
    if payload["selection"]["rejected"]:
        lines += ["", "Rejected by the key: " + ", ".join(str(x) for x in payload["selection"]["rejected"])]
    lines += [
        "",
        "## Gate bands",
        "",
        f"- R width ratio band {REQUIRED_RATIO_BAND}; sigma_noise_only minima "
        f"{NOISE_ONLY_MIN_GEV} GeV; degenerate when within/required < {DEGENERATE_WIDTH_RATIO} "
        f"(noise share is reported but not a gate; it tracks the prior width).",
        f"- C: median slice ratio <= {SLICE_MEDIAN_RATIO_MAX}, worst slice <= {SLICE_MAX_RATIO_MAX}.",
        "- T is report-only and cannot enter the selection key (selection_is_transfer_blind).",
        "",
        "## Provenance",
        "",
        f"- git {payload['git_rev']} (dirty={payload['git_dirty']}), torch {payload['torch_version']}, "
        f"cuda={payload['cuda_available']}",
        f"- reference resolution: {(payload['reference_resolution'] or {}).get('status')}",
        f"- runtime {payload['runtime_seconds']:.1f}s",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, payload: dict) -> None:
    rows = [
        "checkpoint,global_epoch,stage,sha256,R_pass,C_pass,degenerate,identified,"
        "region,within_robust_mev,required_mev,ratio,sigma_noise_only_mev,noise_share,"
        "slice_ratio_median_native,slice_ratio_max_native,in_domain_score"
    ]
    for label, entry in payload["checkpoints"].items():
        for region, block in entry["R"].items():
            gate_r = entry["gates"]["R"][region]
            gate_c = (entry["gates"].get("C") or {}).get(region, {})
            required = block.get("required_robust_gev")
            rows.append(
                ",".join(
                    str(value)
                    for value in (
                        label,
                        entry.get("global_epoch"),
                        entry.get("stage_name"),
                        entry.get("checkpoint_sha256"),
                        entry["gates"]["R_pass"],
                        entry["gates"]["C_pass"],
                        entry["gates"]["degenerate"],
                        entry["gates"]["identified"],
                        region,
                        f"{block['within_robust_median_gev'] * 1000:.4f}",
                        "" if not required else f"{required * 1000:.4f}",
                        _fmt(gate_r["within_std_over_required"], 4),
                        f"{block['sigma_noise_only_gev'] * 1000:.4f}",
                        _fmt(gate_r["noise_variance_fraction"], 4),
                        _fmt(gate_c.get("decode_ratio_median_native"), 4),
                        _fmt(gate_c.get("decode_ratio_max_native"), 4),
                        _fmt(entry.get("in_domain_score"), 6),
                    )
                )
            )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def make_plot(path: Path, payload: dict) -> None:
    labels = list(payload["checkpoints"])
    if not labels:
        return
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    x = np.arange(len(labels))
    regions = sorted({region for entry in payload["checkpoints"].values() for region in entry["R"]})
    width = 0.8 / max(len(regions), 1)
    for offset, region in enumerate(regions):
        ratios = [
            (entry["R"].get(region) or {}).get("within_std_over_required") or np.nan
            for entry in payload["checkpoints"].values()
        ]
        axes[0].bar(x + offset * width, ratios, width, label=region)
    axes[0].axhspan(*REQUIRED_RATIO_BAND, color="tab:green", alpha=0.15)
    axes[0].axhline(1.0, color="black", lw=0.8, ls=":")
    axes[0].set_xticks(x + 0.4 - width / 2)
    axes[0].set_xticklabels(labels, rotation=45, ha="right")
    axes[0].set_ylabel("within-z robust / required")
    axes[0].set_title("Axis R: is the resolution in the noise?")
    axes[0].legend(frameon=False, fontsize=8)
    for offset, region in enumerate(regions):
        shares = [
            (entry["R"].get(region) or {}).get("noise_variance_fraction") or np.nan
            for entry in payload["checkpoints"].values()
        ]
        axes[1].bar(x + offset * width, shares, width, label=region)
    axes[1].axhline(MIN_NOISE_VARIANCE_FRACTION, color="tab:red", lw=0.9, ls="--")
    axes[1].set_xticks(x + 0.4 - width / 2)
    axes[1].set_xticklabels(labels, rotation=45, ha="right")
    axes[1].set_ylabel("noise variance share")
    axes[1].set_title("Axis R: mean map vs noise")
    for offset, region in enumerate(regions):
        medians = [
            ((entry["gates"].get("C") or {}).get(region) or {}).get("decode_ratio_median_native") or np.nan
            for entry in payload["checkpoints"].values()
        ]
        axes[2].bar(x + offset * width, medians, width, label=region)
    axes[2].axhline(SLICE_MEDIAN_RATIO_MAX, color="tab:red", lw=0.9, ls="--")
    axes[2].set_xticks(x + 0.4 - width / 2)
    axes[2].set_xticklabels(labels, rotation=45, ha="right")
    axes[2].set_ylabel("median slice W1 / floor")
    axes[2].set_title("Axis C: conditional closure")
    figure.tight_layout()
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def score_run(
    run_dir: Path,
    *,
    checkpoints: list[tuple[str, Path]],
    output_dir: Path,
    device: torch.device,
    events: int = 256,
    draws: int = 32,
    slice_samples: int = 60000,
    batch_size: int = 4096,
    seed: int = 20261001,
    regions: tuple[str, ...] = ("jpsi", "z"),
    include_upsilon: bool = False,
    skip_slices: bool = False,
    reference_checkpoint: Path | None = None,
    upsilon_prior: Path = UPSILON_PRIOR,
    upsilon_events: int = 400000,
    log=print,
) -> dict:
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"

    def record(message: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        log(line)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    reference_source = reference_checkpoint or checkpoints[0][1]
    reference = compute_data_reference(reference_source, log=lambda message: None)

    first_model, _first_checkpoint, config = load_frozen_model(checkpoints[0][1], torch.device("cpu"))
    del first_model
    samples = fixed_z_samples(
        config,
        regions=regions,
        events=events,
        seed=seed,
        include_upsilon=include_upsilon,
        log=record,
    )
    region_arrays = factories = None
    used_fallback = False
    slice_status = "not_requested" if skip_slices else "pending"
    if not skip_slices:
        try:
            region_arrays, cache_info, _, _, config, used_fallback = load_regions_with_fallback(config, log=record)
            factories = build_loss_factories(config, region_arrays)
            slice_status = "measured"
            record(f"[data] region cache loaded (legacy prior fallback={used_fallback})")
        except Exception as error:  # pragma: no cover - data dependent
            region_arrays = factories = None
            slice_status = "unavailable: " + type(error).__name__ + ": " + str(error)
            record(f"[data] Axis C unavailable ({slice_status}); continuing with Axis R only")

    entries: dict[str, dict] = {}
    for index, (label, path) in enumerate(checkpoints):
        load_start = time.time()
        model, checkpoint, checkpoint_config = load_frozen_model(path, device)
        model.eval()
        native_core, native_tail = native_decoder_multipliers(checkpoint)
        stage = checkpoint.get("stage") or {}
        entry = {
            "label": label,
            "path": str(path),
            "checkpoint_sha256": sha256_file(path),
            "global_epoch": checkpoint.get("global_epoch"),
            "stage_name": stage.get("name") if isinstance(stage, dict) else None,
            "recorded_noise_multipliers": json_ready(checkpoint.get("noise_multipliers") or {}),
            "native_decoder_multipliers": {"core": native_core, "tail": native_tail},
            "in_domain_score": in_domain_score(checkpoint),
        }
        record(
            f"[{index + 1}/{len(checkpoints)}] {label}: ep {entry['global_epoch']} "
            f"{entry['stage_name']} native decoder ({native_core}, {native_tail}) "
            f"in-domain {entry['in_domain_score']} loaded in {time.time() - load_start:.1f}s"
        )
        entry["R"] = axis_response(
            model, checkpoint, samples, reference,
            device=device, draws=draws, batch_size=batch_size, seed=seed, index=index, log=record,
        )
        if factories is not None:
            entry["C"] = axis_conditional(
                model, region_arrays, factories,
                device=device, samples=slice_samples, seed=seed + 200 + index,
                native=(native_core, native_tail), log=record,
            )
        if include_upsilon:
            entry["T"] = axis_transfer(
                model, checkpoint, device=device, prior=upsilon_prior,
                max_events=upsilon_events, seed=seed + 300 + index,
                mass_min=8.5, mass_max=11.5, bin_width=0.02, batch_size=16384, log=record,
            )
        entry["gates"] = entry_gates(entry)
        record(
            f"    gates: R={'PASS' if entry['gates']['R_pass'] else 'FAIL'} "
            f"C={'PASS' if entry['gates']['C_pass'] else 'FAIL'} "
            f"degenerate={entry['gates']['degenerate']} identified={entry['gates']['identified']}"
        )
        entries[label] = entry
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    entry_list = list(entries.values())
    payload = {
        "schema_version": SCHEMA_VERSION,
        "diagnostic": "A0.4 response-identification scorecard",
        "status": "completed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.time() - started,
        "run_dir": str(run_dir),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "seed": seed,
        "events_per_region": events,
        "draws": draws,
        "slice_samples": slice_samples,
        "regions": list(regions),
        "include_upsilon": include_upsilon,
        "legacy_prior_fallback": used_fallback,
        "slice_status": slice_status,
        "reference_resolution": json_ready(reference),
        "reference_checkpoint": str(reference_source),
        "fixed_z_samples": {
            name: {key: value for key, value in sample.items() if key != "z"}
            for name, sample in samples.items()
        },
        "gate_bands": {
            "required_ratio_band": list(REQUIRED_RATIO_BAND),
            "noise_only_min_gev": NOISE_ONLY_MIN_GEV,
            "degenerate_width_ratio": DEGENERATE_WIDTH_RATIO,
            "min_noise_variance_fraction": MIN_NOISE_VARIANCE_FRACTION,
            "slice_median_ratio_max": SLICE_MEDIAN_RATIO_MAX,
            "slice_max_ratio_max": SLICE_MAX_RATIO_MAX,
        },
        "checkpoints": entries,
        "selection": selection_key(entry_list),
        "selection_is_transfer_blind": selection_is_transfer_blind(entry_list),
        "transfer_criteria": transfer_criteria(entry_list) if include_upsilon else {"status": "not_requested"},
        "git_rev": _git_rev(),
        "git_dirty": _git_dirty(),
    }
    (output_dir / "scorecard.json").write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(output_dir / "scorecard.csv", payload)
    write_report(output_dir / "SCORECARD.md", payload)
    make_plot(output_dir / "scorecard.png", payload)
    record(
        f"wrote {output_dir / 'scorecard.json'}, scorecard.csv, SCORECARD.md, scorecard.png; "
        f"selection={payload['selection']['status']} -> {payload['selection']['selected']}"
    )
    return payload


def _git_rev() -> str | None:
    try:
        import subprocess

        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:  # pragma: no cover
        return None


def _git_dirty() -> bool | None:
    try:
        import subprocess

        out = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True)
        return bool(out.strip())
    except Exception:  # pragma: no cover
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", default=None, metavar="LABEL=PATH",
                        help="checkpoint to score (repeatable); default best_model.pt + last_model.pt")
    parser.add_argument("--all-checkpoints", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="default: outputs/cms_Joint/scorecard/<run-dir name>")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--events", type=int, default=256)
    parser.add_argument("--draws", type=int, default=32)
    parser.add_argument("--slice-samples", type=int, default=60000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--regions", default="jpsi,z")
    parser.add_argument("--include-upsilon", action="store_true",
                        help="also run Axis T (report-only) on the held-out Upsilon prior")
    parser.add_argument("--skip-slices", action="store_true")
    parser.add_argument("--reference-checkpoint", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    checkpoints = resolve_checkpoints(run_dir, args.checkpoint, args.all_checkpoints)
    for label, path in checkpoints:
        if not path.exists():
            raise FileNotFoundError(f"checkpoint {label} not found: {path}")
    output_dir = args.output_dir or (REPO_ROOT / "outputs" / "cms_Joint" / "scorecard" / run_dir.name)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    regions = tuple(name.strip() for name in args.regions.split(",") if name.strip())
    payload = score_run(
        run_dir,
        checkpoints=checkpoints,
        output_dir=output_dir.expanduser().resolve(),
        device=device,
        events=args.events,
        draws=args.draws,
        slice_samples=args.slice_samples,
        batch_size=args.batch_size,
        seed=args.seed,
        regions=regions,
        include_upsilon=args.include_upsilon,
        skip_slices=args.skip_slices,
        reference_checkpoint=args.reference_checkpoint,
        log=(lambda message: None) if args.quiet else print,
    )
    print(json.dumps(payload["selection"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

