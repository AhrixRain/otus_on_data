#!/usr/bin/env python
"""C1 non-training probe: sliced-Wasserstein estimator quality for the joint model.

This is the C1 leaf of ``docs/project_tree.md`` ("EBSW / max-SWD /
control-variate SWD").  It does **not** train anything and it does not touch
``scripts/loss.py`` behaviour.  It measures, on seeded synthetic data that
mimics the 8-D joint four-vector problem, how the finite-sample floor and the
detection power of the distributional loss compare between:

* the existing random-slice ``sliced_wasserstein`` at several slice counts;
* a fixed-direction (common-random-numbers) SWD control;
* a fixed orthogonal-direction SWD control;
* learned-direction max-SWD via ``scripts_sota/max_swd.py``, whose directions
  are updated by adversarial ascent inside this probe only.

Two synthetic scenarios
-----------------------

``resonance``
    A narrow invariant-mass resonance (J/psi-like, default 3.0969 GeV with a
    5 MeV truth width) with broad kinematics, pushed through a per-muon
    cylindrical smearing.  The ``shift`` arm adds a small known mass shift
    (default 10 MeV).  This is the physics-like arm.

``hard_direction``
    Two 8-D Gaussians separated by a shift along a random unit direction.  This
    is the arm where random slicing is known to be blind and learned directions
    should win.

For every variant and every replicate the probe evaluates four statistics:

* ``signal``      est(truth, detector)           - the response itself
* ``shift``       est(truth, shifted detector)   - signal + a small known shift
* ``null``        est(truth, independent truth draw) - finite-sample floor
* ``null_b``      est(truth, independent detector draw) - placebo reference
* ``shift_delta`` shift - signal   (paired on the common first sample)
* ``null_delta``  null_b - signal  (paired placebo, same structure, no shift)

and reports mean/std/quantiles, signal-to-floor, and the detection power for
the small shift.

Artifacts are written to a NEW directory (default
``outputs/cms_Joint/swd_variant_probe/``): JSON, REPORT.md, PNG and PDF.
Nothing under ``data/``, no checkpoint, and no existing output directory is
modified.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

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
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from loss import sliced_wasserstein  # noqa: E402
from max_swd import MaxSlicedWasserstein  # noqa: E402


MUON_MASS_GEV = 0.1056583755
FEATURE_DIM = 8
P = 1

DEFAULT_CONFIG: dict = {
    "seed": 20260923,
    "scenarios": ["resonance", "hard_direction"],
    "sample_sizes": [1024, 4096],
    "replicates": 24,
    "feature_dim": FEATURE_DIM,
    "p": P,
    # resonance scenario
    "resonance_mass_gev": 3.0969,
    "resonance_width_gev": 0.005,
    "pt_scale_gev": 2.0,
    "sigma_logpt": 0.01,
    "sigma_eta": 0.005,
    "sigma_phi": 0.005,
    "mass_shift_gev": 0.010,
    "mass_shift_large_gev": 0.050,
    # hard-direction scenario
    "hard_shift": 0.5,
    "hard_small_shift": 0.05,
    # estimator variants
    "random_slices": [8, 32, 64, 256, 1024],
    "fixed_random_slices": 256,
    "orthogonal_slices": 8,
    "max_directions": [8, 32],
    "max_steps": 200,
    "max_lr": 0.5,
    "max_diversity_weight": 0.01,
    "max_update_every": 1,
    "max_direction_grad_clip": 1.0,
}


# ---------------------------------------------------------------------------
# synthetic data
# ---------------------------------------------------------------------------

def mass8(values: np.ndarray) -> np.ndarray:
    """Stable dimuon invariant mass from an (N, 8) [p-, E-, p+, E+] array."""
    values = np.asarray(values, dtype=np.float64)
    energy = values[:, 3] + values[:, 7]
    momentum = values[:, :3] + values[:, 4:7]
    return np.sqrt(np.maximum(energy * energy - np.sum(momentum * momentum, axis=1), 0.0))


def _lorentz_boost(p4: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Boost (E, px, py, pz) rows by velocity ``beta`` (same length)."""
    beta_sq = np.sum(beta * beta, axis=1, keepdims=True)
    gamma = 1.0 / np.sqrt(np.maximum(1.0 - beta_sq, 1e-12))
    energy = p4[:, 0:1]
    momentum = p4[:, 1:]
    beta_dot_p = np.sum(beta * momentum, axis=1, keepdims=True)
    energy_out = gamma * (energy - beta_dot_p)
    coefficient = (gamma - 1.0) * beta_dot_p / np.maximum(beta_sq, 1e-12) - gamma * energy
    momentum_out = momentum + coefficient * beta
    return np.concatenate([energy_out, momentum_out], axis=1)


def resonance_sample(
    rng: np.random.Generator,
    count: int,
    *,
    mass_gev: float = 3.0969,
    width_gev: float = 0.005,
    pt_scale_gev: float = 2.0,
    rapidity_max: float = 2.4,
) -> np.ndarray:
    """A narrow dimuon resonance with broad kinematics, exact two-body decay."""
    masses = mass_gev + width_gev * rng.normal(size=count)
    pt = rng.exponential(pt_scale_gev, size=count)
    rapidity = rng.uniform(-rapidity_max, rapidity_max, size=count)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=count)
    cos_theta = rng.uniform(-1.0, 1.0, size=count)
    phi_star = rng.uniform(0.0, 2.0 * np.pi, size=count)
    momentum_star = np.sqrt(np.maximum(masses * masses / 4.0 - MUON_MASS_GEV**2, 0.0))
    sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)
    p_rest = np.stack(
        [
            momentum_star * sin_theta * np.cos(phi_star),
            momentum_star * sin_theta * np.sin(phi_star),
            momentum_star * cos_theta,
        ],
        axis=1,
    )
    energy_star = (masses / 2.0)[:, None]
    first = np.concatenate([energy_star, p_rest], axis=1)
    second = np.concatenate([energy_star, -p_rest], axis=1)
    transverse_mass = np.sqrt(masses * masses + pt * pt)
    pair = np.stack(
        [
            transverse_mass * np.cosh(rapidity),
            pt * np.cos(phi),
            pt * np.sin(phi),
            transverse_mass * np.sinh(rapidity),
        ],
        axis=1,
    )
    beta = pair[:, 1:] / pair[:, 0:1]
    first_lab = _lorentz_boost(first, beta)
    second_lab = _lorentz_boost(second, beta)
    return np.concatenate(
        [first_lab[:, 1:], first_lab[:, 0:1], second_lab[:, 1:], second_lab[:, 0:1]],
        axis=1,
    ).astype(np.float32)


def smear_cylindrical(
    values: np.ndarray,
    rng: np.random.Generator,
    *,
    sigma_logpt: float = 0.01,
    sigma_eta: float = 0.005,
    sigma_phi: float = 0.005,
) -> np.ndarray:
    """Per-muon Gaussian smearing in (log pT, eta, phi), rebuilt on shell."""
    values = np.asarray(values, dtype=np.float64)
    blocks = []
    for offset in (0, 4):
        px = values[:, offset]
        py = values[:, offset + 1]
        pz = values[:, offset + 2]
        pt = np.hypot(px, py)
        phi = np.arctan2(py, px)
        eta = np.arcsinh(pz / np.maximum(pt, 1e-9))
        pt = pt * np.exp(rng.normal(0.0, sigma_logpt, size=len(values)))
        eta = eta + rng.normal(0.0, sigma_eta, size=len(values))
        phi = phi + rng.normal(0.0, sigma_phi, size=len(values))
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        energy = np.sqrt(px * px + py * py + pz * pz + MUON_MASS_GEV**2)
        blocks.append(np.stack([px, py, pz, energy], axis=1))
    return np.concatenate(blocks, axis=1).astype(np.float32)


def scale_mass(values: np.ndarray, mass_shift_gev: float, mass_gev: float) -> np.ndarray:
    """Scale all four-momenta so the invariant mass shifts by ``mass_shift_gev``."""
    factor = 1.0 + float(mass_shift_gev) / float(mass_gev)
    return (np.asarray(values, dtype=np.float64) * factor).astype(np.float32)


def reference_standardization(
    values: np.ndarray, eps: float = 1e-8
) -> tuple[np.ndarray, np.ndarray]:
    """Per-coordinate mean/std used as the fixed standardization reference."""
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std > eps, std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


def standardize_features(
    values: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Apply the fixed reference standardization (mirrors the joint loss option)."""
    return ((np.asarray(values, dtype=np.float64) - mean) / std).astype(np.float32)


def scenario_reference(
    scenario: str, config: dict, *, reference_count: int = 131072
) -> tuple[np.ndarray, np.ndarray]:
    """Build the fixed standardization reference for a scenario."""
    rng = np.random.default_rng(int(config["seed"]) + 777)
    if scenario == "resonance":
        values = resonance_sample(
            rng,
            reference_count,
            mass_gev=config["resonance_mass_gev"],
            width_gev=config["resonance_width_gev"],
            pt_scale_gev=config["pt_scale_gev"],
        )
    elif scenario == "hard_direction":
        values = rng.normal(size=(reference_count, config["feature_dim"])).astype(np.float32)
    else:
        raise ValueError(f"unknown scenario {scenario!r}")
    return reference_standardization(values)


def hard_direction_sample(
    rng: np.random.Generator, count: int, feature_dim: int, shift: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two isotropic Gaussians separated by ``shift`` along a random unit vector."""
    base = rng.normal(size=(count, feature_dim))
    direction = rng.normal(size=feature_dim)
    direction /= np.linalg.norm(direction)
    shifted = base + shift * direction
    return base.astype(np.float32), shifted.astype(np.float32), direction.astype(np.float32)


def build_replicate(
    scenario: str,
    count: int,
    seed: int,
    config: dict,
    reference_stats: tuple[np.ndarray, np.ndarray],
) -> dict:
    """Build one replicate's independent samples (standardized) and metadata.

    The unpaired joint loss compares independent batches (prior/decoded vs CMS),
    and so does this probe: ``truth`` and ``detector`` are independent draws from
    their distributions, and ``detector`` / ``detector_b`` / ``shifted`` are
    independent draws from the detector / shifted-detector distributions.
    """
    rng = np.random.default_rng(seed)
    reference_mean, reference_std = reference_stats
    if scenario == "resonance":

        def truth_draw() -> np.ndarray:
            return resonance_sample(
                rng,
                count,
                mass_gev=config["resonance_mass_gev"],
                width_gev=config["resonance_width_gev"],
                pt_scale_gev=config["pt_scale_gev"],
            )

        def detector_draw(noise_seed: int, shift_gev: float = 0.0) -> np.ndarray:
            values = truth_draw()
            if shift_gev:
                values = scale_mass(values, shift_gev, config["resonance_mass_gev"])
            return smear_cylindrical(
                values,
                np.random.default_rng(noise_seed),
                sigma_logpt=config["sigma_logpt"],
                sigma_eta=config["sigma_eta"],
                sigma_phi=config["sigma_phi"],
            )

        truth = truth_draw()
        detector = detector_draw(seed + 1)
        detector_b = detector_draw(seed + 2)
        shifted = detector_draw(seed + 3, shift_gev=config["mass_shift_gev"])
        shifted_large = detector_draw(
            seed + 5, shift_gev=config["mass_shift_large_gev"]
        )
        null_truth = truth_draw()
        train_truth = truth_draw()
        train_detector = detector_draw(seed + 4)
        metadata = {
            "truth_mass_mean_gev": float(mass8(truth).mean()),
            "truth_mass_std_gev": float(mass8(truth).std()),
            "detector_mass_mean_gev": float(mass8(detector).mean()),
            "detector_mass_std_gev": float(mass8(detector).std()),
            "shifted_mass_mean_gev": float(mass8(shifted).mean()),
            "mass_shift_gev": float(config["mass_shift_gev"]),
            "direction": None,
        }
    elif scenario == "hard_direction":

        def normal() -> np.ndarray:
            return rng.normal(size=(count, config["feature_dim"])).astype(np.float32)

        direction = rng.normal(size=config["feature_dim"])
        direction /= np.linalg.norm(direction)
        truth = normal()
        detector = normal() + config["hard_shift"] * direction
        detector_b = normal() + config["hard_shift"] * direction
        shifted = (
            normal() + (config["hard_shift"] + config["hard_small_shift"]) * direction
        )
        shifted_large = (
            normal() + (config["hard_shift"] + 2.0 * config["hard_small_shift"]) * direction
        )
        null_truth = normal()
        train_truth = normal()
        train_detector = normal() + config["hard_shift"] * direction
        metadata = {
            "hard_shift": float(config["hard_shift"]),
            "hard_small_shift": float(config["hard_small_shift"]),
            "direction": direction,
        }
    else:
        raise ValueError(f"unknown scenario {scenario!r}")
    standardized = {
        key: standardize_features(value, reference_mean, reference_std)
        for key, value in {
            "truth": truth,
            "detector": detector,
            "detector_b": detector_b,
            "shifted": shifted,
            "shifted_large": shifted_large,
            "null_truth": null_truth,
            "train_truth": train_truth,
            "train_detector": train_detector,
        }.items()
    }
    if metadata["direction"] is not None:
        direction = metadata["direction"].astype(np.float64) / reference_std
        direction = direction / np.linalg.norm(direction)
        metadata["direction"] = direction.astype(np.float32)
    standardized["metadata"] = metadata
    return standardized


def swd_with_directions(
    truth: torch.Tensor,
    pred: torch.Tensor,
    directions: torch.Tensor,
    p: int = 1,
) -> torch.Tensor:
    """Sliced p-Wasserstein at a supplied direction matrix (equal-size batches)."""
    if truth.shape[0] != pred.shape[0]:
        raise ValueError("swd_with_directions requires equal-size batches")
    projected_truth = torch.sort(truth @ directions.t(), dim=0).values
    projected_pred = torch.sort(pred @ directions.t(), dim=0).values
    diff = projected_truth - projected_pred
    if int(p) == 1:
        return diff.abs().mean()
    if int(p) == 2:
        return diff.square().mean()
    raise ValueError("Only p=1 and p=2 are supported")


def random_swd(
    truth: torch.Tensor, pred: torch.Tensor, num_slices: int, p: int, seed: int
) -> float:
    torch.manual_seed(int(seed))
    return float(sliced_wasserstein(truth, pred, int(num_slices), int(p)).detach())


def orthogonal_directions(feature_dim: int, count: int, seed: int) -> torch.Tensor:
    """Fixed orthonormal directions from a QR of a Gaussian matrix."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    matrix = torch.randn(feature_dim, int(count), generator=generator)
    q, _ = torch.linalg.qr(matrix)
    return q.t().contiguous()


def learned_max_swd(
    truth: torch.Tensor,
    pred: torch.Tensor,
    *,
    num_directions: int,
    steps: int,
    lr: float,
    diversity_weight: float,
    update_every: int,
    direction_grad_clip: float,
    p: int,
    seed: int,
) -> tuple[float, torch.Tensor]:
    """Adversarial direction ascent on this pair, then the distance at those directions."""
    torch.manual_seed(int(seed))
    module = MaxSlicedWasserstein(
        feature_dim=truth.shape[1],
        num_directions=int(num_directions),
        p=int(p),
        lr=float(lr),
        diversity_weight=float(diversity_weight),
        update_every=int(update_every),
        direction_grad_clip=float(direction_grad_clip),
    ).to(truth.device)
    for _ in range(int(steps)):
        module.adversarial_step(truth, pred)
    with torch.no_grad():
        value = float(module(truth, pred).detach())
        directions = module.normalized_directions().detach()
    return value, directions


def variant_specs(config: dict) -> list[dict]:
    specs: list[dict] = []
    for slices in config["random_slices"]:
        specs.append({"name": f"swd_{slices}", "kind": "random", "num_slices": int(slices)})
    count = int(config["fixed_random_slices"])
    specs.append(
        {"name": f"swd_fixed_{count}", "kind": "fixed_random", "num_slices": count}
    )
    count = int(config["orthogonal_slices"])
    specs.append(
        {"name": f"swd_orthogonal_{count}", "kind": "orthogonal", "num_slices": count}
    )
    for directions in config["max_directions"]:
        specs.append(
            {
                "name": f"maxswd_{directions}_in",
                "kind": "max",
                "protocol": "in_sample",
                "num_directions": int(directions),
            }
        )
        specs.append(
            {
                "name": f"maxswd_{directions}_heldout",
                "kind": "max",
                "protocol": "heldout",
                "num_directions": int(directions),
            }
        )
    return specs


def train_max_directions(
    truth: torch.Tensor, pred: torch.Tensor, spec: dict, config: dict, seed: int
) -> torch.Tensor:
    """Train max-SWD directions on a separate pair (heldout protocol)."""
    _, directions = learned_max_swd(
        truth,
        pred,
        num_directions=spec["num_directions"],
        steps=config["max_steps"],
        lr=config["max_lr"],
        diversity_weight=config["max_diversity_weight"],
        update_every=config["max_update_every"],
        direction_grad_clip=config["max_direction_grad_clip"],
        p=int(config["p"]),
        seed=seed,
    )
    return directions


def evaluate_variant(
    spec: dict,
    samples_t: dict,
    *,
    config: dict,
    seed: int,
    device: torch.device,
    fixed_directions: torch.Tensor | None,
    orthogonal: torch.Tensor | None,
    heldout_directions: torch.Tensor | None = None,
) -> dict:
    """Evaluate the four statistics of one variant on one replicate."""
    p = int(config["p"])

    def value(
        truth_key: str, pred_key: str, stat_index: int
    ) -> tuple[float, torch.Tensor | None]:
        truth = samples_t[truth_key]
        pred = samples_t[pred_key]
        if spec["kind"] == "random":
            return random_swd(truth, pred, spec["num_slices"], p, seed + stat_index), None
        if spec["kind"] == "fixed_random":
            assert fixed_directions is not None
            with torch.no_grad():
                return float(swd_with_directions(truth, pred, fixed_directions, p)), None
        if spec["kind"] == "orthogonal":
            assert orthogonal is not None
            with torch.no_grad():
                return float(swd_with_directions(truth, pred, orthogonal, p)), None
        if spec["kind"] == "max":
            if spec.get("protocol") == "heldout":
                assert heldout_directions is not None
                with torch.no_grad():
                    return (
                        float(swd_with_directions(truth, pred, heldout_directions, p)),
                        heldout_directions,
                    )
            return learned_max_swd(
                truth,
                pred,
                num_directions=spec["num_directions"],
                steps=config["max_steps"],
                lr=config["max_lr"],
                diversity_weight=config["max_diversity_weight"],
                update_every=config["max_update_every"],
                direction_grad_clip=config["max_direction_grad_clip"],
                p=p,
                seed=seed + stat_index,
            )
        raise ValueError(spec["kind"])

    signal, signal_dirs = value("truth", "detector", 0)
    shifted, _ = value("truth", "shifted", 1)
    null, _ = value("truth", "null_truth", 2)
    null_b, _ = value("truth", "detector_b", 3)
    shifted_large, _ = value("truth", "shifted_large", 4)
    return {
        "signal": signal,
        "shift": shifted,
        "shift_large": shifted_large,
        "null": null,
        "null_b": null_b,
        "signal_directions": signal_dirs,
        "counter": {
            "estimator_calls": 4,
            "adversarial_steps": int(config["max_steps"]) if spec["kind"] == "max" else 0,
        },
    }


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def summarize(values) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot summarize an empty array")
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "sem": float(array.std(ddof=1) / math.sqrt(array.size)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "q05": float(np.quantile(array, 0.05)),
        "q95": float(np.quantile(array, 0.95)),
    }


def detection_power(delta, null_delta, alpha: float = 0.05) -> dict:
    """Fraction of paired shift deltas above the null-delta (1-alpha) quantile."""
    delta = np.asarray(delta, dtype=np.float64)
    null_delta = np.asarray(null_delta, dtype=np.float64)
    if delta.size == 0 or null_delta.size == 0:
        raise ValueError("detection power needs non-empty arrays")
    empirical_threshold = float(np.quantile(null_delta, 1.0 - alpha))
    gaussian_threshold = float(null_delta.mean() + 1.6448536269514722 * null_delta.std(ddof=1))
    return {
        "alpha": float(alpha),
        "null_threshold_empirical": empirical_threshold,
        "null_threshold_gaussian": gaussian_threshold,
        "power_empirical": float(np.mean(delta > empirical_threshold)),
        "power_gaussian": float(np.mean(delta > gaussian_threshold)),
        "shift_delta_mean": float(delta.mean()),
        "shift_delta_std": float(delta.std(ddof=1)) if delta.size > 1 else 0.0,
    }


def aggregate_variant(records: list[dict], spec: dict, config: dict) -> dict:
    signal = np.array([record["signal"] for record in records])
    shifted = np.array([record["shift"] for record in records])
    null = np.array([record["null"] for record in records])
    shifted_large = np.array([record["shift_large"] for record in records])
    null_b = np.array([record["null_b"] for record in records])
    delta = shifted - signal
    delta_large = shifted_large - signal
    # Paired placebo: same common first sample, two independent detector draws.
    null_delta = null_b - signal
    summary_null = summarize(null)
    summary_signal = summarize(signal)
    floor = summary_null["mean"]
    result = {
        "name": spec["name"],
        "kind": spec["kind"],
        "signal": summary_signal,
        "shift": summarize(shifted),
        "null": summary_null,
        "shift_delta": summarize(delta),
        "shift_delta_large": summarize(delta_large),
        "detection": detection_power(delta, null_delta),
        "detection_large": detection_power(delta_large, null_delta),
        "signal_excess": float(summary_signal["mean"] - floor),
        "signal_to_floor": float(summary_signal["mean"] / floor) if floor > 0 else None,
        "signal_z": (
            float((summary_signal["mean"] - floor) / summary_null["std"])
            if summary_null["std"] > 0
            else None
        ),
        "smearing_detection_power": float(
            np.mean(signal > np.quantile(null, 1.0 - 0.05))
        ),
    }
    if spec["kind"] == "max":
        result["num_directions"] = int(spec["num_directions"])
        result["protocol"] = str(spec.get("protocol", "in_sample"))
        result["max_steps"] = int(config["max_steps"])
        result["max_lr"] = float(config["max_lr"])
        result["max_diversity_weight"] = float(config["max_diversity_weight"])
    else:
        result["num_slices"] = int(spec["num_slices"])
    if spec["kind"] == "fixed_random":
        result["direction_seed"] = int(config["seed"])
    return result


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run_probe(config: dict, *, device: torch.device, log=print) -> dict:
    specs = variant_specs(config)
    started = time.time()
    scenarios: dict[str, dict] = {}
    for scenario in config["scenarios"]:
        reference_stats = scenario_reference(scenario, config)
        by_size: dict[str, dict] = {}
        for count in config["sample_sizes"]:
            variant_records: dict[str, list[dict]] = {spec["name"]: [] for spec in specs}
            alignment_records: dict[str, list[tuple[float, float]]] = {}
            replicate_metadata: list[dict] = []
            for replicate in range(int(config["replicates"])):
                replicate_seed = int(config["seed"]) + 1000 * replicate
                built = build_replicate(
                    scenario, int(count), replicate_seed, config, reference_stats
                )
                samples_t = {
                    key: torch.as_tensor(built[key], dtype=torch.float32, device=device)
                    for key in (
                        "truth",
                        "detector",
                        "shifted",
                        "shifted_large",
                        "null_truth",
                        "detector_b",
                        "train_truth",
                        "train_detector",
                    )
                }
                replicate_metadata.append(
                    {key: value for key, value in built["metadata"].items() if key != "direction"}
                )
                direction = built["metadata"]["direction"]
                if direction is not None:
                    direction = direction.astype(np.float64)
                    direction /= np.linalg.norm(direction)
                for spec_index, spec in enumerate(specs):
                    fixed = None
                    orthogonal = None
                    if spec["kind"] == "fixed_random":
                        generator = torch.Generator(device="cpu").manual_seed(
                            int(config["seed"]) + 7
                        )
                        fixed = torch.randn(
                            spec["num_slices"], config["feature_dim"], generator=generator
                        )
                        fixed = F.normalize(fixed, dim=1, eps=1e-12).to(device)
                    if spec["kind"] == "orthogonal":
                        orthogonal = orthogonal_directions(
                            config["feature_dim"],
                            spec["num_slices"],
                            int(config["seed"]) + 11,
                        ).to(device)
                    heldout_directions = None
                    if spec["kind"] == "max" and spec.get("protocol") == "heldout":
                        heldout_directions = train_max_directions(
                            samples_t["train_truth"],
                            samples_t["train_detector"],
                            spec,
                            config,
                            replicate_seed + 100 * spec_index,
                        )
                    record = evaluate_variant(
                        spec,
                        samples_t,
                        config=config,
                        seed=replicate_seed + 100 * spec_index,
                        device=device,
                        fixed_directions=fixed,
                        orthogonal=orthogonal,
                        heldout_directions=heldout_directions,
                    )
                    variant_records[spec["name"]].append(record)
                    if direction is not None and record["signal_directions"] is not None:
                        dots = np.abs(
                            record["signal_directions"].detach().cpu().numpy() @ direction
                        )
                        alignment_records.setdefault(spec["name"], []).append(
                            (float(dots.max()), float(dots.mean()))
                        )
                log(
                    f"[{scenario}/{count}] replicate {replicate + 1}/{config['replicates']} done"
                )
            results = {
                spec["name"]: aggregate_variant(
                    variant_records[spec["name"]], spec, config
                )
                for spec in specs
            }
            for name, values in alignment_records.items():
                results[f"{name}_alignment"] = {
                    "max_abs_cosine_mean": float(np.mean([value[0] for value in values])),
                    "mean_abs_cosine_mean": float(np.mean([value[1] for value in values])),
                }
            by_size[str(count)] = {
                "variants": results,
                "replicate_metadata": replicate_metadata,
            }
        scenarios[scenario] = {"by_sample_size": by_size}
    return {
        "schema_version": 1,
        "probe": "C1 sliced-Wasserstein estimator quality (non-training)",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.time() - started,
        "repo_root": str(REPO_ROOT),
        "git_rev": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain").strip()),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "config": config,
        "scenarios": scenarios,
    }


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return ""


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


# ---------------------------------------------------------------------------
# artifacts
# ---------------------------------------------------------------------------

def write_csv(path: Path, payload: dict) -> None:
    rows = [
        "scenario,sample_size,variant,kind,param,signal_mean,null_mean,floor_ratio,"
        "signal_to_floor,shift_delta_mean,shift_delta_std,power_empirical,power_gaussian,"
        "shift_delta_large_mean,power_large_empirical,power_large_gaussian"
    ]
    for scenario, scenario_data in payload["scenarios"].items():
        for size, size_data in scenario_data["by_sample_size"].items():
            for name, variant in size_data["variants"].items():
                if "alignment" in name or "signal" not in variant:
                    continue
                param = variant.get("num_slices", variant.get("num_directions"))
                rows.append(
                    f"{scenario},{size},{name},{variant['kind']},{param},"
                    f"{variant['signal']['mean']:.8g},{variant['null']['mean']:.8g},"
                    f"{variant['signal_excess']:.8g},{variant['signal_to_floor']:.8g},"
                    f"{variant['shift_delta']['mean']:.8g},{variant['shift_delta']['std']:.8g},"
                    f"{variant['detection']['power_empirical']:.4f},"
                    f"{variant['detection']['power_gaussian']:.4f},"
                    f"{variant['shift_delta_large']['mean']:.8g},"
                    f"{variant['detection_large']['power_empirical']:.4f},"
                    f"{variant['detection_large']['power_gaussian']:.4f}"
                )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_report(path: Path, payload: dict) -> None:
    config = payload["config"]
    lines = [
        "# C1 — sliced-Wasserstein estimator probe (non-training)",
        "",
        f"*{payload['created_utc']} · read-only, seeded, synthetic. "
        f"git `{payload['git_rev'][:8]}` (dirty={payload['git_dirty']}) · "
        f"device `{payload['device']}` · runtime {payload['runtime_seconds']:.1f} s.*",
        "",
        "Labels follow `CLAUDE.md` section 2: `source-verified` (read in code), "
        "`artifact-measured` (this run), `hypothesis` (mechanism), `proposal` (next action).",
        "",
        "## What this measures",
        "",
        f"For each scenario and sample size, {config['replicates']} independent replicates "
        "build independent truth/detector samples. Every estimator is scored on four statistics: "
        "`signal = est(truth, detector)`, `shift = est(truth, shifted detector)`, "
        "`null = est(truth, independent truth draw)` (the finite-sample floor), and "
        "`null_b = est(truth, independent detector draw)`. Shift detection uses the "
        "paired `shift_delta = shift - signal` against the paired placebo "
        "`null_delta = null_b - signal` (same common first sample, two independent "
        "detector draws, no shift); power is the fraction of shift deltas above the "
        "95th percentile of the placebo distribution.",
        "",
        "**Variants.** Random-slice `loss.sliced_wasserstein` at "
        f"{config['random_slices']} slices; a common-random-numbers control with "
        f"{config['fixed_random_slices']} fixed directions; a fixed orthogonal control "
        f"with {config['orthogonal_slices']} directions; learned max-SWD "
        f"(`scripts_sota/max_swd.py`) at {config['max_directions']} directions after "
        f"{config['max_steps']} adversarial steps (lr {config['max_lr']}, diversity "
        f"{config['max_diversity_weight']}). p={config['p']}. Direction ascent happens "
        "only inside this probe; `scripts/loss.py` is imported, not modified.",
        "",
        "## Scenarios",
        "",
        f"- `resonance`: {config['resonance_mass_gev']} GeV resonance, "
        f"{config['resonance_width_gev'] * 1000:.0f} MeV truth width, per-muon smearing "
        f"sigma_logpT={config['sigma_logpt']}, sigma_eta={config['sigma_eta']}, "
        f"sigma_phi={config['sigma_phi']}; shift arms "
        f"{config['mass_shift_gev'] * 1000:.0f} MeV and "
        f"{config['mass_shift_large_gev'] * 1000:.0f} MeV.",
        f"- `hard_direction`: two 8-D Gaussians separated by {config['hard_shift']} along a "
        f"random unit direction, small extra shift {config['hard_small_shift']}.",
        "",
    ]
    for scenario, scenario_data in payload["scenarios"].items():
        lines.append(f"## Results — {scenario}")
        lines.append("")
        for size, size_data in scenario_data["by_sample_size"].items():
            lines.append(f"### N = {size}")
            lines.append("")
            lines.append(
                "| variant | kind | param | signal mean | floor mean | signal/floor | "
                "signal z | shift delta (10 MeV) | power small | power large |"
            )
            lines.append("|---|---|---|---|---|---|---|---|---|---|")
            for name, variant in size_data["variants"].items():
                if "alignment" in name or "signal" not in variant:
                    continue
                param = variant.get("num_slices", variant.get("num_directions"))
                lines.append(
                    f"| {name} | {variant['kind']} | {param} | "
                    f"{variant['signal']['mean']:.6g} | {variant['null']['mean']:.6g} | "
                    f"{variant['signal_to_floor']:.3f} | {variant['signal_z']:.2f} | "
                    f"{variant['shift_delta']['mean']:.6g} ± "
                    f"{variant['shift_delta']['std']:.3g} | "
                    f"{variant['detection']['power_empirical']:.2f} | "
                    f"{variant['detection_large']['power_empirical']:.2f} |"
                )
            lines.append("")
            alignments = {
                name: value
                for name, value in size_data["variants"].items()
                if name.endswith("_alignment")
            }
            if alignments:
                for name, value in alignments.items():
                    lines.append(
                        f"- {name}: max |cos| = {value['max_abs_cosine_mean']:.3f}, "
                        f"mean |cos| = {value['mean_abs_cosine_mean']:.3f}."
                    )
                lines.append("")
    lines.append("## Findings")
    lines.append("")
    lines.extend(_findings(payload))
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- The synthetic probe is 8-D four-vector space with simplified Gaussian smearing; "
        "it is not the 14-D physics-feature space of the joint loss and it does not "
        "include the real batch-cardinality asymmetry."
    )
    lines.append(
        "- Truth and detector batches are independent draws, matching the unpaired joint "
        "loss. The signal therefore includes both the distribution difference and the "
        "finite-sample floor, which is why the raw 8-D signal-to-floor ratio is near 1."
    )
    lines.append(
        "- The learned max-SWD is the repo's averaged-over-directions variant "
        "(`scripts_sota/max_swd.py`), not a max over a single direction. Two protocols are "
        "reported: `_in` trains the directions on the pair being evaluated (what the training "
        "loss literally does; it inflates the floor), and `_heldout` trains them on a separate "
        "pair from the same distributions and evaluates on the target pair (the honest "
        "estimator comparison)."
    )
    lines.append(
        "- This is the bounded C1 configuration used after the parent stopped the full sweep: "
        f"N={config['sample_sizes']}, {config['replicates']} replicates, "
        f"{config['max_steps']} adversarial steps. The full 24-replicate / 4096-event sweep "
        "was not completed and is not reported; the 8-replicate power columns carry a binomial "
        "uncertainty of roughly ±0.17."
    )
    lines.append(
        "- Detection power uses the empirical 95th percentile of "
        f"{config['replicates']} null deltas, so it carries a binomial uncertainty of "
        "roughly ±0.1; the gaussian-threshold column is a smoother companion."
    )
    lines.append(
        "- EBSW (arXiv:2304.13586) and an explicit control-variate estimator are not "
        "implemented here; the fixed-direction arms are the cheap coverage controls. "
        "This is a `proposal` for a follow-up, not a claim about EBSW."
    )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _findings(payload: dict) -> list[str]:
    lines: list[str] = []
    for scenario, scenario_data in payload["scenarios"].items():
        for size, size_data in scenario_data["by_sample_size"].items():
            variants = {
                name: variant
                for name, variant in size_data["variants"].items()
                if "signal" in variant
            }
            if not variants:
                continue
            best_floor = min(variants.values(), key=lambda v: v["null"]["mean"])
            best_ratio = max(variants.values(), key=lambda v: v["signal_to_floor"])
            best_power = max(
                variants.values(),
                key=lambda v: v["detection_large"]["power_empirical"],
            )
            lines.append(
                f"- **{scenario}, N={size}**: lowest floor `{best_floor['name']}` "
                f"(floor {best_floor['null']['mean']:.6g}); best signal/floor "
                f"`{best_ratio['name']}` ({best_ratio['signal_to_floor']:.2f}); "
                f"best large-shift power `{best_power['name']}` "
                f"({best_power['detection_large']['power_empirical']:.2f})."
            )
    lines.append(
        "- **label: hypothesis.** Learned directions trade a higher finite-sample floor "
        "for a larger signal; the useful comparison is the ratio and the paired detection "
        "power, not the raw distance."
    )
    return lines


def make_plot(path: Path, payload: dict) -> None:
    scenarios = payload["scenarios"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {
        "random": "#1f77b4",
        "fixed_random": "#2ca02c",
        "orthogonal": "#9467bd",
        "max": "#d62728",
    }
    scenario = "resonance"
    sizes = list(scenarios[scenario]["by_sample_size"].keys())
    # Panel A: signal vs floor for N = first size.
    axis = axes[0, 0]
    variants = scenarios[scenario]["by_sample_size"][sizes[0]]["variants"]
    names = [name for name in variants if "signal" in variants[name]]
    x = np.arange(len(names))
    axis.bar(
        x - 0.2,
        [variants[name]["signal"]["mean"] for name in names],
        width=0.4,
        color=[colors[variants[name]["kind"]] for name in names],
        label="signal",
    )
    axis.bar(
        x + 0.2,
        [variants[name]["null"]["mean"] for name in names],
        width=0.4,
        color="0.6",
        label="floor",
    )
    axis.set_yscale("log")
    axis.set_xticks(x)
    axis.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axis.set_title(f"{scenario} N={sizes[0]}: signal vs finite-sample floor")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.3)
    # Panel B: signal-to-floor ratio, both sizes.
    axis = axes[0, 1]
    width = 0.35
    for index, size in enumerate(sizes):
        variants = scenarios[scenario]["by_sample_size"][size]["variants"]
        names = [name for name in variants if "signal" in variants[name]]
        axis.bar(
            np.arange(len(names)) + (index - 0.5) * width,
            [variants[name]["signal_to_floor"] for name in names],
            width=width,
            label=f"N={size}",
        )
    axis.set_xticks(np.arange(len(names)))
    axis.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axis.set_title(f"{scenario}: signal-to-floor ratio")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.3)
    # Panel C: shift detection power, resonance.
    axis = axes[1, 0]
    for index, size in enumerate(sizes):
        variants = scenarios[scenario]["by_sample_size"][size]["variants"]
        names = [name for name in variants if "signal" in variants[name]]
        axis.bar(
            np.arange(len(names)) + (index - 0.5) * width,
            [variants[name]["detection_large"]["power_empirical"] for name in names],
            width=width,
            label=f"N={size}",
        )
    axis.set_xticks(np.arange(len(names)))
    axis.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axis.set_ylim(0, 1.05)
    axis.set_title(
        f"{scenario}: detection power, "
        f"{payload['config']['mass_shift_large_gev']*1000:.0f} MeV shift"
    )
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.3)
    # Panel D: hard-direction signal vs true shift.
    axis = axes[1, 1]
    if "hard_direction" in scenarios:
        hard_sizes = list(scenarios["hard_direction"]["by_sample_size"].keys())
        size = hard_sizes[0]
        variants = scenarios["hard_direction"]["by_sample_size"][size]["variants"]
        names = [name for name in variants if "signal" in variants[name]]
        axis.bar(
            np.arange(len(names)),
            [variants[name]["signal"]["mean"] for name in names],
            color=[colors[variants[name]["kind"]] for name in names],
        )
        axis.axhline(
            payload["config"]["hard_shift"], color="black", ls="--", lw=1.5,
            label=f"true shift {payload['config']['hard_shift']}",
        )
        axis.set_xticks(np.arange(len(names)))
        axis.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        axis.set_title(f"hard_direction N={size}: recovered shift")
        axis.legend(fontsize=8)
        axis.grid(axis="y", alpha=0.3)
    fig.suptitle(
        "C1 SWD estimator probe — synthetic, non-training "
        f"({payload['config']['replicates']} replicates, p={payload['config']['p']})"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "cms_Joint" / "swd_variant_probe",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--replicates", type=int, default=DEFAULT_CONFIG["replicates"])
    parser.add_argument("--sample-sizes", default="1024,4096")
    parser.add_argument("--scenarios", default="resonance,hard_direction")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_CONFIG["max_steps"])
    parser.add_argument("--quick", action="store_true", help="small fast smoke configuration")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = dict(DEFAULT_CONFIG)
    config["seed"] = int(args.seed)
    config["replicates"] = int(args.replicates)
    config["sample_sizes"] = [int(value) for value in args.sample_sizes.split(",") if value]
    config["scenarios"] = [value.strip() for value in args.scenarios.split(",") if value.strip()]
    config["max_steps"] = int(args.max_steps)
    if args.quick:
        config["replicates"] = 3
        config["sample_sizes"] = [256]
        config["max_steps"] = 20
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"

    def log(message: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log(f"C1 SWD probe on {device}; output {output_dir}")
    payload = run_probe(config, device=device, log=log)
    payload["command"] = " ".join(sys.argv)
    json_path = output_dir / "swd_variant_probe.json"
    json_path.write_text(json.dumps(json_ready(payload), indent=2) + "\n", encoding="utf-8")
    write_csv(output_dir / "swd_variant_probe.csv", payload)
    write_report(output_dir / "REPORT.md", payload)
    make_plot(output_dir / "swd_variant_probe.png", payload)
    log(f"wrote {json_path}")
    log(f"wrote {output_dir / 'swd_variant_probe.csv'}")
    log(f"wrote {output_dir / 'REPORT.md'}")
    log(f"wrote {output_dir / 'swd_variant_probe.png'}")
    log(f"done in {payload['runtime_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
