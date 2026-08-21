"""Mass-aware checkpoint selection gates (Tier A, item 5).

The current pipeline selects checkpoints from a weighted sum of standardized
loss components.  Those components are dominated by broad coordinate scales,
so a checkpoint with a visibly broken J/psi mass can still be selected (this
was measured for the archived v3.5 run).  These gates make the mass shape a
*hard* selection criterion: if any configured threshold is violated the
validation score becomes ``inf`` and the checkpoint cannot become best.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from physics import invariant_mass_np, validate_daughter_masses

try:
    from scipy.stats import ks_2samp, wasserstein_distance

    HAS_SCIPY = True
except Exception:  # pragma: no cover - fallback used only on minimal installs
    HAS_SCIPY = False


@dataclass(frozen=True)
class GateConfig:
    enabled: bool = True
    max_sim_mass_w1_gev: float = 0.015
    max_sim_mass_ks: float = 0.12
    min_sim_window_fraction: float = 0.90
    max_latent_mass_ks: float = 0.15
    max_latent_mass_w1_gev: float | None = None
    max_latent_mass_width_rel_error: float | None = None
    max_reco_mass_ks: float = 0.25
    max_reco_mass_w1_gev: float | None = None
    max_sim_pair_pt_ks: float | None = None
    max_abs_sim_mass_mean_shift_gev: float = 0.005
    mass_window: tuple[float, float] = (3.0369, 3.1569)
    max_validation_events: int = 16384
    seed: int = 20260816

    @staticmethod
    def from_config(config: dict) -> "GateConfig":
        gates = config.get("sota_selection_gates", {})
        return GateConfig(**gates)


def _w1_quantile(a: np.ndarray, b: np.ndarray, n_quantiles: int = 4097) -> float:
    if HAS_SCIPY:
        return float(wasserstein_distance(a, b))
    q = np.linspace(0.0, 1.0, int(n_quantiles))
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def _ks(a: np.ndarray, b: np.ndarray) -> float:
    if HAS_SCIPY:
        return float(ks_2samp(a, b).statistic)
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    cdf_a = np.searchsorted(a_sorted, np.concatenate([a_sorted, b_sorted]), side="right") / len(a)
    cdf_b = np.searchsorted(b_sorted, np.concatenate([a_sorted, b_sorted]), side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _pair_pt(values: np.ndarray) -> np.ndarray:
    return np.hypot(values[:, 0] + values[:, 4], values[:, 1] + values[:, 5])


@torch.inference_mode()
def _transform_in_chunks(function, values: np.ndarray, device: torch.device, batch_size: int = 8192) -> np.ndarray:
    outputs = []
    for start in range(0, len(values), batch_size):
        batch = torch.as_tensor(values[start : start + batch_size], dtype=torch.float32, device=device)
        outputs.append(function(batch).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def evaluate_mass_gates(
    model,
    x_values: np.ndarray,
    z_values: np.ndarray,
    *,
    daughter_masses,
    gate_config: GateConfig,
    device: torch.device,
    batch_size: int = 8192,
    decoder_draws: int = 1,
) -> tuple[dict, bool]:
    """Evaluate the hard mass gates on fixed validation arrays.

    Returns ``(report, passed)``.  ``decoder_draws > 1`` averages the mass
    histogram over independent decoder draws before computing W1/KS, but the
    current default is one draw for cheap stage-boundary decisions.
    """
    masses = validate_daughter_masses(daughter_masses)
    rng = np.random.default_rng(gate_config.seed)
    max_events = int(gate_config.max_validation_events)
    x_idx = rng.choice(len(x_values), size=min(max_events, len(x_values)), replace=False)
    z_idx = rng.choice(len(z_values), size=min(max_events, len(z_values)), replace=False)
    x_fixed = np.asarray(x_values[x_idx], dtype=np.float32)
    z_fixed = np.asarray(z_values[z_idx], dtype=np.float32)
    model.eval()

    z_encoded = _transform_in_chunks(model.encode, x_fixed, device, batch_size)
    x_reco = _transform_in_chunks(model.decode, z_encoded, device, batch_size)
    sim_parts = [
        _transform_in_chunks(model.decode, z_fixed, device, batch_size)
        for _ in range(max(1, int(decoder_draws)))
    ]

    m_x = invariant_mass_np(x_fixed, daughter_masses=masses, stable=True)
    # Detector x-space is an on-shell reconstructed state, while the MG5
    # z-space contract treats stored energies as authoritative.  This matches
    # plot_run_e_paperstyle.py and prevents validation from selecting on a
    # different mass definition than the final test report.
    m_z = invariant_mass_np(z_fixed, daughter_masses=masses, stable=False)
    m_enc = invariant_mass_np(z_encoded, daughter_masses=masses, stable=False)
    m_reco = invariant_mass_np(x_reco, daughter_masses=masses, stable=True)
    m_sim = np.concatenate(
        [invariant_mass_np(part, daughter_masses=masses, stable=True) for part in sim_parts]
    )

    lo, hi = gate_config.mass_window
    report = {
        "sim_mass_w1_gev": _w1_quantile(m_x, m_sim),
        "sim_mass_ks": _ks(m_x, m_sim),
        "sim_window_fraction": float(np.mean((m_sim >= lo) & (m_sim <= hi))),
        "sim_mean_shift_gev": float(m_sim.mean() - m_x.mean()),
        "latent_mass_w1_gev": _w1_quantile(m_z, m_enc),
        "latent_mass_ks": _ks(m_z, m_enc),
        "latent_window_fraction": float(np.mean((m_enc >= lo) & (m_enc <= hi))),
        "latent_mass_width_rel_error": float(
            abs(m_enc.std() - m_z.std()) / max(float(m_z.std()), 1e-8)
        ),
        "reco_mass_w1_gev": _w1_quantile(m_x, m_reco),
        "reco_mass_ks": _ks(m_x, m_reco),
        "sim_pair_pt_ks": _ks(_pair_pt(x_fixed), _pair_pt(np.concatenate(sim_parts))),
        "decoder_draws": int(decoder_draws),
        "x_events": int(len(m_x)),
        "z_events": int(len(m_z)),
        "sim_events": int(len(m_sim)),
    }

    passed = bool(
        (not gate_config.enabled)
        or (
            report["sim_mass_w1_gev"] <= gate_config.max_sim_mass_w1_gev
            and report["sim_mass_ks"] <= gate_config.max_sim_mass_ks
            and report["sim_window_fraction"] >= gate_config.min_sim_window_fraction
            and report["latent_mass_ks"] <= gate_config.max_latent_mass_ks
            and report["reco_mass_ks"] <= gate_config.max_reco_mass_ks
            and abs(report["sim_mean_shift_gev"]) <= gate_config.max_abs_sim_mass_mean_shift_gev
            and (
                gate_config.max_latent_mass_w1_gev is None
                or report["latent_mass_w1_gev"] <= gate_config.max_latent_mass_w1_gev
            )
            and (
                gate_config.max_latent_mass_width_rel_error is None
                or report["latent_mass_width_rel_error"]
                <= gate_config.max_latent_mass_width_rel_error
            )
            and (
                gate_config.max_reco_mass_w1_gev is None
                or report["reco_mass_w1_gev"] <= gate_config.max_reco_mass_w1_gev
            )
            and (
                gate_config.max_sim_pair_pt_ks is None
                or report["sim_pair_pt_ks"] <= gate_config.max_sim_pair_pt_ks
            )
        )
    )
    report["gate_passed"] = passed
    return report, passed


def gated_score(base_score: float, gate_report: dict | None, passed: bool) -> float:
    """Return ``inf`` when a checkpoint fails the mass gates."""
    if gate_report is None or bool(passed):
        return float(base_score)
    return math.inf


def checkpoint_selection_score(
    base_score: float,
    gate_report: dict | None,
    passed: bool,
    config: dict,
) -> float:
    """Select by the worst normalized direction when Run G requests it.

    The legacy Run E/F behavior remains the default.  ``worst_direction``
    prevents a very good decoder score from compensating for a weak encoder
    (or vice versa).  Each configured target is a positive denominator and
    the best checkpoint minimizes the maximum normalized metric.
    """
    gated = gated_score(base_score, gate_report, passed)
    if not math.isfinite(gated) or gate_report is None:
        return gated
    selection = config.get("checkpoint_selection", {})
    if selection.get("mode", "loss_sum") != "worst_direction":
        return gated
    targets = config.get("sota_selection_targets", {})
    ratios = []
    for metric, target in targets.items():
        if metric not in gate_report:
            raise KeyError(f"Unknown/missing checkpoint target metric: {metric}")
        target = float(target)
        if not math.isfinite(target) or target <= 0.0:
            raise ValueError(f"Checkpoint target {metric} must be positive and finite")
        ratios.append(abs(float(gate_report[metric])) / target)
    if not ratios:
        raise ValueError("worst_direction checkpoint selection needs sota_selection_targets")
    base_weight = float(selection.get("base_loss_weight", 0.0))
    return float(max(ratios) + base_weight * gated)


def write_gate_report(report: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
