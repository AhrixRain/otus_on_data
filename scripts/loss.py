from __future__ import annotations

import difflib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from physics import (
    invariant_mass_np,
    invariant_mass_torch,
    validate_daughter_masses,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
UTILITY_DIR = REPO_ROOT / "utilityFunctions"
if str(UTILITY_DIR) not in sys.path:
    sys.path.insert(0, str(UTILITY_DIR))

from func_utils import anchor_loss  # noqa: E402


P = 2
CANONICAL_LOSS_KIND = "cms_doubleelectron_loss"
JPSI_DIMUON_LOSS_KIND = "cms_jpsi_doublemuon_loss"

# Default per-component weight values. Kept at module level so strict
# configuration validation and the factory share one source of truth.
DEFAULT_SPACE_WEIGHTS = {
    "raw_swd": 0.5,
    "marginal_w1": 0.5,
    "mass_w1": 2.0,
    "resonance_mass_w1": 0.0,
    "physics_swd": 1.0,
    "mass_kin_swd": 0.0,
    "transverse_w1": 0.5,
    "longitudinal_w1": 0.4,
    "tail_w1": 0.15,
    "pair_mass_w1": 2.0,
    "pair_pt_w1": 2.0,
    "lepton_pt_w1": 1.0,
    "delta_phi_w1": 0.5,
    "delta_eta_w1": 0.5,
    "pair_rapidity_w1": 0.5,
    "physics_coord_swd": 0.5,
    "mmd": 0.0,
}

MASS_KIN_SWD_COMPONENT_KEYS = ("mll", "ptll", "yll", "cos_dphi", "sin_dphi")
SELECTION_SCORE_KEYS = ("x_sim", "z_prior", "x_reco", "cycle")

# Non-weight settings accepted at the top level of a loss config (and inside
# per-space overrides, which are merged into the space config unchanged).
KNOWN_LOSS_SETTINGS = frozenset(
    {
        "kind",
        "vanilla_v3_7",
        "vanilla_swae",
        "eps",
        "p",
        "num_slices",
        "tail_frac",
        "resonance_mass_center",
        "resonance_mass_half_width",
        "mmd_scales",
        "standardize_raw_matching",
        "decoder_num_noise_samples",
        "x_reco_physics_w1",
        "mass_kin_swd_components",
        "space_weights",
        "selection_score",
    }
)
KNOWN_LOSS_KEYS = frozenset(DEFAULT_SPACE_WEIGHTS) | KNOWN_LOSS_SETTINGS


def _format_unknown_keys(unknown: set[str], context: str) -> str:
    known = sorted(KNOWN_LOSS_KEYS)
    lines = []
    for key in sorted(unknown):
        nearest = difflib.get_close_matches(key, known, n=1, cutoff=0.4)
        hint = f" (nearest known key: {nearest[0]})" if nearest else ""
        lines.append(f"  - {key}{hint}")
    return (
        f"Unknown {context} key(s):\n"
        + "\n".join(lines)
        + f"\nValid {context} keys: {', '.join(known)}"
    )


def _require_finite_weight(name: str, value: Any) -> None:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"loss weight {name!r} must be finite, got {value!r}")


def validate_loss_config(loss_config: dict[str, Any]) -> None:
    """Strictly validate a loss config before constructing a loss factory.

    Rejects unknown keys (with a nearest-match hint) and obvious invalid
    values, so a typo like ``pair_masss_w1`` fails fast instead of silently
    disabling a term. Legacy ZLossFactory-style configs (kinds other than the
    canonical/Jpsi dilepton kinds) are intentionally skipped: their key space
    predates this validation and none of the checked-in configs use it.
    """
    if not isinstance(loss_config, dict):
        raise ValueError(f"loss config must be a mapping, got {type(loss_config).__name__}")
    kind = loss_config.get("kind")
    if kind is not None and kind not in {
        None,
        CANONICAL_LOSS_KIND,
        JPSI_DIMUON_LOSS_KIND,
        "original_feature_ot_v1",
    }:
        return

    unknown = set(loss_config) - KNOWN_LOSS_KEYS
    if unknown:
        raise ValueError(_format_unknown_keys(unknown, "loss"))

    for key in DEFAULT_SPACE_WEIGHTS:
        if key in loss_config:
            _require_finite_weight(key, loss_config[key])
    if "x_reco_physics_w1" in loss_config:
        _require_finite_weight("x_reco_physics_w1", loss_config["x_reco_physics_w1"])

    space_weights = loss_config.get("space_weights") or {}
    if not isinstance(space_weights, dict):
        raise ValueError("loss.space_weights must be a mapping")
    for space in ("x", "z"):
        if space not in space_weights:
            continue
        override = space_weights[space]
        if not isinstance(override, dict):
            raise ValueError(f"loss.space_weights.{space} must be a mapping")
        unknown = set(override) - KNOWN_LOSS_KEYS
        if unknown:
            raise ValueError(_format_unknown_keys(unknown, f"loss.space_weights.{space}"))
        for key in DEFAULT_SPACE_WEIGHTS:
            if key in override:
                _require_finite_weight(f"space_weights.{space}.{key}", override[key])

    mass_kin = loss_config.get("mass_kin_swd_components") or {}
    if not isinstance(mass_kin, dict):
        raise ValueError("loss.mass_kin_swd_components must be a mapping")
    unknown = set(mass_kin) - set(MASS_KIN_SWD_COMPONENT_KEYS)
    if unknown:
        raise ValueError(_format_unknown_keys(unknown, "loss.mass_kin_swd_components"))
    for key in MASS_KIN_SWD_COMPONENT_KEYS:
        if key in mass_kin:
            value = float(mass_kin[key])
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"loss.mass_kin_swd_components.{key} must be finite and non-negative, "
                    f"got {mass_kin[key]!r}"
                )

    selection_score = loss_config.get("selection_score") or {}
    if not isinstance(selection_score, dict):
        raise ValueError("loss.selection_score must be a mapping")
    unknown = set(selection_score) - set(SELECTION_SCORE_KEYS)
    if unknown:
        raise ValueError(_format_unknown_keys(unknown, "loss.selection_score"))
    for key in SELECTION_SCORE_KEYS:
        if key in selection_score:
            _require_finite_weight(f"selection_score.{key}", selection_score[key])

    for key in ("num_slices", "decoder_num_noise_samples"):
        if key in loss_config:
            value = loss_config[key]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"loss.{key} must be a positive integer, got {value!r}")
    if "eps" in loss_config:
        value = float(loss_config["eps"])
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"loss.eps must be positive and finite, got {loss_config['eps']!r}")
    if "tail_frac" in loss_config:
        value = float(loss_config["tail_frac"])
        if not np.isfinite(value) or not 0.0 < value < 1.0:
            raise ValueError(f"loss.tail_frac must lie in (0, 1), got {loss_config['tail_frac']!r}")
    if "resonance_mass_half_width" in loss_config:
        value = float(loss_config["resonance_mass_half_width"])
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"loss.resonance_mass_half_width must be positive, got "
                f"{loss_config['resonance_mass_half_width']!r}"
            )


def build_ee_physics_features(
    x: torch.Tensor,
    eps: float = 1e-6,
    daughter_masses=None,
    mass_from_energy: bool = False,
) -> dict[str, torch.Tensor]:
    """Differentiable CMS dilepton observables for [l- p4, l+ p4].

    ``mass_from_energy`` selects the pair-mass definition. When True the
    stored energy columns are authoritative and the direct
    ``sqrt(E^2 - p^2)`` expression is used; this is correct for the theory
    z-prior, whose energy columns carry truth-level information that is not
    derivable from the momenta alone (its per-muon energies are not exactly
    on shell). When False the cancellation-free transverse-mass/rapidity
    decomposition from ``physics.invariant_mass_torch`` is used, which is the
    float32-safe choice for detector-level x four-vectors whose energies are
    derived from p and the daughter masses.
    """
    if x.ndim != 2 or x.shape[1] != 8:
        raise ValueError(f"Expected tensor with shape [N, 8], got {tuple(x.shape)}.")

    px_m, py_m, pz_m, e_m = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    px_p, py_p, pz_p, e_p = x[:, 4], x[:, 5], x[:, 6], x[:, 7]

    pt_m = torch.sqrt(torch.clamp(px_m**2 + py_m**2, min=eps))
    pt_p = torch.sqrt(torch.clamp(px_p**2 + py_p**2, min=eps))
    eta_m = torch.asinh(pz_m / torch.clamp(pt_m, min=eps))
    eta_p = torch.asinh(pz_p / torch.clamp(pt_p, min=eps))
    phi_m = torch.atan2(py_m, px_m)
    phi_p = torch.atan2(py_p, px_p)
    delta_phi = torch.atan2(torch.sin(phi_m - phi_p), torch.cos(phi_m - phi_p))
    delta_eta = eta_m - eta_p

    pair_px = px_m + px_p
    pair_py = py_m + py_p
    pair_pz = pz_m + pz_p
    pair_e = e_m + e_p
    if mass_from_energy:
        pair_mass2 = pair_e**2 - pair_px**2 - pair_py**2 - pair_pz**2
        pair_mass = torch.sqrt(torch.clamp(pair_mass2, min=eps))
    else:
        pair_mass = invariant_mass_torch(x, daughter_masses=daughter_masses, eps=eps)
    pair_pt = torch.sqrt(torch.clamp(pair_px**2 + pair_py**2, min=eps))
    rapidity_ratio = torch.clamp(
        torch.clamp(pair_e + pair_pz, min=eps) / torch.clamp(pair_e - pair_pz, min=eps),
        min=eps,
        max=1.0 / eps,
    )
    pair_rapidity = 0.5 * torch.log(rapidity_ratio)

    coord_m = torch.stack(
        [
            torch.log(pt_m + eps),
            eta_m,
            py_m / (pt_m + eps),
            px_m / (pt_m + eps),
            torch.log(torch.clamp(e_m, min=eps)),
        ],
        dim=1,
    )
    coord_p = torch.stack(
        [
            torch.log(pt_p + eps),
            eta_p,
            py_p / (pt_p + eps),
            px_p / (pt_p + eps),
            torch.log(torch.clamp(e_p, min=eps)),
        ],
        dim=1,
    )
    physics_coord_features = torch.cat([coord_m, coord_p], dim=1)

    physics_features = torch.stack(
        [
            pt_m,
            pt_p,
            eta_m,
            eta_p,
            pair_mass,
            pair_pt,
            pair_rapidity,
            torch.cos(delta_phi),
            torch.sin(delta_phi),
            pair_px,
            pair_py,
            pair_pz,
            px_m - px_p,
            py_m - py_p,
            pz_m - pz_p,
        ],
        dim=1,
    )
    pair_features = torch.stack(
        [pair_mass, pair_pt, pt_m, pt_p, delta_phi, delta_eta, pair_rapidity],
        dim=1,
    )
    return {
        "m_ee": pair_mass,
        "pt_ee": pair_pt,
        "pair_rapidity": pair_rapidity,
        "e_minus_pt": pt_m,
        "e_plus_pt": pt_p,
        "delta_phi": delta_phi,
        "delta_eta": delta_eta,
        "physics_features": physics_features,
        "pair_features": pair_features,
        "physics_coord_features": physics_coord_features,
    }


def sliced_wasserstein(
    truth: torch.Tensor,
    pred: torch.Tensor,
    num_slices: int,
    p: int = 2,
    eps: float = 1e-12,
) -> torch.Tensor:
    if truth.ndim != 2 or pred.ndim != 2:
        raise ValueError("sliced_wasserstein expects rank-2 tensors.")
    if truth.shape[1] != pred.shape[1]:
        raise ValueError("sliced_wasserstein inputs must have the same feature dimension.")
    n = min(truth.shape[0], pred.shape[0])
    if n == 0:
        return truth.new_tensor(0.0)
    theta = torch.randn(
        int(num_slices),
        truth.shape[1],
        dtype=truth.dtype,
        device=truth.device,
    )
    theta = theta / torch.clamp(torch.linalg.norm(theta, dim=1, keepdim=True), min=eps)
    truth_proj = truth[:n].matmul(theta.t())
    pred_proj = pred[:n].matmul(theta.t())
    truth_sorted = torch.sort(truth_proj, dim=0)[0]
    pred_sorted = torch.sort(pred_proj, dim=0)[0]
    diff = pred_sorted - truth_sorted
    if int(p) == 1:
        return torch.mean(torch.abs(diff))
    if int(p) == 2:
        return torch.mean(diff**2)
    raise ValueError("Only p=1 and p=2 are supported.")


def _safe_numpy_std(values: np.ndarray) -> np.ndarray:
    std = np.std(values, axis=0)
    return np.where((std > 0.0) & np.isfinite(std), std, 1.0)


def _safe_torch_std(values: torch.Tensor, eps: float) -> torch.Tensor:
    std = values.std(dim=0, unbiased=False)
    return torch.where(
        torch.isfinite(std) & (std > eps),
        std,
        torch.ones_like(std),
    )


def _empirical_quantile(sorted_values: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    """Linear-interpolation empirical quantile function (differentiable).

    ``sorted_values`` is an ascending 1-D tensor of ``n`` samples and
    ``quantiles`` a 1-D tensor of probabilities in [0, 1]. Uses the
    ``(n - 1) * q`` interpolation convention shared with NumPy/scipy, so the
    result matches those offline reference implementations while remaining
    differentiable with respect to ``sorted_values``.
    """
    n = sorted_values.numel()
    positions = quantiles * (n - 1)
    lower = torch.floor(positions).to(torch.long)
    upper = torch.clamp(lower + 1, max=n - 1)
    fraction = positions - lower.to(sorted_values.dtype)
    return (
        sorted_values[lower]
        + fraction * (sorted_values[upper] - sorted_values[lower])
    )


class SpaceFeatureOTLoss:
    """Continuous feature OT loss for one four-vector space."""

    def __init__(
        self,
        train_samples: np.ndarray,
        loss_config: dict[str, Any],
        *,
        name: str,
        eps: float = 1e-6,
        daughter_masses=None,
        mass_from_energy: bool = False,
    ):
        self.name = name
        self.eps = float(loss_config.get("eps", eps))
        self.num_slices = int(loss_config.get("num_slices", 1000))
        self.p = int(loss_config.get("p", P))
        self.mass_from_energy = bool(mass_from_energy)
        self.weights = dict(DEFAULT_SPACE_WEIGHTS)
        self.weights.update(
            {
                key: float(loss_config[key])
                for key in self.weights
                if key in loss_config
            }
        )
        self.daughter_masses = validate_daughter_masses(daughter_masses)
        self.tail_frac = float(loss_config.get("tail_frac", 0.20))
        self.resonance_mass_center = float(
            loss_config.get("resonance_mass_center", 3.0969)
        )
        self.resonance_mass_half_width = float(
            loss_config.get("resonance_mass_half_width", 0.06)
        )
        # mass_kin_swd is a joint sliced-Wasserstein over the pair-level physics
        # bundle [m_ll, pT_ll, y_ll, cos(dphi), sin(dphi)] (physics-feature
        # columns 4..8). The invariant-mass column can be disabled independently
        # via `mass_kin_swd_components: {mll: 0.0}` so a controlled ablation can
        # keep the four non-mass pair-kinematics observables while removing
        # explicit mass supervision. Defaults to all components active, which is
        # byte-for-byte the pre-refactor behavior (SWD over columns 4:9).
        self.mass_kin_swd_columns = {
            "mll": 4,
            "ptll": 5,
            "yll": 6,
            "cos_dphi": 7,
            "sin_dphi": 8,
        }
        mass_kin_component_config = loss_config.get("mass_kin_swd_components", {})
        self.mass_kin_swd_component_weights = {
            name: float(mass_kin_component_config.get(name, 1.0))
            for name in self.mass_kin_swd_columns
        }
        self.mmd_scales = [
            float(value) for value in loss_config.get("mmd_scales", [0.5, 1.0, 2.0, 4.0])
        ]
        # Raw 8-vector SWD/marginal/tail terms are computed in standardized units
        # by default. Set `standardize_raw_matching: false` for a raw-GeV control.
        self.standardize_raw_matching = bool(
            loss_config.get("standardize_raw_matching", True)
        )

        self.raw_mean = np.mean(train_samples, axis=0)
        self.raw_std = _safe_numpy_std(train_samples)

        # Invariant-mass statistics are computed in float64. Detector-level x
        # spaces use the stable formula so boosted float32 pairs cannot distort
        # the standardization scale; theory z-spaces use the stored energy
        # columns, which carry truth-level information.
        mass_train = invariant_mass_np(
            train_samples,
            daughter_masses=None if self.mass_from_energy else self.daughter_masses,
            stable=not self.mass_from_energy,
        )
        self.mass_mean = torch.as_tensor(
            float(np.mean(mass_train, dtype=np.float64)),
            dtype=torch.float32,
        )
        mass_std_value = float(np.std(mass_train, dtype=np.float64))
        if not np.isfinite(mass_std_value) or mass_std_value <= self.eps:
            mass_std_value = 1.0
        self.mass_std = torch.as_tensor(mass_std_value, dtype=torch.float32)

        # Empirical training-sample standard deviation of delta_eta itself,
        # used to normalize the delta_eta W1 term (replaces the ad-hoc
        # std(eta1) + std(eta2) approximation).
        work = np.asarray(train_samples, dtype=np.float64)
        pt1 = np.hypot(work[:, 0], work[:, 1])
        pt2 = np.hypot(work[:, 4], work[:, 5])
        with np.errstate(divide="ignore", invalid="ignore"):
            eta1 = np.arcsinh(work[:, 2] / np.where(pt1 > 0.0, pt1, 1.0))
            eta2 = np.arcsinh(work[:, 6] / np.where(pt2 > 0.0, pt2, 1.0))
        delta_eta = eta1 - eta2
        delta_eta_std = float(np.std(delta_eta, dtype=np.float64))
        if not np.isfinite(delta_eta_std) or delta_eta_std <= self.eps:
            delta_eta_std = 1.0
        self.delta_eta_std = delta_eta_std

        with torch.no_grad():
            train = torch.as_tensor(train_samples, dtype=torch.float32)
            features = self.physics_features(train)
            transverse = self.transverse_features(train)
            longitudinal = self.longitudinal_features(train)
            physics_coord = self.physics_coord_features(train)
            self.feature_mean = features.mean(dim=0).detach()
            self.feature_std = _safe_torch_std(features, self.eps).detach()
            self.transverse_mean = transverse.mean(dim=0).detach()
            self.transverse_std = _safe_torch_std(transverse, self.eps).detach()
            self.longitudinal_mean = longitudinal.mean(dim=0).detach()
            self.longitudinal_std = _safe_torch_std(longitudinal, self.eps).detach()
            self.physics_coord_mean = physics_coord.mean(dim=0).detach()
            self.physics_coord_std = _safe_torch_std(physics_coord, self.eps).detach()

    def set_num_slices(self, num_slices: int) -> None:
        self.num_slices = int(num_slices)

    def to_like(self, value: Any, ref: torch.Tensor) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.detach().to(dtype=ref.dtype, device=ref.device)
        return torch.as_tensor(value, dtype=ref.dtype, device=ref.device)

    def standardize_raw(self, values: torch.Tensor) -> torch.Tensor:
        mean = self.to_like(self.raw_mean, values)
        std = self.to_like(self.raw_std, values)
        return (values - mean) / (std + self.eps)

    def standardize_features(
        self,
        features: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        return (features - self.to_like(mean, features)) / (
            self.to_like(std, features) + self.eps
        )

    def standardize_mass(self, mass: torch.Tensor) -> torch.Tensor:
        return (mass - self.to_like(self.mass_mean, mass)) / (
            self.to_like(self.mass_std, mass) + self.eps
        )

    def paired_mse_standardized(self, truth: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        return torch.mean((self.standardize_raw(truth) - self.standardize_raw(pred)) ** 2)

    def invariant_mass(self, values: torch.Tensor) -> torch.Tensor:
        return build_ee_physics_features(
            values,
            self.eps,
            self.daughter_masses,
            mass_from_energy=self.mass_from_energy,
        )["m_ee"]

    def safe_eta(self, pt: torch.Tensor, pz: torch.Tensor) -> torch.Tensor:
        return torch.asinh(pz / torch.clamp(pt, min=self.eps))

    def safe_rapidity(self, energy: torch.Tensor, pz: torch.Tensor) -> torch.Tensor:
        numerator = torch.clamp(energy + pz, min=self.eps)
        denominator = torch.clamp(energy - pz, min=self.eps)
        ratio = torch.clamp(numerator / denominator, min=self.eps, max=1.0 / self.eps)
        return 0.5 * torch.log(ratio)

    def physics_features(self, values: torch.Tensor) -> torch.Tensor:
        return build_ee_physics_features(
            values,
            self.eps,
            self.daughter_masses,
            mass_from_energy=self.mass_from_energy,
        )["physics_features"]

    def physics_coord_features(self, values: torch.Tensor) -> torch.Tensor:
        return build_ee_physics_features(
            values,
            self.eps,
            self.daughter_masses,
            mass_from_energy=self.mass_from_energy,
        )["physics_coord_features"]

    def transverse_features(self, values: torch.Tensor) -> torch.Tensor:
        px1, py1 = values[:, 0], values[:, 1]
        px2, py2 = values[:, 4], values[:, 5]
        pt1 = torch.sqrt(torch.clamp(px1**2 + py1**2, min=self.eps))
        pt2 = torch.sqrt(torch.clamp(px2**2 + py2**2, min=self.eps))
        pair_px = px1 + px2
        pair_py = py1 + py2
        pair_pt = torch.sqrt(torch.clamp(pair_px**2 + pair_py**2, min=self.eps))
        dot = px1 * px2 + py1 * py2
        cross = px1 * py2 - py1 * px2
        norm = torch.clamp(pt1 * pt2, min=self.eps)
        return torch.stack(
            [
                px1,
                py1,
                px2,
                py2,
                pt1,
                pt2,
                pair_px,
                pair_py,
                pair_pt,
                px1 - px2,
                py1 - py2,
                torch.clamp(dot / norm, min=-1.0, max=1.0),
                torch.clamp(cross / norm, min=-1.0, max=1.0),
            ],
            dim=1,
        )

    def longitudinal_features(self, values: torch.Tensor) -> torch.Tensor:
        px1, py1, pz1, e1 = values[:, 0], values[:, 1], values[:, 2], values[:, 3]
        px2, py2, pz2, e2 = values[:, 4], values[:, 5], values[:, 6], values[:, 7]
        pt1 = torch.sqrt(torch.clamp(px1**2 + py1**2, min=self.eps))
        pt2 = torch.sqrt(torch.clamp(px2**2 + py2**2, min=self.eps))
        pair_pz = pz1 + pz2
        pair_energy = e1 + e2
        return torch.stack(
            [
                pz1,
                pz2,
                e1,
                e2,
                pair_pz,
                pz1 - pz2,
                self.safe_eta(pt1, pz1),
                self.safe_eta(pt2, pz2),
                self.safe_rapidity(pair_energy, pair_pz),
            ],
            dim=1,
        )

    def wasserstein_1d_sorted(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_sorted = torch.sort(a.reshape(-1))[0]
        b_sorted = torch.sort(b.reshape(-1))[0]
        n = min(a_sorted.numel(), b_sorted.numel())
        return torch.mean(torch.abs(a_sorted[:n] - b_sorted[:n]))

    def _marginal_w1_std(
        self,
        truth_std: torch.Tensor,
        pred_std: torch.Tensor,
    ) -> torch.Tensor:
        """Batched 1D Wasserstein-1 over all columns with one sort per array.

        Sorting every column at once and averaging over all N x D entries is
        mathematically identical to sorting each column and averaging per
        column, but replaces D separate torch.sort calls with one.
        """
        truth_sorted = torch.sort(truth_std, dim=0)[0]
        pred_sorted = torch.sort(pred_std, dim=0)[0]
        n = min(truth_sorted.shape[0], pred_sorted.shape[0])
        return torch.mean(torch.abs(truth_sorted[:n] - pred_sorted[:n]))

    def marginal_w1(self, truth: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        if self.standardize_raw_matching:
            truth_std = self.standardize_raw(truth)
            pred_std = self.standardize_raw(pred)
        else:
            truth_std = truth
            pred_std = pred
        return self._marginal_w1_std(truth_std, pred_std)

    def feature_w1(
        self,
        truth_features: torch.Tensor,
        pred_features: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        truth_std = self.standardize_features(truth_features, mean, std)
        pred_std = self.standardize_features(pred_features, mean, std)
        return self._marginal_w1_std(truth_std, pred_std)

    def standardized_physics_column(self, values: torch.Tensor, column: int) -> torch.Tensor:
        features = self.physics_features(values)
        mean = self.to_like(self.feature_mean[column], features)
        std = self.to_like(self.feature_std[column], features)
        return (features[:, column] - mean) / (std + self.eps)

    def tail_wasserstein_abs(self, truth: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        truth_sorted = torch.sort(torch.abs(truth.reshape(-1)))[0]
        pred_sorted = torch.sort(torch.abs(pred.reshape(-1)))[0]
        n = min(truth_sorted.numel(), pred_sorted.numel())
        if n == 0:
            return truth.new_tensor(0.0)
        start = max(0, min(int((1.0 - self.tail_frac) * n), n - 1))
        return torch.mean(torch.abs(truth_sorted[start:n] - pred_sorted[start:n]))

    def _tail_w1_std(
        self,
        truth_std: torch.Tensor,
        pred_std: torch.Tensor,
        dims: tuple[int, ...] = (0, 1, 2, 4, 5, 6),
    ) -> torch.Tensor:
        """Batched tail Wasserstein-1 over the selected columns."""
        truth_sorted = torch.sort(torch.abs(truth_std[:, dims]), dim=0)[0]
        pred_sorted = torch.sort(torch.abs(pred_std[:, dims]), dim=0)[0]
        n = min(truth_sorted.shape[0], pred_sorted.shape[0])
        if n == 0:
            return truth_std.new_tensor(0.0)
        start = max(0, min(int((1.0 - self.tail_frac) * n), n - 1))
        return torch.mean(torch.abs(truth_sorted[start:n] - pred_sorted[start:n]))

    def tail_w1(self, truth: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        if self.standardize_raw_matching:
            truth_std = self.standardize_raw(truth)
            pred_std = self.standardize_raw(pred)
        else:
            truth_std = truth
            pred_std = pred
        return self._tail_w1_std(truth_std, pred_std)

    def resonance_mass_w1(self, truth_mass: torch.Tensor, pred_mass: torch.Tensor) -> torch.Tensor:
        """W1 on the physical invariant mass inside the resonance window.

        The MG5 z-prior mass has sigma ~10 MeV, so a standardized global mass W1
        carries a gradient scale ~100x larger than the other terms and can
        dominate training. This windowed term operates in physical GeV so its
        gradient scale is comparable to the kinematics terms, and it targets
        only the resonance core: it fixes peak position/width without letting
        the mass marginal dominate the full 8-vector joint.

        When the truth and prediction windows contain the same number of
        events the original sorted-pair W1 is returned unchanged. When the
        counts differ (e.g. the model has not yet populated the window), the
        two empirical quantile functions are compared on a shared quantile
        grid, which is the correct empirical W1 for unequal sample counts
        instead of silently truncating to the smaller cardinality.
        """
        center = self.to_like(self.resonance_mass_center, truth_mass)
        half_width = self.to_like(self.resonance_mass_half_width, truth_mass)
        truth_window = (truth_mass >= center - half_width) & (truth_mass <= center + half_width)
        pred_window = (pred_mass >= center - half_width) & (pred_mass <= center + half_width)
        truth_selected = truth_mass[truth_window]
        pred_selected = pred_mass[pred_window]
        if truth_selected.numel() == 0 or pred_selected.numel() == 0:
            # Documented remaining limitation: an empty window returns zero
            # (no gradient) rather than introducing a new occupancy objective.
            return truth_mass.new_tensor(0.0)
        if truth_selected.numel() == pred_selected.numel():
            return self.wasserstein_1d_sorted(truth_selected, pred_selected)
        truth_sorted = torch.sort(truth_selected)[0]
        pred_sorted = torch.sort(pred_selected)[0]
        grid_size = max(truth_sorted.numel(), pred_sorted.numel())
        quantiles = (
            torch.arange(grid_size, dtype=truth_mass.dtype, device=truth_mass.device)
            + 0.5
        ) / grid_size
        truth_quantiles = _empirical_quantile(truth_sorted, quantiles)
        pred_quantiles = _empirical_quantile(pred_sorted, quantiles)
        return torch.mean(torch.abs(truth_quantiles - pred_quantiles))

    def paired_physics_mse_standardized(
        self,
        truth: torch.Tensor,
        pred: torch.Tensor,
    ) -> torch.Tensor:
        """Per-event MSE over standardized physics observables.

        The 15-observable bundle includes invariant mass (column 4). This is a
        paired per-event anchor used by `x_reco_loss` (x -> z -> x), not a
        marginal/narrow-window mass constraint. It is intentionally retained by
        the v3.6A ablation, which removes only explicit distribution-level mass
        supervision (mass_w1, pair_mass_w1, resonance_mass_w1, mass_kin_swd.mll).
        """
        truth_std = self.standardize_features(
            self.physics_features(truth),
            self.feature_mean,
            self.feature_std,
        )
        pred_std = self.standardize_features(
            self.physics_features(pred),
            self.feature_mean,
            self.feature_std,
        )
        return torch.mean((truth_std - pred_std) ** 2)

    def multiscale_mmd(self, truth: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        if self.standardize_raw_matching:
            truth_std = self.standardize_raw(truth)
            pred_std = self.standardize_raw(pred)
        else:
            truth_std = truth
            pred_std = pred
        xx = torch.mm(truth_std, truth_std.t())
        yy = torch.mm(pred_std, pred_std.t())
        xy = torch.mm(truth_std, pred_std.t())
        rx = torch.diag(xx).unsqueeze(0)
        ry = torch.diag(yy).unsqueeze(0)
        dxx = torch.clamp(rx.t() + rx - 2.0 * xx, min=0.0)
        dyy = torch.clamp(ry.t() + ry - 2.0 * yy, min=0.0)
        dxy = torch.clamp(rx.t() + ry - 2.0 * xy, min=0.0)
        loss = truth_std.new_tensor(0.0)
        for scale in self.mmd_scales:
            gamma = 1.0 / max(scale, self.eps)
            loss = loss + torch.exp(-gamma * dxx).mean()
            loss = loss + torch.exp(-gamma * dyy).mean()
            loss = loss - 2.0 * torch.exp(-gamma * dxy).mean()
        return loss / max(1, len(self.mmd_scales))

    def distribution_components(self, truth: torch.Tensor, pred: torch.Tensor) -> dict[str, torch.Tensor]:
        """Differentiable distribution-level OT components between two batches.

        INVARIANT-MASS TRACE (J/psi v3.6A mass ablation reference):

        Explicit distribution-level mass terms:
          mass_w1 / pair_mass_w1  -> W1 on the standardized invariant mass
                                     (identical value; two separate weight knobs).
          resonance_mass_w1       -> W1 on physical mass inside the configured
                                     narrow J/psi window [center +/- half_width].
          mass_kin_swd            -> joint SWD over [m_ll, pT_ll, y_ll,
                                     cos(dphi), sin(dphi)]; the m_ll column is
                                     independently disableable through
                                     `mass_kin_swd_components: {mll: 0.0}`.

        Indirect mass-bearing terms (NOT removed by the v3.6A config):
          physics_swd             -> joint SWD over the 15 physics features,
                                     which include m_ll (column 4).
          physics_coord_swd       -> SWD over per-lepton log(pT), eta, direction
                                     and log(E); mass enters only through the
                                     energy coordinates.
          raw_swd / marginal_w1 /
          mmd                      -> 8-vector distribution matching; mass is a
                                     nonlinear function of the components, which
                                     is the intended physics-learning channel.
        """
        # Build the differentiable physics bundle once per side and reuse it for
        # every component instead of recomputing the feature stack repeatedly.
        truth_named = build_ee_physics_features(
            truth,
            self.eps,
            self.daughter_masses,
            mass_from_energy=self.mass_from_energy,
        )
        pred_named = build_ee_physics_features(
            pred,
            self.eps,
            self.daughter_masses,
            mass_from_energy=self.mass_from_energy,
        )
        if self.standardize_raw_matching:
            truth_std = self.standardize_raw(truth)
            pred_std = self.standardize_raw(pred)
        else:
            truth_std = truth
            pred_std = pred

        truth_features_std = self.standardize_features(
            truth_named["physics_features"],
            self.feature_mean,
            self.feature_std,
        )
        pred_features_std = self.standardize_features(
            pred_named["physics_features"],
            self.feature_mean,
            self.feature_std,
        )

        truth_transverse_std = self.standardize_features(
            self.transverse_features(truth),
            self.transverse_mean,
            self.transverse_std,
        )
        pred_transverse_std = self.standardize_features(
            self.transverse_features(pred),
            self.transverse_mean,
            self.transverse_std,
        )

        truth_longitudinal_std = self.standardize_features(
            self.longitudinal_features(truth),
            self.longitudinal_mean,
            self.longitudinal_std,
        )
        pred_longitudinal_std = self.standardize_features(
            self.longitudinal_features(pred),
            self.longitudinal_mean,
            self.longitudinal_std,
        )

        truth_mass_std = self.standardize_mass(truth_named["m_ee"])
        pred_mass_std = self.standardize_mass(pred_named["m_ee"])

        truth_coord_std = self.standardize_features(
            truth_named["physics_coord_features"],
            self.physics_coord_mean,
            self.physics_coord_std,
        )
        pred_coord_std = self.standardize_features(
            pred_named["physics_coord_features"],
            self.physics_coord_mean,
            self.physics_coord_std,
        )

        eta_norm = self.to_like(self.delta_eta_std, truth) + self.eps
        components = {
            "raw_swd": sliced_wasserstein(truth_std, pred_std, self.num_slices, self.p),
            "marginal_w1": self._marginal_w1_std(truth_std, pred_std),
            "mass_w1": self.wasserstein_1d_sorted(truth_mass_std, pred_mass_std),
            "resonance_mass_w1": self.resonance_mass_w1(
                truth_named["m_ee"],
                pred_named["m_ee"],
            ),
            "physics_swd": sliced_wasserstein(
                truth_features_std,
                pred_features_std,
                self.num_slices,
                self.p,
            ),
            "mass_kin_swd": self._mass_kin_swd(
                truth_features_std,
                pred_features_std,
            ),
            "transverse_w1": self._marginal_w1_std(
                truth_transverse_std,
                pred_transverse_std,
            ),
            "longitudinal_w1": self._marginal_w1_std(
                truth_longitudinal_std,
                pred_longitudinal_std,
            ),
            "tail_w1": self._tail_w1_std(truth_std, pred_std),
            "pair_mass_w1": self.wasserstein_1d_sorted(truth_mass_std, pred_mass_std),
            "pair_pt_w1": self.wasserstein_1d_sorted(
                truth_features_std[:, 5],
                pred_features_std[:, 5],
            ),
            "lepton_pt_w1": 0.5
            * (
                self.wasserstein_1d_sorted(
                    truth_features_std[:, 0],
                    pred_features_std[:, 0],
                )
                + self.wasserstein_1d_sorted(
                    truth_features_std[:, 1],
                    pred_features_std[:, 1],
                )
            ),
            "delta_phi_w1": self.wasserstein_1d_sorted(
                truth_named["delta_phi"] / torch.pi,
                pred_named["delta_phi"] / torch.pi,
            ),
            "delta_eta_w1": self.wasserstein_1d_sorted(
                truth_named["delta_eta"] / eta_norm,
                pred_named["delta_eta"] / eta_norm,
            ),
            "pair_rapidity_w1": self.wasserstein_1d_sorted(
                truth_features_std[:, 6],
                pred_features_std[:, 6],
            ),
            "physics_coord_swd": sliced_wasserstein(
                truth_coord_std,
                pred_coord_std,
                self.num_slices,
                self.p,
            ),
        }
        if self.weights.get("mmd", 0.0) > 0.0:
            components["mmd"] = self.multiscale_mmd(truth, pred)
        else:
            components["mmd"] = truth.new_tensor(0.0)
        return components

    def _mass_kin_swd(
        self,
        truth_features_std: torch.Tensor,
        pred_features_std: torch.Tensor,
    ) -> torch.Tensor:
        """Joint sliced-Wasserstein over the mass-kinematics bundle.

        Selected columns come from `mass_kin_swd_columns` filtered by
        `mass_kin_swd_component_weights` (weights <= 0 drop the observable).
        Dropping the invariant mass leaves [pT_ll, y_ll, cos(dphi), sin(dphi)].
        """
        columns = [
            column
            for name, column in self.mass_kin_swd_columns.items()
            if self.mass_kin_swd_component_weights[name] > 0.0
        ]
        if not columns:
            return truth_features_std.new_tensor(0.0)
        return sliced_wasserstein(
            truth_features_std[:, columns],
            pred_features_std[:, columns],
            self.num_slices,
            self.p,
        )

    def distribution_loss(self, truth: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        components = self.distribution_components(truth, pred)
        loss = truth.new_tensor(0.0)
        for key, value in components.items():
            loss = loss + float(self.weights.get(key, 0.0)) * value
        return loss


class DualSpaceFeatureOTLoss:
    """Loss API with independent x-space and z-space normalization."""

    def __init__(
        self,
        x_train: np.ndarray,
        z_train: np.ndarray,
        loss_config: dict[str, Any],
        daughter_masses=None,
    ):
        validate_loss_config(loss_config)
        self.kind = str(loss_config.get("kind", CANONICAL_LOSS_KIND))
        self.daughter_masses = validate_daughter_masses(daughter_masses)
        # Vanilla OTUS/SWAE mode (v3.7 and v3.8): the only training terms are
        # the raw per-event reconstruction MSE and the 8D sliced-Wasserstein
        # latent term. All other component weights are ignored in this mode.
        #
        # ``vanilla_v3_7`` is the historical name and is retained verbatim:
        # it selects the vanilla mode with the standardized latent SWD that
        # v3.7 used by default. ``vanilla_swae`` is the generalized name and
        # selects the same two-term mode; whether its latent SWD operates on
        # raw or standardized coordinates is controlled explicitly by
        # ``standardize_raw_matching`` (v3.8 sets it false for raw-coordinate
        # latent SWD). Existing v3.7 configs/checkpoints are unaffected.
        self.vanilla_v3_7 = bool(loss_config.get("vanilla_v3_7", False))
        self.vanilla_swae = bool(loss_config.get("vanilla_swae", False)) or self.vanilla_v3_7
        space_weight_overrides = loss_config.get("space_weights", {})
        x_loss_config = dict(loss_config)
        x_loss_config.update(space_weight_overrides.get("x", {}))
        z_loss_config = dict(loss_config)
        z_loss_config.update(space_weight_overrides.get("z", {}))
        self.x_space = SpaceFeatureOTLoss(
            x_train,
            x_loss_config,
            name="x",
            daughter_masses=self.daughter_masses,
            mass_from_energy=False,
        )
        self.z_space = SpaceFeatureOTLoss(
            z_train,
            z_loss_config,
            name="z",
            daughter_masses=self.daughter_masses,
            mass_from_energy=True,
        )
        self.num_slices = int(loss_config.get("num_slices", 1000))
        self.decoder_num_noise_samples = max(1, int(loss_config.get("decoder_num_noise_samples", 1)))
        self.x_reco_physics_w1 = float(loss_config.get("x_reco_physics_w1", 0.0))
        self.latest_components: dict[str, torch.Tensor] = {}
        score_weights = loss_config.get("selection_score", {})
        self.selection_weights = {
            "x_sim": float(score_weights.get("x_sim", 1.0)),
            "z_prior": float(score_weights.get("z_prior", 0.7)),
            "x_reco": float(score_weights.get("x_reco", 0.2)),
            "cycle": float(score_weights.get("cycle", 0.0)),
        }

    def set_num_slices(self, num_slices: int) -> None:
        self.num_slices = int(num_slices)
        self.x_space.set_num_slices(num_slices)
        self.z_space.set_num_slices(num_slices)

    def standardize_x_raw(self, x: torch.Tensor) -> torch.Tensor:
        return self.x_space.standardize_raw(x)

    def standardize_z_raw(self, z: torch.Tensor) -> torch.Tensor:
        return self.z_space.standardize_raw(z)

    def paired_mse_standardized(self, a: torch.Tensor, b: torch.Tensor, standardize_fun) -> torch.Tensor:
        return torch.mean((standardize_fun(a) - standardize_fun(b)) ** 2)

    def reset_components(self) -> None:
        self.latest_components = {}

    def _weighted_distribution_loss(
        self,
        space: SpaceFeatureOTLoss,
        truth: torch.Tensor,
        pred: torch.Tensor,
        prefix: str,
    ) -> torch.Tensor:
        components = space.distribution_components(truth, pred)
        loss = truth.new_tensor(0.0)
        for key, value in components.items():
            self.latest_components[f"{prefix}_{key}"] = value
            weight = float(space.weights.get(key, 0.0))
            self.latest_components[f"{prefix}_{key}_raw"] = value
            self.latest_components[f"{prefix}_{key}_weighted"] = weight * value
            loss = loss + weight * value
        return loss

    def z_prior_loss(self, z_true: torch.Tensor, z_encoded: torch.Tensor) -> torch.Tensor:
        if self.vanilla_swae:
            if z_true.shape[0] != z_encoded.shape[0]:
                raise ValueError(
                    "Vanilla SWAE latent SWD requires equal batch cardinalities: "
                    f"z_true has {z_true.shape[0]} rows, z_encoded has {z_encoded.shape[0]} rows."
                )
            if self.z_space.standardize_raw_matching:
                truth_std = self.z_space.standardize_raw(z_true)
                pred_std = self.z_space.standardize_raw(z_encoded)
            else:
                truth_std = z_true
                pred_std = z_encoded
            value = sliced_wasserstein(
                truth_std,
                pred_std,
                self.num_slices,
                self.z_space.p,
            )
            # The raw (unweighted) SW value is the scalar multiplied by the
            # stage lambda in the trainer. The *_weighted alias here uses the
            # per-component config weight (1.0); the stage-weighted value is
            # logged separately as train_z_loss_weighted.
            self.latest_components["z_raw_swd"] = value
            self.latest_components["z_raw_swd_raw"] = value
            self.latest_components["z_raw_swd_weighted"] = value
            return value
        loss = self._weighted_distribution_loss(self.z_space, z_true, z_encoded, "z")
        # Per-component marginal W1 (standardized), the channel-independent
        # marginal term discussed in the encoder-alignment diagnostic.
        if self.z_space.standardize_raw_matching:
            truth_std = self.z_space.standardize_raw(z_true)
            pred_std = self.z_space.standardize_raw(z_encoded)
        else:
            truth_std = z_true
            pred_std = z_encoded
        marginal_weight = float(self.z_space.weights.get("marginal_w1", 1.0))
        n_components = min(truth_std.shape[1], pred_std.shape[1])
        for j in range(n_components):
            value = self.z_space.wasserstein_1d_sorted(
                truth_std[:, j],
                pred_std[:, j],
            )
            self.latest_components[f"z_component_w1_{j:02d}"] = value
            self.latest_components[f"z_component_w1_{j:02d}_weighted"] = (
                marginal_weight * value / max(1, n_components)
            )
        return loss

    def x_sim_loss(self, x_true: torch.Tensor, x_from_z: torch.Tensor) -> torch.Tensor:
        return self._weighted_distribution_loss(self.x_space, x_true, x_from_z, "x")

    def x_reco_loss(self, x_true: torch.Tensor, x_reco: torch.Tensor) -> torch.Tensor:
        if self.vanilla_swae:
            value = torch.mean((x_true - x_reco) ** 2)
            self.latest_components["x_reco_mse_raw"] = value
            return value
        loss = self.paired_mse_standardized(x_true, x_reco, self.standardize_x_raw)
        if self.x_reco_physics_w1 > 0.0:
            loss = loss + self.x_reco_physics_w1 * self.x_space.paired_physics_mse_standardized(
                x_true,
                x_reco,
            )
        return loss

    def encoder_anchor_loss(self, z_encoded: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
        return anchor_loss(z_encoded, x_true)

    def decoder_anchor_loss(self, z_true: torch.Tensor, x_from_z: torch.Tensor) -> torch.Tensor:
        return anchor_loss(z_true, x_from_z)

    def z_transverse_loss(self, z_true: torch.Tensor, z_encoded: torch.Tensor) -> torch.Tensor:
        return self.z_space.feature_w1(
            self.z_space.transverse_features(z_true),
            self.z_space.transverse_features(z_encoded),
            self.z_space.transverse_mean,
            self.z_space.transverse_std,
        )

    def validation_score(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        return (
            self.selection_weights["x_sim"] * losses["alt_x_loss"]
            + self.selection_weights["z_prior"] * losses["z_loss"]
            + self.selection_weights["x_reco"] * losses["x_loss"]
            + self.selection_weights["cycle"] * losses.get("cycle_loss", losses["x_loss"])
        )

    def __call__(self, z_true: torch.Tensor, z_encoded: torch.Tensor) -> torch.Tensor:
        return self.z_prior_loss(z_true, z_encoded)


class CmsDoubleElectronLossFactory(DualSpaceFeatureOTLoss):
    """Canonical current CMS DoubleElectron OTUS loss."""


class CmsJpsiDoubleMuonLossFactory(DualSpaceFeatureOTLoss):
    """J/psi dimuon alias of the same charge-ordered dilepton OTUS loss."""


class OriginalOtusFeatureLossFactory(CmsDoubleElectronLossFactory):
    """Compatibility alias for older configs/imports."""
