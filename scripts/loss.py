from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
UTILITY_DIR = REPO_ROOT / "utilityFunctions"
if str(UTILITY_DIR) not in sys.path:
    sys.path.insert(0, str(UTILITY_DIR))

from func_utils import anchor_loss  # noqa: E402


P = 2
CANONICAL_LOSS_KIND = "cms_doubleelectron_loss"


def build_ee_physics_features(x: torch.Tensor, eps: float = 1e-6) -> dict[str, torch.Tensor]:
    """Differentiable CMS DoubleElectron observables for [e- p4, e+ p4]."""
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
    pair_mass2 = pair_e**2 - pair_px**2 - pair_py**2 - pair_pz**2
    pair_mass = torch.sqrt(torch.clamp(pair_mass2, min=eps))
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


class SpaceFeatureOTLoss:
    """Continuous feature OT loss for one four-vector space."""

    def __init__(
        self,
        train_samples: np.ndarray,
        loss_config: dict[str, Any],
        *,
        name: str,
        eps: float = 1e-6,
    ):
        self.name = name
        self.eps = float(loss_config.get("eps", eps))
        self.num_slices = int(loss_config.get("num_slices", 1000))
        self.p = int(loss_config.get("p", P))
        self.weights = {
            "raw_swd": 0.5,
            "marginal_w1": 0.5,
            "mass_w1": 2.0,
            "physics_swd": 1.0,
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
        self.weights.update(
            {
                key: float(loss_config[key])
                for key in self.weights
                if key in loss_config
            }
        )
        self.tail_frac = float(loss_config.get("tail_frac", 0.20))
        self.mmd_scales = [
            float(value) for value in loss_config.get("mmd_scales", [0.5, 1.0, 2.0, 4.0])
        ]

        self.raw_mean = np.mean(train_samples, axis=0)
        self.raw_std = _safe_numpy_std(train_samples)

        with torch.no_grad():
            train = torch.as_tensor(train_samples, dtype=torch.float32)
            mass = self.invariant_mass(train)
            features = self.physics_features(train)
            transverse = self.transverse_features(train)
            longitudinal = self.longitudinal_features(train)
            physics_coord = self.physics_coord_features(train)
            self.mass_mean = mass.mean().detach()
            self.mass_std = mass.std(unbiased=False).detach()
            if not bool(torch.isfinite(self.mass_std).item()) or float(self.mass_std) <= self.eps:
                self.mass_std = torch.ones_like(self.mass_std)
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
        return build_ee_physics_features(values, self.eps)["m_ee"]

    def safe_eta(self, pt: torch.Tensor, pz: torch.Tensor) -> torch.Tensor:
        return torch.asinh(pz / torch.clamp(pt, min=self.eps))

    def safe_rapidity(self, energy: torch.Tensor, pz: torch.Tensor) -> torch.Tensor:
        numerator = torch.clamp(energy + pz, min=self.eps)
        denominator = torch.clamp(energy - pz, min=self.eps)
        ratio = torch.clamp(numerator / denominator, min=self.eps, max=1.0 / self.eps)
        return 0.5 * torch.log(ratio)

    def physics_features(self, values: torch.Tensor) -> torch.Tensor:
        return build_ee_physics_features(values, self.eps)["physics_features"]

    def physics_coord_features(self, values: torch.Tensor) -> torch.Tensor:
        return build_ee_physics_features(values, self.eps)["physics_coord_features"]

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

    def marginal_w1(self, truth: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        truth_std = self.standardize_raw(truth)
        pred_std = self.standardize_raw(pred)
        loss = truth_std.new_tensor(0.0)
        for dim in range(truth_std.shape[1]):
            loss = loss + self.wasserstein_1d_sorted(truth_std[:, dim], pred_std[:, dim])
        return loss / truth_std.shape[1]

    def feature_w1(
        self,
        truth_features: torch.Tensor,
        pred_features: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        truth_std = self.standardize_features(truth_features, mean, std)
        pred_std = self.standardize_features(pred_features, mean, std)
        loss = truth_std.new_tensor(0.0)
        for dim in range(truth_std.shape[1]):
            loss = loss + self.wasserstein_1d_sorted(truth_std[:, dim], pred_std[:, dim])
        return loss / truth_std.shape[1]

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

    def tail_w1(self, truth: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        truth_std = self.standardize_raw(truth)
        pred_std = self.standardize_raw(pred)
        loss = truth_std.new_tensor(0.0)
        for dim in (0, 1, 2, 4, 5, 6):
            loss = loss + self.tail_wasserstein_abs(truth_std[:, dim], pred_std[:, dim])
        return loss / 6.0

    def multiscale_mmd(self, truth: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        truth_std = self.standardize_raw(truth)
        pred_std = self.standardize_raw(pred)
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
        truth_std = self.standardize_raw(truth)
        pred_std = self.standardize_raw(pred)
        truth_features = self.physics_features(truth)
        pred_features = self.physics_features(pred)
        truth_transverse = self.transverse_features(truth)
        pred_transverse = self.transverse_features(pred)
        truth_longitudinal = self.longitudinal_features(truth)
        pred_longitudinal = self.longitudinal_features(pred)
        truth_named = build_ee_physics_features(truth, self.eps)
        pred_named = build_ee_physics_features(pred, self.eps)
        truth_coord = truth_named["physics_coord_features"]
        pred_coord = pred_named["physics_coord_features"]
        truth_coord_std = self.standardize_features(
            truth_coord,
            self.physics_coord_mean,
            self.physics_coord_std,
        )
        pred_coord_std = self.standardize_features(
            pred_coord,
            self.physics_coord_mean,
            self.physics_coord_std,
        )
        components = {
            "raw_swd": sliced_wasserstein(truth_std, pred_std, self.num_slices, self.p),
            "marginal_w1": self.marginal_w1(truth, pred),
            "mass_w1": self.wasserstein_1d_sorted(
                self.standardize_mass(self.invariant_mass(truth)),
                self.standardize_mass(self.invariant_mass(pred)),
            ),
            "physics_swd": sliced_wasserstein(
                self.standardize_features(truth_features, self.feature_mean, self.feature_std),
                self.standardize_features(pred_features, self.feature_mean, self.feature_std),
                self.num_slices,
                self.p,
            ),
            "transverse_w1": self.feature_w1(
                truth_transverse,
                pred_transverse,
                self.transverse_mean,
                self.transverse_std,
            ),
            "longitudinal_w1": self.feature_w1(
                truth_longitudinal,
                pred_longitudinal,
                self.longitudinal_mean,
                self.longitudinal_std,
            ),
            "tail_w1": self.tail_w1(truth, pred),
            "pair_mass_w1": self.wasserstein_1d_sorted(
                self.standardize_mass(truth_named["m_ee"]),
                self.standardize_mass(pred_named["m_ee"]),
            ),
            "pair_pt_w1": self.wasserstein_1d_sorted(
                self.standardized_physics_column(truth, 5),
                self.standardized_physics_column(pred, 5),
            ),
            "lepton_pt_w1": 0.5
            * (
                self.wasserstein_1d_sorted(
                    self.standardized_physics_column(truth, 0),
                    self.standardized_physics_column(pred, 0),
                )
                + self.wasserstein_1d_sorted(
                    self.standardized_physics_column(truth, 1),
                    self.standardized_physics_column(pred, 1),
                )
            ),
            "delta_phi_w1": self.wasserstein_1d_sorted(
                truth_named["delta_phi"] / torch.pi,
                pred_named["delta_phi"] / torch.pi,
            ),
            "delta_eta_w1": self.wasserstein_1d_sorted(
                truth_named["delta_eta"]
                / (self.to_like(self.feature_std[2], truth) + self.to_like(self.feature_std[3], truth) + self.eps),
                pred_named["delta_eta"]
                / (self.to_like(self.feature_std[2], pred) + self.to_like(self.feature_std[3], pred) + self.eps),
            ),
            "pair_rapidity_w1": self.wasserstein_1d_sorted(
                self.standardized_physics_column(truth, 6),
                self.standardized_physics_column(pred, 6),
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

    def distribution_loss(self, truth: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        components = self.distribution_components(truth, pred)
        loss = truth.new_tensor(0.0)
        for key, value in components.items():
            loss = loss + float(self.weights.get(key, 0.0)) * value
        return loss


class DualSpaceFeatureOTLoss:
    """Loss API with independent x-space and z-space normalization."""

    def __init__(self, x_train: np.ndarray, z_train: np.ndarray, loss_config: dict[str, Any]):
        self.kind = str(loss_config.get("kind", CANONICAL_LOSS_KIND))
        self.x_space = SpaceFeatureOTLoss(x_train, loss_config, name="x")
        self.z_space = SpaceFeatureOTLoss(z_train, loss_config, name="z")
        self.num_slices = int(loss_config.get("num_slices", 1000))
        self.decoder_num_noise_samples = max(1, int(loss_config.get("decoder_num_noise_samples", 1)))
        self.latest_components: dict[str, torch.Tensor] = {}
        score_weights = loss_config.get("selection_score", {})
        self.selection_weights = {
            "x_sim": float(score_weights.get("x_sim", 1.0)),
            "z_prior": float(score_weights.get("z_prior", 0.7)),
            "x_reco": float(score_weights.get("x_reco", 0.2)),
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
            loss = loss + float(space.weights.get(key, 0.0)) * value
        return loss

    def z_prior_loss(self, z_true: torch.Tensor, z_encoded: torch.Tensor) -> torch.Tensor:
        return self._weighted_distribution_loss(self.z_space, z_true, z_encoded, "z")

    def x_sim_loss(self, x_true: torch.Tensor, x_from_z: torch.Tensor) -> torch.Tensor:
        return self._weighted_distribution_loss(self.x_space, x_true, x_from_z, "x")

    def x_reco_loss(self, x_true: torch.Tensor, x_reco: torch.Tensor) -> torch.Tensor:
        return self.x_space.paired_mse_standardized(x_true, x_reco)

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
        )

    def __call__(self, z_true: torch.Tensor, z_encoded: torch.Tensor) -> torch.Tensor:
        return self.z_prior_loss(z_true, z_encoded)


class CmsDoubleElectronLossFactory(DualSpaceFeatureOTLoss):
    """Canonical current CMS DoubleElectron OTUS loss."""


class OriginalOtusFeatureLossFactory(CmsDoubleElectronLossFactory):
    """Compatibility alias for older configs/imports."""
