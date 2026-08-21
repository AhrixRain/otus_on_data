"""Loss-factory wrapper that adds Tier A neural-OT terms to the current loss.

The wrapper subclasses the current J/psi loss factory so all existing
components, logging conventions, and config validation continue to work.
A new optional ``sota`` block controls the Run E additions::

  loss:
    kind: cms_jpsi_doublemuon_loss
    vanilla_swae: false
    ... existing Run D terms ...
    sota:
      sinkhorn: {enabled: true, weight: 0.5, regularization: 0.05,
                 max_iter: 100, max_batch: 1024}
      monge_gap: {enabled: false, weight: 0.1}
      max_swd: {enabled: true, weight: 0.5, num_directions: 32,
                lr: 0.001, diversity_weight: 0.05}

``SotaLossFactory`` is a plain object like the current factories, not an
``nn.Module``.  Max-SW directions are trained by an internal adversarial
ascent step inside each distribution-loss call, so the existing optimizer
loop needs no changes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

try:
    from .ot import PhysicsGroundCost, SinkhornConfig, entropic_monge_gap, sinkhorn_divergence
    from .max_swd import MaxSlicedWasserstein
except ImportError:  # scripts_sota/ added directly to sys.path
    from ot import PhysicsGroundCost, SinkhornConfig, entropic_monge_gap, sinkhorn_divergence
    from max_swd import MaxSlicedWasserstein


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from loss import CmsJpsiDoubleMuonLossFactory  # noqa: E402


def _bool(value: Any, default: bool = False) -> bool:
    return bool(default if value is None else value)


class SotaLossFactory(CmsJpsiDoubleMuonLossFactory):
    """Run E loss factory: current components + physics-cost neural OT + max-SW."""

    def __init__(self, x_train, z_train, loss_config, daughter_masses=None):
        base_config = dict(loss_config)
        sota = base_config.pop("sota", None) or {}
        super().__init__(x_train, z_train, base_config, daughter_masses)
        self.sota_config = sota
        self.x_ground = PhysicsGroundCost(
            x_train,
            daughter_masses=daughter_masses,
            feature_weights=sota.get("ground_feature_weights"),
        )
        self.z_ground = PhysicsGroundCost(
            z_train,
            daughter_masses=daughter_masses,
            feature_weights=sota.get("ground_feature_weights"),
            mass_from_energy=True,
        )

        max_swd_cfg = sota.get("max_swd") or {}
        self.max_swd_enabled = _bool(max_swd_cfg.get("enabled", True))
        self.max_swd_weight = float(max_swd_cfg.get("weight", 0.5))
        self.max_swd_modules = {
            "x": MaxSlicedWasserstein(
                feature_dim=14,
                num_directions=int(max_swd_cfg.get("num_directions", 32)),
                p=int(max_swd_cfg.get("p", 2)),
                lr=float(max_swd_cfg.get("lr", 1e-3)),
                diversity_weight=float(max_swd_cfg.get("diversity_weight", 0.05)),
                direction_grad_clip=float(max_swd_cfg.get("direction_grad_clip", 1.0)),
                update_every=int(max_swd_cfg.get("update_every", 5)),
                distance_clamp=max_swd_cfg.get("distance_clamp"),
            ),
            "z": MaxSlicedWasserstein(
                feature_dim=14,
                num_directions=int(max_swd_cfg.get("num_directions", 32)),
                p=int(max_swd_cfg.get("p", 2)),
                lr=float(max_swd_cfg.get("lr", 1e-3)),
                diversity_weight=float(max_swd_cfg.get("diversity_weight", 0.05)),
                direction_grad_clip=float(max_swd_cfg.get("direction_grad_clip", 1.0)),
                update_every=int(max_swd_cfg.get("update_every", 5)),
                distance_clamp=max_swd_cfg.get("distance_clamp"),
            ),
        }

        sinkhorn_cfg = sota.get("sinkhorn") or {}
        self.sinkhorn_enabled = _bool(sinkhorn_cfg.get("enabled", True))
        self.sinkhorn_weight = float(sinkhorn_cfg.get("weight", 0.5))
        self.sinkhorn_log_scale = _bool(sinkhorn_cfg.get("log_scale", False))
        self.sinkhorn_config = SinkhornConfig(
            regularization=float(sinkhorn_cfg.get("regularization", 0.05)),
            max_iter=int(sinkhorn_cfg.get("max_iter", 100)),
            tolerance=float(sinkhorn_cfg.get("tolerance", 1e-6)),
            max_batch=int(sinkhorn_cfg.get("max_batch", 1024)),
            debias=_bool(sinkhorn_cfg.get("debias", True), True),
        )
        monge_cfg = sota.get("monge_gap") or {}
        self.monge_enabled = _bool(monge_cfg.get("enabled", False))
        self.monge_weight = float(monge_cfg.get("weight", 0.1))
        self.monge_regularization = float(monge_cfg.get("regularization", 0.05))
        self.monge_max_iter = int(monge_cfg.get("max_iter", 100))

        for module in self.max_swd_modules.values():
            module.to(dtype=torch.float32)

    def _max_swd_device(self, truth: torch.Tensor) -> torch.Tensor:
        return truth.new_tensor(0.0)

    def _sota_terms(self, prefix, ground, max_swd, truth, prediction):
        """Compute and log the Run E distribution terms for one space."""
        value = truth.new_tensor(0.0)
        ground = ground.to(truth.device)
        features_truth = ground.features(truth)
        features_prediction = ground.features(prediction)

        if self.max_swd_enabled and self.max_swd_weight > 0.0:
            max_swd.to(truth.device)
            if torch.is_grad_enabled():
                max_swd.adversarial_step(features_truth, features_prediction)
            max_value = max_swd(features_truth, features_prediction)
            self.latest_components[f"{prefix}_sota_max_swd"] = max_value
            self.latest_components[f"{prefix}_sota_max_swd_weighted"] = (
                self.max_swd_weight * max_value
            )
            value = value + self.max_swd_weight * max_value

        if self.sinkhorn_enabled and self.sinkhorn_weight > 0.0:
            sinkhorn_raw = sinkhorn_divergence(ground, truth, prediction, self.sinkhorn_config)
            sinkhorn_value = torch.log1p(sinkhorn_raw) if self.sinkhorn_log_scale else sinkhorn_raw
            self.latest_components[f"{prefix}_sota_sinkhorn_raw"] = sinkhorn_raw
            self.latest_components[f"{prefix}_sota_sinkhorn"] = sinkhorn_value
            self.latest_components[f"{prefix}_sota_sinkhorn_weighted"] = (
                self.sinkhorn_weight * sinkhorn_value
            )
            value = value + self.sinkhorn_weight * sinkhorn_value

        if self.monge_enabled and self.monge_weight > 0.0:
            monge_value = entropic_monge_gap(
                ground,
                truth,
                prediction,
                regularization=self.monge_regularization,
                max_iter=self.monge_max_iter,
            )
            self.latest_components[f"{prefix}_sota_monge_gap"] = monge_value
            self.latest_components[f"{prefix}_sota_monge_gap_weighted"] = (
                self.monge_weight * monge_value
            )
            value = value + self.monge_weight * monge_value
        return value

    def z_prior_loss(self, z_true, z_encoded):
        base = super().z_prior_loss(z_true, z_encoded)
        extra = self._sota_terms("z", self.z_ground, self.max_swd_modules["z"], z_true, z_encoded)
        self.latest_components["z_sota_extra"] = extra
        return base + extra

    def x_sim_loss(self, x_true, x_from_z):
        base = super().x_sim_loss(x_true, x_from_z)
        extra = self._sota_terms("x", self.x_ground, self.max_swd_modules["x"], x_true, x_from_z)
        self.latest_components["x_sota_extra"] = extra
        return base + extra

    def state_dict(self) -> dict[str, Any]:
        return {
            "x_max_swd": self.max_swd_modules["x"].state_dict(),
            "z_max_swd": self.max_swd_modules["z"].state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if "x_max_swd" in state:
            self.max_swd_modules["x"].load_state_dict(state["x_max_swd"])
        if "z_max_swd" in state:
            self.max_swd_modules["z"].load_state_dict(state["z_max_swd"])
