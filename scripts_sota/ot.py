"""Physics-aware neural-OT components for Run E.

The original OTUS/SWAE objective measures sliced-Wasserstein distance on raw
Cartesian four-momenta.  A narrow resonance such as J/psi -> mu+mu- is a
~30 MeV shell embedded in GeV-scale coordinate distributions, so random
linear projections are nearly blind to it (this was measured for the paper
baseline and is the motivation for this module).

This module therefore implements distribution distances with an explicit
physics-aware ground metric:

  [log pT(mu-), eta(mu-), sin phi(mu-), cos phi(mu-),
   log pT(mu+), eta(mu+), sin phi(mu+), cos phi(mu+),
   log m(mumu), log pair pT, pair rapidity,
   cos delta-phi, sin delta-phi, delta-eta]

The features are standardized on the training sample.  The cost is a weighted
squared Euclidean distance in this space, so a 30 MeV mass difference is a
first-class citizen of the cost instead of an invisible nonlinear shell.

Components
----------
``PhysicsGroundCost``
    Feature builder + train-standardized weighted squared cost matrix.

``sinkhorn_divergence``
    Debiased entropic OT (Sinkhorn divergence) between two minibatches.

``entropic_monge_gap``
    Barycentric-projection Monge gap proxy.  For an entropic plan P the
    barycentric map is T(x_i) = sum_j P_ij y_j / sum_j P_ij.  The gap
    ``mean_i c(x_i, T(x_i)) - OT_eps`` is non-negative for the exact OT cost
    and is used here as a differentiable regularizer that pushes the learned
    conditional map towards a Monge map rather than a spread-out coupling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn

from physics import invariant_mass_torch, validate_daughter_masses


_EPS = 1e-8


def cylindrical_physics_features(
    values: torch.Tensor,
    daughter_masses: Sequence[float] | None = None,
    eps: float = _EPS,
    mass_from_energy: bool = False,
) -> torch.Tensor:
    """Return standardized-ready cylindrical + pair physics features [N, 14].

    The representation is chosen so that every coordinate is either linear,
    log-positive, or a sin/cos pair on S^1; no azimuthal discontinuity enters
    the ground metric.
    """
    if not isinstance(values, torch.Tensor):
        values = torch.as_tensor(values)
    if values.ndim != 2 or values.shape[1] != 8:
        raise ValueError(f"Expected [N, 8] dimuon four-vectors, got {tuple(values.shape)}")
    masses = validate_daughter_masses(daughter_masses)
    p1, p2 = values[:, 0:4], values[:, 4:8]
    pt1 = torch.sqrt(torch.clamp(p1[:, 0] ** 2 + p1[:, 1] ** 2, min=eps))
    pt2 = torch.sqrt(torch.clamp(p2[:, 0] ** 2 + p2[:, 1] ** 2, min=eps))
    eta1 = torch.asinh(p1[:, 2] / torch.clamp(pt1, min=eps))
    eta2 = torch.asinh(p2[:, 2] / torch.clamp(pt2, min=eps))
    phi1 = torch.atan2(p1[:, 1], p1[:, 0])
    phi2 = torch.atan2(p2[:, 1], p2[:, 0])
    delta_phi = torch.atan2(
        torch.sin(phi1 - phi2),
        torch.cos(phi1 - phi2),
    )
    pair = p1 + p2
    pair_pt = torch.sqrt(torch.clamp(pair[:, 0] ** 2 + pair[:, 1] ** 2, min=eps))
    rapidity = 0.5 * torch.log(
        torch.clamp(pair[:, 3] + pair[:, 2], min=eps)
        / torch.clamp(pair[:, 3] - pair[:, 2], min=eps)
    )
    if mass_from_energy:
        mass2 = pair[:, 3] ** 2 - (
            pair[:, 0] ** 2 + pair[:, 1] ** 2 + pair[:, 2] ** 2
        )
        mass = torch.sqrt(torch.clamp(mass2, min=0.0))
    else:
        mass = invariant_mass_torch(values, daughter_masses=masses, eps=eps)
    return torch.cat(
        [
            torch.log(torch.clamp(pt1, min=eps)).unsqueeze(1),
            eta1.unsqueeze(1),
            torch.sin(phi1).unsqueeze(1),
            torch.cos(phi1).unsqueeze(1),
            torch.log(torch.clamp(pt2, min=eps)).unsqueeze(1),
            eta2.unsqueeze(1),
            torch.sin(phi2).unsqueeze(1),
            torch.cos(phi2).unsqueeze(1),
            torch.log(torch.clamp(mass, min=eps)).unsqueeze(1),
            torch.log(torch.clamp(pair_pt, min=eps)).unsqueeze(1),
            rapidity.unsqueeze(1),
            torch.cos(delta_phi).unsqueeze(1),
            torch.sin(delta_phi).unsqueeze(1),
            (eta1 - eta2).unsqueeze(1),
        ],
        dim=1,
    )


@dataclass(frozen=True)
class SinkhornConfig:
    regularization: float = 0.05
    max_iter: int = 200
    tolerance: float = 1e-6
    max_batch: int = 2048
    debias: bool = True


class PhysicsGroundCost(nn.Module):
    """Train-standardized physics ground metric for two dimuon samples."""

    def __init__(
        self,
        train_p4: np.ndarray,
        *,
        daughter_masses: Sequence[float] | None = None,
        feature_weights: Sequence[float] | None = None,
        eps: float = _EPS,
        mass_from_energy: bool = False,
    ):
        super().__init__()
        self.daughter_masses = validate_daughter_masses(daughter_masses)
        self.eps = float(eps)
        self.mass_from_energy = bool(mass_from_energy)
        default_weights = torch.ones(14, dtype=torch.float32)
        # The pair-mass coordinate gets a relative boost so the ground metric
        # explicitly prices 10^-2-scale mass differences in log m.  This is
        # deliberately not a resonance mass constant; it is a feature weight.
        default_weights[8] = 4.0
        if feature_weights is not None:
            weights = torch.as_tensor(feature_weights, dtype=torch.float32)
            if weights.numel() != 14:
                raise ValueError("feature_weights must contain 14 values")
            if bool((weights <= 0).any()):
                raise ValueError("feature_weights must be positive")
        else:
            weights = default_weights
        self.register_buffer("feature_weights", weights)

        train = torch.as_tensor(train_p4, dtype=torch.float32)
        features = cylindrical_physics_features(
            train,
            daughter_masses=self.daughter_masses,
            eps=self.eps,
            mass_from_energy=self.mass_from_energy,
        )
        mean = features.mean(dim=0)
        std = features.std(dim=0, unbiased=False)
        std = torch.where(torch.isfinite(std) & (std > self.eps), std, torch.ones_like(std))
        self.register_buffer("feature_mean", mean)
        self.register_buffer("feature_std", std)

    def features(self, values: torch.Tensor) -> torch.Tensor:
        raw = cylindrical_physics_features(
            values,
            daughter_masses=self.daughter_masses,
            eps=self.eps,
            mass_from_energy=self.mass_from_energy,
        )
        return (raw - self.feature_mean) / self.feature_std

    def pairwise_cost(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Weighted squared Euclidean cost matrix ``[A, B]`` in feature space."""
        fa = self.features(a)
        fb = self.features(b)
        fa = fa * self.feature_weights
        fb = fb * self.feature_weights
        a2 = (fa * fa).sum(dim=1, keepdim=True)
        b2 = (fb * fb).sum(dim=1, keepdim=True)
        return torch.clamp(a2 - 2.0 * fa @ fb.t() + b2.t(), min=0.0)


class CrossDomainPhysicsGroundCost(PhysicsGroundCost):
    """Common feature scale for z->x OT with domain-correct mass semantics.

    ``pairwise_cost(z, x)`` treats the first argument as MG5 stored-energy
    z-space and the second as on-shell detector x-space.
    """

    def __init__(
        self,
        x_train: np.ndarray,
        z_train: np.ndarray,
        *,
        daughter_masses: Sequence[float] | None = None,
        feature_weights: Sequence[float] | None = None,
        eps: float = _EPS,
    ):
        nn.Module.__init__(self)
        self.daughter_masses = validate_daughter_masses(daughter_masses)
        self.eps = float(eps)
        self.mass_from_energy = False
        default_weights = torch.ones(14, dtype=torch.float32)
        default_weights[8] = 4.0
        weights = (
            default_weights
            if feature_weights is None
            else torch.as_tensor(feature_weights, dtype=torch.float32)
        )
        if weights.numel() != 14 or bool((weights <= 0).any()):
            raise ValueError("feature_weights must contain 14 positive values")
        self.register_buffer("feature_weights", weights)
        x_features = cylindrical_physics_features(
            torch.as_tensor(x_train, dtype=torch.float32),
            daughter_masses=self.daughter_masses,
            eps=self.eps,
            mass_from_energy=False,
        )
        z_features = cylindrical_physics_features(
            torch.as_tensor(z_train, dtype=torch.float32),
            daughter_masses=self.daughter_masses,
            eps=self.eps,
            mass_from_energy=True,
        )
        features = torch.cat([x_features, z_features], dim=0)
        mean = features.mean(dim=0)
        std = features.std(dim=0, unbiased=False)
        std = torch.where(torch.isfinite(std) & (std > self.eps), std, torch.ones_like(std))
        self.register_buffer("feature_mean", mean)
        self.register_buffer("feature_std", std)

    def domain_features(self, values: torch.Tensor, *, space: str) -> torch.Tensor:
        if space not in {"x", "z"}:
            raise ValueError("space must be 'x' or 'z'")
        raw = cylindrical_physics_features(
            values,
            daughter_masses=self.daughter_masses,
            eps=self.eps,
            mass_from_energy=(space == "z"),
        )
        return (raw - self.feature_mean) / self.feature_std

    def pairwise_cost(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        fz = self.domain_features(z, space="z") * self.feature_weights
        fx = self.domain_features(x, space="x") * self.feature_weights
        z2 = (fz * fz).sum(dim=1, keepdim=True)
        x2 = (fx * fx).sum(dim=1, keepdim=True)
        return torch.clamp(z2 - 2.0 * fz @ fx.t() + x2.t(), min=0.0)


def _subsample_pair(
    a: torch.Tensor,
    b: torch.Tensor,
    max_batch: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministically subsample two tensors to at most ``max_batch`` rows.

    Sinkhorn has O(n^2) memory.  The feature dimension is only 14, so a
    2048 x 2048 cost matrix is 32 MiB in float32; this is the default budget.
    """
    if max_batch is None or max_batch <= 0:
        return a, b
    n_a, n_b = int(a.shape[0]), int(b.shape[0])
    if n_a <= max_batch and n_b <= max_batch:
        return a, b
    if generator is None:
        generator = torch.Generator(device="cpu")
    idx_a = torch.randperm(n_a, generator=generator)[:max_batch]
    idx_b = torch.randperm(n_b, generator=generator)[:max_batch]
    return a[idx_a], b[idx_b]


@torch.no_grad()
def _sample_indices(
    n: int,
    k: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if generator is None:
        generator = torch.Generator(device="cpu")
    return torch.randperm(n, generator=generator)[:k]


def sinkhorn_divergence(
    ground_cost: PhysicsGroundCost,
    a: torch.Tensor,
    b: torch.Tensor,
    config: SinkhornConfig | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Debiased Sinkhorn divergence between p4 minibatches.

    ``a`` and ``b`` must already live on the same device/dtype as the ground
    cost buffers.  The returned scalar is differentiable with respect to both
    inputs (the cost matrix is built inside the autograd graph).
    """
    cfg = config or SinkhornConfig()
    a_s, b_s = _subsample_pair(a, b, int(cfg.max_batch), generator)
    if cfg.debias:
        aa_cost = ground_cost.pairwise_cost(a_s, a_s)
        bb_cost = ground_cost.pairwise_cost(b_s, b_s)
        # Zero diagonal for the self-transport entropy term (avoids self-cost).
        aa_cost = aa_cost - torch.diag(aa_cost).diag_embed()
        bb_cost = bb_cost - torch.diag(bb_cost).diag_embed()
        cross_cost = _sinkhorn_cost(
            ground_cost.pairwise_cost(a_s, b_s),
            float(cfg.regularization),
            int(cfg.max_iter),
            float(cfg.tolerance),
        )
        self_cost = (
            _sinkhorn_cost(aa_cost, float(cfg.regularization), int(cfg.max_iter), float(cfg.tolerance))
            + _sinkhorn_cost(bb_cost, float(cfg.regularization), int(cfg.max_iter), float(cfg.tolerance))
        ) / 2.0
        return cross_cost - self_cost
    return _sinkhorn_cost(
        ground_cost.pairwise_cost(a_s, b_s),
        float(cfg.regularization),
        int(cfg.max_iter),
        float(cfg.tolerance),
    )


def _sinkhorn_cost(
    cost: torch.Tensor,
    regularization: float,
    max_iter: int,
    tolerance: float,
) -> torch.Tensor:
    """Log-domain Sinkhorn OT cost for a non-negative cost matrix."""
    n, m = cost.shape
    if regularization <= 0.0:
        raise ValueError("Sinkhorn regularization must be positive")
    dtype = cost.dtype
    device = cost.device
    mu = torch.full((n,), 1.0 / n, dtype=dtype, device=device)
    nu = torch.full((m,), 1.0 / m, dtype=dtype, device=device)
    log_mu = torch.log(mu)
    log_nu = torch.log(nu)

    # Stable log-sum-exp implementation.
    u = torch.zeros_like(mu)
    v = torch.zeros_like(nu)
    k = -cost / regularization
    for _ in range(int(max_iter)):
        u_prev = u.clone()
        # v = log(nu) - logsumexp(k^T + u, dim=0)
        v = log_nu - torch.logsumexp(k.t() + u.unsqueeze(0), dim=1)
        u = log_mu - torch.logsumexp(k + v.unsqueeze(0), dim=1)
        if float((u - u_prev).detach().abs().max()) < tolerance:
            break
    # Transport cost: sum_{ij} P_ij C_ij
    log_p = k + u.unsqueeze(1) + v.unsqueeze(0)
    return torch.exp(log_p).mul(cost).sum()


def entropic_ot_plan(
    ground_cost: PhysicsGroundCost,
    a: torch.Tensor,
    b: torch.Tensor,
    regularization: float = 0.05,
    max_iter: int = 200,
    tolerance: float = 1e-6,
) -> torch.Tensor:
    """Return the entropic OT coupling matrix ``P`` for two minibatches."""
    cost = ground_cost.pairwise_cost(a, b)
    n, m = cost.shape
    dtype = cost.dtype
    device = cost.device
    log_mu = torch.full((n,), math.log(1.0 / n), dtype=dtype, device=device)
    log_nu = torch.full((m,), math.log(1.0 / m), dtype=dtype, device=device)
    u = torch.zeros(n, dtype=dtype, device=device)
    v = torch.zeros(m, dtype=dtype, device=device)
    k = -cost / float(regularization)
    for _ in range(int(max_iter)):
        u_prev = u.clone()
        v = log_nu - torch.logsumexp(k.t() + u.unsqueeze(0), dim=1)
        u = log_mu - torch.logsumexp(k + v.unsqueeze(0), dim=1)
        if float((u - u_prev).detach().abs().max()) < float(tolerance):
            break
    return torch.exp(k + u.unsqueeze(1) + v.unsqueeze(0))


def entropic_monge_gap(
    ground_cost: PhysicsGroundCost,
    truth: torch.Tensor,
    prediction: torch.Tensor,
    regularization: float = 0.05,
    max_iter: int = 100,
    tolerance: float = 1e-6,
    detach_plan: bool = False,
) -> torch.Tensor:
    """Differentiable entropic Monge-gap proxy for ``truth`` -> ``prediction``.

    ``truth`` is regarded as the source measure and ``prediction`` as the
    target measure.  The barycentric map is a deterministic map in the convex
    hull of the target atoms; its transport cost cannot be below the optimal
    transport cost, so the non-negative gap measures how far the entropic
    barycentric coupling is from a deterministic map.
    """
    a_s, b_s = _subsample_pair(truth, prediction, 2048)
    plan = entropic_ot_plan(
        ground_cost,
        a_s,
        b_s,
        regularization=regularization,
        max_iter=max_iter,
        tolerance=tolerance,
    )
    if detach_plan:
        plan = plan.detach()
    row_sum = plan.sum(dim=1, keepdim=True).clamp_min(1e-12)
    barycentric = (plan / row_sum) @ b_s
    cost = ground_cost.pairwise_cost(a_s, b_s)
    ot_cost = (plan * cost).sum() / a_s.shape[0]
    fa = ground_cost.features(a_s) * ground_cost.feature_weights
    fb = ground_cost.features(barycentric) * ground_cost.feature_weights
    map_cost = ((fa - fb).square().sum(dim=1)).mean()
    return torch.clamp(map_cost - ot_cost, min=0.0)
