#!/usr/bin/env python
"""Bidirectional minibatch-OT flow matching / stochastic interpolant (Run F1+F2).

The model learns two cylindrical velocity fields:

  * decoder velocity ``v_x(t, u_t | z, eps)``, z -> x
  * encoder velocity ``v_z(t, u_t | x, eps)``, x -> z

Training pairs come from an entropic OT plan between the current z and x
minibatches (decoder uses P, encoder uses P^T), so no per-event truth pairs are
required.  Each path is a noisy linear stochastic interpolant

    u_t = (1-t) z_u + t x_u + gamma_t eps
    gamma_t = sigma sqrt(t(1-t))

on [logpT-, eta-, phi-, logpT+, eta+, phi+].  Phi is transported on S^1; eps
enters the four non-azimuthal coordinates.  Inference integrates the learned
velocity with Euler steps, conditioned on the starting event and a sampled eps.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

try:
    from .ot import CrossDomainPhysicsGroundCost, entropic_ot_plan
except ImportError:  # scripts_sota/ on sys.path directly
    from ot import CrossDomainPhysicsGroundCost, entropic_ot_plan

from physics import validate_daughter_masses

try:
    from .cylindrical_flow import (
        condition_stats_np,
        cylindrical_to_p4,
        p4_to_cylindrical,
    )
except ImportError:
    from cylindrical_flow import (
        condition_stats_np,
        cylindrical_to_p4,
        p4_to_cylindrical,
    )


_EPS = 1e-8
_NONPHI = [0, 1, 3, 4]
_PHI = [2, 5]


def _wrapped_phi(u: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [
            u[:, :2],
            torch.atan2(torch.sin(u[:, 2:3]), torch.cos(u[:, 2:3])),
            u[:, 3:5],
            torch.atan2(torch.sin(u[:, 5:6]), torch.cos(u[:, 5:6])),
        ],
        dim=1,
    )


def interpolate_cylindrical(u0: torch.Tensor, u1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Linear interpolant in cylindrical coordinates; phi interpolates on S^1."""
    alpha = 1.0 - t
    beta = t
    linear = alpha * u0 + beta * u1
    phi_minus = torch.atan2(
        alpha * torch.sin(u0[:, 2:3]) + beta * torch.sin(u1[:, 2:3]),
        alpha * torch.cos(u0[:, 2:3]) + beta * torch.cos(u1[:, 2:3]),
    )
    phi_plus = torch.atan2(
        alpha * torch.sin(u0[:, 5:6]) + beta * torch.sin(u1[:, 5:6]),
        alpha * torch.cos(u0[:, 5:6]) + beta * torch.cos(u1[:, 5:6]),
    )
    return torch.cat([linear[:, :2], phi_minus, linear[:, 3:5], phi_plus], dim=1)


def cylindrical_difference(u0: torch.Tensor, u1: torch.Tensor) -> torch.Tensor:
    linear = u1 - u0
    phi_minus = torch.atan2(
        torch.sin(u1[:, 2:3] - u0[:, 2:3]),
        torch.cos(u1[:, 2:3] - u0[:, 2:3]),
    )
    phi_plus = torch.atan2(
        torch.sin(u1[:, 5:6] - u0[:, 5:6]),
        torch.cos(u1[:, 5:6] - u0[:, 5:6]),
    )
    return torch.cat([linear[:, :2], phi_minus, linear[:, 3:5], phi_plus], dim=1)


def make_velocity_network(input_dim: int, hidden_dims: Sequence[int], activation: type[nn.Module]) -> nn.Module:
    layers: list[nn.Module] = []
    previous = int(input_dim)
    for width in hidden_dims:
        layers.extend([nn.Linear(previous, int(width)), nn.LayerNorm(int(width)), activation()])
        previous = int(width)
    layers.append(nn.Linear(previous, 6))
    net = nn.Sequential(*layers)
    with torch.no_grad():
        net[-1].weight.mul_(1e-4)
        net[-1].bias.zero_()
    return net


class VelocityIntegrator(nn.Module):
    """One directional cylindrical velocity field + Euler sampler."""

    def __init__(self, condition_mean, condition_std, hidden_dims, activation, noise_dim: int, muon_mass: float):
        super().__init__()
        self.register_buffer("condition_mean", torch.as_tensor(condition_mean, dtype=torch.float32))
        self.register_buffer("condition_std", torch.as_tensor(condition_std, dtype=torch.float32))
        self.noise_dim = int(noise_dim)
        self.muon_mass = float(muon_mass)
        # u(6) + condition(14) + noise(noise_dim) + t(1)
        self.net = make_velocity_network(
            6 + int(self.condition_mean.numel()) + int(noise_dim) + 1,
            hidden_dims,
            activation,
        )

    def condition(self, values: torch.Tensor) -> torch.Tensor:
        try:
            from .ot import cylindrical_physics_features
        except ImportError:
            from ot import cylindrical_physics_features
        features = cylindrical_physics_features(
            values,
            daughter_masses=(self.muon_mass, self.muon_mass),
            eps=_EPS,
        )
        return (features - self.condition_mean) / self.condition_std

    def velocity(self, u: torch.Tensor, start: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        cond = self.condition(start)
        tt = t.reshape(-1, 1)
        return self.net(torch.cat([u, cond, eps, tt], dim=1))

    def sample(self, start: torch.Tensor, steps: int, eps: torch.Tensor | None = None) -> torch.Tensor:
        if eps is None:
            eps = torch.randn(len(start), self.noise_dim, dtype=start.dtype, device=start.device)
        u = p4_to_cylindrical(start, eps=_EPS)
        dt = 1.0 / max(1, int(steps))
        for step in range(int(steps)):
            t = torch.full((len(start), 1), step * dt, dtype=start.dtype, device=start.device)
            v = self.velocity(u, start, eps, t)
            u = _wrapped_phi(u + dt * v)
        return cylindrical_to_p4(u, self.muon_mass)


class BidirectionalFlowMatcher(nn.Module):
    def __init__(
        self,
        x_condition_stats,
        z_condition_stats,
        model_config: dict[str, Any],
        muon_mass: float,
        daughter_masses,
        x_train: np.ndarray,
        z_train: np.ndarray,
    ):
        super().__init__()
        self.muon_mass = float(muon_mass)
        self.daughter_masses = validate_daughter_masses(daughter_masses)
        hidden_dims = [int(x) for x in model_config["hidden_dims"]]
        activation = getattr(nn, model_config.get("activation", "SiLU"))
        self.noise_dim = int(model_config.get("noise_dim", 4))
        self.sigma = float(model_config.get("sigma", 0.10))
        self.steps = max(1, int(model_config.get("integration_steps", 8)))
        self.ot_regularization = float(model_config.get("ot_regularization", 0.05))
        self.ot_max_iter = int(model_config.get("ot_max_iter", 50))
        self.ot_max_batch = int(model_config.get("ot_max_batch", 512))
        self.encoder = VelocityIntegrator(
            x_condition_stats[0], x_condition_stats[1], hidden_dims, activation, self.noise_dim, self.muon_mass
        )
        self.decoder = VelocityIntegrator(
            z_condition_stats[0], z_condition_stats[1], hidden_dims, activation, self.noise_dim, self.muon_mass
        )
        self.coupling_ground = CrossDomainPhysicsGroundCost(
            x_train,
            z_train,
            daughter_masses=daughter_masses,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.sample(x, self.steps)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder.sample(z, self.steps)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return z, self.decode(z)

    def _subsample(self, a, b):
        if a.shape[0] <= self.ot_max_batch and b.shape[0] <= self.ot_max_batch:
            return a, b
        generator = torch.Generator(device="cpu")
        ia = torch.randperm(a.shape[0], generator=generator)[: self.ot_max_batch]
        ib = torch.randperm(b.shape[0], generator=generator)[: self.ot_max_batch]
        return a[ia], b[ib]

    def flow_matching_loss(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        zs, xs = self._subsample(z, x)
        plan = entropic_ot_plan(
            self.coupling_ground, zs, xs,
            regularization=self.ot_regularization,
            max_iter=self.ot_max_iter,
        )
        # decoder pairs: z_i -> x_j ; encoder uses the transposed coupling
        idx_x = torch.multinomial(plan + 1e-12, 1).squeeze(1)
        x_target = xs[idx_x]
        loss = self._one_direction_loss(zs, x_target, self.decoder, zs, "forward")
        # encoder direction: x_j -> z_i from transposed plan
        plan_t = plan.t().contiguous()
        idx_z = torch.multinomial(plan_t + 1e-12, 1).squeeze(1)
        z_target = zs[idx_z]
        loss = loss + self._one_direction_loss(xs, z_target, self.encoder, xs, "backward")
        return 0.5 * loss

    def _one_direction_loss(self, start, target, integrator: VelocityIntegrator, condition_p4, direction: str):
        start_u = p4_to_cylindrical(start, eps=_EPS)
        target_u = p4_to_cylindrical(target, eps=_EPS)
        n = start.shape[0]
        t = torch.rand(n, 1, dtype=start.dtype, device=start.device).clamp(0.02, 0.98)
        eps = torch.randn(n, self.noise_dim, dtype=start.dtype, device=start.device)
        gamma = self.sigma * torch.sqrt(t * (1.0 - t) + 1e-6)
        u_t = interpolate_cylindrical(start_u, target_u, t)
        noisy_nonphi = u_t[:, _NONPHI] + gamma * eps
        u_t = torch.cat(
            [noisy_nonphi[:, :2], u_t[:, 2:3], noisy_nonphi[:, 2:4], u_t[:, 5:6]],
            dim=1,
        )
        v_target = cylindrical_difference(start_u, target_u)
        # Brownian-bridge style correction for the noisy non-phi coordinates.
        gamma_prime = self.sigma * (0.5 - t) / torch.sqrt(t * (1.0 - t) + 1e-6)
        residual = u_t[:, _NONPHI] - (
            (1.0 - t) * start_u[:, _NONPHI] + t * target_u[:, _NONPHI]
        )
        corrected_nonphi = v_target[:, _NONPHI] + (gamma_prime / gamma) * residual
        v_target = torch.cat(
            [corrected_nonphi[:, :2], v_target[:, 2:3], corrected_nonphi[:, 2:4], v_target[:, 5:6]],
            dim=1,
        )
        v_pred = integrator.velocity(u_t, condition_p4, eps, t)
        return ((v_pred - v_target) ** 2).mean()


def build_flow_matching_autoencoder(
    model_config: dict[str, Any],
    x_train: np.ndarray,
    z_train: np.ndarray,
    muon_mass: float,
    daughter_masses,
) -> BidirectionalFlowMatcher:
    x_stats = condition_stats_np(x_train, daughter_masses)
    z_stats = condition_stats_np(z_train, daughter_masses)
    return BidirectionalFlowMatcher(
        x_stats,
        z_stats,
        model_config,
        muon_mass,
        daughter_masses,
        x_train,
        z_train,
    )
