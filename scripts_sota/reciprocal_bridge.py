#!/usr/bin/env python
"""Run G0 reciprocal stochastic bridge prototype for unpaired x/z transport.

The Run F model learns two independent velocity fields.  This module instead
uses one shared field with a direction flag, a common cylindrical state space,
and an explicit time-reversal penalty.  Endpoint pairs are sampled from the
same physics-cost entropic-OT plan; no event-level x/z pairs are consumed.

The training objective is a simulation-free stochastic-interpolant bridge
objective with an auxiliary conditional-score head.  It is deliberately named
``reciprocal_bridge`` rather than claiming an exact implementation of any one
published Schrodinger-bridge algorithm.  Run G0 is the contract/prototype run;
method claims require the locked evaluations and ablations in ``run_g0.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

try:
    from .cylindrical_flow import cylindrical_to_p4, p4_to_cylindrical
    from .flow_matching import cylindrical_difference, interpolate_cylindrical
    from .ot import CrossDomainPhysicsGroundCost, cylindrical_physics_features, entropic_ot_plan
except ImportError:  # scripts_sota/ placed directly on sys.path
    from cylindrical_flow import cylindrical_to_p4, p4_to_cylindrical
    from flow_matching import cylindrical_difference, interpolate_cylindrical
    from ot import CrossDomainPhysicsGroundCost, cylindrical_physics_features, entropic_ot_plan

from physics import validate_daughter_masses


_EPS = 1e-8
_NONPHI = [0, 1, 3, 4]


def _wrapped_phi(values: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [
            values[:, :2],
            torch.atan2(torch.sin(values[:, 2:3]), torch.cos(values[:, 2:3])),
            values[:, 3:5],
            torch.atan2(torch.sin(values[:, 5:6]), torch.cos(values[:, 5:6])),
        ],
        dim=1,
    )


def _make_network(input_dim: int, hidden_dims: Sequence[int], activation: type[nn.Module]) -> nn.Module:
    layers: list[nn.Module] = []
    previous = int(input_dim)
    for width in hidden_dims:
        layers.extend([nn.Linear(previous, int(width)), nn.LayerNorm(int(width)), activation()])
        previous = int(width)
    # Six cylindrical velocities plus four non-azimuthal score components.
    layers.append(nn.Linear(previous, 10))
    network = nn.Sequential(*layers)
    with torch.no_grad():
        network[-1].weight.mul_(1e-4)
        network[-1].bias.zero_()
    return network


class ReciprocalBridgeField(nn.Module):
    """Shared time-dependent field used in both transport directions."""

    def __init__(
        self,
        condition_mean: np.ndarray,
        condition_std: np.ndarray,
        hidden_dims: Sequence[int],
        activation: type[nn.Module],
        noise_dim: int,
        muon_mass: float,
    ):
        super().__init__()
        if int(noise_dim) != 4:
            raise ValueError("reciprocal_bridge currently requires noise_dim=4")
        self.noise_dim = int(noise_dim)
        self.muon_mass = float(muon_mass)
        self.register_buffer("condition_mean", torch.as_tensor(condition_mean, dtype=torch.float32))
        self.register_buffer("condition_std", torch.as_tensor(condition_std, dtype=torch.float32))
        # state(6) + source condition(14) + eps(4) + local time(1) + direction(1)
        self.network = _make_network(26, hidden_dims, activation)

    def condition(self, p4: torch.Tensor, *, mass_from_energy: bool) -> torch.Tensor:
        features = cylindrical_physics_features(
            p4,
            daughter_masses=(self.muon_mass, self.muon_mass),
            eps=_EPS,
            mass_from_energy=mass_from_energy,
        )
        return (features - self.condition_mean) / self.condition_std

    def forward(
        self,
        state: torch.Tensor,
        source_p4: torch.Tensor,
        eps: torch.Tensor,
        local_time: torch.Tensor,
        direction: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        direction_column = torch.full_like(local_time.reshape(-1, 1), float(direction))
        inputs = torch.cat(
            [
                state,
                self.condition(source_p4, mass_from_energy=(direction > 0)),
                eps,
                local_time.reshape(-1, 1),
                direction_column,
            ],
            dim=1,
        )
        prediction = self.network(inputs)
        return prediction[:, :6], prediction[:, 6:]


class DirectionalBridgeIntegrator(nn.Module):
    """Thin sampling adapter around the shared reciprocal field."""

    def __init__(
        self,
        field: ReciprocalBridgeField,
        direction: float,
        steps: int,
        muon_mass: float,
    ):
        super().__init__()
        self.field = field
        self.direction = float(direction)
        self.steps = max(1, int(steps))
        self.muon_mass = float(muon_mass)

    def forward(self, source: torch.Tensor, eps: torch.Tensor | None = None) -> torch.Tensor:
        if eps is None:
            eps = torch.randn(
                len(source), self.field.noise_dim, dtype=source.dtype, device=source.device
            )
        state = p4_to_cylindrical(source, eps=_EPS)
        dt = 1.0 / self.steps
        # Midpoint integration is materially more stable than Run F's Euler
        # sampler while retaining a fixed, auditable number of evaluations.
        for step in range(self.steps):
            t0 = torch.full((len(source), 1), step * dt, dtype=source.dtype, device=source.device)
            velocity0, _ = self.field(state, source, eps, t0, self.direction)
            midpoint = _wrapped_phi(state + 0.5 * dt * velocity0)
            tm = t0 + 0.5 * dt
            velocity_mid, _ = self.field(midpoint, source, eps, tm, self.direction)
            state = _wrapped_phi(state + dt * velocity_mid)
        return cylindrical_to_p4(state, self.muon_mass)


class ReciprocalBridgeAutoencoder(nn.Module):
    """One shared stochastic bridge exposed through OTUS encode/decode APIs."""

    def __init__(
        self,
        model_config: dict[str, Any],
        x_train: np.ndarray,
        z_train: np.ndarray,
        muon_mass: float,
        daughter_masses,
    ):
        super().__init__()
        self.muon_mass = float(muon_mass)
        self.daughter_masses = validate_daughter_masses(daughter_masses)
        hidden_dims = [int(value) for value in model_config["hidden_dims"]]
        activation = getattr(nn, model_config.get("activation", "SiLU"))
        self.noise_dim = int(model_config.get("noise_dim", 4))
        self.sigma = float(model_config.get("sigma", 0.10))
        self.score_weight = float(model_config.get("score_weight", 0.10))
        self.reciprocity_weight = float(model_config.get("reciprocity_weight", 0.25))
        self.ot_regularization = float(model_config.get("ot_regularization", 0.05))
        self.ot_max_iter = int(model_config.get("ot_max_iter", 50))
        self.ot_max_batch = int(model_config.get("ot_max_batch", 512))
        self.integration_steps = max(1, int(model_config.get("integration_steps", 12)))

        self.coupling_ground = CrossDomainPhysicsGroundCost(
            x_train,
            z_train,
            daughter_masses=self.daughter_masses,
        )
        mean = self.coupling_ground.feature_mean.detach().cpu().numpy().astype(np.float32)
        std = self.coupling_ground.feature_std.detach().cpu().numpy().astype(np.float32)

        self.field = ReciprocalBridgeField(
            mean, std, hidden_dims, activation, self.noise_dim, self.muon_mass
        )
        # direction +1 is theory z -> detector x; -1 is x -> z.
        self.decoder = DirectionalBridgeIntegrator(
            self.field, +1.0, self.integration_steps, self.muon_mass
        )
        self.encoder = DirectionalBridgeIntegrator(
            self.field, -1.0, self.integration_steps, self.muon_mass
        )
        self.latest_bridge_components: dict[str, float] = {}

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return z, self.decode(z)

    def _subsample(self, x: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if len(x) <= self.ot_max_batch and len(z) <= self.ot_max_batch:
            return x, z
        # Device-local random indices avoid Run F's fixed CPU generator, which
        # repeatedly selected the same prefix-like minibatch subset.
        x_idx = torch.randperm(len(x), device=x.device)[: self.ot_max_batch]
        z_idx = torch.randperm(len(z), device=z.device)[: self.ot_max_batch]
        return x[x_idx], z[z_idx]

    def _direction_loss(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        direction: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        source_u = p4_to_cylindrical(source, eps=_EPS)
        target_u = p4_to_cylindrical(target, eps=_EPS)
        n = len(source)
        t = torch.rand(n, 1, dtype=source.dtype, device=source.device).clamp(0.02, 0.98)
        eps = torch.randn(n, self.noise_dim, dtype=source.dtype, device=source.device)
        gamma = self.sigma * torch.sqrt(t * (1.0 - t) + 1e-6)
        clean_state = interpolate_cylindrical(source_u, target_u, t)
        noisy_nonphi = clean_state[:, _NONPHI] + gamma * eps
        state = torch.cat(
            [
                noisy_nonphi[:, :2],
                clean_state[:, 2:3],
                noisy_nonphi[:, 2:],
                clean_state[:, 5:6],
            ],
            dim=1,
        )

        velocity_target = cylindrical_difference(source_u, target_u)
        gamma_prime = self.sigma * (0.5 - t) / torch.sqrt(t * (1.0 - t) + 1e-6)
        velocity_nonphi = velocity_target[:, _NONPHI] + gamma_prime * eps
        velocity_target = torch.cat(
            [
                velocity_nonphi[:, :2],
                velocity_target[:, 2:3],
                velocity_nonphi[:, 2:],
                velocity_target[:, 5:6],
            ],
            dim=1,
        )
        score_target = -eps / gamma.clamp_min(1e-4)
        velocity, score = self.field(state, source, eps, t, direction)
        flow_loss = torch.mean((velocity - velocity_target) ** 2)
        # gamma^2 weighting keeps score matching finite near bridge endpoints.
        score_loss = torch.mean((gamma * (score - score_target)) ** 2)
        return flow_loss, score_loss, clean_state, t

    def flow_matching_loss(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x_small, z_small = self._subsample(x, z)
        plan = entropic_ot_plan(
            self.coupling_ground,
            z_small,
            x_small,
            regularization=self.ot_regularization,
            max_iter=self.ot_max_iter,
        )
        x_indices = torch.multinomial(plan + 1e-12, 1).squeeze(1)
        x_target = x_small[x_indices]

        forward_flow, forward_score, clean_state, t = self._direction_loss(
            z_small, x_target, +1.0
        )
        reverse_flow, reverse_score, _, _ = self._direction_loss(
            x_target, z_small, -1.0
        )

        zeros = torch.zeros(
            len(z_small), self.noise_dim, dtype=z_small.dtype, device=z_small.device
        )
        forward_velocity, _ = self.field(clean_state, z_small, zeros, t, +1.0)
        reverse_velocity, _ = self.field(clean_state, x_target, zeros, 1.0 - t, -1.0)
        reciprocity = torch.mean((forward_velocity + reverse_velocity) ** 2)

        flow = 0.5 * (forward_flow + reverse_flow)
        score = 0.5 * (forward_score + reverse_score)
        total = flow + self.score_weight * score + self.reciprocity_weight * reciprocity
        self.latest_bridge_components = {
            "bridge_flow": float(flow.detach().cpu()),
            "bridge_score": float(score.detach().cpu()),
            "bridge_reciprocity": float(reciprocity.detach().cpu()),
        }
        return total


def build_reciprocal_bridge_autoencoder(
    model_config: dict[str, Any],
    x_train: np.ndarray,
    z_train: np.ndarray,
    muon_mass: float,
    daughter_masses,
) -> ReciprocalBridgeAutoencoder:
    return ReciprocalBridgeAutoencoder(
        model_config,
        x_train,
        z_train,
        muon_mass,
        daughter_masses,
    )
