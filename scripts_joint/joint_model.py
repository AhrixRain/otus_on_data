"""Shared stochastic residual model without explicit parent-mass conditioning."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

from cylindrical_flow import CylindricalResidualFlowMap
from ot import cylindrical_physics_features
from physics import validate_daughter_masses


CONDITION_FEATURES = {
    "log_pt_minus": 0,
    "eta_minus": 1,
    "sin_phi_minus": 2,
    "cos_phi_minus": 3,
    "log_pt_plus": 4,
    "eta_plus": 5,
    "sin_phi_plus": 6,
    "cos_phi_plus": 7,
    "log_pair_mass": 8,
    "log_pair_pt": 9,
    "pair_rapidity": 10,
    "cos_delta_phi": 11,
    "sin_delta_phi": 12,
    "delta_eta": 13,
}


def resolve_condition_indices(spec: Sequence[str | int] | None) -> tuple[int, ...]:
    if spec is None:
        # Run A deliberately excludes the explicit parent-mass coordinate. The
        # response is conditioned on muon kinematics, not a named resonance or
        # fixed mass anchor. The kinematics still contain physical mass
        # information implicitly; this is not claimed to erase it.
        return tuple(index for index in range(14) if index != CONDITION_FEATURES["log_pair_mass"])
    indices = []
    for value in spec:
        index = CONDITION_FEATURES[value] if isinstance(value, str) else int(value)
        if not 0 <= index < 14:
            raise ValueError(f"Condition feature index outside [0, 13]: {index}")
        indices.append(index)
    if len(set(indices)) != len(indices) or not indices:
        raise ValueError("condition_features must be non-empty and contain no duplicates")
    return tuple(indices)


def _balanced_condition_stats(
    arrays_by_region: dict[str, np.ndarray],
    *,
    daughter_masses,
    mass_from_energy: bool,
    indices: tuple[int, ...],
    max_events_per_region: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Equal-region feature statistics, independent of region event counts."""
    feature_blocks = []
    for name, values in arrays_by_region.items():
        limit = len(values)
        if max_events_per_region is not None:
            limit = min(limit, int(max_events_per_region))
        if limit < 2:
            raise ValueError(f"Region {name!r} needs at least two training events")
        features = cylindrical_physics_features(
            torch.as_tensor(values[:limit], dtype=torch.float32),
            daughter_masses=daughter_masses,
            mass_from_energy=mass_from_energy,
        )[:, indices]
        feature_blocks.append(features)
    # Every block contributes equally even if the underlying datasets differ
    # by orders of magnitude.
    means = torch.stack([block.mean(dim=0) for block in feature_blocks])
    second_moments = torch.stack([(block * block).mean(dim=0) for block in feature_blocks])
    mean = means.mean(dim=0)
    variance = torch.clamp(second_moments.mean(dim=0) - mean * mean, min=1e-8)
    std = torch.sqrt(variance)
    return mean.numpy().astype(np.float32), std.numpy().astype(np.float32)


class JointResidualFlowMap(CylindricalResidualFlowMap):
    """Cylindrical residual map with a fixed, auditable condition mask."""

    def __init__(
        self,
        condition_mean: np.ndarray,
        condition_std: np.ndarray,
        condition_indices: tuple[int, ...],
        model_config: dict[str, Any],
        muon_mass: float,
        daughter_masses,
        *,
        mass_from_energy: bool,
    ):
        super().__init__(
            condition_mean,
            condition_std,
            model_config,
            muon_mass,
            daughter_masses,
        )
        self.register_buffer(
            "condition_indices",
            torch.as_tensor(condition_indices, dtype=torch.long),
        )
        self.mass_from_energy = bool(mass_from_energy)

    def condition(self, values: torch.Tensor) -> torch.Tensor:
        features = cylindrical_physics_features(
            values,
            daughter_masses=self.daughter_masses,
            mass_from_energy=self.mass_from_energy,
        ).index_select(1, self.condition_indices)
        return (features - self.condition_mean) / self.condition_std


class JointDimuonAutoencoder(nn.Module):
    """One encoder and one decoder shared by every configured mass region."""

    def __init__(
        self,
        x_stats: tuple[np.ndarray, np.ndarray],
        z_stats: tuple[np.ndarray, np.ndarray],
        condition_indices: tuple[int, ...],
        model_config: dict[str, Any],
        muon_mass: float,
        daughter_masses,
    ):
        super().__init__()
        self.encoder = JointResidualFlowMap(
            *x_stats,
            condition_indices,
            model_config,
            muon_mass,
            daughter_masses,
            mass_from_energy=False,
        )
        self.decoder = JointResidualFlowMap(
            *z_stats,
            condition_indices,
            model_config,
            muon_mass,
            daughter_masses,
            mass_from_energy=True,
        )

    def set_noise_multipliers(self, core: float, tail: float) -> None:
        """Set the same noise multipliers on both components.

        Kept as the back-compatible entry point: checkpoints written before
        2026-09-04 record a single pair of multipliers, and
        ``restore_joint_checkpoint`` still calls this.
        """
        self.set_component_noise_multipliers(
            encoder_core=core,
            encoder_tail=tail,
            decoder_core=core,
            decoder_tail=tail,
        )

    def set_component_noise_multipliers(
        self,
        *,
        encoder_core: float,
        encoder_tail: float,
        decoder_core: float,
        decoder_tail: float,
    ) -> None:
        """Set encoder and decoder noise independently.

        Sharing one schedule is physically backwards for a detector response:
        the DECODER adds resolution (it must be stochastic to smear a truth
        four-vector into a measured one), while the ENCODER removes it. With
        both deterministic -- which is what stage 1 does, multipliers 0/0 --
        the decoder has no noise source to re-add width, so the cycle
        reconstruction term forbids the encoder from discarding mass width:
        the physically correct map is unreachable, not merely unfound. That is
        the leading explanation for the narrow-prior A/B failure
        (memory.md section 7.2, open question 8), and splitting these two
        schedules is the experiment that tests it.
        """
        self.encoder.set_noise_multipliers(encoder_core, encoder_tail)
        self.decoder.set_noise_multipliers(decoder_core, decoder_tail)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode(x)
        return encoded, self.decode(encoded)


def build_joint_autoencoder(
    model_config: dict[str, Any],
    region_arrays: dict[str, dict[str, np.ndarray]],
    muon_mass: float,
    daughter_masses,
) -> JointDimuonAutoencoder:
    masses = validate_daughter_masses(daughter_masses)
    indices = resolve_condition_indices(model_config.get("condition_features"))
    cap = model_config.get("condition_stats_max_events_per_region", 100000)
    x_stats = _balanced_condition_stats(
        {name: arrays["x_train"] for name, arrays in region_arrays.items()},
        daughter_masses=masses,
        mass_from_energy=False,
        indices=indices,
        max_events_per_region=cap,
    )
    z_stats = _balanced_condition_stats(
        {name: arrays["z_train"] for name, arrays in region_arrays.items()},
        daughter_masses=masses,
        mass_from_energy=True,
        indices=indices,
        max_events_per_region=cap,
    )
    return JointDimuonAutoencoder(
        x_stats,
        z_stats,
        indices,
        model_config,
        muon_mass,
        masses,
    )
