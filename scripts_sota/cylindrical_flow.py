"""Stochastic residual flow in cylindrical coordinates for dimuon p4.

This is the Run E model family.  It keeps Virat's physically bounded residual
parameterization but promotes it from a one-shot stochastic residual MLP to a
short *flow*: ``flow_steps`` conditional residual transforms applied in
sequence to ``u = [log pT-, eta-, phi-, log pT+, eta+, phi+]``.

Each step is conditioned on the current event kinematics through the same 14
physics features as ``scripts_sota.ot``.  Phi is updated through sin/cos so
the flow lives on S^1, and the four-momentum is rebuilt on the muon mass shell
after every step.  The noise model is a Gaussian core plus a Student-t tail
with configurable per-coordinate floors/scales, exactly the parameterization
that the reference notebook introduced; ``set_noise_multipliers`` lets a stage
schedule turn the stochasticity on progressively.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

try:
    from .ot import cylindrical_physics_features
except ImportError:  # scripts_sota/ added directly to sys.path
    from ot import cylindrical_physics_features
from physics import validate_daughter_masses


_EPS = 1e-8
_CYLINDRICAL_DIM = 6


def p4_to_cylindrical(values: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """Convert [N, 8] dimuon p4 to [N, 6] cylindrical coordinates."""
    if values.ndim != 2 or values.shape[1] != 8:
        raise ValueError(f"Expected [N, 8], got {tuple(values.shape)}")
    p1, p2 = values[:, 0:4], values[:, 4:8]
    pt1 = torch.sqrt(torch.clamp(p1[:, 0] ** 2 + p1[:, 1] ** 2, min=eps))
    pt2 = torch.sqrt(torch.clamp(p2[:, 0] ** 2 + p2[:, 1] ** 2, min=eps))
    eta1 = torch.asinh(p1[:, 2] / torch.clamp(pt1, min=eps))
    eta2 = torch.asinh(p2[:, 2] / torch.clamp(pt2, min=eps))
    phi1 = torch.atan2(p1[:, 1], p1[:, 0])
    phi2 = torch.atan2(p2[:, 1], p2[:, 0])
    return torch.stack(
        [torch.log(pt1), eta1, phi1, torch.log(pt2), eta2, phi2],
        dim=1,
    )


def cylindrical_to_p4(coordinates: torch.Tensor, muon_mass: float) -> torch.Tensor:
    """Rebuild on-shell dimuon p4 from [N, 6] cylindrical coordinates."""
    if coordinates.ndim != 2 or coordinates.shape[1] != _CYLINDRICAL_DIM:
        raise ValueError(f"Expected [N, 6], got {tuple(coordinates.shape)}")
    particles = []
    mass = float(muon_mass)
    for start in (0, 3):
        logpt = torch.clamp(coordinates[:, start], min=-7.0, max=9.0)
        eta = torch.clamp(coordinates[:, start + 1], min=-6.0, max=6.0)
        phi = torch.atan2(
            torch.sin(coordinates[:, start + 2]),
            torch.cos(coordinates[:, start + 2]),
        )
        pt = torch.exp(logpt)
        px = pt * torch.cos(phi)
        py = pt * torch.sin(phi)
        pz = pt * torch.sinh(eta)
        energy = torch.sqrt(
            torch.clamp(px * px + py * py + pz * pz + mass * mass, min=_EPS)
        )
        particles.append(torch.stack([px, py, pz, energy], dim=1))
    return torch.cat(particles, dim=1)


def condition_stats_np(train_p4: np.ndarray, daughter_masses) -> tuple[np.ndarray, np.ndarray]:
    """Train mean/std of the 14D physics condition features."""
    tensor = torch.as_tensor(train_p4, dtype=torch.float32)
    features = cylindrical_physics_features(tensor, daughter_masses=daughter_masses, eps=_EPS)
    mean = features.mean(dim=0).numpy().astype(np.float32)
    std = features.std(dim=0, unbiased=False).numpy().astype(np.float32)
    std = np.where(np.isfinite(std) & (std > 1e-7), std, 1.0).astype(np.float32)
    return mean, std


_SIGMA_FLOOR_SPEC_KEYS = frozenset(
    {
        "schema_version",
        "eta_bins",
        "logpt_a",
        "logpt_b",
        "power_c",
        "power_alpha",
        "power_pivot_gev",
        "linear_offset",
        "linear_slope",
        "knee_pt_gev",
        "sigma_cap",
        "pt_max_gev",
        "phi_a",
        "phi_b",
        "eta_a",
        "eta_b",
        "tail_ratio",
        "scale",
        "units",
        "form",
        "mode",
        "provenance",
    }
)


def _validate_sigma_floor_spec(spec: dict) -> dict:
    """Validate a physics resolution kernel spec and normalize defaults.

    The spec is a small eta-binned parameterisation of the per-muon noise
    amplitude in the model's cylindrical coordinates. Two forms are supported:

    * ``logpt_a`` / ``logpt_b``: ``sigma = scale * sqrt(a(eta)^2 + (b(eta)/pT)^2)``
      (a resolution that falls with pT), and
    * ``power_c`` / ``power_alpha``: ``sigma = scale * c(eta) * (pT/pivot)^alpha(eta)``
      (calibrated on CMS: the per-muon log-pT amplitude *increases* with pT).

    The spec is stored as plain Python attributes (never buffers) so that
    checkpoints written without a kernel keep an unchanged state dict.
    """
    if not isinstance(spec, dict):
        raise ValueError("core_sigma_floor_spec must be a mapping")
    unknown = sorted(set(spec) - _SIGMA_FLOOR_SPEC_KEYS)
    if unknown:
        raise ValueError(
            f"core_sigma_floor_spec has unsupported keys {unknown}; allowed: "
            f"{sorted(_SIGMA_FLOOR_SPEC_KEYS)}"
        )
    eta_bins = [float(value) for value in spec.get("eta_bins", [])]
    if len(eta_bins) < 2:
        raise ValueError("core_sigma_floor_spec.eta_bins needs at least two edges")
    if any(b <= a for a, b in zip(eta_bins, eta_bins[1:])):
        raise ValueError("core_sigma_floor_spec.eta_bins must be strictly increasing")
    n_bins = len(eta_bins) - 1
    normalized: dict = {
        "eta_bins": eta_bins,
        "tail_ratio": float(spec.get("tail_ratio", 0.0)),
        "scale": float(spec.get("scale", 1.0)),
        "sigma_cap": float(spec.get("sigma_cap", 0.25)),
        "pt_max_gev": float(spec.get("pt_max_gev", 200.0)),
    }
    if normalized["tail_ratio"] < 0.0:
        raise ValueError("core_sigma_floor_spec.tail_ratio must be non-negative")
    if not normalized["scale"] > 0.0:
        raise ValueError("core_sigma_floor_spec.scale must be positive")
    if not normalized["sigma_cap"] > 0.0:
        raise ValueError("core_sigma_floor_spec.sigma_cap must be positive")
    if not normalized["pt_max_gev"] > 0.0:
        raise ValueError("core_sigma_floor_spec.pt_max_gev must be positive")
    power_mode = "power_c" in spec or "power_alpha" in spec
    linear_mode = "linear_offset" in spec or "linear_slope" in spec
    if power_mode and linear_mode:
        raise ValueError(
            "core_sigma_floor_spec: use only one of power_c/power_alpha or "
            "linear_offset/linear_slope"
        )
    if linear_mode:
        if "linear_offset" not in spec or "linear_slope" not in spec:
            raise ValueError(
                "core_sigma_floor_spec.linear_offset and linear_slope must be "
                "provided together"
            )
        if "logpt_a" in spec or "logpt_b" in spec:
            raise ValueError(
                "core_sigma_floor_spec: use either logpt_a/logpt_b or the "
                "linear_offset/linear_slope form, not both"
            )
        for key in ("linear_offset", "linear_slope"):
            values = [float(value) for value in spec[key]]
            if len(values) != n_bins:
                raise ValueError(
                    f"core_sigma_floor_spec.{key} must have {n_bins} values"
                )
            if not bool(np.isfinite(values).all()):
                raise ValueError(f"core_sigma_floor_spec.{key} must be finite")
            if any(value < 0.0 for value in values):
                raise ValueError(f"core_sigma_floor_spec.{key} must be non-negative")
            normalized[key] = values
        knee = float(spec.get("knee_pt_gev", 0.0))
        if not bool(np.isfinite(knee)) or knee < 0.0:
            raise ValueError(
                "core_sigma_floor_spec.knee_pt_gev must be finite and non-negative"
            )
        normalized["knee_pt_gev"] = knee
        normalized["mode"] = "linear"
    elif power_mode:
        if "power_c" not in spec or "power_alpha" not in spec:
            raise ValueError(
                "core_sigma_floor_spec.power_c and power_alpha must be provided together"
            )
        if "logpt_a" in spec or "logpt_b" in spec:
            raise ValueError(
                "core_sigma_floor_spec: use either logpt_a/logpt_b or "
                "power_c/power_alpha, not both"
            )
        for key in ("power_c", "power_alpha"):
            values = [float(value) for value in spec[key]]
            if len(values) != n_bins:
                raise ValueError(
                    f"core_sigma_floor_spec.{key} must have {n_bins} values"
                )
            if not bool(np.isfinite(values).all()):
                raise ValueError(f"core_sigma_floor_spec.{key} must be finite")
            if key == "power_c" and any(value < 0.0 for value in values):
                raise ValueError("core_sigma_floor_spec.power_c must be non-negative")
            normalized[key] = values
        pivot = float(spec.get("power_pivot_gev", 10.0))
        if not pivot > 0.0:
            raise ValueError("core_sigma_floor_spec.power_pivot_gev must be positive")
        normalized["power_pivot_gev"] = pivot
        normalized["mode"] = "power_law"
    else:
        for key in ("logpt_a", "logpt_b"):
            values = [float(value) for value in spec.get(key, [])]
            if len(values) != n_bins:
                raise ValueError(f"core_sigma_floor_spec.{key} must have {n_bins} values")
            if any(value < 0.0 for value in values):
                raise ValueError(f"core_sigma_floor_spec.{key} must be non-negative")
            normalized[key] = values
        normalized["mode"] = "sqrt"
    for key in ("phi_a", "phi_b", "eta_a", "eta_b"):
        values = [float(value) for value in spec.get(key, [0.0] * n_bins)]
        if len(values) != n_bins:
            raise ValueError(f"core_sigma_floor_spec.{key} must have {n_bins} values")
        if any(value < 0.0 for value in values):
            raise ValueError(f"core_sigma_floor_spec.{key} must be non-negative")
        normalized[key] = values
    return normalized


class CylindricalFlowStep(nn.Module):
    """One conditional stochastic residual transform on cylindrical coords."""

    def __init__(
        self,
        condition_dim: int,
        hidden_dims: Sequence[int],
        activation: type[nn.Module],
        model_config: dict[str, Any],
    ):
        super().__init__()
        layers: list[nn.Module] = []
        previous = condition_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(previous, int(width)), nn.LayerNorm(int(width)), activation()])
            previous = int(width)
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(previous, 18)
        nn.init.normal_(self.head.weight, mean=0.0, std=1e-4)
        with torch.no_grad():
            self.head.bias[:6].zero_()
            self.head.bias[6:12].fill_(float(model_config["core_log_sigma_bias"]))
            self.head.bias[12:18].fill_(float(model_config["tail_log_sigma_bias"]))
        for name in ["mean_residual_limits", "core_sigma_floors", "core_sigma_scales", "tail_sigma_scales"]:
            values = torch.as_tensor(model_config[name], dtype=torch.float32)
            if values.numel() != _CYLINDRICAL_DIM:
                raise ValueError(f"{name} must contain six coordinate values")
            if name.endswith("floors") and bool((values < 0).any()):
                raise ValueError(f"{name} must be non-negative")
            self.register_buffer(name, values)
        # Optional tail floor, default OFF so every checkpoint written before
        # 2026-09-23 loads with an unchanged state dict. When enabled it is a
        # registered buffer and part of the checkpoint; when absent the
        # attribute is None and the forward pass adds nothing.
        tail_floors = model_config.get("tail_sigma_floors")
        if tail_floors is None:
            self.tail_sigma_floors = None
        else:
            values = torch.as_tensor(tail_floors, dtype=torch.float32)
            if values.numel() != _CYLINDRICAL_DIM:
                raise ValueError("tail_sigma_floors must contain six coordinate values")
            if bool((values < 0).any()):
                raise ValueError("tail_sigma_floors must be non-negative")
            self.register_buffer("tail_sigma_floors", values)
        self.student_t_df = float(model_config["student_t_degrees_of_freedom"])
        self.maximum_heavy_noise = float(model_config["maximum_heavy_noise"])
        if self.student_t_df <= 2.0:
            raise ValueError("Student-t degrees of freedom must exceed two")
        # Optional physics resolution kernel (A2.3). Default OFF; stored as
        # plain attributes so checkpoints written without it are unchanged.
        self.sigma_floor_spec = None
        self.freeze_noise_amplitude = bool(model_config.get("freeze_noise_amplitude", False))
        floor_spec = model_config.get("core_sigma_floor_spec")
        if floor_spec is not None:
            self.sigma_floor_spec = _validate_sigma_floor_spec(floor_spec)
        if self.freeze_noise_amplitude and self.sigma_floor_spec is None:
            raise ValueError(
                "freeze_noise_amplitude requires core_sigma_floor_spec "
                "(there is nothing to freeze otherwise)"
            )
        self.core_noise_multiplier = 1.0
        self.tail_noise_multiplier = 1.0

    def set_noise_multipliers(self, core: float, tail: float) -> None:
        self.core_noise_multiplier = float(core)
        self.tail_noise_multiplier = float(tail)

    def _sigma_floor(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Per-coordinate physics noise floor, (N, 6), from the kernel spec.

        Coordinates are [log pT-, eta-, phi-, log pT+, eta+, phi+]. The log-pT
        channel is either ``scale * sqrt(a(eta)^2 + (b(eta)/pT)^2)`` or, in
        power-law mode, ``scale * c(eta) * (pT / pivot)^alpha(eta)``. Tails
        outside the eta table clamp to the last bin so an extrapolated event is
        still bounded.
        """
        spec = self.sigma_floor_spec
        edges = torch.as_tensor(
            spec["eta_bins"], dtype=coordinates.dtype, device=coordinates.device
        )
        scale = float(spec["scale"])
        mode = spec.get("mode", "sqrt")

        def table(name: str, index: torch.Tensor) -> torch.Tensor:
            values = torch.as_tensor(
                spec[name], dtype=coordinates.dtype, device=coordinates.device
            )
            return values[index]

        columns = []
        sigma_cap = float(spec["sigma_cap"])
        pt_max = float(spec["pt_max_gev"])
        for start in (0, 3):
            eta = coordinates[:, start + 1]
            logpt = torch.clamp(coordinates[:, start], min=-7.0, max=9.0)
            # Clamp pT before evaluating the kernel: the amplitude laws grow
            # without bound, and an unclamped high-pT tail event would get an
            # arbitrarily large noise amplitude and blow up the loss.
            pt = torch.clamp(torch.exp(logpt), max=pt_max)
            if mode == "power_law":
                n_bins = len(spec["power_c"])
            elif mode == "linear":
                n_bins = len(spec["linear_offset"])
            else:
                n_bins = len(spec["logpt_a"])
            index = torch.clamp(
                torch.bucketize(eta.abs(), edges, right=False) - 1,
                min=0,
                max=n_bins - 1,
            )
            if mode == "power_law":
                c = table("power_c", index)
                alpha = table("power_alpha", index)
                sigma = scale * c * (pt / float(spec["power_pivot_gev"])) ** alpha
            elif mode == "linear":
                offset = table("linear_offset", index)
                slope = table("linear_slope", index)
                # Optional knee: the per-muon log-pT resolution is pT-flat below
                # knee_pt_gev (multiple-scattering floor) and rises linearly
                # above it. Needed because a pure offset+slope law cannot
                # satisfy J/psi (median muon pT 13.1 GeV), Z (41.0) and
                # Upsilon (4.7) at once (I2, 2026-10-04).
                knee = float(spec.get("knee_pt_gev", 0.0))
                sigma = scale * (
                    offset + slope * torch.clamp(pt - knee, min=0.0)
                )
            else:
                a = table("logpt_a", index)
                b = table("logpt_b", index)
                sigma = scale * torch.sqrt(
                    torch.clamp(a * a + (b / pt) ** 2, min=1e-12)
                )
            columns.append(torch.clamp(sigma, max=sigma_cap))
            eta_a = table("eta_a", index)
            eta_b = table("eta_b", index)
            columns.append(
                torch.clamp(
                    scale
                    * torch.sqrt(
                        torch.clamp(eta_a * eta_a + (eta_b / pt) ** 2, min=1e-12)
                    ),
                    max=sigma_cap,
                )
            )
            phi_a = table("phi_a", index)
            phi_b = table("phi_b", index)
            columns.append(
                torch.clamp(
                    scale
                    * torch.sqrt(
                        torch.clamp(phi_a * phi_a + (phi_b / pt) ** 2, min=1e-12)
                    ),
                    max=sigma_cap,
                )
            )
        return torch.stack(columns, dim=1)

    def forward(self, coordinates: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        response = self.head(self.backbone(condition))
        mean_raw, core_raw, tail_raw = response.split(6, dim=1)
        mean_delta = torch.tanh(mean_raw) * self.mean_residual_limits
        core_sigma = self.core_sigma_floors + F.softplus(core_raw) * self.core_sigma_scales
        tail_sigma = F.softplus(tail_raw) * self.tail_sigma_scales
        if self.tail_sigma_floors is not None:
            tail_sigma = tail_sigma + self.tail_sigma_floors
        if self.sigma_floor_spec is not None:
            physics_floor = self._sigma_floor(coordinates)
            if self.freeze_noise_amplitude:
                core_sigma = physics_floor
                tail_sigma = self.sigma_floor_spec["tail_ratio"] * physics_floor
            else:
                core_sigma = torch.maximum(core_sigma, physics_floor)

        core_noise = torch.randn_like(core_sigma)
        concentration = core_sigma.new_tensor(self.student_t_df / 2.0)
        rate = core_sigma.new_tensor(self.student_t_df / 2.0)
        inverse_scale = torch.distributions.Gamma(concentration, rate).rsample(core_sigma.shape)
        heavy_noise = torch.randn_like(tail_sigma) / torch.sqrt(inverse_scale.clamp_min(1e-5))
        heavy_noise = torch.clamp(
            heavy_noise,
            min=-self.maximum_heavy_noise,
            max=self.maximum_heavy_noise,
        )
        delta = (
            mean_delta
            + self.core_noise_multiplier * core_sigma * core_noise
            + self.tail_noise_multiplier * tail_sigma * heavy_noise
        )
        phi_new = coordinates[:, 2] + delta[:, 2]
        phi_new = torch.atan2(torch.sin(phi_new), torch.cos(phi_new))
        phi_new_p = coordinates[:, 5] + delta[:, 5]
        phi_new_p = torch.atan2(torch.sin(phi_new_p), torch.cos(phi_new_p))
        return torch.stack(
            [
                coordinates[:, 0] + delta[:, 0],
                coordinates[:, 1] + delta[:, 1],
                phi_new,
                coordinates[:, 3] + delta[:, 3],
                coordinates[:, 4] + delta[:, 4],
                phi_new_p,
            ],
            dim=1,
        )


class CylindricalResidualFlowMap(nn.Module):
    """Conditional cylindrical residual flow for one p4 space."""

    def __init__(
        self,
        condition_mean: Sequence[float] | np.ndarray,
        condition_std: Sequence[float] | np.ndarray,
        model_config: dict[str, Any],
        muon_mass: float,
        daughter_masses,
    ):
        super().__init__()
        self.muon_mass = float(muon_mass)
        self.daughter_masses = validate_daughter_masses(daughter_masses)
        self.register_buffer(
            "condition_mean",
            torch.as_tensor(condition_mean, dtype=torch.float32),
        )
        self.register_buffer(
            "condition_std",
            torch.as_tensor(condition_std, dtype=torch.float32),
        )
        activation = getattr(nn, model_config["activation"])
        hidden_dims = [int(value) for value in model_config["hidden_dims"]]
        flow_steps = max(1, int(model_config.get("flow_steps", 2)))
        self.flow_steps = flow_steps
        self.steps = nn.ModuleList(
            [
                CylindricalFlowStep(
                    int(self.condition_mean.numel()),
                    hidden_dims,
                    activation,
                    model_config,
                )
                for _ in range(flow_steps)
            ]
        )

    def set_noise_multipliers(self, core: float, tail: float) -> None:
        for step in self.steps:
            step.set_noise_multipliers(core, tail)

    def condition(self, values: torch.Tensor) -> torch.Tensor:
        features = cylindrical_physics_features(
            values,
            daughter_masses=self.daughter_masses,
            eps=_EPS,
        )
        return (features - self.condition_mean) / self.condition_std

    def forward(self, raw_input: torch.Tensor) -> torch.Tensor:
        coordinates = p4_to_cylindrical(raw_input, eps=_EPS)
        current = raw_input
        for step in self.steps:
            condition = self.condition(current)
            coordinates = step(coordinates, condition)
            current = cylindrical_to_p4(coordinates, self.muon_mass)
        return current


class CylindricalFlowAutoencoder(nn.Module):
    """Encoder/decoder pair made of cylindrical residual flow maps."""

    def __init__(
        self,
        x_condition_stats: tuple[np.ndarray, np.ndarray],
        z_condition_stats: tuple[np.ndarray, np.ndarray],
        model_config: dict[str, Any],
        muon_mass: float,
        daughter_masses,
    ):
        super().__init__()
        self.encoder = CylindricalResidualFlowMap(
            x_condition_stats[0],
            x_condition_stats[1],
            model_config,
            muon_mass,
            daughter_masses,
        )
        self.decoder = CylindricalResidualFlowMap(
            z_condition_stats[0],
            z_condition_stats[1],
            model_config,
            muon_mass,
            daughter_masses,
        )

    def set_noise_multipliers(self, core: float, tail: float) -> None:
        self.encoder.set_noise_multipliers(core, tail)
        self.decoder.set_noise_multipliers(core, tail)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return z, self.decode(z)


def build_cylindrical_flow_autoencoder(
    model_config: dict[str, Any],
    x_train: np.ndarray,
    z_train: np.ndarray,
    muon_mass: float,
    daughter_masses,
) -> CylindricalFlowAutoencoder:
    """Construct the model and derive condition statistics from train splits."""
    x_stats = condition_stats_np(x_train, daughter_masses)
    z_stats = condition_stats_np(z_train, daughter_masses)
    return CylindricalFlowAutoencoder(
        x_stats,
        z_stats,
        model_config,
        muon_mass,
        daughter_masses,
    )
