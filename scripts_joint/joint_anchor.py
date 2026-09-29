"""Frozen mean-map anchor for the joint stochastic stages.

The invariant-mass contract (``run_joint.assert_no_mass_anchor``) forbids the
physical mass from being a model input or a loss anchor. This module provides
the complementary contract: the *deterministic mean map* learned in the first
(noise-free) stage is frozen, and later stages are penalised for moving it.

Why: the marginal objective constrains only the decoded distribution, so the
split between "spread in f(z)" and "spread in the noise" is unidentifiable.
The Run H audit (2026-09-11) measured that the optimiser takes the cheap
solution -- it shrinks ``core_sigma`` to its floor and ``tail_sigma`` by ~5.7x,
moves the detector resolution into the deterministic mean map, and then drifts
that mean map late in stage 3, shifting all three Upsilon peaks down by
~-1.4%. The anchor bounds exactly that drift.

Design rules (the contract):

* the anchor is the *deterministic* map, evaluated with every noise multiplier
  temporarily set to zero, on a fixed, deterministic reference batch;
* the stochastic core/tail channels stay **trainable and switched on** in both
  directions -- the anchor never zeroes a stage's noise schedule;
* the reference is captured from the declared ``reference_stage`` (the
  noiseless warmup) at the moment that stage finishes;
* the penalty is a per-coordinate variance-normalised MSE, so its weight is
  comparable across regions and four-momentum components.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

try:  # scripts_joint/ added directly to sys.path
    from .joint_metrics import (
        current_noise_multipliers,
        set_noise_multipliers,
    )
except ImportError:
    from joint_metrics import (
        current_noise_multipliers,
        set_noise_multipliers,
    )


NOISE_KEYS = ("encoder_core", "encoder_tail", "decoder_core", "decoder_tail")
_ZERO_NOISE = {key: 0.0 for key in NOISE_KEYS}


def _first(value):
    return value[0] if isinstance(value, (tuple, list)) else value


class MeanMapAnchor:
    """Frozen deterministic encoder/decoder map, penalised in later stages."""

    def __init__(self, events_per_region: int = 256, epsilon: float = 1e-6):
        if int(events_per_region) < 2:
            raise ValueError("mean-map anchor needs at least two events per region")
        self.events_per_region = int(events_per_region)
        self.epsilon = float(epsilon)
        self.inputs: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None
        self.reference: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None
        self.scale: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None
        self.reference_stage: str | None = None

    # -- reference construction -------------------------------------------------
    def build_inputs(self, region_arrays: dict[str, dict[str, np.ndarray]], region_order, device: torch.device) -> None:
        """Pick a deterministic, evenly spaced reference batch per region.

        Evenly spaced indices (not the first N) avoid a file-ordering bias and
        keep the reference identical across restarts.
        """
        inputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for name in region_order:
            arrays = region_arrays[name]
            x_values = np.asarray(arrays["x_train"])
            z_values = np.asarray(arrays["z_train"])
            if len(x_values) < 2 or len(z_values) < 2:
                raise ValueError(f"Region {name!r} needs at least two training events")
            x_count = min(self.events_per_region, len(x_values))
            z_count = min(self.events_per_region, len(z_values))
            x_index = np.linspace(0, len(x_values) - 1, x_count).astype(np.int64)
            z_index = np.linspace(0, len(z_values) - 1, z_count).astype(np.int64)
            inputs[name] = (
                torch.as_tensor(np.ascontiguousarray(x_values[x_index]), dtype=torch.float32, device=device),
                torch.as_tensor(np.ascontiguousarray(z_values[z_index]), dtype=torch.float32, device=device),
            )
        self.inputs = inputs

    @property
    def ready(self) -> bool:
        return self.reference is not None and self.inputs is not None

    @torch.no_grad()
    def capture(self, model, reference_stage: str) -> dict[str, float]:
        """Freeze the current deterministic map as the anchor reference."""
        if self.inputs is None:
            raise RuntimeError("MeanMapAnchor.build_inputs must be called before capture")
        was_training = model.training
        previous_noise = current_noise_multipliers(model)
        model.eval()
        set_noise_multipliers(model, _ZERO_NOISE)
        try:
            reference: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            scale: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            for name, (x_ref, z_ref) in self.inputs.items():
                encoded = _first(model.encode(x_ref)).detach()
                decoded = _first(model.decode(z_ref)).detach()
                reference[name] = (encoded.clone(), decoded.clone())
                scale[name] = (
                    self._per_coordinate_variance(encoded),
                    self._per_coordinate_variance(decoded),
                )
        finally:
            set_noise_multipliers(model, previous_noise)
            model.train(was_training)
        self.reference = reference
        self.scale = scale
        self.reference_stage = str(reference_stage)
        return {
            "events_per_region": float(self.events_per_region),
            "regions": float(len(reference)),
        }

    def _per_coordinate_variance(self, values: torch.Tensor) -> torch.Tensor:
        variance = values.var(dim=0, unbiased=False)
        return torch.clamp(variance, min=self.epsilon)

    # -- penalty ----------------------------------------------------------------
    def loss(self, model) -> tuple[torch.Tensor, dict[str, float]]:
        """Variance-normalised MSE of the current deterministic map vs reference."""
        if not self.ready:
            raise RuntimeError(
                "MeanMapAnchor.loss called before the reference was captured; "
                "check mean_map_anchor.reference_stage"
            )
        previous_noise = current_noise_multipliers(model)
        set_noise_multipliers(model, _ZERO_NOISE)
        try:
            total = None
            terms: dict[str, float] = {}
            for name, (x_ref, z_ref) in self.inputs.items():
                encoded = _first(model.encode(x_ref))
                decoded = _first(model.decode(z_ref))
                enc_ref, dec_ref = self.reference[name]
                enc_scale, dec_scale = self.scale[name]
                encoder_term = self._normalized_mse(encoded, enc_ref, enc_scale)
                decoder_term = self._normalized_mse(decoded, dec_ref, dec_scale)
                region_total = encoder_term + decoder_term
                total = region_total if total is None else total + region_total
                terms[f"{name}_encoder"] = float(encoder_term.detach().cpu())
                terms[f"{name}_decoder"] = float(decoder_term.detach().cpu())
        finally:
            set_noise_multipliers(model, previous_noise)
        if total is None:
            raise RuntimeError("MeanMapAnchor has no regions")
        return total, terms

    def _normalized_mse(self, current, reference, scale) -> torch.Tensor:
        residual = current - reference
        return torch.mean(residual * residual / scale)
