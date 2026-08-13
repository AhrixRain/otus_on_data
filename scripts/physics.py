"""Shared, numerically stable invariant-mass helpers for the dilepton pipeline.

The naive definition

    m^2 = E^2 - px^2 - py^2 - pz^2

subtracts two O(TeV^2) quantities to recover an O(GeV^2) invariant mass. For
boosted, low-mass pairs stored in float32 this is catastrophically unstable:
for daughters with |p| ~ 7 TeV the float32 rounding of E and p is already
larger than m^2 itself, so the direct formula produces wrong (or zero) masses.
The pipeline uses 10 MeV J/psi bins, so this matters for the reported metrics.

This module implements a cancellation-free decomposition instead. For two
daughters with transverse momentum pT_i, transverse mass
mT_i = sqrt(m_i^2 + pT_i^2), rapidity y_i = asinh(pz_i / mT_i) (which satisfies
E_i = mT_i cosh(y_i) and pz_i = mT_i sinh(y_i) exactly), azimuthal angle
phi_i, Delta_y = y_1 - y_2, and Delta_phi = phi_1 - phi_2:

    E1 E2 - p1 . p2 = mT1 mT2 cosh(Delta_y) - pT1 pT2 cos(Delta_phi)

so the pair invariant mass squared becomes

    m^2 = m1^2 + m2^2
          + 4 pT1 pT2 [sinh^2(Delta_y/2) + sin^2(Delta_phi/2)]
          + 2 (mT1 mT2 - pT1 pT2) cosh(Delta_y),

with the small transverse-mass correction evaluated in the cancellation-free
rationalized form

    mT1 mT2 - pT1 pT2 =
        (m1^2 m2^2 + m1^2 pT2^2 + m2^2 pT1^2) / (mT1 mT2 + pT1 pT2).

For m1 = m2 = 0 the rapidity reduces to pseudorapidity
(y_i = eta_i = asinh(pz_i / pT_i)) and the expression reduces to

    m^2 = 4 pT1 pT2 [sinh^2(Delta_eta/2) + sin^2(Delta_phi/2)],

the standard boosted massless-dilepton form. Every term above is a product of
non-negative O(1)-ish factors times pT1 pT2, so no large cancellation occurs.
sin^2(Delta_phi/2) is 2pi-periodic, so the azimuthal difference needs no
explicit wrapping.

The NumPy implementation always works in float64 and is exact for zero-pT legs
(it falls back to the direct float64 expression, which is numerically safe at
that precision). The Torch implementation stays in the caller's dtype (float32
on the MPS/CUDA training graph) and floors pT at ``eps`` so that pseudorapidity
and its gradients remain finite; for pT below ``eps`` the eta-based
decomposition is approximate. The physics selection keeps pT >= 2 GeV, so this
floor is never exercised by real data.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch


MUON_MASS_GEV = 0.1056583745
ELECTRON_MASS_GEV = 0.00051099895

CHANNEL_DAUGHTER_MASSES: dict[str, tuple[float, float]] = {
    "muon": (MUON_MASS_GEV, MUON_MASS_GEV),
    "electron": (ELECTRON_MASS_GEV, ELECTRON_MASS_GEV),
}


def validate_daughter_masses(
    daughter_masses: Sequence[float] | None,
) -> tuple[float, float] | None:
    """Return a validated ``(m1, m2)`` tuple, or ``None`` for the massless case."""
    if daughter_masses is None:
        return None
    if isinstance(daughter_masses, (str, bytes)) or len(daughter_masses) != 2:
        raise ValueError(
            f"daughter_masses must be a two-element sequence, got {daughter_masses!r}"
        )
    masses = (float(daughter_masses[0]), float(daughter_masses[1]))
    if not all(np.isfinite(mass) and mass >= 0.0 for mass in masses):
        raise ValueError(f"daughter_masses must be finite and non-negative, got {masses}")
    return masses


def daughter_masses_from_config(config: dict) -> tuple[float, float] | None:
    """Resolve daughter masses for a resolved config.

    Prefers an explicit ``model.daughter_masses`` value. Old configs without the
    field fall back to the physical masses for the configured channel; the model
    builder applies its own legacy-massless default for checkpoint
    compatibility, so this helper is only used for offline mass evaluation.
    """
    model = config.get("model") or {}
    masses = model.get("daughter_masses")
    if masses is not None:
        return validate_daughter_masses(masses)
    channel = str((config.get("data") or {}).get("channel", "electron"))
    normalized = channel.strip().lower().replace("-", "_")
    if normalized in {"muon", "doublemuon", "doublemuons", "mumu", "jpsi_mumu"}:
        return CHANNEL_DAUGHTER_MASSES["muon"]
    if normalized in {"electron", "doubleelectron", "ee"}:
        return CHANNEL_DAUGHTER_MASSES["electron"]
    return None


def _massless_mass2(
    px1: np.ndarray,
    py1: np.ndarray,
    pz1: np.ndarray,
    px2: np.ndarray,
    py2: np.ndarray,
    pz2: np.ndarray,
) -> np.ndarray:
    p1 = np.sqrt(px1**2 + py1**2 + pz1**2)
    p2 = np.sqrt(px2**2 + py2**2 + pz2**2)
    return 2.0 * (p1 * p2 - (px1 * px2 + py1 * py2 + pz1 * pz2))


def invariant_mass_np(
    pairs,
    *,
    daughter_masses: Sequence[float] | None = None,
    stable: bool = True,
) -> np.ndarray:
    """NumPy pair invariant mass for [N, 8] arrays ordered as [p4-, p4+].

    Computes in float64 regardless of the input dtype and returns float64.
    With ``stable=True`` (default) the cancellation-free transverse-mass /
    angular-difference decomposition is used and the stored energy columns are
    ignored. With ``stable=False`` the legacy direct ``E^2 - p^2`` expression
    is used in float64; this is provided only as a float64 four-vector
    reference for tests.

    ``daughter_masses`` selects the physical treatment:
      - ``None``: effectively massless daughters.
      - ``(m1, m2)``: massive daughters with the given masses in GeV.
    """
    values = np.asarray(pairs)
    if values.ndim != 2 or values.shape[1] != 8:
        raise ValueError(
            f"pairs must have shape [N, 8] ordered as [p4-, p4+], got {values.shape}"
        )
    masses = validate_daughter_masses(daughter_masses)
    work = values.astype(np.float64, copy=False)
    px1, py1, pz1 = work[:, 0], work[:, 1], work[:, 2]
    px2, py2, pz2 = work[:, 4], work[:, 5], work[:, 6]

    if not stable:
        p4 = work[:, 0:4] + work[:, 4:8]
        mass2 = p4[:, 3] ** 2 - (p4[:, 0] ** 2 + p4[:, 1] ** 2 + p4[:, 2] ** 2)
        return np.sqrt(np.maximum(mass2, 0.0))

    pt1 = np.hypot(px1, py1)
    pt2 = np.hypot(px2, py2)
    positive = (pt1 > 0.0) & (pt2 > 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        phi1 = np.arctan2(py1, px1)
        phi2 = np.arctan2(py2, px2)
        dphi = phi1 - phi2

    if masses is None:
        with np.errstate(divide="ignore", invalid="ignore"):
            eta1 = np.arcsinh(pz1 / np.where(positive, pt1, 1.0))
            eta2 = np.arcsinh(pz2 / np.where(positive, pt2, 1.0))
            dy = eta1 - eta2
            angular = 4.0 * pt1 * pt2 * (
                np.sinh(dy / 2.0) ** 2 + np.sin(dphi / 2.0) ** 2
            )
        mass2 = angular
    else:
        m1, m2 = masses
        m1sq = m1 * m1
        m2sq = m2 * m2
        pt1sq = pt1 * pt1
        pt2sq = pt2 * pt2
        mt1 = np.sqrt(m1sq + pt1sq)
        mt2 = np.sqrt(m2sq + pt2sq)
        with np.errstate(divide="ignore", invalid="ignore"):
            y1 = np.arcsinh(pz1 / np.where(mt1 > 0.0, mt1, 1.0))
            y2 = np.arcsinh(pz2 / np.where(mt2 > 0.0, mt2, 1.0))
            dy = y1 - y2
            angular = 4.0 * pt1 * pt2 * (
                np.sinh(dy / 2.0) ** 2 + np.sin(dphi / 2.0) ** 2
            )
        numerator = m1sq * m2sq + m1sq * pt2sq + m2sq * pt1sq
        denominator = mt1 * mt2 + pt1 * pt2
        with np.errstate(divide="ignore", invalid="ignore"):
            correction = 2.0 * (numerator / denominator) * np.cosh(dy)
        mass2 = m1sq + m2sq + angular + correction

    if masses is None:
        direct = _massless_mass2(px1, py1, pz1, px2, py2, pz2)
    else:
        m1, m2 = masses
        e1 = np.sqrt(m1 * m1 + px1**2 + py1**2 + pz1**2)
        e2 = np.sqrt(m2 * m2 + px2**2 + py2**2 + pz2**2)
        direct = (
            m1 * m1
            + m2 * m2
            + 2.0 * (e1 * e2 - (px1 * px2 + py1 * py2 + pz1 * pz2))
        )
    mass2 = np.where(positive, mass2, direct)
    return np.sqrt(np.maximum(mass2, 0.0))


def invariant_mass_torch(
    pairs,
    *,
    daughter_masses: Sequence[float] | None = None,
    eps: float = 1e-8,
):
    """Differentiable pair invariant mass for [N, 8] torch tensors.

    Stays in the input dtype (float32 on the MPS/CUDA training graph; float64
    is never introduced). Uses the same cancellation-free decomposition as
    :func:`invariant_mass_np`. ``eps`` floors pT before the pseudorapidity
    computation so that zero-pT legs produce finite values and finite
    gradients; the floor is a documented approximation below ``eps``.

    ``daughter_masses`` mirrors the NumPy signature: ``None`` is massless,
    ``(m1, m2)`` is massive.
    """
    if not isinstance(pairs, torch.Tensor):
        pairs = torch.as_tensor(pairs)
    if pairs.ndim != 2 or pairs.shape[1] != 8:
        raise ValueError(
            f"pairs must have shape [N, 8] ordered as [p4-, p4+], got {tuple(pairs.shape)}"
        )
    masses = validate_daughter_masses(daughter_masses)
    if float(eps) <= 0.0:
        raise ValueError("eps must be positive")

    px1, py1, pz1 = pairs[:, 0], pairs[:, 1], pairs[:, 2]
    px2, py2, pz2 = pairs[:, 4], pairs[:, 5], pairs[:, 6]
    pt1 = torch.sqrt(px1 * px1 + py1 * py1)
    pt2 = torch.sqrt(px2 * px2 + py2 * py2)
    phi1 = torch.atan2(py1, px1)
    phi2 = torch.atan2(py2, px2)
    dphi = phi1 - phi2

    if masses is None:
        pt1_safe = torch.clamp(pt1, min=eps)
        pt2_safe = torch.clamp(pt2, min=eps)
        eta1 = torch.asinh(pz1 / pt1_safe)
        eta2 = torch.asinh(pz2 / pt2_safe)
        dy = eta1 - eta2
        angular = 4.0 * pt1_safe * pt2_safe * (
            torch.sinh(dy / 2.0) ** 2 + torch.sin(dphi / 2.0) ** 2
        )
        mass2 = angular
    else:
        m1 = torch.as_tensor(masses[0], dtype=pairs.dtype, device=pairs.device)
        m2 = torch.as_tensor(masses[1], dtype=pairs.dtype, device=pairs.device)
        m1sq = m1 * m1
        m2sq = m2 * m2
        pt1sq = pt1 * pt1
        pt2sq = pt2 * pt2
        mt1 = torch.sqrt(m1sq + pt1sq)
        mt2 = torch.sqrt(m2sq + pt2sq)
        mt1_safe = torch.clamp(mt1, min=eps)
        mt2_safe = torch.clamp(mt2, min=eps)
        y1 = torch.asinh(pz1 / mt1_safe)
        y2 = torch.asinh(pz2 / mt2_safe)
        dy = y1 - y2
        angular = 4.0 * pt1 * pt2 * (
            torch.sinh(dy / 2.0) ** 2 + torch.sin(dphi / 2.0) ** 2
        )
        numerator = m1sq * m2sq + m1sq * pt2sq + m2sq * pt1sq
        denominator = torch.clamp(mt1 * mt2 + pt1 * pt2, min=eps)
        correction = 2.0 * (numerator / denominator) * torch.cosh(dy)
        mass2 = m1sq + m2sq + angular + correction
    return torch.sqrt(torch.clamp(mass2, min=0.0))
