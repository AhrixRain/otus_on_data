"""SOTA extensions for the OTUS CMS dimuon project (Run E / Tier A).

The modules in this package are intentionally separate from ``scripts/`` so the
provenance pipeline and archived checkpoints are not disturbed while the Tier A
components are developed.

Tier A components:
  - ``ot``: physics-aware ground cost, Sinkhorn divergence, entropic Monge gap.
  - ``max_swd``: learned / adversarially maximized sliced-Wasserstein slicing.
  - ``cylindrical_flow``: stochastic residual flow in cylindrical coordinates.
  - ``evaluation``: C2ST, fixed-z stochasticity, pseudo-pair / cycle coverage.
  - ``selection``: mass-aware checkpoint gates.
"""

__all__ = [
    "ot",
    "max_swd",
    "cylindrical_flow",
    "evaluation",
    "selection",
]
