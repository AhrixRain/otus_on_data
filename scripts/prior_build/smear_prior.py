#!/usr/bin/env python
"""Apply detector-like pT smearing to a prior HDF5 (FDL/zData) and rewrite it.

Smearing model: each muon's pT is scaled by a lognormal-free Gaussian factor,
    pT' = pT * (1 + eps),   eps ~ N(0, a),
direction (eta, phi) unchanged; pz' = pT' * sinh(eta); E' = sqrt(p'^2 + m_mu^2).
The pair-mass width response is calibrated against the CMS data (std ~ 27.9 MeV
in the signal window) via --calibrate.

Usage:
  python scripts/prior_build/smear_prior.py --in in.hdf5 --out smeared.hdf5 [--a 0.013] [--seed 0]
  python scripts/prior_build/smear_prior.py --in in.hdf5 --calibrate [--target-std 0.0279]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.cms_data import load_theory_prior_z  # noqa: E402
from scripts.physics import invariant_mass_np  # noqa: E402

_MU_MASS = 0.105658


def smear_rows(z: np.ndarray, a: float, rng: np.random.Generator) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    for i in (0, 4):  # mu- and mu+ blocks
        px, py, pz = z[:, i], z[:, i + 1], z[:, i + 2]
        pt = np.hypot(px, py)
        eta = np.arcsinh(pz / np.maximum(pt, 1e-12))
        phi = np.arctan2(py, px)
        pt_s = pt * (1.0 + rng.normal(0.0, a, size=len(z)))
        px_s = pt_s * np.cos(phi)
        py_s = pt_s * np.sin(phi)
        pz_s = pt_s * np.sinh(eta)
        out[:, i] = px_s
        out[:, i + 1] = py_s
        out[:, i + 2] = pz_s
        out[:, i + 3] = np.sqrt(px_s**2 + py_s**2 + pz_s**2 + _MU_MASS**2)
    return out


def mass_std_for_a(z: np.ndarray, a: float, n_draws: int = 3) -> float:
    """Average pair-mass std (massive daughters, p-based) over n_draws at smearing a."""
    stds = []
    for seed in range(n_draws):
        rng = np.random.default_rng(seed)
        zs = smear_rows(z, a, rng)
        mass = invariant_mass_np(zs, daughter_masses=(_MU_MASS, _MU_MASS), stable=True)
        stds.append(float(mass.std()))
    return float(np.mean(stds))


def filter_trigger_matched(z: np.ndarray) -> np.ndarray:
    """Apply the trigger-equivalent selection used by filter_theory_prior."""
    pt1 = np.hypot(z[:, 0], z[:, 1])
    pt2 = np.hypot(z[:, 4], z[:, 5])
    eta1 = np.arcsinh(z[:, 2] / np.maximum(pt1, 1e-12))
    eta2 = np.arcsinh(z[:, 6] / np.maximum(pt2, 1e-12))
    mass = invariant_mass_np(z, daughter_masses=(_MU_MASS, _MU_MASS), stable=True)
    keep = ((pt1 > 3.0) & (pt2 > 3.0) & (np.abs(eta1) < 2.4) & (np.abs(eta2) < 2.4)
            & (mass > 3.0369) & (mass < 3.1569))
    return z[keep]


def calibrate(z: np.ndarray, target_std: float, tol: float = 5e-5) -> float:
    lo, hi = 0.0, 0.2
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        s = mass_std_for_a(z, mid)
        if s > target_std:
            hi = mid
        else:
            lo = mid
        if abs(s - target_std) < tol or (hi - lo) < 1e-6:
            return mid
    return 0.5 * (lo + hi)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_file", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--a", type=float, default=None, help="relative pT smearing sigma")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--target-std", type=float, default=0.0279, help="target mass std [GeV]")
    args = parser.parse_args()

    z = load_theory_prior_z(args.in_file).astype(np.float64)
    if args.calibrate or args.a is None:
        z_cal = filter_trigger_matched(z)
        if len(z_cal) < 50:
            raise SystemExit(f"only {len(z_cal)} events pass the trigger-matched filter; cannot calibrate")
        args.a = calibrate(z_cal, args.target_std)
        print(f"calibrated a = {args.a:.4f} on {len(z_cal):,} filtered events -> "
              f"mass std ~ {mass_std_for_a(z_cal, args.a):.4f} GeV (target {args.target_std})")
    if args.out is None:
        return 0

    rng = np.random.default_rng(args.seed)
    zs = smear_rows(z, args.a, rng).astype(np.float32)
    mass = invariant_mass_np(zs, daughter_masses=(_MU_MASS, _MU_MASS), stable=True)
    import h5py
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as h:
        grp = h.create_group("FDL")
        grp.create_dataset("zData", data=zs, dtype=np.float32)
        for key, value in {
            "feature_names": '["mu_minus_px", "mu_minus_py", "mu_minus_pz", "mu_minus_E", '
                             '"mu_plus_px", "mu_plus_py", "mu_plus_pz", "mu_plus_E"]',
            "units": "GeV",
            "particle_order": "mu- (PDG 13) first; mu+ (PDG -13) second",
            "source_file": args.in_file.name,
            "smearing_model": "pT' = pT * (1 + N(0, a)); eta/phi fixed; E = sqrt(p'^2 + m_mu^2)",
            "smearing_a": float(args.a),
            "smearing_seed": int(args.seed),
            "daughter_muon_mass_GeV": _MU_MASS,
            "smeared_mass_mean_GeV": float(mass.mean()),
            "smeared_mass_std_GeV": float(mass.std()),
            "production_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }.items():
            grp.attrs[key] = value
    print(f"wrote {len(zs):,} smeared events (mass mean {mass.mean():.5f}, "
          f"std {mass.std()*1000:.1f} MeV) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
