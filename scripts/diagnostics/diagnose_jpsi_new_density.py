#!/usr/bin/env python
"""Diagnose the Jpsi_new prior paper-style run's mass / pT density failure.

Reads the already-produced plot arrays for a run and prints the evidence that
the coordinate-level losses are mass-blind, that D(z_prior) is extrapolating,
and that the prior has a low-pair-pT support hole.

Example:
    python scripts/diagnostics/diagnose_jpsi_new_density.py \
        --run-dir outputs/cms_Jpsi_new/Jpsi_newprior_paper_20pct
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MASS_LOW = 3.0369
MASS_HIGH = 3.1569
MUON_MASS = 0.105658


def pair_mass_direct(pairs: np.ndarray) -> np.ndarray:
    """E^2 - p^2 pair mass in float64, matching the plot script's z-space convention."""
    a = pairs.astype(np.float64)
    energy = a[:, 3] + a[:, 7]
    px = a[:, 0] + a[:, 4]
    py = a[:, 1] + a[:, 5]
    pz = a[:, 2] + a[:, 6]
    return np.sqrt(np.maximum(energy**2 - px**2 - py**2 - pz**2, 0.0))


def pair_pt(pairs: np.ndarray) -> np.ndarray:
    px = pairs[:, 0] + pairs[:, 4]
    py = pairs[:, 1] + pairs[:, 5]
    return np.sqrt(px**2 + py**2)


def angular_features(pairs: np.ndarray) -> dict[str, np.ndarray]:
    a = pairs.astype(np.float64)
    p1 = a[:, 0:3]
    p2 = a[:, 4:7]
    pt1 = np.hypot(a[:, 0], a[:, 1])
    pt2 = np.hypot(a[:, 4], a[:, 5])
    eta1 = np.arcsinh(a[:, 2] / pt1)
    eta2 = np.arcsinh(a[:, 6] / pt2)
    phi1 = np.arctan2(a[:, 1], a[:, 0])
    phi2 = np.arctan2(a[:, 5], a[:, 4])
    dphi = (phi1 - phi2 + np.pi) % (2.0 * np.pi) - np.pi
    deta = eta1 - eta2
    costh = np.sum(p1 * p2, axis=1) / (
        np.linalg.norm(p1, axis=1) * np.linalg.norm(p2, axis=1)
    )
    return {"deta": deta, "dphi": dphi, "costh": costh}


def ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    idx = np.searchsorted(a, b, side="right")
    return float(
        max(
            np.max(np.abs(idx / len(a) - (np.arange(1, len(b) + 1) / len(b)))),
            np.max(np.abs(idx / len(a) - (np.arange(len(b)) / len(b)))),
        )
    )


def max_projection_ks(a: np.ndarray, b: np.ndarray, num_slices: int = 1000, seed: int = 0) -> float:
    """Max KS over random unit projections; mirrors the SWD slicing family."""
    rng = np.random.default_rng(seed)
    if len(a) < len(b):
        b = b[rng.choice(len(b), len(a), replace=False)]
    elif len(b) < len(a):
        a = a[rng.choice(len(a), len(b), replace=False)]
    best = 0.0
    for _ in range(num_slices):
        theta = rng.normal(size=a.shape[1])
        theta /= np.linalg.norm(theta)
        best = max(best, ks_stat(a @ theta, b @ theta))
    return best


def frac_in_mass_window(mass: np.ndarray) -> float:
    return float(np.mean((mass >= MASS_LOW) & (mass <= MASS_HIGH)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "cms_Jpsi_new" / "Jpsi_newprior_paper_20pct",
    )
    args = parser.parse_args()
    npz_path = args.run_dir / "plots" / "paperstyle_loaded_model_outputs.npz"
    if not npz_path.exists():
        raise SystemExit(f"Missing plot arrays: {npz_path}")
    d = np.load(npz_path)
    x = d["x_plot"]
    z = d["z_plot"]
    ze = d["z_encoded"]
    xr = d["x_reco"]
    xz = d["x_from_z"]

    mz = pair_mass_direct(z)
    mze = pair_mass_direct(ze)
    mx = pair_mass_direct(x)
    mxr = pair_mass_direct(xr)
    mxz = pair_mass_direct(xz)

    print("== z-space: prior vs E(x) ==")
    print(f"  mass mean/std [GeV]:       {mz.mean():.4f} / {mz.std():.4f}   vs   {mze.mean():.4f} / {mze.std():.4f}")
    print(f"  mass-window fraction:      {frac_in_mass_window(mz):.3f}   vs   {frac_in_mass_window(mze):.4f}")
    print(f"  mass KS:                   {ks_stat(mz, mze):.4f}")
    print(f"  max per-component KS:      {max(ks_stat(z[:, i], ze[:, i]) for i in range(8)):.4f}")
    print(f"  max 1000-projection KS:    {max_projection_ks(z, ze):.4f}")

    af_z = angular_features(z)
    af_ze = angular_features(ze)
    print("  angular correlations (prior vs encoded):")
    for key in ("deta", "dphi", "costh"):
        print(
            f"    {key:5s} mean/std: {af_z[key].mean():+.4f}/{af_z[key].std():.4f}"
            f"   vs   {af_ze[key].mean():+.4f}/{af_ze[key].std():.4f}"
        )

    print("\n== x-space cycle: x vs D(E(x)) ==")
    print(f"  per-coordinate residual std [GeV]: {np.std(xr.astype(float) - x.astype(float), axis=0)}")
    print(f"  pair pT KS:                         {ks_stat(pair_pt(x), pair_pt(xr)):.4f}")
    print(f"  mass mean/std [GeV]:                {mx.mean():.4f} / {mx.std():.4f}   vs   {mxr.mean():.4f} / {mxr.std():.4f}")
    print(f"  mass-window fraction:               {frac_in_mass_window(mx):.4f}   vs   {frac_in_mass_window(mxr):.4f}")
    print(f"  mass KS:                            {ks_stat(mx, mxr):.4f}")

    print("\n== x-space generator: x vs D(z_prior) ==")
    print(f"  pair pT KS:                         {ks_stat(pair_pt(x), pair_pt(xz)):.4f}")
    print(f"  mass mean/std [GeV]:                {mx.mean():.4f} / {mx.std():.4f}   vs   {mxz.mean():.4f} / {mxz.std():.4f}")
    print(f"  mass-window fraction:               {frac_in_mass_window(mx):.4f}   vs   {frac_in_mass_window(mxz):.4f}")
    print(f"  mass KS:                            {ks_stat(mx, mxz):.4f}")

    print("\n== low pair-pT support ==")
    ptx = pair_pt(x)
    ptz = pair_pt(z)
    ptxz = pair_pt(xz)
    for threshold in (5.0, 9.7, 10.0):
        print(
            f"  frac pT<{threshold:4.1f} GeV: data {np.mean(ptx < threshold):.4f}"
            f"   prior {np.mean(ptz < threshold):.4f}"
            f"   D(prior) {np.mean(ptxz < threshold):.4f}"
        )
    print(f"  pair-pT minimum [GeV]:       data {ptx.min():.3f}   prior {ptz.min():.3f}   D(prior) {ptxz.min():.3f}")


if __name__ == "__main__":
    main()
