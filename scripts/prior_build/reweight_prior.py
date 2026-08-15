#!/usr/bin/env python
"""Reweight a prior HDF5 so its muon-pT spectrum matches a reference sample.

Per-event weight = prod_muons [ f_ref(pt) / f_prior(pt) ] with KDE densities
(scipy.stats.gaussian_kde on log-pT), capped at --max-weight. Weighted
sampling (no replacement limits beyond counts) converts to an unweighted
sample of --n events.

Usage:
  python scripts/reweight_prior.py --in prior.hdf5 --ref-cache <npz> \
      --out reweighted.hdf5 --n 80000 --max-weight 20 --seed 0
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.cms_data import load_theory_prior_z  # noqa: E402


def muon_pt(z: np.ndarray) -> np.ndarray:
    return np.concatenate([np.hypot(z[:, 0], z[:, 1]), np.hypot(z[:, 4], z[:, 5])])


def build_density(x: np.ndarray):
    """Binned log-pT density (histogram + light smoothing + interpolation).

    Returns a callable density(points_in_logpT) -> array. O(n) build, O(1)
    per-bin eval — the KDE route is intractable for million-scale references.
    """
    from scipy.ndimage import gaussian_filter1d
    x = np.asarray(x, dtype=np.float64)
    x = x[(x > 0) & np.isfinite(x)]
    logx = np.log(x)
    lo, hi = float(np.percentile(logx, 0.1)), float(np.percentile(logx, 99.9))
    bins = np.linspace(lo, hi, 401)
    counts, _ = np.histogram(logx, bins=bins)
    centers = 0.5 * (bins[:-1] + bins[1:])
    dens = gaussian_filter1d(counts.astype(np.float64), sigma=2.0)
    dens /= dens.sum() * (centers[1] - centers[0])

    def density(points: np.ndarray) -> np.ndarray:
        return np.interp(points, centers, dens, left=1e-12, right=1e-12)

    return density


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_file", type=Path, required=True)
    parser.add_argument("--ref-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--max-weight", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    z = load_theory_prior_z(args.in_file).astype(np.float64)
    with np.load(args.ref_cache, allow_pickle=False) as d:
        x = np.concatenate([np.asarray(d[k], dtype=np.float64) for k in ("x_train", "x_val", "x_test")])
    kde_ref = build_density(muon_pt(x))
    kde_prior = build_density(muon_pt(z))
    pt1 = np.hypot(z[:, 0], z[:, 1])
    pt2 = np.hypot(z[:, 4], z[:, 5])
    w = (kde_ref(np.log(pt1)) / kde_prior(np.log(pt1))) * (kde_ref(np.log(pt2)) / kde_prior(np.log(pt2)))
    w = np.clip(w, 0.0, args.max_weight)
    w = w / w.sum()
    rng = np.random.default_rng(args.seed)
    n_out = len(z) if args.n is None else int(args.n)
    idx = rng.choice(len(z), size=n_out, replace=True, p=w)
    z_out = z[idx].astype(np.float32)

    import h5py
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as h:
        grp = h.create_group("FDL")
        grp.create_dataset("zData", data=z_out, dtype=np.float32)
        for key, value in {
            "feature_names": '["mu_minus_px", "mu_minus_py", "mu_minus_pz", "mu_minus_E", "mu_plus_px", "mu_plus_py", "mu_plus_pz", "mu_plus_E"]',
            "units": "GeV",
            "particle_order": "mu- (PDG 13) first; mu+ (PDG -13) second",
            "source_file": args.in_file.name,
            "reweight_target": args.ref_cache.name,
            "reweight_max_weight": float(args.max_weight),
            "reweight_seed": int(args.seed),
            "reweight_mean_weight": float(np.mean(w * len(z))),
            "production_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }.items():
            grp.attrs[key] = value
    before = np.median(muon_pt(z))
    after = np.median(muon_pt(z_out))
    print(f"reweighted {len(z):,} -> {n_out:,} events | muon pT median {before:.2f} -> {after:.2f} GeV")
    print(f"mean weight (eff. sample factor) = {np.mean(w * len(z)):.2f} | wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
