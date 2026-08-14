#!/usr/bin/env python
"""E1 diagnostics: dump kinematics of an MG5 HDF5 theory prior.

Rows are [N, 8] = [mu- px, py, pz, E, mu+ px, py, pz, E] (FDL/zData or zData).
Reports per-event statistics for:
  - stored daughter energies vs |p| (are daughters massless in the file?)
  - pair invariant mass, both the stored-E direct formula (z-space authority)
    and the stable p-based formula (x-space convention)
  - pair pT (vector-sum magnitude), pair rapidity (E, pz based)
  - per-muon pT, |eta|, pz
and optionally applies the same filter as scripts/cms_data.filter_theory_prior
for a given YAML config (theory_prior_selection key) to report the pass rate
and the filtered prior's composition.

Usage:
  python scripts/prior_kinematics.py --file data/cms_jpsi_mumu_mg5_8tev_1M.hdf5
  python scripts/prior_kinematics.py --file data/cms_jpsi_mumu_mg5_8tev_1M.hdf5 \
      --config configs/cms_JpsiDoubleMuons_v3.9_F1_restricted.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.cms_data import filter_theory_prior, load_theory_prior_z  # noqa: E402
from scripts.physics import invariant_mass_np  # noqa: E402


def _stats(name: str, a: np.ndarray) -> None:
    a = np.asarray(a, dtype=np.float64)
    qs = np.percentile(a, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    print(
        f"  {name:24s} mean={a.mean():10.4f} std={a.std():10.4f} "
        f"min={qs[0]:9.4f} p01={qs[1]:9.4f} p05={qs[2]:9.4f} med={qs[4]:9.4f} "
        f"p95={qs[6]:9.4f} p99={qs[7]:9.4f} max={qs[8]:9.4f}"
    )


def analyze(z: np.ndarray, label: str) -> None:
    print(f"\n=== {label}: n={len(z)} ===")
    z = np.asarray(z, dtype=np.float64)
    px1, py1, pz1, e1 = z[:, 0], z[:, 1], z[:, 2], z[:, 3]
    px2, py2, pz2, e2 = z[:, 4], z[:, 5], z[:, 6], z[:, 7]

    pabs1 = np.sqrt(px1**2 + py1**2 + pz1**2)
    pabs2 = np.sqrt(px2**2 + py2**2 + pz2**2)
    _stats("E1/|p1| (stored)", e1 / np.where(pabs1 > 0, pabs1, 1.0))
    _stats("E2/|p2| (stored)", e2 / np.where(pabs2 > 0, pabs2, 1.0))

    # Pair mass: stored-E direct formula (z-space authority).
    pair_e = e1 + e2
    pair_px = px1 + px2
    pair_py = py1 + py2
    pair_pz = pz1 + pz2
    m2 = pair_e**2 - (pair_px**2 + pair_py**2 + pair_pz**2)
    mass_e = np.sqrt(np.maximum(m2, 0.0))
    _stats("pair mass [GeV] (E-based)", mass_e)
    mass_p = invariant_mass_np(z, daughter_masses=None, stable=True)
    _stats("pair mass [GeV] (p-based)", mass_p)

    pt1 = np.hypot(px1, py1)
    pt2 = np.hypot(px2, py2)
    pair_pt = np.hypot(pair_px, pair_py)
    _stats("muon pT [GeV]", np.concatenate([pt1, pt2]))
    _stats("pair pT [GeV] (vector sum)", pair_pt)
    _stats("pair pT [GeV] (scalar sum)", pt1 + pt2)

    pair_rap = 0.5 * np.log((pair_e + pair_pz) / np.where(pair_e - pair_pz > 0, pair_e - pair_pz, 1e-9))
    _stats("pair rapidity", pair_rap)

    eta1 = np.arcsinh(pz1 / np.where(pt1 > 0, pt1, 1.0))
    eta2 = np.arcsinh(pz2 / np.where(pt2 > 0, pt2, 1.0))
    eta = np.concatenate([eta1, eta2])
    _stats("muon eta", eta)
    _stats("muon |eta|", np.abs(eta))
    _stats("muon pz [GeV]", np.concatenate([pz1, pz2]))

    frac_eta24 = float((np.abs(eta) < 2.4).mean())
    frac_pt3 = float(((pt1 > 3.0) & (pt2 > 3.0)).mean())
    frac_pt2 = float(((pt1 > 2.0) & (pt2 > 2.0)).mean())
    frac_win = float(((mass_p > 3.0369) & (mass_p < 3.1569)).mean())
    frac_all = float(frac_pt3 and (np.abs(eta1) < 2.4).mean() and 0)
    print(
        f"  pass rates: both-muons pT>2: {frac_pt2:.3%} | pT>3: {frac_pt3:.3%} "
        f"| |eta|<2.4: {frac_eta24:.3%} | mass in [3.0369,3.1569]: {frac_win:.3%}"
    )
    keep = (pt1 > 3.0) & (pt2 > 3.0) & (np.abs(eta1) < 2.4) & (np.abs(eta2) < 2.4)
    keep &= (mass_p > 3.0369) & (mass_p < 3.1569)
    print(f"  combined v3.9 prior selection keeps {keep.sum()} / {len(z)} ({keep.mean():.3%})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None,
                        help="YAML config whose theory_prior_selection is applied (needs pyyaml)")
    args = parser.parse_args()

    z = load_theory_prior_z(args.file)
    analyze(z, f"{args.file.name} (raw)")

    if args.config is not None:
        import yaml
        with open(args.config) as fh:
            cfg = yaml.safe_load(fh)
        sel = cfg.get("theory_prior_selection")
        if not sel:
            print("\n(no theory_prior_selection in config)")
            return 0
        filtered = filter_theory_prior(z, sel)
        print(f"\nselection applied: {sel}")
        analyze(filtered, f"{args.file.name} (filtered: {len(filtered)} events)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
