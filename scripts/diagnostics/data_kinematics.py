#!/usr/bin/env python
"""E1 diagnostics: CMS data-side kinematics (DoubleMuParked reduced ROOT).

Part A (trigger floor): streams the raw Muon_pt/Muon_eta branches for every
event and reports the pT distribution around the expected trigger legs, plus
the number of muons passing pT>2 / pT>3 with |eta|<2.4.

Part B (signal-region pairs): reuses scripts.cms_data.load_cms_x_data with
the given selection to produce charge-ordered OS pairs in the mass window,
then prints the same kinematic summary as scripts/prior_kinematics.py.

Usage:
  python scripts/data_kinematics.py \
      --root data/Run2012BC_DoubleMuParked_Muons.root \
      --pt-min 2.0 --eta-max 2.4 --mass-min 3.0369 --mass-max 3.1569 \
      --max-selected 1000000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.cms_data import load_cms_x_data  # noqa: E402
from scripts.prior_kinematics import analyze  # noqa: E402


def trigger_floor(root: Path) -> None:
    import awkward as ak
    import uproot

    edges = np.array(
        [0.0, 1.0, 2.0, 2.5, 2.8, 2.9, 3.0, 3.1, 3.2, 3.5, 4.0, 5.0, 10.0, 1000.0]
    )
    hist = np.zeros(len(edges) - 1, dtype=np.int64)
    total_muons = 0
    n_pt2_eta24 = 0
    n_pt3_eta24 = 0
    n_pt25_eta24 = 0
    n_pt35_eta24 = 0
    events = uproot.open(root)["Events"]
    for arrays in events.iterate(["Muon_pt", "Muon_eta"], step_size="200 MB", library="ak"):
        pt = ak.flatten(arrays["Muon_pt"])
        eta = ak.flatten(arrays["Muon_eta"])
        total_muons += len(pt)
        hist += np.histogram(ak.to_numpy(pt), bins=edges)[0].astype(np.int64)
        sel = ak.to_numpy(ak.mask(pt, np.abs(eta) < 2.4, valid_when=False))
        sel = sel[~np.isnan(sel)]
        n_pt2_eta24 += int((sel > 2.0).sum())
        n_pt25_eta24 += int((sel > 2.5).sum())
        n_pt3_eta24 += int((sel > 3.0).sum())
        n_pt35_eta24 += int((sel > 3.5).sum())
    print(f"\n=== Part A: raw muon pT spectrum ({root.name}) ===")
    print(f"  total stored muons: {total_muons}")
    for i in range(len(edges) - 1):
        print(f"  pT in [{edges[i]:6.1f}, {edges[i+1]:6.1f}): {hist[i]:>12d}")
    print(f"  muons with |eta|<2.4: pT>2: {n_pt2_eta24} | pT>2.5: {n_pt25_eta24} | pT>3: {n_pt3_eta24} | pT>3.5: {n_pt35_eta24}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pt-min", type=float, default=2.0)
    parser.add_argument("--eta-max", type=float, default=2.4)
    parser.add_argument("--mass-min", type=float, required=True)
    parser.add_argument("--mass-max", type=float, required=True)
    parser.add_argument("--max-selected", type=int, default=1000000)
    parser.add_argument("--skip-trigger", action="store_true")
    args = parser.parse_args()

    if not args.skip_trigger:
        trigger_floor(args.root)

    selection = {
        "muon_pt_min": args.pt_min,
        "muon_abs_eta_max": args.eta_max,
        "jpsi_mass_min": args.mass_min,
        "jpsi_mass_max": args.mass_max,
    }
    print(f"\nLoading x pairs with selection {selection} (max {args.max_selected})...")
    x = load_cms_x_data(args.root, selection, channel="muon", max_selected=args.max_selected)
    analyze(
        x,
        f"CMS data OS pairs, mass in [{args.mass_min},{args.mass_max}] "
        f"(first {len(x)} of the file)",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
