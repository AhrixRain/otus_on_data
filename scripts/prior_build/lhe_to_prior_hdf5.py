#!/usr/bin/env python
"""MG5 LHE -> prior HDF5 post-processor (FDL/zData writer).

Self-contained LHE parser (no pylhe dependency). Extracts final-state
mu- (pdg 13) / mu+ (pdg -13) pairs, applies the trigger-equivalent selection
(both-muons pT > pt_min, |eta| < eta_max, pair mass in [mass_min, mass_max]),
and writes an HDF5 file compatible with scripts/cms_data.load_theory_prior_z:

  FDL/zData  shape [N, 8] float32
             columns = [mu- px, py, pz, E,  mu+ px, py, pz, E] (GeV)

The LHE energies are kept as stored (the model must have massive muons so
that E-based and p-based pair masses agree). Metadata attributes mirror the
original cms_jpsi_mumu_mg5_8tev_1M.hdf5 conventions.

Usage:
  python scripts/lhe_to_prior_hdf5.py --lhe events.lhe --out out.hdf5 \
      [--pt-min 3.0] [--eta-max 2.4] [--mass-min 3.0369] [--mass-max 3.1569] \
      [--label "p p > jpsiv j"] [--attr key=value ...]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.physics import invariant_mass_np  # noqa: E402

_MU_MASS = 0.105658
_EVENT_RE = re.compile(r"<event>\s*(.*?)</event>", re.S)


def parse_lhe_events(text: str) -> list[np.ndarray]:
    """Yield one [Npart, 13] float array per <event> block."""
    events: list[np.ndarray] = []
    for match in _EVENT_RE.finditer(text):
        lines = match.group(1).strip().splitlines()
        if not lines:
            continue
        rows = []
        for line in lines[1:]:  # first line is the event header
            parts = line.split()
            if len(parts) < 13:
                continue
            rows.append([float(p) for p in parts[:13]])
        if rows:
            events.append(np.asarray(rows, dtype=np.float64))
    return events


def select_dimuons(
    events: list[np.ndarray],
    pt_min: float,
    eta_max: float,
    mass_min: float,
    mass_max: float,
) -> tuple[np.ndarray, dict]:
    """Return charge-ordered [N, 8] rows and a cut-flow summary."""
    rows: list[np.ndarray] = []
    cut_flow = {"events": len(events), "exactly_one_pair": 0, "pt": 0, "eta": 0, "mass": 0}
    for ev in events:
        idup = ev[:, 0]
        istup = ev[:, 1]
        final = ev[(istup == 1) & (np.abs(idup) == 13)]
        if len(final) != 2 or np.sum(idup[(istup == 1) & (np.abs(idup) == 13)] == 13) != 1:
            continue
        cut_flow["exactly_one_pair"] += 1
        mu_minus = final[final[:, 0] == 13][0]
        mu_plus = final[final[:, 0] == -13][0]
        px, py, pz, e = mu_minus[6], mu_minus[7], mu_minus[8], mu_minus[9]
        px2, py2, pz2, e2 = mu_plus[6], mu_plus[7], mu_plus[8], mu_plus[9]
        pt1 = np.hypot(px, py)
        pt2 = np.hypot(px2, py2)
        if pt1 <= pt_min or pt2 <= pt_min:
            continue
        cut_flow["pt"] += 1
        eta1 = np.arcsinh(pz / max(pt1, 1e-9))
        eta2 = np.arcsinh(pz2 / max(pt2, 1e-9))
        if abs(eta1) >= eta_max or abs(eta2) >= eta_max:
            continue
        cut_flow["eta"] += 1
        pair = np.asarray([px, py, pz, e, px2, py2, pz2, e2], dtype=np.float64)
        mass = invariant_mass_np(pair[None, :], daughter_masses=(_MU_MASS, _MU_MASS), stable=True)[0]
        if not (mass_min < mass < mass_max):
            continue
        cut_flow["mass"] += 1
        rows.append(pair)
    if not rows:
        return np.empty((0, 8), dtype=np.float32), cut_flow
    return np.stack(rows).astype(np.float32), cut_flow


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lhe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pt-min", type=float, default=3.0)
    parser.add_argument("--eta-max", type=float, default=2.4)
    parser.add_argument("--mass-min", type=float, default=3.0369)
    parser.add_argument("--mass-max", type=float, default=3.1569)
    parser.add_argument("--label", default="")
    parser.add_argument("--attr", action="append", default=[], metavar="KEY=VALUE",
                        help="extra metadata attribute (repeatable)")
    args = parser.parse_args()

    text = args.lhe.read_text(encoding="utf-8", errors="replace")
    events = parse_lhe_events(text)
    z, cut_flow = select_dimuons(events, args.pt_min, args.eta_max, args.mass_min, args.mass_max)

    import h5py
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as h:
        grp = h.create_group("FDL")
        grp.create_dataset("zData", data=z, dtype=np.float32)
        attrs = {
            "feature_names": '["mu_minus_px", "mu_minus_py", "mu_minus_pz", "mu_minus_E", '
                             '"mu_plus_px", "mu_plus_py", "mu_plus_pz", "mu_plus_E"]',
            "units": "GeV",
            "particle_order": "mu- (PDG 13) first; mu+ (PDG -13) second",
            "lhe_file": args.lhe.name,
            "selection": json.dumps({
                "muon_pt_min": args.pt_min,
                "muon_abs_eta_max": args.eta_max,
                "mass_min": args.mass_min,
                "mass_max": args.mass_max,
            }),
            "cut_flow": json.dumps(cut_flow),
            "n_selected": int(len(z)),
            "daughter_muon_mass_GeV": _MU_MASS,
            "production_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "label": args.label,
        }
        for item in args.attr:
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            attrs[key] = value
        for key, value in attrs.items():
            grp.attrs[key] = value

    print(f"LHE events: {cut_flow['events']:,}")
    print(f"  exactly one mu-/mu+ pair: {cut_flow['exactly_one_pair']:,}")
    print(f"  pT > {args.pt_min}: {cut_flow['pt']:,}")
    print(f"  |eta| < {args.eta_max}: {cut_flow['eta']:,}")
    print(f"  mass in ({args.mass_min},{args.mass_max}): {cut_flow['mass']:,}")
    print(f"Wrote {len(z):,} selected events -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
