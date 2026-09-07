#!/usr/bin/env python
"""Print, save and compare the standard statistic block for a theory prior.

Why this exists
---------------
The 0j1j CKKW-L priors have no generation cards in version control
(memory.md section 9.1). When they are regenerated, "did we reproduce the
prior we have been training on?" has to be answered by numbers, not by
eyeball. This prints one canonical block for any prior HDF5 and can compare it
against a stored reference with explicit tolerances, exiting non-zero on a
mismatch so it can be used as a gate in a runbook.

Mass conventions
----------------
Two live in this repository and they do NOT agree on these files:

``stable``  (default here)   massive muons, momentum-based, stored energies
                             ignored. This is what ``build_ab_priors.py`` uses
                             and therefore what every A/B number and every
                             reference JSON shipped alongside this script is in.
``stored``                   stored energies authoritative, daughters massless.
                             This is what ``cms_data.filter_theory_prior`` uses
                             for its mass window.

Both are printed. Only the ``stable`` value is compared, because that is the
convention the references were built in. Do not mix them.

Usage
-----
    python scripts/prior_build/prior_stats.py --in data/<prior>.hdf5 --select jpsi
    python scripts/prior_build/prior_stats.py --in <new>.hdf5 --select jpsi \
        --compare scripts/prior_build/reference/jpsi_0j1j_reweighted_selected.json
    python scripts/prior_build/prior_stats.py --in <new>.hdf5 --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from scripts.cms_data import filter_theory_prior, load_theory_prior_z  # noqa: E402
from scripts.physics import invariant_mass_np  # noqa: E402

MUON_MASS = 0.105658

# The training selection each joint J/psi region applies to its prior. Kept
# here so a bare reproduction check does not need a run config.
SELECTIONS: dict[str, dict[str, float]] = {
    "jpsi": {
        "muon_pt_min": 3.0,
        "muon_abs_eta_max": 2.4,
        "mass_min": 3.0369,
        "mass_max": 3.1569,
    },
    "upsilon": {
        "muon_pt_min": 3.0,
        "muon_abs_eta_max": 2.4,
        "mass_min": 8.5,
        "mass_max": 11.5,
    },
    "z": {
        "muon_pt_min": 3.0,
        "muon_abs_eta_max": 2.4,
        "mass_min": 70.0,
        "mass_max": 110.0,
    },
}

# Absolute tolerances in the reference's own units; everything else is relative.
ABSOLUTE_TOLERANCE = {
    "mass_mean_gev": 5.0e-4,   # 0.5 MeV on the peak position
    "mass_std_mev": 0.5,       # 0.5 MeV on the width
}
RELATIVE_TOLERANCE = 0.02      # 2% on every median / percentile, matching
                               # the guard in build_ab_priors.py


def _pt(values: np.ndarray, first: int) -> np.ndarray:
    return np.hypot(values[:, first], values[:, first + 1])


def _abs_eta(values: np.ndarray, first: int) -> np.ndarray:
    momentum = np.sqrt(np.sum(values[:, first : first + 3] ** 2, axis=1))
    ratio = np.clip(values[:, first + 2] / np.maximum(momentum, 1e-12), -1 + 1e-7, 1 - 1e-7)
    return np.abs(np.arctanh(ratio))


def prior_stats(z: np.ndarray, *, nominal_mass: float | None = None) -> dict[str, Any]:
    z = np.asarray(z, dtype=np.float64)
    mass = invariant_mass_np(z, daughter_masses=(MUON_MASS, MUON_MASS), stable=True)
    mass_stored = invariant_mass_np(z, daughter_masses=None, stable=False)
    muon_pt = np.concatenate([_pt(z, 0), _pt(z, 4)])
    abs_eta = np.concatenate([_abs_eta(z, 0), _abs_eta(z, 4)])
    pair_pt = np.hypot(z[:, 0] + z[:, 4], z[:, 1] + z[:, 5])
    q75, q25 = np.percentile(mass, [75, 25])

    stats: dict[str, Any] = {
        "events": int(len(z)),
        "mass_mean_gev": float(mass.mean()),
        "mass_std_mev": float(mass.std() * 1000.0),
        "mass_median_gev": float(np.median(mass)),
        "mass_iqr_width_mev": float((q75 - q25) / 1.349 * 1000.0),
        "mass_std_stored_energy_mev": float(mass_stored.std() * 1000.0),
        "muon_pt_median_gev": float(np.median(muon_pt)),
        "muon_pt_p99_gev": float(np.percentile(muon_pt, 99)),
        "pair_pt_median_gev": float(np.median(pair_pt)),
        "pair_pt_p99_gev": float(np.percentile(pair_pt, 99)),
        "abs_eta_median": float(np.median(abs_eta)),
    }
    if nominal_mass is not None:
        for window_mev in (5.0, 10.0):
            inside = np.abs(mass - nominal_mass) < window_mev / 1000.0
            stats[f"frac_within_{int(window_mev)}mev"] = float(np.mean(inside))
    return stats


def compare(stats: dict[str, Any], reference: dict[str, Any]) -> list[str]:
    """Compare only the keys the reference actually carries."""
    failures = []
    for key, expected in reference.items():
        if key.startswith("_") or key not in stats:
            continue
        if not isinstance(expected, (int, float)):
            continue
        actual = float(stats[key])
        expected = float(expected)
        if key == "events":
            continue  # a reproduction check need not match the event count
        if key in ABSOLUTE_TOLERANCE:
            tolerance = ABSOLUTE_TOLERANCE[key]
            ok = abs(actual - expected) <= tolerance
            detail = f"|{actual:.6g} - {expected:.6g}| <= {tolerance:g}"
        else:
            tolerance = RELATIVE_TOLERANCE * abs(expected)
            ok = abs(actual - expected) <= tolerance
            detail = f"|{actual:.6g} - {expected:.6g}| <= {tolerance:.4g} (2%)"
        print(f"  {'PASS' if ok else 'FAIL'}  {key:28s} {detail}")
        if not ok:
            failures.append(key)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_file", required=True, type=Path)
    parser.add_argument(
        "--select",
        choices=sorted(SELECTIONS),
        default=None,
        help="Apply a region's training selection before measuring. Reference "
             "JSONs in this directory are all post-selection.",
    )
    parser.add_argument("--nominal-mass", type=float, default=None,
                        help="Resonance mass in GeV, for the +/-5 and +/-10 MeV fractions.")
    parser.add_argument("--compare", type=Path, default=None)
    parser.add_argument("--json", dest="json_out", type=Path, default=None)
    args = parser.parse_args()

    z = load_theory_prior_z(args.in_file)
    raw_events = len(z)
    if args.select:
        z = filter_theory_prior(z, SELECTIONS[args.select])
        if len(z) == 0:
            raise SystemExit(f"selection {args.select!r} kept no events")

    nominal = args.nominal_mass
    if nominal is None and args.select == "jpsi":
        nominal = 3.0969

    stats = prior_stats(z, nominal_mass=nominal)
    stats["_source_file"] = args.in_file.name
    stats["_selection"] = args.select or "none"
    stats["_events_before_selection"] = int(raw_events)

    print(f"\n{args.in_file.name}")
    print(f"  selection: {stats['_selection']}   "
          f"{raw_events:,} -> {stats['events']:,} events "
          f"({100.0 * stats['events'] / max(raw_events, 1):.2f}% kept)")
    for key, value in stats.items():
        if key.startswith("_") or key == "events":
            continue
        print(f"  {key:28s} {value:12.6g}")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n",
                                 encoding="utf-8")
        print(f"\nWrote {args.json_out}")

    if args.compare is not None:
        reference = json.loads(args.compare.read_text(encoding="utf-8"))
        print(f"\nComparing against {args.compare.name}")
        if reference.get("_selection") not in (None, stats["_selection"]):
            print(f"  WARNING: reference was measured with selection "
                  f"{reference['_selection']!r}, this run used {stats['_selection']!r}")
        failures = compare(stats, reference)
        if failures:
            print(f"\nMISMATCH in {len(failures)} field(s): {', '.join(failures)}")
            return 1
        print("\nAll compared fields within tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
