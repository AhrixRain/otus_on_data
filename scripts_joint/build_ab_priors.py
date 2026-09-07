#!/usr/bin/env python
"""Build the event-matched prior pair for the controlled prior-width A/B test.

Motivation
----------
Joint Runs D and E were trained against ``cms_jpsi_mumu_mg5_8tev_mixed_ptj5``,
a truth prior that had been *hand-smeared* so its pair-mass width (26.2 MeV)
already matched the CMS signal-window width (28.1 MeV). Under that prior the
encoder only had to remove ~7% of the mass width, and both runs selected a
stage-1 deterministic checkpoint as best. Run F swapped in the showered 0j+1j
prior, whose width is 14.7 MeV, and failed the J/psi latent mass gate at every
validation.

Those two facts are confounded: Run F changed the generator, the shower, the
reweighting, the event count *and* the width at once. This script isolates the
width.

Design
------
Both arms are derived from the SAME source file:

  arm "narrow"   -- the source prior, untouched (~14.7 MeV)
  arm "smeared"  -- the same events with a calibrated Gaussian pT smear applied
                    so the pair-mass width lands on ``--target-std``

Each arm then has the training selection applied on its own terms (both muons
pT > 3, |eta| < 2.4, pair mass inside the J/psi window) -- exactly what the
loader would do to either file.

A note on why the arms are NOT event-matched. Requiring an event to pass the
window before *and* after smearing looks stricter, but the window is only
120 MeV wide, so that requirement discards precisely the tail events that carry
the mass width, in both arms at once: it pulled the smeared arm down to 23.2 MeV
and the narrow arm from 14.7 MeV to 12.8 MeV, deforming the quantity under test.
Per-arm selection keeps the narrow arm identical to the prior Run F actually
failed on, at the cost of a ~3% event-count difference between arms -- which is
negligible next to a 1.8x width difference, and is checked below by comparing
the kinematic marginals.

Smearing is calibrated against the width measured AFTER selection, which is also
how the Run D/E prior's quoted 26.2 MeV was obtained (that build reported a
~5.7% smearing-induced window spill-out).

Usage
-----
  python scripts_joint/build_ab_priors.py --calibrate-only
  python scripts_joint/build_ab_priors.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (REPO_ROOT, REPO_ROOT / "scripts" / "prior_build"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from scripts.physics import invariant_mass_np  # noqa: E402
from scripts.cms_data import load_theory_prior_z  # noqa: E402
from smear_prior import smear_rows  # noqa: E402

_MU_MASS = 0.105658
_MASS_MIN, _MASS_MAX = 3.0369, 3.1569

DEFAULT_SOURCE = (
    REPO_ROOT
    / "data"
    / "cms_jpsi_mumu_mg5py8_ckkwl_8tev_inclusive_0j1j_fiducial_3p0369_3p1569_1M_reweighted.hdf5"
)
# 26.2 MeV is the measured pair-mass width of the Run D/E prior
# (cms_jpsi_mumu_mg5_8tev_mixed_ptj5) on its z_test split.
DEFAULT_TARGET_STD = 0.0262


def selection_mask(z: np.ndarray) -> np.ndarray:
    pt1 = np.hypot(z[:, 0], z[:, 1])
    pt2 = np.hypot(z[:, 4], z[:, 5])
    eta1 = np.arcsinh(z[:, 2] / np.maximum(pt1, 1e-12))
    eta2 = np.arcsinh(z[:, 6] / np.maximum(pt2, 1e-12))
    mass = invariant_mass_np(z, daughter_masses=(_MU_MASS, _MU_MASS), stable=True)
    return (
        (pt1 > 3.0)
        & (pt2 > 3.0)
        & (np.abs(eta1) < 2.4)
        & (np.abs(eta2) < 2.4)
        & (mass > _MASS_MIN)
        & (mass < _MASS_MAX)
    )


def effective_mass_std(z: np.ndarray, a: float, seeds: tuple[int, ...] = (0, 1)) -> float:
    """Pair-mass std AFTER the training selection, averaged over ``seeds``.

    This is the quantity the trainer actually sees, and the one the Run D/E
    prior's published width refers to. Calibrating on the pre-selection width
    overshoots, because window spill-out truncates the tails.
    """
    values = []
    for seed in seeds:
        smeared = smear_rows(z, a, np.random.default_rng(seed))
        kept = smeared[selection_mask(smeared)]
        if len(kept) < 100:
            return 0.0
        mass = invariant_mass_np(kept, daughter_masses=(_MU_MASS, _MU_MASS), stable=True)
        values.append(float(mass.std()))
    return float(np.mean(values))


def calibrate_effective(
    z: np.ndarray, target_std: float, *, a_max: float = 0.05, iterations: int = 40
) -> float:
    """Bisect the smearing amplitude against the post-selection width."""
    ceiling = effective_mass_std(z, a_max)
    if ceiling < target_std:
        raise SystemExit(
            f"Target {target_std * 1000:.1f} MeV is unreachable: even a={a_max} only "
            f"reaches {ceiling * 1000:.2f} MeV after selection, because the "
            f"{(_MASS_MAX - _MASS_MIN) * 1000:.0f} MeV window truncates the tails."
        )
    lo, hi = 0.0, a_max
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if effective_mass_std(z, mid) > target_std:
            hi = mid
        else:
            lo = mid
        if (hi - lo) < 1e-6:
            break
    return 0.5 * (lo + hi)


def describe(z: np.ndarray, label: str) -> dict[str, float]:
    mass = invariant_mass_np(z, daughter_masses=(_MU_MASS, _MU_MASS), stable=True)
    pt1 = np.hypot(z[:, 0], z[:, 1])
    pt2 = np.hypot(z[:, 4], z[:, 5])
    pair_pt = np.hypot(z[:, 0] + z[:, 4], z[:, 1] + z[:, 5])
    muon_pt = np.concatenate([pt1, pt2])
    eta = np.abs(
        np.concatenate(
            [
                np.arcsinh(z[:, 2] / np.maximum(pt1, 1e-12)),
                np.arcsinh(z[:, 6] / np.maximum(pt2, 1e-12)),
            ]
        )
    )
    stats = {
        "events": int(len(z)),
        "mass_mean_gev": float(mass.mean()),
        "mass_std_mev": float(mass.std() * 1000.0),
        "muon_pt_median_gev": float(np.median(muon_pt)),
        "muon_pt_p99_gev": float(np.quantile(muon_pt, 0.99)),
        "pair_pt_median_gev": float(np.median(pair_pt)),
        "pair_pt_p99_gev": float(np.quantile(pair_pt, 0.99)),
        "abs_eta_median": float(np.median(eta)),
    }
    print(
        f"  {label:10s} n={stats['events']:>7,}  mass {stats['mass_mean_gev']:.5f}"
        f" +- {stats['mass_std_mev']:5.2f} MeV | muon pT med {stats['muon_pt_median_gev']:5.2f}"
        f" p99 {stats['muon_pt_p99_gev']:6.2f} | pair pT med {stats['pair_pt_median_gev']:5.2f}"
        f" | |eta| med {stats['abs_eta_median']:.3f}"
    )
    return stats


def write_prior(path: Path, z: np.ndarray, attrs: dict) -> str:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        group = handle.create_group("FDL")
        group.create_dataset("zData", data=z.astype(np.float32), dtype=np.float32)
        for key, value in attrs.items():
            group.attrs[key] = value
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"  wrote {path.name}  ({len(z):,} events, sha256 {digest[:16]}...)")
    return digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--target-std", type=float, default=DEFAULT_TARGET_STD)
    parser.add_argument("--tolerance-mev", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument(
        "--calibration-events",
        type=int,
        default=200000,
        help="Subsample size used while bisecting the smearing amplitude.",
    )
    parser.add_argument("--calibrate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.expanduser().resolve()
    if not source.exists():
        raise SystemExit(f"Source prior not found: {source}")
    narrow_path = args.out_dir / "cms_jpsi_ab_narrow.hdf5"
    smeared_path = args.out_dir / "cms_jpsi_ab_smeared.hdf5"
    if not args.calibrate_only and not args.overwrite:
        for path in (narrow_path, smeared_path):
            if path.exists():
                raise SystemExit(f"{path} already exists; pass --overwrite to replace it")

    print(f"Source: {source.name}")
    z = load_theory_prior_z(source).astype(np.float64)
    print(f"  loaded {len(z):,} events")

    narrow = z[selection_mask(z)]
    print(f"  {len(narrow):,} pass the training selection unsmeared (arm 'narrow')")

    # Calibrate on a subsample for speed, then verify on the full set below.
    calibration_rng = np.random.default_rng(0)
    sample_size = min(len(z), int(args.calibration_events))
    sample = z[calibration_rng.choice(len(z), size=sample_size, replace=False)]
    achieved_a = calibrate_effective(sample, args.target_std)
    print(
        f"  calibrated smearing a = {achieved_a:.5f} on {sample_size:,} events "
        f"-> post-selection std ~ {effective_mass_std(sample, achieved_a) * 1000:.2f} MeV "
        f"(target {args.target_std * 1000:.1f})"
    )
    if args.calibrate_only:
        return 0

    z_smeared = smear_rows(z, achieved_a, np.random.default_rng(args.seed))
    smeared = z_smeared[selection_mask(z_smeared)]
    spill = 1.0 - len(smeared) / len(z)
    print(f"  {len(smeared):,} pass the training selection after smearing "
          f"({spill * 100:.2f}% window spill-out)")
    if len(smeared) < 10000:
        raise SystemExit("Too few surviving events; check the source file or target width")

    print("Arm statistics (each arm selected on its own terms):")
    narrow_stats = describe(narrow, "narrow")
    smeared_stats = describe(smeared, "smeared")

    delta = abs(smeared_stats["mass_std_mev"] - args.target_std * 1000.0)
    if delta > args.tolerance_mev:
        raise SystemExit(
            f"Smeared width {smeared_stats['mass_std_mev']:.2f} MeV misses the target "
            f"{args.target_std * 1000:.1f} MeV by {delta:.2f} MeV (tolerance {args.tolerance_mev})"
        )

    checks = {
        "muon pT median": (
            narrow_stats["muon_pt_median_gev"],
            smeared_stats["muon_pt_median_gev"],
            0.02,
        ),
        "pair pT median": (
            narrow_stats["pair_pt_median_gev"],
            smeared_stats["pair_pt_median_gev"],
            0.02,
        ),
        "|eta| median": (
            narrow_stats["abs_eta_median"],
            smeared_stats["abs_eta_median"],
            0.02,
        ),
        # Window spill-out is intrinsic to smearing inside a 120 MeV window, not
        # a defect: the Run D/E prior build reported ~5.7% loss for the same
        # reason. The limit only has to catch a pathological build.
        "event count": (
            float(narrow_stats["events"]),
            float(smeared_stats["events"]),
            0.10,
        ),
    }
    print("Controlled-comparison checks (width must be the only difference):")
    for label, (left, right, limit) in checks.items():
        shift = abs(right - left) / left
        status = "ok" if shift <= limit else "FAIL"
        print(f"  {label:16s} {left:10.4f} vs {right:10.4f}  {shift * 100:5.2f}%  [{status}]")
        if shift > limit:
            raise SystemExit(
                f"{label} differs by {shift * 100:.2f}% between arms (limit "
                f"{limit * 100:.0f}%); the arms would not be a controlled comparison"
            )
    ratio = smeared_stats["mass_std_mev"] / narrow_stats["mass_std_mev"]
    print(f"  mass width ratio smeared/narrow: {ratio:.2f}x  (this is the variable)")

    common = {
        "feature_names": '["mu_minus_px", "mu_minus_py", "mu_minus_pz", "mu_minus_E", '
        '"mu_plus_px", "mu_plus_py", "mu_plus_pz", "mu_plus_E"]',
        "units": "GeV",
        "particle_order": "mu- (PDG 13) first; mu+ (PDG -13) second",
        "daughter_muon_mass_GeV": _MU_MASS,
        "source_file": source.name,
        "ab_test": "prior_mass_width",
        "event_matched": False,
        "selection": "both muons pT>3, |eta|<2.4, pair mass in [3.0369, 3.1569], "
        "applied to each arm independently",
        "production_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    narrow_digest = write_prior(
        narrow_path, narrow, {**common, "arm": "narrow", "smearing_a": 0.0}
    )
    smeared_digest = write_prior(
        smeared_path,
        smeared,
        {
            **common,
            "arm": "smeared",
            "smearing_model": "pT' = pT * (1 + N(0, a)); eta/phi fixed; E = sqrt(p'^2 + m_mu^2)",
            "smearing_a": float(achieved_a),
            "smearing_seed": int(args.seed),
        },
    )

    manifest = {
        "schema_version": 1,
        "purpose": "Controlled prior-mass-width A/B for the cms_Joint series",
        "event_matched": False,
        "selection_note": "each arm selected on its own terms; see the module docstring",
        "source_file": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "seed": int(args.seed),
        "target_mass_std_gev": float(args.target_std),
        "calibrated_smearing_a": float(achieved_a),
        "arms": {
            "narrow": {"path": str(narrow_path), "sha256": narrow_digest, **narrow_stats},
            "smeared": {"path": str(smeared_path), "sha256": smeared_digest, **smeared_stats},
        },
    }
    manifest_path = args.out_dir / "cms_jpsi_ab_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"  manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
