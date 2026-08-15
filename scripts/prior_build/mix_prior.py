#!/usr/bin/env python
"""Mix signal + continuum prior HDF5s at a post-filter fraction and write the final prior.

Both inputs are assumed to be FDL/zData HDF5s already passed through the
trigger-equivalent selection and smearing. The fraction is defined ON THE
FILTERED COUNTS: n_signal_out = round(frac_signal * n_total_out), with the
smaller component driving the total (so no component is upsampled unless
--allow-upsample is given).

Usage:
  python scripts/prior_build/mix_prior.py --signal sig.hdf5 --continuum dy.hdf5 \
      --out mixed.hdf5 [--frac-signal 0.85] [--seed 0]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.cms_data import load_theory_prior_z  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal", type=Path, required=True)
    parser.add_argument("--continuum", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frac-signal", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-upsample", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.frac_signal < 1.0:
        raise SystemExit("--frac-signal must be in (0,1)")

    sig = load_theory_prior_z(args.signal).astype(np.float32)
    con = load_theory_prior_z(args.continuum).astype(np.float32)
    f = args.frac_signal
    n_total = min(len(sig) / f, len(con) / (1 - f))
    n_sig = int(round(n_total * f))
    n_con = int(round(n_total * (1 - f)))
    if n_sig > len(sig) or n_con > len(con):
        if not args.allow_upsample:
            raise SystemExit(
                f"requested mix needs {n_sig} signal + {n_con} continuum events, but inputs "
                f"have {len(sig)} / {len(con)}; generate more events or pass --allow-upsample"
            )

    rng = np.random.default_rng(args.seed)
    idx_sig = rng.choice(len(sig), size=n_sig, replace=n_sig > len(sig))
    idx_con = rng.choice(len(con), size=n_con, replace=n_con > len(con))
    mixed = np.concatenate([sig[idx_sig], con[idx_con]], axis=0)
    mixed = mixed[rng.permutation(len(mixed))].astype(np.float32)

    import h5py
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as h:
        grp = h.create_group("FDL")
        grp.create_dataset("zData", data=mixed, dtype=np.float32)
        for key, value in {
            "feature_names": '["mu_minus_px", "mu_minus_py", "mu_minus_pz", "mu_minus_E", '
                             '"mu_plus_px", "mu_plus_py", "mu_plus_pz", "mu_plus_E"]',
            "units": "GeV",
            "particle_order": "mu- (PDG 13) first; mu+ (PDG -13) second",
            "signal_file": args.signal.name,
            "continuum_file": args.continuum.name,
            "frac_signal_post_filter": float(args.frac_signal),
            "n_signal": int(n_sig),
            "n_continuum": int(n_con),
            "mix_seed": int(args.seed),
            "production_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }.items():
            grp.attrs[key] = value
    print(f"mixed prior: {n_sig:,} signal + {n_con:,} continuum = {len(mixed):,} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
