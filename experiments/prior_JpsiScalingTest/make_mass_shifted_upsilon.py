#!/usr/bin/env python
"""Build mass-shifted true Upsilon priors without changing momenta.

For each Upsilon(1S) event, keep px,py,pz exactly the same and rescale the two
stored energies so the E-based dimuon mass is multiplied by (1 + shift).
This changes the mass seen by the decoder's z-space condition while leaving the
pT/eta/phi kinematics unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    REPO_ROOT
    / "data"
    / "cms_upsilon_mumu_mg5py8_ckkwl_8tev_inclusive_0j1j_fiducial_8p5_11p5_1M.hdf5"
)
OUT_DIR = Path(__file__).resolve().parent / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-prior", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--component", default="upsilon1s")
    parser.add_argument("--shift", type=float, required=True,
                        help="Fractional mass shift, e.g. 0.03 for +3% or -0.05 for -5%.")
    parser.add_argument("--max-events", type=int, default=20000)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> int:
    args = parse_args()
    source_path = args.source_prior.expanduser().resolve()
    output_path = args.output.expanduser().resolve() if args.output is not None else (
        OUT_DIR / f"true_upsilon1s_mass_{args.shift:+.3f}.hdf5"
    )
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(source_path, "r") as source:
        if "FDL/zData" not in source:
            raise KeyError("No FDL/zData")
        z = np.asarray(source["FDL/zData"], dtype=np.float64)
        component = np.asarray(source["FDL/component_id"])
        attrs = {key: decode_attr(value) for key, value in source.attrs.items()}
        raw_mapping = attrs.get("component_id_mapping", "{}")
        mapping = json.loads(raw_mapping) if isinstance(raw_mapping, str) else dict(raw_mapping or {})
        mapping = {str(k): int(v) for k, v in mapping.items()}
        if args.component not in mapping:
            raise KeyError(f"Component {args.component!r} not in mapping {mapping}")
        z = z[component == mapping[args.component]]
        if args.max_events is not None:
            z = z[: int(args.max_events)]

    p1 = z[:, 0:3]
    p2 = z[:, 4:7]
    e1 = z[:, 3]
    e2 = z[:, 7]
    energy_sum = e1 + e2
    p_sum = p1 + p2
    p2_mag = np.sum(p_sum * p_sum, axis=1)
    mass2 = energy_sum**2 - p2_mag
    mass_orig = np.sqrt(np.maximum(mass2, 0.0))
    target_mass = mass_orig * (1.0 + args.shift)
    new_energy_sum = np.sqrt(p2_mag + target_mass**2)
    scale = np.divide(
        new_energy_sum,
        energy_sum,
        out=np.ones_like(energy_sum),
        where=energy_sum > 0,
    )
    z_out = z.copy()
    z_out[:, 3] = e1 * scale
    z_out[:, 7] = e2 * scale

    comp_out = np.zeros(len(z_out), dtype=np.uint8)
    mapping_out = {f"upsilon1s_mass_{args.shift:+.3f}": 0}
    counts = {name: int(np.sum(comp_out == cid)) for name, cid in mapping_out.items()}
    attrs_out = {
        "label": f"True Upsilon(1S) prior with mass shift {args.shift:+.3f}, momenta unchanged",
        "mass_shift": float(args.shift),
        "mass_shift_method": "rescale stored E1,E2 to scale E-based pair mass; px,py,pz unchanged",
        "source_prior": str(source_path),
        "source_prior_sha256": file_sha256(source_path),
        "source_component": args.component,
        "component_id_mapping": json.dumps(mapping_out),
        "component_counts": json.dumps(counts),
        "component_names": json.dumps(list(mapping_out)),
    }

    with h5py.File(output_path, "w") as dest:
        group = dest.create_group("FDL")
        group.create_dataset("zData", data=np.asarray(z_out, dtype=np.float32),
                             compression="gzip", compression_opts=4, shuffle=True)
        group.create_dataset("component_id", data=comp_out, compression="gzip")
        for key, value in attrs_out.items():
            group.attrs[key] = value

    print("Wrote mass-shifted Upsilon:", output_path)
    print("  shift:", args.shift, "events:", len(z_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
