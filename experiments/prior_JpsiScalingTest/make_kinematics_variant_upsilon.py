#!/usr/bin/env python
"""Build Upsilon(1S) variants with changed kinematics but the same mass.

Method: take true Upsilon(1S) events and Lorentz-boost each full dimuon event
so that the pair transverse momentum is set to a chosen value while the
E-based pair mass and pair rapidity are unchanged.

The output file has the same FDL/zData format as the other priors.
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
    parser.add_argument("--pair-pt", type=float, required=True,
                        help="Target pair pT in GeV, e.g. 10 or 80.")
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


def boost_particles(p4, beta):
    """Boost p4 arrays [N,4] columns [px,py,pz,E] by per-event beta [N,3]."""
    px = p4[:, 0]
    py = p4[:, 1]
    pz = p4[:, 2]
    energy = p4[:, 3]
    beta_sq = np.sum(beta * beta, axis=1)
    gamma = 1.0 / np.sqrt(np.maximum(1.0 - beta_sq, 1e-12))
    bp = beta[:, 0] * px + beta[:, 1] * py + beta[:, 2] * pz
    factor = np.where(
        beta_sq > 1e-12,
        (gamma - 1.0) / np.maximum(beta_sq, 1e-12),
        0.0,
    )
    new_px = px + factor * bp * beta[:, 0] + gamma * energy * beta[:, 0]
    new_py = py + factor * bp * beta[:, 1] + gamma * energy * beta[:, 1]
    new_pz = pz + factor * bp * beta[:, 2] + gamma * energy * beta[:, 2]
    new_energy = gamma * (energy + bp)
    return np.stack([new_px, new_py, new_pz, new_energy], axis=1)


def main() -> int:
    args = parse_args()
    source_path = args.source_prior.expanduser().resolve()
    output_path = args.output.expanduser().resolve() if args.output is not None else (
        OUT_DIR / f"true_upsilon1s_kinematics_pt{args.pair_pt:g}.hdf5"
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

    p1 = z[:, 0:4]
    p2 = z[:, 4:8]
    pair = p1 + p2
    pair_e = pair[:, 3]
    pair_p = pair[:, 0:3]
    pair_mass2 = pair_e**2 - np.sum(pair_p * pair_p, axis=1)
    pair_mass = np.sqrt(np.maximum(pair_mass2, 0.0))

    # Original pair rapidity.
    rapidity = 0.5 * np.log(
        np.maximum(pair_e + pair_p[:, 2], 1e-10)
        / np.maximum(pair_e - pair_p[:, 2], 1e-10)
    )

    # 1) Boost to the pair rest frame.
    beta_rest = -pair_p / np.maximum(pair_e[:, None], 1e-10)
    p1_rest = boost_particles(p1, beta_rest)
    p2_rest = boost_particles(p2, beta_rest)

    # 2) Boost back to a target frame with the requested pair pT.
    pair_pt_orig = np.hypot(pair_p[:, 0], pair_p[:, 1])
    cos_phi = np.divide(pair_p[:, 0], pair_pt_orig, out=np.ones_like(pair_pt_orig), where=pair_pt_orig > 0)
    sin_phi = np.divide(pair_p[:, 1], pair_pt_orig, out=np.zeros_like(pair_pt_orig), where=pair_pt_orig > 0)
    target_pt = np.full(len(z), float(args.pair_pt), dtype=np.float64)
    tx = target_pt * cos_phi
    ty = target_pt * sin_phi
    mt = np.sqrt(pair_mass**2 + target_pt**2)
    tz = mt * np.sinh(rapidity)
    te = mt * np.cosh(rapidity)
    beta_target = np.stack([tx / te, ty / te, tz / te], axis=1)
    p1_out = boost_particles(p1_rest, beta_target)
    p2_out = boost_particles(p2_rest, beta_target)
    z_out = np.concatenate([p1_out, p2_out], axis=1)

    # Verify E-based mass is preserved.
    pair_out = p1_out + p2_out
    mass_out = np.sqrt(np.maximum(pair_out[:, 3]**2 - np.sum(pair_out[:, 0:3]**2, axis=1), 0.0))
    max_mass_diff = float(np.max(np.abs(mass_out - pair_mass)))

    comp_out = np.zeros(len(z_out), dtype=np.uint8)
    mapping_out = {f"upsilon1s_kinematics_pt{args.pair_pt:g}": 0}
    counts = {name: int(np.sum(comp_out == cid)) for name, cid in mapping_out.items()}
    attrs_out = {
        "label": f"True Upsilon(1S) prior with kinematics changed to pair pT {args.pair_pt:g} GeV, mass preserved",
        "pair_pt_target_gev": float(args.pair_pt),
        "kinematics_method": "Lorentz boost to pair rest frame then to target pair pT; E-based mass and rapidity preserved",
        "max_abs_mass_change_gev": max_mass_diff,
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

    print("Wrote kinematics variant:", output_path)
    print("  pair pT target:", args.pair_pt, "events:", len(z_out))
    print("  max abs mass change [GeV]:", max_mass_diff)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
