#!/usr/bin/env python
"""Build a fake Upsilon(1S) prior by scaling the J/psi CKKW-L prior signal.

The scaling follows the idea recorded in memory.md (section 4, Plan 1):
take the J/psi signal component and scale every four-vector by

    R = M_Upsilon(1S) / M_J/psi

so the fake prior mass peak lands near the Upsilon(1S) mass. The fake file is
written in the same `FDL/zData` + `FDL/component_id` format used by the
decoder utilities, so it can be fed to the frozen Run E decoder unchanged.
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
    / "cms_jpsi_mumu_mg5py8_ckkwl_8tev_inclusive_0j1j_fiducial_3p0369_3p1569_1M_reweighted.hdf5"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs" / "fake_jpsi_to_upsilon1s.hdf5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-prior",
        type=Path,
        default=DEFAULT_SOURCE,
        help="J/psi prior HDF5 with FDL/zData and FDL/component_id.",
    )
    parser.add_argument(
        "--component",
        default="jpsi",
        help="Component name to scale; default jpsi.",
    )
    parser.add_argument(
        "--mass-target",
        type=float,
        default=9.4603,
        help="Target mass, default Upsilon(1S) PDG mass in GeV.",
    )
    parser.add_argument(
        "--mass-source",
        type=float,
        default=3.0969,
        help="Source mass, default J/psi PDG mass in GeV.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=20000,
        help="Maximum number of signal events to keep after component selection.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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
    output_path = args.output.expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(source_path, "r") as source:
        if "FDL/zData" not in source:
            raise KeyError("Source prior has no FDL/zData")
        z = np.asarray(source["FDL/zData"], dtype=np.float64)
        component = source.get("FDL/component_id")
        attrs = {key: decode_attr(value) for key, value in source.attrs.items()}
        if component is not None:
            component_ids = np.asarray(component)
            raw_mapping = attrs.get("component_id_mapping", "{}")
            mapping = (
                json.loads(raw_mapping)
                if isinstance(raw_mapping, str)
                else dict(raw_mapping or {})
            )
            mapping = {str(k): int(v) for k, v in mapping.items()}
            if args.component not in mapping:
                raise KeyError(
                    f"Component {args.component!r} not in source mapping {mapping}"
                )
            keep = component_ids == mapping[args.component]
        else:
            keep = np.ones(len(z), dtype=bool)
        z_selected = z[keep]
        if len(z_selected) == 0:
            raise ValueError("No events remain after component selection")

        if args.max_events is not None:
            z_selected = z_selected[: int(args.max_events)]

    scale = args.mass_target / args.mass_source
    fake = z_selected * scale
    fake = np.asarray(fake, dtype=np.float32)

    # All rows are the selected fake signal component.
    component_ids_out = np.zeros(len(fake), dtype=np.uint8)
    component_mapping_out = {f"fake_jpsi_to_upsilon1s": 0}
    component_counts = {name: int(np.sum(component_ids_out == cid)) for name, cid in component_mapping_out.items()}

    attrs_out = {
        "label": f"Fake J/psi CKKW-L prior scaled to Upsilon(1S) mass",
        "scale": float(scale),
        "mass_source_gev": float(args.mass_source),
        "mass_target_gev": float(args.mass_target),
        "source_prior": str(source_path),
        "source_prior_sha256": file_sha256(source_path),
        "source_component": args.component,
        "component_id_mapping": json.dumps(component_mapping_out),
        "component_counts": json.dumps(component_counts),
        "component_names": json.dumps(list(component_mapping_out)),
        "event_level": "scaled post-parton-shower stable muon four-vectors; scaling only",
    }
    for key in ("four_vector_convention", "daughter_muon_mass_GeV"):
        if key in attrs:
            attrs_out[key] = attrs[key]

    with h5py.File(output_path, "w") as destination:
        group = destination.create_group("FDL")
        group.create_dataset("zData", data=fake, compression="gzip", compression_opts=4, shuffle=True)
        group.create_dataset("component_id", data=component_ids_out, compression="gzip")
        for key, value in attrs_out.items():
            group.attrs[key] = value

    print("Wrote fake prior:", output_path)
    print("  events:", len(fake))
    print("  scale factor:", scale)
    print("  component mapping:", component_mapping_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
