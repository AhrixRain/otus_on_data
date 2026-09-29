#!/usr/bin/env python
"""Extract a small Upsilon(1S) component subset in the decoder-input format.

The fake J/psi prior contains only signal-like events. To compare it with the
real Upsilon transfer on equal footing, this script writes a true-Upsilon(1S)
subset file with the same number of component events. It does not scale or
otherwise modify the events.
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
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs" / "true_upsilon1s_subset.hdf5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-prior", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--component", default="upsilon1s")
    parser.add_argument("--max-events", type=int, default=20000)
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
        component = np.asarray(source["FDL/component_id"])
        attrs = {key: decode_attr(value) for key, value in source.attrs.items()}
        raw_mapping = attrs.get("component_id_mapping", "{}")
        mapping = (
            json.loads(raw_mapping) if isinstance(raw_mapping, str) else dict(raw_mapping or {})
        )
        mapping = {str(k): int(v) for k, v in mapping.items()}
        if args.component not in mapping:
            raise KeyError(f"Component {args.component!r} not in mapping {mapping}")
        keep = component == mapping[args.component]
        z_selected = z[keep]
        if len(z_selected) == 0:
            raise ValueError("No events remain after component selection")
        if args.max_events is not None:
            z_selected = z_selected[: int(args.max_events)]

    component_mapping_out = {args.component: 0}
    component_ids_out = np.zeros(len(z_selected), dtype=np.uint8)
    component_counts = {
        name: int(np.sum(component_ids_out == cid)) for name, cid in component_mapping_out.items()
    }

    attrs_out = {
        "label": f"True Upsilon prior subset: {args.component}",
        "source_prior": str(source_path),
        "source_prior_sha256": file_sha256(source_path),
        "source_component": args.component,
        "component_id_mapping": json.dumps(component_mapping_out),
        "component_counts": json.dumps(component_counts),
        "component_names": json.dumps(list(component_mapping_out)),
        "event_level": attrs.get("event_level", "post-parton-shower stable muons"),
    }

    with h5py.File(output_path, "w") as destination:
        group = destination.create_group("FDL")
        group.create_dataset("zData", data=np.asarray(z_selected, dtype=np.float32),
                             compression="gzip", compression_opts=4, shuffle=True)
        group.create_dataset("component_id", data=component_ids_out, compression="gzip")
        for key, value in attrs_out.items():
            group.attrs[key] = value

    print("Wrote true Upsilon subset:", output_path)
    print("  events:", len(z_selected))
    print("  component mapping:", component_mapping_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
