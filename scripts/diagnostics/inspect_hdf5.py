#!/usr/bin/env python
"""Dump the structure of an HDF5 file, and optionally check a pairing.

Written for step 3 (memory.md section 7): nobody in this project has opened
``data/ppzee.hdf5`` in the current era, and ``paired_closure.py``'s docstring
assumes an ``FDL`` / ``ROL`` layout that is an assumption, not a fact. Also
useful on the unified priors.

    python scripts/diagnostics/inspect_hdf5.py data/ppzee.hdf5
    python scripts/diagnostics/inspect_hdf5.py data/ppzee.hdf5 --stats
    python scripts/diagnostics/inspect_hdf5.py data/ppzee.hdf5 --correlate FDL/zData FDL/xData

``--correlate`` is the pairing sanity check: if two [N, 8] arrays are
event-by-event partners their columns correlate strongly (section 7.4 measured
0.977-0.991 on the upstream results archive). If they do not, the rows are not
partners and nothing downstream is valid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _fmt(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.ndarray):
        return f"<array {value.shape} {value.dtype}>"
    return value


def dump(path: Path, stats: bool, sample: int) -> dict:
    import h5py

    out: dict = {"file": str(path), "datasets": {}, "attrs": {}, "groups": []}
    with h5py.File(path, "r") as handle:
        print(f"=== {path} ===")
        if handle.attrs:
            print("\n-- root attributes --")
            for key in handle.attrs:
                value = _fmt(handle.attrs[key])
                out["attrs"][key] = str(value)[:400]
                print(f"  {key} = {str(value)[:200]}")

        print("\n-- tree --")

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  [D] {name:40s} shape={obj.shape} dtype={obj.dtype}")
                entry = {"shape": list(obj.shape), "dtype": str(obj.dtype)}
                if obj.attrs:
                    entry["attrs"] = {k: str(_fmt(obj.attrs[k]))[:200] for k in obj.attrs}
                if stats and obj.ndim in (1, 2) and obj.size:
                    n = min(sample, obj.shape[0])
                    block = np.asarray(obj[:n], dtype=np.float64)
                    if block.ndim == 1:
                        block = block[:, None]
                    entry["column_stats"] = [
                        {
                            "min": float(np.nanmin(col)),
                            "median": float(np.nanmedian(col)),
                            "max": float(np.nanmax(col)),
                        }
                        for col in block.T
                    ]
                    for i, s in enumerate(entry["column_stats"]):
                        print(f"        col {i}: min {s['min']:.4g}  "
                              f"med {s['median']:.4g}  max {s['max']:.4g}")
                out["datasets"][name] = entry
            else:
                print(f"  [G] {name}")
                out["groups"].append(name)

        handle.visititems(visit)
    return out


def _parse_ref(ref: str) -> tuple[str, slice]:
    """``FDL`` -> all columns; ``ROL[0:8]`` -> that column slice.

    ppzee stores FDL as [N, 8] and ROL as [N, 12], so a pairing check has to
    name the columns rather than assume the shapes agree.
    """
    if "[" not in ref:
        return ref, slice(None)
    name, _, rest = ref.partition("[")
    body = rest.rstrip("]")
    parts = [piece.strip() for piece in body.split(":")]
    if len(parts) != 2:
        raise SystemExit(f"column slice must be NAME[a:b], got {ref!r}")
    start = int(parts[0]) if parts[0] else None
    stop = int(parts[1]) if parts[1] else None
    return name, slice(start, stop)


def correlate(path: Path, left: str, right: str, sample: int) -> dict:
    import h5py

    left_name, left_cols = _parse_ref(left)
    right_name, right_cols = _parse_ref(right)
    with h5py.File(path, "r") as handle:
        for name in (left_name, right_name):
            if name not in handle:
                raise SystemExit(
                    f"{name} not in {path}. Available: "
                    f"{', '.join(sorted(handle.keys()))}"
                )
        n = min(sample, handle[left_name].shape[0], handle[right_name].shape[0])
        a = np.asarray(handle[left_name][:n], dtype=np.float64)[:, left_cols]
        b = np.asarray(handle[right_name][:n], dtype=np.float64)[:, right_cols]
    if a.shape != b.shape:
        raise SystemExit(
            f"shape mismatch after slicing: {a.shape} vs {b.shape}. "
            "Use NAME[a:b] to select matching column counts."
        )

    print(f"\n-- pairing check: {left} vs {right} over {n} rows --")
    per_column = []
    for i in range(a.shape[1]):
        x, y = a[:, i], b[:, i]
        if x.std() == 0 or y.std() == 0:
            rho = float("nan")
        else:
            rho = float(np.corrcoef(x, y)[0, 1])
        per_column.append(rho)
        print(f"  col {i}: rho = {rho:.4f}")

    # A shuffled control: if the pairing is real, breaking it must destroy rho.
    rng = np.random.default_rng(0)
    shuffled = b[rng.permutation(len(b))]
    control = []
    for i in range(a.shape[1]):
        x, y = a[:, i], shuffled[:, i]
        control.append(float(np.corrcoef(x, y)[0, 1]) if x.std() and y.std() else float("nan"))
    print(f"  shuffled control (should be ~0): "
          f"{', '.join(f'{c:.3f}' for c in control)}")
    print("\n  Section 7.4 measured 0.977-0.991 on the upstream archive.")
    print("  If these are near the shuffled control, the rows are NOT partners.")
    return {"per_column": per_column, "shuffled_control": control, "rows": int(n)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", type=Path)
    p.add_argument("--stats", action="store_true", help="Per-column min/median/max.")
    p.add_argument("--correlate", nargs=2, metavar=("LEFT", "RIGHT"),
                   help="Two datasets to check as event-by-event partners. "
                        "Accepts a column slice: 'ROL[0:8]'. Same dataset twice "
                        "is allowed, e.g. 'FDL[0:2]' vs 'FDL[4:6]' to test "
                        "whether the transverse momenta are exactly opposite.")
    p.add_argument("--sample", type=int, default=200_000,
                   help="Rows to read for stats and correlation.")
    p.add_argument("--json", dest="json_out", type=Path, default=None)
    args = p.parse_args(argv)

    report = dump(args.path, args.stats, args.sample)
    if args.correlate:
        report["pairing"] = correlate(args.path, *args.correlate, args.sample)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
