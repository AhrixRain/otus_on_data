#!/usr/bin/env python
"""Verify that a config's data staging is complete on THIS machine (read-only).

Why this exists.  On 2026-10-04 a server run died inside
cms_data.file_fingerprint on data/legacy/priors/jpsi_unified_bare_tms10.hdf5
after the priors had been staged at data/.  A --dry-run also catches that, but
only after streaming the 2.1 GB CMS ROOT file.  This checks the layout in one
pass and reports every missing file and every missing HDF5 dataset at once,
which is what you want on a fresh host before spending GPU time.

It also checks the failure mode that is silent by construction: TWO resolvers
exist for a prior path and they do not have to agree.

  * cms_data.data_cache_metadata fingerprints the CONFIGURED path directly,
    with no fallback.  That value builds the cache key and the contract digest.
  * fixed_z_noise_budget.resolve_prior_path tries data/<configured>, then
    data/legacy/<basename>, then data/<basename>.

If they land on different files the run loads one prior while fingerprinting
another, and the contract digest describes a file nobody read.  That is
reported as a DISAGREEMENT, not as a pass.

Usage:
    python scripts_joint/check_data_layout.py --run H_kneeKernel
    python scripts_joint/check_data_layout.py --all
    python scripts_joint/check_data_layout.py --config configs_joint/cms_Joint_runH_D3b.yaml
    python scripts_joint/check_data_layout.py --all --json data_layout.json

Exit 0 when every checked config is complete; 1 when any file or dataset is
missing or the two resolvers disagree; 2 on a usage error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def find_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "configs_joint").is_dir() and (candidate / "scripts_joint").is_dir():
            return candidate
    raise RuntimeError("Could not locate the OTUS repository root")


REPO_ROOT = find_repo_root()
for _directory in (
    REPO_ROOT / "scripts",
    REPO_ROOT / "scripts_sota",
    REPO_ROOT / "scripts_joint",
):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from cms_data import (  # noqa: E402
    LEGACY_JPSI_PRIOR_RELATIVE,
    load_config,
    resolve_legacy_prior_path,
)
from joint_data import resolve_joint_config  # noqa: E402
from fixed_z_noise_budget import resolve_prior_path  # noqa: E402


def human(size: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024.0 or unit == "GiB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GiB"


def hdf5_datasets(path: Path) -> tuple[dict[str, bool], str | None]:
    try:
        import h5py
    except ImportError:
        return {}, "h5py not installed"
    try:
        with h5py.File(path, "r") as handle:
            present = {
                "FDL/zData": "FDL/zData" in handle,
                "FDL/component_id": "FDL/component_id" in handle,
                "FDL/weight": "FDL/weight" in handle,
            }
            counts = {
                key: int(handle[key].shape[0]) for key in present if present[key]
            }
        return present, None if counts.get("FDL/zData") else "FDL/zData is empty"
    except Exception as error:  # noqa: BLE001
        return {}, f"unreadable: {error}"


def check_config(path: Path) -> dict:
    label = path.name
    entry: dict = {"config": str(path.relative_to(REPO_ROOT)), "problems": [], "rows": []}
    try:
        config = resolve_joint_config(load_config(path))
    except Exception as error:  # noqa: BLE001
        entry["problems"].append(f"could not resolve the config: {error}")
        return entry

    data_root = Path(config.get("paths", {}).get("data_root", "data"))
    entry["data_root"] = str(data_root)

    def primary(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else (REPO_ROOT / data_root / candidate)

    root_value = config.get("paths", {}).get("cms_root_file")
    if root_value:
        target = primary(str(root_value))
        exists = target.exists()
        entry["rows"].append(
            {
                "item": "cms_root_file",
                "configured": str(root_value),
                "resolved": str(target),
                "exists": exists,
                "size": human(target.stat().st_size) if exists else "-",
            }
        )
        if not exists:
            entry["problems"].append(f"cms_root_file missing: {target}")

    components_enabled = bool(config.get("prior_components", {}).get("enabled"))
    for region, region_config in (config.get("regions") or {}).items():
        paths = (region_config or {}).get("paths") or {}
        prior_value = paths.get("theory_prior_file")
        if not prior_value:
            continue
        configured = str(prior_value)
        fingerprint_target = primary(configured)
        row: dict = {
            "item": f"prior:{region}",
            "configured": configured,
            "resolved": str(fingerprint_target),
            "exists": fingerprint_target.exists(),
            "size": human(fingerprint_target.stat().st_size)
            if fingerprint_target.exists()
            else "-",
        }
        if not row["exists"]:
            entry["problems"].append(
                f"region {region!r}: the configured prior is missing at {fingerprint_target}"
            )
        loader_target, _described, used_fallback = resolve_prior_path(config, region)
        row["loader_resolved"] = str(loader_target)
        row["loader_used_fallback"] = bool(used_fallback)
        if loader_target != fingerprint_target:
            entry["problems"].append(
                f"region {region!r}: RESOLVER DISAGREEMENT - the cache fingerprint would "
                f"use {fingerprint_target} but the loader reads {loader_target}"
            )
        required = ["FDL/zData"]
        if components_enabled and region in (config.get("prior_components") or {}):
            required += ["FDL/component_id", "FDL/weight"]
        if row["exists"]:
            present, error = hdf5_datasets(fingerprint_target)
            row["datasets"] = {name: present.get(name, False) for name in required}
            if error:
                entry["problems"].append(f"region {region!r}: {error}")
            else:
                missing = [name for name in required if not present.get(name)]
                if missing:
                    entry["problems"].append(
                        f"region {region!r}: {fingerprint_target.name} lacks {missing} "
                        f"(prior_components.enabled={components_enabled})"
                    )
        entry["rows"].append(row)

        # A second, easily-missed dependency: with signal_fraction_source
        # legacy_prior_effective the mixer opens ANOTHER file, only to read its
        # FDL composition attributes. It defaults to a module constant, so a
        # config that never mentions the file still needs it on disk.
        component_spec = (config.get("prior_components") or {}).get(region) or {}
        if (
            isinstance(component_spec, dict)
            and component_spec.get("signal_fraction_source") == "legacy_prior_effective"
        ):
            legacy_value = str(
                component_spec.get("legacy_prior_file", LEGACY_JPSI_PRIOR_RELATIVE)
            )
            # Resolve with the SAME helper the mixer uses, so the report can
            # never disagree with what would actually be opened. The helper is
            # layout-independent: data/<configured>, data/legacy/<name>, then
            # data/<name>.
            try:
                legacy_target = resolve_legacy_prior_path(legacy_value, data_root)
                legacy_exists = True
                legacy_error = None
            except FileNotFoundError as error:
                legacy_target = primary(legacy_value)
                legacy_exists = False
                legacy_error = str(error)
            entry["rows"].append(
                {
                    "item": f"legacy-mix:{region}",
                    "configured": legacy_value,
                    "resolved": str(legacy_target),
                    "exists": legacy_exists,
                    "size": human(legacy_target.stat().st_size) if legacy_exists else "-",
                    "note": "read for its FDL composition attributes only",
                }
            )
            if not legacy_exists:
                entry["problems"].append(f"region {region!r}: {legacy_error}")
            else:
                try:
                    import h5py

                    with h5py.File(legacy_target, "r") as handle:
                        group = handle["FDL"] if "FDL" in handle else handle
                        attrs = dict(group.attrs)
                        has_component = "component_id" in group
                    if not has_component and not any(
                        key in attrs
                        for key in ("n_signal", "n_continuum", "frac_signal_post_filter")
                    ):
                        entry["problems"].append(
                            f"region {region!r}: {legacy_target.name} carries neither "
                            "component_id nor the legacy composition attributes"
                        )
                except Exception as error:  # noqa: BLE001
                    entry["problems"].append(
                        f"region {region!r}: {legacy_target.name} unreadable: {error}"
                    )
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", help="short run id, resolved against configs_joint/cms_Joint_run<ID>.yaml")
    group.add_argument("--config", type=Path, nargs="+", help="one or more config paths")
    group.add_argument("--all", action="store_true", help="every configs_joint/*.yaml")
    parser.add_argument("--json", type=Path, default=None, help="also write the report as JSON")
    args = parser.parse_args()

    if args.run:
        configs = list((REPO_ROOT / "configs_joint").glob(f"cms_Joint_run{args.run}.yaml"))
        if not configs:
            available = sorted(p.name[len("cms_Joint_run"):-len(".yaml")] for p in (REPO_ROOT / "configs_joint").glob("cms_Joint_run*.yaml"))
            print(f"unknown run {args.run!r}; available: {', '.join(available)}", file=sys.stderr)
            return 2
    elif args.config:
        configs = [p if p.is_absolute() else REPO_ROOT / p for p in args.config]
    else:
        configs = sorted((REPO_ROOT / "configs_joint").glob("*.yaml"))

    reports = []
    failed = 0
    for path in configs:
        if not path.exists():
            print(f"missing config: {path}", file=sys.stderr)
            failed += 1
            continue
        report = check_config(path)
        reports.append(report)
        status = "OK" if not report["problems"] else "FAIL"
        print(f"\n{report['config']}   [{status}]   data_root={report.get('data_root', '?')}")
        for row in report["rows"]:
            datasets = row.get("datasets")
            extra = ""
            if datasets is not None:
                extra = "  datasets: " + ", ".join(
                    f"{name.split('/')[-1]}{'' if ok else '(MISSING)'}" for name, ok in datasets.items()
                )
            fallback = "  [loader used a FALLBACK path]" if row.get("loader_used_fallback") else ""
            print(f"    {row['item']:18s} {row['resolved']}")
            print(f"      {'present' if row['exists'] else 'MISSING':8s} {row['size']:>10s}{extra}{fallback}")
        for problem in report["problems"]:
            print(f"    PROBLEM: {problem}")
        failed += len(report["problems"])

    if args.json:
        args.json.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")
    print(f"\n{len(reports)} config(s) checked, {failed} problem(s)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
