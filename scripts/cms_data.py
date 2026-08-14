from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

# Explicit cache/data-pipeline schema version. Bump whenever the selection,
# pairing, splitting, or caching semantics change in this module, so that
# on-disk caches written by older code can never be reused silently. The
# semantic fingerprint below hashes this module's source together with this
# version, so the constant is primarily a human-readable statement of intent.
DATA_PIPELINE_CACHE_VERSION = 3

# Mass-window tolerance (GeV) used when validating cached x arrays. The stable
# invariant-mass formula is accurate to well below 1 MeV for float32 inputs,
# so one 10 MeV bin is a generous but still meaningful guard.
MASS_WINDOW_VALIDATION_TOL_GEV = 0.01

_EXPECTED_CACHE_KEYS = ("x_train", "x_val", "x_test", "z_train", "z_val", "z_test")


def load_config(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        config = yaml.safe_load(text)
    else:
        config = json.loads(text)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    parent_ref = config.pop("extends", None)
    if parent_ref:
        parent_path = (path.parent / str(parent_ref)).expanduser().resolve()
        parent = load_config(parent_path)
        config = _deep_merge(parent, config)
    config["_config_path"] = str(path)
    return config


def _deep_merge(parent: Any, child: Any) -> Any:
    """Recursively merge ``child`` over ``parent``.

    Dictionaries merge key-by-key. Lists whose items are all dicts carrying a
    ``name`` merge item-by-item by name (child overrides the matching parent
    item and appends new named items). Any other value is replaced by the
    child, so plain lists and scalars behave like normal overrides.
    """
    if isinstance(parent, dict) and isinstance(child, dict):
        merged = deepcopy(parent)
        for key, value in child.items():
            if key in merged:
                merged[key] = _deep_merge(merged[key], value)
            else:
                merged[key] = deepcopy(value)
        return merged
    if (
        isinstance(parent, list)
        and isinstance(child, list)
        and parent
        and child
        and all(isinstance(item, dict) and "name" in item for item in [*parent, *child])
    ):
        merged = deepcopy(parent)
        merged_names = {item["name"] for item in merged}
        for item in child:
            if item["name"] in merged_names:
                idx = next(
                    index
                    for index, existing in enumerate(merged)
                    if existing["name"] == item["name"]
                )
                merged[idx] = _deep_merge(merged[idx], item)
            else:
                merged.append(deepcopy(item))
        return merged
    return deepcopy(child)


def resolve_path(value: str | Path, base_dir: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    base = REPO_ROOT if base_dir is None else base_dir
    return (base / path).resolve()


def resolve_config(config: dict[str, Any], overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = deepcopy(config)
    resolved.pop("_config_path", None)
    paths = resolved.setdefault("paths", {})
    repo_root = resolve_path(paths.get("repo_root", "."), REPO_ROOT)
    data_root = resolve_path(paths.get("data_root", "."), repo_root)
    paths["repo_root"] = str(repo_root)
    paths["data_root"] = str(data_root)

    paths["cms_root_file"] = str(resolve_path(paths["cms_root_file"], data_root))
    if paths.get("theory_prior_files"):
        paths["theory_prior_files"] = [
            str(resolve_path(item, data_root)) for item in paths["theory_prior_files"]
        ]
        if paths.get("theory_prior_file"):
            paths["theory_prior_file"] = str(resolve_path(paths["theory_prior_file"], data_root))
    else:
        paths["theory_prior_file"] = str(resolve_path(paths["theory_prior_file"], data_root))

    output_root = paths.get("output_root", "outputs/cms_doubleelectron")
    paths["output_root"] = str(resolve_path(output_root, repo_root))

    if overrides:
        for key, value in overrides.items():
            if value is not None:
                resolved[key] = value
    return resolved


def save_resolved_config(config: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")


def p4_array(particle) -> np.ndarray:
    import awkward as ak

    pt = ak.to_numpy(particle.pt)
    eta = ak.to_numpy(particle.eta)
    phi = ak.to_numpy(particle.phi)
    mass = ak.to_numpy(particle.mass)

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    energy = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
    return np.stack([px, py, pz, energy], axis=1)


def _pair_p4_rows(
    particles,
    mass_min: float,
    mass_max: float,
) -> np.ndarray:
    """Return charge-ordered opposite-sign pairs as [negative p4, positive p4]."""
    import awkward as ak

    pairs = ak.combinations(particles, 2, fields=["p1", "p2"])
    os_pairs = pairs[(pairs.p1.charge * pairs.p2.charge) < 0]
    negative = ak.where(os_pairs.p1.charge < 0, os_pairs.p1, os_pairs.p2)
    positive = ak.where(os_pairs.p1.charge > 0, os_pairs.p1, os_pairs.p2)

    pair_mass = _pair_mass_ak(negative, positive)

    mass_window = (pair_mass > float(mass_min)) & (pair_mass < float(mass_max))
    negative = ak.flatten(negative[mass_window])
    positive = ak.flatten(positive[mass_window])
    if len(negative) == 0:
        return np.empty((0, 8), dtype=np.float32)
    return np.concatenate([p4_array(negative), p4_array(positive)], axis=1)


def _pair_mass_ak(particle1, particle2) -> np.ndarray:
    """Stable pair invariant mass for awkward particle records.

    Mirrors the cancellation-free transverse-mass / rapidity decomposition in
    ``scripts/physics.py`` (``invariant_mass_np``), so the selection uses the
    same physical definition as evaluation:

        m^2 = m1^2 + m2^2
              + 4 pT1 pT2 [sinh^2(Delta_y/2) + sin^2(Delta_phi/2)]
              + 2 (mT1 mT2 - pT1 pT2) cosh(Delta_y)

    with y_i = asinh(pz_i / mT_i) and mT_i = sqrt(m_i^2 + pT_i^2). For massless
    daughters y reduces to pseudorapidity. Every term is a product of
    non-negative O(1)-scale factors times pT1 pT2, so boosted low-mass pairs
    do not suffer the float32 ``E^2 - p^2`` cancellation.

    Both loaders select ``pt > pt_min > 0``, so the pT -> 0 guard used by the
    generic NumPy helper is not needed here.
    """
    pt1 = particle1.pt
    pt2 = particle2.pt
    m1 = particle1.mass
    m2 = particle2.mass
    dphi = particle1.phi - particle2.phi
    m1sq = m1 * m1
    m2sq = m2 * m2
    pt1sq = pt1 * pt1
    pt2sq = pt2 * pt2
    mt1 = np.sqrt(m1sq + pt1sq)
    mt2 = np.sqrt(m2sq + pt2sq)
    # pz = pT * sinh(eta) exactly, so rapidity is available from the recorded
    # pT/eta/mass fields without an explicit pz branch.
    y1 = np.arcsinh(pt1 * np.sinh(particle1.eta) / mt1)
    y2 = np.arcsinh(pt2 * np.sinh(particle2.eta) / mt2)
    dy = y1 - y2
    angular = 4.0 * pt1 * pt2 * (np.sinh(dy / 2.0) ** 2 + np.sin(dphi / 2.0) ** 2)
    numerator = m1sq * m2sq + m1sq * pt2sq + m2sq * pt1sq
    denominator = mt1 * mt2 + pt1 * pt2
    mass2 = m1sq + m2sq + angular + 2.0 * (numerator / denominator) * np.cosh(dy)
    return np.sqrt(mass2)


def load_cms_electron_x_data(
    cms_root_file: Path,
    selection: dict[str, Any],
    max_selected: int | None = None,
    step_size: str | int = "100 MB",
) -> np.ndarray:
    """Load Z->ee candidates from a CMS ROOT file with chunked iteration.

    Mirrors the muon loader: events are read in bounded chunks, selection is
    applied per event inside each chunk (pairing is per event, so the result is
    identical to a full in-memory load), and iteration stops early once
    ``max_selected`` rows are collected.
    """
    import awkward as ak
    import uproot

    if not cms_root_file.exists():
        raise FileNotFoundError(f"CMS ROOT file not found: {cms_root_file}")

    events = uproot.open(cms_root_file)["Events"]
    branches = [
        "nElectron",
        "Electron_pt",
        "Electron_eta",
        "Electron_phi",
        "Electron_mass",
        "Electron_charge",
        "Electron_pfRelIso03_all",
        "Electron_dxy",
        "Electron_dz",
    ]
    pieces: list[np.ndarray] = []
    selected_count = 0
    for arrays in events.iterate(branches, step_size=step_size, library="ak"):
        electrons = ak.zip(
            {
                "pt": arrays["Electron_pt"],
                "eta": arrays["Electron_eta"],
                "phi": arrays["Electron_phi"],
                "mass": arrays["Electron_mass"],
                "charge": arrays["Electron_charge"],
                "pfRelIso03_all": arrays["Electron_pfRelIso03_all"],
                "dxy": arrays["Electron_dxy"],
                "dz": arrays["Electron_dz"],
            }
        )
        abs_eta = np.abs(electrons.eta)
        outside_ecal_gap = ~((abs_eta > 1.4442) & (abs_eta < 1.566))
        selected = electrons[
            (electrons.pt > selection["electron_pt_min"])
            & (abs_eta < selection["electron_abs_eta_max"])
            & outside_ecal_gap
            & (electrons.pfRelIso03_all < selection["electron_iso_max"])
            & (np.abs(electrons.dxy) < selection["electron_dxy_max"])
            & (np.abs(electrons.dz) < selection["electron_dz_max"])
        ]
        rows = _pair_p4_rows(
            selected,
            mass_min=float(selection["z_mass_min"]),
            mass_max=float(selection["z_mass_max"]),
        )
        if max_selected is not None and int(max_selected) > 0:
            remaining = int(max_selected) - selected_count
            rows = rows[:remaining]
        if len(rows):
            pieces.append(rows)
            selected_count += len(rows)
        if max_selected is not None and int(max_selected) > 0 and selected_count >= int(max_selected):
            break

    if not pieces:
        return np.empty((0, 8), dtype=np.float32)
    return np.concatenate(pieces, axis=0)


def load_cms_muon_x_data(
    cms_root_file: Path,
    selection: dict[str, Any],
    max_selected: int | None = None,
    step_size: str | int = "100 MB",
) -> np.ndarray:
    """Load J/psi dimuon candidates from the reduced DoubleMuParked ROOT file.

    The checked-in reduced file contains only the six basic Muon_* branches, so
    the selection deliberately does not assume isolation, ID, or impact-parameter
    fields. Iteration keeps the 61M-event input from being materialized at once.
    """
    import awkward as ak
    import uproot

    if not cms_root_file.exists():
        raise FileNotFoundError(f"CMS ROOT file not found: {cms_root_file}")

    branches = [
        "nMuon",
        "Muon_pt",
        "Muon_eta",
        "Muon_phi",
        "Muon_mass",
        "Muon_charge",
    ]
    events = uproot.open(cms_root_file)["Events"]
    available = set(events.keys())
    missing = [branch for branch in branches if branch not in available]
    if missing:
        raise KeyError(
            f"Muon channel requires ROOT branches {missing}; available branches: "
            f"{sorted(available)}"
        )

    pieces: list[np.ndarray] = []
    selected_count = 0
    for arrays in events.iterate(branches, step_size=step_size, library="ak"):
        muons = ak.zip(
            {
                "pt": arrays["Muon_pt"],
                "eta": arrays["Muon_eta"],
                "phi": arrays["Muon_phi"],
                "mass": arrays["Muon_mass"],
                "charge": arrays["Muon_charge"],
            }
        )
        keep = (muons.pt > float(selection["muon_pt_min"])) & (
            np.abs(muons.eta) < float(selection["muon_abs_eta_max"])
        )
        if selection.get("muon_pt_max") is not None:
            # Optional junk-tail guard. The reduced skim contains
            # misreconstructed muons with multi-TeV pT inside the mass window
            # (unphysical at 8 TeV); the MG5 prior's muon pT support ends at
            # ~94 GeV, so events beyond that can never be matched by the
            # transport and poison the raw-coordinate SWD gradients.
            keep = keep & (muons.pt < float(selection["muon_pt_max"]))
        selected = muons[keep]
        rows = _pair_p4_rows(
            selected,
            mass_min=float(selection["jpsi_mass_min"]),
            mass_max=float(selection["jpsi_mass_max"]),
        )
        if max_selected is not None and int(max_selected) > 0:
            remaining = int(max_selected) - selected_count
            rows = rows[:remaining]
        if len(rows):
            pieces.append(rows)
            selected_count += len(rows)
        if max_selected is not None and int(max_selected) > 0 and selected_count >= int(max_selected):
            break

    if not pieces:
        return np.empty((0, 8), dtype=np.float32)
    return np.concatenate(pieces, axis=0)


def load_cms_x_data(
    cms_root_file: Path,
    selection: dict[str, Any],
    channel: str = "electron",
    max_selected: int | None = None,
    step_size: str | int = "100 MB",
) -> np.ndarray:
    normalized_channel = str(channel).strip().lower().replace("-", "_")
    if normalized_channel in {"electron", "doubleelectron", "ee"}:
        return load_cms_electron_x_data(
            cms_root_file,
            selection,
            max_selected=max_selected,
            step_size=step_size,
        )
    if normalized_channel in {"muon", "doublemuon", "doublemuons", "mumu", "jpsi_mumu"}:
        return load_cms_muon_x_data(
            cms_root_file,
            selection,
            max_selected=max_selected,
            step_size=step_size,
        )
    raise ValueError(
        f"Unknown CMS data channel {channel!r}. Expected electron/ee or muon/mumu."
    )


def file_fingerprint(path: str | Path) -> dict[str, Any]:
    """Stable identity for a data file: resolved path, size, and mtime."""
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def sha256_fingerprint(path: str | Path) -> str:
    """Full-content SHA-256 fingerprint (used by the explicit preflight)."""
    resolved = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_theory_prior(path: str | Path) -> dict[str, Any]:
    """Read-only structural report for an MG5 HDF5 prior file."""
    resolved = Path(path).expanduser().resolve()
    report: dict[str, Any] = {
        "path": str(resolved),
        "exists": resolved.exists(),
    }
    if not resolved.exists():
        report["error"] = "file not found"
        return report
    report["size_bytes"] = int(resolved.stat().st_size)
    report["sha256"] = sha256_fingerprint(resolved)
    try:
        import h5py

        with h5py.File(resolved, "r") as handle:
            report["top_level_keys"] = sorted(handle.keys())
            dataset_key: str | None = None
            if "FDL" in handle and isinstance(handle["FDL"], h5py.Group) and "zData" in handle["FDL"]:
                dataset_key = "FDL/zData"
                dataset = handle["FDL/zData"]
            elif "zData" in handle:
                dataset_key = "zData"
                dataset = handle["zData"]
            else:
                report["error"] = "Could not find z prior. Expected FDL/zData or zData."
                return report
            array = np.asarray(dataset)
            report["dataset_key"] = dataset_key
            report["shape"] = list(array.shape)
            report["dtype"] = str(array.dtype)
            report["eight_dimensional_compatible"] = (
                array.ndim >= 2 and array.shape[1] >= 8
            )
            report["all_finite"] = bool(np.isfinite(array[:]).all())
            report["finite_check"] = "full-array" if report["all_finite"] else "failed"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def inspect_cms_root(
    path: str | Path,
    selection: dict[str, Any],
    channel: str = "muon",
) -> dict[str, Any]:
    """Read-only structural report for a CMS ROOT file (no selection pass)."""
    resolved = Path(path).expanduser().resolve()
    report: dict[str, Any] = {
        "path": str(resolved),
        "exists": resolved.exists(),
    }
    if not resolved.exists():
        report["error"] = "file not found"
        return report
    report["size_bytes"] = int(resolved.stat().st_size)
    report["sha256"] = sha256_fingerprint(resolved)
    normalized_channel = str(channel).strip().lower().replace("-", "_")
    if normalized_channel in {"muon", "doublemuon", "doublemuons", "mumu", "jpsi_mumu"}:
        required_branches = [
            "nMuon",
            "Muon_pt",
            "Muon_eta",
            "Muon_phi",
            "Muon_mass",
            "Muon_charge",
        ]
    else:
        required_branches = [
            "nElectron",
            "Electron_pt",
            "Electron_eta",
            "Electron_phi",
            "Electron_mass",
            "Electron_charge",
            "Electron_pfRelIso03_all",
            "Electron_dxy",
            "Electron_dz",
        ]
    try:
        import uproot

        with uproot.open(resolved) as root_file:
            report["tree_names"] = sorted(root_file.keys())
            events = root_file["Events"]
            available = set(events.keys())
            report["total_event_count"] = int(events.num_entries)
            report["required_branches"] = required_branches
            report["missing_branches"] = sorted(set(required_branches) - available)
            report["branches_present"] = sorted(available)
            if report["missing_branches"]:
                report["error"] = (
                    "missing required branches: " + ", ".join(report["missing_branches"])
                )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def data_cache_metadata(config: dict[str, Any], num_samples: int | None) -> dict[str, Any]:
    """Everything that can change the selected/split arrays.

    In addition to the resolved configuration and source-file fingerprints,
    the metadata records the explicit pipeline schema version and a semantic
    fingerprint of this module's source. The fingerprint invalidates entries
    produced by older selection/pairing/splitting code even when the
    configuration text is unchanged.
    """
    paths = config["paths"]
    channel = str(config.get("data", {}).get("channel", "electron"))
    selection_key = "muon_selection" if channel.lower() in {
        "muon",
        "doublemuon",
        "doublemuons",
        "mumu",
        "jpsi_mumu",
    } else "electron_selection"
    metadata: dict[str, Any] = {
        "version": DATA_PIPELINE_CACHE_VERSION,
        "pipeline_fingerprint": _pipeline_semantic_fingerprint(),
        "num_samples": None if num_samples is None else int(num_samples),
        "float_type": config.get("float_type", "float32"),
        "seed": int(config.get("seed", 0)),
        "data_split": config["data_split"],
        "data": config.get("data", {"channel": "electron"}),
        selection_key: config[selection_key],
        "cms_root_file": file_fingerprint(paths["cms_root_file"]),
    }
    if paths.get("theory_prior_files"):
        metadata["theory_prior_files"] = [
            file_fingerprint(item) for item in paths["theory_prior_files"]
        ]
        metadata["theory_prior_weights"] = paths.get("theory_prior_weights")
        metadata["theory_prior_mixture_seed"] = int(config.get("seed", 0)) + 17
    else:
        metadata["theory_prior_file"] = file_fingerprint(paths["theory_prior_file"])
    metadata["theory_prior_selection"] = config.get("theory_prior_selection")
    return metadata


def _pipeline_semantic_fingerprint() -> str:
    """Deterministic fingerprint of the data-pipeline implementation.

    Hashes the explicit schema version together with the source of this
    module, which contains every function that defines selection/pairing/
    splitting semantics (``p4_array``, ``_pair_p4_rows``, ``_pair_mass_ak``,
    the ROOT/HDF5 loaders, ``apply_num_samples``, ``split_unpaired``, and
    ``load_and_split``). Editing any of them produces a different cache key,
    so stale arrays cannot be reused silently.
    """
    payload = f"{DATA_PIPELINE_CACHE_VERSION}\n".encode("utf-8") + Path(__file__).read_bytes()
    return hashlib.sha256(payload).hexdigest()


def data_cache_key(metadata: dict[str, Any]) -> str:
    payload = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def _configured_mass_window(config: dict[str, Any]) -> tuple[float, float] | None:
    """Return the channel mass window from the selection config, if complete."""
    data_config = config.get("data", {})
    channel = str(data_config.get("channel", "electron"))
    normalized = channel.strip().lower().replace("-", "_")
    if normalized in {"muon", "doublemuon", "doublemuons", "mumu", "jpsi_mumu"}:
        selection = config.get("muon_selection", {})
        low, high = selection.get("jpsi_mass_min"), selection.get("jpsi_mass_max")
    else:
        selection = config.get("electron_selection", {})
        low, high = selection.get("z_mass_min"), selection.get("z_mass_max")
    if low is None or high is None:
        return None
    return (float(low), float(high))


def _validate_cached_arrays(
    arrays: dict[str, np.ndarray],
    config: dict[str, Any],
) -> str | None:
    """Validate cached split arrays; return a problem description or None.

    Checks the expected key set, rank-2 shape with eight columns, finiteness,
    dtype compatibility with ``float_type``, minimum row counts, configured
    split caps, and (when the selection config declares a mass window) that
    the x arrays live inside it up to a small numerical tolerance.
    """
    from physics import daughter_masses_from_config, invariant_mass_np

    problems: list[str] = []
    actual = set(arrays)
    expected = set(_EXPECTED_CACHE_KEYS)
    if actual != expected:
        problems.append(f"unexpected array keys {sorted(actual ^ expected)}")

    try:
        expected_dtype = np.dtype(config.get("float_type", "float32"))
    except TypeError:
        expected_dtype = None
        problems.append(f"invalid float_type {config.get('float_type')!r}")

    for key in _EXPECTED_CACHE_KEYS:
        if key not in arrays:
            continue
        arr = arrays[key]
        if arr.ndim != 2 or arr.shape[1] != 8:
            problems.append(f"{key} has shape {arr.shape}, expected (N, 8)")
        elif arr.shape[0] < 1:
            problems.append(f"{key} is empty")
        if not np.isfinite(arr).all():
            problems.append(f"{key} contains non-finite values")
        if expected_dtype is not None and arr.dtype != expected_dtype:
            problems.append(f"{key} dtype {arr.dtype} does not match {expected_dtype}")

    split_config = config.get("data_split", {})
    for cap_key, array_keys in (
        ("train_max", ("x_train", "z_train")),
        ("val_max", ("x_val", "z_val")),
        ("test_max", ("x_test", "z_test")),
    ):
        cap = split_config.get(cap_key)
        if cap is None:
            continue
        for key in array_keys:
            if key in arrays and len(arrays[key]) > int(cap):
                problems.append(f"{key} has {len(arrays[key])} rows, exceeding {cap_key}={cap}")

    mass_window = _configured_mass_window(config)
    if mass_window is not None:
        low, high = mass_window
        masses = daughter_masses_from_config(config)
        for key in ("x_train", "x_val", "x_test"):
            if key not in arrays:
                continue
            arr = arrays[key]
            if arr.ndim != 2 or arr.shape[1] != 8 or len(arr) == 0:
                continue
            mass = invariant_mass_np(arr, daughter_masses=masses, stable=True)
            finite = mass[np.isfinite(mass)]
            if len(finite) == 0:
                problems.append(f"{key} has no finite invariant masses")
                continue
            mass_min = float(finite.min())
            mass_max = float(finite.max())
            tolerance = MASS_WINDOW_VALIDATION_TOL_GEV
            if mass_min < low - tolerance or mass_max > high + tolerance:
                problems.append(
                    f"{key} invariant-mass range [{mass_min:.4f}, {mass_max:.4f}] "
                    f"outside configured window [{low}, {high}] "
                    f"(tolerance {tolerance} GeV)"
                )
    return "; ".join(problems) if problems else None


def _write_cache_entry(
    cache_path: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> None:
    """Atomically write the metadata JSON and NPZ cache entry."""
    tmp_metadata = metadata_path.with_name(metadata_path.name + ".tmp")
    tmp_metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_metadata.replace(metadata_path)

    tmp_npz = cache_path.with_name(cache_path.name + ".tmp.npz")
    np.savez(tmp_npz, **{key: arrays[key] for key in _EXPECTED_CACHE_KEYS})
    tmp_npz.replace(cache_path)


def load_and_split_cached(
    config: dict[str, Any],
    num_samples: int | None,
    cache_dir: Path | None,
    use_cache: bool,
    log=print,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """load_and_split() backed by a keyed on-disk cache of the split arrays.

    The cache key covers the resolved config, selection, file fingerprints,
    seed, and sample cap, so a hit always returns the same arrays a fresh load
    would produce. Defaults to <output_root>/.plot_cache so train/eval/plot
    share the same artifacts.
    """
    if not use_cache:
        log("Data cache disabled; loading selected CMS and MG5 rows from source files.")
        return load_and_split(config, num_samples=num_samples), {
            "enabled": False,
            "hit": False,
            "status": "disabled",
        }

    if cache_dir is None:
        cache_dir = Path(config["paths"]["output_root"]) / ".plot_cache"
    cache_dir = cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    metadata = data_cache_metadata(config, num_samples)
    key = data_cache_key(metadata)
    cache_path = cache_dir / f"selected_split_{key}.npz"
    metadata_path = cache_dir / f"selected_split_{key}.json"
    stored_metadata: dict[str, Any] | None = None
    metadata_reason: str | None = None
    metadata_missing = False
    if metadata_path.exists():
        try:
            stored_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            metadata_reason = f"metadata unreadable: {exc}"
    else:
        metadata_missing = True
    if stored_metadata is not None and stored_metadata != metadata:
        metadata_reason = "metadata mismatch (selection/data-pipeline semantics changed)"

    if metadata_reason is None and not metadata_missing and cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                arrays = {key: cached[key] for key in _EXPECTED_CACHE_KEYS}
        except Exception as exc:
            status = "content_invalid"
            invalid_reason = f"cached NPZ unreadable: {exc}"
        else:
            validation_problem = _validate_cached_arrays(arrays, config)
            if validation_problem is None:
                log(f"Loading selected/split data cache: {cache_path}")
                log(
                    "Loaded cache shapes: "
                    + ", ".join(f"{key}={value.shape}" for key, value in arrays.items())
                )
                return arrays, {
                    "enabled": True,
                    "hit": True,
                    "status": "hit",
                    "path": str(cache_path),
                    "key": key,
                }
            status = "content_invalid"
            invalid_reason = f"cached content failed validation: {validation_problem}"
    elif metadata_reason is not None or (
        metadata_missing and (cache_path.exists() or metadata_path.exists())
    ):
        status = "metadata_invalid"
        invalid_reason = metadata_reason or "companion metadata JSON is missing"
    else:
        status = "miss"
        invalid_reason = "no cache entry"

    log(f"Data cache {status} ({invalid_reason}); reading ROOT/HDF5 inputs and applying selection.")
    arrays = load_and_split(config, num_samples=num_samples)
    log(f"Writing selected/split data cache: {cache_path}")
    _write_cache_entry(cache_path, metadata_path, metadata, arrays)
    return arrays, {
        "enabled": True,
        "hit": False,
        "status": status,
        "path": str(cache_path),
        "key": key,
        "reason": invalid_reason,
    }


def filter_theory_prior(
    z_data: np.ndarray,
    selection: dict[str, Any] | None,
) -> np.ndarray:
    """Apply a fiducial/trigger-equivalent selection to a theory prior array.

    Rows are ``[N, 8] = [mu- px, py, pz, E, mu+ px, py, pz, E]``. The
    selection keys mirror the data-side muon selection
    (``muon_pt_min``, ``muon_abs_eta_max``) plus an optional pair mass window
    (``mass_min``, ``mass_max``). The pair mass uses the z-space convention:
    stored energies are authoritative and daughters are massless (matching
    ``DualSpaceFeatureOTLoss`` z-space ``mass_from_energy=True``).

    With ``selection`` falsy the input is returned unchanged (historical
    behavior for configs without ``theory_prior_selection``).
    """
    if not selection:
        return z_data
    z = np.asarray(z_data, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] != 8 or len(z) == 0:
        raise ValueError(f"theory prior must be [N, 8], got {z.shape}")
    px1, py1, pz1 = z[:, 0], z[:, 1], z[:, 2]
    px2, py2, pz2 = z[:, 4], z[:, 5], z[:, 6]
    energy1, energy2 = z[:, 3], z[:, 7]
    pt1 = np.hypot(px1, py1)
    pt2 = np.hypot(px2, py2)
    pabs1 = np.sqrt(px1**2 + py1**2 + pz1**2)
    pabs2 = np.sqrt(px2**2 + py2**2 + pz2**2)
    eta1 = np.arctanh(np.clip(pz1 / pabs1, -1.0 + 1e-7, 1.0 - 1e-7))
    eta2 = np.arctanh(np.clip(pz2 / pabs2, -1.0 + 1e-7, 1.0 - 1e-7))
    mass2 = (energy1 + energy2) ** 2 - (
        (px1 + px2) ** 2 + (py1 + py2) ** 2 + (pz1 + pz2) ** 2
    )
    mass = np.sqrt(np.maximum(mass2, 0.0))

    keep = np.ones(len(z), dtype=bool)
    pt_min = selection.get("muon_pt_min")
    if pt_min is not None:
        pt_min = float(pt_min)
        keep &= (pt1 > pt_min) & (pt2 > pt_min)
    eta_max = selection.get("muon_abs_eta_max")
    if eta_max is not None:
        eta_max = float(eta_max)
        keep &= (np.abs(eta1) < eta_max) & (np.abs(eta2) < eta_max)
    mass_min = selection.get("mass_min")
    mass_max = selection.get("mass_max")
    if mass_min is not None and mass_max is not None:
        keep &= (mass > float(mass_min)) & (mass < float(mass_max))

    filtered = z_data[keep]
    if len(filtered) == 0:
        raise ValueError(
            "theory_prior_selection removed every prior event; relax the prior "
            "selection or check the prior file."
        )
    return filtered


def load_theory_prior_z(theory_prior_file: Path) -> np.ndarray:
    import h5py

    if not theory_prior_file.exists():
        raise FileNotFoundError(f"MG5 HDF5 prior file not found: {theory_prior_file}")

    with h5py.File(theory_prior_file, "r") as f:
        if "FDL" in f and isinstance(f["FDL"], h5py.Group) and "zData" in f["FDL"]:
            z_data = np.asarray(f["FDL/zData"])
        elif "zData" in f:
            z_data = np.asarray(f["zData"])
        else:
            raise KeyError("Could not find z prior. Expected FDL/zData or zData.")
    return z_data[:, :8]


def load_theory_prior_z_mixture(
    theory_prior_files: list[str | Path],
    weights: list[float] | None,
    seed: int,
) -> np.ndarray:
    arrays = [load_theory_prior_z(Path(path)) for path in theory_prior_files]
    if not arrays:
        raise ValueError("theory_prior_files is empty.")
    if weights is None:
        weights_arr = np.ones(len(arrays), dtype=float) / len(arrays)
    else:
        weights_arr = np.asarray(weights, dtype=float)
        if weights_arr.shape != (len(arrays),):
            raise ValueError("theory_prior_weights must match theory_prior_files length.")
        total_weight = weights_arr.sum()
        if not np.isfinite(total_weight) or total_weight <= 0.0:
            raise ValueError("theory_prior_weights must sum to a positive finite value.")
        weights_arr = weights_arr / total_weight

    total = int(sum(len(array) for array in arrays))
    counts = np.floor(weights_arr * total).astype(int)
    while counts.sum() < total:
        counts[int(np.argmax(weights_arr * total - counts))] += 1
    rng = np.random.default_rng(seed)
    pieces = []
    for array, count in zip(arrays, counts):
        replace = count > len(array)
        idx = rng.choice(len(array), size=int(count), replace=replace)
        pieces.append(array[idx])
    mixed = np.concatenate(pieces, axis=0)
    return mixed[rng.permutation(len(mixed))]


def apply_num_samples(arr: np.ndarray, num_samples: int | None) -> np.ndarray:
    if num_samples is None or int(num_samples) <= 0:
        return arr
    return arr[: min(len(arr), int(num_samples))]


def split_unpaired(
    arr: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    train_max: int | None = None,
    val_max: int | None = None,
    test_max: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = arr[rng.permutation(len(arr))]
    ratio_train_size = int(len(shuffled) * train_ratio)
    ratio_val_size = int(len(shuffled) * val_ratio)
    ratio_test_size = len(shuffled) - ratio_train_size - ratio_val_size
    train_size = (
        min(ratio_train_size, int(train_max))
        if train_max is not None
        else ratio_train_size
    )
    val_size = (
        min(ratio_val_size, int(val_max)) if val_max is not None else ratio_val_size
    )
    test_size = (
        min(ratio_test_size, int(test_max)) if test_max is not None else ratio_test_size
    )
    return (
        shuffled[:train_size],
        shuffled[train_size : train_size + val_size],
        shuffled[train_size + val_size : train_size + val_size + test_size],
    )


def load_and_split(config: dict[str, Any], num_samples: int | None = None) -> dict[str, np.ndarray]:
    paths = config["paths"]
    data_config = config.get("data", {})
    channel = str(data_config.get("channel", "electron"))
    normalized_channel = channel.strip().lower().replace("-", "_")
    if normalized_channel in {"electron", "doubleelectron", "ee"}:
        selection = config["electron_selection"]
    elif normalized_channel in {"muon", "doublemuon", "doublemuons", "mumu", "jpsi_mumu"}:
        selection = config["muon_selection"]
    else:
        raise ValueError(
            f"Unknown CMS data channel {channel!r}. Expected electron/ee or muon/mumu."
        )
    x_data = load_cms_x_data(
        Path(paths["cms_root_file"]),
        selection,
        channel=channel,
        max_selected=num_samples,
        step_size=data_config.get("root_step_size", "100 MB"),
    )
    if paths.get("theory_prior_files"):
        z_data = load_theory_prior_z_mixture(
            paths["theory_prior_files"],
            paths.get("theory_prior_weights"),
            seed=int(config.get("seed", 0)) + 17,
        )
    else:
        z_data = load_theory_prior_z(Path(paths["theory_prior_file"]))
    z_data = filter_theory_prior(z_data, config.get("theory_prior_selection"))

    x_data = apply_num_samples(x_data, num_samples)
    z_data = apply_num_samples(z_data, num_samples)
    if len(x_data) < 3 or len(z_data) < 3:
        raise ValueError("Need at least 3 selected CMS and MG5 events after --num-samples.")

    dtype = np.dtype(config.get("float_type", "float32")).name
    split_config = config["data_split"]
    seed = int(config.get("seed", 0))
    x_train, x_val, x_test = split_unpaired(
        x_data,
        float(split_config["train_ratio"]),
        float(split_config["val_ratio"]),
        seed,
        train_max=split_config.get("train_max"),
        val_max=split_config.get("val_max"),
        test_max=split_config.get("test_max"),
    )
    z_train, z_val, z_test = split_unpaired(
        z_data,
        float(split_config["train_ratio"]),
        float(split_config["val_ratio"]),
        seed + 1,
        train_max=split_config.get("train_max"),
        val_max=split_config.get("val_max"),
        test_max=split_config.get("test_max"),
    )
    arrays = {
        "x_train": x_train,
        "x_val": x_val,
        "x_test": x_test,
        "z_train": z_train,
        "z_val": z_val,
        "z_test": z_test,
    }
    return {key: value.astype(dtype, copy=False) for key, value in arrays.items()}


def array_stats(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    return mean, np.where(std == 0, 1.0, std)
