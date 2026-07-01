from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


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
    config["_config_path"] = str(path)
    return config


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


def load_cms_x_data(cms_root_file: Path, selection: dict[str, float]) -> np.ndarray:
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
    arrays = events.arrays(branches, library="ak")

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

    pairs = ak.combinations(selected, 2, fields=["e1", "e2"])
    os_pairs = pairs[(pairs.e1.charge * pairs.e2.charge) < 0]
    e_minus = ak.where(os_pairs.e1.charge < 0, os_pairs.e1, os_pairs.e2)
    e_plus = ak.where(os_pairs.e1.charge > 0, os_pairs.e1, os_pairs.e2)

    px = e_minus.pt * np.cos(e_minus.phi) + e_plus.pt * np.cos(e_plus.phi)
    py = e_minus.pt * np.sin(e_minus.phi) + e_plus.pt * np.sin(e_plus.phi)
    pz = e_minus.pt * np.sinh(e_minus.eta) + e_plus.pt * np.sinh(e_plus.eta)
    e1 = np.sqrt((e_minus.pt * np.cosh(e_minus.eta)) ** 2 + e_minus.mass**2)
    e2 = np.sqrt((e_plus.pt * np.cosh(e_plus.eta)) ** 2 + e_plus.mass**2)
    mass2 = (e1 + e2) ** 2 - px**2 - py**2 - pz**2
    m_ee = np.sqrt(ak.where(mass2 > 0, mass2, 0))

    z_window = (m_ee > selection["z_mass_min"]) & (m_ee < selection["z_mass_max"])
    e_minus = ak.flatten(e_minus[z_window])
    e_plus = ak.flatten(e_plus[z_window])
    return np.concatenate([p4_array(e_minus), p4_array(e_plus)], axis=1)


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = arr[rng.permutation(len(arr))]
    train_size = int(len(shuffled) * train_ratio)
    val_size = int(len(shuffled) * val_ratio)
    return (
        shuffled[:train_size],
        shuffled[train_size : train_size + val_size],
        shuffled[train_size + val_size :],
    )


def load_and_split(config: dict[str, Any], num_samples: int | None = None) -> dict[str, np.ndarray]:
    paths = config["paths"]
    x_data = load_cms_x_data(Path(paths["cms_root_file"]), config["electron_selection"])
    if paths.get("theory_prior_files"):
        z_data = load_theory_prior_z_mixture(
            paths["theory_prior_files"],
            paths.get("theory_prior_weights"),
            seed=int(config.get("seed", 0)) + 17,
        )
    else:
        z_data = load_theory_prior_z(Path(paths["theory_prior_file"]))

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
    )
    z_train, z_val, z_test = split_unpaired(
        z_data,
        float(split_config["train_ratio"]),
        float(split_config["val_ratio"]),
        seed + 1,
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
