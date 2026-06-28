"""Server entrypoint for CMS DoubleElectron OTUS training.

The training behavior is configured by train_config.json so data paths,
hyperparameters, and stage ordering can be changed without editing this file.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train OTUS on CMS DoubleElectron data.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("train_config.json"),
        help="Path to the JSON training config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data and build the model/loaders, then exit before training.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def apply_environment(config: dict) -> None:
    env = config.get("environment", {})
    mapping = {
        "cuda_visible_devices": "CUDA_VISIBLE_DEVICES",
        "kmp_duplicate_lib_ok": "KMP_DUPLICATE_LIB_OK",
        "omp_num_threads": "OMP_NUM_THREADS",
        "mkl_num_threads": "MKL_NUM_THREADS",
        "pytorch_enable_mps_fallback": "PYTORCH_ENABLE_MPS_FALLBACK",
    }
    for config_key, env_key in mapping.items():
        value = env.get(config_key)
        if value is not None:
            os.environ[env_key] = str(value)


def select_device(requested: str) -> "torch.device":
    requested = str(requested or "auto").lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("Requested CUDA, but CUDA is not available. Falling back to CPU.")
        return torch.device("cpu")
    if device.type == "mps":
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not mps_available:
            print("Requested MPS, but MPS is not available. Falling back to CPU.")
            return torch.device("cpu")
    return device


def resolve_path(path_value: str, base_dir: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def first_tensor(value):
    if isinstance(value, (tuple, list)):
        return value[0]
    return value


def split_unpaired(arr, train_ratio: float, val_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    arr = arr[rng.permutation(len(arr))]

    train_size = int(len(arr) * train_ratio)
    val_size = int(len(arr) * val_ratio)

    train = arr[:train_size]
    val = arr[train_size : train_size + val_size]
    test = arr[train_size + val_size :]
    return train, val, test


def p4_array(particle):
    pt = ak.to_numpy(particle.pt)
    eta = ak.to_numpy(particle.eta)
    phi = ak.to_numpy(particle.phi)
    mass = ak.to_numpy(particle.mass)

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    energy = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
    return np.stack([px, py, pz, energy], axis=1)


def load_cms_x_data(cms_root_file: Path, selection: dict):
    if not cms_root_file.exists():
        raise FileNotFoundError(f"CMS ROOT file not found: {cms_root_file}")

    cms_file = uproot.open(cms_root_file)
    events = cms_file["Events"]

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

    x_data = np.concatenate([p4_array(e_minus), p4_array(e_plus)], axis=1)
    print("Loaded CMS ROOT file:", cms_root_file)
    print("CMS events:", events.num_entries)
    print("CMS x_data shape:", x_data.shape)
    return x_data


def load_theory_prior_z(theory_prior_file: Path):
    if not theory_prior_file.exists():
        raise FileNotFoundError(f"MG5 HDF5 prior file not found: {theory_prior_file}")

    with h5py.File(theory_prior_file, "r") as f:
        if "FDL" in f and isinstance(f["FDL"], h5py.Group) and "zData" in f["FDL"]:
            z_data = np.asarray(f["FDL/zData"])
        elif "zData" in f:
            z_data = np.asarray(f["zData"])
        else:
            raise KeyError("Could not find z prior. Expected FDL/zData or zData.")

    z_data = z_data[:, :8]
    print("Loaded MG5 z prior:", theory_prior_file)
    print("MG5 z_data shape:", z_data.shape)
    return z_data


class ResampleTensorLoader:
    """Independent resampling loader for unpaired OTUS training."""

    def __init__(self, tensor, batch_size: int, steps_per_epoch: int):
        self.tensor = tensor
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.n = int(tensor.shape[0])
        self.dataset = tensor

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            idx = torch.randint(
                low=0,
                high=self.n,
                size=(self.batch_size,),
                device=self.tensor.device,
            )
            yield self.tensor.index_select(0, idx)


def make_training_tensor(arr, name: str, loader_device, pin_cpu_memory: bool):
    tensor = torch.as_tensor(arr)
    if loader_device.type == "cuda":
        tensor = tensor.to(loader_device)
    elif pin_cpu_memory and torch.cuda.is_available():
        tensor = tensor.pin_memory()
    print(f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}")
    return tensor


def build_loaders(config: dict, x_train, x_val, z_train, z_val, device_obj):
    loader_config = config["loaders"]
    train_batch_size = min(
        int(loader_config["train_batch_size"]),
        max(1, min(len(x_train), len(z_train))),
    )
    eval_batch_size = min(
        int(loader_config["eval_batch_size"]),
        max(1, min(len(x_val), len(z_val))),
    )

    steps_per_epoch = math.ceil(max(len(x_train), len(z_train)) / train_batch_size)
    eval_steps_per_epoch = math.ceil(max(len(x_val), len(z_val)) / eval_batch_size)

    preload_to_gpu = bool(loader_config.get("preload_data_to_gpu", True)) and device_obj.type == "cuda"
    loader_device = device_obj if preload_to_gpu else torch.device("cpu")
    pin_cpu_memory = (
        bool(loader_config.get("pin_memory_if_cpu_loader", True))
        and loader_device.type == "cpu"
        and torch.cuda.is_available()
    )

    print("loader_device:", loader_device)
    print("pin_cpu_memory:", pin_cpu_memory)

    x_train_tensor = make_training_tensor(x_train, "x_train", loader_device, pin_cpu_memory)
    z_train_tensor = make_training_tensor(z_train, "z_train", loader_device, pin_cpu_memory)
    x_val_tensor = make_training_tensor(x_val, "x_val", loader_device, pin_cpu_memory)
    z_val_tensor = make_training_tensor(z_val, "z_val", loader_device, pin_cpu_memory)

    train_loaders = (
        ResampleTensorLoader(x_train_tensor, train_batch_size, steps_per_epoch),
        ResampleTensorLoader(z_train_tensor, train_batch_size, steps_per_epoch),
    )
    eval_loaders = (
        ResampleTensorLoader(x_val_tensor, eval_batch_size, eval_steps_per_epoch),
        ResampleTensorLoader(z_val_tensor, eval_batch_size, eval_steps_per_epoch),
    )

    print("train_batch_size:", train_batch_size)
    print("eval_batch_size:", eval_batch_size)
    print("training batches per epoch:", len(train_loaders[0]))
    print("eval batches:", len(eval_loaders[0]))
    return train_loaders, eval_loaders


class ZLossFactory:
    def __init__(self, x_train, z_train, loss_config: dict):
        self.x_train_mean = np.mean(x_train, axis=0)
        self.x_train_std = np.where(np.std(x_train, axis=0) == 0, 1.0, np.std(x_train, axis=0))
        self.z_train_mean = np.mean(z_train, axis=0)
        self.z_train_std = np.where(np.std(z_train, axis=0) == 0, 1.0, np.std(z_train, axis=0))

        self.weights = {
            "lambda_z8": 0.5,
            "lambda_marginal": 0.8,
            "lambda_mass": 1.8,
            "lambda_phys": 0.7,
            "lambda_transverse": 0.8,
            "lambda_longitudinal": 0.8,
            "lambda_tail": 0.2,
        }
        self.weights.update(loss_config)
        self.num_slices = 1000
        self.p = 2
        self.eps = 1e-6

        with torch.no_grad():
            z_train_torch = torch.as_tensor(z_train, dtype=torch.float32)
            m_z_train = self.dilepton_mass_torch(z_train_torch)
            self.m_z_mean = m_z_train.mean().detach()
            self.m_z_std = m_z_train.std().detach()

            fz_train = self.z_physics_features(z_train_torch)
            self.fz_mean = fz_train.mean(dim=0).detach()
            self.fz_std = fz_train.std(dim=0).detach()

            trans_train = self.z_transverse_features(z_train_torch)
            self.trans_mean = trans_train.mean(dim=0).detach()
            self.trans_std = trans_train.std(dim=0).detach()

            long_train = self.z_longitudinal_features(z_train_torch)
            self.long_mean = long_train.mean(dim=0).detach()
            self.long_std = long_train.std(dim=0).detach()

        print("MG5 z mass mean/std:", self.m_z_mean, self.m_z_std)

    def set_num_slices(self, num_slices: int) -> None:
        self.num_slices = int(num_slices)

    def to_like(self, value, ref):
        if isinstance(value, torch.Tensor):
            return value.detach().to(dtype=ref.dtype, device=ref.device)
        return torch.as_tensor(value, dtype=ref.dtype, device=ref.device)

    def wasserstein_1d_sorted(self, a, b):
        a = torch.sort(a.reshape(-1))[0]
        b = torch.sort(b.reshape(-1))[0]
        n = min(a.numel(), b.numel())
        return torch.mean(torch.abs(a[:n] - b[:n]) ** self.p)

    def weighted_marginal_wd(self, a, b, weights):
        weights = self.to_like(weights, a)
        loss = a.new_tensor(0.0)
        for j in range(a.shape[1]):
            loss = loss + weights[j] * self.wasserstein_1d_sorted(a[:, j], b[:, j])
        return loss / torch.sum(weights)

    def tail_wasserstein_abs(self, a, b, tail_frac=0.20):
        a = torch.sort(torch.abs(a.reshape(-1)))[0]
        b = torch.sort(torch.abs(b.reshape(-1)))[0]
        n = min(a.numel(), b.numel())
        start = int((1.0 - tail_frac) * n)
        start = max(0, min(start, n - 1))
        return torch.mean(torch.abs(a[start:n] - b[start:n]) ** self.p)

    def standardize_x_raw(self, x):
        mean = self.to_like(self.x_train_mean, x)
        std = self.to_like(self.x_train_std, x)
        return (x - mean) / (std + self.eps)

    def standardize_z_raw(self, z):
        mean = self.to_like(self.z_train_mean, z)
        std = self.to_like(self.z_train_std, z)
        return (z - mean) / (std + self.eps)

    def paired_mse_standardized(self, a, b, standardize_fun):
        return torch.mean((standardize_fun(a) - standardize_fun(b)) ** 2)

    def dilepton_mass_torch(self, z):
        px = z[:, 0] + z[:, 4]
        py = z[:, 1] + z[:, 5]
        pz = z[:, 2] + z[:, 6]
        energy = z[:, 3] + z[:, 7]
        m2 = energy**2 - px**2 - py**2 - pz**2
        return torch.sqrt(torch.clamp(m2, min=self.eps))

    def safe_eta_from_pt_pz(self, pt, pz):
        p = torch.sqrt(pt**2 + pz**2 + self.eps)
        return torch.log(torch.clamp(p + pz, min=self.eps) / (pt + self.eps))

    def z_physics_features(self, z):
        px1, py1, pz1, energy1 = z[:, 0], z[:, 1], z[:, 2], z[:, 3]
        px2, py2, pz2, energy2 = z[:, 4], z[:, 5], z[:, 6], z[:, 7]

        pt1 = torch.sqrt(px1**2 + py1**2 + self.eps)
        pt2 = torch.sqrt(px2**2 + py2**2 + self.eps)
        eta1 = self.safe_eta_from_pt_pz(pt1, pz1)
        eta2 = self.safe_eta_from_pt_pz(pt2, pz2)

        px = px1 + px2
        py = py1 + py2
        pz = pz1 + pz2
        energy = energy1 + energy2
        mll = torch.sqrt(torch.clamp(energy**2 - px**2 - py**2 - pz**2, min=self.eps))
        ptll = torch.sqrt(px**2 + py**2 + self.eps)
        yll = 0.5 * torch.log(
            torch.clamp(energy + pz, min=self.eps) / torch.clamp(energy - pz, min=self.eps)
        )

        dot = px1 * px2 + py1 * py2
        cross = px1 * py2 - py1 * px2
        cos_dphi = torch.clamp(dot / (pt1 * pt2 + self.eps), min=-1.0, max=1.0)
        sin_dphi = torch.clamp(cross / (pt1 * pt2 + self.eps), min=-1.0, max=1.0)

        return torch.stack([pt1, eta1, pt2, eta2, mll, ptll, yll, cos_dphi, sin_dphi], dim=1)

    def z_transverse_features(self, z):
        px1, py1 = z[:, 0], z[:, 1]
        px2, py2 = z[:, 4], z[:, 5]
        pt1 = torch.sqrt(px1**2 + py1**2 + self.eps)
        pt2 = torch.sqrt(px2**2 + py2**2 + self.eps)
        px_sum = px1 + px2
        py_sum = py1 + py2
        ptll = torch.sqrt(px_sum**2 + py_sum**2 + self.eps)
        px_diff = px1 - px2
        py_diff = py1 - py2
        dot = px1 * px2 + py1 * py2
        cross = px1 * py2 - py1 * px2
        cos_dphi = torch.clamp(dot / (pt1 * pt2 + self.eps), min=-1.0, max=1.0)
        sin_dphi = torch.clamp(cross / (pt1 * pt2 + self.eps), min=-1.0, max=1.0)
        return torch.stack(
            [
                px1,
                py1,
                px2,
                py2,
                pt1,
                pt2,
                px_sum,
                py_sum,
                ptll,
                px_diff,
                py_diff,
                cos_dphi,
                sin_dphi,
            ],
            dim=1,
        )

    def z_longitudinal_features(self, z):
        px1, py1, pz1, energy1 = z[:, 0], z[:, 1], z[:, 2], z[:, 3]
        px2, py2, pz2, energy2 = z[:, 4], z[:, 5], z[:, 6], z[:, 7]
        pt1 = torch.sqrt(px1**2 + py1**2 + self.eps)
        pt2 = torch.sqrt(px2**2 + py2**2 + self.eps)
        eta1 = self.safe_eta_from_pt_pz(pt1, pz1)
        eta2 = self.safe_eta_from_pt_pz(pt2, pz2)
        pz_sum = pz1 + pz2
        pz_diff = pz1 - pz2
        energy = energy1 + energy2
        yll = 0.5 * torch.log(
            torch.clamp(energy + pz_sum, min=self.eps) / torch.clamp(energy - pz_sum, min=self.eps)
        )
        return torch.stack([pz1, pz2, pz_sum, pz_diff, eta1, eta2, yll], dim=1)

    def standardize_mass(self, mass):
        mean = self.to_like(self.m_z_mean, mass)
        std = self.to_like(self.m_z_std, mass)
        return (mass - mean) / (std + self.eps)

    def standardize_features(self, features, mean, std):
        mean = self.to_like(mean, features)
        std = self.to_like(std, features)
        return (features - mean) / (std + self.eps)

    def z_channel_marginal_loss(self, z, z_tilde):
        z_s = self.standardize_z_raw(z)
        zt_s = self.standardize_z_raw(z_tilde)
        channel_weights = [6.0, 6.0, 3.0, 0.5, 6.0, 6.0, 3.0, 0.5]
        return self.weighted_marginal_wd(z_s, zt_s, channel_weights)

    def z_transverse_loss(self, z, z_tilde):
        f_z = self.standardize_features(self.z_transverse_features(z), self.trans_mean, self.trans_std)
        f_zt = self.standardize_features(self.z_transverse_features(z_tilde), self.trans_mean, self.trans_std)
        weights = [5.0, 5.0, 5.0, 5.0, 2.0, 2.0, 3.0, 3.0, 2.0, 3.0, 3.0, 1.0, 1.0]
        loss_marginal = self.weighted_marginal_wd(f_z, f_zt, weights)
        loss_joint = sliced_wd(f_z, f_zt, self.num_slices, self.p)

        z_s = self.standardize_z_raw(z)
        zt_s = self.standardize_z_raw(z_tilde)
        loss_tail = (
            self.tail_wasserstein_abs(z_s[:, 0], zt_s[:, 0])
            + self.tail_wasserstein_abs(z_s[:, 1], zt_s[:, 1])
            + self.tail_wasserstein_abs(z_s[:, 4], zt_s[:, 4])
            + self.tail_wasserstein_abs(z_s[:, 5], zt_s[:, 5])
        ) / 4.0
        return loss_marginal + 0.5 * loss_joint + 0.5 * loss_tail

    def z_longitudinal_loss(self, z, z_tilde):
        f_z = self.standardize_features(self.z_longitudinal_features(z), self.long_mean, self.long_std)
        f_zt = self.standardize_features(self.z_longitudinal_features(z_tilde), self.long_mean, self.long_std)
        weights = [4.0, 4.0, 3.0, 2.0, 2.0, 2.0, 2.0]
        loss_marginal = self.weighted_marginal_wd(f_z, f_zt, weights)
        loss_joint = sliced_wd(f_z, f_zt, self.num_slices, self.p)

        z_s = self.standardize_z_raw(z)
        zt_s = self.standardize_z_raw(z_tilde)
        loss_tail = (
            self.tail_wasserstein_abs(z_s[:, 2], zt_s[:, 2])
            + self.tail_wasserstein_abs(z_s[:, 6], zt_s[:, 6])
        ) / 2.0
        return loss_marginal + 0.5 * loss_joint + 0.5 * loss_tail

    def __call__(self, z, z_tilde):
        z_std = self.standardize_z_raw(z)
        zt_std = self.standardize_z_raw(z_tilde)

        loss_z_8d = sliced_wd(z_std, zt_std, self.num_slices, self.p)
        loss_marginal = self.z_channel_marginal_loss(z, z_tilde)

        m_z = self.standardize_mass(self.dilepton_mass_torch(z))
        m_zt = self.standardize_mass(self.dilepton_mass_torch(z_tilde))
        loss_mass = self.wasserstein_1d_sorted(m_z, m_zt)

        f_z = self.standardize_features(self.z_physics_features(z), self.fz_mean, self.fz_std)
        f_zt = self.standardize_features(self.z_physics_features(z_tilde), self.fz_mean, self.fz_std)
        loss_phys = sliced_wd(f_z, f_zt, self.num_slices, self.p)

        loss_transverse = self.z_transverse_loss(z, z_tilde)
        loss_longitudinal = self.z_longitudinal_loss(z, z_tilde)

        loss_tail = z_std.new_tensor(0.0)
        for j in range(8):
            loss_tail = loss_tail + self.tail_wasserstein_abs(z_std[:, j], zt_std[:, j])
        loss_tail = loss_tail / 8.0

        return (
            self.weights["lambda_z8"] * loss_z_8d
            + self.weights["lambda_marginal"] * loss_marginal
            + self.weights["lambda_mass"] * loss_mass
            + self.weights["lambda_phys"] * loss_phys
            + self.weights["lambda_transverse"] * loss_transverse
            + self.weights["lambda_longitudinal"] * loss_longitudinal
            + self.weights["lambda_tail"] * loss_tail
        )


def build_model(config: dict, x_dim: int, z_dim: int, x_train_mean, x_train_std, z_train_mean, z_train_std):
    model_config = config["model"]
    hidden_layer_dims = int(model_config["num_hidden_layers"]) * [int(model_config["dim_per_hidden_layer"])]
    activation = getattr(torch.nn, model_config.get("activation", "ReLU"))

    common_kwargs = {
        "x_dim": x_dim,
        "z_dim": z_dim,
        "raw_io": bool(model_config.get("raw_io", True)),
        "x_stats": np.stack([x_train_mean, x_train_std]),
        "z_stats": np.stack([z_train_mean, z_train_std]),
        "x_inv_masses": np.zeros(2),
        "z_inv_masses": np.zeros(2),
        "stoch_enc": bool(model_config.get("stoch_enc", True)),
        "stoch_dec": bool(model_config.get("stoch_dec", True)),
        "activation": activation,
    }

    sigma_fun = model_config.get("sigma_fun")
    if sigma_fun:
        common_kwargs["sigma_fun"] = sigma_fun

    class_name = model_config.get("class", "CondNoiseAutoencoder")
    if class_name == "CondNoiseAutoencoder":
        from models import CondNoiseAutoencoder

        model = CondNoiseAutoencoder(hidden_layer_dims=hidden_layer_dims, **common_kwargs)
    elif class_name == "Autoencoder":
        from models import Autoencoder, CondNoiseMLP, StochasticResNet

        conditional_models = {
            "CondNoiseMLP": CondNoiseMLP,
            "StochasticResNet": StochasticResNet,
        }
        conditional_name = model_config.get("conditional_model", "CondNoiseMLP")
        if conditional_name not in conditional_models:
            raise ValueError(f"Unknown conditional_model: {conditional_name}")
        model = Autoencoder(
            ConditionalModel=conditional_models[conditional_name],
            encoder_hidden_layer_dims=hidden_layer_dims,
            **common_kwargs,
        )
    else:
        raise ValueError(f"Unknown model class: {class_name}")

    print("Using model class:", class_name)
    print("Hidden layer dims:", hidden_layer_dims)
    return model


def set_trainable(model, train_encoder=True, train_decoder=True):
    for param in model.encoder.parameters():
        param.requires_grad_(train_encoder)
    for param in model.decoder.parameters():
        param.requires_grad_(train_decoder)


def make_stage_config(config: dict, stage: dict):
    model_config = config["model"]
    return {
        "num_hidden_layers": model_config["num_hidden_layers"],
        "dim_per_hidden_layer": model_config["dim_per_hidden_layer"],
        "epochs": stage["epochs"],
        "lr": stage["lr"],
        "beta": stage["beta"],
        "lamb": stage["lamb"],
        "tau": stage["tau"],
        "rho": stage["rho"],
        "nu_e": stage["nu_e"],
        "nu_d": stage["nu_d"],
        "num_slices": stage["num_slices"],
    }


def ensure_history(history):
    if history is None:
        history = {}
    for key in ["epoch", "train_loss", "train_x_loss", "train_z_loss", "eval_loss"]:
        history.setdefault(key, [])
        if not isinstance(history[key], list):
            history[key] = list(history[key])
    return history


def as_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def last_epoch_from_history(history) -> int:
    if history is None or "epoch" not in history or len(history["epoch"]) == 0:
        return 0
    return int(max(as_float(v) for v in history["epoch"]))


def append_history(history, epoch, train_loss, train_x_loss, train_z_loss, eval_loss):
    history = ensure_history(history)
    history["epoch"].append(epoch)
    history["train_loss"].append(as_float(train_loss))
    history["train_x_loss"].append(as_float(train_x_loss))
    history["train_z_loss"].append(as_float(train_z_loss))
    history["eval_loss"].append(as_float(eval_loss))
    return history


def z_cycle_losses_for_batch(model, x_batch, z_batch, stage: dict, loss_factory: ZLossFactory):
    z_from_x = first_tensor(model.encode(x_batch))
    x_reco = first_tensor(model.decode(z_from_x))
    loss_x_reco = loss_factory.paired_mse_standardized(
        x_reco,
        x_batch,
        loss_factory.standardize_x_raw,
    )

    loss_z_prior = loss_factory(z_batch, z_from_x)

    x_from_z = first_tensor(model.decode(z_batch))
    z_cycle = first_tensor(model.encode(x_from_z))
    loss_z_cycle = loss_factory.paired_mse_standardized(
        z_cycle,
        z_batch,
        loss_factory.standardize_z_raw,
    )

    loss_transverse = loss_factory.z_transverse_loss(z_batch, z_from_x)
    loss_total = (
        stage.get("beta", 1.0) * loss_x_reco
        + stage.get("lamb", 1.0) * loss_z_prior
        + stage.get("z_cycle_weight", 0.0) * loss_z_cycle
        + stage.get("transverse_weight", 0.0) * loss_transverse
    )

    return {
        "total": loss_total,
        "x_reco": loss_x_reco,
        "z_prior": loss_z_prior,
        "z_cycle": loss_z_cycle,
        "transverse": loss_transverse,
    }


def run_z_cycle_stage(model, train_loaders, eval_loaders, stage: dict, history, loss_factory, device_obj):
    freeze_encoder = stage.get("freeze_encoder", False)
    freeze_decoder = stage.get("freeze_decoder", False)
    set_trainable(model, train_encoder=not freeze_encoder, train_decoder=not freeze_decoder)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError(f"Stage {stage['name']} has no trainable parameters.")

    optimizer = torch.optim.Adam(trainable_params, lr=stage["lr"])
    history = ensure_history(history)
    start_epoch = last_epoch_from_history(history)

    print("Custom z-cycle stage")
    print("Train encoder:", not freeze_encoder)
    print("Train decoder:", not freeze_decoder)
    print("z_cycle_weight:", stage.get("z_cycle_weight", 0.0))
    print("transverse_weight:", stage.get("transverse_weight", 0.0))

    final_eval = None
    for local_epoch in range(1, int(stage["epochs"]) + 1):
        model.train()
        sums = {"total": 0.0, "x_reco": 0.0, "z_prior": 0.0, "z_cycle": 0.0, "transverse": 0.0}
        nbatches = 0

        for x_batch, z_batch in zip(train_loaders[0], train_loaders[1]):
            x_batch = x_batch.to(device_obj)
            z_batch = z_batch.to(device_obj)

            optimizer.zero_grad()
            losses = z_cycle_losses_for_batch(model, x_batch, z_batch, stage, loss_factory)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=10.0)
            optimizer.step()

            for key in sums:
                sums[key] += as_float(losses[key])
            nbatches += 1

        for key in sums:
            sums[key] /= max(1, nbatches)

        should_log = (
            local_epoch == 1
            or local_epoch == int(stage["epochs"])
            or local_epoch % int(stage["log_freq"]) == 0
        )
        if not should_log:
            continue

        model.eval()
        eval_sums = {"total": 0.0, "x_reco": 0.0, "z_prior": 0.0, "z_cycle": 0.0, "transverse": 0.0}
        eval_batches = 0

        with torch.no_grad():
            for x_eval, z_eval in zip(eval_loaders[0], eval_loaders[1]):
                x_eval = x_eval.to(device_obj)
                z_eval = z_eval.to(device_obj)
                eval_losses_batch = z_cycle_losses_for_batch(model, x_eval, z_eval, stage, loss_factory)
                for key in eval_sums:
                    eval_sums[key] += as_float(eval_losses_batch[key])
                eval_batches += 1

        for key in eval_sums:
            eval_sums[key] /= max(1, eval_batches)

        absolute_epoch = start_epoch + local_epoch
        history = append_history(
            history,
            epoch=absolute_epoch,
            train_loss=sums["total"],
            train_x_loss=sums["x_reco"],
            train_z_loss=sums["z_prior"] + sums["z_cycle"] + sums["transverse"],
            eval_loss=eval_sums["total"],
        )
        final_eval = {
            "loss": eval_sums["total"],
            "x_reco": eval_sums["x_reco"],
            "z_prior": eval_sums["z_prior"],
            "z_cycle": eval_sums["z_cycle"],
            "transverse": eval_sums["transverse"],
        }

        print(
            f"epoch {absolute_epoch:04d} "
            f"(local {local_epoch:04d}) | "
            f"train_total={sums['total']:.4e} | "
            f"train_x={sums['x_reco']:.4e} | "
            f"train_zprior={sums['z_prior']:.4e} | "
            f"train_zcycle={sums['z_cycle']:.4e} | "
            f"train_trans={sums['transverse']:.4e} | "
            f"eval_total={eval_sums['total']:.4e}"
        )

    return final_eval, history


def history_to_numpy(values):
    out = []
    for value in values:
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                out.append(value.detach().cpu().item())
            else:
                out.append(value.detach().cpu().numpy())
        else:
            out.append(value)
    return np.asarray(out, dtype=float)


def generate_model_arrays(model, x_train, x_val, x_test, z_train, z_val, z_test):
    torch.manual_seed(0)
    np.random.seed(0)
    model.to("cpu")
    model.encoder.output_stats.to("cpu")
    model.decoder.output_stats.to("cpu")
    model.eval()

    all_arrs = {
        "train": {"x": x_train, "z": z_train},
        "val": {"x": x_val, "z": z_val},
        "test": {"x": x_test, "z": z_test},
    }
    with torch.no_grad():
        for data_key in ("train", "val", "test"):
            arrs = all_arrs[data_key]
            z_tensor = torch.from_numpy(arrs["z"])
            x_tensor = torch.from_numpy(arrs["x"])
            arrs["z_decoded"] = first_tensor(model.decode(z_tensor))
            arrs["x_encoded"] = first_tensor(model.encode(x_tensor))
            arrs["x_reconstructed"] = first_tensor(model.decode(arrs["x_encoded"]))

            for field, arr in list(arrs.items()):
                if isinstance(arr, torch.Tensor):
                    arrs[field] = arr.detach().cpu().numpy()

    return all_arrs


def save_outputs(config: dict, config_path: Path, save_dir: Path, model, all_arrs: dict, history: dict):
    save_dir.mkdir(parents=True, exist_ok=True)

    model_save_path = save_dir / "model_state_dict.pkl"
    torch.save(model.state_dict(), model_save_path)
    print("Saved model weights to", model_save_path)

    full_model_save_path = save_dir / "model_full.pt"
    torch.save(model, full_model_save_path)
    print("Saved full model to", full_model_save_path)

    results_save_path = save_dir / "results_test.npz"
    np.savez(results_save_path, **all_arrs["test"])
    print("Model test results saved at", results_save_path)

    history_save_path = save_dir / "history.npz"
    history_np = {key: history_to_numpy(value) for key, value in history.items()}
    np.savez(history_save_path, **history_np)
    print("Training history saved at", history_save_path)

    copied_config_path = save_dir / "train_config.json"
    shutil.copy2(config_path, copied_config_path)
    print("Training config copied to", copied_config_path)

    metadata_path = save_dir / "training_metadata.json"
    metadata = {
        "finished_at_unix": time.time(),
        "config_path": str(config_path),
        "dataset_name": config["paths"]["dataset_name"],
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print("Training metadata saved at", metadata_path)


def summarize_mass_outputs(all_arrs: dict) -> None:
    arrs = all_arrs["val"]
    for name, values in {
        "CMS x": Zboson_mass(arrs["x"]),
        "x-z-x": Zboson_mass(arrs["x_reconstructed"]),
        "z-x": Zboson_mass(arrs["z_decoded"]),
    }.items():
        bins = np.linspace(70, 110, 81)
        counts, edges = np.histogram(values, bins=bins)
        peak = 0.5 * (edges[np.argmax(counts)] + edges[np.argmax(counts) + 1])
        print(
            f"{name:12s} "
            f"mean={np.mean(values):7.3f}, "
            f"std={np.std(values):7.3f}, "
            f"peak={peak:7.3f}, "
            f"frac_70_110={np.mean((values > 70) & (values < 110)):6.3f}"
        )


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    apply_environment(config)

    global ak, h5py, np, sliced_wd, torch, train_and_val, uproot, Zboson_mass
    import awkward as ak
    import h5py
    import numpy as np
    import torch
    import uproot

    repo_root = Path(__file__).resolve().parents[2]
    utility_dir = repo_root / "utilityFunctions"
    if not (utility_dir / "func_utils.py").exists():
        raise FileNotFoundError(f"Cannot find utilityFunctions at {utility_dir}")
    sys.path.insert(0, str(utility_dir))

    from func_utils import Zboson_mass, sliced_wd
    from ppzee_utils import train_and_val

    base_dir = config_path.parent
    paths = config["paths"]
    cms_root_file = resolve_path(paths["cms_root_file"], base_dir)
    theory_prior_file = resolve_path(paths["theory_prior_file"], base_dir)
    save_dir = resolve_path(paths["output_root"], base_dir) / paths["dataset_name"]

    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)

    float_type = np.dtype(config.get("float_type", "float32")).name
    device_obj = select_device(config.get("device", "auto"))
    print("Using utilityFunctions from:", utility_dir)
    print("Using device:", device_obj)
    print("Using dtype:", float_type)

    x_data = load_cms_x_data(cms_root_file, config["electron_selection"])
    z_data = load_theory_prior_z(theory_prior_file)

    assert x_data.ndim == 2 and x_data.shape[1] == 8, f"x_data should have shape (N, 8), got {x_data.shape}"
    assert z_data.ndim == 2 and z_data.shape[1] == 8, f"z_data should have shape (N, 8), got {z_data.shape}"
    assert np.all(np.isfinite(x_data)), "x_data contains NaN or inf"
    assert np.all(np.isfinite(z_data)), "z_data contains NaN or inf"

    split_config = config["data_split"]
    x_train, x_val, x_test = split_unpaired(
        x_data,
        split_config["train_ratio"],
        split_config["val_ratio"],
        seed=seed,
    )
    z_train, z_val, z_test = split_unpaired(
        z_data,
        split_config["train_ratio"],
        split_config["val_ratio"],
        seed=seed + 1,
    )

    x_train, x_val, x_test, z_train, z_val, z_test = [
        arr.astype(float_type)
        for arr in (x_train, x_val, x_test, z_train, z_val, z_test)
    ]

    x_train_mean = np.mean(x_train, axis=0)
    x_train_std = np.where(np.std(x_train, axis=0) == 0, 1.0, np.std(x_train, axis=0))
    z_train_mean = np.mean(z_train, axis=0)
    z_train_std = np.where(np.std(z_train, axis=0) == 0, 1.0, np.std(z_train, axis=0))

    print("x_train shape:", x_train.shape)
    print("z_train shape:", z_train.shape)
    print("x_val shape:", x_val.shape)
    print("z_val shape:", z_val.shape)
    print("x_test shape:", x_test.shape)
    print("z_test shape:", z_test.shape)

    model = build_model(
        config,
        x_dim=int(x_data.shape[1]),
        z_dim=int(z_data.shape[1]),
        x_train_mean=x_train_mean,
        x_train_std=x_train_std,
        z_train_mean=z_train_mean,
        z_train_std=z_train_std,
    )
    model.to(device_obj)

    loss_factory = ZLossFactory(x_train, z_train, config.get("loss", {}))
    train_loaders, eval_loaders = build_loaders(config, x_train, x_val, z_train, z_val, device_obj)

    if args.dry_run:
        print("Dry run complete. Exiting before training.")
        return

    history = None
    eval_losses = None
    enabled_stages = [stage["name"] for stage in config["stages"] if stage.get("enabled", True)]
    print("Enabled stages:", enabled_stages)

    for stage in config["stages"]:
        if not stage.get("enabled", True):
            print(f"Skipping disabled stage: {stage['name']}")
            continue

        print("\n" + "=" * 80)
        print("Starting training stage:", stage["name"])
        print("Mode:", stage.get("mode", "standard"))
        print("=" * 80)

        loss_factory.set_num_slices(stage["num_slices"])
        mode = stage.get("mode", "standard")

        if mode == "standard":
            freeze_encoder = stage.get("freeze_encoder", False)
            freeze_decoder = stage.get("freeze_decoder", False)
            set_trainable(model, train_encoder=not freeze_encoder, train_decoder=not freeze_decoder)

            print("Train encoder:", not freeze_encoder)
            print("Train decoder:", not freeze_decoder)

            stage_config = make_stage_config(config, stage)
            trainable_params = [param for param in model.parameters() if param.requires_grad]
            if not trainable_params:
                raise ValueError(f"Stage {stage['name']} has no trainable parameters.")
            optimizer = torch.optim.Adam(trainable_params, lr=stage_config["lr"])

            eval_losses, history = train_and_val(
                model,
                train_loaders,
                eval_loaders,
                stage_config,
                optimizer,
                verbose=True,
                prev_hist=history,
                log_freq=stage["log_freq"],
                lr_decay=stage["lr_decay"],
                z_loss_fun=loss_factory,
                device=device_obj,
            )
        elif mode == "z_cycle":
            eval_losses, history = run_z_cycle_stage(
                model,
                train_loaders,
                eval_loaders,
                stage,
                history,
                loss_factory,
                device_obj,
            )
        else:
            raise ValueError(f"Unknown training stage mode: {mode}")

    set_trainable(model, train_encoder=True, train_decoder=True)

    print("\nTraining complete.")
    print("Final eval losses:", eval_losses)

    all_arrs = generate_model_arrays(model, x_train, x_val, x_test, z_train, z_val, z_test)
    summarize_mass_outputs(all_arrs)
    save_outputs(config, config_path, save_dir, model, all_arrs, history)


if __name__ == "__main__":
    main()
