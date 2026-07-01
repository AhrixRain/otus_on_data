from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cms_model import set_trainable


REPO_ROOT = Path(__file__).resolve().parents[1]
UTILITY_DIR = REPO_ROOT / "utilityFunctions"
import sys

if str(UTILITY_DIR) not in sys.path:
    sys.path.insert(0, str(UTILITY_DIR))

from func_utils import anchor_loss, data_loss, sliced_wd  # noqa: E402
from feature_ot_loss import OriginalOtusFeatureLossFactory  # noqa: E402


P = 2


def first_tensor(value):
    if isinstance(value, (tuple, list)):
        return value[0]
    return value


def as_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


class ResampleTensorLoader:
    def __init__(self, tensor: torch.Tensor, batch_size: int, steps_per_epoch: int):
        self.tensor = tensor
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.n = int(tensor.shape[0])

    def __len__(self) -> int:
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


def build_loaders(
    config: dict[str, Any],
    arrays: dict[str, np.ndarray],
    batch_size_override: int | None,
    device: torch.device,
) -> tuple[tuple[ResampleTensorLoader, ResampleTensorLoader], tuple[ResampleTensorLoader, ResampleTensorLoader], dict[str, Any]]:
    loader_config = config["loaders"]
    train_batch_size = int(batch_size_override or loader_config["train_batch_size"])
    eval_batch_size = int(loader_config["eval_batch_size"])

    train_batch_size = min(
        train_batch_size,
        max(1, min(len(arrays["x_train"]), len(arrays["z_train"]))),
    )
    eval_batch_size = min(
        eval_batch_size,
        max(1, min(len(arrays["x_val"]), len(arrays["z_val"]))),
    )
    steps_per_epoch = max(
        1,
        math.ceil(max(len(arrays["x_train"]), len(arrays["z_train"])) / train_batch_size),
    )
    eval_steps_per_epoch = max(
        1,
        math.ceil(max(len(arrays["x_val"]), len(arrays["z_val"])) / eval_batch_size),
    )

    preload = bool(loader_config.get("preload_data_to_accelerator", False))
    loader_device = device if preload and device.type in {"cuda"} else torch.device("cpu")
    tensor_kwargs = {"dtype": torch.float32}
    tensors = {
        key: torch.as_tensor(value, **tensor_kwargs).to(loader_device)
        for key, value in arrays.items()
        if key in {"x_train", "x_val", "z_train", "z_val"}
    }
    train_loaders = (
        ResampleTensorLoader(tensors["x_train"], train_batch_size, steps_per_epoch),
        ResampleTensorLoader(tensors["z_train"], train_batch_size, steps_per_epoch),
    )
    eval_loaders = (
        ResampleTensorLoader(tensors["x_val"], eval_batch_size, eval_steps_per_epoch),
        ResampleTensorLoader(tensors["z_val"], eval_batch_size, eval_steps_per_epoch),
    )
    info = {
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "steps_per_epoch": steps_per_epoch,
        "eval_steps_per_epoch": eval_steps_per_epoch,
        "loader_device": str(loader_device),
    }
    return train_loaders, eval_loaders, info


class ZLossFactory:
    def __init__(self, x_train: np.ndarray, z_train: np.ndarray, loss_config: dict[str, float]):
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
        start = max(0, min(int((1.0 - tail_frac) * n), n - 1))
        return torch.mean(torch.abs(a[start:n] - b[start:n]) ** self.p)

    def standardize_x_raw(self, x):
        return (x - self.to_like(self.x_train_mean, x)) / (self.to_like(self.x_train_std, x) + self.eps)

    def standardize_z_raw(self, z):
        return (z - self.to_like(self.z_train_mean, z)) / (self.to_like(self.z_train_std, z) + self.eps)

    def paired_mse_standardized(self, a, b, standardize_fun):
        return torch.mean((standardize_fun(a) - standardize_fun(b)) ** 2)

    def dilepton_mass_torch(self, z):
        px = z[:, 0] + z[:, 4]
        py = z[:, 1] + z[:, 5]
        pz = z[:, 2] + z[:, 6]
        energy = z[:, 3] + z[:, 7]
        return torch.sqrt(torch.clamp(energy**2 - px**2 - py**2 - pz**2, min=self.eps))

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
            [px1, py1, px2, py2, pt1, pt2, px_sum, py_sum, ptll, px_diff, py_diff, cos_dphi, sin_dphi],
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
            torch.clamp(energy + pz_sum, min=self.eps)
            / torch.clamp(energy - pz_sum, min=self.eps)
        )
        return torch.stack([pz1, pz2, pz_sum, pz_diff, eta1, eta2, yll], dim=1)

    def standardize_mass(self, mass):
        return (mass - self.to_like(self.m_z_mean, mass)) / (self.to_like(self.m_z_std, mass) + self.eps)

    def standardize_features(self, features, mean, std):
        return (features - self.to_like(mean, features)) / (self.to_like(std, features) + self.eps)

    def z_channel_marginal_loss(self, z, z_tilde):
        channel_weights = [6.0, 6.0, 3.0, 0.5, 6.0, 6.0, 3.0, 0.5]
        return self.weighted_marginal_wd(
            self.standardize_z_raw(z),
            self.standardize_z_raw(z_tilde),
            channel_weights,
        )

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
        loss_mass = self.wasserstein_1d_sorted(
            self.standardize_mass(self.dilepton_mass_torch(z)),
            self.standardize_mass(self.dilepton_mass_torch(z_tilde)),
        )
        loss_phys = sliced_wd(
            self.standardize_features(self.z_physics_features(z), self.fz_mean, self.fz_std),
            self.standardize_features(self.z_physics_features(z_tilde), self.fz_mean, self.fz_std),
            self.num_slices,
            self.p,
        )
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

    def z_prior_loss(self, z_true, z_encoded):
        return self(z_true, z_encoded)

    def x_sim_loss(self, x_true, x_from_z):
        return sliced_wd(
            self.standardize_x_raw(x_true),
            self.standardize_x_raw(x_from_z),
            self.num_slices,
            P,
        )

    def x_reco_loss(self, x_true, x_reco):
        return data_loss(x_true, x_reco, P)

    def encoder_anchor_loss(self, z_encoded, x_true):
        return anchor_loss(z_encoded, x_true)

    def decoder_anchor_loss(self, z_true, x_from_z):
        return anchor_loss(z_true, x_from_z)

    def validation_score(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        return losses["z_loss"] + losses["alt_x_loss"]


def build_loss_factory(x_train: np.ndarray, z_train: np.ndarray, loss_config: dict[str, Any]):
    if loss_config.get("kind") == "original_feature_ot_v1":
        return OriginalOtusFeatureLossFactory(x_train, z_train, loss_config)
    return ZLossFactory(x_train, z_train, loss_config)


def make_stage_config(config: dict[str, Any], stage: dict[str, Any]) -> dict[str, Any]:
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


class HistoryLogger:
    fieldnames = [
        "epoch",
        "stage",
        "train_loss",
        "train_x_loss",
        "train_z_loss",
        "train_alt_x_loss",
        "train_x_constraint_loss",
        "eval_loss",
        "eval_x_loss",
        "eval_z_loss",
        "eval_alt_x_loss",
        "eval_selection_score",
    ]

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.rows: list[dict[str, Any]] = []
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def append(self, row: dict[str, Any]) -> None:
        clean = {key: row.get(key, "") for key in self.fieldnames}
        self.rows.append(clean)
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(clean)

    def history(self) -> dict[str, list[Any]]:
        return {key: [row[key] for row in self.rows] for key in self.fieldnames}


def train_standard_epoch(model, optimizer, x_loader, z_loader, stage, loss_factory, device):
    model.train()
    sums = {
        "loss": 0.0,
        "x_loss": 0.0,
        "z_loss": 0.0,
        "alt_x_loss": 0.0,
        "x_constraint_loss": 0.0,
    }
    nbatches = 0
    for x, z in zip(x_loader, z_loader):
        x = x.to(device)
        z = z.to(device)
        optimizer.zero_grad()
        z_tilde = first_tensor(model.encode(x))
        z_loss = loss_factory.z_prior_loss(z, z_tilde) if stage["lamb"] > 0 else x.new_tensor(0.0)
        if stage["beta"] > 0:
            x_tilde = first_tensor(model.decode(z_tilde))
            x_loss = loss_factory.x_reco_loss(x, x_tilde)
        else:
            x_loss = x.new_tensor(0.0)
        encoder_anchor = (
            loss_factory.encoder_anchor_loss(z_tilde, x)
            if stage["nu_e"] > 0
            else x.new_tensor(0.0)
        )
        decoder_anchor = x.new_tensor(0.0)
        alt_x_loss = x.new_tensor(0.0)
        x_constraint_loss = x.new_tensor(0.0)
        if stage["tau"] > 0 or stage["rho"] > 0 or stage["nu_d"] > 0:
            model_x = first_tensor(model.decode(z))
            alt_x_loss = (
                loss_factory.x_sim_loss(x, model_x)
                if stage["tau"] > 0
                else x.new_tensor(0.0)
            )
            decoder_anchor = (
                loss_factory.decoder_anchor_loss(z, model_x)
                if stage["nu_d"] > 0
                else x.new_tensor(0.0)
            )
        loss = (
            stage["beta"] * x_loss
            + stage["lamb"] * z_loss
            + stage["tau"] * alt_x_loss
            + stage["rho"] * x_constraint_loss
            + stage["nu_e"] * encoder_anchor
            + stage["nu_d"] * decoder_anchor
        )
        loss.backward()
        optimizer.step()
        sums["loss"] += as_float(loss)
        sums["x_loss"] += as_float(x_loss)
        sums["z_loss"] += as_float(z_loss)
        sums["alt_x_loss"] += as_float(alt_x_loss)
        sums["x_constraint_loss"] += as_float(x_constraint_loss)
        nbatches += 1
    return {key: value / max(1, nbatches) for key, value in sums.items()}


def eval_standard_epoch(model, x_loader, z_loader, loss_factory, device):
    model.eval()
    sums = {
        "loss": 0.0,
        "x_loss": 0.0,
        "z_loss": 0.0,
        "alt_x_loss": 0.0,
        "selection_score": 0.0,
    }
    nbatches = 0
    with torch.no_grad():
        for x, z in zip(x_loader, z_loader):
            x = x.to(device)
            z = z.to(device)
            z_tilde = first_tensor(model.encode(x))
            x_tilde = first_tensor(model.decode(z_tilde))
            x_loss = loss_factory.x_reco_loss(x, x_tilde)
            z_loss = loss_factory.z_prior_loss(z, z_tilde)
            alt_x_loss = loss_factory.x_sim_loss(x, first_tensor(model.decode(z)))
            loss = loss_factory.validation_score(
                {
                    "x_loss": x_loss,
                    "z_loss": z_loss,
                    "alt_x_loss": alt_x_loss,
                }
            )
            sums["loss"] += as_float(loss)
            sums["x_loss"] += as_float(x_loss)
            sums["z_loss"] += as_float(z_loss)
            sums["alt_x_loss"] += as_float(alt_x_loss)
            sums["selection_score"] += as_float(loss)
            nbatches += 1
    return {key: value / max(1, nbatches) for key, value in sums.items()}


def z_cycle_losses_for_batch(model, x_batch, z_batch, stage, loss_factory):
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
        "loss": loss_total,
        "x_loss": loss_x_reco,
        "z_loss": loss_z_prior + loss_z_cycle + loss_transverse,
        "z_prior": loss_z_prior,
        "z_cycle": loss_z_cycle,
        "transverse": loss_transverse,
    }


def train_z_cycle_epoch(model, optimizer, trainable_params, x_loader, z_loader, stage, loss_factory, device):
    model.train()
    sums = {"loss": 0.0, "x_loss": 0.0, "z_loss": 0.0}
    nbatches = 0
    for x, z in zip(x_loader, z_loader):
        x = x.to(device)
        z = z.to(device)
        optimizer.zero_grad()
        losses = z_cycle_losses_for_batch(model, x, z, stage, loss_factory)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=10.0)
        optimizer.step()
        for key in sums:
            sums[key] += as_float(losses[key])
        nbatches += 1
    return {key: value / max(1, nbatches) for key, value in sums.items()}


def eval_z_cycle_epoch(model, x_loader, z_loader, stage, loss_factory, device):
    model.eval()
    sums = {"loss": 0.0, "x_loss": 0.0, "z_loss": 0.0}
    nbatches = 0
    with torch.no_grad():
        for x, z in zip(x_loader, z_loader):
            x = x.to(device)
            z = z.to(device)
            losses = z_cycle_losses_for_batch(model, x, z, stage, loss_factory)
            for key in sums:
                sums[key] += as_float(losses[key])
            nbatches += 1
    return {key: value / max(1, nbatches) for key, value in sums.items()}


def train_all_stages(
    model,
    config,
    train_loaders,
    eval_loaders,
    loss_factory,
    device,
    logger: HistoryLogger,
    save_callback,
    progress_callback=None,
) -> tuple[dict[str, list[Any]], float | None, int]:
    history_epoch = 0
    best_eval_loss: float | None = None
    total_epochs = sum(
        int(stage["epochs"]) for stage in config["stages"] if stage.get("enabled", True)
    )
    for stage in config["stages"]:
        if not stage.get("enabled", True):
            continue
        stage_epochs = int(stage["epochs"])
        early_config = stage.get("early_stopping", {})
        early_enabled = bool(early_config.get("enabled", False))
        early_patience = max(1, int(early_config.get("patience", 10)))
        early_min_delta = float(early_config.get("min_delta", 0.0))
        early_bad_checks = 0
        stage_best_eval_loss: float | None = None
        loss_factory.set_num_slices(stage["num_slices"])
        set_trainable(
            model,
            train_encoder=not bool(stage.get("freeze_encoder", False)),
            train_decoder=not bool(stage.get("freeze_decoder", False)),
        )
        trainable_params = [param for param in model.parameters() if param.requires_grad]
        if not trainable_params:
            raise ValueError(f"Stage {stage['name']} has no trainable parameters.")
        optimizer = torch.optim.Adam(trainable_params, lr=float(stage["lr"]))
        scheduler = None
        if stage.get("lr_decay", False):
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda epoch: 1 / (1 + 0.1 * epoch),
            )

        for local_epoch in range(1, stage_epochs + 1):
            if stage.get("mode", "standard") == "z_cycle":
                train_losses = train_z_cycle_epoch(
                    model,
                    optimizer,
                    trainable_params,
                    train_loaders[0],
                    train_loaders[1],
                    stage,
                    loss_factory,
                    device,
                )
                eval_fn = eval_z_cycle_epoch
            else:
                train_losses = train_standard_epoch(
                    model,
                    optimizer,
                    train_loaders[0],
                    train_loaders[1],
                    make_stage_config(config, stage),
                    loss_factory,
                    device,
                )
                eval_fn = eval_standard_epoch
            if scheduler is not None:
                scheduler.step()

            history_epoch += 1
            should_log = (
                local_epoch == 1
                or local_epoch == stage_epochs
                or local_epoch % int(stage.get("log_freq", 10)) == 0
            )
            eval_losses = None
            eval_loss = None
            selection_score = None
            if should_log:
                if stage.get("mode", "standard") == "z_cycle":
                    eval_losses = eval_fn(
                        model,
                        eval_loaders[0],
                        eval_loaders[1],
                        stage,
                        loss_factory,
                        device,
                    )
                    eval_alt_x_loss = ""
                else:
                    eval_losses = eval_fn(
                        model,
                        eval_loaders[0],
                        eval_loaders[1],
                        loss_factory,
                        device,
                    )
                    eval_alt_x_loss = eval_losses["alt_x_loss"]

                selection_score = float(eval_losses.get("selection_score", eval_losses["loss"]))
                eval_loss = selection_score
                row = {
                    "epoch": history_epoch,
                    "stage": stage["name"],
                    "train_loss": train_losses["loss"],
                    "train_x_loss": train_losses["x_loss"],
                    "train_z_loss": train_losses["z_loss"],
                    "train_alt_x_loss": train_losses.get("alt_x_loss", ""),
                    "train_x_constraint_loss": train_losses.get("x_constraint_loss", ""),
                    "eval_loss": eval_loss,
                    "eval_x_loss": eval_losses["x_loss"],
                    "eval_z_loss": eval_losses["z_loss"],
                    "eval_alt_x_loss": eval_alt_x_loss,
                    "eval_selection_score": selection_score,
                }
                logger.append(row)
                is_best = best_eval_loss is None or selection_score < best_eval_loss
                if is_best:
                    best_eval_loss = selection_score
                save_callback(history_epoch, selection_score, is_best)

                if early_enabled:
                    stage_improved = (
                        stage_best_eval_loss is None
                        or selection_score < stage_best_eval_loss - early_min_delta
                    )
                    if stage_improved:
                        stage_best_eval_loss = selection_score
                        early_bad_checks = 0
                    else:
                        early_bad_checks += 1

            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": stage["name"],
                        "epoch": local_epoch,
                        "epochs_in_stage": stage_epochs,
                        "global_epoch": history_epoch,
                        "total_epochs": total_epochs,
                        "percent": (100.0 * history_epoch / total_epochs) if total_epochs else 100.0,
                        "train_loss": float(train_losses["loss"]),
                        "eval_loss": eval_loss,
                        "best_eval_loss": best_eval_loss,
                        "evaluated": should_log,
                    }
                )
            elif should_log:
                print(
                    f"epoch {history_epoch:04d} | {stage['name']} | "
                    f"train_loss={train_losses['loss']:.4e} | eval_loss={eval_loss:.4e}"
                )
            if early_enabled and eval_loss is not None and early_bad_checks >= early_patience:
                print(
                    f"Early stopping {stage['name']} at epoch {local_epoch}/{stage_epochs} "
                    f"after {early_bad_checks} validation checks without improvement.",
                    flush=True,
                )
                break
    return logger.history(), best_eval_loss, history_epoch
