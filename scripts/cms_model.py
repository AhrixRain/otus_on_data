from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
UTILITY_DIR = REPO_ROOT / "utilityFunctions"
if str(UTILITY_DIR) not in sys.path:
    sys.path.insert(0, str(UTILITY_DIR))

from models import Autoencoder, CondNoiseAutoencoder, CondNoiseMLP, StochasticResNet  # noqa: E402


def build_model(
    config: dict[str, Any],
    x_train_mean: np.ndarray,
    x_train_std: np.ndarray,
    z_train_mean: np.ndarray,
    z_train_std: np.ndarray,
    x_dim: int = 8,
    z_dim: int = 8,
) -> torch.nn.Module:
    model_config = config["model"]
    hidden_layer_dims = int(model_config["num_hidden_layers"]) * [
        int(model_config["dim_per_hidden_layer"])
    ]
    activation = getattr(torch.nn, model_config.get("activation", "ReLU"))

    common_kwargs: dict[str, Any] = {
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
    if "sigma_floor" in model_config:
        common_kwargs["sigma_floor"] = float(model_config["sigma_floor"])

    class_name = model_config.get("class", "CondNoiseAutoencoder")
    if class_name == "CondNoiseAutoencoder":
        return CondNoiseAutoencoder(hidden_layer_dims=hidden_layer_dims, **common_kwargs)
    if class_name == "Autoencoder":
        conditional_models = {
            "CondNoiseMLP": CondNoiseMLP,
            "StochasticResNet": StochasticResNet,
        }
        conditional_name = model_config.get("conditional_model", "CondNoiseMLP")
        if conditional_name not in conditional_models:
            raise ValueError(f"Unknown conditional_model: {conditional_name}")
        return Autoencoder(
            ConditionalModel=conditional_models[conditional_name],
            encoder_hidden_layer_dims=hidden_layer_dims,
            **common_kwargs,
        )
    raise ValueError(f"Unknown model class: {class_name}")


def set_trainable(model: torch.nn.Module, train_encoder: bool = True, train_decoder: bool = True) -> None:
    for param in model.encoder.parameters():
        param.requires_grad_(train_encoder)
    for param in model.decoder.parameters():
        param.requires_grad_(train_decoder)


def checkpoint_payload(
    model: torch.nn.Module,
    config: dict[str, Any],
    stats: dict[str, np.ndarray],
    epoch: int,
    eval_loss: float | None,
    device_report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "config": config,
        "stats": {key: value.tolist() for key, value in stats.items()},
        "epoch": int(epoch),
        "eval_loss": None if eval_loss is None else float(eval_loss),
        "device_report": device_report,
    }


def load_model_from_checkpoint(
    checkpoint_path: Path,
    config: dict[str, Any] | None,
    map_location: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model_config = checkpoint.get("config") if config is None else config
    if model_config is None:
        raise ValueError("Checkpoint does not contain config; pass --config.")

    stats = {
        key: np.asarray(value, dtype=np.float32)
        for key, value in checkpoint["stats"].items()
    }
    model = build_model(
        model_config,
        x_train_mean=stats["x_train_mean"],
        x_train_std=stats["x_train_std"],
        z_train_mean=stats["z_train_mean"],
        z_train_std=stats["z_train_std"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, model_config, stats, checkpoint
