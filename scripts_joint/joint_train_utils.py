"""Shared scheduling/coverage utilities for the ``cms_Joint`` trainer.

Pure helpers extracted from ``joint_trainer.py`` so the training loop file can
focus on the actual training/validation/checkpoint logic.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import torch

def _first(value):
    return value[0] if isinstance(value, (tuple, list)) else value


def _as_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def _scheduled_value(spec: Any, local_epoch: int, epochs: int) -> float:
    if not isinstance(spec, dict):
        return float(spec)
    start = float(spec["start"])
    end = float(spec.get("end", start))
    t = 0.0 if epochs <= 1 else float(local_epoch - 1) / float(epochs - 1)
    schedule = str(spec.get("schedule", "linear")).lower()
    if schedule == "linear":
        return start + (end - start) * t
    if schedule == "cosine":
        return end + (start - end) * (1.0 + math.cos(math.pi * t)) / 2.0
    if schedule == "constant":
        return start
    raise ValueError(f"Unknown schedule {schedule!r}")


def _region_weights(config: dict[str, Any]) -> dict[str, float]:
    raw = config.get("region_weights", {})
    values = {
        name: float(raw.get(name, 1.0)) for name in config["region_order"]
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in values.values()):
        raise ValueError("Every joint region weight must be finite and positive")
    total = sum(values.values())
    return {name: value / total for name, value in values.items()}


def _sample_batch(
    values: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    device: torch.device,
) -> torch.Tensor:
    replace = len(values) < batch_size
    indices = rng.choice(len(values), size=batch_size, replace=replace)
    return torch.as_tensor(
        np.ascontiguousarray(values[indices]), dtype=torch.float32, device=device
    )


def _expand_indices(
    indices: np.ndarray,
    length: int,
    min_events: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pad a per-update index batch to at least ``min_events`` with replacement.

    Why this exists (flat-loss mechanism, 2026-09-10): a full-pass joint epoch
    sizes every stream's per-update batch from the largest partition. The
    J/psi prior has 71,659 rows against 2.9M CMS rows, so its per-update batch
    is ~417 events while the CMS side is ~16,858. The finite-sample floor of
    every distributional loss term evaluated against that stream is then set by
    the 417-event empirical quantile function, not by model error: a *perfect*
    J/psi model still reports L_z ~ 0.344 and L_x ~ 0.353 at those
    cardinalities, versus ~0.108 at equal 8,192-event batches. Padding the small
    side with replacement lowers the floor and the gradient variance. It does
    not add information (the same unique rows are still visited once per epoch),
    it removes estimator noise.

    ``min_events <= 0`` is the back-compatible default: the batch is returned
    unchanged, so every pre-2026-09-10 run is bit-reproducible.
    """
    min_events = int(min_events)
    if min_events <= 0 or len(indices) >= min_events:
        return indices
    extra = rng.integers(
        0,
        int(length),
        size=min_events - len(indices),
        dtype=np.int64,
    )
    return np.concatenate([indices, extra])


def _coverage_indices(
    length: int,
    start: int,
    count: int,
    *,
    seed: int,
    stream: int,
) -> np.ndarray:
    """Return a deterministic no-repeat stream over successive full cycles.

    The selected arrays are already shuffled by the locked data split. Each
    cycle applies an additional seeded affine permutation, which is bijective
    because its multiplier is chosen coprime to the array length. This avoids
    materializing multi-million-row permutations every epoch while ensuring
    every row is visited once before that stream repeats.
    """
    length = int(length)
    start = int(start)
    count = int(count)
    if length <= 0 or start < 0 or count <= 0:
        raise ValueError("coverage stream requires length/count > 0 and start >= 0")
    positions = np.arange(start, start + count, dtype=np.int64)
    cycles = positions // length
    offsets = positions % length
    indices = np.empty(count, dtype=np.int64)
    for cycle in np.unique(cycles):
        rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), int(stream), int(cycle)])
        )
        if length == 1:
            multiplier = 1
            shift = 0
        else:
            multiplier = int(rng.integers(1, length))
            while math.gcd(multiplier, length) != 1:
                multiplier = 1 if multiplier + 1 >= length else multiplier + 1
            shift = int(rng.integers(0, length))
        mask = cycles == cycle
        indices[mask] = (multiplier * offsets[mask] + shift) % length
    return indices


def _coverage_batches(
    values: np.ndarray,
    batch_size: int,
    steps: int,
    epoch_index: int,
    *,
    seed: int,
    stream: int,
) -> np.ndarray:
    draws_per_epoch = int(batch_size) * int(steps)
    start = (int(epoch_index) - 1) * draws_per_epoch
    return _coverage_indices(
        len(values), start, draws_per_epoch, seed=seed, stream=stream
    ).reshape(int(steps), int(batch_size))


def _full_pass_step_count(
    region_arrays: dict[str, dict[str, np.ndarray]],
    region_order: list[str],
    batch_size: int,
) -> int:
    """Return the updates needed to visit every training row once.

    A joint epoch uses one x batch and one z batch from every region per
    optimizer update. The largest training partition determines the number of
    updates; every other partition is divided into the same number of balanced
    (usually smaller) batches. This avoids repeating a shorter prior merely to
    keep pace with a larger CMS partition.
    """
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("loaders.train_batch_size must be positive")
    lengths = [
        len(region_arrays[name][key])
        for name in region_order
        for key in ("x_train", "z_train")
    ]
    if not lengths or min(lengths) < 1:
        raise ValueError("full-pass epochs require non-empty x/z training partitions")
    steps = max(math.ceil(length / batch_size) for length in lengths)
    if steps > min(lengths):
        raise ValueError(
            "full-pass epoch would require an empty per-update batch in a shorter "
            "training partition; increase train_batch_size or provide more events"
        )
    return int(steps)


def _full_pass_batches(
    values: np.ndarray,
    steps: int,
    epoch_index: int,
    *,
    seed: int,
    stream: int,
) -> list[np.ndarray]:
    """Partition one deterministic epoch permutation into balanced batches."""
    length = len(values)
    indices = _coverage_indices(
        length,
        (int(epoch_index) - 1) * length,
        length,
        seed=seed,
        stream=stream,
    )
    batches = [np.asarray(part, dtype=np.int64) for part in np.array_split(indices, steps)]
    if any(len(part) == 0 for part in batches):
        raise ValueError("full-pass epoch produced an empty batch")
    return batches


def _build_optimizer(parameters, stage: dict[str, Any]):
    name = str(stage.get("optimizer", "adamw")).lower()
    kwargs = {
        "lr": float(stage["lr"]),
        "betas": tuple(float(v) for v in stage.get("adam_betas", (0.9, 0.999))),
        "eps": float(stage.get("adam_eps", 1.0e-8)),
        "weight_decay": float(stage.get("weight_decay", 0.0)),
    }
    if name == "adam":
        return torch.optim.Adam(parameters, **kwargs)
    if name == "adamw":
        return torch.optim.AdamW(parameters, **kwargs)
    raise ValueError(f"Unknown optimizer {name!r}")


def _build_scheduler(optimizer, stage: dict[str, Any], start_epoch: int):
    name = str(stage.get("lr_schedule", "none")).lower()
    epochs = max(1, int(stage["epochs"]))
    warmup = min(max(0, int(stage.get("lr_warmup_epochs", 0))), max(0, epochs - 1))
    if name == "none":
        return None
    if name != "cosine":
        raise ValueError(f"Unknown lr_schedule {name!r}")
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs - warmup),
        eta_min=float(stage["lr"]) * float(stage.get("lr_min_factor", 0.05)),
    )
    if warmup:
        linear = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=float(stage.get("lr_warmup_start_factor", 0.2)),
            end_factor=1.0,
            total_iters=warmup,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [linear, cosine], milestones=[warmup]
        )
    else:
        scheduler = cosine
    if start_epoch > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(start_epoch - 1):
                scheduler.step()
    return scheduler


def _set_trainable(model, stage: dict[str, Any]) -> None:
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(not bool(stage.get("freeze_encoder", False)))
    for parameter in model.decoder.parameters():
        parameter.requires_grad_(not bool(stage.get("freeze_decoder", False)))
