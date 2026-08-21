from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cms_model import set_trainable
from loss import (  # noqa: E402
    CANONICAL_LOSS_KIND,
    JPSI_DIMUON_LOSS_KIND,
    CmsDoubleElectronLossFactory,
    CmsJpsiDoubleMuonLossFactory,
)



def first_tensor(value):
    if isinstance(value, (tuple, list)):
        return value[0]
    return value


def as_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def flat_grad_vector(grads) -> torch.Tensor | None:
    if grads is None:
        return None
    parts = [
        grad.detach().reshape(-1)
        for grad in grads
        if grad is not None and grad.numel() > 0
    ]
    if not parts:
        return None
    return torch.cat(parts)


def grad_norm(grads) -> float:
    if grads is None:
        return 0.0
    vector = flat_grad_vector(grads)
    return 0.0 if vector is None else float(vector.norm())


def grad_cosine(grads_a, grads_b) -> float | None:
    a = flat_grad_vector(grads_a)
    b = flat_grad_vector(grads_b)
    if a is None or b is None:
        return None
    denom = a.norm() * b.norm()
    if not bool(denom > 0.0):
        return None
    return float((a @ b / denom).item())


def encoder_grads_for(loss, model) -> list[torch.Tensor | None]:
    params = [param for param in model.encoder.parameters() if param.requires_grad]
    if not params or loss is None:
        return []
    return list(
        torch.autograd.grad(
            loss,
            params,
            retain_graph=True,
            allow_unused=True,
        )
    )


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


_EPOCH_SEED_STRIDE = 1_000_003


def _aligned_batch_sizes(n_ref: int, batch_size: int) -> list[int]:
    """Batch-size schedule aligned to a reference domain of ``n_ref`` rows.

    Returns one entry per training/evaluation step. Every entry except the
    last is ``batch_size``; the last entry is the incomplete remainder
    (or ``batch_size`` when ``n_ref`` is divisible). The paired loader in the
    other domain uses the same schedule so every comparison, including the
    final one, receives equal batch cardinalities.
    """
    if n_ref <= 0:
        raise ValueError(f"Reference domain size must be positive, got {n_ref}.")
    steps = max(1, math.ceil(n_ref / batch_size))
    sizes = [batch_size] * (steps - 1)
    sizes.append(n_ref - (steps - 1) * batch_size)
    return sizes


class _CycleSafeTensorLoader:
    """Shared cursor logic for loaders that traverse a permutation cyclically.

    A batch of size ``s`` is filled by advancing a cursor through the current
    permutation; if the cursor reaches the end, the loader continues into the
    next permutation. Because every batch size is at most ``self.n`` (the
    batch size is clamped to the smaller domain in ``build_loaders``), a batch
    never duplicates a row within itself.
    """

    def __init__(self, tensor: torch.Tensor, batch_sizes: list[int]):
        self.tensor = tensor
        self.batch_sizes = [int(size) for size in batch_sizes]
        self.n = int(tensor.shape[0])
        if self.n <= 0:
            raise ValueError("Loader tensor must contain at least one row.")
        for size in self.batch_sizes:
            if size <= 0:
                raise ValueError(f"Batch sizes must be positive, got {self.batch_sizes}.")
            if size > self.n:
                raise ValueError(
                    f"Batch size {size} exceeds domain size {self.n}; "
                    "clamp the batch size to the smaller domain first."
                )

    def __len__(self) -> int:
        return len(self.batch_sizes)


class ShuffledNoReplacementLoader(_CycleSafeTensorLoader):
    """Deterministic, epoch-wise shuffled traversal without replacement.

    Each epoch derives a fresh seeded ``torch.Generator`` from
    ``seed + epoch * _EPOCH_SEED_STRIDE`` and draws a random permutation of
    every event. Batches are yielded in order against the shared schedule, so
    the larger (reference) domain is visited exactly once before it would
    reshuffle. If this domain is smaller, the cursor cycles into a freshly
    reshuffled permutation whenever it exhausts one, so every event is still
    visited at least once per epoch. The number of repeated events for a
    smaller domain is ``len(reference schedule total) - n``.

    Validation and test loaders must instead use
    ``DeterministicSequentialLoader``.
    """

    def __init__(self, tensor: torch.Tensor, batch_sizes: list[int], seed: int):
        super().__init__(tensor, batch_sizes)
        self.seed = int(seed)
        self.epoch = 0
        # Total rows this loader yields in one epoch against the shared
        # schedule; events beyond its own n are repeats caused by the unequal
        # domain policy.
        self.repeated_events = max(0, sum(self.batch_sizes) - self.n)

    def __iter__(self):
        epoch_seed = self.seed + self.epoch * _EPOCH_SEED_STRIDE
        self.epoch += 1
        generator = torch.Generator().manual_seed(epoch_seed)
        permutation = torch.randperm(self.n, generator=generator)
        position = 0
        for size in self.batch_sizes:
            chunks = []
            remaining = size
            while remaining > 0:
                take = min(remaining, self.n - position)
                chunks.append(permutation[position : position + take])
                position += take
                remaining -= take
                if position >= self.n:
                    permutation = torch.randperm(self.n, generator=generator)
                    position = 0
            yield self.tensor.index_select(0, torch.cat(chunks).to(self.tensor.device))


class DeterministicSequentialLoader(_CycleSafeTensorLoader):
    """Deterministic validation/test traversal without random resampling.

    Yields the domain in stored order against the shared schedule. A domain
    smaller than the reference repeats from the beginning (with no shuffling)
    so that both complete splits are covered at least once and every batch
    keeps equal cardinality. Two iterations over the same loader produce
    byte-identical batches.
    """

    def __init__(self, tensor: torch.Tensor, batch_sizes: list[int]):
        super().__init__(tensor, batch_sizes)
        self.repeated_events = max(0, sum(self.batch_sizes) - self.n)

    def __iter__(self):
        position = 0
        for size in self.batch_sizes:
            chunks = []
            remaining = size
            while remaining > 0:
                take = min(remaining, self.n - position)
                chunks.append(
                    torch.arange(
                        position,
                        position + take,
                        dtype=torch.long,
                        device=self.tensor.device,
                    )
                )
                position += take
                remaining -= take
                if position >= self.n:
                    position = 0
            yield self.tensor.index_select(0, torch.cat(chunks))


TRAIN_SAMPLER_POLICIES = {
    "random_with_replacement",
    "shuffled_without_replacement",
}
EVAL_SAMPLER_POLICIES = {
    "random_with_replacement",
    "deterministic_sequential",
}


def build_loaders(
    config: dict[str, Any],
    arrays: dict[str, np.ndarray],
    batch_size_override: int | None,
    device: torch.device,
) -> tuple[tuple[Any, Any], tuple[Any, Any], dict[str, Any]]:
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

    train_sampler = str(
        loader_config.get("train_sampler", "random_with_replacement")
    )
    eval_sampler = str(
        loader_config.get("eval_sampler", "random_with_replacement")
    )
    if train_sampler not in TRAIN_SAMPLER_POLICIES:
        raise ValueError(
            f"Unknown loaders.train_sampler {train_sampler!r}; expected one of "
            f"{sorted(TRAIN_SAMPLER_POLICIES)}."
        )
    if eval_sampler not in EVAL_SAMPLER_POLICIES:
        raise ValueError(
            f"Unknown loaders.eval_sampler {eval_sampler!r}; expected one of "
            f"{sorted(EVAL_SAMPLER_POLICIES)}."
        )

    preload = bool(loader_config.get("preload_data_to_accelerator", False))
    loader_device = device if preload and device.type in {"cuda"} else torch.device("cpu")
    tensor_kwargs = {"dtype": torch.float32}
    tensors = {
        key: torch.as_tensor(value, **tensor_kwargs).to(loader_device)
        for key, value in arrays.items()
        if key in {"x_train", "x_val", "z_train", "z_val"}
    }

    seed = int(config.get("seed", 0))
    train_repeated_x = None
    train_repeated_z = None
    if train_sampler == "random_with_replacement":
        train_loaders = (
            ResampleTensorLoader(tensors["x_train"], train_batch_size, steps_per_epoch),
            ResampleTensorLoader(tensors["z_train"], train_batch_size, steps_per_epoch),
        )
    else:
        train_batch_sizes = _aligned_batch_sizes(
            max(len(arrays["x_train"]), len(arrays["z_train"])),
            train_batch_size,
        )
        train_loaders = (
            ShuffledNoReplacementLoader(tensors["x_train"], train_batch_sizes, seed),
            ShuffledNoReplacementLoader(tensors["z_train"], train_batch_sizes, seed + 1),
        )
        train_repeated_x = train_loaders[0].repeated_events
        train_repeated_z = train_loaders[1].repeated_events

    eval_repeated_x = None
    eval_repeated_z = None
    if eval_sampler == "random_with_replacement":
        eval_loaders = (
            ResampleTensorLoader(tensors["x_val"], eval_batch_size, eval_steps_per_epoch),
            ResampleTensorLoader(tensors["z_val"], eval_batch_size, eval_steps_per_epoch),
        )
    else:
        eval_batch_sizes = _aligned_batch_sizes(
            max(len(arrays["x_val"]), len(arrays["z_val"])),
            eval_batch_size,
        )
        eval_loaders = (
            DeterministicSequentialLoader(tensors["x_val"], eval_batch_sizes),
            DeterministicSequentialLoader(tensors["z_val"], eval_batch_sizes),
        )
        eval_repeated_x = eval_loaders[0].repeated_events
        eval_repeated_z = eval_loaders[1].repeated_events

    info = {
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "steps_per_epoch": steps_per_epoch,
        "eval_steps_per_epoch": eval_steps_per_epoch,
        "loader_device": str(loader_device),
        "train_sampler": train_sampler,
        "eval_sampler": eval_sampler,
        "train_repeated_x": train_repeated_x,
        "train_repeated_z": train_repeated_z,
        "eval_repeated_x": eval_repeated_x,
        "eval_repeated_z": eval_repeated_z,
    }
    return train_loaders, eval_loaders, info


def build_loss_factory(
    x_train: np.ndarray,
    z_train: np.ndarray,
    loss_config: dict[str, Any],
    daughter_masses=None,
):
    kind = loss_config.get("kind", CANONICAL_LOSS_KIND)
    if kind == JPSI_DIMUON_LOSS_KIND:
        return CmsJpsiDoubleMuonLossFactory(x_train, z_train, loss_config, daughter_masses)
    if kind in {None, CANONICAL_LOSS_KIND, "original_feature_ot_v1"}:
        return CmsDoubleElectronLossFactory(x_train, z_train, loss_config, daughter_masses)
    raise ValueError(f"Unknown loss kind {kind!r}; expected {CANONICAL_LOSS_KIND!r} or {JPSI_DIMUON_LOSS_KIND!r}.")


def _schedule_fraction(local_epoch: int, stage_epochs: int) -> float:
    """Map epoch 1..N to schedule coordinate [0, 1]."""
    if stage_epochs <= 1:
        return 0.0
    return float(local_epoch - 1) / float(stage_epochs - 1)


def _resolve_scheduled_value(spec: Any, t: float, name: str) -> float:
    """Evaluate a scalar coefficient or a {start, end, schedule} spec.

    Supported schedules:
      - linear:    start + (end - start) * t
      - cosine:    end + (start - end) * (1 + cos(pi * t)) / 2
      - constant:  start
    The coordinate ``t`` is 0.0 at the first epoch and 1.0 at the last epoch
    of the stage. Scalar coefficients are kept for full backward compatibility.
    """
    if isinstance(spec, dict):
        try:
            start = float(spec["start"])
            end = float(spec.get("end", start))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Stage coefficient {name} schedule needs numeric start/end: {spec}") from exc
        kind = str(spec.get("schedule", "linear"))
        if not all(math.isfinite(value) and value >= 0.0 for value in (start, end)):
            raise ValueError(f"Stage coefficient {name} schedule values must be finite and non-negative.")
        if kind == "linear":
            value = start + (end - start) * t
        elif kind == "cosine":
            value = end + (start - end) * (1.0 + math.cos(math.pi * t)) / 2.0
        elif kind == "constant":
            value = start
        else:
            raise ValueError(
                f"Unknown schedule {kind} for stage coefficient {name}; "
                "expected linear, cosine, or constant."
            )
        return float(value)
    return float(spec)


def make_stage_loss_config(
    stage: dict[str, Any],
    local_epoch: int | None = None,
    stage_epochs: int | None = None,
) -> dict[str, Any]:
    """Resolve the per-stage loss coefficients for one training epoch.

    The next-round simplified objective is configured with explicit keys::

        alpha        -> coefficient shared by L(z, E(x)) and L(x, D(z))
        lambda_cycle -> coefficient for L(x, D(E(x)))

    Legacy configs keep using ``lamb`` / ``tau`` / ``beta``; the two naming
    schemes are mutually exclusive inside one stage.
    """
    if local_epoch is None:
        t = 0.0
    else:
        epochs = int(stage.get("epochs", stage_epochs or 1))
        t = _schedule_fraction(int(local_epoch), max(1, epochs))

    if "alpha" in stage:
        if "lamb" in stage or "tau" in stage:
            raise ValueError(
                f"Stage {stage.get('name', '?')}: use either 'alpha' or legacy "
                "'lamb'/'tau' coefficients, not both."
            )
        alpha = _resolve_scheduled_value(stage["alpha"], t, "alpha")
        lamb = tau = alpha
    else:
        alpha = None
        lamb = _resolve_scheduled_value(stage.get("lamb", 0.0), t, "lamb")
        tau = _resolve_scheduled_value(stage.get("tau", 0.0), t, "tau")

    if "lambda_cycle" in stage and "beta" in stage:
        raise ValueError(
            f"Stage {stage.get('name', '?')}: use either 'lambda_cycle' or legacy "
            "'beta', not both."
        )
    lambda_cycle_spec = stage.get("lambda_cycle", stage.get("beta", 0.0))
    lambda_cycle = _resolve_scheduled_value(lambda_cycle_spec, t, "lambda_cycle")

    return {
        "alpha": alpha,
        "lambda_cycle": lambda_cycle,
        # Effective legacy field names consumed by train_standard_epoch.
        "beta": lambda_cycle,
        "lamb": lamb,
        "tau": tau,
        "rho": _resolve_scheduled_value(stage.get("rho", 0.0), t, "rho"),
        "nu_e": _resolve_scheduled_value(stage.get("nu_e", 0.0), t, "nu_e"),
        "nu_d": _resolve_scheduled_value(stage.get("nu_d", 0.0), t, "nu_d"),
        "num_slices": int(stage.get("num_slices", 1000)),
    }


class HistoryLogger:
    component_fields = [
        f"train_{space}_{component}"
        for space in ("x", "z")
        for component in (
            "raw_swd",
            "pair_swd",
            "marginal_w1",
            "mass_w1",
            "resonance_mass_w1",
            "physics_swd",
            "mass_kin_swd",
            "transverse_w1",
            "longitudinal_w1",
            "tail_w1",
            "pair_mass_w1",
            "pair_pt_w1",
            "lepton_pt_w1",
            "delta_phi_w1",
            "delta_eta_w1",
            "pair_rapidity_w1",
            "physics_coord_swd",
        )
    ]
    weighted_component_fields = [
        f"{field}_weighted"
        for field in component_fields
    ]
    latent_component_fields = [
        f"z_component_w1_{j:02d}"
        for j in range(8)
    ]
    latent_component_weighted_fields = [
        f"z_component_w1_{j:02d}_weighted"
        for j in range(8)
    ]
    fieldnames = [
        "epoch",
        "stage",
        "train_loss",
        "train_reference_loss",
        "train_x_loss",
        "train_z_loss",
        "train_z_loss_weighted",
        "train_alt_x_loss",
        "train_alt_x_loss_weighted",
        "train_x_constraint_loss",
        "train_x_reco_mass_huber_raw",
        "train_x_reco_mass_huber_weighted",
        "eval_loss",
        "eval_reference_loss",
        "eval_x_loss",
        "eval_z_loss",
        "eval_alt_x_loss",
        "eval_selection_score",
        "grad_norm_encoder_latent",
        "grad_norm_encoder_reco",
        "grad_norm_encoder_anchor",
        "grad_norm_encoder_latent_weighted",
        "grad_norm_encoder_reco_weighted",
        "grad_norm_encoder_anchor_weighted",
        "grad_norm_encoder_total",
        "grad_norm_decoder_total",
        "grad_cosine_latent_reco",
        "v37_lambda",
        "v38_lambda",
        "effective_alpha",
        "effective_lambda_cycle",
        "learning_rate",
        "batch_size",
        "num_slices",
        "n_train_x",
        "n_train_z",
        "n_val_x",
        "n_val_z",
        "global_step",
    ] + component_fields + weighted_component_fields + latent_component_fields + latent_component_weighted_fields

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


def train_standard_epoch(
    model,
    optimizer,
    x_loader,
    z_loader,
    stage,
    loss_factory,
    device,
    step_callback=None,
):
    model.train()
    sums = {
        "loss": 0.0,
        "reference_loss": 0.0,
        "x_loss": 0.0,
        "z_loss": 0.0,
        "z_loss_weighted": 0.0,
        "alt_x_loss": 0.0,
        "alt_x_loss_weighted": 0.0,
        "x_constraint_loss": 0.0,
        "grad_norm_encoder_latent": 0.0,
        "grad_norm_encoder_reco": 0.0,
        "grad_norm_encoder_anchor": 0.0,
        "grad_norm_encoder_latent_weighted": 0.0,
        "grad_norm_encoder_reco_weighted": 0.0,
        "grad_norm_encoder_anchor_weighted": 0.0,
        "grad_norm_encoder_total": 0.0,
        "grad_norm_decoder_total": 0.0,
        "grad_cosine_latent_reco": 0.0,
    }
    nbatches = 0
    steps_in_epoch = min(len(x_loader), len(z_loader))
    vanilla = bool(getattr(loss_factory, "vanilla_swae", False))
    for x, z in zip(x_loader, z_loader):
        x = x.to(device)
        z = z.to(device)
        if hasattr(loss_factory, "reset_components"):
            loss_factory.reset_components()
        optimizer.zero_grad()

        # All three raw losses are always evaluated. The scheduled weights
        # only affect the gradient (``loss``); the raw terms are logged so a
        # fixed-weight reference remains comparable across the cosine schedule.
        z_tilde = first_tensor(model.encode(x))
        z_loss = loss_factory.z_prior_loss(z, z_tilde)
        x_tilde = first_tensor(model.decode(z_tilde))
        x_loss = loss_factory.x_reco_loss(x, x_tilde)

        encoder_anchor = (
            loss_factory.encoder_anchor_loss(z_tilde, x)
            if stage["nu_e"] > 0
            else x.new_tensor(0.0)
        )
        decoder_anchor = x.new_tensor(0.0)
        alt_x_loss = x.new_tensor(0.0)
        x_constraint_loss = x.new_tensor(0.0)

        needs_direct_decoder = (
            vanilla
            or stage["tau"] > 0
            or stage["rho"] > 0
            or stage["nu_d"] > 0
        )
        if needs_direct_decoder:
            decoder_samples = max(1, int(getattr(loss_factory, "decoder_num_noise_samples", 1)))
            if decoder_samples > 1 and stage["tau"] > 0:
                model_x = torch.cat(
                    [first_tensor(model.decode(z)) for _ in range(decoder_samples)],
                    dim=0,
                )
                x_for_distribution = x.repeat((decoder_samples, 1))
            else:
                model_x = first_tensor(model.decode(z))
                x_for_distribution = x
            alt_x_loss = loss_factory.x_sim_loss(x_for_distribution, model_x)
            decoder_anchor = (
                loss_factory.decoder_anchor_loss(z, model_x[: len(z)])
                if stage["nu_d"] > 0
                else x.new_tensor(0.0)
            )

        reference_loss = x_loss + z_loss + alt_x_loss
        loss = (
            stage["beta"] * x_loss
            + stage["lamb"] * z_loss
            + stage["tau"] * alt_x_loss
            + stage["rho"] * x_constraint_loss
            + stage["nu_e"] * encoder_anchor
            + stage["nu_d"] * decoder_anchor
        )

        encoder_params = [param for param in model.encoder.parameters() if param.requires_grad]
        grad_latent = None
        grad_reco = None
        grad_anchor = None
        if encoder_params:
            if stage["lamb"] > 0 and z_loss.requires_grad:
                grad_latent = encoder_grads_for(z_loss, model)
            if stage["beta"] > 0 and x_loss.requires_grad:
                grad_reco = encoder_grads_for(x_loss, model)
            if stage["nu_e"] > 0 and encoder_anchor.requires_grad:
                grad_anchor = encoder_grads_for(encoder_anchor, model)
        loss.backward()
        total_grad_norm = 0.0
        if encoder_params:
            total_grad_norm = grad_norm(
                [param.grad for param in encoder_params if param.grad is not None]
            )
        decoder_params = [param for param in model.decoder.parameters() if param.requires_grad]
        decoder_total_norm = grad_norm(
            [param.grad for param in decoder_params if param.grad is not None]
        )
        optimizer.step()

        sums["loss"] += as_float(loss)
        sums["reference_loss"] += as_float(reference_loss)
        sums["x_loss"] += as_float(x_loss)
        sums["z_loss"] += as_float(z_loss)
        sums["z_loss_weighted"] += stage["lamb"] * as_float(z_loss)
        sums["alt_x_loss"] += as_float(alt_x_loss)
        sums["alt_x_loss_weighted"] += stage["tau"] * as_float(alt_x_loss)
        sums["x_constraint_loss"] += as_float(x_constraint_loss)
        sums["grad_norm_encoder_latent"] += grad_norm(grad_latent)
        sums["grad_norm_encoder_reco"] += grad_norm(grad_reco)
        sums["grad_norm_encoder_anchor"] += grad_norm(grad_anchor)
        sums["grad_norm_encoder_latent_weighted"] += stage["lamb"] * grad_norm(grad_latent)
        sums["grad_norm_encoder_reco_weighted"] += stage["beta"] * grad_norm(grad_reco)
        sums["grad_norm_encoder_anchor_weighted"] += stage["nu_e"] * grad_norm(grad_anchor)
        sums["grad_norm_encoder_total"] += total_grad_norm
        sums["grad_norm_decoder_total"] += decoder_total_norm
        cosine = grad_cosine(grad_latent, grad_reco)
        sums["grad_cosine_latent_reco"] += 0.0 if cosine is None else cosine
        for key, value in getattr(loss_factory, "latest_components", {}).items():
            sums.setdefault(key, 0.0)
            sums[key] += as_float(value)
        nbatches += 1
        if step_callback is not None:
            step_callback(
                {
                    key: value / max(1, nbatches)
                    for key, value in sums.items()
                },
                nbatches,
                steps_in_epoch,
            )
    return {key: value / max(1, nbatches) for key, value in sums.items()}


def eval_standard_epoch(model, x_loader, z_loader, loss_factory, device):
    """Evaluate all three raw losses and a fixed-weight selection score.

    The training schedule is deliberately not used here. Checkpoint selection
    compares ``selection_score`` with fixed ``loss.selection_score`` weights,
    and ``reference_loss`` is the schedule-independent alpha=lambda=1 target
    (raw x_sim + z_prior + x_reco). Both can therefore be compared across
    epochs and across runs with different cosine schedules.
    """
    model.eval()
    sums = {
        "loss": 0.0,
        "reference_loss": 0.0,
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
            reference_loss = x_loss + z_loss + alt_x_loss
            loss = loss_factory.validation_score(
                {
                    "x_loss": x_loss,
                    "z_loss": z_loss,
                    "alt_x_loss": alt_x_loss,
                    "cycle_loss": x_loss,
                }
            )
            sums["loss"] += as_float(loss)
            sums["reference_loss"] += as_float(reference_loss)
            sums["x_loss"] += as_float(x_loss)
            sums["z_loss"] += as_float(z_loss)
            sums["alt_x_loss"] += as_float(alt_x_loss)
            sums["selection_score"] += as_float(loss)
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
    stage_checkpoint_callback=None,
    diagnostics_callback=None,
    diagnostic_every: int = 10,
) -> tuple[dict[str, list[Any]], float | None, int]:
    history_epoch = 0
    history_step = 0
    best_eval_loss: float | None = None
    vanilla_v3_7 = bool(config.get("loss", {}).get("vanilla_v3_7", False))
    vanilla_swae_flag = bool(config.get("loss", {}).get("vanilla_swae", False))
    total_epochs = sum(
        int(stage["epochs"]) for stage in config["stages"] if stage.get("enabled", True)
    )
    steps_in_train_epoch = min(len(train_loaders[0]), len(train_loaders[1]))
    total_steps = total_epochs * steps_in_train_epoch
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

        if stage.get("mode", "standard") != "standard":
            raise ValueError("Unsupported stage mode " + repr(stage.get("mode")) + "; only 'standard' is supported.")

        for local_epoch in range(1, stage_epochs + 1):
            def report_train_step(train_losses, step, steps_in_epoch):
                nonlocal history_step
                history_step += 1
                if progress_callback is None:
                    return
                progress_callback(
                    {
                        "event": "train_step",
                        "stage": stage["name"],
                        "epoch": local_epoch,
                        "epochs_in_stage": stage_epochs,
                        "global_epoch": history_epoch + 1,
                        "total_epochs": total_epochs,
                        "step": step,
                        "steps_in_epoch": steps_in_epoch,
                        "global_step": history_step,
                        "total_steps": total_steps,
                        "percent": (100.0 * history_step / total_steps) if total_steps else 100.0,
                        "train_loss": float(train_losses["loss"]),
                        "train_reference_loss": float(train_losses.get("reference_loss", 0.0)),
                        "eval_loss": None,
                        "best_eval_loss": best_eval_loss,
                        "evaluated": False,
                    }
                )

            stage_loss_config = make_stage_loss_config(stage, local_epoch, stage_epochs)
            train_losses = train_standard_epoch(
                model,
                optimizer,
                train_loaders[0],
                train_loaders[1],
                stage_loss_config,
                loss_factory,
                device,
                report_train_step,
            )
            eval_fn = eval_standard_epoch
            if scheduler is not None:
                scheduler.step()

            history_epoch += 1
            if (
                diagnostics_callback is not None
                and (
                    history_epoch % max(1, int(diagnostic_every)) == 0
                    or local_epoch == stage_epochs
                )
            ):
                diagnostics_callback(model, history_epoch, stage["name"], loss_factory)
            should_log = (
                local_epoch == 1
                or local_epoch == stage_epochs
                or local_epoch % int(stage.get("log_freq", 10)) == 0
            )
            eval_losses = None
            eval_loss = None
            selection_score = None
            if should_log:
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
                    "train_reference_loss": train_losses.get("reference_loss", ""),
                    "train_x_loss": train_losses["x_loss"],
                    "train_z_loss": train_losses["z_loss"],
                    "train_z_loss_weighted": train_losses.get(
                        "z_loss_weighted",
                        train_losses.get("z_loss", 0.0) * stage_loss_config["lamb"],
                    ),
                    "train_alt_x_loss": train_losses.get("alt_x_loss", ""),
                    "train_alt_x_loss_weighted": train_losses.get("alt_x_loss_weighted", ""),
                    "train_x_constraint_loss": train_losses.get("x_constraint_loss", ""),
                    "train_x_reco_mass_huber_raw": train_losses.get("x_reco_mass_huber_raw", ""),
                    "train_x_reco_mass_huber_weighted": train_losses.get("x_reco_mass_huber_weighted", ""),
                    "eval_loss": eval_loss,
                    "eval_reference_loss": eval_losses.get("reference_loss", ""),
                    "eval_x_loss": eval_losses["x_loss"],
                    "eval_z_loss": eval_losses["z_loss"],
                    "eval_alt_x_loss": eval_alt_x_loss,
                    "eval_selection_score": selection_score,
                }
                for component_field in HistoryLogger.component_fields:
                    component_key = component_field.removeprefix("train_")
                    row[component_field] = train_losses.get(component_key, "")
                for component_field in HistoryLogger.weighted_component_fields:
                    component_key = component_field.removeprefix("train_")
                    row[component_field] = train_losses.get(component_key, "")
                for gradient_field in (
                    "grad_norm_encoder_latent",
                    "grad_norm_encoder_reco",
                    "grad_norm_encoder_anchor",
                    "grad_norm_encoder_latent_weighted",
                    "grad_norm_encoder_reco_weighted",
                    "grad_norm_encoder_anchor_weighted",
                    "grad_norm_encoder_total",
                    "grad_norm_decoder_total",
                    "grad_cosine_latent_reco",
                ):
                    row[gradient_field] = train_losses.get(gradient_field, "")
                for latent_field in (
                    HistoryLogger.latent_component_fields
                    + HistoryLogger.latent_component_weighted_fields
                ):
                    row[latent_field] = train_losses.get(latent_field, "")
                row["v37_lambda"] = stage_loss_config["lamb"] if vanilla_v3_7 else ""
                row["v38_lambda"] = stage_loss_config["lamb"] if vanilla_swae_flag else ""
                row["effective_alpha"] = stage_loss_config.get("alpha", "")
                row["effective_lambda_cycle"] = stage_loss_config["lambda_cycle"]
                row["learning_rate"] = stage["lr"]
                row["batch_size"] = config.get("loader_info", {}).get(
                    "train_batch_size", ""
                )
                row["num_slices"] = stage["num_slices"]
                row["n_train_x"] = train_loaders[0].n
                row["n_train_z"] = train_loaders[1].n
                row["n_val_x"] = eval_loaders[0].n
                row["n_val_z"] = eval_loaders[1].n
                row["global_step"] = history_step
                logger.append(row)
                is_best = best_eval_loss is None or selection_score < best_eval_loss
                if is_best:
                    best_eval_loss = selection_score
                save_callback(history_epoch, selection_score, is_best, eval_losses)

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
                if local_epoch == stage_epochs and stage_checkpoint_callback is not None:
                    stage_checkpoint_callback(stage["name"], history_epoch, selection_score)

            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "epoch_end",
                        "stage": stage["name"],
                        "epoch": local_epoch,
                        "epochs_in_stage": stage_epochs,
                        "global_epoch": history_epoch,
                        "total_epochs": total_epochs,
                        "step": steps_in_train_epoch,
                        "steps_in_epoch": steps_in_train_epoch,
                        "global_step": history_step,
                        "total_steps": total_steps,
                        "percent": (100.0 * history_epoch / total_epochs) if total_epochs else 100.0,
                        "train_loss": float(train_losses["loss"]),
                        "train_reference_loss": float(train_losses.get("reference_loss", 0.0)),
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
