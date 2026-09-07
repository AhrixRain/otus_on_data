#!/usr/bin/env python
"""Decode the labeled Upsilon z prior with a frozen cms_Joint checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch


HERE = Path(__file__).resolve().parent


def find_repo_root() -> Path:
    for candidate in (HERE, *HERE.parents):
        if (candidate / "scripts_joint").is_dir() and (candidate / "scripts_sota").is_dir():
            return candidate
    raise RuntimeError("Could not locate the OTUS repository root")


REPO_ROOT = find_repo_root()
for directory in (
    REPO_ROOT / "scripts",
    REPO_ROOT / "scripts_sota",
    REPO_ROOT / "scripts_joint",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from joint_model import JointDimuonAutoencoder  # noqa: E402
from joint_trainer import restore_joint_checkpoint  # noqa: E402


DEFAULT_PRIOR = (
    REPO_ROOT
    / "data"
    / "cms_upsilon_mumu_mg5_8tev_inclusive_3S_continuum_ptj5_fiducial_8p5_11p5_1M.hdf5"
)
DEFAULT_RUN_DIR = REPO_ROOT / "outputs" / "cms_Joint" / "Run_C_fullScale"
DEFAULT_OUTPUT = HERE / "decoded" / "upsilon_prior_decoded_xspace.hdf5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Defaults to <run-dir>/best_model.pt",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(requested)


def _numpy_buffer(state: dict[str, torch.Tensor], key: str) -> np.ndarray:
    if key not in state:
        raise KeyError(f"Checkpoint is missing {key!r}")
    return state[key].detach().cpu().numpy()


def load_frozen_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict")
    config = checkpoint.get("config")
    if not isinstance(state, dict) or not isinstance(config, dict):
        raise ValueError("Checkpoint lacks model_state_dict or its resolved config")

    encoder_indices = tuple(
        int(value) for value in _numpy_buffer(state, "encoder.condition_indices").tolist()
    )
    decoder_indices = tuple(
        int(value) for value in _numpy_buffer(state, "decoder.condition_indices").tolist()
    )
    if encoder_indices != decoder_indices:
        raise ValueError("Encoder and decoder condition masks differ unexpectedly")
    if 8 in decoder_indices:
        raise ValueError("Frozen joint contract violation: explicit pair mass is conditioned")

    model_config = config["model"]
    model = JointDimuonAutoencoder(
        (
            _numpy_buffer(state, "encoder.condition_mean"),
            _numpy_buffer(state, "encoder.condition_std"),
        ),
        (
            _numpy_buffer(state, "decoder.condition_mean"),
            _numpy_buffer(state, "decoder.condition_std"),
        ),
        encoder_indices,
        model_config,
        float(config.get("muon_mass_gev", 0.1056583755)),
        model_config.get("daughter_masses", [0.1056583755, 0.1056583755]),
    ).to(device)
    restore_joint_checkpoint(model, checkpoint)
    model.eval()
    return model, checkpoint, config


def invariant_mass(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    energy = values[:, 3] + values[:, 7]
    momentum = values[:, :3] + values[:, 4:7]
    return np.sqrt(np.maximum(energy * energy - np.sum(momentum * momentum, axis=1), 0.0))


def mass_summary(masses: np.ndarray) -> dict[str, float | int]:
    finite = masses[np.isfinite(masses)]
    return {
        "events": int(len(masses)),
        "finite_events": int(len(finite)),
        "mean_gev": float(np.mean(finite)),
        "std_gev": float(np.std(finite)),
        "median_gev": float(np.median(finite)),
        "q001_gev": float(np.quantile(finite, 0.001)),
        "q999_gev": float(np.quantile(finite, 0.999)),
    }


def _json_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> int:
    args = parse_args()
    prior_path = args.prior.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else run_dir / "best_model.pt"
    )
    output_path = args.output.expanduser().resolve()
    summary_path = output_path.with_suffix(".summary.json")
    if not prior_path.exists():
        raise FileNotFoundError(f"Upsilon prior not found: {prior_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Frozen checkpoint not found: {checkpoint_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists; pass --overwrite to replace it: {output_path}")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.max_events is not None and args.max_events < 1:
        raise ValueError("max-events must be positive when supplied")

    device = choose_device(args.device)
    model, checkpoint, config = load_frozen_model(checkpoint_path, device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".partial")
    if temporary_path.exists():
        temporary_path.unlink()

    z_masses_parts: list[np.ndarray] = []
    x_masses_parts: list[np.ndarray] = []
    with h5py.File(prior_path, "r") as source:
        if "FDL/zData" not in source:
            raise KeyError("Expected FDL/zData in the Upsilon prior")
        z_source = source["FDL/zData"]
        component_source = source.get("FDL/component_id")
        event_count = len(z_source)
        if args.max_events is not None:
            event_count = min(event_count, int(args.max_events))
        source_attrs = {key: _json_attr(value) for key, value in source.attrs.items()}
        source_attrs.update(
            {
                key: _json_attr(value)
                for key, value in z_source.attrs.items()
                if key not in source_attrs
            }
        )
        source_component_counts = source_attrs.get("component_counts")
        selected_component_ids = None
        output_component_counts = None
        if component_source is not None:
            selected_component_ids = np.asarray(component_source[:event_count])
            raw_mapping = source_attrs.get("component_id_mapping", "{}")
            component_mapping = (
                json.loads(raw_mapping) if isinstance(raw_mapping, str) else dict(raw_mapping)
            )
            output_component_counts = {
                str(name): int(np.sum(selected_component_ids == int(identifier)))
                for name, identifier in component_mapping.items()
            }
            source_attrs["component_counts"] = json.dumps(
                output_component_counts, sort_keys=True
            )
            source_attrs["component_target_fractions"] = json.dumps(
                {
                    name: count / event_count
                    for name, count in output_component_counts.items()
                },
                sort_keys=True,
            )

        try:
            with h5py.File(temporary_path, "w") as destination:
                group = destination.create_group("FDL")
                chunk_rows = min(max(1024, args.batch_size), event_count)
                compression = {"compression": "gzip", "compression_opts": 4, "shuffle": True}
                z_output = group.create_dataset(
                    "zData",
                    shape=(event_count, 8),
                    dtype=np.float32,
                    chunks=(chunk_rows, 8),
                    **compression,
                )
                x_output = group.create_dataset(
                    "xData",
                    shape=(event_count, 8),
                    dtype=np.float32,
                    chunks=(chunk_rows, 8),
                    **compression,
                )
                component_output = None
                if component_source is not None:
                    component_output = group.create_dataset(
                        "component_id",
                        shape=(event_count,),
                        dtype=component_source.dtype,
                        chunks=(chunk_rows,),
                        **compression,
                    )

                with torch.inference_mode():
                    for start in range(0, event_count, args.batch_size):
                        stop = min(start + args.batch_size, event_count)
                        z_batch = np.asarray(z_source[start:stop, :8], dtype=np.float32)
                        tensor = torch.as_tensor(z_batch, dtype=torch.float32, device=device)
                        x_batch = model.decode(tensor).detach().cpu().numpy().astype(np.float32)
                        if not np.isfinite(x_batch).all():
                            raise FloatingPointError(f"Non-finite decoded values in rows {start}:{stop}")
                        z_output[start:stop] = z_batch
                        x_output[start:stop] = x_batch
                        if component_output is not None:
                            component_output[start:stop] = selected_component_ids[start:stop]
                        z_masses_parts.append(invariant_mass(z_batch))
                        x_masses_parts.append(invariant_mass(x_batch))
                        if start == 0 or stop == event_count or stop % (10 * args.batch_size) == 0:
                            print(f"Decoded {stop:,}/{event_count:,} events on {device}")

                holdout_status = str(
                    config.get("holdout", {}).get("status", "unspecified")
                )
                evaluation_scope = (
                    "locked_zero_shot"
                    if holdout_status == "locked_zero_shot"
                    else "post_unblinding_heldout_transfer"
                )
                provenance = {
                    "schema_version": 1,
                    "purpose": "Frozen cms_Joint Upsilon prior decode",
                    "evaluation_scope": evaluation_scope,
                    "holdout_status": holdout_status,
                    "source_prior": str(prior_path),
                    "source_prior_sha256": file_sha256(prior_path),
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": file_sha256(checkpoint_path),
                    "run_dir": str(run_dir),
                    "run_label": config.get("run_label"),
                    "checkpoint_stage": checkpoint.get("stage", {}).get("name"),
                    "checkpoint_global_epoch": checkpoint.get("global_epoch"),
                    "checkpoint_stage_epoch": checkpoint.get("stage_epoch"),
                    "checkpoint_noise_multipliers": checkpoint.get("noise_multipliers"),
                    "joint_contract_sha256": checkpoint.get("joint_contract_sha256"),
                    "device": str(device),
                    "seed": int(args.seed),
                    "events": int(event_count),
                    "component_labels_preserved": component_source is not None,
                    "source_component_counts": source_component_counts,
                    "output_component_counts": output_component_counts,
                    "condition_features": config.get("model", {}).get("condition_features"),
                    "explicit_pair_mass_conditioned": False,
                    "source_composition_scope": source_attrs.get("composition_scope"),
                }
                for target in (destination, group, z_output, x_output):
                    target.attrs["columns"] = str(
                        source_attrs.get(
                            "columns",
                            "mu_minus_px,mu_minus_py,mu_minus_pz,mu_minus_E,"
                            "mu_plus_px,mu_plus_py,mu_plus_pz,mu_plus_E",
                        )
                    )
                    target.attrs["units"] = str(source_attrs.get("units", "GeV"))
                for key in (
                    "component_counts",
                    "component_id_mapping",
                    "component_names",
                    "component_target_fractions",
                    "composition_scope",
                    "composition_source",
                    "daughter_muon_mass_GeV",
                    "mass_window_GeV",
                    "particle_order",
                ):
                    if key in source_attrs:
                        destination.attrs[key] = source_attrs[key]
                        group.attrs[key] = source_attrs[key]
                        if component_output is not None:
                            component_output.attrs[key] = source_attrs[key]
                destination.attrs["decode_provenance"] = json.dumps(provenance, sort_keys=True)
                group.attrs["decode_provenance"] = json.dumps(provenance, sort_keys=True)
                x_output.attrs["space"] = "x (decoded detector/data space)"
                z_output.attrs["space"] = "z (input theory/prior space)"
                destination.flush()
        except Exception:
            if temporary_path.exists():
                temporary_path.unlink()
            raise

    temporary_path.replace(output_path)
    z_masses = np.concatenate(z_masses_parts)
    x_masses = np.concatenate(x_masses_parts)
    summary = {
        "schema_version": 1,
        "output": str(output_path),
        "output_sha256": file_sha256(output_path),
        "source_prior": str(prior_path),
        "checkpoint": str(checkpoint_path),
        "run_label": config.get("run_label"),
        "checkpoint_stage": checkpoint.get("stage", {}).get("name"),
        "checkpoint_global_epoch": checkpoint.get("global_epoch"),
        "checkpoint_noise_multipliers": checkpoint.get("noise_multipliers"),
        "evaluation_scope": (
            "locked_zero_shot"
            if config.get("holdout", {}).get("status") == "locked_zero_shot"
            else "post_unblinding_heldout_transfer"
        ),
        "device": str(device),
        "seed": int(args.seed),
        "z_mass": mass_summary(z_masses),
        "decoded_x_mass": mass_summary(x_masses),
        "interpretation_note": (
            "One stochastic decoder draw is stored per prior event. Component IDs are copied "
            "without alteration. The checkpoint is loaded read-only and is not tuned on Upsilon. "
            "The evaluation scope is inherited from the run's holdout metadata."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
