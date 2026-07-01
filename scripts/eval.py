from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cms_data import load_and_split, load_config, resolve_config, save_resolved_config
from cms_model import load_model_from_checkpoint
from cms_training import first_tensor
from device_utils import device_report, select_device
from metrics import invariant_mass, plot_mass_ratio, plot_residual, residual_metrics, write_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CMS DoubleElectron OTUS mass residuals.")
    parser.add_argument("--config", type=Path, required=True, help="YAML/JSON config path.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="best_model.pt or last_model.pt.")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Evaluation output directory.")
    parser.add_argument("--num-samples", type=int, default=None, help="Limit selected CMS and MG5 rows.")
    parser.add_argument("--bins", type=int, default=80, help="Mass histogram bins.")
    parser.add_argument("--mass-range", nargs=2, type=float, default=(70.0, 110.0), metavar=("LOW", "HIGH"))
    parser.add_argument(
        "--include-inverse-check",
        action="store_true",
        help="Also write the eval-only E(D(z_test)) vs z_test inverse check.",
    )
    return parser.parse_args()


def transform_in_batches(
    transform,
    values: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    outputs = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.as_tensor(
                values[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            transformed = first_tensor(transform(batch))
            outputs.append(transformed.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def decode_in_batches(model, z: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    return transform_in_batches(model.decode, z, device, batch_size)


def encode_in_batches(model, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    return transform_in_batches(model.encode, x, device, batch_size)


def write_mass_comparison(
    output_dir: Path,
    name: str,
    truth: np.ndarray,
    pred: np.ndarray,
    bins: int,
    mass_range: tuple[float, float],
    min_truth_count: int,
    labels: tuple[str, str],
    write_outputs: bool = True,
) -> tuple[dict, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    comparison_dir = output_dir / "comparisons" / name
    truth_mass = invariant_mass(truth)
    pred_mass = invariant_mass(pred)
    metrics, metric_arrays = residual_metrics(
        truth_mass,
        pred_mass,
        bins=bins,
        mass_range=mass_range,
        min_truth_count=min_truth_count,
    )
    metrics["comparison"] = name
    metrics["truth_label"] = labels[0]
    metrics["pred_label"] = labels[1]
    if write_outputs:
        comparison_dir.mkdir(parents=True, exist_ok=True)
        write_metrics(metrics, comparison_dir / "metrics.json")
        plot_mass_ratio(
            metric_arrays,
            comparison_dir / "mass_ratio.png",
            truth_label=labels[0],
            pred_label=labels[1],
        )
        plot_residual(metric_arrays, comparison_dir / "residual.png")
        np.savez(
            comparison_dir / "mass_histograms.npz",
            truth_mass=truth_mass,
            pred_mass=pred_mass,
            **metric_arrays,
        )
    return metrics, metric_arrays, truth_mass, pred_mass


def main() -> None:
    args = parse_args()
    config = resolve_config(load_config(args.config))
    device = select_device(args.device)
    report = device_report(device)

    checkpoint_path = args.checkpoint.expanduser().resolve()
    model, model_config, _stats, checkpoint = load_model_from_checkpoint(
        checkpoint_path,
        config=config,
        map_location=torch.device("cpu"),
    )
    model.to(device)

    arrays = load_and_split(model_config, num_samples=args.num_samples)
    batch_size = int(model_config.get("loaders", {}).get("eval_batch_size", 20000))
    z_decoded = decode_in_batches(model, arrays["z_test"], device, batch_size)

    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
    else:
        output_dir = checkpoint_path.parent / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluation_config = model_config.get("evaluation", {})
    loss_kind = model_config.get("loss", {}).get("kind")
    write_comparisons = bool(
        evaluation_config.get(
            "write_comparison_outputs",
            loss_kind in {"original_feature_ot_v1", "cms_doubleelectron_loss"},
        )
    )
    include_inverse_check = bool(
        args.include_inverse_check or evaluation_config.get("include_inverse_check", False)
    )

    mass_range = (float(args.mass_range[0]), float(args.mass_range[1]))
    min_truth_count = int(model_config.get("evaluation", {}).get("min_truth_count", 20))
    metrics, metric_arrays, truth_mass, pred_mass = write_mass_comparison(
        output_dir,
        "simulation",
        arrays["x_test"],
        z_decoded,
        bins=int(args.bins),
        mass_range=mass_range,
        min_truth_count=min_truth_count,
        labels=("x_test CMS", "D(z_test MG5)"),
        write_outputs=write_comparisons,
    )
    metrics.update(
        {
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_eval_loss": checkpoint.get("eval_loss"),
            "device_report": report,
            "split": "test",
        }
    )

    save_resolved_config(model_config, output_dir / "config.resolved.json")
    write_metrics(metrics, output_dir / "metrics.json")
    plot_mass_ratio(metric_arrays, output_dir / "mass_ratio.png")
    plot_residual(metric_arrays, output_dir / "residual.png")
    np.savez(
        output_dir / "mass_histograms.npz",
        truth_mass=truth_mass,
        pred_mass=pred_mass,
        **metric_arrays,
    )

    if write_comparisons:
        write_metrics(metrics, output_dir / "comparisons" / "simulation" / "metrics.json")
        comparison_metrics = {"simulation": metrics}
        z_encoded = encode_in_batches(model, arrays["x_test"], device, batch_size)
        x_reconstructed = decode_in_batches(model, z_encoded, device, batch_size)
        reco_metrics, _, _, _ = write_mass_comparison(
            output_dir,
            "reconstruction",
            arrays["x_test"],
            x_reconstructed,
            bins=int(args.bins),
            mass_range=mass_range,
            min_truth_count=min_truth_count,
            labels=("x_test CMS", "D(E(x_test CMS))"),
        )
        comparison_metrics["reconstruction"] = reco_metrics
        unfold_metrics, _, _, _ = write_mass_comparison(
            output_dir,
            "unfolding",
            arrays["z_test"],
            z_encoded,
            bins=int(args.bins),
            mass_range=mass_range,
            min_truth_count=min_truth_count,
            labels=("z_test MG5", "E(x_test CMS)"),
        )
        comparison_metrics["unfolding"] = unfold_metrics
        if include_inverse_check:
            z_inverse = encode_in_batches(model, z_decoded, device, batch_size)
            inverse_metrics, _, _, _ = write_mass_comparison(
                output_dir,
                "inverse_check",
                arrays["z_test"],
                z_inverse,
                bins=int(args.bins),
                mass_range=mass_range,
                min_truth_count=min_truth_count,
                labels=("z_test MG5", "E(D(z_test MG5))"),
            )
            comparison_metrics["inverse_check"] = inverse_metrics
        write_metrics(comparison_metrics, output_dir / "comparisons" / "metrics.json")

    print("Using device:", device)
    print("Device report:", json.dumps(report, sort_keys=True))
    print("Wrote metrics:", output_dir / "metrics.json")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
