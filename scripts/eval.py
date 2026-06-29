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
    return parser.parse_args()


def decode_in_batches(model, z: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(z), batch_size):
            batch = torch.as_tensor(z[start : start + batch_size], dtype=torch.float32, device=device)
            decoded = first_tensor(model.decode(batch))
            outputs.append(decoded.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


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

    truth_mass = invariant_mass(arrays["x_test"])
    pred_mass = invariant_mass(z_decoded)
    metrics, metric_arrays = residual_metrics(
        truth_mass,
        pred_mass,
        bins=int(args.bins),
        mass_range=(float(args.mass_range[0]), float(args.mass_range[1])),
        min_truth_count=int(model_config.get("evaluation", {}).get("min_truth_count", 20)),
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

    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
    else:
        output_dir = checkpoint_path.parent / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
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

    print("Using device:", device)
    print("Device report:", json.dumps(report, sort_keys=True))
    print("Wrote metrics:", output_dir / "metrics.json")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
