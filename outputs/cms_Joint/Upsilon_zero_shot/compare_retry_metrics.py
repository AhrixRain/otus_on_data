#!/usr/bin/env python
"""Compare the full 0j+1j Upsilon retry with the legacy ptj5 baseline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_OLD_METRICS = HERE / "quantitative_z_to_x" / "z_to_x_metrics.json"
DEFAULT_NEW_METRICS = (
    HERE / "0j1j_full_retry" / "quantitative_z_to_x" / "z_to_x_metrics.json"
)
DEFAULT_OLD_DECODED = HERE / "decoded" / "upsilon_prior_decoded_xspace.hdf5"
DEFAULT_NEW_DECODED = (
    HERE / "0j1j_full_retry" / "decoded" / "upsilon_0j1j_prior_decoded_xspace.hdf5"
)
DEFAULT_OUTPUT = HERE / "0j1j_full_retry" / "comparison_to_ptj5"
PT_EDGES = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, np.inf])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-metrics", type=Path, default=DEFAULT_OLD_METRICS)
    parser.add_argument("--new-metrics", type=Path, default=DEFAULT_NEW_METRICS)
    parser.add_argument("--old-decoded", type=Path, default=DEFAULT_OLD_DECODED)
    parser.add_argument("--new-decoded", type=Path, default=DEFAULT_NEW_DECODED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_pair(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["FDL/zData"]), np.asarray(handle["FDL/xData"])


def mass(values: np.ndarray) -> np.ndarray:
    energy = values[:, 3] + values[:, 7]
    momentum = values[:, :3] + values[:, 4:7]
    return np.sqrt(np.maximum(energy * energy - np.sum(momentum * momentum, axis=1), 0.0))


def pair_pt(values: np.ndarray) -> np.ndarray:
    return np.hypot(values[:, 0] + values[:, 4], values[:, 1] + values[:, 5])


def conditional_response(z: np.ndarray, x: np.ndarray) -> list[dict[str, float | int | str]]:
    pt = pair_pt(z)
    residual = mass(x) - mass(z)
    rows = []
    for low, high in zip(PT_EDGES[:-1], PT_EDGES[1:]):
        selected = (pt >= low) & (pt < high)
        if not np.any(selected):
            continue
        values = residual[selected]
        rows.append(
            {
                "pt_low_gev": float(low),
                "pt_high_gev": None if np.isinf(high) else float(high),
                "pt_label": f"{low:g}-{high:g}" if np.isfinite(high) else f">={low:g}",
                "events": int(np.sum(selected)),
                "mean_bias_gev": float(np.mean(values)),
                "median_bias_gev": float(np.median(values)),
                "q16_bias_gev": float(np.quantile(values, 0.16)),
                "q84_bias_gev": float(np.quantile(values, 0.84)),
            }
        )
    return rows


def extract_metrics(old: dict, new: dict) -> list[dict[str, float | str]]:
    selectors = {
        "mass_ks": lambda value: value["distribution_report"]["mass"]["ks"],
        "mass_w1_gev": lambda value: value["distribution_report"]["mass"]["w1_gev"],
        "pair_pt_ks": lambda value: value["distribution_report"]["pair_pt"]["ks"],
        "pair_pt_w1_gev": lambda value: value["distribution_report"]["pair_pt"]["w1_gev"],
        "feature_ks_mean": lambda value: value["distribution_report"]["feature_ks"]["mean"],
        "feature_ks_max": lambda value: value["distribution_report"]["feature_ks"]["max"],
        "projected_w1_mean": lambda value: value["distribution_report"]["projected_w1"]["mean"],
        "linear_c2st_auc": lambda value: value["distribution_report"]["c2st"]["linear"]["c2st_auc_mean"],
        "mlp_c2st_auc": lambda value: value["distribution_report"]["c2st"]["mlp"]["c2st_auc_mean"],
        "decoded_x_window_efficiency": lambda value: value["selection"]["decoded_x_window_efficiency"],
    }
    rows = []
    for name, selector in selectors.items():
        old_value = float(selector(old))
        new_value = float(selector(new))
        rows.append(
            {
                "metric": name,
                "ptj5": old_value,
                "full_0j1j": new_value,
                "relative_change": (new_value - old_value) / old_value,
            }
        )
    return rows


def plot_metric_comparison(rows: list[dict], output_dir: Path) -> None:
    lookup = {row["metric"]: row for row in rows}
    panels = [
        ("KS distances (lower is better)", ["mass_ks", "pair_pt_ks", "feature_ks_mean", "feature_ks_max"]),
        ("Wasserstein summaries (lower is better)", ["mass_w1_gev", "pair_pt_w1_gev", "projected_w1_mean"]),
        ("C2ST AUC (0.5 is ideal)", ["linear_c2st_auc", "mlp_c2st_auc"]),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for axis, (title, names) in zip(axes, panels):
        locations = np.arange(len(names))
        width = 0.36
        axis.bar(locations - width / 2, [lookup[name]["ptj5"] for name in names], width, label="Legacy ptj5")
        axis.bar(locations + width / 2, [lookup[name]["full_0j1j"] for name in names], width, label="Full 0j+1j")
        axis.set_xticks(locations, [name.replace("_", "\n") for name in names], fontsize=8)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)
    figure.savefig(output_dir / "baseline_vs_0j1j_metrics.png", dpi=180)
    plt.close(figure)


def plot_conditional_response(old_rows: list[dict], new_rows: list[dict], output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    labels = [row["pt_label"] for row in new_rows]
    label_position = {label: index for index, label in enumerate(labels)}
    for rows, label, color in (
        (old_rows, "Legacy ptj5", "#2563eb"),
        (new_rows, "Full 0j+1j", "#d97706"),
    ):
        positions = np.asarray([label_position[row["pt_label"]] for row in rows])
        medians = np.asarray([row["median_bias_gev"] for row in rows])
        lower = medians - np.asarray([row["q16_bias_gev"] for row in rows])
        upper = np.asarray([row["q84_bias_gev"] for row in rows]) - medians
        axis.errorbar(
            positions,
            medians * 1000.0,
            yerr=np.stack([lower, upper]) * 1000.0,
            marker="o",
            linewidth=1.5,
            capsize=3,
            color=color,
            label=label,
        )
    axis.set_xticks(np.arange(len(labels)), labels)
    axis.axhline(0.0, color="0.4", linestyle=":")
    axis.set_xlabel(r"Input-prior $p_T(\mu\mu)$ bin [GeV]")
    axis.set_ylabel(r"Median $m^x_{\mu\mu}-m^z_{\mu\mu}$ [MeV]")
    axis.set_title("Frozen decoder mass response versus input pair pT")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    figure.savefig(output_dir / "baseline_vs_0j1j_mass_bias_by_input_pair_pt.png", dpi=180)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    old_metrics_path = args.old_metrics.expanduser().resolve()
    new_metrics_path = args.new_metrics.expanduser().resolve()
    old_decoded_path = args.old_decoded.expanduser().resolve()
    new_decoded_path = args.new_decoded.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    old_metrics, new_metrics = load_json(old_metrics_path), load_json(new_metrics_path)
    old_z, old_x = load_pair(old_decoded_path)
    new_z, new_x = load_pair(new_decoded_path)
    rows = extract_metrics(old_metrics, new_metrics)
    old_response = conditional_response(old_z, old_x)
    new_response = conditional_response(new_z, new_x)

    with (output_dir / "baseline_vs_0j1j_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["metric", "ptj5", "full_0j1j", "relative_change"])
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "mass_response_by_input_pair_pt.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["sample", "pt_low_gev", "pt_high_gev", "pt_label", "events", "mean_bias_gev", "median_bias_gev", "q16_bias_gev", "q84_bias_gev"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for sample, response in (("ptj5", old_response), ("full_0j1j", new_response)):
            for row in response:
                writer.writerow({"sample": sample, **row})
    plot_metric_comparison(rows, output_dir)
    plot_conditional_response(old_response, new_response, output_dir)

    summary = {
        "schema_version": 1,
        "old_metrics": str(old_metrics_path),
        "new_metrics": str(new_metrics_path),
        "metrics": rows,
        "mass_response_by_input_pair_pt": {"ptj5": old_response, "full_0j1j": new_response},
        "conclusion": (
            "The 0j1j prior substantially improves pair-pT and multivariate closure, but exposes "
            "an out-of-distribution low-pT decoder response that shifts dimuon mass downward."
        ),
    }
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote retry comparison to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
