#!/usr/bin/env python
"""Generate the legacy paper-style density/ratio suite for a joint run."""


from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (
    REPO_ROOT,
    REPO_ROOT / "scripts",
    REPO_ROOT / "scripts_sota",
    REPO_ROOT / "scripts_joint",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import plot as pplot  # noqa: E402
from cms_data import load_config  # noqa: E402
from device_utils import select_device  # noqa: E402
from joint_data import load_joint_regions, resolve_joint_config  # noqa: E402
from joint_metrics import transform_batches  # noqa: E402
from joint_model import build_joint_autoencoder  # noqa: E402
from joint_trainer import restore_joint_checkpoint  # noqa: E402


COMPONENT_TITLES = [
    r"$\mu^- p_x$",
    r"$\mu^- p_y$",
    r"$\mu^- p_z$",
    r"$\mu^- E$",
    r"$\mu^+ p_x$",
    r"$\mu^+ p_y$",
    r"$\mu^+ p_z$",
    r"$\mu^+ E$",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split", choices=("all", "val", "test"), default="test")
    parser.add_argument("--max-events", type=int, default=15000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--component-bins", type=int, default=80)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--interop-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--counts", action="store_true")
    return parser.parse_args()


def _split(arrays: dict[str, np.ndarray], prefix: str, split: str) -> np.ndarray:
    if split == "all":
        return np.concatenate(
            [arrays[f"{prefix}_train"], arrays[f"{prefix}_val"], arrays[f"{prefix}_test"]],
            axis=0,
        )
    return arrays[f"{prefix}_{split}"]


def _equal_subset(
    x: np.ndarray,
    z: np.ndarray,
    max_events: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(x), len(z), int(max_events))
    if n < 2:
        raise ValueError("Paper-style plots need at least two x and z events")
    rng = np.random.default_rng(seed)
    x_idx = rng.choice(len(x), n, replace=False)
    z_idx = rng.choice(len(z), n, replace=False)
    return (
        np.asarray(x[x_idx], dtype=np.float32),
        np.asarray(z[z_idx], dtype=np.float32),
    )


def _symmetric_bins(arrays: list[np.ndarray], index: int, count: int) -> np.ndarray:
    values = np.concatenate([a[:, index] for a in arrays])
    values = values[np.isfinite(values)]
    bound = float(np.quantile(np.abs(values), 0.998))
    if not np.isfinite(bound) or bound <= 0:
        bound = 1.0
    return np.linspace(-1.03 * bound, 1.03 * bound, int(count) + 1)


def _energy_bins(arrays: list[np.ndarray], index: int, count: int) -> np.ndarray:
    values = np.concatenate([a[:, index] for a in arrays])
    values = values[np.isfinite(values)]
    high = float(np.quantile(values, 0.998))
    if not np.isfinite(high) or high <= 0:
        high = 1.0
    return np.linspace(0.0, 1.03 * high, int(count) + 1)


def _mass_settings(region_config: dict, region: str) -> tuple[float, float, float, float, str]:
    selection = region_config["theory_prior_selection"]
    low = float(selection["mass_min"])
    high = float(selection["mass_max"])
    if region.lower() == "jpsi":
        return low, high, 0.002, 3.0969, r"$m_{J/\psi}$"
    return low, high, 0.5, 91.1876, r"$m_Z$"


def _make_region_plots(
    *,
    model,
    arrays: dict[str, np.ndarray],
    region: str,
    region_config: dict,
    run_label: str,
    output_dir: Path,
    device: torch.device,
    split: str,
    max_events: int,
    batch_size: int,
    component_bins: int,
    seed: int,
    density: bool,
    daughter_masses,
) -> dict:
    x, z = _equal_subset(
        pplot.finite_8d(_split(arrays, "x", split), f"{region} x"),
        pplot.finite_8d(_split(arrays, "z", split), f"{region} z"),
        max_events,
        seed,
    )

    torch.manual_seed(seed)
    z_from_x = transform_batches(model.encode, x, device=device, batch_size=batch_size)
    torch.manual_seed(seed + 1)
    x_reco = transform_batches(model.decode, z_from_x, device=device, batch_size=batch_size)
    torch.manual_seed(seed + 2)
    x_from_z = transform_batches(model.decode, z, device=device, batch_size=batch_size)

    output_dir.mkdir(parents=True, exist_ok=True)
    channel = str(region_config.get("label", region)).replace("->", r"$\to$")
    title_prefix = f"{run_label} — {channel}"
    mass_low, mass_high, mass_width, reference_mass, reference_name = _mass_settings(
        region_config, region
    )
    mass_bins = np.arange(mass_low, mass_high + 0.5 * mass_width, mass_width)
    mass_label = r"$m_{\mu\mu}$ [GeV]"
    reference_label = f"{reference_name}={reference_mass:.4f} GeV"

    m_x = pplot.inv_mass_ee(x, daughter_masses=daughter_masses, stable=True)
    m_x_reco = pplot.inv_mass_ee(x_reco, daughter_masses=daughter_masses, stable=True)
    m_x_from_z = pplot.inv_mass_ee(x_from_z, daughter_masses=daughter_masses, stable=True)
    m_z = pplot.inv_mass_ee(z, daughter_masses=daughter_masses, stable=False)
    m_z_from_x = pplot.inv_mass_ee(z_from_x, daughter_masses=daughter_masses, stable=False)

    pt_x = pplot.pt_ee(x)
    pt_x_reco = pplot.pt_ee(x_reco)
    pt_x_from_z = pplot.pt_ee(x_from_z)
    pt_high = float(
        np.quantile(np.concatenate([pt_x, pt_x_reco, pt_x_from_z]), 0.998)
    )
    pt_high = max(pt_high * 1.03, 1.0)
    pt_bins = np.linspace(0.0, pt_high, component_bins + 1)

    bins_y = _symmetric_bins([x, x_reco, x_from_z, z, z_from_x], 5, component_bins)
    bins_z = _symmetric_bins([x, x_reco, x_from_z, z, z_from_x], 6, component_bins)
    bins_e = _energy_bins([x, x_reco, x_from_z, z, z_from_x], 7, component_bins)

    common_x_labels = {
        "truth_label": "CMS data: x",
        "pred1_label": r"Data cycle: $x \rightarrow \tilde{z} \rightarrow \tilde{x}$",
        "pred2_label": r"Decoded prior: $z \rightarrow \tilde{x}^{\prime}$",
    }
    common_z_labels = {
        "truth_label": "Theory prior: z",
        "pred_label": r"Encoded data: $x \rightarrow \tilde{z}$",
    }

    pplot.paper_ratio_plot_double(
        truth=m_x,
        pred1=m_x_reco,
        pred2=m_x_from_z,
        bins=mass_bins,
        xlabel=mass_label,
        title=f"{title_prefix}: x-space mass",
        path=output_dir / "paperstyle_xspace_mass_density_ratio.png",
        density=density,
        xlim=(mass_low, mass_high),
        residual_ylim=(-0.05, 0.05),
        reference_x=reference_mass,
        reference_label=reference_label,
        **common_x_labels,
    )
    pplot.paper_ratio_plot_double(
        truth=pt_x,
        pred1=pt_x_reco,
        pred2=pt_x_from_z,
        bins=pt_bins,
        xlabel=r"$p_T(\mu\mu)$ [GeV]",
        title=f"{title_prefix}: dilepton transverse momentum",
        path=output_dir / "paperstyle_xspace_pt_density_ratio.png",
        density=density,
        xlim=(0.0, pt_high),
        ratio_ylim=(0.0, 2.0),
        **common_x_labels,
    )

    component_specs = [
        (5, bins_y, r"$\mu^+\ p_y$ [GeV]", "py"),
        (6, bins_z, r"$\mu^+\ p_z$ [GeV]", "pz"),
        (7, bins_e, r"$\mu^+\ E$ [GeV]", "E"),
    ]
    for index, bins, xlabel, suffix in component_specs:
        xlim = (float(bins[0]), float(bins[-1]))
        pplot.paper_ratio_plot_single(
            truth=z[:, index],
            pred=z_from_x[:, index],
            bins=bins,
            xlabel=xlabel,
            title=f"{title_prefix}: z-space closure — {xlabel}",
            path=output_dir / f"paperstyle_zspace_pos_{suffix}_ratio.png",
            density=density,
            xlim=xlim,
            ratio_ylim=(0.5, 1.5),
            **common_z_labels,
        )
        pplot.paper_ratio_plot_double(
            truth=x[:, index],
            pred1=x_reco[:, index],
            pred2=x_from_z[:, index],
            bins=bins,
            xlabel=xlabel,
            title=f"{title_prefix}: x-space closure — {xlabel}",
            path=output_dir / f"paperstyle_xspace_pos_{suffix}_ratio.png",
            density=density,
            xlim=xlim,
            ratio_ylim=(0.5, 1.5),
            **common_x_labels,
        )

    pplot.paper_ratio_plot_single(
        truth=m_z,
        pred=m_z_from_x,
        bins=mass_bins,
        xlabel=mass_label,
        title=f"{title_prefix}: z-space mass check",
        path=output_dir / "paperstyle_zspace_mass_density_ratio.png",
        density=density,
        xlim=(mass_low, mass_high),
        ratio_ylim=(0.0, 2.0),
        reference_x=reference_mass,
        reference_label=reference_label,
        **common_z_labels,
    )
    pplot.plot_all_components(
        [
            (z, "Theory prior: z", pplot.TRUTH_STYLE),
            (z_from_x, r"Encoded data: $x \rightarrow \tilde{z}$", pplot.ENC_STYLE),
        ],
        f"{title_prefix}: z-space component closure",
        output_dir / "all8_zspace_components_density.png",
        component_bins,
        density,
        COMPONENT_TITLES,
    )
    pplot.plot_all_components(
        [
            (x, "CMS data: x", pplot.TRUTH_STYLE),
            (x_reco, "Data cycle", pplot.ENCDEC_STYLE),
            (x_from_z, "Decoded prior", pplot.DEC_STYLE),
        ],
        f"{title_prefix}: x-space component closure",
        output_dir / "all8_xspace_components_density.png",
        component_bins,
        density,
        COMPONENT_TITLES,
    )

    return {
        "events_per_distribution": int(len(x)),
        "split": split,
        "mass_range_gev": [mass_low, mass_high],
        "mass_bin_width_gev": mass_width,
        "files": sorted(path.name for path in output_dir.glob("*.png")),
    }


def main() -> int:
    args = parse_args()
    torch.set_num_threads(max(1, int(args.threads)))
    try:
        torch.set_num_interop_threads(max(1, int(args.interop_threads)))
    except RuntimeError:
        pass

    config_path = args.config.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else checkpoint_path.parent / "plots_paperstyle"
    )
    density = not args.counts
    config = resolve_joint_config(load_config(config_path))
    arrays, _, _ = load_joint_regions(config, num_samples=None, use_cache=True)
    device = select_device(args.device)
    model = build_joint_autoencoder(
        config["model"],
        arrays,
        float(config.get("muon_mass_gev", 0.1056583755)),
        config["model"].get("daughter_masses"),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    restore_joint_checkpoint(model, checkpoint)
    model.eval()

    manifest = {
        "run": str(config.get("run_label", checkpoint_path.parent.name)),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("global_epoch"),
        "stage": (checkpoint.get("stage") or {}).get("name"),
        "density": density,
        "regions": {},
    }
    for offset, region in enumerate(config["region_order"]):
        print(f"Generating paper-style plots for {manifest['run']} / {region} ...", flush=True)
        manifest["regions"][region] = _make_region_plots(
            model=model,
            arrays=arrays[region],
            region=region,
            region_config=config["regions"][region],
            run_label=manifest["run"],
            output_dir=output_dir / region,
            device=device,
            split=args.split,
            max_events=args.max_events,
            batch_size=args.batch_size,
            component_bins=args.component_bins,
            seed=args.seed + 1000 * offset,
            density=density,
            daughter_masses=config["model"].get("daughter_masses"),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "paperstyle_manifest.json"
    manifest_path.write_text(
        json.dumps(pplot.json_safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    total = sum(len(region["files"]) for region in manifest["regions"].values())
    print(f"Wrote {total} paper-style plots to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
