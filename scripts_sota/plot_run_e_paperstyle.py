#!/usr/bin/env python
"""Paper-style x-space / z-space plot suite for cylindrical-flow Run E.

This is the Run E equivalent of ``scripts/plot.py``.  It reuses the original
plotting primitives but loads the ``CylindricalFlowAutoencoder`` checkpoint
directly.  Run with ``--device cpu`` while a training process owns MPS.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "scripts_sota"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import matplotlib
matplotlib.use("Agg")

import plot as pplot  # noqa: E402  (original plotting primitives)
from cms_data import load_and_split_cached, load_config, resolve_config, save_resolved_config  # noqa: E402
from device_utils import device_report, select_device  # noqa: E402
from physics import daughter_masses_from_config  # noqa: E402
from run_e import build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu", help="Use cpu while training is running.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--split", choices=("all", "val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--interop-threads", type=int, default=4)
    parser.add_argument("--max-x-events", type=int, default=5000000)
    parser.add_argument("--max-z-events", type=int, default=5000000)
    parser.add_argument("--mass-low", type=float, default=None)
    parser.add_argument("--mass-high", type=float, default=None)
    parser.add_argument("--mass-bin-width", type=float, default=None)
    parser.add_argument("--counts", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def predict_batches(model, mode, arr, device, batch_size):
    out = []
    for start in range(0, len(arr), batch_size):
        batch = torch.as_tensor(np.ascontiguousarray(arr[start:start+batch_size]), dtype=torch.float32, device=device)
        if mode == "encode":
            value = model.encode(batch)
        elif mode == "decode":
            value = model.decode(batch)
        elif mode == "reconstruct":
            value = model.decode(model.encode(batch))
        else:
            raise ValueError(mode)
        if isinstance(value, tuple):
            value = value[0]
        out.append(value.detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def main() -> int:
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.interop_threads)
    density = not args.counts
    device = select_device(args.device)
    report = device_report(device)

    config = resolve_config(load_config(args.config))
    ck = torch.load(args.checkpoint.expanduser().resolve(), map_location="cpu", weights_only=False)
    ck_config = ck.get("config") or config
    masses = daughter_masses_from_config(ck_config)
    output_dir = args.output_dir or (args.checkpoint.parent / "plots_paperstyle_while_training")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model condition statistics must come from the training selection used by
    # the checkpoint.  The CLI config may request a wider plotting selection
    # (e.g. sidebands), so load both sets explicitly.
    train_arrays, _train_cache = load_and_split_cached(
        ck_config, num_samples=args.num_samples, cache_dir=None, use_cache=True
    )
    arrays, cache_info = load_and_split_cached(
        config, num_samples=args.num_samples, cache_dir=None, use_cache=True
    )
    x_split = pplot.finite_8d(pplot.select_split(arrays, "x", args.split), "x")
    z_split = pplot.finite_8d(pplot.select_split(arrays, "z", args.split), "z")
    seed = int(ck_config.get("seed", 0))
    np.random.seed(seed)
    x_plot = pplot.random_subset(x_split, args.max_x_events, seed)
    z_plot = pplot.random_subset(z_split, args.max_z_events, seed + 1)
    z_for_x = pplot.random_subset(z_split, len(x_plot), seed + 2)

    model = build_model(ck_config, train_arrays).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    batch_size = max(1, int(args.batch_size or ck_config.get("loaders", {}).get("eval_batch_size", 8192)))

    z_encoded = predict_batches(model, "encode", x_plot, device, batch_size)
    x_reco = predict_batches(model, "reconstruct", x_plot, device, batch_size)
    x_from_z = predict_batches(model, "decode", z_for_x, device, batch_size)

    evaluation = config.get("evaluation", {})
    mass_low = float(evaluation.get("mass_range", [2.9, 3.3])[0] if args.mass_low is None else args.mass_low)
    mass_high = float(evaluation.get("mass_range", [2.9, 3.3])[1] if args.mass_high is None else args.mass_high)
    mass_bin_width = float(evaluation.get("mass_bin_width", 0.005) if args.mass_bin_width is None else args.mass_bin_width)
    min_truth_count = int(evaluation.get("min_truth_count", 20))
    mass_label = str(evaluation.get("mass_label", "m(mumu) [GeV]"))
    channel_title = str(evaluation.get("channel_title", "CMS DoubleMuParked: J/psi -> mumu"))
    reference_mass = evaluation.get("reference_mass")
    reference_mass = None if reference_mass is None else float(reference_mass)
    reference_label = str(evaluation.get("reference_label", "m_J/psi"))
    component_titles = [
        r"$\mu^-\ p_x$", r"$\mu^-\ p_y$", r"$\mu^-\ p_z$", r"$\mu^-\ E$",
        r"$\mu^+\ p_x$", r"$\mu^+\ p_y$", r"$\mu^+\ p_z$", r"$\mu^+\ E$",
    ]

    mass_bins = np.arange(mass_low, mass_high + mass_bin_width, mass_bin_width)
    pt_bins = np.linspace(0.0, 100.0, 101)
    bins_y = np.array([-100, -60] + [-50 + 5*i for i in range(21)] + [60, 100], dtype=float)
    bins_z = np.array([-400] + [-250 + 20*i for i in range(26)] + [400], dtype=float)
    bins_e = np.array([0] + [20 + 10*i for i in range(26)] + [400], dtype=float)

    m_x = pplot.inv_mass_ee(x_plot, daughter_masses=masses)
    m_x_reco = pplot.inv_mass_ee(x_reco, daughter_masses=masses)
    m_x_from_z = pplot.inv_mass_ee(x_from_z, daughter_masses=masses)
    m_z_prior = pplot.inv_mass_ee(z_plot, daughter_masses=masses, stable=False)
    m_x_to_z = pplot.inv_mass_ee(z_encoded, daughter_masses=masses, stable=False)
    pt_x = pplot.pt_ee(x_plot)
    pt_x_reco = pplot.pt_ee(x_reco)
    pt_x_from_z = pplot.pt_ee(x_from_z)

    xmass_info = pplot.paper_ratio_plot_double(
        truth=m_x, pred1=m_x_reco, pred2=m_x_from_z, bins=mass_bins,
        xlabel=mass_label, title=f"{channel_title}: x-space mass",
        path=output_dir/"paperstyle_xspace_mass_density_ratio.png", density=density,
        xlim=(mass_low, mass_high), residual_ylim=(-0.05, 0.05),
        reference_x=reference_mass,
        reference_label=None if reference_mass is None else f"{reference_label}={reference_mass:.4f} GeV",
    )
    pplot.paper_ratio_plot_double(
        truth=pt_x, pred1=pt_x_reco, pred2=pt_x_from_z, bins=pt_bins,
        xlabel=r"$p_T(\mu\mu)$ [GeV]", title=f"{channel_title}: dilepton transverse momentum",
        path=output_dir/"paperstyle_xspace_pt_density_ratio.png", density=density,
        xlim=(0.0, 100.0), ratio_ylim=(0.0, 2.0),
    )
    for idx, bins, xlabel, xlim, fname in [
        (5, bins_y, r"$\mu^+\ p_y$ [GeV]", (-100.0,100.0), "paperstyle_zspace_pos_py_ratio.png"),
        (6, bins_z, r"$\mu^+\ p_z$ [GeV]", (-400.0,400.0), "paperstyle_zspace_pos_pz_ratio.png"),
        (7, bins_e, r"$\mu^+\ E$ [GeV]", (0.0,400.0), "paperstyle_zspace_pos_E_ratio.png"),
    ]:
        pplot.paper_ratio_plot_single(
            truth=z_plot[:,idx], pred=z_encoded[:,idx], bins=bins, xlabel=xlabel,
            title=f"z-space closure: {xlabel}", path=output_dir/fname, density=density,
            xlim=xlim, ratio_ylim=(0.5,1.5),
        )
    for idx, bins, xlabel, xlim, fname in [
        (5, bins_y, r"$\mu^+\ p_y$ [GeV]", (-100.0,100.0), "paperstyle_xspace_pos_py_ratio.png"),
        (6, bins_z, r"$\mu^+\ p_z$ [GeV]", (-400.0,400.0), "paperstyle_xspace_pos_pz_ratio.png"),
        (7, bins_e, r"$\mu^+\ E$ [GeV]", (0.0,400.0), "paperstyle_xspace_pos_E_ratio.png"),
    ]:
        pplot.paper_ratio_plot_double(
            truth=x_plot[:,idx], pred1=x_reco[:,idx], pred2=x_from_z[:,idx], bins=bins,
            xlabel=xlabel, title=f"x-space closure: {xlabel}", path=output_dir/fname,
            density=density, xlim=xlim, ratio_ylim=(0.5,1.5),
        )
    pplot.paper_ratio_plot_single(
        truth=m_z_prior, pred=m_x_to_z, bins=mass_bins, xlabel=mass_label,
        title=r"z-space mass check: MG5 z vs OTUS $x \rightarrow \tilde{z}$",
        path=output_dir/"paperstyle_zspace_mass_density_ratio.png", density=density,
        xlim=(mass_low, mass_high), ratio_ylim=(0.0,2.0),
        reference_x=reference_mass,
        reference_label=None if reference_mass is None else f"{reference_label}={reference_mass:.4f} GeV",
    )
    pplot.plot_all_components(
        [(z_plot, "MG5 z", pplot.TRUTH_STYLE), (z_encoded, "CMS x -> z", pplot.ENC_STYLE)],
        "z-space component check: MG5 z vs OTUS x -> z",
        output_dir/"all8_zspace_components_density.png", 80, density, component_titles,
    )
    pplot.plot_all_components(
        [(x_plot, "CMS x", pplot.TRUTH_STYLE), (x_reco, "x -> z -> x", pplot.ENCDEC_STYLE), (x_from_z, "z -> x", pplot.DEC_STYLE)],
        "x-space component closure: CMS x vs OTUS outputs",
        output_dir/"all8_xspace_components_density.png", 80, density, component_titles,
    )

    zdir = output_dir/"zspace_validation"; zdir.mkdir(parents=True, exist_ok=True)
    pplot.paper_ratio_plot_single(
        truth=m_z_prior, pred=m_x_to_z, bins=mass_bins, xlabel=mass_label,
        title="z-space mass check: MG5 z vs OTUS x -> z",
        path=zdir/"zspace_mass_mg5_vs_x2z.png", density=density, xlim=(mass_low,mass_high), ratio_ylim=(0.0,2.0),
    )
    pplot.plot_all_components(
        [(z_plot, "MG5 z", pplot.TRUTH_STYLE), (z_encoded, "CMS x -> z", pplot.ENC_STYLE)],
        "z-space component check: MG5 z vs OTUS x -> z",
        zdir/"zspace_components_mg5_vs_x2z.png", 80, density, component_titles,
    )

    observable_summaries = {
        "m_mumu": pplot.histogram_observable_summary(m_x, m_x_from_z, mass_bins, density, min_truth_count),
        "pT_mumu": pplot.histogram_observable_summary(pt_x, pt_x_from_z, pt_bins, density, min_truth_count),
        "mu_minus_E": pplot.histogram_observable_summary(x_plot[:,3], x_from_z[:,3], bins_e, density, min_truth_count),
        "mu_plus_E": pplot.histogram_observable_summary(x_plot[:,7], x_from_z[:,7], bins_e, density, min_truth_count),
        "mu_minus_py": pplot.histogram_observable_summary(x_plot[:,1], x_from_z[:,1], bins_y, density, min_truth_count),
        "mu_plus_py": pplot.histogram_observable_summary(x_plot[:,5], x_from_z[:,5], bins_y, density, min_truth_count),
        "mu_minus_pz": pplot.histogram_observable_summary(x_plot[:,2], x_from_z[:,2], bins_z, density, min_truth_count),
        "mu_plus_pz": pplot.histogram_observable_summary(x_plot[:,6], x_from_z[:,6], bins_z, density, min_truth_count),
        "zspace_mass": pplot.histogram_observable_summary(m_z_prior, m_x_to_z, mass_bins, density, min_truth_count),
    }
    z_component_summaries = {}
    for j in range(8):
        values = np.concatenate([z_plot[:,j], z_encoded[:,j]])
        lo, hi = np.nanpercentile(values, [0.5,99.5])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = np.nanmin(values), np.nanmax(values)
        zbins = np.linspace(lo, hi, 81)
        z_component_summaries[f"z_component_{j:02d}"] = pplot.histogram_observable_summary(z_plot[:,j], z_encoded[:,j], zbins, density, min_truth_count)

    ks_mass, p_mass = pplot.maybe_ks(m_z_prior, m_x_to_z)
    ks_pt_reco, p_pt_reco = pplot.maybe_ks(pt_x, pt_x_reco)
    ks_pt_zx, p_pt_zx = pplot.maybe_ks(pt_x, pt_x_from_z)
    residual_reco = pplot.residual_quality_summary(xmass_info["truth"], xmass_info["pred1"], xmass_info["truth_counts"], xmass_info["pred1_counts"], m_x, m_x_reco, min_truth_count)
    residual_zx = pplot.residual_quality_summary(xmass_info["truth"], xmass_info["pred2"], xmass_info["truth_counts"], xmass_info["pred2_counts"], m_x, m_x_from_z, min_truth_count)

    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": ck.get("epoch"),
        "stage": (ck.get("stage") or {}).get("name"),
        "device_report": report,
        "split": args.split,
        "data_cache": cache_info,
        "density": density,
        "has_scipy": pplot.HAS_SCIPY,
        "shapes": {k: list(v.shape) for k,v in [("x_plot",x_plot),("z_plot",z_plot),("z_encoded",z_encoded),("x_reco",x_reco),("x_from_z",x_from_z)]},
        "mass": {
            "z_mg5_mean": float(m_z_prior.mean()), "z_mg5_std": float(m_z_prior.std()),
            "x_to_z_mean": float(m_x_to_z.mean()), "x_to_z_std": float(m_x_to_z.std()),
            "x_mean": float(m_x.mean()), "x_std": float(m_x.std()),
            "x_reco_mean": float(m_x_reco.mean()), "x_reco_std": float(m_x_reco.std()),
            "x_from_z_mean": float(m_x_from_z.mean()), "x_from_z_std": float(m_x_from_z.std()),
            "zspace_ks_statistic": ks_mass, "zspace_ks_pvalue": p_mass,
        },
        "pt": {"x_vs_reco_ks_statistic": ks_pt_reco, "x_vs_reco_ks_pvalue": p_pt_reco,
               "x_vs_zx_ks_statistic": ks_pt_zx, "x_vs_zx_ks_pvalue": p_pt_zx},
        "xspace_mass_residual": {"x_to_z_to_x": residual_reco, "z_to_x": residual_zx},
        "observable_residuals": observable_summaries,
        "zspace_component_residuals": z_component_summaries,
    }
    (output_dir/"paperstyle_summary.json").write_text(json.dumps(pplot.json_safe(summary), indent=2, sort_keys=True)+"\n", encoding="utf-8")
    np.savez_compressed(
        output_dir/"paperstyle_loaded_model_outputs.npz",
        x_plot=x_plot, z_plot=z_plot, z_encoded=z_encoded, x_reco=x_reco, x_from_z=x_from_z,
        m_z_prior=m_z_prior, m_x_to_z=m_x_to_z, m_x=m_x, m_x_reco=m_x_reco, m_x_from_z=m_x_from_z,
        pt_x=pt_x, pt_x_reco=pt_x_reco, pt_x_from_z=pt_x_from_z,
    )
    save_resolved_config(ck_config, output_dir/"config.resolved.json")
    print(json.dumps(summary, indent=2, sort_keys=True)[:4000])
    print("Wrote paper-style plots to", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
