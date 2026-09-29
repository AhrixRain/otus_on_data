#!/usr/bin/env python
"""Compare fake J/psi-scaled-to-Upsilon decoding with true Upsilon decoding.

For a given frozen checkpoint, this loads the decoded HDF5 files:

  FDL/zData -> the original (fake or true) prior event
  FDL/xData -> the decoder output for that event

and computes per-event mass shifts and robust width ratios. The mass
convention is the repository's stable massive-muon invariant mass, matching
the quantitative Upsilon transfer reports.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys_path = str(REPO_ROOT)
import sys

if sys_path not in sys.path:
    sys.path.insert(0, sys_path)
scripts_path = str(REPO_ROOT / "scripts")
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from scripts.physics import invariant_mass_np  # noqa: E402

MUON_MASS = 0.105658
MASS_MIN = 8.5
MASS_MAX = 11.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fake-decoded", type=Path, required=True)
    parser.add_argument("--true-decoded", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-plot", type=Path, default=None)
    parser.add_argument("--output-prior-plot", type=Path, default=None)
    parser.add_argument(
        "--cms-data",
        type=Path,
        default=REPO_ROOT / "experiments" / "cms_upsilon" / "data" / "Ymumu.csv",
    )
    return parser.parse_args()


def load_pairs(path: Path):
    with h5py.File(path, "r") as handle:
        for key in ("FDL/zData", "FDL/xData", "FDL/component_id"):
            if key not in handle:
                raise KeyError(f"{path} is missing {key}")
        z = np.asarray(handle["FDL/zData"], dtype=np.float64)
        x = np.asarray(handle["FDL/xData"], dtype=np.float64)
        component = np.asarray(handle["FDL/component_id"])
    if z.shape != x.shape:
        raise ValueError(f"z/x shape mismatch in {path}")
    return z, x, component


def robust_width_mev(values: np.ndarray) -> float:
    q75, q25 = np.percentile(values, [75, 25])
    return float((q75 - q25) / 1.349 * 1000.0)


def mass(values: np.ndarray) -> np.ndarray:
    return invariant_mass_np(values, daughter_masses=(MUON_MASS, MUON_MASS), stable=True)


def pair_pt(values: np.ndarray) -> np.ndarray:
    return np.hypot(values[:, 0] + values[:, 4], values[:, 1] + values[:, 5])


def load_cms_masses_csv(path: Path) -> np.ndarray:
    """Load the CMS Y/upsilon outreach CSV and compute stable dimuon masses."""
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    data = np.atleast_1d(data)
    names = {name.lower(): name for name in data.dtype.names}
    p4 = np.column_stack(
        [
            data[names["px1"]], data[names["py1"]],
            data[names["pz1"]], data[names["e1"]],
            data[names["px2"]], data[names["py2"]],
            data[names["pz2"]], data[names["e2"]],
        ]
    ).astype(np.float64, copy=False)
    m = mass(p4)
    return m[(m >= MASS_MIN) & (m <= MASS_MAX)]


def summarize(name: str, z: np.ndarray, x: np.ndarray):
    mz = mass(z)
    mx = mass(x)
    shift = mx - mz
    window = (mz >= MASS_MIN) & (mz <= MASS_MAX)
    # The shift is meaningful only where both input and output are in/near the
    # window; we report the full labelled-component shift as in the Upsilon
    # reports, plus retention.
    retained = (mx >= MASS_MIN) & (mx <= MASS_MAX)
    summary = {
        "name": name,
        "events": int(len(z)),
        "input_mass_median_gev": float(np.median(mz)),
        "input_mass_mean_gev": float(np.mean(mz)),
        "output_mass_median_gev": float(np.median(mx)),
        "output_mass_mean_gev": float(np.mean(mx)),
        "median_mass_shift_mev": float(np.median(shift)) * 1000.0,
        "mean_mass_shift_mev": float(np.mean(shift)) * 1000.0,
        "median_mass_shift_percent": float(np.median(shift) / np.median(mz) * 100.0),
        "mean_mass_shift_percent": float(np.mean(shift) / np.mean(mz) * 100.0),
        "input_robust_width_mev": robust_width_mev(mz),
        "output_robust_width_mev": robust_width_mev(mx),
        "input_std_mev": float(np.std(mz) * 1000.0),
        "output_std_mev": float(np.std(mx) * 1000.0),
        "output_window_retention": float(np.mean(retained)),
        "input_pair_pt_median_gev": float(np.median(pair_pt(z))),
        "output_pair_pt_median_gev": float(np.median(pair_pt(x))),
    }
    return summary, mz, mx, shift


def main() -> int:
    args = parse_args()
    fake_path = args.fake_decoded.expanduser().resolve()
    true_path = args.true_decoded.expanduser().resolve()
    if not fake_path.exists():
        raise FileNotFoundError(fake_path)
    if not true_path.exists():
        raise FileNotFoundError(true_path)

    zf, xf, cf = load_pairs(fake_path)
    zt, xt, ct = load_pairs(true_path)
    if not np.all(cf == 0):
        print(f"Warning: fake decoded component ids are not all zero ({np.unique(cf)})")
    if not np.all(ct == 0):
        print(f"Warning: true decoded component ids are not all zero ({np.unique(ct)})")

    fake_summary, mzf, mxf, shift_f = summarize("fake_jpsi_scaled_to_upsilon1s", zf, xf)
    true_summary, mzt, mxt, shift_t = summarize("true_upsilon1s", zt, xt)

    rows = [fake_summary, true_summary]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps({"checkpoint": args.checkpoint, "rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nCheckpoint: {args.checkpoint}")
    for row in rows:
        print(f"\n{row['name']}")
        for key, value in row.items():
            if key == "name":
                continue
            if isinstance(value, float):
                print(f"  {key:32s} {value:.6g}")
            else:
                print(f"  {key:32s} {value}")

    if args.output_plot is not None:
        import matplotlib
        matplotlib.use("Agg")
        import plot as pplot

        cms_masses = load_cms_masses_csv(args.cms_data.expanduser().resolve())
        bins = np.linspace(MASS_MIN, MASS_MAX, 161)
        pplot.paper_ratio_plot_double(
            truth=cms_masses,
            pred1=mxf,
            pred2=mxt,
            bins=bins,
            xlabel=r"$m_{\mu\mu}$ [GeV]",
            title=f"Run E {args.checkpoint} decoder: fake decoded vs true decoded vs CMS",
            path=args.output_plot,
            density=True,
            truth_label="CMS Y data",
            pred1_label="Fake J/psi-scaled decoded",
            pred2_label="True Upsilon decoded",
            pred1_style=dict(color="deepskyblue", linewidth=1.8, linestyle="--"),
            pred2_style=dict(color="darkviolet", linewidth=1.8, linestyle="-."),
            xlim=(MASS_MIN, MASS_MAX),
            ratio_ylim=(0.0, 3.0),
            residual_ylim=(-1.0, 2.0),
            reference_x=9.4603,
            reference_label=r"$m_{\Upsilon(1S)}=9.4603$ GeV",
        )
        print("Saved:", args.output_plot)

    if args.output_prior_plot is not None:
        import matplotlib
        matplotlib.use("Agg")
        import plot as pplot

        args.output_prior_plot.parent.mkdir(parents=True, exist_ok=True)
        pplot.paper_ratio_plot_single(
            truth=mzt,
            pred=mzf,
            bins=np.linspace(MASS_MIN, MASS_MAX, 161),
            xlabel=r"$m_{\mu\mu}$ [GeV]",
            title="Fake J/psi scaled prior vs true Upsilon(1S) prior",
            path=args.output_prior_plot,
            density=True,
            truth_label="True Upsilon(1S) prior",
            pred_label="Fake J/psi scaled prior",
            pred_style=dict(color="deepskyblue", linewidth=1.8, linestyle="--"),
            xlim=(MASS_MIN, MASS_MAX),
            ratio_ylim=(0.0, 2.0),
            reference_x=9.4603,
            reference_label=r"$m_{\Upsilon(1S)}=9.4603$ GeV",
        )
        print("Saved:", args.output_prior_plot)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
