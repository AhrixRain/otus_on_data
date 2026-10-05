#!/usr/bin/env python
"""S2c + Upsilon quadrature target (read-only).

Two questions from docs/calibration_target_2026-10-04.md section 6:

A. S2c - fit the J/psi dimuon mass peak on the LOCKED split with the shape CMS
   actually uses (Crystal Ball core + exponential background), for both the CMS
   data (x_test) and the honest narrow prior (z_test). This separates "the tails
   are detector" from "the tails are truth" and decides empirically which of the
   P4 readings is the response requirement:

       E (adopted): J/psi 23.81 MeV  (std quadrature)
       A:           J/psi 28.06 MeV  (data std; the shipped kernel target is 28.1)
       F:           J/psi 30.02 MeV  (robust quadrature)

B. Upsilon quadrature target - the region prior's std (110.29 MeV) exceeds the
   fitted peak sigma (84.42 MeV), so no std-based target exists. Restricted to
   the signal component's CORE, a target becomes definable at all; this script
   measures the core and reports the resulting requirement.

Nothing is trained; nothing under data/, outputs/ or any checkpoint is modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

REPO_ROOT = Path(__file__).resolve().parents[1]
for _directory in (
    REPO_ROOT / "scripts",
    REPO_ROOT / "scripts_sota",
    REPO_ROOT / "scripts_joint",
    REPO_ROOT / "scripts_joint" / "upsilon",
):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from fixed_z_noise_budget import (  # noqa: E402
    load_reference_cluster_arrays,
    mass8,
)

DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "outputs"
    / "cms_Joint"
    / "Run_H_cycleNoNoise"
    / "best_RunHcycleNoNoise_stage2_stochastic_core.pt"
)
CONTINUUM_PRIOR = REPO_ROOT / "data" / "upsilon_prior_continuumReweighted.hdf5"


def crystal_ball(x, mu, sigma, alpha, n):
    z = (x - mu) / sigma
    out = np.empty_like(z)
    mask = z > -alpha
    out[mask] = np.exp(-0.5 * z[mask] ** 2)
    a = abs(alpha)
    tail = (n / a) ** n * np.exp(-0.5 * a * a) * (n / a - a - z[~mask]) ** (-n)
    out[~mask] = tail
    return out


def model(x, mu, sigma, alpha, n, amp, b0, b1):
    return amp * crystal_ball(x, mu, sigma, alpha, n) + np.exp(b0 + b1 * x)


def fit_peak(masses, *, window, bins, sigma0, label, log=print):
    lo, hi = window
    selected = masses[(masses > lo) & (masses < hi)]
    counts, edges = np.histogram(selected, bins=bins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    errors = np.sqrt(np.maximum(counts, 1.0))
    amplitude0 = float(np.max(counts)) * sigma0["sigma"] * np.sqrt(2 * np.pi) / width
    side = (centers < sigma0["mu"] - 5 * sigma0["sigma"]) | (
        centers > sigma0["mu"] + 5 * sigma0["sigma"]
    )
    side_level = float(np.median(counts[side])) if bool(side.any()) else 1.0
    start = np.array(
        [
            sigma0["mu"],
            sigma0["sigma"],
            1.5,
            5.0,
            max(amplitude0, 1.0),
            np.log(max(side_level, 1.0)),
            0.0,
        ]
    )
    lower = np.array([sigma0["mu"] - 0.05, sigma0["sigma"] * 0.3, 0.3, 1.5, 0.0, -25.0, -200.0])
    upper = np.array(
        [sigma0["mu"] + 0.05, sigma0["sigma"] * 3.0, 6.0, 40.0, np.inf, 25.0, 200.0]
    )
    start = np.clip(start, lower + 1e-9, upper - 1e-9)

    def residual(params):
        return (model(centers, *params) - counts) / errors

    result = least_squares(residual, start, bounds=(lower, upper), max_nfev=20000)
    mu, sigma, alpha, n, amp, b0, b1 = result.x
    dof = max(len(counts) - len(result.x), 1)
    chi2 = float(np.sum(result.fun**2))
    jac = result.jac
    try:
        covariance = np.linalg.pinv(jac.T @ jac) * (chi2 / dof)
        sigma_err = float(np.sqrt(max(covariance[1, 1], 0.0)))
    except np.linalg.LinAlgError:
        sigma_err = float("nan")
    log(
        f"{label}: n={len(selected)} mu={mu:.5f} sigma={sigma*1000:.2f} MeV "
        f"(+/- {sigma_err*1000:.2f}) alpha={alpha:.3f} n={n:.2f} chi2/dof={chi2/dof:.3f}"
    )
    return {
        "label": label,
        "n_events": int(len(selected)),
        "window_gev": list(window),
        "bins": bins,
        "mu_gev": float(mu),
        "sigma_gev": float(sigma),
        "sigma_err_gev": sigma_err,
        "alpha": float(alpha),
        "n_tail": float(n),
        "chi2_per_dof": float(chi2 / dof),
        "sigma_mev": float(sigma * 1000.0),
        "sigma_err_mev": float(sigma_err * 1000.0),
        "efficiency_retained": float(len(selected) / max(len(masses), 1)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jpsi-window", default="2.9,3.3")
    parser.add_argument("--jpsi-bins", type=int, default=80)
    parser.add_argument("--upsilon-signature", type=float, default=0.08442)
    parser.add_argument("--upsilon-alternate", type=float, default=0.08824)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = load_reference_cluster_arrays(args.checkpoint)
    masses_x = mass8(arrays["jpsi"]["x_test"])
    masses_z = mass8(arrays["jpsi"]["z_test"])
    window = tuple(float(v) for v in args.jpsi_window.split(","))

    data_fit = fit_peak(
        masses_x,
        window=window,
        bins=args.jpsi_bins,
        sigma0={"mu": 3.0969, "sigma": 0.031},
        label="cms_x_test (detector level)",
    )
    prior_fit = fit_peak(
        masses_z,
        window=window,
        bins=args.jpsi_bins,
        sigma0={"mu": 3.0969, "sigma": 0.003},
        label="prior_z_test (truth level, honest narrow prior)",
    )

    sd, sp = data_fit["sigma_gev"], prior_fit["sigma_gev"]
    empirical_response = float(np.sqrt(max(sd * sd - sp * sp, 0.0)))
    readings = {
        "E_std_quadrature": 0.02381,
        "A_data_std": 0.02806,
        "F_robust_quadrature": 0.03002,
        "G_primary_literature": 0.031,
    }
    comparison = {
        key: {
            "reading_gev": value,
            "empirical_response_over_reading": empirical_response / value,
        }
        for key, value in readings.items()
    }
    closest = min(comparison, key=lambda k: abs(np.log(comparison[k]["empirical_response_over_reading"])))

    # ---- B. Upsilon signal-component quadrature target -------------------
    import h5py

    upsilon = {}
    with h5py.File(CONTINUUM_PRIOR, "r") as source:
        z = np.asarray(source["FDL/zData"][:], dtype=np.float64)
        component = np.asarray(source["FDL/component_id"][:])
    for index, state in enumerate(("upsilon1s", "upsilon2s", "upsilon3s")):
        selected = z[component == index]
        masses = mass8(selected)
        robust = float((np.quantile(masses, 0.84) - np.quantile(masses, 0.16)) / 2.0)
        std = float(np.std(masses))
        core = masses[np.abs(masses - np.median(masses)) < 3.0 * robust]
        core_std = float(np.std(core))
        upsilon[state] = {
            "n_prior_events": int(len(selected)),
            "prior_mean_gev": float(np.mean(masses)),
            "prior_std_gev": std,
            "prior_robust_gev": robust,
            "prior_core_std_3robust_gev": core_std,
            "prior_core_n": int(len(core)),
            "quadrature_target_from_core_std_gev": float(
                np.sqrt(max(args.upsilon_signature**2 - core_std**2, 0.0))
            ),
            "quadrature_undefined_from_full_std": bool(std >= args.upsilon_signature),
            "fitted_sigma_reference_gev": args.upsilon_signature,
            "fitted_sigma_alternate_gev": args.upsilon_alternate,
        }

    payload = {
        "schema_version": 1,
        "diagnostic": "S2c J/psi peak fit + Upsilon signal-component target",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "jpsi": {
            "data_fit": data_fit,
            "prior_fit": prior_fit,
            "empirical_response_sigma_gev": empirical_response,
            "empirical_response_sigma_mev": empirical_response * 1000.0,
            "readings": readings,
            "comparison": comparison,
            "closest_reading": closest,
            "interpretation": (
                "sigma_x^2 = sigma_z^2 + sigma_response^2; the fitted CB core width "
                "of the CMS data minus the fitted truth prior width is the response "
                "requirement measured on the locked split, with the tails "
                "attributed by the same function on both sides."
            ),
        },
        "upsilon": upsilon,
    }
    (output_dir / "s2c_jpsi_fit.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# S2c - fitted J/psi peak sigma, and the Upsilon signal-component target",
        "",
        f"*{payload['created_utc']} - read-only fit on the locked splits, no training. "
        f"Checkpoint `{Path(args.checkpoint).name}` used only to resolve the locked "
        f"region cache.*",
        "",
        "## A. Crystal Ball + exponential fit, window "
        f"{window[0]}-{window[1]} GeV, {args.jpsi_bins} bins",
        "",
        "| sample | n | mu [GeV] | sigma [MeV] | alpha | n_tail | chi2/dof |",
        "|---|---|---|---|---|---|---|",
        f"| CMS x_test | {data_fit['n_events']} | {data_fit['mu_gev']:.5f} | "
        f"**{data_fit['sigma_mev']:.2f} +/- {data_fit['sigma_err_mev']:.2f}** | "
        f"{data_fit['alpha']:.3f} | {data_fit['n_tail']:.2f} | {data_fit['chi2_per_dof']:.3f} |",
        f"| prior z_test | {prior_fit['n_events']} | {prior_fit['mu_gev']:.5f} | "
        f"**{prior_fit['sigma_mev']:.2f} +/- {prior_fit['sigma_err_mev']:.2f}** | "
        f"{prior_fit['alpha']:.3f} | {prior_fit['n_tail']:.2f} | {prior_fit['chi2_per_dof']:.3f} |",
        "",
        f"Empirical response sigma = sqrt(sigma_x^2 - sigma_z^2) = "
        f"**{empirical_response*1000:.2f} MeV**.",
        "",
        "| reading | value [MeV] | empirical / reading |",
        "|---|---|---|",
    ]
    for key, entry in comparison.items():
        lines.append(
            f"| {key} | {entry['reading_gev']*1000:.2f} | "
            f"{entry['empirical_response_over_reading']:.3f} |"
        )
    lines += ["", f"Closest reading: **{closest}**.", "", "## B. Upsilon signal-component target", ""]
    lines += [
        "| state | prior std [MeV] | prior robust [MeV] | core std (3 robust) [MeV] | "
        "quadrature target from core [MeV] | std-based target defined? |",
        "|---|---|---|---|---|---|",
    ]
    for state, entry in upsilon.items():
        lines.append(
            f"| {state} | {entry['prior_std_gev']*1000:.2f} | "
            f"{entry['prior_robust_gev']*1000:.2f} | "
            f"{entry['prior_core_std_3robust_gev']*1000:.2f} | "
            f"**{entry['quadrature_target_from_core_std_gev']*1000:.2f}** | "
            f"{'NO' if entry['quadrature_undefined_from_full_std'] else 'yes'} |"
        )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"empirical response {empirical_response*1000:.2f} MeV; closest reading {closest}; "
        f"wrote {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
