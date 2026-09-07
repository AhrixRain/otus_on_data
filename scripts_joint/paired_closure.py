#!/usr/bin/env python
"""Per-event closure against withheld truth pairs.

What this answers
-----------------
Every metric the joint pipeline currently reports is a MARGINAL: it compares
two distributions. The 2026-09-04 A/B showed marginals can be satisfied by a
map that does nothing (memory.md section 7.2). A marginal test cannot
distinguish

    "the encoder inverted the detector response event by event"

from

    "the encoder produced a set of four-vectors whose histogram happens to
     match the prior".

Only paired truth separates those, and paired truth exists on disk: the
upstream OTUS datasets (``data/ppzee*.hdf5``, ``data/ppttbar*.hdf5``, and the
saved arrays in ``experiments/ppzee/otus_results-dataset=ppzee_test.npz``)
carry event-by-event (z, x) partners produced by MadGraph + Pythia + Delphes.
Training stays unpaired exactly as it is now; the pairing is used only at
scoring time.

The one number that matters
---------------------------
    residual_rms_vs_identity = rms(mass(pred) - mass(z_true))
                             / rms(mass(x_input) - mass(z_true))

    < 1   the map moved events toward their own truth partners
    = 1   no better per event than handing back the detector-level event
    > 1   the map moved events away from their truth partners

The denominator is the detector resolution itself, so this is a direct,
units-free statement about whether resolution was inverted. Marginal KS and W1
are reported next to it so the two can be read together: a model can be
excellent on the first and useless on the second, and that gap is the finding.

Modes
-----
Upstream reference (NumPy only, no model, no GPU) -- score the published OTUS
result on its own paired benchmark:

    python scripts_joint/paired_closure.py \\
        --results-npz experiments/ppzee/otus_results-dataset=ppzee_test.npz

Generic -- score any saved prediction against its truth partners:

    python scripts_joint/paired_closure.py --pairs-npz PAIRS --pred-npz PRED

``PAIRS`` must contain ``z`` and ``x`` of shape [N, 8]; ``PRED`` must contain
``z_pred`` of the same shape and, optionally, ``z_pred_draws`` of shape
[D, N, 8] for the posterior diagnostics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts_joint") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts_joint"))

from identity_reference import ks_distance, wasserstein_1d  # noqa: E402


def direct_mass(values: np.ndarray) -> np.ndarray:
    """E^2 - |p|^2 on the stored four-vectors, in float64.

    The upstream ppzee/ppttbar arrays store their own energies and are not in
    the CMS pipeline's stable-mass convention, so the stored energy is used as
    given rather than recomputed from a daughter mass.
    """
    values = np.asarray(values, dtype=np.float64)
    pair = values[:, 0:4] + values[:, 4:8]
    return np.sqrt(
        np.clip(pair[:, 3] ** 2 - np.sum(pair[:, 0:3] ** 2, axis=1), 0.0, None)
    )


def _residual_stats(prefix: str, residual: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(np.mean(residual)),
        f"{prefix}_rms": float(np.sqrt(np.mean(residual**2))),
        f"{prefix}_std": float(np.std(residual)),
        f"{prefix}_mad": float(np.median(np.abs(residual - np.median(residual)))),
    }


def paired_closure(
    z_true: np.ndarray,
    x_input: np.ndarray | None,
    z_pred: np.ndarray,
    *,
    z_pred_draws: np.ndarray | None = None,
    mass_fn=direct_mass,
) -> dict[str, Any]:
    """Per-event and marginal closure for one encoder direction."""
    if len(z_true) != len(z_pred):
        raise ValueError("z_true and z_pred must be event-aligned")
    # ``x_input`` is the map's own input, used as the do-nothing reference.
    # Pass None when there is no meaningful identity -- scoring a cycle
    # D(E(x)) against x, for instance, where the identity residual is
    # identically zero and every ratio against it is degenerate.
    if x_input is not None and len(x_input) != len(z_true):
        raise ValueError("x_input must be event-aligned with z_true")

    mass_true = mass_fn(z_true)
    mass_pred = mass_fn(z_pred)
    model_residual = mass_pred - mass_true

    report: dict[str, Any] = {
        "events": int(len(z_true)),
        "has_identity_reference": x_input is not None,
        "per_event_mass": {},
        "per_event_components": {},
        "marginal_mass": {},
    }
    report["per_event_mass"].update(_residual_stats("model", model_residual))
    report["per_event_mass"]["truth_correlation"] = float(
        np.corrcoef(mass_pred, mass_true)[0, 1]
    )
    model_component = np.sqrt(np.mean((z_pred - z_true) ** 2, axis=0))
    report["per_event_components"]["model_rms_per_column"] = model_component.tolist()
    report["marginal_mass"] = {
        "model_ks": ks_distance(mass_true, mass_pred),
        "model_w1_gev": wasserstein_1d(mass_true, mass_pred),
        "model_std_gev": float(np.std(mass_pred)),
        "truth_std_gev": float(np.std(mass_true)),
    }

    identity_rms = 0.0
    if x_input is not None:
        mass_input = mass_fn(x_input)
        identity_residual = mass_input - mass_true
        report["per_event_mass"].update(
            _residual_stats("identity", identity_residual)
        )
        identity_rms = report["per_event_mass"]["identity_rms"]
        model_rms = report["per_event_mass"]["model_rms"]
        report["per_event_mass"]["residual_rms_vs_identity"] = (
            model_rms / identity_rms if identity_rms > 0 else float("inf")
        )
        report["per_event_mass"]["identity_truth_correlation"] = float(
            np.corrcoef(mass_input, mass_true)[0, 1]
        )
        # Four-vector columns, so a mass-only success cannot hide a broken map.
        identity_component = np.sqrt(np.mean((x_input - z_true) ** 2, axis=0))
        ratio = model_component / np.maximum(identity_component, 1.0e-12)
        report["per_event_components"].update(
            {
                "identity_rms_per_column": identity_component.tolist(),
                "rms_vs_identity_per_column": ratio.tolist(),
                "rms_vs_identity_mean": float(np.mean(ratio)),
            }
        )
        report["marginal_mass"].update(
            {
                "identity_ks": ks_distance(mass_true, mass_input),
                "identity_w1_gev": wasserstein_1d(mass_true, mass_input),
                "identity_std_gev": float(np.std(mass_input)),
            }
        )

    if z_pred_draws is not None:
        draws = np.asarray(z_pred_draws, dtype=np.float64)
        if draws.ndim != 3 or draws.shape[1:] != z_true.shape:
            raise ValueError("z_pred_draws must have shape [D, N, 8]")
        draw_mass = np.stack([mass_fn(draw) for draw in draws])
        posterior_mean = draw_mass.mean(axis=0)
        posterior_std = draw_mass.std(axis=0, ddof=1)
        mean_residual = posterior_mean - mass_true
        pull = mean_residual / np.maximum(posterior_std, 1.0e-12)
        inside = np.abs(mass_true - posterior_mean) <= posterior_std
        report["posterior"] = {
            "draws": int(draws.shape[0]),
            "mean_residual_rms": float(np.sqrt(np.mean(mean_residual**2))),
            "mean_residual_rms_vs_identity": (
                float(np.sqrt(np.mean(mean_residual**2)) / identity_rms)
                if identity_rms > 0
                else float("inf")
            ),
            "posterior_std_median_gev": float(np.median(posterior_std)),
            "pull_mean": float(np.mean(pull)),
            "pull_std": float(np.std(pull)),
            "coverage_1sigma": float(np.mean(inside)),
            "coverage_1sigma_nominal": 0.6827,
        }
    return report


def print_report(label: str, report: dict[str, Any]) -> None:
    per_event = report["per_event_mass"]
    marginal = report["marginal_mass"]
    print(f"\n=== {label} ({report['events']} paired events) ===")
    print("per-event mass residual against the withheld partner [GeV]")
    print(f"  model                    mean {per_event['model_mean']:+8.4f}"
          f"   rms {per_event['model_rms']:8.4f}"
          f"   correlation {per_event['truth_correlation']:.4f}")
    if report["has_identity_reference"]:
        print(f"  identity (hand back x)   mean {per_event['identity_mean']:+8.4f}"
              f"   rms {per_event['identity_rms']:8.4f}"
              f"   correlation {per_event['identity_truth_correlation']:.4f}")
        print(f"  residual_rms_vs_identity {per_event['residual_rms_vs_identity']:.4f}"
              "   (< 1 inverted resolution; >= 1 did not)")
        print("  four-vector column rms relative to identity:")
        print("    " + "  ".join(
            f"{value:.3f}"
            for value in report["per_event_components"]["rms_vs_identity_per_column"]
        ))
    else:
        print("  no identity reference for this direction (see paired_closure docs)")
    print("marginal mass (what the pipeline gates on today)")
    print(f"  model    KS {marginal['model_ks']:.4f}   W1 {marginal['model_w1_gev']:.4f}"
          f"   std {marginal['model_std_gev']:.4f}   (truth std {marginal['truth_std_gev']:.4f})")
    if report["has_identity_reference"]:
        print(f"  identity KS {marginal['identity_ks']:.4f}"
              f"   W1 {marginal['identity_w1_gev']:.4f}"
              f"   std {marginal['identity_std_gev']:.4f}")
    if "posterior" in report:
        posterior = report["posterior"]
        print(f"posterior over {posterior['draws']} draws")
        print(f"  mean-residual rms {posterior['mean_residual_rms']:.4f}"
              f"  (vs identity {posterior['mean_residual_rms_vs_identity']:.4f})")
        print(f"  pull mean {posterior['pull_mean']:+.4f} std {posterior['pull_std']:.4f}"
              f"   1-sigma coverage {posterior['coverage_1sigma']:.4f}"
              f" (nominal {posterior['coverage_1sigma_nominal']})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-npz",
        type=Path,
        help="Upstream OTUS results archive carrying z, x and x_encoded.",
    )
    parser.add_argument("--pairs-npz", type=Path, help="Archive with z and x.")
    parser.add_argument("--pred-npz", type=Path, help="Archive with z_pred.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    reports: dict[str, Any] = {}
    if args.results_npz is not None:
        data = np.load(args.results_npz)
        required = ("z", "x", "x_encoded")
        missing = [key for key in required if key not in data.files]
        if missing:
            raise SystemExit(f"{args.results_npz} is missing {missing}")
        reports["upstream_encoder"] = paired_closure(
            data["z"], data["x"], data["x_encoded"]
        )
        print_report("upstream OTUS encoder E(x) vs truth partner", reports["upstream_encoder"])
        if "x_reconstructed" in data.files:
            # The cycle is scored against the DETECTOR event, not the truth
            # event: D(E(x)) is meant to return x. Reported because a good
            # cycle with a bad encoder is exactly the failure mode we found.
            reports["upstream_cycle"] = paired_closure(
                data["x"], None, data["x_reconstructed"]
            )
            print_report(
                "upstream OTUS cycle D(E(x)) vs the detector event it came from",
                reports["upstream_cycle"],
            )
    elif args.pairs_npz is not None and args.pred_npz is not None:
        pairs = np.load(args.pairs_npz)
        pred = np.load(args.pred_npz)
        reports["model"] = paired_closure(
            pairs["z"],
            pairs["x"],
            pred["z_pred"],
            z_pred_draws=pred["z_pred_draws"] if "z_pred_draws" in pred.files else None,
        )
        print_report("model E(x) vs truth partner", reports["model"])
    else:
        parser.error("Pass --results-npz, or both --pairs-npz and --pred-npz")

    if args.out is not None:
        args.out.write_text(
            json.dumps(reports, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
