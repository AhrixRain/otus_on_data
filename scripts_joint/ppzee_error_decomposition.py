#!/usr/bin/env python
"""Reproducible per-event error decomposition on the ppzee paired bench.

Meeting point 5 asked whether the prediction error is "error inside the
components accumulated" or something else. This script answers that on the
only place where CMS-style paired truth exists: the retained 160,000 held-out
ppzee pairs (``pairs_test.npz``: truth ``z`` and detector ``x``) and the frozen
Run E encoder output (``pred_test.npz``: ``z_pred = E(x)``).

It writes ``error_decomposition.json`` and ``REPORT.md`` into a NEW output
directory and never touches the existing paired-closure artifacts.

Decomposition used
------------------
1. Aggregate
       r_identity = m(x) - m(z)          (hand the detector event back)
       r_model    = m(E(x)) - m(z)       (our encoder)
   bias_removed_fraction = 1 - mean(r_model) / mean(r_identity)
   scatter_ratio         = std(r_model) / std(r_identity)
   residual_rms_vs_identity = rms(r_model) / rms(r_identity)

2. Inherited vs added
       r_model = r_identity + added
   95% covariance share on r_identity means the encoder is mostly passing the
   detector response through instead of inverting it per event.

3. Component decomposition in the model's own coordinates
   The four-vector linearization m^2 = E^2 - p^2 is numerically
   ill-conditioned for a ~91 GeV dimuon (large cancellation). The massless
   dimuon identity

       m^2 = 2 pT- pT+ (cosh(dEta) - cos(dPhi))

   is well-conditioned and matches the model's [log pT, eta, phi] working
   coordinates:

       dm/m = 0.5 (dlog pT- + dlog pT+)                  (pT term)
              + 0.5 [sinh(dEta) d(dEta)
                     + sin(dPhi) d(dPhi)] / denom        (angular term)

   We report the R^2 of (pT + angular) against the exact fractional residual.

4. Common vs differential log-pT
       common       = 0.5 (dlog pT- + dlog pT+)
       differential = 0.5 (dlog pT- - dlog pT+)
   For a back-to-back pair the mass depends on the common mode; the
   differential mode largely cancels. A detector whose per-muon log-pT errors
   are anti-correlated has its mass-inert differential noise; an encoder whose
   residual becomes correlated has converted it into a mass-moving scale error.

Sources: outputs/cms_Joint/ppzee/paired_closure/{pairs_test,pred_test}.npz and
the committed paired_closure_decomposition.json (read for reference only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAIRS = (
    REPO_ROOT / "outputs" / "cms_Joint" / "ppzee" / "paired_closure" / "pairs_test.npz"
)
DEFAULT_PRED = (
    REPO_ROOT / "outputs" / "cms_Joint" / "ppzee" / "paired_closure" / "pred_test.npz"
)
DEFAULT_EXISTING = (
    REPO_ROOT
    / "outputs"
    / "cms_Joint"
    / "ppzee"
    / "paired_closure_decomposition.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "outputs" / "cms_Joint" / "ppzee" / "error_decomposition"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--pred", type=Path, default=DEFAULT_PRED)
    parser.add_argument(
        "--existing-json",
        type=Path,
        default=DEFAULT_EXISTING,
        help="Committed aggregate decomposition, read for cross-checking only.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def invariant_mass(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    energy = values[:, 3] + values[:, 7]
    momentum = values[:, :3] + values[:, 4:7]
    return np.sqrt(np.maximum(energy * energy - np.sum(momentum * momentum, axis=1), 0.0))


def _log_pt_eta_phi(values: np.ndarray, base: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (log pT, eta, phi) for one muon in a [N, 8] charge-ordered block."""
    px = values[:, base + 0]
    py = values[:, base + 1]
    pz = values[:, base + 2]
    pt = np.maximum(np.hypot(px, py), 1.0e-12)
    return np.log(pt), np.arcsinh(pz / pt), np.arctan2(py, px)


def _wrap_phi(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def _skewness(values: np.ndarray) -> float:
    std = values.std()
    if std <= 0.0:
        return 0.0
    return float(((values - values.mean()) ** 3).mean() / std**3)


def _quantiles(values: np.ndarray) -> dict[str, float]:
    q = np.percentile(values, [0.1, 1.0, 16.0, 50.0, 84.0, 99.0, 99.9])
    return {
        "q0.1_gev": float(q[0]),
        "q1_gev": float(q[1]),
        "q16_gev": float(q[2]),
        "median_gev": float(q[3]),
        "q84_gev": float(q[4]),
        "q99_gev": float(q[5]),
        "q99.9_gev": float(q[6]),
        "left_tail_q16_minus_q1_gev": float(q[2] - q[1]),
        "right_tail_q99_minus_q84_gev": float(q[5] - q[4]),
    }


def _shape(values: np.ndarray) -> dict[str, float]:
    return {
        "events": int(len(values)),
        "mean_gev": float(values.mean()),
        "std_gev": float(values.std()),
        "skewness": _skewness(values),
        **_quantiles(values),
    }


def _fractional_terms(z: np.ndarray, delta: np.ndarray) -> dict[str, np.ndarray]:
    """Model-space mass decomposition of a per-event four-vector shift."""
    lm0, em0, pm0 = _log_pt_eta_phi(z, 0)
    lp0, ep0, pp0 = _log_pt_eta_phi(z, 4)
    shifted = z + delta
    lm1, em1, pm1 = _log_pt_eta_phi(shifted, 0)
    lp1, ep1, pp1 = _log_pt_eta_phi(shifted, 4)

    dlogpt_m = lm1 - lm0
    dlogpt_p = lp1 - lp0
    common = 0.5 * (dlogpt_m + dlogpt_p)
    differential = 0.5 * (dlogpt_m - dlogpt_p)

    delta_eta = ep0 - em0
    delta_phi = pp0 - pm0
    d_delta_eta = (ep1 - em1) - (ep0 - em0)
    d_delta_phi = (pp1 - pm1) - (pp0 - pm0)
    denom = np.cosh(delta_eta) - np.cos(delta_phi)
    angular = (
        0.5
        * (np.sinh(delta_eta) * d_delta_eta + np.sin(delta_phi) * d_delta_phi)
        / denom
    )
    return {
        "log_pt_minus": dlogpt_m,
        "log_pt_plus": dlogpt_p,
        "common_log_pt": common,
        "differential_log_pt": differential,
        "angular": angular,
        "d_eta_minus": em1 - em0,
        "d_eta_plus": ep1 - ep0,
        "d_phi_minus": _wrap_phi(pm1 - pm0),
        "d_phi_plus": _wrap_phi(pp1 - pp0),
    }


def _r2(target: np.ndarray, prediction: np.ndarray) -> float:
    variance = float(np.var(target))
    if variance <= 0.0:
        return float("nan")
    return float(1.0 - np.var(target - prediction) / variance)


def _covariance_share(component: np.ndarray, target: np.ndarray) -> float:
    variance = float(np.var(target))
    if variance <= 0.0:
        return float("nan")
    return float(np.cov(component, target)[0, 1] / variance)


def _affine_oracle_ratio(predictor_mass: np.ndarray, truth_mass: np.ndarray, identity_rms: float):
    design = np.vstack([predictor_mass, np.ones_like(predictor_mass)]).T
    coef, *_ = np.linalg.lstsq(design, truth_mass, rcond=None)
    residual = (design @ coef) - truth_mass
    return {
        "slope": float(coef[0]),
        "intercept_gev": float(coef[1]),
        "residual_rms_gev": float(np.sqrt((residual**2).mean())),
        "ratio_vs_identity": float(np.sqrt((residual**2).mean()) / identity_rms),
    }


def build_report(pairs_path: Path, pred_path: Path, existing_path: Path) -> dict:
    pairs = np.load(pairs_path, allow_pickle=True)
    z = np.asarray(pairs["z"], dtype=np.float64)
    x = np.asarray(pairs["x"], dtype=np.float64)
    pred = np.asarray(np.load(pred_path, allow_pickle=True)["z_pred"], dtype=np.float64)
    if not (len(z) == len(x) == len(pred)):
        raise ValueError(
            f"Cardinality mismatch: z={len(z)} x={len(x)} pred={len(pred)}"
        )

    mass_z = invariant_mass(z)
    mass_x = invariant_mass(x)
    mass_pred = invariant_mass(pred)

    r_identity = mass_x - mass_z
    r_model = mass_pred - mass_z
    identity_rms = float(np.sqrt((r_identity**2).mean()))
    model_rms = float(np.sqrt((r_model**2).mean()))
    frac_identity = r_identity / mass_z
    frac_model = r_model / mass_z
    added = frac_model - frac_identity

    aggregate = {
        "events": int(len(z)),
        "identity": {
            "mean_gev": float(r_identity.mean()),
            "std_gev": float(r_identity.std()),
            "rms_gev": identity_rms,
        },
        "model": {
            "mean_gev": float(r_model.mean()),
            "std_gev": float(r_model.std()),
            "rms_gev": model_rms,
        },
        "bias_removed_fraction": float(1.0 - r_model.mean() / r_identity.mean()),
        "scatter_ratio_model_over_identity": float(r_model.std() / r_identity.std()),
        "residual_rms_vs_identity": float(model_rms / identity_rms),
        "constant_offset_oracle_ratio": float(r_identity.std() / identity_rms),
        "affine_oracle_on_identity": _affine_oracle_ratio(mass_x, mass_z, identity_rms),
        "affine_oracle_on_model": _affine_oracle_ratio(mass_pred, mass_z, identity_rms),
        "affine_oracle_on_model_optimistic_note": (
            "fitted and scored on the same 160k pairs; it is an upper bound, not "
            "a held-out oracle."
        ),
    }

    inherited = {
        "corr_frac_model_frac_identity": float(
            np.corrcoef(frac_model, frac_identity)[0, 1]
        ),
        "r2_frac_model_explained_by_identity": _r2(frac_model, frac_identity),
        "covariance_share_identity": _covariance_share(frac_identity, frac_model),
        "covariance_share_encoder_added": _covariance_share(added, frac_model),
        "encoder_added_rms": float(np.sqrt((added**2).mean())),
        "encoder_added_mean": float(added.mean()),
    }

    terms_identity = _fractional_terms(z, x - z)
    terms_model = _fractional_terms(z, pred - z)
    component_block: dict[str, dict] = {}
    for label, terms, frac in (
        ("identity_detector_x_minus_z", terms_identity, frac_identity),
        ("model_encoder_pred_minus_z", terms_model, frac_model),
    ):
        total = terms["common_log_pt"] + terms["angular"]
        component_block[label] = {
            "residual_fractional_rms": float(np.sqrt((frac**2).mean())),
            "residual_fractional_mean": float(frac.mean()),
            "per_muon_log_pt_std": {
                "mu_minus": float(terms["log_pt_minus"].std()),
                "mu_plus": float(terms["log_pt_plus"].std()),
            },
            "corr_log_pt_minus_plus": float(
                np.corrcoef(terms["log_pt_minus"], terms["log_pt_plus"])[0, 1]
            ),
            "eta_std": {
                "mu_minus": float(terms["d_eta_minus"].std()),
                "mu_plus": float(terms["d_eta_plus"].std()),
            },
            "phi_std": {
                "mu_minus": float(terms["d_phi_minus"].std()),
                "mu_plus": float(terms["d_phi_plus"].std()),
            },
            "common_log_pt_std": float(terms["common_log_pt"].std()),
            "differential_log_pt_std": float(terms["differential_log_pt"].std()),
            "common_over_differential_variance_ratio": float(
                terms["common_log_pt"].var() / max(terms["differential_log_pt"].var(), 1e-300)
            ),
            "mass_component_terms": {
                "common_log_pt": {
                    "rms": float(np.sqrt((terms["common_log_pt"] ** 2).mean())),
                    "covariance_share": _covariance_share(terms["common_log_pt"], frac),
                },
                "angular": {
                    "rms": float(np.sqrt((terms["angular"] ** 2).mean())),
                    "covariance_share": _covariance_share(terms["angular"], frac),
                },
                "differential_log_pt": {
                    "rms": float(np.sqrt((terms["differential_log_pt"] ** 2).mean())),
                    "covariance_share": _covariance_share(
                        terms["differential_log_pt"], frac
                    ),
                },
            },
            "r2_common_plus_angular": _r2(frac, total),
            "r2_common_only": _r2(frac, terms["common_log_pt"]),
            "remainder_rms": float(np.sqrt(((frac - total) ** 2).mean())),
        }

    shape = {
        "truth_z": _shape(mass_z),
        "detector_x": _shape(mass_x),
        "encoder_pred": _shape(mass_pred),
    }

    existing = None
    if existing_path.exists():
        existing = json.loads(existing_path.read_text(encoding="utf-8"))

    report = {
        "schema_version": 1,
        "provenance": {
            "pairs": {"path": str(pairs_path), "sha256": file_sha256(pairs_path)},
            "pred": {"path": str(pred_path), "sha256": file_sha256(pred_path)},
            "existing_decomposition": (
                {"path": str(existing_path), "contents": existing}
                if existing is not None
                else None
            ),
        },
        "aggregate": aggregate,
        "inherited_vs_added": inherited,
        "components": component_block,
        "shape": shape,
    }

    # A direct cross-check against the committed aggregate numbers.
    if existing:
        report["cross_check_vs_committed"] = {
            "identity_mean_gev": [aggregate["identity"]["mean_gev"], existing.get("identity", {}).get("mean")],
            "identity_std_gev": [aggregate["identity"]["std_gev"], existing.get("identity", {}).get("std")],
            "model_mean_gev": [aggregate["model"]["mean_gev"], existing.get("model", {}).get("mean")],
            "model_std_gev": [aggregate["model"]["std_gev"], existing.get("model", {}).get("std")],
            "residual_rms_vs_identity": [
                aggregate["residual_rms_vs_identity"],
                existing.get("residual_rms_vs_identity"),
            ],
        }
    return report


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def write_markdown(report: dict, output_path: Path) -> None:
    agg = report["aggregate"]
    inh = report["inherited_vs_added"]
    comp = report["components"]
    shape = report["shape"]
    ref = (report["provenance"]["existing_decomposition"] or {}).get("contents") or {}

    lines = [
        "# ppzee per-event error decomposition",
        "",
        "Generated by `scripts_joint/ppzee_error_decomposition.py` from the retained",
        "160,000 held-out ppzee pairs. Labels follow `CLAUDE.md` section 2: every",
        "number below is **artifact-measured** unless marked otherwise.",
        "",
        "## 1. Aggregate mass residual `m(map) - m(z)` (units GeV unless noted)",
        "",
        "| residual | mean | std | rms |",
        "|---|---|---|---|",
        f"| identity `x` | {_fmt(agg['identity']['mean_gev'])} | {_fmt(agg['identity']['std_gev'])} | {_fmt(agg['identity']['rms_gev'])} |",
        f"| encoder `E(x)` | {_fmt(agg['model']['mean_gev'])} | {_fmt(agg['model']['std_gev'])} | {_fmt(agg['model']['rms_gev'])} |",
        "",
        f"- `bias_removed_fraction` = **{_fmt(agg['bias_removed_fraction'], 3)}**",
        f"- `scatter_ratio_model_over_identity` = **{_fmt(agg['scatter_ratio_model_over_identity'], 3)}**",
        f"- `residual_rms_vs_identity` = **{_fmt(agg['residual_rms_vs_identity'], 4)}**",
        f"- constant-offset oracle ratio = {_fmt(agg['constant_offset_oracle_ratio'], 4)}",
        f"- affine oracle on identity ratio = {_fmt(agg['affine_oracle_on_identity']['ratio_vs_identity'], 4)}",
        f"- affine oracle on model ratio = {_fmt(agg['affine_oracle_on_model']['ratio_vs_identity'], 4)} (optimistic, fitted in-sample)",
        "",
        f"Committed reference `paired_closure_decomposition.json`: "
        f"`residual_rms_vs_identity` = {_fmt(ref.get('residual_rms_vs_identity'), 4)}, "
        f"`bias_removed_fraction` = {_fmt(ref.get('bias_removed_fraction'), 3)}, "
        f"`scatter_ratio_model_over_identity` = {_fmt(ref.get('scatter_ratio_model_over_identity'), 3)}, "
        f"`oracle_floor_ratio` = {_fmt(ref.get('oracle_floor_ratio'), 4)} "
        "(the last one is reported, not recomputed here).",
        "",
        "## 2. Inherited vs encoder-added",
        "",
        f"- corr(fractional model residual, fractional identity residual) = **{_fmt(inh['corr_frac_model_frac_identity'], 4)}**",
        f"- R^2 of the model residual explained by the identity residual = **{_fmt(inh['r2_frac_model_explained_by_identity'], 4)}**",
        f"- covariance share on the identity residual = **{_fmt(inh['covariance_share_identity'], 3)}**",
        f"- covariance share on the encoder-added piece = **{_fmt(inh['covariance_share_encoder_added'], 3)}**",
        f"- encoder-added rms = {_fmt(inh['encoder_added_rms'], 5)} (mean {_fmt(inh['encoder_added_mean'], 5)})",
        "",
        "## 3. Component decomposition in the model's own coordinates",
        "",
        "| quantity | detector `x-z` | encoder `E(x)-z` |",
        "|---|---|---|",
        f"| per-muon log-pT std (mu-) | {_fmt(comp['identity_detector_x_minus_z']['per_muon_log_pt_std']['mu_minus'], 4)} | {_fmt(comp['model_encoder_pred_minus_z']['per_muon_log_pt_std']['mu_minus'], 4)} |",
        f"| per-muon log-pT std (mu+) | {_fmt(comp['identity_detector_x_minus_z']['per_muon_log_pt_std']['mu_plus'], 4)} | {_fmt(comp['model_encoder_pred_minus_z']['per_muon_log_pt_std']['mu_plus'], 4)} |",
        f"| corr(mu- log-pT, mu+ log-pT) | {_fmt(comp['identity_detector_x_minus_z']['corr_log_pt_minus_plus'], 4)} | {_fmt(comp['model_encoder_pred_minus_z']['corr_log_pt_minus_plus'], 4)} |",
        f"| common-mode log-pT std | {_fmt(comp['identity_detector_x_minus_z']['common_log_pt_std'], 4)} | {_fmt(comp['model_encoder_pred_minus_z']['common_log_pt_std'], 4)} |",
        f"| differential log-pT std | {_fmt(comp['identity_detector_x_minus_z']['differential_log_pt_std'], 4)} | {_fmt(comp['model_encoder_pred_minus_z']['differential_log_pt_std'], 4)} |",
        f"| var(common)/var(differential) | {_fmt(comp['identity_detector_x_minus_z']['common_over_differential_variance_ratio'], 3)} | {_fmt(comp['model_encoder_pred_minus_z']['common_over_differential_variance_ratio'], 3)} |",
        f"| covariance share: common log-pT | {_fmt(comp['identity_detector_x_minus_z']['mass_component_terms']['common_log_pt']['covariance_share'], 3)} | {_fmt(comp['model_encoder_pred_minus_z']['mass_component_terms']['common_log_pt']['covariance_share'], 3)} |",
        f"| covariance share: angular | {_fmt(comp['identity_detector_x_minus_z']['mass_component_terms']['angular']['covariance_share'], 3)} | {_fmt(comp['model_encoder_pred_minus_z']['mass_component_terms']['angular']['covariance_share'], 3)} |",
        f"| covariance share: differential log-pT | {_fmt(comp['identity_detector_x_minus_z']['mass_component_terms']['differential_log_pt']['covariance_share'], 3)} | {_fmt(comp['model_encoder_pred_minus_z']['mass_component_terms']['differential_log_pt']['covariance_share'], 3)} |",
        f"| R^2 (common + angular) | {_fmt(comp['identity_detector_x_minus_z']['r2_common_plus_angular'], 3)} | {_fmt(comp['model_encoder_pred_minus_z']['r2_common_plus_angular'], 3)} |",
        f"| remainder rms | {_fmt(comp['identity_detector_x_minus_z']['remainder_rms'], 5)} | {_fmt(comp['model_encoder_pred_minus_z']['remainder_rms'], 5)} |",
        "",
        "All standardizations above are fractional (log pT, not pT), so the columns",
        "are directly comparable between the detector and the encoder.",
        "",
        "## 4. Mass shape",
        "",
        "| distribution | median | std | skewness | left tail | right tail |",
        "|---|---|---|---|---|---|",
    ]
    for label, key in (("truth `z`", "truth_z"), ("detector `x`", "detector_x"), ("encoder `E(x)`", "encoder_pred")):
        s = shape[key]
        lines.append(
            f"| {label} | {_fmt(s['median_gev'], 4)} | {_fmt(s['std_gev'], 5)} | "
            f"{_fmt(s['skewness'], 3)} | {_fmt(s['left_tail_q16_minus_q1_gev'], 4)} | "
            f"{_fmt(s['right_tail_q99_minus_q84_gev'], 4)} |"
        )
    lines += [
        "",
        "## 5. Reading",
        "",
        "1. The encoder removes most of the global mass bias "
        f"({_fmt(100.0 * agg['bias_removed_fraction'], 1)}%) but not the per-event scatter "
        f"(ratio {_fmt(agg['scatter_ratio_model_over_identity'], 3)}).",
        f"2. {_fmt(100.0 * inh['covariance_share_identity'], 1)}% of the model residual "
        "variance is inherited from the detector response; only "
        f"{_fmt(100.0 * inh['covariance_share_encoder_added'], 1)}% is encoder-added.",
        "3. The detector's per-muon log-pT errors are anti-correlated, so most of their "
        "variance is differential (mass-inert). The encoder's residual is strongly "
        "correlated, i.e. almost purely common-mode, which is the mass-moving channel. "
        "The error is therefore not independent per-component accumulation; it is "
        "organized into a coherent pair-level scale error.",
        "",
        "Full machine-readable numbers: `error_decomposition.json` in this directory.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} already exists and is non-empty; pass --overwrite to replace."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(
        args.pairs.expanduser().resolve(),
        args.pred.expanduser().resolve(),
        args.existing_json.expanduser().resolve(),
    )
    (output_dir / "error_decomposition.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_markdown(report, output_dir / "REPORT.md")

    agg = report["aggregate"]
    inh = report["inherited_vs_added"]
    print(f"Wrote {output_dir / 'error_decomposition.json'}")
    print(f"Wrote {output_dir / 'REPORT.md'}")
    print(
        "aggregate: identity rms %.4f GeV, model rms %.4f GeV, "
        "ratio %.4f, bias removed %.1f%%, scatter ratio %.3f"
        % (
            agg["identity"]["rms_gev"],
            agg["model"]["rms_gev"],
            agg["residual_rms_vs_identity"],
            100.0 * agg["bias_removed_fraction"],
            agg["scatter_ratio_model_over_identity"],
        )
    )
    print(
        "inherited vs added: corr %.4f, identity covariance share %.3f, "
        "encoder-added share %.3f"
        % (
            inh["corr_frac_model_frac_identity"],
            inh["covariance_share_identity"],
            inh["covariance_share_encoder_added"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
