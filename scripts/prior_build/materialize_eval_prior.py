#!/usr/bin/env python
"""Materialise a ready-to-consume evaluation array from a unified prior.

Why this exists
---------------
The Session 29 unified priors are shipped **loose-generated, labelled, unmixed
and weighted** (docs/unified_prior_method.md sections 1, 3, 10):

* no analysis selection is applied - the window, muon pT threshold and |eta| cut
  are offline choices;
* ``FDL/component_id`` labels the components and no mixture fraction is baked in;
* ``FDL/weight`` holds the CKKW-L Sudakov weight, and it is **not** applied.

Every downstream consumer in this repository reads ``FDL/zData`` directly and
does none of those three things. ``cms_data.load_theory_prior_z`` returns
``z_data[:, :8]``; ``scripts_joint/upsilon/decode_prior.py`` reads
``FDL/zData`` and carries ``component_id`` through without acting on it. Feeding
a unified prior to either of them silently uses the wrong events at the wrong
ratio with the weights discarded.

This script does the three things once, writes a new file, and records exactly
what it did. Consumers stay dumb: point ``--prior`` at the output.

    python scripts/prior_build/materialize_eval_prior.py \
        --in  data/priors/upsilon_unified_bare_tms10.hdf5 \
        --out data/priors/upsilon_unified_bare_tms10_eval_legacyWindow.hdf5 \
        --window legacy --signal-fraction-from \
        data/cms_upsilon_mumu_mg5py8_ckkwl_8tev_inclusive_0j1j_fiducial_8p5_11p5_1M.hdf5

Design notes
------------
**Selection** replicates ``cms_data.filter_theory_prior`` exactly - stored
energies authoritative, daughters massless - because that is the convention the
training pipeline applies. The replication is *asserted* against that function
rather than assumed; see ``_selection_mask``. Two mass conventions live in this
repository and they do not agree (memory.md section 9); this script is in the
``stored`` one, on purpose, and says so in the output attributes.

**Unweighting defaults to accept-reject**, not importance resampling, so the
output contains **no duplicated rows**. Duplication is what makes the legacy 1M
reweighted prior carry a Kish ESS of 63,037 against 1,000,000 rows, and a random
split over duplicated rows puts copies of the same event on both sides. An
evaluation file has no train/test split, but the property is cheap to keep and
expensive to lose track of.

**The mixture fraction is applied here and recorded here.** It never enters the
generated file. Inter-signal ratios are preserved when a region has several
signal components (Upsilon 1S/2S/3S).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA = "unified-prior-eval-1"


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------
def _selection_mask(z: np.ndarray, selection: dict) -> np.ndarray:
    """Boolean keep-mask matching ``cms_data.filter_theory_prior`` row for row.

    ``filter_theory_prior`` returns filtered rows, not a mask, so it cannot
    carry ``component_id`` and ``weight`` along. This reproduces its arithmetic
    and is verified against it by ``verify_against_cms_data``.
    """
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] != 8 or len(z) == 0:
        raise ValueError(f"theory prior must be [N, 8], got {z.shape}")
    px1, py1, pz1 = z[:, 0], z[:, 1], z[:, 2]
    px2, py2, pz2 = z[:, 4], z[:, 5], z[:, 6]
    energy1, energy2 = z[:, 3], z[:, 7]
    pt1 = np.hypot(px1, py1)
    pt2 = np.hypot(px2, py2)
    pabs1 = np.sqrt(px1**2 + py1**2 + pz1**2)
    pabs2 = np.sqrt(px2**2 + py2**2 + pz2**2)
    eta1 = np.arctanh(np.clip(pz1 / pabs1, -1.0 + 1e-7, 1.0 - 1e-7))
    eta2 = np.arctanh(np.clip(pz2 / pabs2, -1.0 + 1e-7, 1.0 - 1e-7))
    mass2 = (energy1 + energy2) ** 2 - (
        (px1 + px2) ** 2 + (py1 + py2) ** 2 + (pz1 + pz2) ** 2
    )
    mass = np.sqrt(np.maximum(mass2, 0.0))

    keep = np.ones(len(z), dtype=bool)
    if selection.get("muon_pt_min") is not None:
        pt_min = float(selection["muon_pt_min"])
        keep &= (pt1 > pt_min) & (pt2 > pt_min)
    if selection.get("muon_abs_eta_max") is not None:
        eta_max = float(selection["muon_abs_eta_max"])
        keep &= (np.abs(eta1) < eta_max) & (np.abs(eta2) < eta_max)
    if selection.get("mass_min") is not None and selection.get("mass_max") is not None:
        keep &= (mass > float(selection["mass_min"])) & (mass < float(selection["mass_max"]))
    return keep


def verify_against_cms_data(z: np.ndarray, selection: dict, keep: np.ndarray) -> None:
    """Assert the mask reproduces ``cms_data.filter_theory_prior`` exactly."""
    from scripts.cms_data import filter_theory_prior

    reference = filter_theory_prior(z, selection)
    ours = z[keep]
    if reference.shape != ours.shape or not np.array_equal(reference, ours):
        raise AssertionError(
            "selection mask disagrees with cms_data.filter_theory_prior "
            f"({ours.shape} vs {reference.shape}). The prior would be cut "
            "differently here than in training; refusing to write."
        )


# --------------------------------------------------------------------------
# weights
# --------------------------------------------------------------------------
def kish_ess(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64)
    total = w.sum()
    if total <= 0:
        return 0.0
    return float(total * total / np.square(w).sum())


def unweight_accept_reject(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Keep event i with probability w_i / w_max. No duplicated rows, ever."""
    w = np.asarray(weights, dtype=np.float64)
    if np.any(w < 0):
        raise ValueError(
            "negative generator weights present; accept-reject cannot unweight "
            "them. Investigate before proceeding - do not silently clip."
        )
    w_max = w.max()
    if w_max <= 0:
        raise ValueError("all generator weights are zero")
    return rng.random(len(w)) < (w / w_max)


# --------------------------------------------------------------------------
# mixture fraction
# --------------------------------------------------------------------------
def measure_signal_fraction(path: Path, selection: dict) -> float:
    """Effective signal fraction of a reference prior, after the same cut."""
    import h5py

    with h5py.File(path, "r") as handle:
        z = np.asarray(handle["FDL/zData"])[:, :8]
        if "FDL/component_id" not in handle:
            raise KeyError(f"{path} has no FDL/component_id; cannot measure a fraction")
        cid = np.asarray(handle["FDL/component_id"])
        mapping = json.loads(handle.attrs.get("component_id_mapping", "{}"))
    keep = _selection_mask(z, selection)
    cid = cid[keep]
    continuum_ids = {int(v) for k, v in mapping.items() if "continuum" in str(k).lower()}
    if not continuum_ids:
        raise ValueError(f"{path}: no component named *continuum* in {mapping}")
    signal = np.isin(cid, list(continuum_ids), invert=True).sum()
    return float(signal) / float(len(cid))


def apply_signal_fraction(
    cid: np.ndarray,
    continuum_ids: set[int],
    target: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Index array realising ``target`` total signal fraction.

    Inter-signal ratios are preserved: whichever side is over-represented is
    subsampled, the other is kept whole, so no event is ever duplicated.
    """
    if not (0.0 < target < 1.0):
        raise ValueError(f"signal fraction must be in (0, 1), got {target}")
    is_continuum = np.isin(cid, list(continuum_ids))
    sig_idx = np.flatnonzero(~is_continuum)
    con_idx = np.flatnonzero(is_continuum)
    if len(sig_idx) == 0 or len(con_idx) == 0:
        raise ValueError("need both signal and continuum events to set a fraction")

    # keep all continuum, subsample signal   -> n_sig = f/(1-f) * n_con
    want_sig = int(round(target / (1.0 - target) * len(con_idx)))
    if want_sig <= len(sig_idx):
        chosen_sig = rng.choice(sig_idx, size=want_sig, replace=False)
        chosen_con = con_idx
    else:
        # keep all signal, subsample continuum -> n_con = (1-f)/f * n_sig
        want_con = int(round((1.0 - target) / target * len(sig_idx)))
        chosen_sig = sig_idx
        chosen_con = rng.choice(con_idx, size=min(want_con, len(con_idx)), replace=False)
    out = np.concatenate([chosen_sig, chosen_con])
    out.sort()
    return out


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def _emit(report: dict, out_file: Path | None = None) -> None:
    """Always show the funnel - a failure without its numbers is unusable."""
    print(json.dumps(report, indent=2))
    if out_file is not None:
        path = out_file.with_suffix(".funnel.json")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"funnel written to {path}")
        except OSError:
            pass


def _abort(report: dict, out_file: Path | None, message: str) -> "SystemExit":
    report["aborted"] = message
    _emit(report, out_file)
    return SystemExit(message)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="in_file", type=Path, required=True)
    p.add_argument("--out", dest="out_file", type=Path, required=True)
    p.add_argument("--window", choices=("legacy", "new", "explicit"), default="legacy",
                   help="Take the mass window from the input file's "
                        "analysis_window_legacy_GeV / analysis_window_new_GeV "
                        "attribute, or pass --mass-min/--mass-max.")
    p.add_argument("--mass-min", type=float, default=None)
    p.add_argument("--mass-max", type=float, default=None)
    p.add_argument("--muon-pt-min", type=float, default=None,
                   help="Defaults to the input's analysis_muon_pt_GeV attribute.")
    p.add_argument("--muon-abs-eta-max", type=float, default=2.4)
    p.add_argument("--unweight", choices=("accept_reject", "none"),
                   default="accept_reject")
    p.add_argument("--min-ess-ratio", type=float, default=0.5,
                   help="Abort if Kish ESS / N falls below this.")
    p.add_argument("--weight-cap-quantile", type=float, default=None,
                   help="Cap generator weights at this quantile before "
                        "accept-reject (e.g. 0.999). Kish ESS is a bulk "
                        "statistic and can look healthy while a handful of "
                        "outliers set w_max and destroy the accept-reject "
                        "efficiency, which is mean(w)/max(w). Capping trades a "
                        "quantified bias - the weight above the cap - for "
                        "efficiency. The bias is reported; read it.")
    p.add_argument("--diagnose", action="store_true",
                   help="Report the funnel and exit without writing.")
    p.add_argument("--signal-fraction", type=float, default=None)
    p.add_argument("--signal-fraction-from", type=Path, default=None,
                   help="Measure the fraction from this reference prior, after "
                        "the same selection.")
    p.add_argument("--min-events", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=20260906)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip-cms-data-check", action="store_true",
                   help="Skip the filter_theory_prior cross-check (for unit "
                        "tests with synthetic fixtures).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    import h5py

    args = parse_args(argv)
    if args.out_file.exists() and not args.overwrite:
        raise SystemExit(f"{args.out_file} exists; pass --overwrite")
    rng = np.random.default_rng(args.seed)
    report: dict = {"schema_version": SCHEMA, "source": str(args.in_file)}

    with h5py.File(args.in_file, "r") as handle:
        if "FDL/zData" not in handle:
            raise KeyError("expected FDL/zData")
        z = np.asarray(handle["FDL/zData"])[:, :8]
        cid = (np.asarray(handle["FDL/component_id"])
               if "FDL/component_id" in handle else None)
        weight = (np.asarray(handle["FDL/weight"], dtype=np.float64)
                  if "FDL/weight" in handle else None)
        src_attrs = {k: handle.attrs[k] for k in handle.attrs}

    def attr(name, default=None):
        value = src_attrs.get(name, default)
        return value.decode() if isinstance(value, bytes) else value

    # ---- selection -------------------------------------------------------
    if args.window == "explicit":
        mass_min, mass_max = args.mass_min, args.mass_max
        if mass_min is None or mass_max is None:
            raise SystemExit("--window explicit needs --mass-min and --mass-max")
    else:
        key = f"analysis_window_{args.window}_GeV"
        raw = attr(key)
        if raw is None:
            raise SystemExit(f"input has no {key}; pass --window explicit")
        window = json.loads(raw) if isinstance(raw, str) else list(raw)
        mass_min, mass_max = float(window[0]), float(window[1])
    pt_min = args.muon_pt_min
    if pt_min is None:
        pt_min = attr("analysis_muon_pt_GeV")
        if pt_min is None:
            raise SystemExit("input has no analysis_muon_pt_GeV; pass --muon-pt-min")
        pt_min = float(pt_min)

    selection = {
        "mass_min": mass_min,
        "mass_max": mass_max,
        "muon_pt_min": pt_min,
        "muon_abs_eta_max": args.muon_abs_eta_max,
    }
    report["selection"] = selection
    report["selection_convention"] = (
        "stored energies authoritative, daughters massless - identical to "
        "cms_data.filter_theory_prior (memory.md section 9)"
    )

    # Generation cuts must be strictly looser than the selection, or the file
    # cannot support this window and the result would be silently truncated.
    gen = attr("generation_cuts")
    if gen:
        gen = json.loads(gen) if isinstance(gen, str) else gen
        gmin, gmax = gen.get("mass_window_GeV", [None, None])
        if gmin is not None and not (gmin < mass_min and gmax > mass_max):
            raise SystemExit(
                f"generation window {gen['mass_window_GeV']} does not strictly "
                f"contain the requested {[mass_min, mass_max]}; the edges would "
                "be truncated with no migration margin."
            )
        gpt = gen.get("lepton_pt_min_GeV")
        if gpt is not None and not gpt < pt_min:
            raise SystemExit(
                f"generation ptl {gpt} is not below the requested muon pT "
                f"{pt_min}; no migration margin on the threshold."
            )
        report["generation_cuts"] = gen

    report["n_source"] = int(len(z))
    keep = _selection_mask(z, selection)
    if not args.skip_cms_data_check:
        verify_against_cms_data(z, selection, keep)
    z, cid = z[keep], (cid[keep] if cid is not None else None)
    weight = weight[keep] if weight is not None else None
    report["n_after_selection"] = int(len(z))

    # ---- weights ---------------------------------------------------------
    if weight is None:
        report["unweighting"] = "no FDL/weight in source"
    elif args.unweight == "none":
        report["unweighting"] = "none - weights carried through unapplied"
        report["kish_ess"] = kish_ess(weight)
    else:
        ess = kish_ess(weight)
        ratio = ess / len(weight)
        w_mean, w_max = float(weight.mean()), float(weight.max())
        report["kish_ess"] = ess
        report["kish_ess_ratio"] = ratio
        report["weight_mean"] = w_mean
        report["weight_max"] = w_max
        report["weight_max_over_mean"] = (w_max / w_mean) if w_mean else None
        report["accept_reject_efficiency_raw"] = (w_mean / w_max) if w_max else 0.0
        report["weight_quantiles"] = {
            str(q): float(np.quantile(weight, q))
            for q in (0.5, 0.9, 0.99, 0.999, 0.9999, 1.0)
        }
        if ratio < args.min_ess_ratio:
            raise _abort(report, args.out_file,
                f"Kish ESS/N = {ratio:.4f} is below --min-ess-ratio "
                f"{args.min_ess_ratio}. Unweighting would throw away most of the "
                "sample; carry the weights into the estimator instead.")

        working = weight
        if args.weight_cap_quantile is not None:
            cap = float(np.quantile(weight, args.weight_cap_quantile))
            above = weight > cap
            report["weight_cap_quantile"] = args.weight_cap_quantile
            report["weight_cap"] = cap
            report["weight_above_cap_events"] = int(above.sum())
            # The bias introduced: how much of the total weight is discarded by
            # flattening the tail. This is the number that decides whether the
            # cap is acceptable, not the efficiency gain.
            report["weight_above_cap_bias_fraction_of_total"] = float(
                (weight[above] - cap).sum() / weight.sum()
            ) if above.any() else 0.0
            working = np.minimum(weight, cap)
            report["accept_reject_efficiency_capped"] = float(
                working.mean() / working.max()
            )

        mask = unweight_accept_reject(working, rng)
        report["unweighting"] = ("accept_reject_capped"
                                 if args.weight_cap_quantile is not None
                                 else "accept_reject")
        report["accept_reject_efficiency_achieved"] = float(mask.mean())
        report["duplicated_rows"] = 0
        z, cid = z[mask], (cid[mask] if cid is not None else None)
        weight = None
        report["n_after_unweighting"] = int(len(z))

    # ---- mixture fraction ------------------------------------------------
    mapping = attr("component_id_mapping")
    mapping = json.loads(mapping) if isinstance(mapping, str) else (mapping or {})
    continuum_ids = {int(v) for k, v in mapping.items() if "continuum" in str(k).lower()}
    target = args.signal_fraction
    if args.signal_fraction_from is not None:
        if target is not None:
            raise SystemExit("pass only one of --signal-fraction / --signal-fraction-from")
        target = measure_signal_fraction(args.signal_fraction_from, selection)
        report["signal_fraction_source"] = str(args.signal_fraction_from)
    if target is None or cid is None or not continuum_ids:
        report["signal_fraction_applied"] = None
        report["signal_fraction_note"] = (
            "no fraction applied - single-component region, or none requested"
        )
    else:
        idx = apply_signal_fraction(cid, continuum_ids, target, rng)
        z, cid = z[idx], cid[idx]
        achieved = float(np.isin(cid, list(continuum_ids), invert=True).mean())
        report["signal_fraction_requested"] = float(target)
        report["signal_fraction_applied"] = achieved
        report["n_after_fraction"] = int(len(z))

    if args.diagnose:
        report["diagnose_only"] = True
        _emit(report, args.out_file)
        return 0

    if len(z) < args.min_events:
        raise _abort(report, args.out_file,
            f"only {len(z)} events survive; below --min-events "
            f"{args.min_events}. The funnel above shows which stage collapsed - "
            "read accept_reject_efficiency_raw and weight_max_over_mean first: a "
            "healthy Kish ESS with a large weight_max_over_mean means a few "
            "outlier weights are setting w_max, and --weight-cap-quantile 0.999 "
            "is the fix. If the selection is what cut it, the window or the "
            "muon pT threshold is the problem, not the weights.")

    # ---- write -----------------------------------------------------------
    report["number_of_events"] = int(len(z))
    report["seed"] = args.seed
    report["produced_utc"] = datetime.now(timezone.utc).isoformat()
    if cid is not None:
        report["component_counts_out"] = {
            str(name): int((cid == int(ident)).sum()) for name, ident in mapping.items()
        }

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out_file, "w") as out:
        group = out.create_group("FDL")
        group.create_dataset("zData", data=z.astype(np.float32), compression="gzip")
        if cid is not None:
            group.create_dataset("component_id", data=cid.astype(np.int8),
                                 compression="gzip")
        if weight is not None:
            group.create_dataset("weight", data=weight.astype(np.float64),
                                 compression="gzip")
        for key, value in src_attrs.items():
            out.attrs[f"source_{key}"] = value
        for key, value in report.items():
            out.attrs[key] = (json.dumps(value)
                              if isinstance(value, (dict, list)) else value)
        out.attrs["derived_by"] = "scripts/prior_build/materialize_eval_prior.py"
        out.attrs["selection_baked_in"] = True
        out.attrs["ready_for_decode_prior"] = True

    json_path = args.out_file.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nwrote {args.out_file}\nwrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
