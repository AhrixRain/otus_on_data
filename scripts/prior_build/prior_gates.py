#!/usr/bin/env python3
"""Run the method book section 11 acceptance gates on a unified prior.

Six gates, each recorded whether it passes or fails.  A prior that fails a gate
is not shipped under a clean name -- the caller writes it with `_REJECTED` in the
filename and `status: REJECTED` in its manifest.  **Nothing in this script may be
adjusted to make a gate pass.**  The thresholds are fixed here, in one place,
with the measurement that set them written next to them.

  1. Signal width.  Every signal component's robust width (IQR/1.349) below
     SIGNAL_WIDTH_MAX_MEV.

     Calibrated 2026-09-06, not taken from Gamma.  The truth muons are *bare* --
     post-shower, after lepton QED FSR -- so the truth line is a Breit-Wigner
     convolved with a radiative tail and a threshold derived from Gamma alone
     would reject a correct file.  artifact-measured robust widths:
       correct bare files : shipped J/psi signal 0.107 MeV; shipped Upsilon
                            1S/2S/3S 0.317 / 0.353 / 0.423 MeV; a sample
                            converted here from recovered HepMC 0.648 MeV
       genuine smears     : A/B smeared arm 28.03 MeV; Program B hand-smeared
                            29.34 MeV
     5.0 MeV passes the widest correct file by 7.7x and rejects the smallest
     genuine smear by 5.6x.

     NOTE, and it matters: the widely repeated claim that the shipped Upsilon
     prior's signal components are "110-192 MeV wide" is a **standard deviation**,
     not a width.  Their robust widths are 0.317-0.423 MeV.  That prior is not
     smeared, and this gate would have passed it.

  2. Mass convention.  E-based and p-based pair masses agree to better than
     MASS_CONVENTION_TOL relative.  This is the massive-muon check: it fails
     loudly if the restrict card lost `MM`.

     **It has to be evaluated at full precision, not on the stored file.**  The
     method book section 10 schema is `[N, 8] float32`, and for a boosted pair
     `m^2 = E^2 - p^2` is a cancellation: at the Z, E_pair can be several hundred
     GeV against a 91 GeV mass, so float32's ~1e-7 relative precision on E is
     amplified into ~1e-3 on m in the extreme-boost tail.  artifact-measured
     2026-09-06 on this run's Z prior: worst deviation on the stored file
     1.4975e-03, and the float32 round-off FLOOR -- computed by putting the
     muons exactly on shell in float64, rounding to float32, and re-running the
     identical comparison -- is **1.4975e-03**, a ratio of 1.000.  The stored-file
     number is round-off and contains no physics.  The same artifact is present in
     every shipped prior (Z 1M 2.04e-05, Upsilon 1M 3.67e-04, **J/psi 500k
     4.13e-03**), so a stored-file gate at 1e-6 would reject every prior this
     project has ever used, including the ones its results rest on.

     So the gate is taken from the converter's check, which runs in float64 on the
     HepMC records before anything is cast, and is recorded per component in the
     manifest as `mass_check_worst_relative`.  The stored-file value is reported
     next to its float32 floor as information.  This is the `CLAUDE.md` section 4
     discipline: never quote a gate number without its reference.

  3. Merging scan.  Read from the scan report; not recomputed here.  A FAILED
     scan is **recorded and carried as a systematic, and is not blocking** --
     that is the decision rule this run operates under
     (`docs/unified_prior_cli_prompt_autonomous.md`): "no scale passes for a
     region -> use the method book rule value max(10, M/4), record
     `TMS_scan: FAILED`, carry the full spread as the systematic, continue.  This
     is a finding, not a blocker."

  4. Legacy containment.  Both the legacy and the new analysis windows must be
     applicable offline with a non-empty margin on every edge -- mass window,
     muon pT threshold and |eta|.  This is what lets these priors be compared
     against joint Runs D and E at all.

  5. Pair-pT against CMS: measured and recorded, **never corrected**.  This gate
     reports the disagreement; it does not require it to be small, because
     correcting it is exactly what method book section 13 forbids.

  6. Provenance: cards, versions and checksums present.

Usage:
    python scripts/prior_build/prior_gates.py --prior data/priors/z_*.hdf5 \\
        --region z --scan-report ~/mg5work/scan_report.json --json out.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

MUON_MASS = 0.105658
SIGNAL_WIDTH_MAX_MEV = 5.0      # gate 1, calibrated above
MASS_CONVENTION_TOL = 1e-6      # gate 2

REGION = {
    "jpsi": dict(
        analysis_pt=3.0, eta=2.4,
        window_new=(2.942, 3.252), window_legacy=(3.0369, 3.1569),
        signal_components={0: ("jpsi", 3.0969)},
        cms_cache="outputs/cms_Joint/.region_cache/jpsi/selected_split_4eb219e4bb8f537fe576.npz",
        cms_window=(3.0369, 3.1569), cms_pt=3.0),
    "z": dict(
        analysis_pt=25.0, eta=2.4,
        window_new=(83.70, 98.68), window_legacy=(70.0, 110.0),
        signal_components={},
        cms_cache="outputs/cms_Joint/.region_cache/z/selected_split_aec522b7204a5843ba5f.npz",
        cms_window=(70.0, 110.0), cms_pt=25.0),
    "upsilon": dict(
        analysis_pt=3.0, eta=2.4,
        window_new=(8.987, 10.873), window_legacy=(8.5, 11.5),
        signal_components={0: ("upsilon1s", 9.4604), 1: ("upsilon2s", 10.0234),
                           2: ("upsilon3s", 10.3551)},
        cms_csv="experiments/cms_upsilon/data/Ymumu.csv",
        cms_window=(8.5, 11.5), cms_pt=3.0),
}


def kinematics(z):
    z = np.asarray(z, dtype=np.float64)
    pt1 = np.hypot(z[:, 0], z[:, 1])
    pt2 = np.hypot(z[:, 4], z[:, 5])

    def eta(pz, pt):
        p = np.sqrt(pt ** 2 + pz ** 2)
        return 0.5 * np.log(np.clip((p + pz) / np.clip(p - pz, 1e-12, None), 1e-12, None))

    e_mom = (np.sqrt((z[:, 0:3] ** 2).sum(1) + MUON_MASS ** 2)
             + np.sqrt((z[:, 4:7] ** 2).sum(1) + MUON_MASS ** 2))
    p = z[:, 0:3] + z[:, 4:7]
    mass_p = np.sqrt(np.clip(e_mom ** 2 - (p ** 2).sum(1), 0.0, None))
    e_sto = z[:, 3] + z[:, 7]
    mass_e = np.sqrt(np.clip(e_sto ** 2 - (p ** 2).sum(1), 0.0, None))
    pair_pt = np.hypot(z[:, 0] + z[:, 4], z[:, 1] + z[:, 5])
    return dict(pt1=pt1, pt2=pt2, eta1=eta(z[:, 2], pt1), eta2=eta(z[:, 6], pt2),
                mass=mass_p, mass_stored=mass_e, pair_pt=pair_pt)


def robust_width_mev(mass):
    q75, q25 = np.percentile(mass, [75, 25])
    return float((q75 - q25) / 1.349 * 1000.0)


def load_cms(cfg):
    if "cms_cache" in cfg:
        d = np.load(cfg["cms_cache"])
        return np.concatenate([d["x_train"], d["x_val"], d["x_test"]]).astype(np.float64)
    rows = np.genfromtxt(cfg["cms_csv"], delimiter=",", names=True)
    # csv columns: E1,px1,py1,pz1 ... and Q1 gives the charge, so the mu- is the
    # one with Q = -1.  Written as (mu-, mu+) to match the prior's convention.
    out = np.empty((rows.size, 8))
    q1 = rows["Q1"]
    a = np.column_stack([rows["px1"], rows["py1"], rows["pz1"], rows["E1"]])
    b = np.column_stack([rows["px2"], rows["py2"], rows["pz2"], rows["E2"]])
    minus = np.where((q1 < 0)[:, None], a, b)
    plus = np.where((q1 < 0)[:, None], b, a)
    out[:, 0:4] = minus
    out[:, 4:8] = plus
    return out


def ks(a, b):
    a, b = np.sort(a), np.sort(b)
    allv = np.concatenate([a, b])
    d = float(np.abs(np.searchsorted(a, allv, "right") / a.size
                     - np.searchsorted(b, allv, "right") / b.size).max())
    return d, d * np.sqrt(a.size * b.size / (a.size + b.size))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prior", required=True)
    ap.add_argument("--region", required=True, choices=sorted(REGION))
    ap.add_argument("--scan-report", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    cfg = REGION[args.region]

    with h5py.File(args.prior, "r") as fh:
        z = fh["FDL/zData"][:]
        cid = fh["FDL/component_id"][:]
        attrs = {k: fh.attrs[k] for k in fh.attrs}
    k = kinematics(z)
    gates = {}
    print(f"PRIOR  {args.prior}")
    print(f"REGION {args.region}   N = {len(z)}   components "
          f"{ {int(c): int((cid == c).sum()) for c in np.unique(cid)} }")

    # ---------------- gate 1: signal width ----------------
    print("\n[gate 1] signal robust width  (threshold "
          f"{SIGNAL_WIDTH_MAX_MEV} MeV, calibrated -- see module docstring)")
    rows, ok1 = [], True
    if not cfg["signal_components"]:
        print("   region has no signal component (method book s12.2); "
              "gate NOT APPLICABLE")
        gates["signal_width"] = dict(status="NOT_APPLICABLE",
                                     reason="region ships one continuum component")
    else:
        for comp_id, (name, nominal) in cfg["signal_components"].items():
            sel = cid == comp_id
            if sel.sum() < 100:
                print(f"   {name:>10}: only {int(sel.sum())} events -- CANNOT EVALUATE")
                rows.append(dict(component=name, status="CANNOT_EVALUATE",
                                 n=int(sel.sum())))
                ok1 = False
                continue
            w = robust_width_mev(k["mass"][sel])
            good = w < SIGNAL_WIDTH_MAX_MEV
            ok1 &= good
            print(f"   {name:>10}: robust width {w:8.4f} MeV   "
                  f"std {k['mass'][sel].std()*1000:9.3f} MeV   "
                  f"{'PASS' if good else 'FAIL'}")
            rows.append(dict(component=name, robust_width_mev=w,
                             std_mev=float(k["mass"][sel].std() * 1000),
                             n=int(sel.sum()), status="PASS" if good else "FAIL"))
        gates["signal_width"] = dict(status="PASS" if ok1 else "FAIL",
                                     threshold_mev=SIGNAL_WIDTH_MAX_MEV, components=rows)

    # ---------------- gate 2: mass convention ----------------
    print("\n[gate 2] E-based vs p-based pair mass (massive-muon / lost-MM check)")
    # (a) the gate proper: the converter's float64 check on the HepMC records,
    #     recorded per component in the manifest.
    side = Path(args.prior).with_suffix(".json")
    conv_checks = {}
    if side.exists():
        man = json.loads(side.read_text())
        for name, comp in (man.get("per_component") or {}).items():
            src = comp.get("source_manifest") or {}
            for cname, c in (src.get("per_component") or {}).items():
                v = c.get("mass_check_worst_relative")
                if v is not None:
                    conv_checks[name] = float(v)
    if conv_checks:
        worst_conv = max(conv_checks.values())
        ok2 = worst_conv < MASS_CONVENTION_TOL
        for name, v in sorted(conv_checks.items()):
            print(f"   full precision, at conversion, {name:>12}: {v:.3e}")
        print(f"   worst {worst_conv:.3e}  (tolerance {MASS_CONVENTION_TOL:.0e})  "
              f"{'PASS' if ok2 else 'FAIL'}")
    else:
        worst_conv = None
        ok2 = False
        print("   NO CONVERTER CHECK FOUND IN THE MANIFEST -- cannot evaluate the "
              "gate at full precision.  FAIL (recorded as a gap, not as a physics "
              "failure).")
    # (b) information only: the stored float32 file, against its own round-off floor
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(k["mass"] - k["mass_stored"]) / np.maximum(k["mass"], 1e-12)
    worst_stored = float(np.nanmax(rel))
    zf = z.copy()
    z64 = np.asarray(z, dtype=np.float64)
    zf[:, 3] = np.sqrt((z64[:, 0:3] ** 2).sum(1) + MUON_MASS ** 2).astype(np.float32)
    zf[:, 7] = np.sqrt((z64[:, 4:7] ** 2).sum(1) + MUON_MASS ** 2).astype(np.float32)
    kf = kinematics(zf)
    with np.errstate(divide="ignore", invalid="ignore"):
        relf = np.abs(kf["mass"] - kf["mass_stored"]) / np.maximum(kf["mass"], 1e-12)
    floor = float(np.nanmax(relf))
    print(f"   stored float32 file: worst {worst_stored:.3e}, median "
          f"{float(np.nanmedian(rel)):.3e}")
    print(f"   float32 round-off floor (exactly on-shell muons): {floor:.3e}"
          f"   ratio {worst_stored/floor if floor else float('nan'):.3f}"
          "   <- 1.0 means the stored number is pure round-off")
    gates["mass_convention"] = dict(
        status="PASS" if ok2 else "FAIL",
        worst_relative_full_precision=worst_conv,
        per_component_full_precision=conv_checks,
        tolerance=MASS_CONVENTION_TOL,
        stored_float32_worst=worst_stored,
        stored_float32_roundoff_floor=floor,
        stored_over_floor=(worst_stored / floor) if floor else None,
        note="The gate is the full-precision check the converter runs on the HepMC "
             "records. The stored-file number is float32 round-off from the "
             "[N,8] float32 schema and is reported for reference only.")

    # ---------------- gate 3: merging scan ----------------
    print("\n[gate 3] merging scan")
    if args.scan_report and Path(args.scan_report).exists():
        rep = json.loads(Path(args.scan_report).read_text())
        r = rep.get(args.region, {})
        st = r.get("status", "MISSING")
        print(f"   scan status for this region: {st}")
        if "spread_acceptance_relative" in r:
            print(f"   systematic carried: acceptance spread "
                  f"{r['spread_acceptance_relative']*100:.2f}%, pair-pT median spread "
                  f"{r['spread_pair_pt_median_relative']*100:.2f}%")
        gates["merging_scan"] = dict(status=st, detail=r)
    else:
        print("   NO SCAN REPORT SUPPLIED")
        gates["merging_scan"] = dict(status="MISSING")

    # ---------------- gate 4: legacy containment ----------------
    print("\n[gate 4] legacy containment -- both windows applicable offline "
          "with a non-empty margin")
    base = ((k["pt1"] > cfg["analysis_pt"]) & (k["pt2"] > cfg["analysis_pt"])
            & (np.abs(k["eta1"]) < cfg["eta"]) & (np.abs(k["eta2"]) < cfg["eta"]))
    ok4, wrows = True, []
    for label, win in (("legacy", cfg["window_legacy"]), ("new", cfg["window_new"])):
        sel = base & (k["mass"] > win[0]) & (k["mass"] < win[1])
        n = int(sel.sum())
        # margin: events must exist on BOTH sides of each edge, otherwise the
        # window sits on the edge of the generated sample and migration into it
        # is not modelled.
        below = int((base & (k["mass"] <= win[0])).sum())
        above = int((base & (k["mass"] >= win[1])).sum())
        good = n > 0 and below > 0 and above > 0
        ok4 &= good
        print(f"   {label:>7} window {win[0]:8.4f}-{win[1]:8.4f}: {n:8d} events, "
              f"margin below {below:7d}, above {above:7d}   "
              f"{'PASS' if good else 'FAIL'}")
        wrows.append(dict(window=label, low=win[0], high=win[1], n_selected=n,
                          margin_below=below, margin_above=above,
                          status="PASS" if good else "FAIL"))
    # and the pT / eta edges
    pt_margin = int((base.sum() > 0) and
                    ((np.minimum(k["pt1"], k["pt2"]) < cfg["analysis_pt"]).sum() > 0))
    eta_margin = int((np.maximum(np.abs(k["eta1"]), np.abs(k["eta2"]))
                      >= cfg["eta"]).sum() > 0)
    print(f"   muon-pT threshold {cfg['analysis_pt']}: events below it exist -> "
          f"{'PASS' if pt_margin else 'FAIL'}")
    print(f"   |eta| cut {cfg['eta']}: events outside it exist -> "
          f"{'PASS' if eta_margin else 'FAIL'}")
    ok4 &= bool(pt_margin) and bool(eta_margin)
    gates["legacy_containment"] = dict(status="PASS" if ok4 else "FAIL", windows=wrows,
                                       pt_margin=bool(pt_margin),
                                       eta_margin=bool(eta_margin))

    # ---------------- gate 5: pair pT vs CMS, measured not corrected ----------------
    print("\n[gate 5] pair pT against CMS -- MEASURED AND RECORDED, NOT CORRECTED")
    try:
        xc = load_cms(cfg)
        kc = kinematics(xc)
        csel = ((kc["pt1"] > cfg["cms_pt"]) & (kc["pt2"] > cfg["cms_pt"])
                & (np.abs(kc["eta1"]) < cfg["eta"]) & (np.abs(kc["eta2"]) < cfg["eta"])
                & (kc["mass"] > cfg["cms_window"][0]) & (kc["mass"] < cfg["cms_window"][1]))
        psel = (base & (k["mass"] > cfg["cms_window"][0])
                & (k["mass"] < cfg["cms_window"][1]))
        pc, pp = kc["pair_pt"][csel], k["pair_pt"][psel]
        d, sig = ks(pp, pc)
        print(f"   CMS events {pc.size}, prior events {pp.size} "
              f"(window {cfg['cms_window']}, pT > {cfg['cms_pt']})")
        print(f"   {'quantile':>10} {'prior':>10} {'CMS':>10} {'ratio':>8}")
        for q in (5, 25, 50, 75, 95):
            a, b = np.percentile(pp, q), np.percentile(pc, q)
            print(f"   {'p'+str(q):>10} {a:10.4f} {b:10.4f} {a/b:8.3f}")
        for t in (5.0, 10.0, 22.8):
            print(f"   frac below {t:5.1f} GeV: prior {np.mean(pp<t):.4f}   "
                  f"CMS {np.mean(pc<t):.4f}")
        print(f"   pair-pT KS D = {d:.5f} ({sig:.1f} sigma).  RECORDED, NOT CORRECTED.")
        gates["pair_pt_vs_cms"] = dict(
            status="MEASURED", ks_d=d, ks_sigma=float(sig),
            n_cms=int(pc.size), n_prior=int(pp.size),
            prior_median=float(np.median(pp)), cms_median=float(np.median(pc)),
            median_ratio=float(np.median(pp) / np.median(pc)),
            prior_frac_below_5=float(np.mean(pp < 5.0)),
            cms_frac_below_5=float(np.mean(pc < 5.0)))
    except Exception as exc:
        print(f"   CANNOT EVALUATE: {exc}")
        gates["pair_pt_vs_cms"] = dict(status="CANNOT_EVALUATE", reason=str(exc))

    # ---------------- gate 6: provenance ----------------
    print("\n[gate 6] provenance attributes present")
    required = ["truth_variant", "generation_cuts", "component_id_mapping",
                "component_counts", "daughter_muon_mass_GeV", "converter",
                "components_are_mixed", "reweighting_applied",
                "artificial_smearing_applied"]
    missing = [r for r in required if r not in attrs]
    ok6 = not missing
    print(f"   {'all present' if ok6 else 'MISSING: ' + ', '.join(missing)}   "
          f"{'PASS' if ok6 else 'FAIL'}")
    gates["provenance"] = dict(status="PASS" if ok6 else "FAIL", missing=missing)

    # A failed merging scan is a recorded finding carried as a systematic, not a
    # blocker -- see the gate 3 note above.  Everything else blocks.
    NON_BLOCKING = {"merging_scan"}
    blocking = [n for n, g in gates.items()
                if g["status"] not in ("PASS", "MEASURED", "NOT_APPLICABLE")
                and n not in NON_BLOCKING]
    overall = "PASS" if not blocking else "REJECTED"
    print(f"\nOVERALL: {overall}" + (f"   (blocking: {', '.join(blocking)})" if blocking else ""))
    if args.json:
        Path(args.json).write_text(json.dumps(
            dict(prior=args.prior, region=args.region, overall=overall,
                 blocking=blocking, gates=gates), indent=2))
        print(f"wrote {args.json}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
