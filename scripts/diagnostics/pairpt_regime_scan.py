#!/usr/bin/env python3
"""Stage 1 of docs/unified_prior_cli_prompt.md: measure the two merging-scale
regime hypotheses of docs/unified_prior_method.md section 5.1 against the priors
already on disk, before any generation.

Read-only.  Nothing under data/ or outputs/ is modified.

What it measures, per region, after applying that region's recorded analysis
selection:
  1. pair-pT quantiles and the fraction below 5.0 / 10.0 / 22.8 / 45.6 GeV;
  2. the pair-pT floor (minimum and p01) against the analytic kinematic floor;
  3. a kink test at the generated ptj = Merging:TMS = 5.0 GeV, done as a
     blind-window fit with a control-pull check -- a fit whose control rms pull
     is far from 1 cannot support any statement about the blind window;
  4. per component where FDL/component_id exists, including a signal-vs-continuum
     pair-pT KS, which is the assumption the section-5 TMS scan rests on.

Usage:  python scripts/diagnostics/pairpt_regime_scan.py [--json OUT]
"""
import argparse, json
import h5py
import numpy as np

THRESHOLDS = (5.0, 10.0, 22.8, 45.6)

REGIONS = {
    "jpsi": dict(
        path="data/cms_jpsi_mumu_mg5py8_ckkwl_8tev_inclusive_0j1j_fiducial_3p0369_3p1569_500k.hdf5",
        mass=3.0969, pt_min=3.0, eta_max=2.4, mass_window=(3.0369, 3.1569),
        components={0: "jpsi", 1: "continuum"}, continuum_id=1,
    ),
    "upsilon": dict(
        path="data/cms_upsilon_mumu_mg5py8_ckkwl_8tev_inclusive_0j1j_fiducial_8p5_11p5_1M.hdf5",
        mass=9.4604, pt_min=3.0, eta_max=2.4, mass_window=(8.5, 11.5),
        components={0: "upsilon1s", 1: "upsilon2s", 2: "upsilon3s", 3: "continuum"},
        continuum_id=3,
    ),
    "z": dict(
        path="data/cms_dymumu_mg5py8_ckkwl_8tev_inclusive_0j1j_fiducial_70_110_1M.hdf5",
        mass=91.1876, pt_min=25.0, eta_max=2.4, mass_window=(70.0, 110.0),
        components=None, continuum_id=None,
    ),
}


def kinematics(z):
    z = z.astype(np.float64)
    minus, plus = z[:, 0:4], z[:, 4:8]
    total = minus + plus

    def pt(v):
        return np.hypot(v[:, 0], v[:, 1])

    def eta(v, v_pt):
        pz = v[:, 2]
        p = np.sqrt(v_pt ** 2 + pz ** 2)
        return 0.5 * np.log(np.clip((p + pz) / np.clip(p - pz, 1e-12, None), 1e-12, None))

    pt1, pt2 = pt(minus), pt(plus)
    m2 = total[:, 3] ** 2 - total[:, 0] ** 2 - total[:, 1] ** 2 - total[:, 2] ** 2
    return dict(pt1=pt1, pt2=pt2, eta1=eta(minus, pt1), eta2=eta(plus, pt2),
                pair_pt=pt(total), mass=np.sqrt(np.clip(m2, 0.0, None)))


def kinematic_floor(mass, pt_min):
    """Minimum pair pT for a two-body decay of `mass` with both muons above
    `pt_min`.  Pair pT = 0 needs pT1 = pT2 = p, dPhi = 180, so
    M^2 = 2 p^2 (cosh dEta + 1), maximised at dEta = 0 with p = M/2 = p*.
    So the floor is zero iff p* >= pt_min; otherwise it sits at
    pT1 = pT2 = pt_min, dEta = 0."""
    p_star = 0.5 * mass
    if p_star >= pt_min:
        return 0.0, p_star
    cos_dphi = 1.0 - mass ** 2 / (2.0 * pt_min ** 2)
    return float(np.sqrt(2.0 * pt_min ** 2 * (1.0 + cos_dphi))), p_star


def ks_two_sample(a, b):
    a, b = np.sort(a), np.sort(b)
    allv = np.concatenate([a, b])
    ca = np.searchsorted(a, allv, "right") / a.size
    cb = np.searchsorted(b, allv, "right") / b.size
    d = float(np.abs(ca - cb).max())
    n_eff = a.size * b.size / (a.size + b.size)
    return d, d * np.sqrt(n_eff)


def kink_test(pair_pt, blind=(4.5, 5.75), fit_range=(2.5, 15.0), degree=5, min_count=20):
    """Blind-window smoothness test in log(count) vs log(pair pT).

    The control rms pull is reported alongside the blind-window pull and must be
    read first: a stiff fit produces large blind-window pulls that are fit
    systematics, not a merging hole.

    The test is refused outright when the sample's own kinematic turn-on falls
    inside the blind window, because an acceptance edge and a merging hole are
    then the same measurement.  That is the J/psi case."""
    if pair_pt.min() > blind[0]:
        return dict(applicable=False,
                    reason=f"sample minimum {pair_pt.min():.3f} GeV lies inside the blind "
                           f"window {blind}; the acceptance turn-on and a merging hole are "
                           f"degenerate here")
    edges = np.arange(fit_range[0], fit_range[1] + 1e-9, 0.25)
    counts, _ = np.histogram(pair_pt, bins=edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    counts = counts.astype(float)
    usable = counts > min_count
    in_blind = (centres >= blind[0]) & (centres <= blind[1])
    fit_mask = usable & ~in_blind
    if fit_mask.sum() < degree + 3:
        return dict(applicable=False, reason="too few usable bins outside the blind window",
                    n_fit_bins=int(fit_mask.sum()))
    x = np.log(centres)
    coef = np.polyfit(x[fit_mask], np.log(counts[fit_mask]), degree)
    pred = np.exp(np.polyval(coef, x))
    pull = (counts - pred) / np.sqrt(np.maximum(pred, 1.0))
    blind_mask = usable & in_blind
    return dict(applicable=True, degree=degree, blind=list(blind),
                control_rms_pull=float(pull[fit_mask].std()),
                n_control_bins=int(fit_mask.sum()),
                n_blind_bins=int(blind_mask.sum()),
                blind_mean_pull=float(pull[blind_mask].mean()) if blind_mask.any() else None,
                blind_pulls=[float(v) for v in pull[blind_mask]],
                centres=centres.tolist(), counts=counts.tolist())


def summarise(pair_pt, label, n_reference):
    q = np.percentile(pair_pt, [1, 5, 25, 50, 75, 99])
    row = dict(label=label, n=int(pair_pt.size),
               fraction_of_selected=float(pair_pt.size / max(n_reference, 1)),
               minimum=float(pair_pt.min()),
               p01=float(q[0]), p05=float(q[1]), p25=float(q[2]),
               median=float(q[3]), p75=float(q[4]), p99=float(q[5]))
    for t in THRESHOLDS:
        row[f"fraction_below_{t}"] = float((pair_pt < t).mean())
    return row


def run_region(name, cfg):
    with h5py.File(cfg["path"], "r") as f:
        z = f["FDL/zData"][:]
        cid = f["FDL/component_id"][:] if "component_id" in f["FDL"] else None
    k = kinematics(z)
    lo, hi = cfg["mass_window"]
    sel = ((k["pt1"] > cfg["pt_min"]) & (k["pt2"] > cfg["pt_min"])
           & (np.abs(k["eta1"]) < cfg["eta_max"]) & (np.abs(k["eta2"]) < cfg["eta_max"])
           & (k["mass"] > lo) & (k["mass"] < hi))
    pair_pt = k["pair_pt"][sel]
    floor, p_star = kinematic_floor(cfg["mass"], cfg["pt_min"])
    rows = [summarise(pair_pt, "ALL", pair_pt.size)]
    ks_rows = []
    if cid is not None:
        for cid_value, comp_name in cfg["components"].items():
            mask = sel & (cid == cid_value)
            if mask.sum() > 50:
                rows.append(summarise(k["pair_pt"][mask], comp_name, pair_pt.size))
        cont = k["pair_pt"][sel & (cid == cfg["continuum_id"])]
        for cid_value, comp_name in cfg["components"].items():
            if cid_value == cfg["continuum_id"]:
                continue
            sig = k["pair_pt"][sel & (cid == cid_value)]
            if sig.size > 50:
                d, stat = ks_two_sample(sig, cont)
                ks_rows.append(dict(component=comp_name, n=int(sig.size),
                                    ks_d=d, ks_d_sqrt_neff=float(stat)))
    mass = k["mass"][sel]
    return dict(
        path=cfg["path"], file_n=int(z.shape[0]), n_selected=int(sel.sum()),
        selection_survival=float(sel.mean()),
        selection=dict(muon_pt_min=cfg["pt_min"], muon_abs_eta_max=cfg["eta_max"],
                       mass_window=list(cfg["mass_window"])),
        p_star=float(p_star), kinematic_pair_pt_floor=floor,
        rows=rows, signal_vs_continuum_ks=ks_rows,
        kink=kink_test(pair_pt),
        mass=dict(mean=float(mass.mean()), std_MeV=float(mass.std() * 1000.0),
                  robust_width_MeV=float((np.percentile(mass, 75) - np.percentile(mass, 25))
                                         / 1.349 * 1000.0)),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", help="write the full result, including kink histograms, here")
    args = ap.parse_args()

    out = {}
    for name, cfg in REGIONS.items():
        r = run_region(name, cfg)
        out[name] = r
        print("=" * 96)
        print(f"{name}:  {r['path']}")
        print(f"  analysis selection {r['selection']}")
        print(f"  survives re-application: {r['n_selected']} / {r['file_n']} "
              f"= {r['selection_survival']:.6f}")
        print(f"  p* = M/2 = {r['p_star']:.3f} GeV;  analytic kinematic pair-pT floor "
              f"= {r['kinematic_pair_pt_floor']:.3f} GeV")
        head = ["component", "n", "frac", "min", "p01", "p05", "p25", "median", "p75", "p99"] \
               + [f"<{t}" for t in THRESHOLDS]
        print("  " + " ".join(f"{h:>10s}" for h in head))
        for row in r["rows"]:
            cells = [row["label"], str(row["n"]), f"{row['fraction_of_selected']:.4f}",
                     f"{row['minimum']:.3f}", f"{row['p01']:.3f}", f"{row['p05']:.3f}",
                     f"{row['p25']:.3f}", f"{row['median']:.3f}", f"{row['p75']:.3f}",
                     f"{row['p99']:.3f}"] \
                    + [f"{row[f'fraction_below_{t}']:.4f}" for t in THRESHOLDS]
            print("  " + " ".join(f"{c:>10s}" for c in cells))
        for ks_row in r["signal_vs_continuum_ks"]:
            print(f"  signal-vs-continuum pair-pT KS: {ks_row['component']:>10s} "
                  f"D={ks_row['ks_d']:.4f}  D*sqrt(n_eff)={ks_row['ks_d_sqrt_neff']:.2f}")
        kink = r["kink"]
        if kink["applicable"]:
            print(f"  kink test at ptj = TMS = 5.0: control rms pull={kink['control_rms_pull']:.2f} "
                  f"(n={kink['n_control_bins']}), blind-window mean pull="
                  f"{kink['blind_mean_pull']:+.2f} (n={kink['n_blind_bins']})")
        else:
            print(f"  kink test NOT APPLICABLE: {kink['reason']}")
        print(f"  mass: mean={r['mass']['mean']:.5f} GeV  std={r['mass']['std_MeV']:.2f} MeV  "
              f"robust(IQR/1.349)={r['mass']['robust_width_MeV']:.2f} MeV")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
