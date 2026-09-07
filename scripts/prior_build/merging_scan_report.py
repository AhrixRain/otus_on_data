#!/usr/bin/env python3
"""Stage 4: score the merging-scale insensitivity scan.

Method book section 5: "Generate at 0.5x, 1x and 2x TMS.  Require the merged
pair-pT spectrum and the fiducial acceptance to agree within statistical
uncertainty across the three.  Take the central value; carry the spread as a
systematic."

Two things this script is careful about.

**Agreement is judged against a same-TMS floor, not against zero.**  Two
independent 100k samples of the *same* distribution do not give KS D = 0.  So a
raw D carries no claim on its own -- the same lesson as `CLAUDE.md` section 4.
The floor is estimated by splitting each sample in half and comparing the halves,
which costs nothing and makes "within statistical uncertainty" a measured
statement rather than an assertion.

**The Upsilon signal cross-check is a derivative, not a similarity.**  The
assumption the scan rests on is that d(spectrum)/d(TMS) is common to signal and
continuum.  Similar spectra support that but do not prove it, so the Upsilon
region also has signal runs at 0.5x and 2x TMS and this script compares the
*shift* between them against the continuum's shift over the same interval.

Usage:
    python scripts/prior_build/merging_scan_report.py --dir ~/mg5work/converted
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

MUON_MASS = 0.105658

# The analysis selection each region's fiducial acceptance is measured against.
# The LOOSEST window that may be applied offline, so the acceptance is the one
# that actually gates Stage 5's event counts.
ANALYSIS = {
    "jpsi": dict(pt=3.0, eta=2.4, window=(2.942, 3.252)),
    "z": dict(pt=25.0, eta=2.4, window=(70.0, 110.0)),
    "upsilon": dict(pt=3.0, eta=2.4, window=(8.5, 11.5)),
}

SCAN = {
    "jpsi": ("scan_jpsi_cont_tms{}", [5.0, 10.0, 20.0], 10.0),
    "z": ("scan_z_cont_tms{}", [5.0, 11.4, 22.8, 45.6], 22.8),
    "upsilon": ("scan_ups_cont_tms{}", [5.0, 10.0, 20.0], 10.0),
}
UPSILON_SIGNAL = ("scan_ups1s_tms{}", [5.0, 20.0])


def kinematics(z):
    z = np.asarray(z, dtype=np.float64)
    pt1 = np.hypot(z[:, 0], z[:, 1])
    pt2 = np.hypot(z[:, 4], z[:, 5])

    def eta(pz, pt):
        p = np.sqrt(pt ** 2 + pz ** 2)
        return 0.5 * np.log(np.clip((p + pz) / np.clip(p - pz, 1e-12, None), 1e-12, None))

    e = (np.sqrt((z[:, 0:3] ** 2).sum(1) + MUON_MASS ** 2)
         + np.sqrt((z[:, 4:7] ** 2).sum(1) + MUON_MASS ** 2))
    p = z[:, 0:3] + z[:, 4:7]
    mass = np.sqrt(np.clip(e ** 2 - (p ** 2).sum(1), 0.0, None))
    pair_pt = np.hypot(z[:, 0] + z[:, 4], z[:, 1] + z[:, 5])
    return pt1, pt2, eta(z[:, 2], pt1), eta(z[:, 6], pt2), pair_pt, mass


def load(path, region):
    with h5py.File(path, "r") as fh:
        z = fh["FDL/zData"][:]
    pt1, pt2, e1, e2, pair_pt, mass = kinematics(z)
    cut = ANALYSIS[region]
    sel = ((pt1 > cut["pt"]) & (pt2 > cut["pt"])
           & (np.abs(e1) < cut["eta"]) & (np.abs(e2) < cut["eta"])
           & (mass > cut["window"][0]) & (mass < cut["window"][1]))
    return dict(n_converted=len(z), acceptance=float(sel.mean()),
                n_selected=int(sel.sum()), pair_pt=pair_pt[sel])


def ks(a, b):
    a, b = np.sort(a), np.sort(b)
    allv = np.concatenate([a, b])
    d = float(np.abs(np.searchsorted(a, allv, "right") / a.size
                     - np.searchsorted(b, allv, "right") / b.size).max())
    return d, d * np.sqrt(a.size * b.size / (a.size + b.size))


def same_sample_floor(x, rng, trials=6):
    """KS between two halves of the same sample: what 'no difference' looks like."""
    out = []
    for _ in range(trials):
        idx = rng.permutation(x.size)
        half = x.size // 2
        out.append(ks(x[idx[:half]], x[idx[half:2 * half]])[0])
    return float(np.mean(out)), float(np.std(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(Path.home() / "mg5work" / "converted"))
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    root = Path(args.dir).expanduser()
    rng = np.random.default_rng(20260906)
    report = {}

    for region, (pattern, points, rule) in SCAN.items():
        print("=" * 92)
        print(f"REGION {region}  (continuum component; method book rule TMS = {rule})")
        data = {}
        for tms in points:
            path = root / (pattern.format(tms) + ".hdf5")
            if not path.exists():
                print(f"  TMS {tms}: MISSING {path}")
                continue
            data[tms] = load(path, region)
        if len(data) < 2:
            print("  fewer than two scan points present; scan NOT EVALUABLE")
            report[region] = dict(status="NOT_EVALUABLE")
            continue

        print(f"  {'TMS':>7} {'converted':>10} {'selected':>9} {'acceptance':>11} "
              f"{'ppT median':>11} {'ppT p25':>9} {'ppT p75':>9} {'frac<5':>8}")
        for tms, d in data.items():
            p = d["pair_pt"]
            print(f"  {tms:7.1f} {d['n_converted']:10d} {d['n_selected']:9d} "
                  f"{d['acceptance']:11.4f} {np.median(p):11.4f} "
                  f"{np.percentile(p,25):9.4f} {np.percentile(p,75):9.4f} "
                  f"{np.mean(p<5.0):8.4f}")

        ref = rule if rule in data else sorted(data)[len(data) // 2]
        floor, floor_sd = same_sample_floor(data[ref]["pair_pt"], rng)
        print(f"\n  same-TMS KS floor at TMS {ref} (two halves of one sample): "
              f"D = {floor:.5f} +- {floor_sd:.5f}")
        print(f"  {'TMS':>7} {'KS D vs ref':>12} {'D / floor':>10} "
              f"{'acceptance':>11} {'acc. shift':>11} {'sigma':>8}")
        acc_ref = data[ref]["acceptance"]
        n_ref = data[ref]["n_converted"]
        rows = []
        for tms, d in data.items():
            d_ks, _ = ks(d["pair_pt"], data[ref]["pair_pt"])
            ratio = d_ks / floor if floor > 0 else float("nan")
            dacc = d["acceptance"] - acc_ref
            sig = np.sqrt(d["acceptance"] * (1 - d["acceptance"]) / d["n_converted"]
                          + acc_ref * (1 - acc_ref) / n_ref)
            nsig = dacc / sig if sig > 0 else 0.0
            print(f"  {tms:7.1f} {d_ks:12.5f} {ratio:10.2f} {d['acceptance']:11.4f} "
                  f"{dacc:+11.4f} {nsig:+8.2f}")
            rows.append(dict(tms=tms, ks_d=d_ks, ks_over_floor=ratio,
                             acceptance=d["acceptance"], acc_shift=dacc,
                             acc_sigma=float(nsig),
                             pair_pt_median=float(np.median(d["pair_pt"])),
                             n_converted=d["n_converted"], n_selected=d["n_selected"]))

        accs = [d["acceptance"] for d in data.values()]
        meds = [float(np.median(d["pair_pt"])) for d in data.values()]
        spread_acc = (max(accs) - min(accs)) / np.mean(accs)
        spread_med = (max(meds) - min(meds)) / np.mean(meds)
        insensitive = all(r["ks_over_floor"] < 3.0 for r in rows) and \
                      all(abs(r["acc_sigma"]) < 3.0 for r in rows)
        print(f"\n  spread across scan points: acceptance {spread_acc*100:.2f}% relative, "
              f"pair-pT median {spread_med*100:.2f}% relative")
        print(f"  VERDICT: {'INSENSITIVE (scan passes)' if insensitive else 'SENSITIVE (scan FAILS)'}")
        report[region] = dict(status="PASS" if insensitive else "FAIL",
                              reference_tms=ref, ks_floor=floor, rows=rows,
                              spread_acceptance_relative=float(spread_acc),
                              spread_pair_pt_median_relative=float(spread_med))

    # ---- the Upsilon signal cross-check: is d(spectrum)/d(TMS) common? ----
    print("=" * 92)
    print("UPSILON signal-vs-continuum TMS RESPONSE (method book s5: turns the "
          "scan's assumption into a measurement)")
    pat, pts = UPSILON_SIGNAL
    sig = {}
    for tms in pts:
        p = root / (pat.format(tms) + ".hdf5")
        if p.exists():
            sig[tms] = load(p, "upsilon")
    cont = {}
    for tms in pts:
        p = root / (SCAN["upsilon"][0].format(tms) + ".hdf5")
        if p.exists():
            cont[tms] = load(p, "upsilon")
    if len(sig) == 2 and len(cont) == 2:
        lo, hi = pts
        d_sig, _ = ks(sig[lo]["pair_pt"], sig[hi]["pair_pt"])
        d_con, _ = ks(cont[lo]["pair_pt"], cont[hi]["pair_pt"])
        f_sig, _ = same_sample_floor(sig[hi]["pair_pt"], rng)
        f_con, _ = same_sample_floor(cont[hi]["pair_pt"], rng)
        print(f"  response to TMS {lo} -> {hi}, as KS between the two TMS points:")
        print(f"    signal    D = {d_sig:.5f}  (same-TMS floor {f_sig:.5f}, "
              f"ratio {d_sig/f_sig:.2f})")
        print(f"    continuum D = {d_con:.5f}  (same-TMS floor {f_con:.5f}, "
              f"ratio {d_con/f_con:.2f})")
        for name, dd in (("signal", sig), ("continuum", cont)):
            m_lo = np.median(dd[lo]["pair_pt"]); m_hi = np.median(dd[hi]["pair_pt"])
            a_lo = dd[lo]["acceptance"]; a_hi = dd[hi]["acceptance"]
            print(f"    {name:>9}: pair-pT median {m_lo:.4f} -> {m_hi:.4f} "
                  f"({(m_hi-m_lo)/m_lo*100:+.2f}%)   acceptance {a_lo:.4f} -> {a_hi:.4f} "
                  f"({(a_hi-a_lo)/a_lo*100:+.2f}%)")
        report["upsilon_signal_response"] = dict(
            ks_signal=d_sig, ks_continuum=d_con,
            floor_signal=f_sig, floor_continuum=f_con,
            signal_median_shift_pct=float((np.median(sig[hi]["pair_pt"])
                                           - np.median(sig[lo]["pair_pt"]))
                                          / np.median(sig[lo]["pair_pt"]) * 100),
            continuum_median_shift_pct=float((np.median(cont[hi]["pair_pt"])
                                              - np.median(cont[lo]["pair_pt"]))
                                             / np.median(cont[lo]["pair_pt"]) * 100))
    else:
        print("  signal or continuum scan points missing; cross-check NOT EVALUABLE")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
