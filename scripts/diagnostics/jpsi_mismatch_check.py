#!/usr/bin/env python
"""One-off diagnostic: CMS data vs MG5 prior mismatch inside the J/psi ~3.09 GeV region.

Reuses the version-3 split caches in outputs/cms_JpsiDoubleMuons/archive/.plot_cache:
  - 0ed04817d72bb34f84ed : v3.9 signal-region selection (muon pT in (2,100], |eta|<2.4,
                           mass in [3.0369,3.1569]) + trigger-matched filtered prior
  - 410ce075cc7cc2e067c8 : full window [2.6,3.5], same muon pT/eta cuts, no prior filter

Prints: prior mass delta-ness, data mass resolution/continuum decomposition in the
window, kinematic support comparison, and joint-support coverage of data by prior.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.physics import invariant_mass_np  # noqa: E402

CACHE = ROOT / "outputs/cms_JpsiDoubleMuons/archive/.plot_cache"
PRIOR = ROOT / "data/cms_jpsi_mumu_mg5_8tev_1M.hdf5"

COLS = ["mu- px", "mu- py", "mu- pz", "mu- E", "mu+ px", "mu+ py", "mu+ pz", "mu+ E"]


def load_npz(key: str) -> dict[str, np.ndarray]:
    with np.load(CACHE / f"selected_split_{key}.npz", allow_pickle=False) as d:
        return {k: np.asarray(d[k], dtype=np.float64) for k in d.keys()}


def mass_e(z: np.ndarray) -> np.ndarray:
    m2 = (z[:, 3] + z[:, 7]) ** 2 - (z[:, 0] + z[:, 4]) ** 2 - (z[:, 1] + z[:, 5]) ** 2 - (z[:, 2] + z[:, 6]) ** 2
    return np.sqrt(np.maximum(m2, 0.0))


def stats(name: str, a: np.ndarray) -> None:
    q = np.percentile(a, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    print(f"    {name:22s} mean={a.mean():10.3f} std={a.std():9.3f} "
          f"p01={q[1]:8.3f} p05={q[2]:8.3f} med={q[4]:8.3f} "
          f"p95={q[6]:8.3f} p99={q[7]:8.3f} max={q[8]:9.3f}")


def main() -> None:
    print("=" * 80)
    print("J/PSI ~3.09 GeV REGION: CMS DATA vs MG5 PRIOR MISMATCH CHECK")
    print("=" * 80)

    # ---------------- PRIOR ----------------
    from scripts.cms_data import load_theory_prior_z
    zfull = load_theory_prior_z(PRIOR).astype(np.float64)
    print(f"\n[1] MG5 signal prior, full file (n={len(zfull):,}):")
    m_e = mass_e(zfull)
    m_p = invariant_mass_np(zfull, daughter_masses=None, stable=True)
    print(f"    mass E-based: mean={m_e.mean():.5f}  std={m_e.std()*1000:.2f} MeV  "
          f"min={m_e.min():.5f}  max={m_e.max():.5f}")
    print(f"    mass p-based: mean={m_p.mean():.5f}  std={m_p.std()*1000:.2f} MeV  "
          f"min={m_p.min():.5f}  max={m_p.max():.5f}")
    frac_win = float(np.mean((m_p > 3.0369) & (m_p < 3.1569)))
    print(f"    frac with p-based mass in [3.0369,3.1569]: {frac_win*100:.2f}%")

    cache = load_npz("0ed04817d72bb34f84ed")
    z = np.concatenate([cache["z_train"], cache["z_val"], cache["z_test"]])
    print(f"\n[2] Filtered prior (trigger-matched: pT>3, |eta|<2.4, mass window; n={len(z):,}):")
    m_ze = mass_e(z)
    m_zp = invariant_mass_np(z, daughter_masses=None, stable=True)
    print(f"    mass E-based: mean={m_ze.mean():.5f}  std={m_ze.std()*1000:.2f} MeV")
    print(f"    mass p-based: mean={m_zp.mean():.5f}  std={m_zp.std()*1000:.2f} MeV")

    # ---------------- DATA: signal window ----------------
    xsig = np.concatenate([cache["x_train"], cache["x_val"], cache["x_test"]])
    msig = invariant_mass_np(xsig, daughter_masses=None, stable=True)
    print(f"\n[3] CMS data, signal window [3.0369,3.1569], muon pT in (2,100] (n={len(xsig):,}):")
    print(f"    mass: mean={msig.mean():.5f}  std={msig.std()*1000:.1f} MeV  median={np.median(msig):.5f}")
    hist, edges = np.histogram(msig, bins=60, range=(3.0369, 3.1569))
    peak = (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1]) / 2
    print(f"    peak bin center (60 bins/120 MeV): {peak:.4f}")
    q = np.percentile(msig, [16, 50, 84])
    print(f"    quantiles 16/50/84%: {q[0]:.5f} / {q[1]:.5f} / {q[2]:.5f}  "
          f"(q84-q16)/2 = {(q[2]-q[0])/2*1000:.1f} MeV")

    # ---------------- DATA: full window, sideband decomposition ----------------
    full = load_npz("410ce075cc7cc2e067c8")
    xfull = np.concatenate([full["x_train"], full["x_val"], full["x_test"]])
    mfull = invariant_mass_np(xfull, daughter_masses=None, stable=True)
    print(f"\n[4] CMS data, full window [2.6,3.5] (n={len(xfull):,}):")
    n_lo = np.sum(mfull < 3.0369)
    n_sig = np.sum((mfull >= 3.0369) & (mfull <= 3.1569))
    n_hi = np.sum(mfull > 3.1569)
    print(f"    low sideband [2.6,3.0369): {n_lo:,} ({n_lo/len(xfull)*100:.1f}%)")
    print(f"    signal region [3.0369,3.1569]: {n_sig:,} ({n_sig/len(xfull)*100:.1f}%)")
    print(f"    high sideband (3.1569,3.5]: {n_hi:,} ({n_hi/len(xfull)*100:.1f}%)")
    bkg_density = (n_lo / (3.0369 - 2.6) + n_hi / (3.5 - 3.1569)) / 2
    bkg_under = bkg_density * 0.12
    sig_under = n_sig - bkg_under
    print(f"    linear sideband est.: bkg density {bkg_density:.0f}/GeV -> "
          f"{bkg_under:.0f} continuum events under the signal region")
    print(f"    => signal frac in [3.0369,3.1569]: {sig_under/n_sig*100:.1f}%  "
          f"| continuum frac: {bkg_under/n_sig*100:.1f}%")
    core = (mfull > 3.0869) & (mfull < 3.1069)
    wings = (mfull >= 3.0369) & (mfull <= 3.1569) & ~core
    print(f"    data in +/-10 MeV of 3.0969: {np.sum(core):,} ({np.sum(core)/n_sig*100:.1f}% of signal region)")
    print(f"    data in window wings: {np.sum(wings):,} ({np.sum(wings)/n_sig*100:.1f}% of signal region)")

    # ---------------- kinematics inside the signal window ----------------
    print("\n[5] Kinematics inside the signal window (data vs filtered prior):")
    pt1 = np.hypot(xsig[:, 0], xsig[:, 1]); pt2 = np.hypot(xsig[:, 4], xsig[:, 5])
    eta1 = np.arcsinh(xsig[:, 2] / np.maximum(pt1, 1e-9)); eta2 = np.arcsinh(xsig[:, 6] / np.maximum(pt2, 1e-9))
    pair_pt = np.hypot(xsig[:, 0] + xsig[:, 4], xsig[:, 1] + xsig[:, 5])
    pair_e = xsig[:, 3] + xsig[:, 7]; pair_pz = xsig[:, 2] + xsig[:, 6]
    pair_rap = 0.5 * np.log((pair_e + pair_pz) / np.maximum(pair_e - pair_pz, 1e-9))
    print("  --- CMS data (signal window) ---")
    stats("muon pT [GeV]", np.concatenate([pt1, pt2]))
    stats("pair pT [GeV]", pair_pt)
    stats("pair rapidity", pair_rap)
    stats("muon |eta|", np.abs(np.concatenate([eta1, eta2])))
    print("  --- filtered MG5 prior (5,995 ev) ---")
    zpt1 = np.hypot(z[:, 0], z[:, 1]); zpt2 = np.hypot(z[:, 4], z[:, 5])
    zeta1 = np.arcsinh(z[:, 2] / np.maximum(zpt1, 1e-9)); zeta2 = np.arcsinh(z[:, 6] / np.maximum(zpt2, 1e-9))
    zpair_pt = np.hypot(z[:, 0] + z[:, 4], z[:, 1] + z[:, 5])
    zpair_e = z[:, 3] + z[:, 7]; zpair_pz = z[:, 2] + z[:, 6]
    zpair_rap = 0.5 * np.log((zpair_e + zpair_pz) / np.maximum(zpair_e - zpair_pz, 1e-9))
    stats("muon pT [GeV]", np.concatenate([zpt1, zpt2]))
    stats("pair pT [GeV]", zpair_pt)
    stats("pair rapidity", zpair_rap)
    stats("muon |eta|", np.abs(np.concatenate([zeta1, zeta2])))

    # ---------------- per-coordinate support coverage ----------------
    print("\n[6] Per-coordinate support coverage (data vs filtered prior):")
    lo = np.percentile(z, 0.5, axis=0)
    hi = np.percentile(z, 99.5, axis=0)
    inside = np.all((xsig >= lo) & (xsig <= hi), axis=1)
    print(f"    data events fully inside prior [p0.5,p99.5] box: {np.sum(inside):,} / {len(xsig):,} ({np.mean(inside)*100:.1f}%)")
    for i, name in enumerate(COLS):
        frac_in = np.mean((xsig[:, i] >= lo[i]) & (xsig[:, i] <= hi[i]))
        print(f"    {name:8s} prior[{lo[i]:8.2f},{hi[i]:8.2f}]  data in range: {frac_in*100:5.1f}%  "
              f"(data p01={np.percentile(xsig[:, i], 1):8.2f} p99={np.percentile(xsig[:, i], 99):8.2f})")
    # joint support with full-range box
    zmin = z.min(axis=0); zmax = z.max(axis=0)
    inside_full = np.all((xsig >= zmin) & (xsig <= zmax), axis=1)
    print(f"    data events fully inside prior [min,max] box: {np.sum(inside_full):,} / {len(xsig):,} ({np.mean(inside_full)*100:.1f}%)")
    # how much of the data lies beyond the prior's pair-pT max
    zpair_pt_max = zpair_pt.max()
    print(f"    data pair pT above prior max ({zpair_pt_max:.1f} GeV): {np.mean(pair_pt > zpair_pt_max)*100:.2f}%")
    # resolution smearing scale: prior delta -> data 28 MeV. Compare with J/psi natural width.
    print(f"\n[7] Resolution note: prior is a delta (p-based std {m_zp.std()*1000:.1f} MeV, natural width ~0.09 MeV),")
    print(f"    data core is broadened to ~{(q[2]-q[0])/2*1000:.0f} MeV by detector resolution,")
    print(f"    i.e. the transport must smear the prior by ~{(q[2]-q[0])/2:.3f} GeV to match the data peak.")

    # ---------------- FULL-window comparison (v3.5/v3.8 training config) ----------------
    print("\n[8] FULL-window [2.6,3.5]: CMS data vs UNFILTERED MG5 prior (v3.5/v3.8 setup)")
    print("  --- CMS data, full window ---")
    fpt1 = np.hypot(xfull[:, 0], xfull[:, 1]); fpt2 = np.hypot(xfull[:, 4], xfull[:, 5])
    feta = np.abs(np.concatenate([
        np.arcsinh(xfull[:, 2] / np.maximum(fpt1, 1e-9)),
        np.arcsinh(xfull[:, 6] / np.maximum(fpt2, 1e-9)),
    ]))
    fpair_pt = np.hypot(xfull[:, 0] + xfull[:, 4], xfull[:, 1] + xfull[:, 5])
    stats("mass [GeV]", mfull)
    stats("muon pT [GeV]", np.concatenate([fpt1, fpt2]))
    stats("pair pT [GeV]", fpair_pt)
    stats("muon |eta|", feta)
    print("  --- MG5 prior, full 1M file (unfiltered) ---")
    zfpt1 = np.hypot(zfull[:, 0], zfull[:, 1]); zfpt2 = np.hypot(zfull[:, 4], zfull[:, 5])
    zfeta = np.abs(np.concatenate([
        np.arcsinh(zfull[:, 2] / np.maximum(zfpt1, 1e-9)),
        np.arcsinh(zfull[:, 6] / np.maximum(zfpt2, 1e-9)),
    ]))
    zfpair_pt = np.hypot(zfull[:, 0] + zfull[:, 4], zfull[:, 1] + zfull[:, 5])
    stats("mass [GeV]", m_e)
    stats("muon pT [GeV]", np.concatenate([zfpt1, zfpt2]))
    stats("pair pT [GeV]", zfpair_pt)
    stats("muon |eta|", zfeta)
    # per-coordinate coverage of full-window data by unfiltered prior
    print("  --- support coverage of full-window data by unfiltered prior [p0.5,p99.5] box ---")
    zlo = np.percentile(zfull, 0.5, axis=0); zhi = np.percentile(zfull, 99.5, axis=0)
    for i, name in enumerate(COLS):
        frac_in = np.mean((xfull[:, i] >= zlo[i]) & (xfull[:, i] <= zhi[i]))
        print(f"    {name:8s} prior[{zlo[i]:8.2f},{zhi[i]:8.2f}]  data in range: {frac_in*100:5.1f}%")
    jinside = np.all((xfull >= zlo) & (xfull <= zhi), axis=1)
    print(f"    full-window data events fully inside prior box: {np.sum(jinside):,} / {len(xfull):,} ({np.mean(jinside)*100:.1f}%)")
    print(f"    full-window data pair pT above prior max ({zfpair_pt.max():.1f} GeV): {np.mean(fpair_pt > zfpair_pt.max())*100:.2f}%")
    # composition summary
    n_win_signal = sig_under
    print("\n[9] Composition summary (full window [2.6,3.5]):")
    print(f"    sideband est. of J/psi signal in window: {n_win_signal/len(xfull)*100:.1f}%  "
          f"| continuum: {100 - n_win_signal/len(xfull)*100:.1f}%")
    print(f"    prior composition: 100% on-shell J/psi, 0% continuum -> the two populations")
    print("    do NOT match; the SWAE must crush ~57% continuum onto a signal-only prior.")

    # ---------------- population-level pass-rate / hardness asymmetry ----------------
    print("\n[10] Population asymmetry (same filter applied to both sides)")
    zpt1f = np.hypot(zfull[:, 0], zfull[:, 1]); zpt2f = np.hypot(zfull[:, 4], zfull[:, 5])
    zeta1f = np.arcsinh(zfull[:, 2] / np.maximum(zpt1f, 1e-9))
    zeta2f = np.arcsinh(zfull[:, 6] / np.maximum(zpt2f, 1e-9))
    zkeep = ((zpt1f > 3) & (zpt2f > 3) & (np.abs(zeta1f) < 2.4) & (np.abs(zeta2f) < 2.4)
             & (m_e > 3.0369) & (m_e < 3.1569))
    xpt1 = np.hypot(xfull[:, 0], xfull[:, 1]); xpt2 = np.hypot(xfull[:, 4], xfull[:, 5])
    xeta1 = np.arcsinh(xfull[:, 2] / np.maximum(xpt1, 1e-9))
    xeta2 = np.arcsinh(xfull[:, 6] / np.maximum(xpt2, 1e-9))
    xkeep = ((xpt1 > 3) & (xpt2 > 3) & (np.abs(xeta1) < 2.4) & (np.abs(xeta2) < 2.4)
             & (mfull > 3.0369) & (mfull < 3.1569))
    print(f"    same filter (pT>3, |eta|<2.4, mass window) keeps: data {np.mean(xkeep)*100:.1f}%  "
          f"| prior {np.mean(zkeep)*100:.2f}%")
    zpt_all = np.concatenate([zpt1f, zpt2f])
    xpt_all = np.concatenate([xpt1, xpt2])
    print(f"    prior muon pT p99 = {np.percentile(zpt_all, 99):.2f} GeV -> "
          f"{np.mean(xpt_all > np.percentile(zpt_all, 99))*100:.1f}% of data muons are HARDER than prior p99")
    print(f"    prior muon pT p95 = {np.percentile(zpt_all, 95):.2f} GeV -> "
          f"{np.mean(xpt_all > np.percentile(zpt_all, 95))*100:.1f}% of data muons harder than prior p95")
    xin = (mfull > 3.0369) & (mfull < 3.1569)
    zin = zkeep
    xpair_pt = np.hypot(xfull[:, 0] + xfull[:, 4], xfull[:, 1] + xfull[:, 5])
    zpair_pt_f = np.hypot(zfull[:, 0] + zfull[:, 4], zfull[:, 1] + zfull[:, 5])
    print(f"    in-window pair pT: data median={np.median(xpair_pt[xin]):.1f} p99={np.percentile(xpair_pt[xin], 99):.1f} | "
          f"filtered prior median={np.median(zpair_pt_f[zin]):.1f} p99={np.percentile(zpair_pt_f[zin], 99):.1f}")
    print(f"    in-window muon pT: data median={np.median(np.concatenate([xpt1[xin], xpt2[xin]])):.1f} | "
          f"filtered prior median={np.median(np.concatenate([zpt1f[zin], zpt2f[zin]])):.1f}")
    print("    -> the data is ~2.5x harder than the prior even after trigger matching")


if __name__ == "__main__":
    main()
