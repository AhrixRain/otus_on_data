#!/usr/bin/env python
"""Z->ee health check: CMS data vs MG5 prior mismatch, mirroring the J/psi check.

Reuses outputs/cms_doubleelectron/.plot_cache selected_split_9bb63a1e38cf0cd9b9c2
(version 1, full data+prior). Prints: prior metadata, mass/kinematics, composition
(on-peak vs off-peak), support coverage, pass-rate asymmetry, and a verdict vs J/psi.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.physics import invariant_mass_np
from scripts.cms_data import load_theory_prior_z

CACHE = ROOT / "outputs/cms_doubleelectron/.plot_cache/selected_split_9bb63a1e38cf0cd9b9c2.npz"
PRIOR = ROOT / "data/cms_dyee_mg5_8tev_dy1j_ptj5_fiducial_70_110.hdf5"
COLS = ["e- px", "e- py", "e- pz", "e- E", "e+ px", "e+ py", "e+ pz", "e+ E"]


def stats(name, a):
    q = np.percentile(a, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    print("    %-22s mean=%10.3f std=%9.3f p01=%8.3f p05=%8.3f med=%8.3f p95=%8.3f p99=%8.3f max=%9.3f"
          % (name, a.mean(), a.std(), q[1], q[2], q[4], q[6], q[7], q[8]))


def emass(z):
    m2 = (z[:, 3] + z[:, 7]) ** 2 - (z[:, 0] + z[:, 4]) ** 2 - (z[:, 1] + z[:, 5]) ** 2 - (z[:, 2] + z[:, 6]) ** 2
    return np.sqrt(np.maximum(m2, 0))


def main():
    print("=" * 80)
    print("Z->ee HEALTH CHECK: CMS DATA vs MG5 PRIOR")
    print("=" * 80)

    # ---- prior metadata ----
    import h5py
    with h5py.File(PRIOR, "r") as h:
        def walk(name, obj):
            if hasattr(obj, "attrs") and obj.attrs:
                print("  %s -> %s" % (name, dict(obj.attrs)))
        print("\n[0] Prior HDF5 attributes:")
        h.visititems(walk)

    zfull = load_theory_prior_z(PRIOR).astype(np.float64)
    print("\n[1] Prior, full file (n=%d):" % len(zfull))
    m_ze = emass(zfull)
    m_zp = invariant_mass_np(zfull, stable=True)
    print("    mass E-based: mean=%.5f std=%.3f GeV" % (m_ze.mean(), m_ze.std()))
    print("    mass p-based: mean=%.5f std=%.3f GeV" % (m_zp.mean(), m_zp.std()))
    dE1 = zfull[:, 3] - np.sqrt(zfull[:, 0]**2 + zfull[:, 1]**2 + zfull[:, 2]**2)
    dE2 = zfull[:, 7] - np.sqrt(zfull[:, 4]**2 + zfull[:, 5]**2 + zfull[:, 6]**2)
    dE = np.concatenate([dE1, dE2])
    print("    stored E - |p| per electron: mean=%.3f MeV std=%.3f MeV max|.|=%.2f MeV"
          % (dE.mean()*1e3, dE.std()*1e3, np.abs(dE).max()*1e3))
    print("    frac mass in [70,110]: E-based %.2f%% | p-based %.2f%%"
          % (np.mean((m_ze > 70) & (m_ze < 110))*100, np.mean((m_zp > 70) & (m_zp < 110))*100))

    # ---- cached arrays ----
    with np.load(CACHE, allow_pickle=False) as d:
        x = np.concatenate([np.asarray(d[k], dtype=np.float64) for k in ("x_train", "x_val", "x_test")])
        z = np.concatenate([np.asarray(d[k], dtype=np.float64) for k in ("z_train", "z_val", "z_test")])
    m_xe = emass(x)
    m_xp = invariant_mass_np(x, stable=True)
    m_ze_c = emass(z)
    m_zp_c = invariant_mass_np(z, stable=True)
    print("\n[2] Data vs prior mass distributions (training arrays, float32->f64 recast):")
    stats("data mass E-based [GeV]", m_xe)
    stats("data mass p-based [GeV]", m_xp)
    stats("prior mass E-based [GeV]", m_ze_c)
    stats("prior mass p-based [GeV]", m_zp_c)
    bins = np.linspace(70, 110, 81)
    c, e = np.histogram(m_xp, bins=bins)
    cent = 0.5 * (e[:-1] + e[1:])
    print("    data p-based mass peak bin: %.2f GeV (0.5 GeV bins), count=%d" % (cent[np.argmax(c)], c.max()))
    # core width via quantiles around the peak
    onpeak = (m_xp > 80) & (m_xp < 100)
    q = np.percentile(m_xp[onpeak], [16, 50, 84])
    print("    data on-peak [80,100] quantiles 16/50/84: %.3f / %.3f / %.3f -> half-width %.3f GeV"
          % (q[0], q[1], q[2], (q[2]-q[0])/2))

    # ---- composition: on-peak vs off-peak ----
    print("\n[3] Composition of the [70,110] window:")
    n_lo = np.sum(m_xp < 80); n_pk = np.sum((m_xp >= 80) & (m_xp <= 100)); n_hi = np.sum(m_xp > 100)
    print("    data: [70,80): %d (%.1f%%) | [80,100]: %d (%.1f%%) | (100,110]: %d (%.1f%%)"
          % (n_lo, n_lo/len(x)*100, n_pk, n_pk/len(x)*100, n_hi, n_hi/len(x)*100))
    nz_lo = np.sum(m_zp_c < 80); nz_pk = np.sum((m_zp_c >= 80) & (m_zp_c <= 100)); nz_hi = np.sum(m_zp_c > 100)
    print("    prior: [70,80): %d (%.1f%%) | [80,100]: %d (%.1f%%) | (100,110]: %d (%.1f%%)"
          % (nz_lo, nz_lo/len(z)*100, nz_pk, nz_pk/len(z)*100, nz_hi, nz_hi/len(z)*100))
    # off-peak density ratio: data vs prior (how well matched is the continuum)
    dens_lo_x = n_lo / 10; dens_hi_x = n_hi / 10
    dens_lo_z = nz_lo / 10; dens_hi_z = nz_hi / 10
    print("    off-peak densities [/GeV]: data low=%.0f high=%.0f | prior low=%.0f high=%.0f"
          % (dens_lo_x, dens_hi_x, dens_lo_z, dens_hi_z))

    # ---- kinematics ----
    print("\n[4] Kinematics (data vs prior):")
    xpt1 = np.hypot(x[:, 0], x[:, 1]); xpt2 = np.hypot(x[:, 4], x[:, 5])
    xeta = np.abs(np.concatenate([np.arcsinh(x[:, 2]/np.maximum(xpt1, 1e-9)), np.arcsinh(x[:, 6]/np.maximum(xpt2, 1e-9))]))
    xpair_pt = np.hypot(x[:, 0] + x[:, 4], x[:, 1] + x[:, 5])
    print("  --- data ---")
    stats("electron pT [GeV]", np.concatenate([xpt1, xpt2]))
    stats("pair pT [GeV]", xpair_pt)
    stats("electron |eta|", xeta)
    zpt1 = np.hypot(z[:, 0], z[:, 1]); zpt2 = np.hypot(z[:, 4], z[:, 5])
    zeta = np.abs(np.concatenate([np.arcsinh(z[:, 2]/np.maximum(zpt1, 1e-9)), np.arcsinh(z[:, 6]/np.maximum(zpt2, 1e-9))]))
    zpair_pt = np.hypot(z[:, 0] + z[:, 4], z[:, 1] + z[:, 5])
    print("  --- prior ---")
    stats("electron pT [GeV]", np.concatenate([zpt1, zpt2]))
    stats("pair pT [GeV]", zpair_pt)
    stats("electron |eta|", zeta)

    # ---- support coverage ----
    print("\n[5] Per-coordinate support coverage (data vs prior [p0.5,p99.5] box):")
    lo = np.percentile(z, 0.5, axis=0); hi = np.percentile(z, 99.5, axis=0)
    inside = np.all((x >= lo) & (x <= hi), axis=1)
    print("    data events fully inside prior box: %d / %d (%.1f%%)" % (inside.sum(), len(x), inside.mean()*100))
    for i, name in enumerate(COLS):
        frac = np.mean((x[:, i] >= lo[i]) & (x[:, i] <= hi[i]))
        print("    %-8s prior[%8.2f,%8.2f]  data in range: %5.1f%%" % (name, lo[i], hi[i], frac*100))

    # ---- barrel vs endcap scale check (skim calibration) ----
    print("\n[6] Barrel/endcap fine-mode split (5 MeV bins, [80,100]):")
    xeta1 = np.arcsinh(x[:, 2] / np.maximum(xpt1, 1e-9))
    xeta2 = np.arcsinh(x[:, 6] / np.maximum(xpt2, 1e-9))
    for name, sel in [
        ("both barrel", (np.abs(xeta1) < 1.4442) & (np.abs(xeta2) < 1.4442)),
        ("both endcap", (np.abs(xeta1) > 1.566) & (np.abs(xeta1) < 2.5) & (np.abs(xeta2) > 1.566) & (np.abs(xeta2) < 2.5)),
    ]:
        mm = m_xp[sel]
        cc, ee = np.histogram(mm[(mm > 80) & (mm < 100)], bins=4000)
        ccc = 0.5 * (ee[:-1] + ee[1:])
        print("    %-12s n=%7d  fine mode=%.3f  median=%.3f  mean=%.3f  (Z pole=91.1876)"
              % (name, sel.sum(), ccc[np.argmax(cc)], np.median(mm), mm.mean()))
    c1, e1 = np.histogram(m_xp[(m_xp > 80) & (m_xp < 100)], bins=4000)
    c1c = 0.5 * (e1[:-1] + e1[1:])
    print("    overall data fine mode=%.3f (+%.0f MeV vs pole) | prior fine mode=%.3f"
          % (c1c[np.argmax(c1)], (c1c[np.argmax(c1)] - 91.1876) * 1000,
             (lambda mm: (lambda h, e: 0.5 * (e[np.argmax(h)] + e[np.argmax(h) + 1]))(*np.histogram(mm[(mm > 80) & (mm < 100)], bins=4000)))(m_zp_c)))

    # ---- pass-rate asymmetry: apply the data-side selection to the prior ----
    print("\n[7] Pass-rate asymmetry:")
    keepz = (zpt1 > 20) & (zpt2 > 20) & (np.abs(np.arcsinh(z[:, 2]/np.maximum(zpt1, 1e-9))) < 2.5) & (np.abs(np.arcsinh(z[:, 6]/np.maximum(zpt2, 1e-9))) < 2.5) & (m_zp_c > 70) & (m_zp_c < 110)
    print("    prior passing e pT>20 + |eta|<2.5 + mass window: %.2f%% (iso/dxy/dz N/A at truth level)" % (keepz.mean()*100))
    zpt_all = np.concatenate([zpt1, zpt2]); xpt_all = np.concatenate([xpt1, xpt2])
    print("    prior electron pT p99 = %.2f GeV -> %.1f%% of data electrons harder"
          % (np.percentile(zpt_all, 99), np.mean(xpt_all > np.percentile(zpt_all, 99))*100))
    print("    pair pT: data median=%.1f p99=%.1f | prior median=%.1f p99=%.1f"
          % (np.median(xpair_pt), np.percentile(xpair_pt, 99), np.median(zpair_pt), np.percentile(zpair_pt, 99)))

    # ---- alternate prior file (no dy1j) ----
    z2 = load_theory_prior_z(ROOT / "data/cms_dyee_mg5_8tev_fiducial_70_110.hdf5").astype(np.float64)
    m2 = invariant_mass_np(z2, stable=True)
    print("\n[8] Alternate prior (cms_dyee_mg5_8tev_fiducial_70_110.hdf5, no dy1j):")
    print("    n=%d  mean=%.3f  std=%.3f  median=%.3f  frac in [70,110]: %.1f%%"
          % (len(z2), m2.mean(), m2.std(), np.median(m2), np.mean((m2 > 70) & (m2 < 110)) * 100))


if __name__ == "__main__":
    main()
