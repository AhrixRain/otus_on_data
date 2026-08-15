#!/usr/bin/env python
"""Float32-view check: mass spread of prior/data exactly as the training pipeline sees it."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.physics import invariant_mass_np
from scripts.cms_data import load_theory_prior_z


def emass(z):
    m2 = (z[:, 3] + z[:, 7]) ** 2 - (z[:, 0] + z[:, 4]) ** 2 - (z[:, 1] + z[:, 5]) ** 2 - (z[:, 2] + z[:, 6]) ** 2
    return np.sqrt(np.maximum(m2, 0))


def main():
    print("=== PRIOR: how the TRAINING sees z ===")
    zfull64 = load_theory_prior_z(Path("data/cms_jpsi_mumu_mg5_8tev_1M.hdf5")).astype(np.float64)
    zfull32 = zfull64.astype(np.float32)
    print("full prior 1M: E-based mass std  f64=%.2f MeV  f32=%.2f MeV" % (emass(zfull64).std()*1000, emass(zfull32).std()*1000))
    print("               p-based mass std  f64=%.2f MeV  f32=%.2f MeV" % (invariant_mass_np(zfull64, stable=True).std()*1000, invariant_mass_np(zfull32, stable=True).std()*1000))
    dE1 = zfull64[:, 3] - np.sqrt(zfull64[:, 0]**2 + zfull64[:, 1]**2 + zfull64[:, 2]**2)
    dE2 = zfull64[:, 7] - np.sqrt(zfull64[:, 4]**2 + zfull64[:, 5]**2 + zfull64[:, 6]**2)
    dE = np.concatenate([dE1, dE2])
    print("prior stored E - |p| per muon (f64): mean=%.2f MeV std=%.2f MeV max|.|=%.2f MeV" % (dE.mean()*1e3, dE.std()*1e3, np.abs(dE).max()*1e3))

    with np.load("outputs/cms_JpsiDoubleMuons/archive/.plot_cache/selected_split_0ed04817d72bb34f84ed.npz", allow_pickle=False) as d:
        zf32 = np.concatenate([np.asarray(d[k]) for k in ("z_train", "z_val", "z_test")])
    print("filtered prior %d (float32): E-based mass std=%.2f MeV | p-based std=%.2f MeV" % (len(zf32), emass(zf32).std()*1000, invariant_mass_np(zf32, stable=True).std()*1000))
    print("                            E-based mean=%.5f  p-based mean=%.5f" % (emass(zf32).mean(), invariant_mass_np(zf32, stable=True).mean()))

    print()
    print("=== DATA: how the TRAINING sees x (float32 cache) ===")
    with np.load("outputs/cms_JpsiDoubleMuons/archive/.plot_cache/selected_split_410ce075cc7cc2e067c8.npz", allow_pickle=False) as d:
        x32 = np.asarray(d["x_test"])
    x64 = x32.astype(np.float64)
    m_p32 = invariant_mass_np(x32, stable=True)
    inwin = (m_p32 > 3.0369) & (m_p32 < 3.1569)
    m_e32 = emass(x32)
    print("full-window x_test (f32): E-based mean=%.5f std=%.1f MeV | p-based mean=%.5f std=%.1f MeV" % (m_e32.mean(), m_e32.std()*1000, m_p32.mean(), m_p32.std()*1000))
    print("in-window (f32, n=%d): E-based mean=%.5f std=%.1f MeV | p-based mean=%.5f std=%.1f MeV" % (inwin.sum(), m_e32[inwin].mean(), m_e32[inwin].std()*1000, m_p32[inwin].mean(), m_p32[inwin].std()*1000))
    dEx1 = x64[:, 3] - np.sqrt(x64[:, 0]**2 + x64[:, 1]**2 + x64[:, 2]**2)
    dEx2 = x64[:, 7] - np.sqrt(x64[:, 4]**2 + x64[:, 5]**2 + x64[:, 6]**2)
    dEx = np.concatenate([dEx1, dEx2])
    print("data stored E - |p| per muon: mean=%.3f MeV std=%.2f MeV max|.|=%.2f GeV" % (dEx.mean()*1e3, dEx.std()*1e3, np.abs(dEx).max()))
    dm = np.abs(m_e32[inwin] - m_p32[inwin])
    print("in-window |E-based - p-based| mass: mean=%.2f MeV max=%.1f MeV" % (dm.mean()*1000, dm.max()*1000))


if __name__ == "__main__":
    main()
