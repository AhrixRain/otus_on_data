import h5py, numpy as np
from pathlib import Path

def mass8(v):
    v = np.asarray(v, dtype=np.float64)
    e = v[:, 3] + v[:, 7]
    p = v[:, :3] + v[:, 4:7]
    return np.sqrt(np.maximum(e * e - np.sum(p * p, axis=1), 0.0))

files = [
    ("config now: legacy/priors/jpsi_unified_bare_tms10", "data/legacy/priors/jpsi_unified_bare_tms10.hdf5"),
    ("screenshot: root ..._1M_reweighted",               "data/cms_jpsi_mumu_mg5py8_ckkwl_8tev_inclusive_0j1j_fiducial_3p0369_3p1569_1M_reweighted.hdf5"),
    ("A/B smeared arm",                                  "data/legacy/cms_jpsi_ab_smeared.hdf5"),
    ("A/B narrow arm",                                   "data/legacy/cms_jpsi_ab_narrow.hdf5"),
    ("config now: legacy/priors/z_unified_bare_tms22p8", "data/legacy/priors/z_unified_bare_tms22p8.hdf5"),
    ("screenshot: root cms_dymumu_..._70_110_1M",        "data/cms_dymumu_mg5py8_ckkwl_8tev_inclusive_0j1j_fiducial_70_110_1M.hdf5"),
    ("config-eval: upsilon_prior_continuumReweighted",   "data/upsilon_prior_continuumReweighted.hdf5"),
]
print(f"{'file':52s} {'n':>9s} {'std MeV':>9s} {'robust MeV':>11s} {'q02 MeV':>9s} {'weight?':>8s} {'comp?':>6s}")
for label, rel in files:
    p = Path(rel)
    if not p.exists():
        print(f"{label:52s}   (not present locally)")
        continue
    with h5py.File(p, "r") as f:
        z = f["FDL/zData"][:]
        has_w = "FDL/weight" in f
        has_c = "FDL/component_id" in f
        m = mass8(z)
        # narrow the window to the resonance of interest so the numbers are comparable
        if m.mean() < 5:
            m = m[(m > 2.8) & (m < 3.4)]
        else:
            m = m[(m > 8.0) & (m < 11.5)]
    if len(m) == 0:
        print(f"{label:52s}   (empty window)")
        continue
    robust = (np.quantile(m, 0.84) - np.quantile(m, 0.16)) / 2
    q02 = np.quantile(m, 0.02)
    print(f"{label:52s} {len(m):9d} {m.std()*1000:9.2f} {robust*1000:11.2f} "
          f"{(m.mean()-q02)*1000:9.2f} {str(has_w):>8s} {str(has_c):>6s}")
print()
print("(window 2.8-3.4 GeV for the J/psi files, 8.0-11.5 GeV for Upsilon; 'robust' = (q84-q16)/2)")
