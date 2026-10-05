import h5py, numpy as np
from pathlib import Path

def mass8(v):
    v = np.asarray(v, dtype=np.float64)
    e = v[:, 3] + v[:, 7]
    p = v[:, :3] + v[:, 4:7]
    return np.sqrt(np.maximum(e * e - np.sum(p * p, axis=1), 0.0))

CASES = [
    ("config jpsi: legacy/priors/jpsi_unified_bare_tms10", "data/legacy/priors/jpsi_unified_bare_tms10.hdf5", (2.8, 3.4)),
    ("screenshot jpsi: root ..._1M_reweighted",            "data/cms_jpsi_mumu_mg5py8_ckkwl_8tev_inclusive_0j1j_fiducial_3p0369_3p1569_1M_reweighted.hdf5", (2.8, 3.4)),
    ("A/B smeared",                                        "data/legacy/cms_jpsi_ab_smeared.hdf5", (2.8, 3.4)),
    ("config z: legacy/priors/z_unified_bare_tms22p8",     "data/legacy/priors/z_unified_bare_tms22p8.hdf5", (70.0, 110.0)),
    ("screenshot z: root cms_dymumu_..._70_110_1M",        "data/cms_dymumu_mg5py8_ckkwl_8tev_inclusive_0j1j_fiducial_70_110_1M.hdf5", (70.0, 110.0)),
    ("eval upsilon: continuumReweighted",                  "data/upsilon_prior_continuumReweighted.hdf5", (8.0, 11.5)),
]
for label, rel, win in CASES:
    p = Path(rel)
    if not p.exists():
        print(f"{label}\n    (not present)\n"); continue
    print(f"{label}   [{p.stat().st_size/2**20:.1f} MB]")
    with h5py.File(p, "r") as f:
        z = f["FDL/zData"][:]
        comp = f["FDL/component_id"][:] if "FDL/component_id" in f else None
        w = f["FDL/weight"][:] if "FDL/weight" in f else None
        m = mass8(z)
        groups = [(None, np.ones(len(m), bool))] if comp is None else [
            (int(c), comp == c) for c in np.unique(comp)]
        for cid, mask in groups:
            mm = m[mask]
            mm = mm[(mm > win[0]) & (mm < win[1])]
            if len(mm) < 50:
                print(f"    component {cid}: only {len(mm)} events in {win} GeV  (skipped)")
                continue
            robust = (np.quantile(mm, 0.84) - np.quantile(mm, 0.16)) / 2
            ess = None
            if w is not None:
                ww = w[mask]; ww = ww[np.isfinite(ww)]
                if ww.size and ww.sum() > 0:
                    ess = float(ww.sum()**2 / np.sum(ww**2))
            print(f"    component {cid}: n={len(mm):>9d}  std {mm.std()*1000:8.2f} MeV  "
                  f"robust {robust*1000:8.2f} MeV  mean {mm.mean():.4f} GeV"
                  + (f"  ESS/N {ess/len(mm):.3f}" if ess else ""))
    print()
