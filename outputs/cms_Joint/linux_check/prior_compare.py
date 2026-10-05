import h5py, numpy as np
from pathlib import Path
files = {
    "ROOT/legacy (screenshot) cms_jpsi_..._1M_reweighted": "data/cms_jpsi_mumu_mg5py8_ckkwl_8tev_inclusive_0j1j_fiducial_3p0369_3p1569_1M_reweighted.hdf5",
    "legacy/priors  jpsi_unified_bare_tms10":              "data/legacy/priors/jpsi_unified_bare_tms10.hdf5",
    "ROOT/legacy (screenshot) cms_dymumu_..._70_110_1M":   "data/cms_dymumu_mg5py8_ckkwl_8tev_inclusive_0j1j_fiducial_70_110_1M.hdf5",
    "legacy/priors  z_unified_bare_tms22p8":               "data/legacy/priors/z_unified_bare_tms22p8.hdf5",
}
for label, rel in files.items():
    p = Path(rel)
    if not p.exists():
        print(f"{label}\n   MISSING locally\n"); continue
    with h5py.File(p, "r") as f:
        def walk(g, pre=""):
            out = []
            for k in g:
                obj = g[k]
                if isinstance(obj, h5py.Dataset):
                    out.append((pre + k, obj.shape))
                else:
                    out += walk(obj, pre + k + "/")
            return out
        entries = walk(f)
        print(f"{label}   [{p.stat().st_size/2**20:.1f} MB]")
        for name, shape in entries:
            print(f"   {name:28s} {shape}")
        attrs = {k: (v if not isinstance(v, bytes) else v.decode()) for k, v in f.attrs.items()}
        for k in ("source_file", "smearing", "TMS", "note", "reference_file"):
            if k in attrs:
                print(f"   attr {k} = {str(attrs[k])[:80]}")
        print()
