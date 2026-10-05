import json
import torch
from pathlib import Path
ck = torch.load(Path("outputs/cms_Joint/Run_H_D3b/best_model.pt"), map_location="cpu", weights_only=False)
cfg = ck.get("config", {})
print("=== paths embedded in a Windows-produced checkpoint ===")
for key in ("cms_root_file", "output_root", "data_root", "cache_root"):
    if key in cfg.get("paths", {}):
        print(f"  paths.{key} = {cfg['paths'][key]!r}")
for region in ("jpsi", "z"):
    rp = cfg.get("regions", {}).get(region, {}).get("paths", {})
    for k, v in rp.items():
        print(f"  regions.{region}.paths.{k} = {v!r}")
print()
print("=== what POSIX thinks of those ===")
for label, value in (("cms_root_file", cfg.get("paths", {}).get("cms_root_file")),
                     ("jpsi prior", cfg.get("regions", {}).get("jpsi", {}).get("paths", {}).get("theory_prior_file"))):
    if value is None:
        continue
    import posixpath
    is_abs = posixpath.isabs(str(value))
    print(f"  {label}: {value!r}")
    print(f"     posix absolute? {is_abs}   -> {'OK' if is_abs else 'NOT absolute: a POSIX join would yield /repo/' + str(value)}")
