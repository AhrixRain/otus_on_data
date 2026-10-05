import json, os
from pathlib import Path
cfg = json.load(open(r"C:\Users\AhrixMarin\AppData\Local\Temp\otus_clone_check\outputs\cms_Joint\Run_H_kneeKernel_dryrun\config.resolved.json"))
root = Path(r"C:\Users\AhrixMarin\Desktop\otus")
found = {}
def walk(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            walk(v, path + "/" + str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk(v, path + f"[{i}]")
    elif isinstance(node, str):
        if any(tok in node.lower() for tok in (".hdf5", ".root", ".npz")):
            found[path] = node
walk(cfg)
print("=== data files this arm resolves ===")
seen = set()
total = 0
for key, value in sorted(found.items()):
    p = Path(value)
    if not p.is_absolute():
        p = root / "data" / value
    if str(p) in seen:
        continue
    seen.add(str(p))
    size = p.stat().st_size / 2**20 if p.exists() else None
    if size:
        total += size
    print(f"  {'OK ' if p.exists() else 'MISS'} {size if size else '-':>9} MB  {value}   <- {key}")
print(f"\n  total: {total:.0f} MB across {len(seen)} files")
print("\n=== data/legacy/priors/ contents (needed by the joint configs) ===")
pri = root / "data" / "legacy" / "priors"
for p in sorted(pri.iterdir()) if pri.is_dir() else []:
    if p.is_file():
        print(f"  {p.stat().st_size/2**20:8.2f} MB  {p.name}")
