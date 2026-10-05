import os
from collections import defaultdict
from pathlib import Path
root = Path(r"C:\Users\AhrixMarin\Desktop\otus\outputs")
by_ext = defaultdict(lambda: [0, 0])
total = 0; count = 0
for base, dirs, files in os.walk(root):
    for name in files:
        p = Path(base) / name
        try: size = p.stat().st_size
        except OSError: continue
        ext = p.suffix.lower() or "(none)"
        by_ext[ext][0] += 1; by_ext[ext][1] += size
        total += size; count += 1
print(f"outputs/ total: {total/2**30:.2f} GiB in {count} files (working tree; NOT in git: 1388)")
print()
print(f"{'ext':8s} {'files':>7s} {'GiB':>8s}")
for ext, (n, s) in sorted(by_ext.items(), key=lambda kv: -kv[1][1])[:14]:
    print(f"{ext:8s} {n:7d} {s/2**30:8.3f}")
print()
print("=== what a SERVER actually needs, by category ===")
cats = {
    ".region_cache caches": [], "checkpoints .pt": [], "decoded/transfer .hdf5": [],
    "provenance *.json": [], "figures .png/.pdf": [], "reports .md": [], "other": [],
}
for base, dirs, files in os.walk(root):
    for name in files:
        p = Path(base) / name
        rel = str(p.relative_to(root)); low = rel.lower()
        try: s = p.stat().st_size
        except OSError: continue
        if ".region_cache" in low: cats[".region_cache caches"].append((rel, s))
        elif name.endswith(".pt"): cats["checkpoints .pt"].append((rel, s))
        elif name.endswith(".hdf5"): cats["decoded/transfer .hdf5"].append((rel, s))
        elif name.endswith(".json"): cats["provenance *.json"].append((rel, s))
        elif name.endswith((".png", ".pdf")): cats["figures .png/.pdf"].append((rel, s))
        elif name.endswith(".md"): cats["reports .md"].append((rel, s))
        else: cats["other"].append((rel, s))
for k, v in cats.items():
    print(f"{k:26s} {len(v):5d} files  {sum(s for _, s in v)/2**30:7.3f} GiB")
