import json, torch
from pathlib import Path
run = Path("outputs/cms_Joint/Run_H_kneeKernel")
rows = []
for p in sorted(run.glob("*.pt")):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    js = ck.get("joint_selection") or {}
    score = js.get("selection_score")
    worst = js.get("worst_region")
    regions = js.get("regions") or {}
    rows.append((p.name, ck.get("global_epoch"), score, worst,
                 {k: (v.get("score") if isinstance(v, dict) else v) for k, v in regions.items()}))
print(f"{'checkpoint':50s} {'ep':>4s} {'score':>8s} {'worst':>6s}  regions")
for name, ep, score, worst, regions in sorted(rows, key=lambda r: (r[2] is None, r[2])):
    fmt = " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in regions.items())
    print(f"{name:50s} {str(ep):>4s} {score if score is None else round(score,4)!s:>8s} {str(worst):>6s}  {fmt}")
print()
print("D3b reference (memory): stage scores 4.337 / 4.551 / 3.886 -> selected global epoch 180")
