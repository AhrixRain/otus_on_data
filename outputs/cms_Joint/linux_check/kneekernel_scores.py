import json
from pathlib import Path
run = Path("outputs/cms_Joint/Run_H_kneeKernel")
h = json.loads((run / "history.json").read_text(encoding="utf-8"))
print("epochs:", len(h))
scored = [r for r in h if isinstance(r.get("joint_selection"), (int, float))]
print("rows carrying a selection score:", len(scored))
for r in scored:
    stage = r["stage"]
    name = stage.get("name") if isinstance(stage, dict) else stage
    print(f"  ep {r['global_epoch']:3d} {name[-24:]:26s} score {r['joint_selection']:.4f}")
best = min(scored, key=lambda r: r["joint_selection"])
print("\nARG-MIN selection score:", best["global_epoch"], best["joint_selection"])
ev = json.loads((run / "joint_evaluation.json").read_text(encoding="utf-8"))
print("\njoint_evaluation keys:", sorted(ev.keys())[:12])
