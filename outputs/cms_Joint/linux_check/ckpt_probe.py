import torch, json
from pathlib import Path
base = Path("outputs/cms_Joint/Run_H_kneeKernel")
hist = json.load(open(base / "history.json"))
print("history rows:", len(hist), "| last global_epoch", hist[-1].get("global_epoch"), "| stage", hist[-1].get("stage"))
for name in ("last_model.pt", "best_model.pt",
             "last_RunHkneeKernel_stage3_stochastic_tail.pt",
             "best_RunHkneeKernel_stage3_stochastic_tail.pt"):
    p = base / name
    if not p.exists():
        print(f"{name:52s} MISSING")
        continue
    ck = torch.load(p, map_location="cpu", weights_only=False)
    keys = sorted(k for k in ck.keys() if k not in ("model", "model_state_dict"))
    print(f"{name:52s} epoch={ck.get('global_epoch')} stage={ck.get('stage')} stage_epoch={ck.get('stage_epoch')}")
    print(f"{'':52s} has_optimizer={'optimizer' in ck or 'optimizer_state_dict' in ck} keys={keys}")
