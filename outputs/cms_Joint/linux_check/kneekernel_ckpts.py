import torch
from pathlib import Path
run = Path("outputs/cms_Joint/Run_H_kneeKernel")
names = ["best_model.pt", "last_model.pt",
         "best_RunHkneeKernel_stage1_deterministic_warmup.pt",
         "best_RunHkneeKernel_stage2_stochastic_core.pt",
         "best_RunHkneeKernel_stage3_stochastic_tail.pt"]
print(f"{'checkpoint':50s} {'epoch':>6s} {'stage':>34s} {'selection':>10s}")
for name in names:
    p = run / name
    if not p.exists():
        print(f"{name:50s} MISSING"); continue
    ck = torch.load(p, map_location="cpu", weights_only=False)
    stage = ck.get("stage")
    sname = stage.get("name") if isinstance(stage, dict) else stage
    sel = ck.get("joint_selection")
    sel = sel.get("score") if isinstance(sel, dict) else sel
    print(f"{name:50s} {str(ck.get('global_epoch')):>6s} {str(sname):>34s} {str(sel)[:10]:>10s}")
    if isinstance(ck.get("joint_selection"), dict):
        print("      joint_selection keys:", sorted(ck["joint_selection"].keys())[:10])
