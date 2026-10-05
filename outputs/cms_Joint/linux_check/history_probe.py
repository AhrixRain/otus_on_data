import json
from collections import Counter
h = json.load(open("outputs/cms_Joint/Run_H_kneeKernel/history.json"))
print("rows:", len(h))
print("epoch range:", h[0].get("global_epoch"), "->", h[-1].get("global_epoch"))
eras = Counter(r.get("global_epoch") for r in h)
dups = {e: c for e, c in eras.items() if c > 1}
print("duplicate epochs:", dups if dups else "none")
print("last 6 rows:")
for r in h[-6:]:
    print("  epoch", r.get("global_epoch"), "| stage_epoch", r.get("stage_epoch"),
          "| train_loss", r.get("train_loss"), "| eval_loss", r.get("eval_loss"))
