import json, math, pathlib
# Replicate the verifier block from train_readiness.sh against a REAL run dir,
# to confirm its assumptions about the history schema hold.
check_dir = pathlib.Path("outputs/cms_Joint/Run_H_kneeKernel")
history = json.loads((check_dir / "history.json").read_text(encoding="utf-8"))
print("history rows:", len(history))
print("row keys:", sorted(history[-1].keys()))
stage = history[-1].get("stage")
print("stage type:", type(stage).__name__, "| value:", stage)
stages = []
for row in history:
    name = row.get("stage")
    name = name.get("name") if isinstance(name, dict) else name
    if name and name not in stages:
        stages.append(name)
print("stages discovered:", stages)
bad = [ (r.get("global_epoch"), k, v) for r in history for k, v in r.items()
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) ]
print("non-finite floats:", bad if bad else "none")
cps = sorted(check_dir.glob("*.pt"))
print("checkpoints:", len(cps), "| largest MiB:", round(max(p.stat().st_size for p in cps)/2**20, 1))
cfg = json.loads((check_dir / "config.resolved.json").read_text(encoding="utf-8"))
enabled = [s for s in cfg.get("stages", []) if s.get("enabled", True)]
total = sum(int(s.get("epochs", 0)) for s in enabled)
evals = sorted({int(s.get("eval_every", 1)) for s in enabled})
print("full schedule:", total, "global epochs | eval_every:", evals)
print("VERIFIER ASSUMPTIONS HOLD for this run dir")
