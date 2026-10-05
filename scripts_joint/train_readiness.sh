#!/usr/bin/env bash
# TRAIN READINESS: prove that a REAL training can run here, end to end.
#
#   bash scripts_joint/train_readiness.sh <RUN_ID> [auto|cuda|cpu] [--epochs N]
#                                         [--num-samples N] [--keep] [--tests]
#
# Why not just --smoke: --smoke uses 1000 events and a tiny model, so it proves the
# config parses and the loss is wired, but it does NOT prove that the real cache
# builds, that full-size per-update batches fit, that an epoch finishes in a sane
# time, or that a checkpoint is written. This script runs the arm's own config on
# the arm's own data for a few epochs per stage.
#
# WHAT IT PROVES / CHECKS
#   1. the full-data region cache builds (the expensive first read of the 2.1 GB
#      CMS ROOT file). The cache key is content-derived and lives in the shared
#      outputs/cms_Joint/.region_cache, so the real run afterwards REUSES it;
#   2. every stage runs, including the stochastic ones, at the config's real
#      per-update batch sizes;
#   3. losses stay finite (NaN/Inf scan over history.json);
#   4. at least one checkpoint is written -- hence --epochs must be >= the stages'
#      eval_every (default 5). A shorter run would prove nothing about writing;
#   5. the locked Upsilon region stays closed;
#   6. wall time, extrapolated to the arm's full schedule.
#
# CAVEATS
#   * --epochs is PER STAGE (run_joint.py:176-179 sets every enabled stage to N),
#     so "--epochs 5" on a three-stage config is 15 global epochs.
#   * --num-samples smaller than the config value produces a DIFFERENT cache key,
#     so the real run would rebuild the cache. Use it only for a quick look.
#   * Writes outputs/cms_Joint/<RUN_ID>_traincheck and deletes it at the end
#     unless --keep is given.
#
# exit 0 all green | 1 dry-run failed | 2 test training failed | 3 environment
# | 4 verification failed | 5 unit suite failed

set -uo pipefail

RUN_ID=""; DEVICE="auto"; EPOCHS=5; NUM_SAMPLES=""; KEEP=0; TESTS=0
while [ $# -gt 0 ]; do
  case "$1" in
    --epochs) EPOCHS="$2"; shift 2 ;;
    --num-samples) NUM_SAMPLES="$2"; shift 2 ;;
    --keep) KEEP=1; shift ;;
    --tests) TESTS=1; shift ;;
    -h|--help) sed -n '2,36p' "$0"; exit 0 ;;
    auto|cuda|cpu) DEVICE="$1"; shift ;;
    -*) echo "unknown option: $1" >&2; exit 3 ;;
    *) if [ -z "$RUN_ID" ]; then RUN_ID="$1"; else echo "only one RUN_ID" >&2; exit 3; fi; shift ;;
  esac
done
if [ -z "$RUN_ID" ]; then
  echo "usage: bash scripts_joint/train_readiness.sh <RUN_ID> [auto|cuda|cpu] [--epochs N] [--num-samples N] [--keep] [--tests]" >&2
  exit 3
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT" || exit 3

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if   command -v python  >/dev/null 2>&1; then PY=python
  elif command -v python3 >/dev/null 2>&1; then PY=python3
  else echo "FAIL: no python on PATH (activate the venv)" >&2; exit 3
  fi
fi
export PYTHONUTF8=1
if [ -z "${LANG:-}" ]; then export LANG=en_US.UTF-8; fi

CONFIG="configs_joint/cms_Joint_${RUN_ID}.yaml"
CHECK_NAME="${RUN_ID}_traincheck"
CHECK_DIR="outputs/cms_Joint/${CHECK_NAME}"
LOG_DIR="logs"; mkdir -p "$LOG_DIR"
LOG_DRY="$LOG_DIR/traincheck_${RUN_ID}_dryrun.log"
LOG_RUN="$LOG_DIR/traincheck_${RUN_ID}_run.log"

if [ -n "$NUM_SAMPLES" ]; then SAMPLES_LABEL="$NUM_SAMPLES"; else SAMPLES_LABEL="<config default = full data>"; fi

banner() { echo; echo "=== $* ==============================================="; }

echo "================================================================================"
echo " TRAIN READINESS  $RUN_ID    $(date '+%Y-%m-%d %H:%M:%S')"
echo " repo     $REPO_ROOT"
echo " python   $("$PY" -c 'import sys; print(sys.version.split()[0])')   device $DEVICE"
echo " epochs   $EPOCHS per stage   num_samples $SAMPLES_LABEL"
echo "================================================================================"

# ---- environment --------------------------------------------------------------
[ -f "$CONFIG" ] || { echo "FAIL: config not found: $CONFIG" >&2; exit 3; }
[ -d data ] || { echo "FAIL: data/ absent (gitignored; stage it manually)" >&2; exit 3; }
if ! "$PY" -c "import yaml" 2>/dev/null; then
  echo "FAIL: PyYAML missing (hard import in scripts/cms_data.py, absent from requirements-cms.txt)" >&2
  echo "      fix: $PY -m pip install PyYAML==6.0.3" >&2
  exit 3
fi
"$PY" - <<'PYEOF' || exit 3
import sys
try:
    import torch
except Exception as error:
    print(f"FAIL: torch not importable: {error}")
    sys.exit(1)
print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print(f"GPU {torch.cuda.get_device_name(0)}  free {free/2**30:.1f} / {total/2**30:.1f} GiB")
PYEOF
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1; fi
DF_FREE=$(df -Pk . | awk 'NR==2 {printf "%.1f", $4/1048576}')
echo "disk free: $DF_FREE GiB  (the full cache plus checkpoints need a few GiB)"

# ---- 1. dry-run ---------------------------------------------------------------
banner "[1/4] dry-run (config + data contract)"
"$PY" scripts_joint/run_joint.py --run "$RUN_ID" --device "$DEVICE" --dry-run 2>&1 | tee "$LOG_DRY"
dry_status=${PIPESTATUS[0]}
if [ "$dry_status" -ne 0 ] || ! grep -q "dry-run passed" "$LOG_DRY"; then
  echo "FAIL: dry-run (exit $dry_status). See $LOG_DRY" >&2
  exit 1
fi
echo "OK: dry-run passed"

# ---- 2. real-data bounded training -------------------------------------------
banner "[2/4] real training, $EPOCHS epochs per stage (final evaluation skipped)"
EXTRA=()
if [ -n "$NUM_SAMPLES" ]; then EXTRA=(--num-samples "$NUM_SAMPLES"); fi
START_TS=$(date +%s)
"$PY" scripts_joint/run_joint.py --run "$RUN_ID" --device "$DEVICE" \
  --run-name "$CHECK_NAME" --epochs "$EPOCHS" --skip-evaluation "${EXTRA[@]}" 2>&1 | tee "$LOG_RUN"
run_status=${PIPESTATUS[0]}
END_TS=$(date +%s)
if [ "$run_status" -ne 0 ]; then
  echo "FAIL: training exited $run_status after $((END_TS-START_TS))s. Log: $LOG_RUN" >&2
  exit 2
fi
echo "OK: training loop finished in $((END_TS-START_TS))s"

# ---- 3. verification ----------------------------------------------------------
banner "[3/4] verification"
"$PY" - "$CHECK_DIR" "$EPOCHS" "$LOG_RUN" <<'PYEOF'
import json, math, pathlib, re, sys

check_dir, per_stage, log_path = pathlib.Path(sys.argv[1]), int(sys.argv[2]), pathlib.Path(sys.argv[3])
problems, notes = [], []

history_path = check_dir / "history.json"
history = []
if not history_path.exists():
    problems.append(f"no history.json in {check_dir}")
else:
    history = json.loads(history_path.read_text(encoding="utf-8"))
    notes.append(f"history rows: {len(history)}")
    stages = []
    for row in history:
        name = row.get("stage")
        name = name.get("name") if isinstance(name, dict) else name
        if name and name not in stages:
            stages.append(name)
    notes.append(f"stages exercised: {len(stages)} -> {stages}")
    bad = []
    for row in history:
        for key, value in row.items():
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                bad.append(f"epoch {row.get('global_epoch')} {key}={value}")
    if bad:
        problems.append("non-finite values: " + "; ".join(bad[:5]))
    else:
        notes.append("all logged floats finite (no NaN/Inf)")

    seconds = [float(r["seconds"]) for r in history
               if isinstance(r.get("seconds"), (int, float)) and r["seconds"] > 0]
    peak = [float(r["peak_cuda_reserved_gb"]) for r in history
            if isinstance(r.get("peak_cuda_reserved_gb"), (int, float))]
    if seconds:
        notes.append(f"epoch wall time: mean {sum(seconds)/len(seconds):.1f} s "
                     f"(min {min(seconds):.1f}, max {max(seconds):.1f}) over {len(seconds)} epochs")
    if peak:
        notes.append(f"peak CUDA reserved: {max(peak):.2f} GiB")

checkpoints = sorted(check_dir.glob("*.pt"))
if not checkpoints:
    problems.append("NO checkpoint written -- --epochs is below eval_every, so this run "
                    "proved nothing about checkpointing")
else:
    notes.append(f"checkpoints written: {len(checkpoints)} "
                 f"(largest {max(p.stat().st_size for p in checkpoints)/2**20:.1f} MiB)")

text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
if "Upsilon was not opened" not in text:
    problems.append("the Upsilon-exclusion line is absent from the log")
else:
    notes.append("Upsilon region stayed closed")

resolved = check_dir / "config.resolved.json"
if resolved.exists():
    cfg = json.loads(resolved.read_text(encoding="utf-8"))
    enabled = [s for s in cfg.get("stages", []) if s.get("enabled", True)]
    total = sum(int(s.get("epochs", 0)) for s in enabled)
    evals = sorted({int(s.get("eval_every", 1)) for s in enabled})
    notes.append(f"full schedule of this config: {total} global epochs; eval_every {evals}")
    if per_stage < max(evals):
        problems.append(f"--epochs {per_stage} < max eval_every {max(evals)}")
    notes.append(f"per-stage epochs requested {per_stage} -> {len(history)} global epochs logged")
    if seconds and total:
        mean_s = sum(seconds) / len(seconds)
        hours = mean_s * total / 3600.0
        notes.append(f"EXTRAPOLATED full run: {total} epochs x {mean_s:.1f} s = {hours:.2f} h "
                     f"(cache build already paid by this test)")
        if peak:
            gib = max(peak)
            notes.append(f"Slurm suggestion: --gres=gpu:1 --mem=16G --time={int(hours*1.4)+1:02d}:00:00 "
                         f"(estimated ~{gib:.1f} GiB VRAM; the config self-limits to a fraction of the device)")

print("  summary:")
for note in notes:
    print(f"    - {note}")
if problems:
    print("  PROBLEMS:")
    for problem in problems:
        print(f"    ! {problem}")
    sys.exit(1)
print("  OK: verification clean")
PYEOF
verify_status=$?
if [ "$verify_status" -ne 0 ]; then
  echo "FAIL: verification; keeping $CHECK_DIR for inspection" >&2
  exit 4
fi

# ---- 4. optional suite + cleanup ---------------------------------------------
if [ "$TESTS" -eq 1 ]; then
  banner "[4/4] unit suite"
  if ! "$PY" -m unittest discover -s tests -v; then
    echo "FAIL: unit suite; keeping $CHECK_DIR" >&2
    exit 5
  fi
  echo "OK: unit suite green"
else
  banner "[4/4] unit suite skipped (pass --tests)"
fi

if [ "$KEEP" -eq 1 ]; then
  echo "keeping $CHECK_DIR (--keep)"
elif [ -d "$CHECK_DIR" ]; then
  case "$CHECK_DIR" in
    *_traincheck) rm -rf "$CHECK_DIR"; echo "removed throwaway $CHECK_DIR" ;;
    *) echo "REFUSING to delete $CHECK_DIR (name does not end in _traincheck)" >&2 ;;
  esac
fi

echo
echo "================================================================================"
echo " READINESS GREEN: $RUN_ID -- real training works on this machine"
echo " full run:"
echo "   $PY scripts_joint/run_joint.py --run $RUN_ID --device $DEVICE"
echo "================================================================================"
