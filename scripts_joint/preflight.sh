#!/usr/bin/env bash
# Preflight a joint training arm on Linux: config+data dry-run, then a 1-epoch/stage
# smoke, then (optionally) the unit suite, then removal of the transient directories.
#
#   bash scripts_joint/preflight.sh <RUN_ID> [device] [--keep] [--tests]
#
#     <RUN_ID>   resolves configs_joint/cms_Joint_<RUN_ID>.yaml
#     device     auto (default) | cuda | cpu
#     --keep     keep the *_dryrun / *_smoke directories after success
#     --tests    run "python -m unittest discover -s tests" before the cleanup
#                (CLAUDE.md section 3 couples the two: a preflight directory is
#                throwaway only once the checks AND the tests are green)
#
#   exit 0  both green (and tests green, if requested)
#   exit 1  dry-run failed        (config, data, contract guard)
#   exit 2  smoke failed          (model build, loss wiring, one real update)
#   exit 3  environment problem   (no interpreter, no config, no data, no PyYAML)
#   exit 4  unit suite failed
#
# SERVER NOTES
#   * Run the SMOKE inside the GPU allocation. On Slurm that means
#     "srun --gres=gpu:1 bash scripts_joint/preflight.sh <RUN_ID>" or the same
#     line inside the sbatch body - not on a login node, where torch sees no GPU
#     and "--device cuda" aborts loudly (correct behaviour, not a bug).
#     The dry-run itself is CPU-only (config + data selection).
#   * The region cache key hashes the ABSOLUTE path, mtime_ns and size of the
#     inputs, so every machine builds its own cache. The 1000-sample caches a
#     dry-run/smoke create are NOT reused by a full run -- that is expected.
#   * Set LANG/PYTHONUTF8 yourself in a systemd unit or cron job (this script
#     exports them) or Python opens text files as ASCII.
#   * If HDF5 writes fail with errno 38 on a network/FUSE share, export
#     HDF5_USE_FILE_LOCKING=FALSE before running.
#   * git tracks these .sh files as mode 100644, so call it as
#     "bash scripts_joint/preflight.sh", never "./scripts_joint/preflight.sh".

set -uo pipefail

RUN_ID=""; DEVICE="auto"; KEEP=0; TESTS=0
for arg in "$@"; do
  case "$arg" in
    --keep)  KEEP=1 ;;
    --tests) TESTS=1 ;;
    auto|cuda|cpu) DEVICE="$arg" ;;
    -h|--help) sed -n '2,33p' "$0"; exit 0 ;;
    -*) echo "unknown option: $arg" >&2; exit 3 ;;
    *)  if [ -z "$RUN_ID" ]; then RUN_ID="$arg"; else echo "only one RUN_ID" >&2; exit 3; fi ;;
  esac
done
if [ -z "$RUN_ID" ]; then
  echo "usage: bash scripts_joint/preflight.sh <RUN_ID> [auto|cuda|cpu] [--keep] [--tests]" >&2
  exit 3
fi

# ---- repo root, interpreter, locale -----------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT" || exit 3

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if   command -v python  >/dev/null 2>&1; then PY=python
  elif command -v python3 >/dev/null 2>&1; then PY=python3
  else echo "FAIL: no python interpreter on PATH (activate your venv first)" >&2; exit 3
  fi
fi
export PYTHONUTF8=1
if [ -z "${LANG:-}" ]; then export LANG=en_US.UTF-8; fi

CONFIG="configs_joint/cms_Joint_${RUN_ID}.yaml"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_DRY="$LOG_DIR/preflight_${RUN_ID}_dryrun.log"
LOG_SMOKE="$LOG_DIR/preflight_${RUN_ID}_smoke.log"

echo "================================================================================"
echo " preflight $RUN_ID   $(date '+%Y-%m-%d %H:%M:%S')"
echo " repo   $REPO_ROOT"
echo " python $("$PY" -c 'import sys; print(sys.version.split()[0], sys.executable)')"
echo " device $DEVICE"
echo "================================================================================"

# ---- environment checks ------------------------------------------------------
if [ ! -f "$CONFIG" ]; then
  echo "FAIL: config not found: $CONFIG" >&2
  exit 3
fi
if [ ! -d data ]; then
  echo "FAIL: data/ is absent. It is gitignored and must be staged separately" >&2
  echo "      (the 2.1 GB Run2012BC_DoubleMuParked_Muons.root plus the priors)." >&2
  exit 3
fi
if [ ! -f data/Run2012BC_DoubleMuParked_Muons.root ]; then
  echo "WARN: data/Run2012BC_DoubleMuParked_Muons.root is missing; the dry-run fails"
  echo "      unless the config resolves its CMS input elsewhere."
fi
if ! "$PY" -c "import yaml" 2>/dev/null; then
  echo "FAIL: PyYAML is not installed in this interpreter." >&2
  echo "      It is a hard import in scripts/cms_data.py but is MISSING from" >&2
  echo "      requirements-cms.txt. Install it explicitly:" >&2
  echo "        $PY -m pip install PyYAML==6.0.3" >&2
  exit 3
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
else
  echo "GPU : nvidia-smi not found (use --device cpu for the dry-run, or allocate a GPU)"
fi
if ! "$PY" -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())" 2>/dev/null; then
  echo "WARN: could not import torch (is the venv activated?)"
fi
echo

# ---- 1. dry-run: config resolution + data contract ---------------------------
echo "--[1/3] dry-run ------------------------------------------------------------"
"$PY" scripts_joint/run_joint.py --run "$RUN_ID" --device "$DEVICE" --dry-run 2>&1 | tee "$LOG_DRY"
dry_status=${PIPESTATUS[0]}
if [ "$dry_status" -ne 0 ] || ! grep -q "dry-run passed" "$LOG_DRY"; then
  echo
  echo "FAIL: dry-run did not report success (exit $dry_status). Full log: $LOG_DRY" >&2
  exit 1
fi
echo "OK: dry-run passed (config + data contract). Log: $LOG_DRY"
echo

# ---- 2. smoke: one epoch per stage ------------------------------------------
echo "--[2/3] smoke (1 epoch/stage) ----------------------------------------------"
"$PY" scripts_joint/run_joint.py --run "$RUN_ID" --device "$DEVICE" --smoke 2>&1 | tee "$LOG_SMOKE"
smoke_status=${PIPESTATUS[0]}
if [ "$smoke_status" -ne 0 ] || ! grep -q "complete:" "$LOG_SMOKE"; then
  echo
  echo "FAIL: smoke did not complete (exit $smoke_status). Full log: $LOG_SMOKE" >&2
  exit 2
fi
if ! grep -q "Upsilon was not opened by this run" "$LOG_SMOKE"; then
  echo "WARN: the smoke did not print the Upsilon-exclusion line; verify by hand that"
  echo "      the locked Upsilon region stayed closed."
fi
echo "OK: smoke complete. Log: $LOG_SMOKE"
echo

# ---- 3. optional unit suite --------------------------------------------------
if [ "$TESTS" -eq 1 ]; then
  echo "--[3/3] unit suite ---------------------------------------------------------"
  if ! "$PY" -m unittest discover -s tests -v; then
    echo "FAIL: unit suite not green; keeping the preflight directories" >&2
    exit 4
  fi
  echo "OK: unit suite green"
  echo
else
  echo "--[3/3] unit suite skipped (pass --tests to run it) ------------------------"
  echo
fi

# ---- cleanup (CLAUDE.md section 3: preflight directories are transient) ------
if [ "$KEEP" -eq 1 ]; then
  echo "keeping preflight directories (--keep)"
else
  echo "-- cleanup -----------------------------------------------------------------"
  "$PY" scripts_joint/clean_preflight.py --root outputs --apply
fi
echo
echo "================================================================================"
echo " PREFLIGHT GREEN: $RUN_ID"
echo " next (inside the GPU allocation):"
echo "   cd $REPO_ROOT"
echo "   $PY scripts_joint/run_joint.py --run $RUN_ID --device $DEVICE"
echo "================================================================================"
