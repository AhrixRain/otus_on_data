#!/bin/bash
#SBATCH --job-name=otus-grid
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/grid-%j.out
#
# Launch several joint training arms on ONE GPU, in parallel.
#
#   bash scripts_joint/run_grid.sh                 # on an allocated node
#   sbatch scripts_joint/run_grid.sh               # the same file as a Slurm job
#   bash scripts_joint/run_grid.sh H_kneeStd ...    # any subset of arms
#
# Why parallel is safe here (measured, memory.md Session 78/79):
#   * one run peaks at 3.27 GiB, the card has ~96 GiB;
#   * 4080 Laptop 81 s/epoch -> RTX PRO 6000 35.7 s/epoch is only 2.3x while the
#     card is 6-8x faster, so the pipeline is launch-bound, not compute-bound;
#   * HPC3 charges per GPU-hour, not per process, so extra arms on the same GPU
#     are free until the GPU saturates.
# Check saturation with: nvidia-smi --query-gpu=utilization.gpu,memory.used -l 5
#
# Every arm writes its own outputs/cms_Joint/<run_name>/ and its own log; the
# region cache is read-only here (already warm from the readiness test), so the
# processes do not contend for it.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

ARMS=("${@:-H_kneeStd H_kneeStdA H_kneeStdTail10 H_kneeStdTail00}")
PY="${PYTHON:-python}"
command -v "$PY" >/dev/null 2>&1 || PY=python3
mkdir -p logs

echo "=============================================================="
echo " grid: ${ARMS[*]}"
echo " host: $(hostname)   python: $("$PY" -c 'import sys;print(sys.version.split()[0])')"
echo " gpu : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo " start: $(date '+%F %T')"
echo "=============================================================="

pids=()
names=()
for arm in "${ARMS[@]}"; do
  log="logs/${arm}.log"
  echo "launching $arm -> $log"
  "$PY" scripts_joint/run_joint.py --run "$arm" --device cuda > "$log" 2>&1 &
  pids+=("$!")
  names+=("$arm")
done

failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "OK    ${names[$i]}"
  else
    code=$?
    echo "FAIL  ${names[$i]} (exit $code) - see logs/${names[$i]}.log"
    failed=$((failed + 1))
  fi
done

echo "=============================================================="
echo " end: $(date '+%F %T')   failures: $failed"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo " readouts (run after all arms finish):"
for arm in "${names[@]}"; do
  echo "   python scripts_joint/response_scorecard.py --run-dir outputs/cms_Joint/Run_${arm} --all-checkpoints --device cuda"
done
echo "   python scripts_joint/identity_baseline.py --run-dir outputs/cms_Joint/Run_${arm}"
echo "=============================================================="
exit $((failed > 0))
