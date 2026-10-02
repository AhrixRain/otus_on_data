#!/bin/bash
#SBATCH -J otus-train
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -t 12:00:00
#SBATCH --account=CHANGE_ME_gpu
#SBATCH --output=logs/slurm-%x.%j.out
#SBATCH --error=logs/slurm-%x.%j.err
#SBATCH --mail-type=fail,end
#SBATCH --mail-user=CHANGE_ME@uci.edu
#
# OTUS joint training on UCI HPC3.
#
#   cd /pub/$USER/otus
#   mkdir -p logs
#   sbatch --export=ALL,OTUS_RUN=H,OTUS_MODE=smoke deploy/hpc3/train_joint.sbatch
#
# Repository facts this script is built around (all measured on the Windows
# workstation, see memory.md and the runbook):
#   * one process, one GPU: there is no DDP / torchrun / multiprocessing
#     anywhere under scripts_joint, so --gres=gpu:1 is the whole story;
#   * Run H peaked at 3.21 GB CUDA reserved and 2.11 GB allocated, so every
#     HPC3 GPU (16 GB V100 and up) is 5x oversized -- the config's
#     cuda_memory_limit_gb: 11 is renormalised per device by
#     run_joint.configure_cuda_memory_limit and needs no edit;
#   * Run H took 13,551 s (3.76 h) over 180 epochs on an RTX 4080 Laptop.
#     The 12 h limit covers a slower GPU, the first-run region-cache build,
#     and a resume after preemption.
#
# Set OTUS_MODE=train only when you mean it (CLAUDE.md section 3: no long
# training runs without explicit instruction). dry and smoke write to
# <run_name>_dryrun / <run_name>_smoke and cannot touch a real run directory.
#
# Environment overrides, all via --export:
#   OTUS_RUN     run id resolved against configs_joint/cms_Joint_<ID>.yaml (required)
#   OTUS_MODE    dry | smoke | train                (default: smoke)
#   OTUS_RESUME  1 to --resume a killed run's last_model.pt (train mode only)
#   OTUS_PULL    1 to git fetch + fast-forward before training (default: 1)
#   OTUS_ENV_PREFIX  conda prefix (default /pub/$USER/otus-conda/envs/otus-cms)
#
# This script never runs git add/commit/push. Committing run artifacts is a
# manual step on the cluster (runbook section 6).

set -uo pipefail

OTUS_RUN="${OTUS_RUN:-}"
OTUS_MODE="${OTUS_MODE:-smoke}"
OTUS_RESUME="${OTUS_RESUME:-0}"
OTUS_PULL="${OTUS_PULL:-1}"
OTUS_ENV_PREFIX="${OTUS_ENV_PREFIX:-/pub/${USER}/otus-conda/envs/otus-cms}"
CONDA_MODULE="${OTUS_CONDA_MODULE:-miniconda3/23.5.2}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}" || exit 1

mkdir -p logs
export OTUS_MODE
export OTUS_START_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
job_start_epoch="$(date +%s)"

echo "=============================================================="
echo "job        : ${SLURM_JOB_ID:-local} (${SLURM_JOB_NAME:-interactive})"
echo "node       : ${SLURMD_NODENAME:-$(hostname)}"
echo "partition  : ${SLURM_JOB_PARTITION:-unknown}"
echo "started    : $(date -Is)"
echo "repo       : ${REPO_ROOT}"
echo "commit     : ${OTUS_START_SHA}"
echo "run / mode : ${OTUS_RUN} / ${OTUS_MODE}"
echo "=============================================================="

case "${OTUS_MODE}" in
    dry|smoke|train) ;;
    *) echo "ERROR: OTUS_MODE must be dry, smoke or train (got '${OTUS_MODE}')" >&2; exit 1 ;;
esac
if [[ -z "${OTUS_RUN}" ]]; then
    echo "ERROR: OTUS_RUN is required, e.g. --export=ALL,OTUS_RUN=H,OTUS_MODE=smoke" >&2
    exit 1
fi
if [[ ! -x "${OTUS_ENV_PREFIX}/bin/python" ]]; then
    echo "ERROR: no environment at ${OTUS_ENV_PREFIX}. Run deploy/hpc3/env_create.sh first." >&2
    exit 1
fi

# ------------------------------------------------------- environment ------
# RCIC: load versioned modules inside the submit script, never in .bashrc, and
# never alongside conda. https://rcic.uci.edu/software/modules.html
module purge
module load "${CONDA_MODULE}"
if [[ -f "${HOME}/.mycondainit-23.5.2" ]]; then
    # shellcheck disable=SC1090
    source "${HOME}/.mycondainit-23.5.2"
else
    eval "$(conda shell.bash hook)"
fi
# shellcheck disable=SC1091
conda activate "${OTUS_ENV_PREFIX}"

# The torch wheels carry their own CUDA runtime, so no cuda/* module is loaded.
# Loading one can prepend an older libcudart/libstdc++ ahead of the wheel's.
echo "python     : $(command -v python)"
python -c "import torch;print('torch      :', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"

# ------------------------------------------------------- code freshness ----
# Optional fast-forward before training, skipped when there is no .git (the
# normal state for a plain scp/rsync copy) or when OTUS_PULL=0.
if [[ -d .git ]]; then
    if ! git diff --quiet || ! git diff --cached --quiet; then
        echo "WARNING: the working tree has uncommitted changes, so provenance.json"
        echo "         cannot name the exact code that ran:"
        git status --short
    fi
    if [[ "${OTUS_PULL}" == "1" ]]; then
        branch="$(git rev-parse --abbrev-ref HEAD)"
        if git fetch origin "${branch}" --quiet 2>/dev/null; then
            if ! git merge --ff-only "origin/${branch}" --quiet 2>/dev/null; then
                echo "ERROR: ${branch} has diverged from origin/${branch}. Reconcile it" >&2
                echo "       on the cluster by hand before training." >&2
                exit 1
            fi
            echo "code       : fast-forwarded to $(git rev-parse --short HEAD)"
        else
            echo "code       : git fetch failed (offline or no credentials); using the tree as-is"
        fi
    else
        echo "code       : OTUS_PULL=0, no fetch; using the working tree as-is"
    fi
else
    echo "code       : no .git directory (scp/rsync copy); skipping the fast-forward"
fi

# ------------------------------------------------------- train -------------
cmd=(python scripts_joint/run_joint.py --run "${OTUS_RUN}" --device cuda)
case "${OTUS_MODE}" in
    dry)   cmd+=(--dry-run) ;;
    smoke) cmd+=(--smoke) ;;
    train) : ;;
esac

# Resuming a preempted free-gpu job. run_joint.py --resume reads
# <output_dir>/last_model.pt and continues the configured stage schedule; it
# fails fast if that checkpoint is absent, which is the honest outcome for a
# job that died before its first validation.
if [[ "${OTUS_RESUME}" == "1" ]]; then
    if [[ "${OTUS_MODE}" != "train" ]]; then
        echo "WARNING: OTUS_RESUME=1 is only meaningful with OTUS_MODE=train; ignoring it." >&2
    else
        echo "note: resuming from the existing run directory's last_model.pt"
        cmd+=(--resume)
    fi
fi

echo
echo "==> ${cmd[*]}"
"${cmd[@]}"
train_status=$?
echo "==> run_joint.py exited ${train_status} after $(( $(date +%s) - job_start_epoch )) s"

# ------------------------------------------------------- hand-off -----------
# No git is run from this job. Committing and pushing the run artifacts is a
# deliberate manual step on the cluster -- see the runbook section 6 for the
# command sequence and for the one non-obvious part (outputs/**/*.json is
# gitignored, so a plain `git add` silently drops every metric file).
echo
echo "==> finished: $(date -Is)"
echo "==> artifacts (commit them by hand; this job does not touch git):"
echo "    outputs/cms_Joint/*/history.json            per-epoch metrics"
echo "    outputs/cms_Joint/*/joint_evaluation.json   final per-region report"
echo "    outputs/cms_Joint/*/provenance.json         code/env/cache provenance"
echo "    outputs/cms_Joint/*/plots*/                 figures"
echo "    logs/$(basename "${SLURM_JOB_NAME:-otus-train}").${SLURM_JOB_ID:-0}.out"
echo "==> suggested: cd $(pwd) && git status --short outputs/"
exit "${train_status}"
