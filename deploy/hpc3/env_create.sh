#!/bin/bash
# Create the OTUS training environment on UCI HPC3.
#
# RUN THIS INSIDE AN INTERACTIVE COMPUTE NODE, never on a login node.
# Installing into a login node is banned by the HPC3 acceptable-use policy
# ("Any R or conda/mamba installation of packages or environments",
# https://rcic.uci.edu/account/acceptable-use.html) and RCIC states the
# install "will likely fail" there.
#
#   ssh <UCInetID>@hpc3.rcic.uci.edu
#   srun -c 4 -p free --mem=16G --time=01:00:00 --pty /bin/bash -i
#   cd /pub/$USER/otus && bash deploy/hpc3/env_create.sh
#
# The environment is a *prefix* environment under DFS (/pub), not a named
# environment under $HOME, for two reasons:
#   * the torch cu126 wheels + pip cache are several GB and RCIC forbids
#     storing large data in $HOME (50 GB quota, shared with snapshots);
#   * a /pub/$USER path is identical on every compute node, so the batch
#     script never has to guess where the environment lives.
#
# Re-running it is safe: an existing prefix is reused, the requirements are
# re-applied.

set -euo pipefail

OTUS_ENV_PREFIX="${OTUS_ENV_PREFIX:-/pub/${USER}/otus-conda/envs/otus-cms}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/pub/${USER}/otus-conda/pip-cache}"
REPO_ROOT="${OTUS_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CONDA_MODULE="${OTUS_CONDA_MODULE:-miniconda3/23.5.2}"

echo "host          : $(hostname)"
echo "repo          : ${REPO_ROOT}"
echo "env prefix    : ${OTUS_ENV_PREFIX}"
echo "pip cache     : ${PIP_CACHE_DIR}"
echo "conda module  : ${CONDA_MODULE}"
echo

if [[ "$(hostname)" == login-* ]]; then
    echo "ERROR: this is a login node. Re-run inside 'srun -c 4 -p free --mem=16G --time=01:00:00 --pty /bin/bash -i'." >&2
    exit 1
fi

# RCIC: never load a python or other module alongside conda, and always load a
# versioned module name. https://rcic.uci.edu/software/modules.html
module purge
module load "${CONDA_MODULE}"

# RCIC's recommended pattern: keep the 'conda initialize' block out of
# .bashrc and source it explicitly. Fall back to the shell hook if the user
# has not done that yet.
if [[ -f "${HOME}/.mycondainit-23.5.2" ]]; then
    # shellcheck disable=SC1090
    source "${HOME}/.mycondainit-23.5.2"
else
    eval "$(conda shell.bash hook)"
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda is not on PATH after loading ${CONDA_MODULE}." >&2
    exit 1
fi

mkdir -p "$(dirname "${OTUS_ENV_PREFIX}")" "${PIP_CACHE_DIR}"

if [[ ! -x "${OTUS_ENV_PREFIX}/bin/python" ]]; then
    echo "==> creating prefix environment (python 3.10, matching the Windows host)"
    conda create -y -p "${OTUS_ENV_PREFIX}" python=3.10
else
    echo "==> reusing existing prefix environment"
fi

# shellcheck disable=SC1091
conda activate "${OTUS_ENV_PREFIX}"

python -m pip install --upgrade pip
python -m pip install \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    -r "${REPO_ROOT}/requirements-cms.txt"

# Keep $HOME light: RCIC warns that a dirty conda cache "can reasult in $HOME
# overquota" (https://rcic.uci.edu/software/user-installed.html).
conda clean -a -f -y

echo
echo "==> environment report"
python - <<'PY'
import sys

import torch

print("python        :", sys.version.split()[0])
print("torch         :", torch.__version__, "(cuda", torch.version.cuda + ")")
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    capability = torch.cuda.get_device_capability(0)
    print("device        :", torch.cuda.get_device_name(0), "sm_%d%d" % capability)
print("built arches  :", ", ".join(torch.cuda.get_arch_list()))
PY
echo
echo "Environment ready. Activate it in every batch job with:"
echo "  module purge && module load ${CONDA_MODULE}"
echo "  conda activate ${OTUS_ENV_PREFIX}"
