#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Launch an AlphaFlow experiment on a single GPU (RTX 5090 / Blackwell).
#
# Usage:
#   bash scripts/train_single_gpu.sh                                  # smoke test preset
#   bash scripts/train_single_gpu.sh alphaflow-sigmoid-latentspace-B-2-single-gpu
#   bash scripts/train_single_gpu.sh alphaflow-sigmoid-latentspace-B-2-single-gpu \
#        dataset.batch_gpu=16 training.max_steps=100000
#
# Everything after the experiment name is passed through to Hydra, so any
# config value can be overridden on the command line.
#
# The presets in infra/experiments/experiments-alphaflow.yaml already set
#   num_gpus=1, direct_launch=true, wandb.enabled=false
# so no `torchrun` / multi-GPU launcher is needed.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- Make `import src...` work regardless of how the script is called. ------
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# --- Project root used by configs/env/user.yaml (datasets/snapshots/output). --
export ALPHAFLOW_PROJECT_PATH="${ALPHAFLOW_PROJECT_PATH:-${REPO_ROOT}}"

# --- GPU memory allocator: reduces fragmentation on small free-VRAM slices. --
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# --- Put caches inside the repo (safe on shared servers / small home quotas). -
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${REPO_ROOT}/.cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${REPO_ROOT}/.cache}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

EXP="${1:-alphaflow-smoke-test-single-gpu}"
shift || true

mkdir -p experiments data

if [[ ! -f configs/env/user.yaml ]]; then
  echo "[ERROR] configs/env/user.yaml is missing."
  exit 1
fi

echo "=============================================================="
echo " Experiment : ${EXP}"
echo " Repo       : ${REPO_ROOT}"
echo " Python     : $(command -v python) ($(python -c 'import sys; print(sys.version.split()[0])'))"
echo " Extra args : $*"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo " GPU:"
  nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu \
             --format=csv,noheader | sed 's/^/   /'
fi
echo "=============================================================="
echo

exec python infra/launch.py "${EXP}" num_gpus=1 direct_launch=true wandb.enabled=false "$@"
