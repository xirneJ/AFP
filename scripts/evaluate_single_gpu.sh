#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Evaluate a folder of generated images (FID / FDD / FCD) on a single GPU.
#
# Results are saved automatically under the unified experiments root:
#   runs/<run_name>/metrics/eval-<nfe>.json        (combined)
#   runs/<run_name>/metrics/metric-<metric>.jsonl  (append-only history)
# (This works because <gen_dir> is expected to be under runs/<run>/generate/...;
#  otherwise a `metrics/` folder is created next to the generated images.)
#
# Usage:
#   bash scripts/evaluate_single_gpu.sh runs/_released/alphaflow-B-2-cfg/nfe1
#   METRICS="fid50k_full" bash scripts/evaluate_single_gpu.sh <gen_dir>
#
# Environment variables (all optional):
#   METRICS   comma-separated metric names
#             (default: fid50k_full,dinofid50k_full,fcd50k_full)
#   RES       image resolution, default 256
#   DATA_SRC  if you only have the ImageNet val split, set this to it, e.g.
#             DATA_SRC=$PWD/data/imagenet/val/ (otherwise uses data/imagenet/val)
#   VAL_SPLIT if "0", uses the ImageNet train split as the real distribution
#             (the repo's default, matches the official evaluate.sh and is much
#             slower the first time; default here: 1 = standard FID vs val 50k)
#
# Any extra Hydra overrides are forwarded, e.g.
#   bash scripts/evaluate_single_gpu.sh <gen_dir> \
#     'metrics=[fid50k_full]' dataset.use_val_data_for_eval_stream=false
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export ALPHAFLOW_PROJECT_PATH="${ALPHAFLOW_PROJECT_PATH:-${REPO_ROOT}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${REPO_ROOT}/.cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${REPO_ROOT}/.cache}"
# China mirror + local cache by default (avoids long hf.co HEAD-request timeouts; SD-VAE is cached).
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

GEN_DIR="${1:-}"
if [[ -z "${GEN_DIR}" ]]; then
  echo "Usage: bash scripts/evaluate_single_gpu.sh <generated_image_dir> [hydra overrides...]" >&2
  exit 1
fi
shift || true

if [[ ! -d "${GEN_DIR}" ]]; then
  echo "[ERROR] generated image directory not found: ${GEN_DIR}" >&2
  exit 1
fi
GEN_DIR="$(cd "${GEN_DIR}" && pwd)"

METRICS="${METRICS:-fid50k_full,dinofid50k_full,fcd50k_full}"
RES="${RES:-256}"
VAL_SPLIT="${VAL_SPLIT:-1}"

EXTRA=()
if [[ -n "${DATA_SRC:-}" ]]; then
  EXTRA+=("dataset.src=${DATA_SRC}")
fi
if [[ "${VAL_SPLIT}" == "1" ]]; then
  EXTRA+=("dataset.use_val_data_for_eval_stream=true")
else
  EXTRA+=("dataset.use_val_data_for_eval_stream=false")
fi

echo "=============================================================="
echo " Generated dir : ${GEN_DIR}"
echo " Metrics       : ${METRICS}  (resolution ${RES}x${RES})"
echo " Real data     : $([[ "${VAL_SPLIT}" == "1" ]] && echo 'ImageNet val (standard FID)' || echo 'ImageNet train (repo default)')"
echo " Extra         : ${EXTRA[*]:-none} $*"
echo "=============================================================="
echo

exec python scripts/evaluate.py \
  "dataset.resolution=[1,${RES},${RES}]" \
  "gen_dataset.src=${GEN_DIR}/" \
  "gen_dataset.resolution=[1,${RES},${RES}]" \
  "gen_dataset.label_shape=[]" \
  "metrics=[${METRICS}]" \
  "hydra.run.dir=${REPO_ROOT}/runs/_hydra/eval" \
  hydra.output_subdir=null \
  ${EXTRA[@]+"${EXTRA[@]}"} \
  "$@"
