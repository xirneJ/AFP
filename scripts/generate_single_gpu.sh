#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Generate samples from a released AlphaFlow checkpoint on a *single* GPU.
#
# 1) Download the checkpoints into ./snapshots/ (see docs/SINGLE_GPU.md):
#      snapshots/alphaflow-B-2-cfg.pt
#      snapshots/sd_vae_ft_ema.pt
#
# 2) Run:
#      bash scripts/generate_single_gpu.sh
#
# Environment variables (all optional):
#   CKPT        checkpoint path           (default: snapshots/alphaflow-B-2-cfg.pt)
#   OUT         output directory          (default: runs/<run>/generate/nfe<NFE> when the
#                                          checkpoint lives under runs/,
#                                          else runs/_released/<ckpt>/nfe<NFE>)
#   SEEDS       seeds to generate         (default: 0-7)
#   NFE         number of sampling steps  (default: 1; use 2 for the 2-NFE variant)
#   BATCH_SIZE  generation batch size     (default: 1; raise ~8 on a 32 GiB card)
#   CFG_SCALE   classifier-free guidance  (default: unset -> keep checkpoint config;
#                                                released -cfg checkpoints are usually
#                                                sampled with scale 1.0 because guidance
#                                                is baked into training)
#   DIT_DTYPE   cast DiT backbone to bf16 to save VRAM (default: unset; try bfloat16
#               if you only have ~2-3 GiB free)
#
# Any extra CLI args are forwarded to scripts/generate.py.
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
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

CKPT="${CKPT:-snapshots/alphaflow-B-2-cfg.pt}"
SEEDS="${SEEDS:-0-7}"
NFE="${NFE:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
CFG_SCALE="${CFG_SCALE:-}"
DIT_DTYPE="${DIT_DTYPE:-}"

# --- Where to write the generated images (under the unified experiments root) --
#   * checkpoint inside <repo>/runs/<run>/...  ->  runs/<run>/generate/nfe<NFE>/
#   * released checkpoint in ./snapshots/...   ->  runs/_released/<ckpt>/nfe<NFE>/
# Override explicitly with OUT=<dir>. (If you moved experiments_root in
# configs/env/user.yaml, pass OUT explicitly.)
EXPERIMENTS_ROOT="${ALPHAFLOW_EXPERIMENTS_ROOT:-${REPO_ROOT}/runs}"
if [[ -z "${OUT:-}" ]]; then
  CKPT_STEM="$(basename "${CKPT}")"
  CKPT_STEM="${CKPT_STEM%.*}"
  CKPT_DIR="$(cd "$(dirname "${CKPT}")" 2>/dev/null && pwd || true)"
  if [[ -n "${CKPT_DIR}" && "${CKPT_DIR}" == "${EXPERIMENTS_ROOT}"/* ]]; then
    REL="${CKPT_DIR#"${EXPERIMENTS_ROOT}"/}"
    RUN_NAME="${REL%%/*}"
    OUT="${EXPERIMENTS_ROOT}/${RUN_NAME}/generate/nfe${NFE}"
  else
    OUT="${EXPERIMENTS_ROOT}/_released/${CKPT_STEM}/nfe${NFE}"
  fi
fi
mkdir -p "${OUT}"

if [[ ! -f "${CKPT}" ]]; then
  echo "[ERROR] checkpoint not found: ${CKPT}"
  echo "        Download it first, see docs/SINGLE_GPU.md ('Download checkpoints')."
  exit 1
fi
if [[ ! -f snapshots/sd_vae_ft_ema.pt ]]; then
  echo "[WARN] snapshots/sd_vae_ft_ema.pt is missing."
  echo "       Latent-space (LDM) checkpoints need it to decode images."
fi

EXTRA=()
if [[ -n "${CFG_SCALE}" ]]; then
  EXTRA+=("sampling.cfg.scale=${CFG_SCALE}")
fi
if [[ -n "${DIT_DTYPE}" ]]; then
  EXTRA+=("inference_dtype=${DIT_DTYPE}")
fi

echo "=============================================================="
echo " Checkpoint : ${CKPT}"
echo " Output     : ${OUT}"
echo " Seeds      : ${SEEDS}  (batch_size=${BATCH_SIZE}, NFE=${NFE})"
echo " Extra      : ${EXTRA[*]:-none} $*"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo " GPU:"
  nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader | sed 's/^/   /'
fi
echo "=============================================================="
echo

exec python scripts/generate.py \
  ckpt.snapshot_path="${CKPT}" \
  output_dir="${OUT}" \
  seeds="${SEEDS}" \
  batch_size="${BATCH_SIZE}" \
  distributed=false \
  sampling=recflow \
  sampling.sigma_noise=1 \
  sampling.num_steps="${NFE}" \
  sampling.enable_trajectory_sampling=true \
  sampling.enable_consistency_sampling=false \
  hydra.run.dir="${EXPERIMENTS_ROOT}/_hydra/generate" \
  hydra.output_subdir=null \
  ${EXTRA[@]+"${EXTRA[@]}"} \
  "$@"
