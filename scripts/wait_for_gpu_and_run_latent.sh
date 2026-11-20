#!/usr/bin/env bash
# Waits until the GPU has enough free memory (and is not busy), then runs the real-latent
# 50-step profiling for Corrected Velocity (A0) and Corrected State (A1).
#
# Nothing is started until the GPU is actually free, so this is safe to leave running in the
# background while another user's job occupies the card.
#
# Usage:
#   chmod +x scripts/wait_for_gpu_and_run_latent.sh
#   nohup bash scripts/wait_for_gpu_and_run_latent.sh > runs/_wait_gpu.log 2>&1 &
#   tail -f runs/_wait_gpu.log
#
# Options:
#   --min-free-mib 16384    required free VRAM (MiB), default 16 GiB
#   --max-util 40           max GPU utilization (%) to consider the card idle; 100 disables the check
#   --sustain-checks 3      consecutive successful polls required before starting
#   --poll-sec 60           polling interval
#   --batch-gpu 32          dataset.batch_gpu (global batch stays 256; lower it if memory is tight)
#   --run-1k                after the 50-step runs, also run the 1k-step A0/A1 sanity (default: off)
#   --stages 50            comma-separated stages: 50,1k (e.g. --stages 50,1k)
#
# The script re-checks the GPU before each arm: if another process grabbed the memory in the
# meantime, it goes back to waiting instead of risking OOM.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MIN_FREE_MIB=16384
MAX_UTIL=40
SUSTAIN_CHECKS=3
POLL_SEC=60
BATCH_GPU=32
STAGES="50"
GPU_ID=0
PRESET="alphaflow-sigmoid-latentspace-B-2-single-gpu"
INIT="$REPO_ROOT/snapshots/alphaflow-B-2-finetune-init.pt"
ALPHA_OVERRIDES=(
  "loss.alpha.scheduler=sigmoid"
  "loss.alpha.initial_value=1.0"
  "loss.alpha.end_value=0"
  "loss.alpha.change_init_steps=0"
  "loss.alpha.change_end_steps=1200"
  "loss.alpha.clamp_value=0.005"
  "loss.alpha.gamma=25.0"
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --min-free-mib) MIN_FREE_MIB="$2"; shift 2 ;;
    --max-util) MAX_UTIL="$2"; shift 2 ;;
    --sustain-checks) SUSTAIN_CHECKS="$2"; shift 2 ;;
    --poll-sec) POLL_SEC="$2"; shift 2 ;;
    --batch-gpu) BATCH_GPU="$2"; shift 2 ;;
    --stages) STAGES="$2"; shift 2 ;;
    --gpu-id) GPU_ID="$2"; shift 2 ;;
    --init) INIT="$2"; shift 2 ;;
    --preset) PRESET="$2"; shift 2 ;;
    -h|--help) sed -n '1,40p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1"; exit 2 ;;
  esac
done

mkdir -p runs
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

log() { echo "[$(date '+%F %T')] $*"; }

# Prevent double launches.
LOCK_FILE=/tmp/alphaflow_wait_gpu.lock
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  log "Another wait_for_gpu_and_run_latent.sh instance is already running. Exiting."
  exit 1
fi

gpu_free_mib() {
  nvidia-smi -i "$GPU_ID" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ' || true
}

gpu_util_pct() {
  nvidia-smi -i "$GPU_ID" --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ' || true
}

gpu_ready() {
  local free util
  free="$(gpu_free_mib)"
  util="$(gpu_util_pct)"
  if [[ ! "$free" =~ ^[0-9]+$ || ! "$util" =~ ^[0-9]+$ ]]; then
    return 1
  fi
  if (( free < MIN_FREE_MIB )); then
    return 1
  fi
  if (( MAX_UTIL < 100 && util > MAX_UTIL )); then
    return 1
  fi
  return 0
}

wait_for_gpu() {
  local phase="$1" streak=0 free util
  log "Waiting for GPU ${GPU_ID}: need free >= ${MIN_FREE_MIB} MiB, util <= ${MAX_UTIL}% (or 100 to disable), ${SUSTAIN_CHECKS} consecutive polls."
  while true; do
    free="$(gpu_free_mib)"; util="$(gpu_util_pct)"
    if gpu_ready; then
      streak=$((streak + 1))
      log "[$phase] poll: free=${free} MiB util=${util}% -> ready (${streak}/${SUSTAIN_CHECKS})"
      if (( streak >= SUSTAIN_CHECKS )); then
        log "[$phase] GPU is free enough. Proceeding."
        return 0
      fi
    else
      streak=0
      log "[$phase] poll: free=${free} MiB util=${util}% -> not ready"
    fi
    sleep "$POLL_SEC"
  done
}

# --- prerequisites -------------------------------------------------------
if [[ ! -d data/imagenet/train ]]; then
  log "ERROR: data/imagenet/train is missing. Run the data preparation first (docs/LATENT_VALIDATION_PLAN.md)."
  exit 1
fi

if [[ ! -f "$INIT" ]]; then
  log "Creating the unified fine-tune snapshot (CPU only): $INIT"
  python scripts/make_finetune_snapshot.py snapshots/alphaflow-B-2.pt "$INIT" || {
    log "ERROR: failed to create $INIT"; exit 1; }
fi

COMMON_OVERRIDES() {
  local steps="$1" tick="$2" save_samples="$3"
  COMMON=(
    "training.resume.on_start_ckpt_path=$INIT"
    "training.resume.whole_state=false"
    "model.ema_rampup_ratio=null"
    "training.max_steps=$steps"
    "training.steps_per_tick=$tick"
    "training.freqs.metrics=null"
    "training.freqs.metrics_rare=null"
    "training.metrics_final=[]"
    "training.freqs.loss_val=null"
    "training.freqs.loss_per_sigma=null"
    "training.freqs.snapshot=null"
    "training.freqs.snapshot_latest=null"
    "training.freqs.save_samples=$save_samples"
    "dataset.batch_gpu=$BATCH_GPU"
    "${ALPHA_OVERRIDES[@]}"
  )
}

run_arm() {
  local desc="$1" stages="$2"; shift 2
  log "Launching ${desc} (stages=${stages})..."
  if ! bash scripts/train_single_gpu.sh "$PRESET" \
      "desc=$desc" \
      "${COMMON[@]}" \
      "$@"; then
    log "ERROR: ${desc} failed. Stopping."
    exit 1
  fi
  log "${desc} finished."
}

# --- stages --------------------------------------------------------------
IFS=',' read -r -a STAGE_LIST <<< "$STAGES"
for stage in "${STAGE_LIST[@]}"; do
  case "$stage" in
    50)
      COMMON_OVERRIDES 50 5 null
      wait_for_gpu "50-step"
      run_arm "stage1-latent-speed-velocity" "50"
      wait_for_gpu "50-step"
      run_arm "stage1-latent-speed-state" "50" \
        "model.parameterization=state" "training.resume.allow_missing_keys=true"
      ;;
    1k)
      COMMON_OVERRIDES 1000 10 1000
      wait_for_gpu "1k-velocity"
      run_arm "stage1-latent-1k-velocity" "1k"
      wait_for_gpu "1k-state"
      run_arm "stage1-latent-1k-state" "1k" \
        "model.parameterization=state" "training.resume.allow_missing_keys=true"
      log "Running Part II summary..."
      python scripts/summarize_stage1_sanity.py \
        --velocity-dir runs/stage1-latent-1k-velocity \
        --state-dir runs/stage1-latent-1k-state \
        --expected-steps 1000 || true
      ;;
    *)
      log "ERROR: unknown stage '$stage' (supported: 50,1k)"; exit 2 ;;
  esac
done

log "=== SUMMARY (50-step tick lines) ==="
for d in runs/stage1-latent-speed-velocity runs/stage1-latent-speed-state; do
  if [[ -f "$d/log.txt" ]]; then
    log "--- $d/log.txt ---"
    grep 'step ' "$d/log.txt" | tail -n 12 || true
  fi
done
log "All requested stages finished. Review the logs, then decide about the next stage."
