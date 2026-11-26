#!/usr/bin/env bash
# Stage 5: after the A0 gate passed, run A1 (state+shared) and A2 (state+EMA teacher) 20k each,
# sequentially, with the validated fine-tune recipe:
#   constant alpha=0, ema_rampup_ratio=0.05, Adam lr=1e-4, warmup=500, batch_gpu=32, 20k steps.
# Sample-image generation during training is disabled (it triggered an intermittent NCCL
# new_group failure at step 15000); fakes/FID are produced after training instead.
#
# Usage:
#   nohup bash scripts/run_stage5_a1_a2.sh > runs/_stage5_a1_a2.log 2>&1 &
#   tail -F runs/_stage5_a1_a2.log
#
# Options:
#   --min-free-mib 24000  required free VRAM before each arm
#   --max-util 20         max utilization to consider the GPU idle (100 disables)
#   --sustain-checks 2    consecutive ready polls
#   --poll-sec 30         polling interval
#   --batch-gpu 32
#   --stages a1,a2        subset/order
#   --no-wait             skip the idle-GPU wait (use only if you know the GPU is free)
#   --no-lock             skip the duplicate-instance lock

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MIN_FREE_MIB=24000
MAX_UTIL=20
SUSTAIN_CHECKS=2
POLL_SEC=30
BATCH_GPU=32
STAGES="a1,a2"
NO_WAIT=0
NO_LOCK=0
INIT="$REPO_ROOT/snapshots/alphaflow-B-2-finetune-init.pt"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --min-free-mib) MIN_FREE_MIB="$2"; shift 2 ;;
    --max-util) MAX_UTIL="$2"; shift 2 ;;
    --sustain-checks) SUSTAIN_CHECKS="$2"; shift 2 ;;
    --poll-sec) POLL_SEC="$2"; shift 2 ;;
    --batch-gpu) BATCH_GPU="$2"; shift 2 ;;
    --stages) STAGES="$2"; shift 2 ;;
    --no-wait) NO_WAIT=1; shift ;;
    --no-lock) NO_LOCK=1; shift ;;
    -h|--help) sed -n '1,30p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1"; exit 2 ;;
  esac
done

mkdir -p runs
log() { echo "[$(date '+%F %T')] $*"; }

LOCK_DIR=/tmp/alphaflow_stage5_a1a2.lockdir
if (( NO_LOCK == 0 )); then
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    echo $$ > "$LOCK_DIR/pid"; trap 'rm -rf "$LOCK_DIR"' EXIT INT TERM
  else
    other="$(cat "$LOCK_DIR/pid" 2>/dev/null || true)"
    if [[ -n "$other" ]] && kill -0 "$other" 2>/dev/null; then
      log "Another run_stage5_a1_a2.sh instance (pid ${other}) is running. Exiting."; exit 1
    fi
    log "Removing stale lock directory."; rm -rf "$LOCK_DIR"
    mkdir "$LOCK_DIR" 2>/dev/null && { echo $$ > "$LOCK_DIR/pid"; trap 'rm -rf "$LOCK_DIR"' EXIT INT TERM; }
  fi
fi

gpu_free_mib() { nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ' || true; }
gpu_util_pct() { nvidia-smi -i 0 --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ' || true; }

wait_for_gpu() {
  local phase="$1" streak=0 free util
  if (( NO_WAIT == 1 )); then
    log "[$phase] --no-wait given; starting immediately."; return 0
  fi
  log "[$phase] waiting for idle GPU: free >= ${MIN_FREE_MIB} MiB, util <= ${MAX_UTIL}%, ${SUSTAIN_CHECKS} consecutive polls."
  while true; do
    free="$(gpu_free_mib)"; util="$(gpu_util_pct)"
    if [[ "$free" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ ]] \
       && (( free >= MIN_FREE_MIB )) && (( MAX_UTIL >= 100 || util <= MAX_UTIL )); then
      streak=$((streak + 1))
      log "[$phase] poll: free=${free} MiB util=${util}% -> ready (${streak}/${SUSTAIN_CHECKS})"
      if (( streak >= SUSTAIN_CHECKS )); then
        log "[$phase] GPU is idle enough. Proceeding."; return 0
      fi
    else
      streak=0
      log "[$phase] poll: free=${free} MiB util=${util}% -> not ready"
    fi
    sleep "$POLL_SEC"
  done
}

if [[ ! -f "$INIT" ]]; then
  log "Creating the unified fine-tune snapshot (CPU only)..."
  python scripts/make_finetune_snapshot.py snapshots/alphaflow-B-2.pt "$INIT" || { log "ERROR: failed to create $INIT"; exit 1; }
fi

COMMON=(
  "training.resume.on_start_ckpt_path=$INIT"
  "training.resume.whole_state=false"
  "model.ema_rampup_ratio=0.05"
  "model.lr_scheduler.num_warmup_steps=500"
  "training.max_steps=20000"
  "training.steps_per_tick=10"
  "training.freqs.metrics=null"
  "training.freqs.metrics_rare=null"
  "training.metrics_final=[]"
  "training.freqs.loss_val=null"
  "training.freqs.loss_per_sigma=null"
  "training.freqs.snapshot=null"
  "training.freqs.snapshot_latest=1000"
  "training.freqs.save_samples=null"
  "dataset.batch_gpu=$BATCH_GPU"
  "loss.alpha.scheduler=constant"
  "loss.alpha.initial_value=0.0"
)

run_arm() {
  local desc="$1"; shift
  local run_dir="runs/$desc"
  if compgen -G "$run_dir/snapshot-*.pt" > /dev/null; then
    log "ERROR: $run_dir already has snapshots; delete it first for a fresh run."; exit 1
  fi
  log "Launching ${desc} (batch_gpu=${BATCH_GPU}, steps=20000, constant alpha=0, ramped EMA)..."
  if ! bash scripts/train_single_gpu.sh alphaflow-sigmoid-latentspace-B-2-single-gpu \
      "desc=${desc}" "${COMMON[@]}" "$@" > "runs/_${desc}.out" 2>&1; then
    log "ERROR: ${desc} failed; see runs/_${desc}.out"; exit 1
  fi
  log "${desc} finished OK."
}

IFS=',' read -r -a ARM_LIST <<< "$STAGES"
for arm in "${ARM_LIST[@]}"; do
  case "$arm" in
    a1)
      wait_for_gpu "a1"
      run_arm "stage5-a1-20k-a0const-ramp" \
        "model.parameterization=state" "training.resume.allow_missing_keys=true"
      ;;
    a2)
      wait_for_gpu "a2"
      run_arm "stage5-a2-20k-a0const-ramp" \
        "model.parameterization=state" "loss.target_branch=ema" "training.resume.allow_missing_keys=true"
      ;;
    *) log "ERROR: unknown arm '$arm' (supported: a1,a2)"; exit 2 ;;
  esac
done

log "All requested arms finished."
for d in runs/stage5-a1-20k-a0const-ramp runs/stage5-a2-20k-a0const-ramp; do
  [[ -d "$d" ]] || continue
  log "--- $d ---"
  ls -lh "$d"/snapshot-*.pt 2>/dev/null || true
  grep 'step ' "$d/log.txt" 2>/dev/null | tail -n 2 || true
done
