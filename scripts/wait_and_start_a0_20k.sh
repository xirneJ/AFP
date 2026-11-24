#!/usr/bin/env bash
# Waits until the GPU is idle, then starts the Stage-5 A0 20k run (constant alpha=0 + ramped EMA,
# batch_gpu=32) automatically.
#
# Idle condition (polled every 30 s by default):
#   memory.free >= 26000 MiB  AND  utilization.gpu <= 20 %  for 2 consecutive polls.
# The b32 run peaks at ~20.5 GiB reserved (step-1 init peak); 26 GiB free leaves headroom.
#
# Usage:
#   nohup bash scripts/wait_and_start_a0_20k.sh > runs/_wait_a0_20k.log 2>&1 &
#   tail -F runs/_wait_a0_20k.log
#
# Options:
#   --min-free-mib 26000   required free VRAM
#   --max-util 20          required idle utilization (%); 100 disables the check
#   --sustain-checks 2     consecutive ready polls required
#   --poll-sec 30          polling interval
#   --desc stage5-a0-20k-a0const-ramp
#   --no-lock              skip the duplicate-instance lock

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MIN_FREE_MIB=26000
MAX_UTIL=20
SUSTAIN_CHECKS=2
POLL_SEC=30
DESC="stage5-a0-20k-a0const-ramp"
NO_LOCK=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --min-free-mib) MIN_FREE_MIB="$2"; shift 2 ;;
    --max-util) MAX_UTIL="$2"; shift 2 ;;
    --sustain-checks) SUSTAIN_CHECKS="$2"; shift 2 ;;
    --poll-sec) POLL_SEC="$2"; shift 2 ;;
    --desc) DESC="$2"; shift 2 ;;
    --no-lock) NO_LOCK=1; shift ;;
    -h|--help) sed -n '1,32p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1"; exit 2 ;;
  esac
done

mkdir -p runs
log() { echo "[$(date '+%F %T')] $*"; }

LOCK_DIR=/tmp/alphaflow_wait_a0.lockdir
if (( NO_LOCK == 0 )); then
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    echo $$ > "$LOCK_DIR/pid"; trap 'rm -rf "$LOCK_DIR"' EXIT INT TERM
  else
    other="$(cat "$LOCK_DIR/pid" 2>/dev/null || true)"
    if [[ -n "$other" ]] && kill -0 "$other" 2>/dev/null; then
      log "Another wait_and_start_a0_20k.sh instance (pid ${other}) is running. Exiting."
      exit 1
    fi
    log "Removing stale lock directory."; rm -rf "$LOCK_DIR"
    mkdir "$LOCK_DIR" 2>/dev/null && { echo $$ > "$LOCK_DIR/pid"; trap 'rm -rf "$LOCK_DIR"' EXIT INT TERM; }
  fi
fi

gpu_free_mib() { nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ' || true; }
gpu_util_pct() { nvidia-smi -i 0 --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ' || true; }

INIT="$REPO_ROOT/snapshots/alphaflow-B-2-finetune-init.pt"
if [[ ! -f "$INIT" ]]; then
  log "Creating the unified fine-tune snapshot (CPU only)..."
  python scripts/make_finetune_snapshot.py snapshots/alphaflow-B-2.pt "$INIT" || { log "ERROR: failed to create $INIT"; exit 1; }
fi

# Refuse to silently resume an old run with the same desc: require a clean run dir.
RUN_DIR="runs/$DESC"
if compgen -G "$RUN_DIR/snapshot-*.pt" > /dev/null; then
  log "ERROR: $RUN_DIR already has snapshots; delete it first if you really want a fresh run."; exit 1
fi

log "Waiting for idle GPU: free >= ${MIN_FREE_MIB} MiB, util <= ${MAX_UTIL}%, ${SUSTAIN_CHECKS} consecutive polls, poll every ${POLL_SEC}s."
streak=0
while true; do
  free="$(gpu_free_mib)"; util="$(gpu_util_pct)"
  if [[ "$free" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ ]] \
     && (( free >= MIN_FREE_MIB )) \
     && (( MAX_UTIL >= 100 || util <= MAX_UTIL )); then
    streak=$((streak + 1))
    log "poll: free=${free} MiB util=${util}% -> ready (${streak}/${SUSTAIN_CHECKS})"
    if (( streak >= SUSTAIN_CHECKS )); then
      log "GPU is idle enough. Starting the A0 20k run."
      break
    fi
  else
    streak=0
    log "poll: free=${free} MiB util=${util}% -> not ready"
  fi
  sleep "$POLL_SEC"
done

OUT_LOG="$REPO_ROOT/runs/_stage5_a0_20k.out"
log "Launching ${DESC} (batch_gpu=32, 20000 steps). Training output -> ${OUT_LOG}"
if bash scripts/train_single_gpu.sh alphaflow-sigmoid-latentspace-B-2-single-gpu \
    "desc=${DESC}" \
    "training.resume.on_start_ckpt_path=${INIT}" \
    "training.resume.whole_state=false" \
    "model.ema_rampup_ratio=0.05" \
    "model.lr_scheduler.num_warmup_steps=500" \
    "training.max_steps=20000" \
    "training.steps_per_tick=10" \
    "training.freqs.metrics=null" \
    "training.freqs.metrics_rare=null" \
    "training.metrics_final=[]" \
    "training.freqs.loss_val=null" \
    "training.freqs.loss_per_sigma=null" \
    "training.freqs.snapshot=10000" \
    "training.freqs.snapshot_latest=1000" \
    "training.freqs.save_samples=5000" \
    "dataset.batch_gpu=32" \
    "loss.alpha.scheduler=constant" \
    "loss.alpha.initial_value=0.0" \
    > "$OUT_LOG" 2>&1; then
  log "A0 20k finished OK. Snapshots: $(ls -1 "$RUN_DIR"/snapshot-*.pt 2>/dev/null | tr '\n' ' ')"
else
  log "ERROR: A0 20k failed; see $OUT_LOG"
  exit 1
fi
log "Done. Next: evaluate fid2048 at snapshot-00001000 / snapshot-00020000 (EMA + student)."
