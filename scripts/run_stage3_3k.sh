#!/usr/bin/env bash
# Runs the three A0/A1/A2 experiments sequentially on a single GPU (default 3k steps; reusable for
# the 20k stage via --steps 20000 --warmup 500 --run-name-prefix stage4).
#
#   A0: velocity + shared target  (no teacher)
#   A1: state    + shared target  (no teacher)
#   A2: state    + EMA teacher    (loss.target_branch=ema)
#
# All three use the SAME released init snapshot, the SAME proportional alpha schedule
# (change_end_steps=<steps>), the SAME LR warmup, the SAME batch size and the SAME
# logging/checkpoint settings. The ONLY intended variables are the parameterization and
# the target branch.
#
# Resume support: if runs/<desc>/snapshot-*.pt already exists, that arm resumes from the latest
# snapshot with whole_state=true (optimizer/scheduler/EMA/teacher restored); otherwise it starts
# fresh from --init with whole_state=false. For long 20k runs use --snapshot-latest 1000 so an
# interrupted arm can be resumed by simply rerunning the same command.
#
# Usage:
#   nohup bash scripts/run_stage3_3k.sh --batch-gpu 16 > runs/_stage3_3k.log 2>&1 &
#   # 20k:
#   nohup bash scripts/run_stage3_3k.sh --steps 20000 --warmup 500 --save-samples 5000 \
#     --snapshot-latest 1000 --run-name-prefix stage4 --batch-gpu 16 \
#     > runs/_stage4_20k.log 2>&1 &
#
# Options:
#   --batch-gpu 16        dataset.batch_gpu (must match the profiling decision; default 16)
#   --steps 3000          total optimizer steps per arm
#   --warmup 75           LR warmup steps (official ratio: 2.5% of budget)
#   --save-samples 1000   freqs.save_samples (fakes at multiples; null disables)
#   --snapshot-latest null  freqs.snapshot_latest (e.g. 1000 for resumable long runs)
#   --desc-suffix 3k      run-dir suffix (default: <steps/1000>k)
#   --min-free-mib 24000  required free VRAM before starting (idle GPU)
#   --max-util 20         max GPU utilization (%) to consider the card idle (100 disables)
#   --sustain-checks 3    consecutive ready polls required
#   --poll-sec 60         polling interval
#   --stages a0,a1,a2     subset/order of arms to run (default: a0,a1,a2)
#   --run-name-prefix stage3   desc prefix (desc=<prefix>-a0-latent-<suffix>, ...)
#   --init PATH           init snapshot (default snapshots/alphaflow-B-2-finetune-init.pt)
#   --no-lock             skip the duplicate-instance lock (only if you are sure)
#
# After the runs the script writes runs/_<prefix>_report.txt and prints it.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MIN_FREE_MIB=24000
MAX_UTIL=20
SUSTAIN_CHECKS=3
POLL_SEC=60
BATCH_GPU=16
STAGES="a0,a1,a2"
PREFIX="stage3"
INIT="$REPO_ROOT/snapshots/alphaflow-B-2-finetune-init.pt"
PRESET="alphaflow-sigmoid-latentspace-B-2-single-gpu"
STEPS=3000
TICK=10
WARMUP=75
SAVE_SAMPLES=1000
SNAPSHOT_LATEST=null
DESC_SUFFIX=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --batch-gpu) BATCH_GPU="$2"; shift 2 ;;
    --min-free-mib) MIN_FREE_MIB="$2"; shift 2 ;;
    --max-util) MAX_UTIL="$2"; shift 2 ;;
    --sustain-checks) SUSTAIN_CHECKS="$2"; shift 2 ;;
    --poll-sec) POLL_SEC="$2"; shift 2 ;;
    --stages) STAGES="$2"; shift 2 ;;
    --run-name-prefix) PREFIX="$2"; shift 2 ;;
    --init) INIT="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --save-samples) SAVE_SAMPLES="$2"; shift 2 ;;
    --snapshot-latest) SNAPSHOT_LATEST="$2"; shift 2 ;;
    --desc-suffix) DESC_SUFFIX="$2"; shift 2 ;;
    --no-lock) NO_LOCK=1; shift ;;
    -h|--help) sed -n '1,40p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1"; exit 2 ;;
  esac
done

NO_LOCK="${NO_LOCK:-0}"
DESC_SUFFIX="${DESC_SUFFIX:-$((STEPS / 1000))k}"

mkdir -p runs
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

log() { echo "[$(date '+%F %T')] $*"; }

# Directory-based lock: unlike flock on an inherited fd, a child process (e.g. `sleep`) cannot keep
# the lock alive after the main script is killed. Stale locks are detected via the PID file.
LOCK_DIR=/tmp/alphaflow_stage3_3k.lockdir
acquire_lock() {
  local pidfile="${LOCK_DIR}/pid" other
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    echo $$ > "$pidfile"
    trap 'rm -rf "$LOCK_DIR"' EXIT INT TERM
    return 0
  fi
  other="$(cat "$pidfile" 2>/dev/null || true)"
  if [[ -n "$other" ]] && kill -0 "$other" 2>/dev/null; then
    log "Another run_stage3_3k.sh instance (pid ${other}) is already running. Exiting."
    log "(If you are sure there is none, rerun with --no-lock.)"
    return 1
  fi
  log "Removing stale lock directory (pid ${other:-unknown} is not alive)."
  rm -rf "$LOCK_DIR"
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    echo $$ > "$pidfile"
    trap 'rm -rf "$LOCK_DIR"' EXIT INT TERM
    return 0
  fi
  log "Failed to acquire the lock directory."; return 1
}

if (( NO_LOCK == 0 )); then
  acquire_lock || exit 1
else
  log "WARNING: --no-lock given; skipping the duplicate-instance lock."
fi

gpu_free_mib() { nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ' || true; }
gpu_util_pct() { nvidia-smi -i 0 --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ' || true; }

gpu_ready() {
  local free util
  free="$(gpu_free_mib)"; util="$(gpu_util_pct)"
  [[ "$free" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ ]] || return 1
  (( free >= MIN_FREE_MIB )) || return 1
  (( MAX_UTIL >= 100 || util <= MAX_UTIL )) || return 1
  return 0
}

wait_for_gpu() {
  local phase="$1" streak=0 free util
  log "Waiting for GPU 0: free >= ${MIN_FREE_MIB} MiB, util <= ${MAX_UTIL}% (or 100 to disable), ${SUSTAIN_CHECKS} consecutive polls."
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
  log "ERROR: data/imagenet/train is missing."; exit 1
fi
if [[ ! -f "$INIT" ]]; then
  log "Creating the unified fine-tune snapshot (CPU only): $INIT"
  python scripts/make_finetune_snapshot.py snapshots/alphaflow-B-2.pt "$INIT" || { log "ERROR: failed to create $INIT"; exit 1; }
fi

COMMON=(
  "model.ema_rampup_ratio=null"
  "model.lr_scheduler.num_warmup_steps=$WARMUP"
  "training.max_steps=$STEPS"
  "training.steps_per_tick=$TICK"
  "training.freqs.metrics=null"
  "training.freqs.metrics_rare=null"
  "training.metrics_final=[]"
  "training.freqs.loss_val=null"
  "training.freqs.loss_per_sigma=null"
  "training.freqs.snapshot=null"
  "training.freqs.snapshot_latest=$SNAPSHOT_LATEST"
  "training.freqs.save_samples=$SAVE_SAMPLES"
  "dataset.batch_gpu=$BATCH_GPU"
  "loss.alpha.scheduler=sigmoid"
  "loss.alpha.initial_value=1.0"
  "loss.alpha.end_value=0"
  "loss.alpha.change_init_steps=0"
  "loss.alpha.change_end_steps=$STEPS"
  "loss.alpha.clamp_value=0.005"
  "loss.alpha.gamma=25.0"
)

run_arm() {
  local desc="$1"; shift
  local run_dir="runs/$desc"
  local resume_args=()
  if compgen -G "$run_dir/snapshot-*.pt" > /dev/null; then
    log "Found existing snapshots in $run_dir -> resuming from the latest one (whole_state=true)."
    resume_args=("training.resume.on_start_ckpt_path=null" "training.resume.whole_state=true")
  else
    resume_args=("training.resume.on_start_ckpt_path=$INIT" "training.resume.whole_state=false")
  fi
  log "Launching ${desc} (batch_gpu=${BATCH_GPU}, steps=${STEPS})..."
  if ! bash scripts/train_single_gpu.sh "$PRESET" "desc=$desc" "${resume_args[@]}" "${COMMON[@]}" "$@"; then
    log "ERROR: ${desc} failed. Stopping (the remaining arms will NOT run)."
    exit 1
  fi
  log "${desc} finished."
}

IFS=',' read -r -a ARM_LIST <<< "$STAGES"
for arm in "${ARM_LIST[@]}"; do
  case "$arm" in
    a0)
      wait_for_gpu "a0"
      run_arm "${PREFIX}-a0-latent-${DESC_SUFFIX}"
      ;;
    a1)
      wait_for_gpu "a1"
      run_arm "${PREFIX}-a1-latent-${DESC_SUFFIX}" "model.parameterization=state" "training.resume.allow_missing_keys=true"
      ;;
    a2)
      wait_for_gpu "a2"
      run_arm "${PREFIX}-a2-latent-${DESC_SUFFIX}" "model.parameterization=state" "loss.target_branch=ema" "training.resume.allow_missing_keys=true"
      ;;
    *)
      log "ERROR: unknown arm '$arm' (supported: a0,a1,a2)"; exit 2 ;;
  esac
done

# --- report --------------------------------------------------------------
log "All requested arms finished. Building report..."
REPORT="runs/_${PREFIX}_report.txt"
python - "$REPORT" "$PREFIX" "$BATCH_GPU" "$STEPS" "$DESC_SUFFIX" <<'PY'
import glob
import os
import re
import sys

report_path, prefix, batch_gpu, expected_steps, suffix = (
    sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])
arms = ['a0', 'a1', 'a2']
run_dir_of = lambda arm: f'runs/{prefix}-{arm}-latent-{suffix}'
lines = [f'=== Stage 3 report: {prefix} (steps={expected_steps}, batch_gpu={batch_gpu}) ===', '']
ok = True

def read_ticks(path):
    ticks = []
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if ' step ' not in line or 'loss_total=' not in line:
                continue
            d = {}
            for k in ('step', 'iter/sec', 'samples/sec', 'gpumem', 'reserved', 'loss_total',
                      'loss_trajectory_FM', 'loss_trajectory_consistency', 'grad_norm',
                      'grad_net_num_nans', 'grad_net_num_infs', 'teacher_ema_beta',
                      'teacher_ema_updates_total', 'teacher_student_rel_dist', 'eval_ema_beta'):
                m = re.search(rf'(?:^|\s){re.escape(k)}=([^\s]+)', line)
                if m is None:
                    # Some fields are printed as "key value" (e.g. "iter/sec 0.14", "gpumem 6.10").
                    m = re.search(rf'(?:^|\s){re.escape(k)} ([^\s]+)', line)
                if m:
                    d[k] = m.group(1)
            ticks.append(d)
    return ticks

def last(ticks, key):
    for t in reversed(ticks):
        if key in t:
            return t[key]
    return 'NA'

for arm in arms:
    run_dir = run_dir_of(arm)
    log_txt = f'{run_dir}/log.txt'
    lines.append(f'--- {arm}: {run_dir} ---')
    if not os.path.isfile(log_txt):
        lines.append('  MISSING log.txt'); ok = False; lines.append(''); continue
    ticks = read_ticks(log_txt)
    if not ticks:
        lines.append('  no ticks parsed'); ok = False; lines.append(''); continue
    steps = [int(t['step']) for t in ticks]
    nans = sum(int(t.get('grad_net_num_nans', 0)) for t in ticks)
    infs = sum(int(t.get('grad_net_num_infs', 0)) for t in ticks)
    losses = [float(t['loss_total']) for t in ticks if t.get('loss_total') not in (None, 'NA')]
    grads = [float(t['grad_norm']) for t in ticks if t.get('grad_norm') not in (None, 'NA')]
    lines.append(f'  steps: {min(steps)}..{max(steps)} ({len(steps)} ticks)')
    lines.append(f'  iter/sec last={last(ticks, "iter/sec")}  samples/sec last={last(ticks, "samples/sec")}')
    lines.append(f'  gpumem max={max((t.get("gpumem") for t in ticks if t.get("gpumem")), key=float, default="NA")}  '
                 f'reserved max={max((t.get("reserved") for t in ticks if t.get("reserved")), key=float, default="NA")}')
    lines.append(f'  loss_total first={losses[0]:.6f} last={losses[-1]:.6f} min={min(losses):.6f} max={max(losses):.6f}')
    lines.append(f'  grad_norm first={grads[0]:.4f} last={grads[-1]:.4f} max={max(grads):.4f}')
    lines.append(f'  NaNs={nans} Infs={infs}')
    if int(max(steps)) < expected_steps:
        lines.append(f'  FAIL: did not reach {expected_steps} steps'); ok = False
    if nans or infs:
        lines.append('  FAIL: NaN/Inf in gradients'); ok = False
    if arm == 'a2':
        lines.append(f'  teacher_ema_beta last={last(ticks, "teacher_ema_beta")}  '
                     f'teacher_ema_updates_total last={last(ticks, "teacher_ema_updates_total")}  '
                     f'teacher_student_rel_dist last={last(ticks, "teacher_student_rel_dist")}')
        if last(ticks, 'teacher_ema_updates_total') not in ('NA', str(expected_steps)):
            lines.append(f'  WARN: teacher updates total != {expected_steps}')
    snap = sorted(glob.glob(f'{run_dir}/snapshot-*.pt'))
    if snap:
        lines.append(f'  snapshot: {snap[-1]} ({os.path.getsize(snap[-1]) / 1e9:.2f} GB)')
    else:
        lines.append('  FAIL: no snapshot saved'); ok = False
    lines.append('')

# A0 vs A1 loss closeness (soft)
dir_a0 = f'{run_dir_of("a0")}/log.txt'
dir_a1 = f'{run_dir_of("a1")}/log.txt'
ticks_a0 = read_ticks(dir_a0) if os.path.isfile(dir_a0) else []
ticks_a1 = read_ticks(dir_a1) if os.path.isfile(dir_a1) else []
if ticks_a0 and ticks_a1:
    pairs = [(float(a['loss_total']), float(b['loss_total'])) for a, b in zip(ticks_a0, ticks_a1)
             if a.get('loss_total') and b.get('loss_total')]
    if pairs:
        rel = sorted(abs(a - b) / max(abs(a), 1e-12) for a, b in pairs)
        lines.append(f'A0 vs A1 loss_total rel diff: median={rel[len(rel)//2]:.3e} max={rel[-1]:.3e}')
lines.append('')
lines.append('RESULT: ' + ('PASS' if ok else 'FAILURES, see above'))
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('\n'.join(lines))
PY

log "Report written to $REPORT. Review it, then decide about the 20k stage."
