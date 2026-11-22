#!/usr/bin/env bash
# Idle-GPU profiling for the 20k decision: A0/A2 x batch_gpu 16/32, 50 steps each.
#
# Decision rule (see docs/STAGE3_FORMAL_TRAINING_PLAN.md §4.5):
#   use batch_gpu=32 ONLY if, for A2, peak `reserved` < 28 GiB, no OOM/NaN/Inf, and
#   samples/sec improves by >= 10% over batch_gpu=16; otherwise use batch_gpu=16 for all arms.
#
# Usage (idle GPU recommended; the script waits for it):
#   nohup bash scripts/run_profiling.sh > runs/_profiling.log 2>&1 &
#   tail -f runs/_profiling.log
#
# Options:
#   --arms a0,a2          arms to profile (default a0,a2)
#   --batch-gpus 16,32    batch sizes (default 16,32)
#   --steps 50            profiling steps per run (default 50)
#   --min-free-mib 24000  required free VRAM (idle GPU)
#   --max-util 20         max GPU utilization to consider the card idle (100 disables)
#   --sustain-checks 3    consecutive ready polls required
#   --poll-sec 60         polling interval
#   --no-lock             skip the duplicate-instance lock
#
# Writes runs/_profiling_report.txt and prints it.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ARMS="a0,a2"
BATCH_GPUS="16,32"
STEPS=50
MIN_FREE_MIB=24000
MAX_UTIL=20
SUSTAIN_CHECKS=3
POLL_SEC=60
NO_LOCK=0
PREFIX="prof"
INIT="$REPO_ROOT/snapshots/alphaflow-B-2-finetune-init.pt"
PRESET="alphaflow-sigmoid-latentspace-B-2-single-gpu"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arms) ARMS="$2"; shift 2 ;;
    --batch-gpus) BATCH_GPUS="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --min-free-mib) MIN_FREE_MIB="$2"; shift 2 ;;
    --max-util) MAX_UTIL="$2"; shift 2 ;;
    --sustain-checks) SUSTAIN_CHECKS="$2"; shift 2 ;;
    --poll-sec) POLL_SEC="$2"; shift 2 ;;
    --no-lock) NO_LOCK=1; shift ;;
    -h|--help) sed -n '1,40p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1"; exit 2 ;;
  esac
done

mkdir -p runs
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

log() { echo "[$(date '+%F %T')] $*"; }

LOCK_DIR=/tmp/alphaflow_profiling.lockdir
acquire_lock() {
  local pidfile="${LOCK_DIR}/pid" other
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    echo $$ > "$pidfile"; trap 'rm -rf "$LOCK_DIR"' EXIT INT TERM; return 0
  fi
  other="$(cat "$pidfile" 2>/dev/null || true)"
  if [[ -n "$other" ]] && kill -0 "$other" 2>/dev/null; then
    log "Another run_profiling.sh instance (pid ${other}) is running. Exiting."; return 1
  fi
  log "Removing stale lock directory (pid ${other:-unknown} is not alive)."
  rm -rf "$LOCK_DIR"
  mkdir "$LOCK_DIR" 2>/dev/null || return 1
  echo $$ > "$pidfile"; trap 'rm -rf "$LOCK_DIR"' EXIT INT TERM; return 0
}
if (( NO_LOCK == 0 )); then acquire_lock || exit 1; fi

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
        log "[$phase] GPU is free enough. Proceeding."; return 0
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

run_one() {
  local arm="$1" bg="$2" desc
  desc="${PREFIX}-${arm}-b${bg}"
  local overrides=(
    "training.resume.on_start_ckpt_path=$INIT"
    "training.resume.whole_state=false"
    "model.ema_rampup_ratio=null"
    "model.lr_scheduler.num_warmup_steps=10"
    "training.max_steps=$STEPS"
    "training.steps_per_tick=5"
    "training.freqs.metrics=null"
    "training.freqs.metrics_rare=null"
    "training.metrics_final=[]"
    "training.freqs.loss_val=null"
    "training.freqs.loss_per_sigma=null"
    "training.freqs.snapshot=null"
    "training.freqs.snapshot_latest=null"
    "training.freqs.save_samples=null"
    "dataset.batch_gpu=$bg"
    "loss.alpha.scheduler=sigmoid"
    "loss.alpha.initial_value=1.0"
    "loss.alpha.end_value=0"
    "loss.alpha.change_init_steps=0"
    "loss.alpha.change_end_steps=$STEPS"
    "loss.alpha.clamp_value=0.005"
    "loss.alpha.gamma=25.0"
  )
  if [[ "$arm" == "a2" ]]; then
    overrides+=("model.parameterization=state" "loss.target_branch=ema" "training.resume.allow_missing_keys=true")
  fi
  wait_for_gpu "$desc"
  log "Launching ${desc} (batch_gpu=$bg, steps=$STEPS)..."
  rm -rf "runs/$desc"
  bash scripts/train_single_gpu.sh "$PRESET" "desc=$desc" "${overrides[@]}" > "runs/_${desc}.out" 2>&1
  local rc=$?
  echo "$rc" > "runs/_${desc}.exit"
  if (( rc != 0 )); then
    log "WARNING: ${desc} exited with code ${rc} (likely OOM); recorded and continuing."
  else
    log "${desc} finished."
  fi
}

IFS=',' read -r -a ARM_LIST <<< "$ARMS"
IFS=',' read -r -a BG_LIST <<< "$BATCH_GPUS"
for arm in "${ARM_LIST[@]}"; do
  for bg in "${BG_LIST[@]}"; do
    run_one "$arm" "$bg"
  done
done

# --- report --------------------------------------------------------------
log "Building profiling report..."
REPORT=runs/_profiling_report.txt
python - "$REPORT" "$PREFIX" "$ARMS" "$BATCH_GPUS" "$STEPS" <<'PY'
import os
import re
import statistics
import sys

report_path, prefix, arms_s, bgs_s, steps = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
arms = [a.strip() for a in arms_s.split(',') if a.strip()]
bgs = [int(b) for b in bgs_s.split(',') if b.strip()]
lines = [f'=== GPU profiling report ({prefix}): steps={steps} ===', '']
ok = True

def parse_ticks(path):
    out = []
    if not os.path.isfile(path):
        return out
    for line in open(path, encoding='utf-8', errors='replace'):
        if ' step ' not in line or 'loss_total=' not in line:
            continue
        d = {}
        for key in ('iter/sec', 'samples/sec', 'gpumem', 'reserved', 'loss_total', 'grad_norm',
                    'grad_net_num_nans', 'grad_net_num_infs'):
            m = re.search(rf'(?:^|\s){re.escape(key)}=([^\s]+)', line) or \
                re.search(rf'(?:^|\s){re.escape(key)} ([^\s]+)', line)
            if m:
                d[key] = m.group(1)
        m = re.search(r'\bstep (\d+)', line)
        if m:
            d['step'] = int(m.group(1))
        out.append(d)
    return out

def med_last(ticks, key):
    vals = []
    for t in ticks:
        try:
            vals.append(float(t[key]))
        except (KeyError, ValueError):
            pass
    if not vals:
        return None
    tail = vals[-min(5, len(vals)):]
    return statistics.median(tail)

results = {}
for arm in arms:
    for bg in bgs:
        desc = f'{prefix}-{arm}-b{bg}'
        run_dir = f'runs/{desc}'
        ticks = parse_ticks(f'{run_dir}/log.txt')
        exit_code = None
        if os.path.isfile(f'runs/_{desc}.exit'):
            try:
                exit_code = int(open(f'runs/_{desc}.exit').read().strip())
            except ValueError:
                exit_code = None
        if not ticks:
            lines.append(f'--- {desc} ---  FAIL: no log ticks (exit={exit_code})'); ok = False
            results[(arm, bg)] = None
            continue
        steps_done = max(t['step'] for t in ticks)
        nans = sum(int(float(t.get('grad_net_num_nans', 0))) for t in ticks)
        infs = sum(int(float(t.get('grad_net_num_infs', 0))) for t in ticks)
        gpu_max = max((float(t['gpumem']) for t in ticks if 'gpumem' in t), default=None)
        res_max = max((float(t['reserved']) for t in ticks if 'reserved' in t), default=None)
        it = med_last(ticks, 'iter/sec')
        sps = med_last(ticks, 'samples/sec')
        results[(arm, bg)] = dict(steps=steps_done, nans=nans, infs=infs, gpu=gpu_max, reserved=res_max,
                                  iter=it, samples=sps, exit=exit_code)
        lines.append(f'--- {desc} (exit={exit_code}) ---')
        lines.append(f'  steps={steps_done}/{steps}  iter/sec={it:.3f}  samples/sec={sps:.1f}')
        lines.append(f'  gpumem max={gpu_max:.2f} GB  reserved max={res_max:.2f} GB  NaNs={nans} Infs={infs}')
        if steps_done < steps or (exit_code not in (0, None)) or nans or infs:
            lines.append('  FAIL: run did not cleanly complete')
            ok = False
        lines.append('')

lines.append('=== b32 decision (per arm, compared with b16) ===')
if 16 in bgs and 32 in bgs:
    for arm in arms:
        b16 = results.get((arm, 16))
        b32 = results.get((arm, 32))
        if not b16 or not b32 or not b16.get('samples') or not b32.get('samples'):
            lines.append(f'{arm}: insufficient data -> default to batch_gpu=16')
            continue
        gain = b32['samples'] / b16['samples'] - 1.0
        reserved_ok = b32['reserved'] is not None and b32['reserved'] < 28.0
        sps_ok = gain >= 0.10
        clean_ok = (b32['exit'] in (0, None) and b32['nans'] == 0 and b32['infs'] == 0
                    and b32['steps'] >= steps)
        verdict = 'b32 OK' if (reserved_ok and sps_ok and clean_ok) else 'b16 recommended'
        lines.append(f'{arm}: samples/sec b16={b16["samples"]:.1f} b32={b32["samples"]:.1f} '
                     f'-> gain={gain * 100:+.1f}%; b32 reserved={b32["reserved"] if b32["reserved"] else "NA"} GB; '
                     f'reserved<28GiB={reserved_ok} gain>=10%={sps_ok} clean={clean_ok} => {verdict}')
else:
    lines.append('need both batch_gpus 16 and 32 for the comparison')

lines.append('')
lines.append('RESULT: ' + ('profiling complete (see decision lines)' if ok else 'profiling had FAILURES'))
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('\n'.join(lines))
PY

log "Report written to $REPORT. Send it for the batch-size decision; do NOT start 20k before that."
