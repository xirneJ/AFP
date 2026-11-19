#!/usr/bin/env python3
"""Summarizes the stage-1 short-training (1-2k steps) sanity runs.

This is the *second* part of the stage-1 verification report:
  Part I  (hard): mathematical / numerical / code equivalence -> `scripts/check_state_equivalence.py`
  Part II (soft): 1-2k step training sanity (loss, gradient norm, stability) -> this script

The two runs are NOT expected to match pointwise; the parameterizations differ (and `state`
has an extra output head), so only stability/consistency indicators are compared.

Usage (from the repository root):
    python scripts/summarize_stage1_sanity.py \
        --velocity-dir runs/stage1-smoke-velocity \
        --state-dir runs/stage1-smoke-state \
        --expected-steps 1000

The tick log lines (written to <run>/log.txt every `training.steps_per_tick` steps) contain
`step N ... loss 0.123 loss_total=... loss_trajectory_FM=... grad_norm=...` fields.
"""
import argparse
import json
import math
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STEP_RE = re.compile(r'\bstep\s+(\d+)\b')
ITER_RE = re.compile(r'iter/sec\s+([\d.]+)')
KV_RE = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*)=([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|nan|inf|-inf)\b')
FINISH_RE = re.compile(r'Training is done|All done|Exiting\.\.\.')

INTERESTING_KEYS = [
    'loss_total', 'loss_trajectory_FM', 'loss_trajectory_consistency', 'loss_trajectory_sum',
    'grad_norm',
]


def parse_run(run_dir: str):
    log_path = os.path.join(run_dir, 'log.txt')
    if not os.path.isfile(log_path):
        raise FileNotFoundError(f'No log.txt in {run_dir}')
    records = []
    finished = False
    with open(log_path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if FINISH_RE.search(line):
                finished = True
            step_match = STEP_RE.search(line)
            if step_match is None:
                continue
            kvs = {k: float(v) for k, v in KV_RE.findall(line)}
            if 'loss_total' not in kvs:
                continue
            iter_match = ITER_RE.search(line)
            records.append({
                'step': int(step_match.group(1)),
                'iter_per_sec': float(iter_match.group(1)) if iter_match else None,
                **{k: kvs.get(k) for k in INTERESTING_KEYS},
            })
    records.sort(key=lambda r: r['step'])
    return records, finished, log_path


def series(records, key):
    return [r['step'] for r in records if r[key] is not None], [r[key] for r in records if r[key] is not None]


def finite(values):
    return all(v is not None and math.isfinite(v) for v in values)


def rel(a, b):
    if a is None or b is None:
        return float('nan')
    denom = abs(b) + 1e-12
    return abs(a - b) / denom


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--velocity-dir', required=True, help='Run dir of the velocity baseline (e.g. runs/stage1-smoke-velocity).')
    parser.add_argument('--state-dir', required=True, help='Run dir of the state model (e.g. runs/stage1-smoke-state).')
    parser.add_argument('--expected-steps', type=int, default=1000)
    parser.add_argument('--initial-window', type=int, default=10, help='Number of first common ticks used for the initial-loss comparison.')
    args = parser.parse_args()

    v_records, v_finished, v_log = parse_run(args.velocity_dir)
    s_records, s_finished, s_log = parse_run(args.state_dir)
    if not v_records or not s_records:
        print('ERROR: could not parse any tick record with `loss_total=...` from the logs.')
        print('Make sure the runs were started with the updated logging (stage-1 branch) and with')
        print('`training.steps_per_tick=10` so that the curves are dense enough.')
        return 1

    print('=== Part II: 1-2k step sanity report ===')
    print(f'velocity log: {v_log}')
    print(f'state    log: {s_log}')
    print()

    hard_failures = []

    for name, records, finished in (('velocity', v_records, v_finished), ('state', s_records, s_finished)):
        steps = [r['step'] for r in records]
        print(f'[{name}] ticks: {len(records)}, step range: {min(steps)}..{max(steps)}')
        loss_v = [r['loss_total'] for r in records]
        grad_v = [r['grad_norm'] for r in records]
        iters = [r['iter_per_sec'] for r in records if r['iter_per_sec'] is not None]
        print(f'[{name}] loss_total: first={loss_v[0]:.6g}, last={loss_v[-1]:.6g}, min={min(loss_v):.6g}, max={max(loss_v):.6g}')
        if grad_v and all(g is not None for g in grad_v):
            print(f'[{name}] grad_norm: first={grad_v[0]:.4g}, last={grad_v[-1]:.4g}, max={max(grad_v):.4g}')
        if iters:
            print(f'[{name}] iter/sec: mean={sum(iters) / len(iters):.2f}')
        ok_steps = max(steps) >= args.expected_steps
        ok_finite = finite(loss_v) and finite([g for g in grad_v if g is not None])
        ok_finished = finished or max(steps) >= args.expected_steps
        print(f'[{name}] {"PASS" if ok_steps else "FAIL"} completed >= {args.expected_steps} steps')
        print(f'[{name}] {"PASS" if ok_finite else "FAIL"} loss / grad norm are finite (no NaN/Inf)')
        print(f'[{name}] {"PASS" if ok_finished else "FAIL"} training finished / reached target')
        if not (ok_steps and ok_finite and ok_finished):
            hard_failures.append(name)
        print()

    # Align by step for the soft comparison.
    v_by_step = {r['step']: r for r in v_records}
    s_by_step = {r['step']: r for r in s_records}
    common = sorted(set(v_by_step) & set(s_by_step))
    print(f'common ticks for comparison: {len(common)}')

    if common:
        diffs = [rel(s_by_step[s]['loss_total'], v_by_step[s]['loss_total']) for s in common]
        initial = diffs[:args.initial_window]
        print('--- soft comparison (NOT required to be pointwise identical) ---')
        print(f'initial-loss rel diff (first {len(initial)} ticks): median={sorted(initial)[len(initial) // 2]:.4g}, max={max(initial):.4g}')
        print(f'whole-run loss rel diff: median={sorted(diffs)[len(diffs) // 2]:.4g}, max={max(diffs):.4g}')
        print(f'loss_total at first common step: velocity={v_by_step[common[0]]["loss_total"]:.6g}, state={s_by_step[common[0]]["loss_total"]:.6g}')
        print(f'loss_total at last  common step: velocity={v_by_step[common[-1]]["loss_total"]:.6g}, state={s_by_step[common[-1]]["loss_total"]:.6g}')
        v_grad = [v_by_step[s]['grad_norm'] for s in common if v_by_step[s]['grad_norm'] is not None]
        s_grad = [s_by_step[s]['grad_norm'] for s in common if s_by_step[s]['grad_norm'] is not None]
        if v_grad and s_grad:
            print(f'grad_norm over common ticks: velocity max={max(v_grad):.4g} / state max={max(s_grad):.4g}')
        print()
        print('Interpretation: the two parameterizations are mathematically equivalent at '
              'initialisation, but they have different output heads and receive gradients from '
              'different branches, so the curves are only expected to be *close*, not identical. '
              'The hard criteria are: finite losses/gradients, no divergence, and both runs '
              'reaching the target number of steps.')

    summary = {
        'velocity_dir': os.path.abspath(args.velocity_dir),
        'state_dir': os.path.abspath(args.state_dir),
        'expected_steps': args.expected_steps,
        'common_ticks': len(common),
        'hard_failures': hard_failures,
    }
    summary_path = os.path.join(args.state_dir, 'sanity-summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSaved summary JSON to {summary_path}')

    if hard_failures:
        print(f'RESULT: FAILED (runs with hard-check failures: {", ".join(hard_failures)})')
        return 1
    print('RESULT: Part II sanity checks passed (equivalence itself is verified by Part I).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
