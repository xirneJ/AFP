#!/usr/bin/env python3
"""Audit: loading the released velocity checkpoint into the State (dual-head) model.

This checks the production resume path used by A1/A2 when starting from
`snapshots/alphaflow-B-2-finetune-init.pt` (a released B/2 velocity checkpoint):

  1. which keys are missing / unexpected with `allow_missing_keys=true`
     (must be exactly the two `linear_state` keys per module pair, nothing else);
  2. that the state head is re-initialised from the *loaded* velocity head with
     `W_state == -W_velocity` and `b_state == -b_velocity` (element-wise, bitwise);
  3. that the resulting full network still satisfies
     `f_state(z_t, r, t) == z_t - (t-r) * u_velocity(z_t, r, t)`
     exactly on the released weights;
  4. that the EMA copy is loaded and re-initialised the same way.

Usage (server, repository root; read-only, no training):
    python scripts/check_released_state_init.py
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.environ.setdefault('ALPHAFLOW_PROJECT_PATH', _REPO_ROOT)
os.environ.setdefault('HF_HUB_OFFLINE', '1')

import torch
from hydra import compose, initialize

from src.structs import EasyDict, ResumeConfig, TensorGroup
from src.utils import distributed as dist, misc
from src.training import network_utils
from infra.utils import recursive_instantiate
from infra.launch import compute_overrides

EXPERIMENT = 'alphaflow-sigmoid-latentspace-B-2-single-gpu'
OVERRIDES = [
    'wandb.enabled=false',
    'training.dp.strategy=null',
    'training.max_steps=1',
    'training.freqs.metrics=null',
    'training.freqs.metrics_rare=null',
    'model.parameterization=state',
]
SNAPSHOT = os.path.join(_REPO_ROOT, 'snapshots', 'alphaflow-B-2-finetune-init.pt')

RESULTS = []


def report(name: str, ok: bool, detail: str = ''):
    RESULTS.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f'  |  {detail}' if detail else ''))


def head_pairs(module):
    """Yields (name, velocity_linear, state_linear) for every dual-head FinalLayer."""
    unwrapped = misc.unwrap_module(module)
    for name, sub in unwrapped.named_modules():
        linear = getattr(sub, 'linear', None)
        linear_state = getattr(sub, 'linear_state', None)
        if isinstance(linear, torch.nn.Linear) and isinstance(linear_state, torch.nn.Linear):
            yield name or '<root>', linear, linear_state


def check_negation(tag: str, module) -> int:
    count = 0
    for name, lin, lin_s in head_pairs(module):
        w_diff = float((lin_s.weight.detach().float() + lin.weight.detach().float()).abs().max())
        b_diff = 0.0 if lin.bias is None else float((lin_s.bias.detach().float() + lin.bias.detach().float()).abs().max())
        report(f'{tag}: W_state == -W_velocity', w_diff == 0.0, f'head={name}, max|W_s+W_v|={w_diff:.3e}')
        report(f'{tag}: b_state == -b_velocity', b_diff == 0.0, f'head={name}, max|b_s+b_v|={b_diff:.3e}')
        count += 1
    return count


def main():
    if not os.path.isfile(SNAPSHOT):
        print(f'ERROR: missing {SNAPSHOT}')
        return 1
    with initialize(config_path='../configs', version_base='1.2'):
        cfg = compose('train.yaml', overrides=compute_overrides(EXPERIMENT, OVERRIDES))
    recursive_instantiate(cfg)
    cfg = EasyDict.init_recursively(cfg)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if device.type == 'cuda':
        dist.init()
        dist.init_random_state_and_cuda(seed=0, cudnn_benchmark=False, allow_tf32=False)
    torch.manual_seed(0)
    print(f'Device: {device}')
    print(f'Snapshot: {SNAPSHOT}')

    snapshot = torch.load(SNAPSHOT, map_location='cpu', weights_only=False)
    print(f'Snapshot keys: {sorted(snapshot.keys())}, cfg.parameterization='
          f'{snapshot["cfg"].get("model", {}).get("parameterization", "velocity")}')
    report('snapshot has net/ema state dicts', 'net' in snapshot and 'ema' in snapshot)

    nets = network_utils.init_networks(cfg, device)
    report('model built with dual state head',
           getattr(misc.unwrap_module(nets.net).model, 'parameterization', None) == 'state'
           and getattr(misc.unwrap_module(nets.net).model.final_layer, 'linear_state', None) is not None)

    # Reproduce the production load path: load_state_dict_ + re-init of the missing state head.
    resume_cfg = ResumeConfig(ckpt_path=SNAPSHOT, whole_state=False, allow_missing_keys=True,
                              allow_unexpected_keys=False)
    modules = {'net': nets.net, 'ema': nets.ema}
    for name, module in modules.items():
        if module is None:
            continue
        # Inspect the incompatible keys *before* the production load (same state dict).
        incompatible = misc.unwrap_module(module).load_state_dict(snapshot[name], strict=False)
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        print(f'[{name}] missing keys ({len(missing)}): {missing}')
        print(f'[{name}] unexpected keys ({len(unexpected)}): {unexpected}')
        # The only acceptable missing keys are the state-head parameters.
        missing_ok = all('linear_state' in k for k in missing) and any('linear_state' in k for k in missing)
        report(f'[{name}] only state-head keys are missing', missing_ok)
        report(f'[{name}] no unexpected keys', len(unexpected) == 0)
        report(f'[{name}] exactly 2 missing state-head keys',
               len([k for k in missing if 'linear_state' in k]) == 2,
               f'num_linear_state_missing={len([k for k in missing if "linear_state" in k])}')

    # Production path: loads the state dict and re-initialises the missing state heads from the
    # *loaded* velocity heads (this is what `load_module_state` does).
    for name, module in modules.items():
        if module is not None:
            network_utils.load_module_state(cfg, module, snapshot, name, resume_cfg)

    num_heads = 0
    for name, module in modules.items():
        if module is not None:
            num_heads += check_negation(name, module)
    report('every module pair has exactly one dual head', num_heads == 2 * sum(m is not None for m in modules.values()),
           f'checked {num_heads} heads')

    # Full-network equivalence on the released weights: f_state == z_t - (t-r) * u_velocity.
    net = misc.unwrap_module(nets.net).eval()
    b = 2
    z = torch.randn(b, 1, 4, 32, 32, device=device)
    t = torch.tensor([0.9, 0.7], device=device)
    r = torch.tensor([0.3, 0.2], device=device)
    cond = TensorGroup({'label': torch.nn.functional.one_hot(torch.arange(b, device=device), 1000).float()})
    with torch.no_grad():
        u = net(z, sigma=t, sigma_next=r, cond=cond)
        f = net(z, sigma=t, sigma_next=r, cond=cond, head='state')
    dt = (t - r).reshape(b, 1, 1, 1, 1)
    expected = z - dt * u
    max_abs = float((f - expected).abs().max())
    denom = float(expected.abs().max())
    report('released weights: f_state == z_t - (t-r) * u_velocity', max_abs < 1e-5,
           f'max|diff|={max_abs:.3e} (max|f|={denom:.3f})')
    report('released weights: state output is finite', bool(torch.isfinite(f).all()))

    print()
    failed = [name for name, ok in RESULTS if not ok]
    if failed:
        print(f'RESULT: FAILED ({len(failed)}/{len(RESULTS)})')
        for name in failed:
            print(f'  - {name}')
        return 1
    print(f'RESULT: ALL {len(RESULTS)} CHECKS PASSED — released velocity ckpt -> state head is exact.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
