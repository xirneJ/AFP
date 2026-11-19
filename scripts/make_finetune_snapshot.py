#!/usr/bin/env python3
"""Converts a released AlphaFlow checkpoint into a training snapshot.

The released checkpoints (e.g. ``snapshots/alphaflow-B-2.pt``) only contain
``{'ema', 'cfg'}``, while the training code expects a snapshot with ``{'net', 'ema', 'cfg'}``
when resuming via ``training.resume.on_start_ckpt_path``. This script maps the released
EMA weights to both ``net`` and ``ema`` so that fine-tuning can start from them.

Usage:
    python scripts/make_finetune_snapshot.py snapshots/alphaflow-B-2.pt snapshots/alphaflow-B-2-finetune-init.pt

Then fine-tune with:
    training.resume.on_start_ckpt_path=<abs path to the converted snapshot>
    training.resume.whole_state=false
    training.resume.allow_missing_keys=true   # only needed for `model.parameterization=state`
"""
import argparse
import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main():
    parser = argparse.ArgumentParser(description='Convert a released AlphaFlow checkpoint into a training snapshot.')
    parser.add_argument('src', help='Path to the released checkpoint (keys: ema, cfg).')
    parser.add_argument('dst', help='Output path of the training snapshot (keys: net, ema, cfg).')
    args = parser.parse_args()

    print(f'Loading {args.src} ...')
    snapshot = torch.load(args.src, map_location='cpu', weights_only=False)
    assert 'cfg' in snapshot, f'Expected a `cfg` entry in {args.src}, got keys: {list(snapshot.keys())}'

    if 'net' in snapshot and snapshot['net'] is not None:
        net_state = snapshot['net']
        print('Using the existing `net` state dict.')
    elif 'ema' in snapshot and snapshot['ema'] is not None:
        net_state = snapshot['ema']
        print('No `net` state dict found; using the released EMA weights for both `net` and `ema`.')
    else:
        raise KeyError(f'Neither `net` nor `ema` found in {args.src}: {list(snapshot.keys())}')

    ema_state = snapshot.get('ema', net_state)
    out = {
        'net': net_state,
        'ema': ema_state,
        'cfg': snapshot['cfg'],
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    torch.save(out, args.dst)
    size_mb = os.path.getsize(args.dst) / 1e6
    print(f'Saved training snapshot to {args.dst} ({size_mb:.0f} MB).')
    print('Fine-tune overrides:')
    print(f'  training.resume.on_start_ckpt_path={os.path.abspath(args.dst)}')
    print('  training.resume.whole_state=false')
    print('  training.resume.allow_missing_keys=true   # only for model.parameterization=state')


if __name__ == '__main__':
    main()
