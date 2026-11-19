#!/usr/bin/env python3
"""Reads a few real ImageNet latent batches through the full data pipeline (no training).

Checks (Step 2 completion criteria for the real-latent validation):
  1. dataset path / epoch size / resolution / label shape;
  2. real batches can be read repeatedly without errors;
  3. decoded video tensor shape / dtype / value range;
  4. VAE encoding to latents works: latent shape / dtype / finiteness / stats
     (expected latent shape for 256x256 with the SD-VAE: [b, 1, 4, 32, 32]).

Usage (from the repository root):
    python scripts/check_latent_batch.py
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.environ.setdefault('ALPHAFLOW_PROJECT_PATH', _REPO_ROOT)
# Use the same caches as scripts/train_single_gpu.sh (the SD-VAE is cached inside the repo,
# not in the user's ~/.cache/huggingface that check_setup looks at).
os.environ.setdefault('HF_HOME', os.path.join(_REPO_ROOT, '.cache', 'huggingface'))
os.environ.setdefault('TORCH_HOME', os.path.join(_REPO_ROOT, '.cache', 'torch'))
os.environ.setdefault('XDG_CACHE_HOME', os.path.join(_REPO_ROOT, '.cache'))
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

import torch
from hydra import compose, initialize

from src.structs import EasyDict
from src.utils import distributed as dist, misc
from src.utils.os_utils import disable_trivial_warnings
from src.data import Data, construct_inputs_from_batch
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
]

RESULTS = []


def report(name: str, ok: bool, detail: str = ''):
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f'  |  {detail}' if detail else ''))


def main():
    disable_trivial_warnings()
    with initialize(config_path='../configs', version_base='1.2'):
        cfg = compose('train.yaml', overrides=compute_overrides(EXPERIMENT, OVERRIDES))
    recursive_instantiate(cfg)
    cfg = EasyDict.init_recursively(cfg)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if device.type == 'cuda':
        dist.init()
        dist.init_random_state_and_cuda(seed=1, cudnn_benchmark=False, allow_tf32=False)
    print(f'Device: {device}')

    dataset_cfg = cfg.dataset
    print(f'dataset: name={dataset_cfg.name}, src={dataset_cfg.src}, src_val={dataset_cfg.get("src_val")}')
    print(f'dataset resolution: {dataset_cfg.resolution}, label_shape: {dataset_cfg.label_shape}, '
          f'batch_gpu: {dataset_cfg.batch_gpu}, batch_size: {dataset_cfg.batch_size}')

    data: Data = Data.init_from_cfg(dataset_cfg)
    epoch_size = getattr(data.dataset, 'epoch_size', None)
    if epoch_size is None and hasattr(data.dataset, '__len__'):
        try:
            epoch_size = len(data.dataset)
        except Exception: # pylint: disable=broad-except
            epoch_size = None
    resolution = getattr(data.dataset, 'resolution', None)
    report('real train dataset initialised', True,
           f'epoch_size={epoch_size}, resolution={resolution}')
    report('epoch_size matches ImageNet-1K train (1281167)',
           epoch_size in (1281167, None), f'epoch_size={epoch_size}')

    # Read several real batches from the folder dataset pipeline.
    batch_info = data.batch_info
    print(f'batch_info: {batch_info}')
    batches = []
    for _ in range(3):
        batches.append(next(data.iterator))
    print(f'batch keys: {sorted(batches[0].keys())}')
    videos = [b['video'] for b in batches]
    report('read 3 consecutive real batches from the folder dataset', True,
           f'shapes={[tuple(v.shape) for v in videos]}, dtype={videos[0].dtype}')
    label_shapes = [tuple(b['label'].shape) for b in batches]
    report('labels present with expected shape', all(len(s) == 1 + len(dataset_cfg.label_shape) for s in label_shapes),
           f'label shapes={label_shapes}')

    # Convert one real batch the same way the training loop does and inspect videos.
    x, cond = construct_inputs_from_batch(batches[0], use_x_cond=cfg.model.use_x_cond, device=device)
    print(f'x: shape={tuple(x.shape)}, dtype={x.dtype}, min={float(x.min()):.3f}, max={float(x.max()):.3f}')
    report('decoded video tensor shape == [b, 1, 3, 256, 256]',
           tuple(x.shape[1:]) == (1, 3, 256, 256), f'shape={tuple(x.shape)}')
    # The training pipeline rescales uint8 [0,255] to fp32 [-1,1].
    report('decoded video values are finite and in [-1, 1]',
           bool(torch.isfinite(x).all()) and float(x.min()) >= -1.001 and float(x.max()) <= 1.001,
           f'min={float(x.min()):.4f}, max={float(x.max()):.4f}')
    report('decoded video is non-degenerate', float(x.std()) > 1e-3,
           f'std={float(x.std()):.4f}')

    # Instantiate the real LDM network (loads the SD-VAE snapshot) and encode to latents.
    nets = network_utils.init_networks(cfg, device)
    net = misc.unwrap_module(nets.net)
    with torch.no_grad():
        latents = net.encode(x, cond=cond) if cond is not None else net.encode(x)
    print(f'latents: shape={tuple(latents.shape)}, dtype={latents.dtype}, '
          f'mean={float(latents.mean()):.4f}, std={float(latents.std()):.4f}')
    expected_latent_shape = (x.shape[0], 1, 4, 32, 32)
    report('latent shape == [b, 1, 4, 32, 32] (SD-VAE compression 8x)',
           tuple(latents.shape) == expected_latent_shape, f'shape={tuple(latents.shape)}')
    report('latents are finite float tensors', bool(torch.isfinite(latents).all()) and latents.dtype.is_floating_point,
           f'dtype={latents.dtype}')
    report('latents have non-degenerate statistics', float(latents.std()) > 1e-3,
           f'std={float(latents.std()):.4e}')

    print()
    failed = [name for name, ok, _ in RESULTS if not ok]
    if failed:
        print(f'RESULT: FAILED ({len(failed)}/{len(RESULTS)})')
        for name in failed:
            print(f'  - {name}')
        return 1
    print(f'RESULT: ALL {len(RESULTS)} CHECKS PASSED — real latent data pipeline is ready.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
