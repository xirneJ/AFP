#!/usr/bin/env python3
"""Stage-1 equivalence checks for the "state" output parameterization.

The state model predicts the endpoint state directly:
    f_theta(z_t, r, t) = z_t - (t-r) * u_theta(z_t, r, t)
while the official model predicts the interval-average velocity u. With the second
(state) head initialized as the negated velocity head, the two models are the same
function. This script verifies (on a small pixel-space SiT, no checkpoints/data needed):

  1. Prediction conversion:  f(z_t,r,t) == z_t - (t-r) * u(z_t,r,t)
  2. Un-scaled loss:         ||f - f_tgt||^2 / (t-r)^2 == ||u - u_tgt||^2
  3. Adaptive weighting:     same weights => same final loss
  4. Gradients:              trunk gradients identical; state-head gradient == -velocity-head grad
  5. Sampling:               state multi-step sampling == velocity Euler sampling (same head)
                             [checked in a follow-up once a trained model exists]

Run (from the repository root, GPU recommended but CPU is supported):
    python scripts/check_state_equivalence.py
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.environ.setdefault('ALPHAFLOW_PROJECT_PATH', _REPO_ROOT)

import torch
from hydra import compose, initialize
from omegaconf import OmegaConf

from src.structs import EasyDict
from src.utils import distributed as dist
from src.utils.os_utils import construct_class_by_name, disable_trivial_warnings
from src.training import network_utils
from infra.utils import recursive_instantiate
from infra.launch import compute_overrides

# A tiny pixel-space SiT with the alphaflow loss: no VAE, no dataset, no checkpoint needed.
# We start from the registry smoke-test preset (pixel-space, 64x64, dummy dataset) and shrink it.
SMOKE_EXPERIMENT = 'alphaflow-smoke-test-single-gpu'
EXTRA_OVERRIDES = [
    'model.parameterization=state',
    'model.num_blocks=2',
    'model.dim=128',
    'model.num_heads=2',
    'model.act_dtype=float32',
    'model.autocast=false',
    'model.use_ema=false',
    'model.self_cond_probability=0.0',
    'training.dp.strategy=null',
    'wandb.enabled=false',
]

RESULTS = []


def report(name: str, ok: bool, detail: str = ''):
    RESULTS.append((name, ok, detail))
    status = 'PASS' if ok else 'FAIL'
    print(f'[{status}] {name}' + (f'  |  {detail}' if detail else ''))


def max_rel_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = b.abs().mean().clamp_min(1e-12)
    return float((a - b).abs().max() / denom)


def build_net_and_loss(cfg: EasyDict, device: torch.device):
    network_kwargs, loss_kwargs = network_utils.instantiate_network_kwargs(cfg)
    interface_kwargs = dict(resolution=cfg.dataset.resolution, img_channels=3, label_shape=cfg.dataset.label_shape)
    net = construct_class_by_name(**network_kwargs, **interface_kwargs).to(device)
    net.train()
    loss = construct_class_by_name(**loss_kwargs).to(device)
    return net, loss


def make_batch(net, batch_size: int, t_values: list[float], interval_ratio: float, device: torch.device):
    """Builds a deterministic (x, z_t, t, r, v_true) batch on the linear path."""
    spatial = net.input_shape # (t, c, h, w)
    x = torch.randn(batch_size, *spatial, device=device)
    noise = torch.randn(batch_size, *spatial, device=device)
    t = torch.tensor(t_values, device=device, dtype=torch.float32).view(batch_size, 1, 1, 1, 1)
    r = t * interval_ratio
    z_t = (1 - t) * x + t * noise
    velocity = noise - x # true velocity v = eps - x
    return x, z_t, t, r, velocity


def check_prediction_conversion(net, device):
    b = 4
    x, z_t, t, r, _ = make_batch(net, b, [0.95, 0.8, 0.6, 0.4], 0.5, device)
    cond = net.construct_dummy_cond(b, device)
    with torch.no_grad():
        u = net(z_t, sigma=t, sigma_next=r, cond=cond, head='velocity')
        f = net(z_t, sigma=t, sigma_next=r, cond=cond, head='state')
        expected_f = z_t - (t - r) * u
    diff = max_rel_diff(f, expected_f)
    report('prediction conversion f == z_t - (t-r) u', diff < 1e-5, f'max rel diff = {diff:.3e}')


def check_loss_and_gradients(net, loss, device):
    b = 6
    t_values = [0.95, 0.85, 0.75, 0.6, 0.5, 0.4]
    alpha = 0.5
    x, z_t, t, r, velocity = make_batch(net, b, t_values, 0.4, device)
    cond = net.construct_dummy_cond(b, device)
    dt = alpha * (t - r)

    with torch.no_grad():
        velocity_cfg = loss._compute_velocity_cfg(velocity=velocity, x_t=z_t, t=t, cond=cond, augment_labels=None, net=net, batch_size=b)
        mean_velocity = loss._compute_mean_velocity_d(z_t, velocity_cfg, r, t, dt, cond, None, net) # u_target

    # --- velocity model forward/backward -------------------------------------
    net.zero_grad(set_to_none=True)
    pred_v = net(z_t, sigma_next=r, sigma=t, cond=cond)
    L_v = ((pred_v - mean_velocity) ** 2).flatten(1).mean(1) # [b]
    weight_v = alpha / (L_v.detach() + loss.cfg.adaptive_loss_weight_eps)
    loss_v = (weight_v * L_v).mean()
    loss_v.backward()
    grads_v = {n: p.grad.detach().clone() for n, p in net.named_parameters() if p.grad is not None}

    # --- state model forward/backward ----------------------------------------
    net.zero_grad(set_to_none=True)
    pred_s = net(z_t, sigma_next=r, sigma=t, cond=cond, head='state') # f_theta(z_t, r, t)
    interval = t - r
    f_target = z_t - interval * mean_velocity # z_t - (t-r) u_target
    L_s = ((pred_s - f_target) ** 2 / interval ** 2).flatten(1).mean(1) # [b]
    weight_s = alpha / (L_s.detach() + loss.cfg.adaptive_loss_weight_eps)
    loss_s = (weight_s * L_s).mean()
    loss_s.backward()
    grads_s = {n: p.grad.detach().clone() for n, p in net.named_parameters() if p.grad is not None}

    # 2. un-scaled losses
    diff = max_rel_diff(L_s, L_v)
    report('un-scaled loss  ||f-f_tgt||^2/(t-r)^2 == ||u-u_tgt||^2', diff < 1e-5, f'max rel diff = {diff:.3e}')

    # 3. adaptive weights / final losses
    weight_diff = float((weight_s - weight_v).abs().max() / weight_v.abs().mean().clamp_min(1e-12))
    loss_diff = float((loss_s - loss_v).abs() / loss_v.abs().clamp_min(1e-12))
    report('adaptive-weighted loss identical', weight_diff < 1e-5 and loss_diff < 1e-5,
           f'weight diff = {weight_diff:.3e}, loss diff = {loss_diff:.3e}')

    # 4a. trunk gradients (same shared parameters)
    trunk_names = [n for n in grads_v if 'final_layer.linear' not in n]
    trunk_ok = all(torch.allclose(grads_v[n], grads_s[n], rtol=1e-4, atol=1e-6) for n in trunk_names)
    worst = max(float((grads_v[n] - grads_s[n]).abs().max()) for n in trunk_names)
    report(f'shared-trunk gradients identical ({len(trunk_names)} tensors)', trunk_ok, f'max abs diff = {worst:.3e}')

    # 4b. state head gradient should be the negated velocity head gradient
    gv = grads_v.get('model.final_layer.linear.weight')
    gs = grads_s.get('model.final_layer.linear_state.weight')
    head_ok = gv is not None and gs is not None and torch.allclose(gs, -gv, rtol=1e-4, atol=1e-6)
    head_diff = float((gs + gv).abs().max()) if (gs is not None and gv is not None) else float('nan')
    report('state-head grad == -velocity-head grad', head_ok, f'max |g_state + g_velocity| = {head_diff:.3e}')


def main():
    disable_trivial_warnings()
    with initialize(config_path='../configs', version_base='1.2'):
        overrides = compute_overrides(SMOKE_EXPERIMENT, EXTRA_OVERRIDES)
        cfg = compose('train.yaml', overrides=overrides)
    recursive_instantiate(cfg)
    cfg = EasyDict.init_recursively(cfg)
    assert str(cfg.model.parameterization) == 'state', cfg.model.parameterization
    assert float(cfg.model.sampling.sigma_noise) == 1.0, \
        f'The equivalence checks assume the linear path z_t=(1-t)x+t*eps (sigma_noise=1), got {cfg.model.sampling.sigma_noise}.'

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        dist.init()
        dist.init_random_state_and_cuda(seed=1, cudnn_benchmark=False, allow_tf32=False)

    torch.manual_seed(0)
    net, loss = build_net_and_loss(cfg, device)
    # Perturb all parameters so that the checks are not run on the (partially zero-initialised)
    # vanilla init: this mimics a loaded checkpoint where the velocity head is meaningful.
    with torch.no_grad():
        for p in net.parameters():
            p.normal_(mean=0.0, std=0.02)
    num_state_heads = network_utils.init_state_head_from_velocity_(net)
    assert num_state_heads == 1, f'Expected exactly one dual head, got {num_state_heads}'
    net.eval() # no dropout => deterministic checks
    print(f'State heads initialized from velocity head: {num_state_heads}')
    print(f'Network parameters: {sum(p.numel() for p in net.parameters()):,}')
    print()

    check_prediction_conversion(net, device)
    check_loss_and_gradients(net, loss, device)

    print()
    failed = [name for name, ok, _ in RESULTS if not ok]
    if failed:
        print(f'RESULT: FAILED ({len(failed)}/{len(RESULTS)} checks)')
        for name in failed:
            print(f'  - {name}')
        return 1
    print(f'RESULT: ALL {len(RESULTS)} CHECKS PASSED')
    print('Stage-1 (strictly equivalent state parameterization) is implemented correctly.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
