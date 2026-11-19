#!/usr/bin/env python3
"""Stage-1 verification suite: state parameterization vs (corrected) velocity baseline.

This script implements the requested audit/equivalence tests on a small pixel-space SiT
(no dataset, no checkpoint, ~1 minute on GPU):

  A. Batch-alignment audit with three variants:
       - Original Official  : the upstream `cat([c-targets, d-targets])` pairing,
       - Corrected Velocity : per-sample target assembly (production velocity path),
       - Corrected State    : per-sample target assembly + state head (production state path).
     Tests realizable mask patterns (alpha scalar => mask_c is a prefix or the whole batch) and
     adversarial non-uniform patterns (interleaved mask_c/mask_d) to verify per-sample pairing.

  B. Branch-by-branch math/code equivalence (Velocity vs State):
       - 0 < alpha <= 1 d-branch, - alpha == 0 JVP local-expansion branch, - r == t FM branch,
       - CFG-guided velocity construction, - clamp_utgt clipping position/state conversion,
       - adaptive weighting: un-scaled loss, weights, detached denominator, final loss.

  C. Numerical & gradient tests:
       - full-network output mapping f_state == z_t-(t-r)u (precond/skip/adaLN level, C0),
         including a counter-test showing the relation breaks when the heads are not negations,
       - per-sample prediction conversion, output-gradient relation, parameter gradients,
       - detached adaptive-weight denominator, NaN/Inf/division-by-zero checks.

  D. Sampling equivalence (no extra NFE):
       - state sampler vs velocity sampler for NFE-1/NFE-2, with and without CFG,
         comparing EVERY intermediate step (x_next and x_denoised), not only the final state,
       - state sampler output vs analytic z_t-(t-r)u.

  E. Realistic branch-combination matrix:
       - alpha=0 + CFG, alpha=1 target-branch skip, JVP branch without clipping,
       - ratio_fm=0 / ratio_fm=1 boundaries (with and without CFG).

Run from the repository root (GPU recommended):
    python scripts/check_state_equivalence.py
"""
import copy
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.environ.setdefault('ALPHAFLOW_PROJECT_PATH', _REPO_ROOT)

import torch
from hydra import compose, initialize

from src.structs import EasyDict, TensorGroup
from src.utils import distributed as dist, misc
from src.utils.os_utils import construct_class_by_name, disable_trivial_warnings
from src.training import network_utils
from src.inference import reverse_recflow_step
from infra.utils import recursive_instantiate
from infra.launch import compute_overrides

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
TOL_FP32 = 1e-5
TOL_DIFF = 1e-3 # tolerance for "these two quantities are equal up to fp noise"


def report(name: str, ok: bool, detail: str = ''):
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f'  |  {detail}' if detail else ''))


def rel_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.detach(), b.detach()
    denom = b.abs().mean().clamp_min(1e-12)
    return float((a - b).abs().max() / denom)


def assert_finite(name: str, *tensors) -> bool:
    ok = all(bool(torch.isfinite(t).all()) for t in tensors if t is not None)
    report(f'finite check: {name}', ok)
    return ok


# ---------------------------------------------------------------------------
# Setup: one velocity net and one state net that represent the same function.
# ---------------------------------------------------------------------------

def build_setup(device):
    with initialize(config_path='../configs', version_base='1.2'):
        overrides = compute_overrides(SMOKE_EXPERIMENT, EXTRA_OVERRIDES)
        cfg = compose('train.yaml', overrides=overrides)
    recursive_instantiate(cfg)
    cfg = EasyDict.init_recursively(cfg)
    assert str(cfg.model.parameterization) == 'state', cfg.model.parameterization
    assert float(cfg.model.sampling.sigma_noise) == 1.0, 'Equivalence tests assume sigma_noise == 1.'

    net_v = build_net(cfg, device, 'velocity')
    torch.manual_seed(0)
    with torch.no_grad():
        for p in net_v.parameters():
            p.normal_(mean=0.0, std=0.02) # meaningful, non-zero weights (mimics a loaded checkpoint)

    net_s = build_net(cfg, device, 'state')
    state_sd = net_s.state_dict()
    for k, v in net_v.state_dict().items():
        assert k in state_sd, f'Missing key in state model: {k}'
        state_sd[k] = v.clone()
    net_s.load_state_dict(state_sd)
    num_heads = network_utils.init_state_head_from_velocity_(net_s)
    assert num_heads == 1, f'Expected exactly one dual head, got {num_heads}'

    loss_v = build_loss(cfg, device, 'velocity')
    loss_s = build_loss(cfg, device, 'state')
    return cfg, net_v, net_s, loss_v, loss_s


def build_net(cfg, device, parameterization):
    model_cfg = copy.deepcopy(cfg.model)
    model_cfg.parameterization = parameterization
    net = construct_class_by_name(
        class_name=model_cfg.target_class, cfg=model_cfg,
        resolution=cfg.dataset.resolution, img_channels=3, label_shape=cfg.dataset.label_shape,
    ).to(device)
    net.eval()
    return net


def build_loss(cfg, device, parameterization, **loss_overrides):
    loss_cfg = copy.deepcopy(cfg.loss)
    loss_cfg.model = copy.deepcopy(cfg.model)
    loss_cfg.model.parameterization = parameterization
    for key, value in loss_overrides.items():
        setattr(loss_cfg, key, value)
    return construct_class_by_name(class_name=cfg.loss.target, cfg=loss_cfg).to(device)


# ---------------------------------------------------------------------------
# Deterministic batches with controllable mask patterns.
# ---------------------------------------------------------------------------

def make_batch(net, device, kind: str, seed: int = 0):
    """Returns tensors t, t_next, dt (all [b,1,1,1,1]) and the scalar alpha."""
    torch.manual_seed(seed)
    b = 6
    x = torch.randn(b, *net.input_shape, device=device)
    device_ = device
    t = torch.empty(b, 1, 1, 1, 1, device=device_)
    t_next = torch.empty(b, 1, 1, 1, 1, device=device_)
    dt = torch.zeros(b, 1, 1, 1, 1, device=device_)
    alpha = 0.5

    def set_pair(i, t_val, ratio=0.4):
        t[i].fill_(t_val)
        t_next[i].fill_(t_val * ratio)

    if kind == 'fm':
        alpha = 0.0
        for i, tv in enumerate([0.95, 0.85, 0.75, 0.65, 0.55, 0.45]):
            t[i].fill_(tv)
            t_next[i].fill_(tv)
    elif kind == 'd':
        alpha = 0.5
        for i, tv in enumerate([0.95, 0.85, 0.75]):
            t[i].fill_(tv)
            t_next[i].fill_(tv)
        for i, tv in enumerate([0.80, 0.60, 0.40]):
            set_pair(3 + i, tv, ratio=0.4)
            dt[3 + i] = alpha * (t[3 + i] - t_next[3 + i])
    elif kind == 'alpha0':
        alpha = 0.0
        for i, tv in enumerate([0.95, 0.85, 0.75]):
            t[i].fill_(tv)
            t_next[i].fill_(tv)
        for i, tv in enumerate([0.80, 0.60, 0.40]):
            set_pair(3 + i, tv, ratio=0.4) # dt == 0 => JVP branch (t > r)
    elif kind == 'd_all':
        # All samples on the consistency branch (no FM samples): used for the parameter-gradient
        # comparison so that both models receive gradients through exactly one head.
        alpha = 0.5
        for i, tv in enumerate([0.95, 0.85, 0.75, 0.65, 0.55, 0.45]):
            set_pair(i, tv, ratio=0.4)
            dt[i] = alpha * (t[i] - t_next[i])
    elif kind == 'alpha1':
        # All samples with alpha == 1: dt == t-r exactly (target branch is skipped in the code).
        alpha = 1.0
        for i, tv in enumerate([0.95, 0.85, 0.75, 0.65, 0.55, 0.45]):
            set_pair(i, tv, ratio=0.4)
            dt[i] = 1.0 * (t[i] - t_next[i])
    elif kind == 'alpha0_all':
        # All samples on the JVP branch (t > r, dt == 0, alpha == 0): corresponds to ratio_fm=0.
        alpha = 0.0
        for i, tv in enumerate([0.95, 0.85, 0.75, 0.65, 0.55, 0.45]):
            set_pair(i, tv, ratio=0.4)
    elif kind == 'mixed':
        # Synthetic mix used only to test clipping semantics: 2 FM (c), 2 JVP (c), 2 d (clipped).
        alpha = 0.5
        for i, tv in enumerate([0.95, 0.85]):
            t[i].fill_(tv)
            t_next[i].fill_(tv)
        for i, tv in enumerate([0.80, 0.70]):
            set_pair(2 + i, tv, ratio=0.4) # dt == 0, t > r -> JVP (c-branch, not clipped)
        for i, tv in enumerate([0.60, 0.45]):
            set_pair(4 + i, tv, ratio=0.4)
            dt[4 + i] = alpha * (t[4 + i] - t_next[4 + i])
    elif kind == 'interleaved':
        # Adversarial pattern: mask_c/mask_d interleaved; not producible by the official sampler
        # (alpha is a scalar per step), but it is exactly the pattern that exposes target mispairing.
        alpha = 0.5
        for i in range(6):
            tv = 0.95 - 0.1 * i
            if i % 2 == 0:
                t[i].fill_(tv)
                t_next[i].fill_(tv)
            else:
                set_pair(i, tv, ratio=0.4)
                dt[i] = alpha * (t[i] - t_next[i])
    elif kind == 'prefix_suffix':
        # Adversarial: mask_c = prefix + suffix (FM samples up front, JVP samples at the end).
        alpha = 0.5
        for i, tv in enumerate([0.95, 0.85]):
            t[i].fill_(tv)
            t_next[i].fill_(tv)
        for i, tv in enumerate([0.75, 0.65]):
            set_pair(2 + i, tv, ratio=0.4)
            dt[2 + i] = 0.2 * (t[2 + i] - t_next[2 + i])
        for i, tv in enumerate([0.55, 0.45]):
            set_pair(4 + i, tv, ratio=0.4)
    else:
        raise ValueError(kind)

    return EasyDict(x=x, t=t, t_next=t_next, dt=dt, alpha=alpha)


def reference_parts(loss, net, batch, cond, seed: int = 1234):
    """Computes per-sample targets with an ordering-independent reference implementation."""
    torch.manual_seed(seed) # The production forward draws the noise as its first random op.
    x, t, t_next, dt = batch.x, batch.t, batch.t_next, batch.dt
    alpha = batch.alpha
    b = x.shape[0]
    x_aug = loss._maybe_encode(net, x, cond)
    noise = torch.randn_like(x_aug)
    x_t = loss.apply_noise(x_aug, noise * t, t, net)
    velocity = noise - x_aug
    velocity_cfg = loss._compute_velocity_cfg(
        velocity=velocity, x_t=x_t, t=t, cond=cond, augment_labels=None, net=net, batch_size=b,
    )

    mask_c = (dt == 0).flatten()
    mask_d = ~mask_c
    # Corrected (per-sample) targets, assembled in the original batch order.
    targets_corrected = torch.empty_like(x_t)
    if mask_c.any():
        targets_corrected[mask_c] = loss._compute_mean_velocity_c(
            x_t[mask_c], t_next[mask_c], t[mask_c], velocity_cfg[mask_c], cond[mask_c], None, net,
        )
    if mask_d.any():
        targets_corrected[mask_d] = loss._compute_mean_velocity_d(
            x_t[mask_d], velocity_cfg[mask_d], t_next[mask_d], t[mask_d], dt[mask_d], cond[mask_d], None, net,
        )
    # Legacy official pairing: cat([c-targets, d-targets]) (only valid when mask_c is a prefix).
    targets_c = loss._compute_mean_velocity_c(
        x_t[mask_c], t_next[mask_c], t[mask_c], velocity_cfg[mask_c], cond[mask_c], None, net,
    ) if mask_c.any() else torch.empty(0, *x_t.shape[1:], device=x_t.device)
    targets_d = loss._compute_mean_velocity_d(
        x_t[mask_d], velocity_cfg[mask_d], t_next[mask_d], t[mask_d], dt[mask_d], cond[mask_d], None, net,
    ) if mask_d.any() else torch.empty(0, *x_t.shape[1:], device=x_t.device)
    targets_legacy = torch.cat([targets_c, targets_d], dim=0)
    weights_legacy = torch.cat(
        [torch.ones(len(targets_c), device=x_t.device), torch.full((len(targets_d),), alpha, device=x_t.device)],
        dim=0,
    )
    # Fully ordering-independent reference: one sample at a time (recomputes the helper output).
    ref_targets = []
    for i in range(b):
        sl = slice(i, i + 1)
        if bool(mask_c[i]):
            ref_targets.append(loss._compute_mean_velocity_c(
                x_t[sl], t_next[sl], t[sl], velocity_cfg[sl], cond[sl], None, net,
            ))
        else:
            ref_targets.append(loss._compute_mean_velocity_d(
                x_t[sl], velocity_cfg[sl], t_next[sl], t[sl], dt[sl], cond[sl], None, net,
            ))
    ref_targets = torch.cat(ref_targets, dim=0)

    return EasyDict(
        x_aug=x_aug, noise=noise, x_t=x_t, velocity=velocity, velocity_cfg=velocity_cfg,
        mask_c=mask_c, mask_d=mask_d, alpha=alpha,
        targets=EasyDict(corrected=targets_corrected, legacy=targets_legacy, ref=ref_targets),
        weights_legacy=weights_legacy,
    )


def production_forward(loss, net, batch, cond, seed: int = 1234):
    """Runs the real loss.forward with the times injected (monkeypatched sampler)."""
    def fixed_sampler(batch_size, cur_step, device):
        _ = batch_size, cur_step, device
        return batch.t, batch.t_next, batch.dt, batch.alpha
    loss.sample_traj_params = fixed_sampler
    torch.manual_seed(seed)
    return loss.forward(net, batch.x, cond, cur_step=0)


def per_sample_velocity_loss(pred, targets, weights, eps):
    L = ((pred - targets) ** 2).flatten(1).mean(1) # [b]
    w = weights / (L.detach() + eps)
    return L, w, w * L


# ---------------------------------------------------------------------------
# A. Batch-alignment audit.
# ---------------------------------------------------------------------------

def section_a_alignment(net_v, net_s, loss_v, loss_s, device):
    print('\n=== A. Batch-alignment audit (Original Official / Corrected Velocity / Corrected State) ===')
    eps = float(loss_v.cfg.adaptive_loss_weight_eps)
    b = 6
    cond = net_v.construct_dummy_cond(b, device)
    realizable = {'fm', 'd', 'alpha0'}
    patterns = ['fm', 'd', 'alpha0', 'interleaved', 'prefix_suffix']
    for kind in patterns:
        batch = make_batch(net_v, device, kind)
        parts = reference_parts(loss_v, net_v, batch, cond)
        is_fm = torch.isclose(batch.t, batch.t_next).flatten()
        interval = (batch.t - batch.t_next)

        # Predictions (velocity model, and the state-head equivalent for the state model).
        with torch.no_grad():
            pred_v = net_v(parts.x_t, sigma=batch.t, sigma_next=batch.t_next, cond=cond)
            pred_s = torch.empty_like(parts.x_t)
            if (~is_fm).any():
                pred_s[~is_fm] = net_s(parts.x_t[~is_fm], sigma=batch.t[~is_fm], sigma_next=batch.t_next[~is_fm],
                                       cond=cond[~is_fm], head='state')
            if is_fm.any():
                pred_s[is_fm] = net_v(parts.x_t[is_fm], sigma=batch.t[is_fm], sigma_next=batch.t_next[is_fm], cond=cond[is_fm])

        # Corrected velocity loss (reference and production).
        ref_weights = torch.where(torch.isclose(batch.dt.flatten(), torch.zeros(b, device=device)),
                                  torch.ones(b, device=device), torch.full((b,), batch.alpha, device=device))
        Lv_ref, wv_ref, lossv_ref = per_sample_velocity_loss(pred_v, parts.targets.ref, ref_weights, eps)
        out_v = production_forward(loss_v, net_v, batch, cond)

        # Corrected state loss (reference and production).
        f_tgt = parts.x_t - interval * parts.targets.ref
        Ls_ref = torch.empty(b, device=device)
        if (~is_fm).any():
            Ls_ref[~is_fm] = ((pred_s[~is_fm] - f_tgt[~is_fm]) ** 2 / interval[~is_fm] ** 2).flatten(1).mean(1)
        if is_fm.any():
            Ls_ref[is_fm] = ((pred_s[is_fm] - parts.targets.ref[is_fm]) ** 2).flatten(1).mean(1)
        ws_ref = ref_weights / (Ls_ref.detach() + eps)
        losss_ref = ws_ref * Ls_ref
        out_s = production_forward(loss_s, net_s, batch, cond)

        # Legacy official loss: predictions in original order vs cat([c-targets, d-targets]).
        Lv_legacy, wv_legacy, lossv_legacy = per_sample_velocity_loss(pred_v, parts.targets.legacy, parts.weights_legacy, eps)

        # --- checks -----------------------------------------------------
        # (1) corrected target assembly == ordering-independent per-sample reference.
        # NOTE: the per-sample reference runs the backbone with batch size 1 instead of the full
        # batch, so cuBLAS/SDPA kernels and reduction orders differ; fp32 differences up to ~1e-4
        # (relative) are expected and are NOT an implementation error (verified by the fact that the
        # production loss matches the reference to ~1e-7). We therefore use a 1e-3 tolerance here.
        d_corr_ref = rel_diff(parts.targets.corrected, parts.targets.ref)
        report(f'[{kind}] corrected targets == per-sample reference targets (batch-size fp32 numerics)',
               d_corr_ref < TOL_DIFF, f'max rel diff = {d_corr_ref:.3e}')
        # (2) production velocity loss == corrected reference loss
        d_prod_v = rel_diff(out_v['total'], lossv_ref)
        report(f'[{kind}] production Velocity loss == corrected reference',
               d_prod_v < TOL_DIFF, f'max rel diff = {d_prod_v:.3e}')
        # (3) production state loss == corrected state reference
        d_prod_s = rel_diff(out_s['total'], losss_ref)
        report(f'[{kind}] production State loss == corrected reference',
               d_prod_s < TOL_DIFF, f'max rel diff = {d_prod_s:.3e}')
        # (4) State == Velocity (same function, same weighting) per sample
        d_sv = rel_diff(out_s['total'], out_v['total'])
        report(f'[{kind}] State loss == Velocity loss (per sample)',
               d_sv < TOL_DIFF, f'max rel diff = {d_sv:.3e}')
        # (5) legacy official vs corrected: equal on realizable patterns, different on adversarial
        d_legacy_corr = rel_diff(parts.targets.legacy, parts.targets.corrected)
        d_legacy_loss = rel_diff(lossv_legacy, lossv_ref)
        if kind in realizable:
            report(f'[{kind}] (realizable) original-official pairing == corrected',
                   d_legacy_corr < TOL_FP32, f'target rel diff = {d_legacy_corr:.3e}')
        else:
            n_mismatch = int((parts.targets.legacy - parts.targets.corrected).abs().flatten(1).max(1).values.gt(1e-6).sum())
            report(f'[{kind}] (adversarial) original-official pairing MISPAIRS samples (expected)',
                   d_legacy_corr > TOL_DIFF and n_mismatch > 0,
                   f'target rel diff = {d_legacy_corr:.3e}, mismatched samples = {n_mismatch}/{b}')
        # (6) no NaN/Inf
        assert_finite(f'[{kind}] losses (v/s/legacy)', out_v['total'], out_s['total'], lossv_legacy)


# ---------------------------------------------------------------------------
# B. Branch-by-branch equivalence: d-branch / alpha=0 JVP / FM / CFG / clamp / adaptive.
# ---------------------------------------------------------------------------

def _state_ref_loss(pred_s, parts, is_fm, interval, eps, ref_weights):
    b = pred_s.shape[0]
    f_tgt = parts.x_t - interval * parts.targets.ref
    L = torch.empty(b, device=pred_s.device)
    if (~is_fm).any():
        L[~is_fm] = ((pred_s[~is_fm] - f_tgt[~is_fm]) ** 2 / interval[~is_fm] ** 2).flatten(1).mean(1)
    if is_fm.any():
        L[is_fm] = ((pred_s[is_fm] - parts.targets.ref[is_fm]) ** 2).flatten(1).mean(1)
    w = ref_weights / (L.detach() + eps)
    return L, w, w * L


def run_case(cfg, net_v, net_s, cond, device, kind, loss_overrides=None):
    """Builds one deterministic branch-combination case and returns all intermediate quantities."""
    loss_overrides = loss_overrides or {}
    loss_v = build_loss(cfg, device, 'velocity', **loss_overrides)
    loss_s = build_loss(cfg, device, 'state', **loss_overrides)
    eps = float(loss_v.cfg.adaptive_loss_weight_eps)
    batch = make_batch(net_v, device, kind)
    parts = reference_parts(loss_v, net_v, batch, cond)
    b = batch.x.shape[0]
    is_fm = torch.isclose(batch.t, batch.t_next).flatten()
    interval = batch.t - batch.t_next
    ref_weights = torch.where(
        torch.isclose(batch.dt.flatten(), torch.zeros(b, device=device)),
        torch.ones(b, device=device), torch.full((b,), batch.alpha, device=device),
    )
    with torch.no_grad():
        pred_v = net_v(parts.x_t, sigma=batch.t, sigma_next=batch.t_next, cond=cond)
        pred_s = torch.empty_like(parts.x_t)
        if (~is_fm).any():
            pred_s[~is_fm] = net_s(parts.x_t[~is_fm], sigma=batch.t[~is_fm],
                                   sigma_next=batch.t_next[~is_fm], cond=cond[~is_fm], head='state')
        if is_fm.any():
            pred_s[is_fm] = net_v(parts.x_t[is_fm], sigma=batch.t[is_fm],
                                  sigma_next=batch.t_next[is_fm], cond=cond[is_fm])
    Lv, wv, lossv = per_sample_velocity_loss(pred_v, parts.targets.ref, ref_weights, eps)
    Ls, ws, losss = _state_ref_loss(pred_s, parts, is_fm, interval, eps, ref_weights)
    out_v = production_forward(loss_v, net_v, batch, cond)
    out_s = production_forward(loss_s, net_s, batch, cond)
    return EasyDict(
        batch=batch, parts=parts, is_fm=is_fm, interval=interval, ref_weights=ref_weights,
        loss_v=loss_v, loss_s=loss_s, eps=eps,
        Lv=Lv, wv=wv, lossv=lossv, Ls=Ls, ws=ws, losss=losss,
        out_v=out_v, out_s=out_s, pred_v=pred_v, pred_s=pred_s,
    )


def report_case_equivalence(name, res):
    """Reports the common velocity-vs-state equivalence checks for one case."""
    d_unscaled = rel_diff(res.Ls, res.Lv)
    d_weights = rel_diff(res.ws, res.wv)
    d_final = rel_diff(res.losss, res.lossv)
    d_prod_v = rel_diff(res.out_v['total'], res.lossv)
    d_prod_s = rel_diff(res.out_s['total'], res.losss)
    d_sv = rel_diff(res.out_s['total'], res.out_v['total'])
    report(f'[{name}] un-scaled loss equal', d_unscaled < TOL_FP32, f'max rel diff = {d_unscaled:.3e}')
    report(f'[{name}] adaptive weights equal', d_weights < TOL_FP32, f'max rel diff = {d_weights:.3e}')
    report(f'[{name}] adaptive-weighted final loss equal', d_final < TOL_DIFF, f'max rel diff = {d_final:.3e}')
    report(f'[{name}] production V loss == reference', d_prod_v < TOL_DIFF, f'max rel diff = {d_prod_v:.3e}')
    report(f'[{name}] production S loss == reference', d_prod_s < TOL_DIFF, f'max rel diff = {d_prod_s:.3e}')
    report(f'[{name}] State loss == Velocity loss (per sample)', d_sv < TOL_DIFF, f'max rel diff = {d_sv:.3e}')
    assert_finite(f'[{name}] losses/preds', res.out_v['total'], res.out_s['total'], res.pred_v, res.pred_s)


# ---------------------------------------------------------------------------
# C0. Full-network output mapping: state head vs velocity head (preconditioning,
#     skip connection, adaLN scale/shift), and the "equality only at init" fact.
# ---------------------------------------------------------------------------

def section_c0_full_mapping(net_s, device):
    print('\n=== C0. Full-network output mapping f_state == z_t - (t-r) u_velocity ===')
    b = 4
    cond = net_s.construct_dummy_cond(b, device)
    # Use one-hot conditioned labels to exercise the conditional branch too.
    cond.label = torch.zeros(b, *net_s.label_shape, device=device)
    for i in range(b):
        cond.label[i, i % int(net_s.label_shape[0])] = 1.0
    torch.manual_seed(11)
    z = torch.randn(b, *net_s.input_shape, device=device)

    final_layer = net_s.model.final_layer
    w_v, b_v = final_layer.linear.weight, final_layer.linear.bias
    w_s, b_s = final_layer.linear_state.weight, final_layer.linear_state.bias
    exact_negation = bool(torch.equal(w_s, -w_v) and torch.equal(b_s, -b_v))
    report('head parameters are exact negations (W_s == -W_v, b_s == -b_v)', exact_negation,
           f'max |W_s+W_v|={float((w_s + w_v).detach().abs().max()):.3e}, '
           f'max |b_s+b_v|={float((b_s + b_v).detach().abs().max()):.3e}')
    shared_modulation = (final_layer.linear is not final_layer.linear_state) and \
                        hasattr(final_layer, 'adaLN_modulation') and hasattr(final_layer, 'norm_final')
    report('heads live in the same FinalLayer (adaLN modulation + norm_final are applied once, '
           'the heads are the only separate modules)', shared_modulation)

    # Capture (a) the backbone input, (b) the shared final-norm/adaLN features fed into each head,
    # (c) each head's raw token-space output, and (d) the backbone image-space output (post-head).
    # This proves the mapping holds at the *complete network output* level (preconditioning + skip
    # connection + shared scale/shift), not merely "weights negated".
    captured = {}
    backbone_inputs = {}
    backbone_outputs = {}

    def backbone_hook(module, args, kwargs, output):
        _ = module, kwargs
        backbone_inputs['last'] = args[0].detach().clone()
        backbone_outputs['last'] = (output[0] if isinstance(output, tuple) else output).detach().clone()

    def make_head_hook(tag):
        def head_hook(module, args, output):
            _ = module
            captured[tag] = {'input': args[0].detach().clone(), 'raw': output.detach().clone()}
        return head_hook

    h_backbone = net_s.model.register_forward_hook(backbone_hook, with_kwargs=True)
    h_head_v = final_layer.linear.register_forward_hook(make_head_hook('velocity'))
    h_head_s = final_layer.linear_state.register_forward_hook(make_head_hook('state'))

    cases = [(1.0, 0.0), (0.8, 0.3), (0.5, 0.499), (0.9, 0.1)]
    try:
        for t_val, r_val in cases:
            t = torch.full((b, 1, 1, 1, 1), t_val, device=device)
            r = torch.full((b, 1, 1, 1, 1), r_val, device=device)
            with torch.no_grad():
                u = net_s(z, sigma=t, sigma_next=r, cond=cond, head='velocity')
                backbone_input_velocity = backbone_inputs['last'].clone()
                features_v, raw_tokens_v = captured['velocity']['input'], captured['velocity']['raw']
                raw_v = backbone_outputs['last'].clone() # image-space backbone output = u
                f = net_s(z, sigma=t, sigma_next=r, cond=cond, head='state')
                backbone_input_state = backbone_inputs['last'].clone()
                features_s, raw_tokens_s = captured['state']['input'], captured['state']['raw']
                raw_s = backbone_outputs['last'].clone() # image-space backbone output = raw_state
            same_backbone_input = torch.equal(backbone_input_velocity, backbone_input_state)
            same_head_features = torch.equal(features_v, features_s)
            raw_negated_image = rel_diff(raw_s, -raw_v)
            raw_negated_tokens = rel_diff(raw_tokens_s, -raw_tokens_v)
            # State precond: f = c_skip*z_t + c_out*raw_s with c_skip=1, c_out=(t-r), c_in=1.
            reconstructed = z + (t - r) * raw_s
            d_recon = rel_diff(f, reconstructed)
            d_map = rel_diff(f, z - (t - r) * u)
            report(f'[t={t_val}, r={r_val}] backbone input identical for both heads', same_backbone_input)
            report(f'[t={t_val}, r={r_val}] shared final-norm/adaLN features identical for both heads',
                   same_head_features)
            report(f'[t={t_val}, r={r_val}] raw state output == -raw velocity output (token space)',
                   raw_negated_tokens < TOL_FP32, f'max rel diff = {raw_negated_tokens:.3e}')
            report(f'[t={t_val}, r={r_val}] raw state output == -raw velocity output (image space)',
                   raw_negated_image < TOL_FP32, f'max rel diff = {raw_negated_image:.3e}')
            report(f'[t={t_val}, r={r_val}] f == z_t + (t-r) raw_state (precond/skip reconstruction)',
                   d_recon < TOL_FP32, f'max rel diff = {d_recon:.3e}')
            report(f'[t={t_val}, r={r_val}] f_state == z_t - (t-r) u_velocity (full network output)',
                   d_map < TOL_FP32, f'max rel diff = {d_map:.3e}')
    finally:
        h_backbone.remove()
        h_head_v.remove()
        h_head_s.remove()

    # Degenerate target-time case r == t: the state precond has c_out = t-r = 0, so the state head
    # is not meaningful there; the FM branch uses the velocity head instead. The code path guards
    # against this with an assertion (positive interval required), which we verify here.
    t = torch.full((b, 1, 1, 1, 1), 0.7, device=device)
    raised = False
    try:
        with torch.no_grad():
            _ = net_s(z, sigma=t, sigma_next=t, cond=cond, head='state')
    except AssertionError:
        raised = True
    report('degenerate r == t: state head rejects the zero interval (FM branch uses the velocity head)',
           raised)

    # Counter-test: the mapping must BREAK once the heads are not negations. This proves the
    # relation is an initialisation-time identity, not a property of the architecture.
    t = torch.full((b, 1, 1, 1, 1), 0.8, device=device)
    r = torch.full((b, 1, 1, 1, 1), 0.3, device=device)
    with torch.no_grad():
        f_before = net_s(z, sigma=t, sigma_next=r, cond=cond, head='state')
        u_before = net_s(z, sigma=t, sigma_next=r, cond=cond, head='velocity')
        final_layer.linear_state.weight.add_(0.1) # perturb the state head only
        f_perturbed = net_s(z, sigma=t, sigma_next=r, cond=cond, head='state')
        d_perturbed = rel_diff(f_perturbed, z - (t - r) * u_before)
        # restore the exact negation
        final_layer.linear_state.weight.copy_(-final_layer.linear.weight)
        final_layer.linear_state.bias.copy_(-final_layer.linear.bias)
    report('mapping no longer holds when the state head is not the negated velocity head (expected)',
           d_perturbed > TOL_DIFF, f'rel diff after perturbing state head = {d_perturbed:.3e}')

    # NOTE (also documented): after training the two heads are independent (the velocity head is
    # only trained on the FM branch, the state head only on the consistency branch), so the
    # equality f == z_t-(t-r)u is NOT preserved; it only serves as the initialisation equivalence.


# ---------------------------------------------------------------------------
# B. Branch-by-branch equivalence: d-branch / alpha=0 JVP / FM / CFG / clamp / adaptive.
# ---------------------------------------------------------------------------

def section_b_branches(net_v, net_s, cfg, device):
    print('\n=== B. Branch equivalence (Velocity vs State) ===')
    b = 6
    cond = net_v.construct_dummy_cond(b, device)
    scenarios = [
        ('d-branch (0<alpha<=1)', 'd', {}),
        ('alpha=0 JVP branch', 'alpha0', {}),
        ('FM branch (r=t)', 'fm', {}),
        ('CFG-guided velocity', 'd', {'cfg_params': EasyDict(omega=0.8, kappa=0.2, t_min=0.0, t_max=1.0)}),
        ('clamp_utgt active', 'd_all', {'clamp_utgt': 0.05}),
    ]
    for name, kind, loss_overrides in scenarios:
        res = run_case(cfg, net_v, net_s, cond, device, kind, loss_overrides)
        report_case_equivalence(name, res)
        # CFG scenario: make sure guidance actually changed the target (test is meaningful)
        if 'CFG' in name:
            d_guided = rel_diff(res.parts.velocity_cfg, res.parts.velocity)
            report(f'[{name}] guidance actually modifies the target velocity', d_guided > TOL_DIFF,
                   f'rel diff(velocity_cfg, velocity) = {d_guided:.3e}')
        # clamp scenario: clipping is active in velocity space and preserved by the state conversion
        if 'clamp' in name:
            max_tgt = float(res.parts.targets.ref.abs().max())
            f_tgt = res.parts.x_t - res.interval * res.parts.targets.ref
            u_recovered = (res.parts.x_t - f_tgt) / res.interval
            d_recover = rel_diff(u_recovered, res.parts.targets.ref)
            report(f'[{name}] state conversion preserves clamped target',
                   max_tgt <= float(res.loss_v.cfg.clamp_utgt) + 1e-6 and d_recover < TOL_FP32,
                   f'max |u_tgt| = {max_tgt:.4f}, recover rel diff = {d_recover:.3e}')


# ---------------------------------------------------------------------------
# E. Realistic branch-combination tests (alpha=0 + CFG, alpha=1 skip, JVP no clipping,
#    ratio_fm=0/1 boundaries).
# ---------------------------------------------------------------------------

def section_e_combinations(net_v, net_s, cfg, device):
    print('\n=== E. Real branch-combination matrix ===')
    b = 6
    cond = net_v.construct_dummy_cond(b, device)

    # E1: alpha=0 (JVP) + CFG guidance active.
    res = run_case(cfg, net_v, net_s, cond, device, 'alpha0',
                   {'cfg_params': EasyDict(omega=0.8, kappa=0.2, t_min=0.0, t_max=1.0)})
    report_case_equivalence('alpha=0 + CFG', res)
    d_guided = rel_diff(res.parts.velocity_cfg, res.parts.velocity)
    jvp_samples = int(((res.batch.dt.flatten() == 0) & (~res.is_fm)).sum())
    report('[alpha=0 + CFG] guidance active and JVP branch present',
           d_guided > TOL_DIFF and jvp_samples > 0,
           f'guidance rel diff = {d_guided:.3e}, JVP samples = {jvp_samples}/{b}')

    # E2: alpha=1 -> target branch is skipped (`u_tgt == clip(velocity_cfg)`), state converts to z_r.
    res = run_case(cfg, net_v, net_s, cond, device, 'alpha1')
    report_case_equivalence('alpha=1 target skip', res)
    clamp = float(res.loss_v.cfg.clamp_utgt)
    clipped_v = torch.clip(res.parts.velocity_cfg, -clamp, clamp)
    d_skip = rel_diff(res.parts.targets.ref, clipped_v)
    report('[alpha=1 target skip] target == clip(velocity_cfg) (target branch skipped)', d_skip < TOL_FP32,
           f'max rel diff = {d_skip:.3e}')
    # The state target is then f_tgt = z_t-(t-r)u_tgt = z_r (true endpoint state, no target net call).
    f_tgt = res.parts.x_t - res.interval * res.parts.targets.ref
    d_state = rel_diff(f_tgt, res.parts.x_t - res.interval * clipped_v)
    report('[alpha=1 target skip] state target f_tgt == z_t-(t-r)clip(velocity_cfg)', d_state < TOL_FP32,
           f'max rel diff = {d_state:.3e}')

    # E3: clipping semantics: c-branch (FM / JVP) is NOT clipped; d-branch IS clipped in velocity space.
    batch = make_batch(net_v, device, 'mixed')
    loss_small = build_loss(cfg, device, 'velocity', clamp_utgt=0.05)
    loss_large = build_loss(cfg, device, 'velocity', clamp_utgt=4.0)
    parts_small = reference_parts(loss_small, net_v, batch, cond)
    parts_large = reference_parts(loss_large, net_v, batch, cond)
    mask_c = (batch.dt == 0).flatten()
    mask_d = ~mask_c
    d_c = rel_diff(parts_small.targets.ref[mask_c], parts_large.targets.ref[mask_c]) if mask_c.any() else 0.0
    max_c = float(parts_large.targets.ref[mask_c].abs().max()) if mask_c.any() else 0.0
    report('[clipping semantics] c-branch (FM + JVP) targets are NOT clipped',
           d_c < TOL_FP32 and max_c > 0.05,
           f'clamp 0.05 vs 4.0 rel diff = {d_c:.3e}, max |u_tgt|(c) = {max_c:.4f} (exceeds clamp, not clipped)')
    if mask_d.any():
        d_d = rel_diff(parts_small.targets.ref[mask_d], parts_large.targets.ref[mask_d])
        max_d = float(parts_large.targets.ref[mask_d].abs().max())
        report('[clipping semantics] d-branch targets ARE clipped to clamp_utgt', d_d > TOL_DIFF and max_d > 0.05,
               f'rel diff = {d_d:.3e}, max |u_tgt|(d) = {max_d:.4f}')
        report('[clipping semantics] clipped d-branch target obeys clamp bound',
               float(parts_small.targets.ref[mask_d].abs().max()) <= 0.05 + 1e-6,
               f'max |clip(u_tgt)| = {float(parts_small.targets.ref[mask_d].abs().max()):.4f}')

    # E4: ratio_fm = 0 boundary (all samples on the MF/consistency branch).
    for kind, label in (('d_all', 'ratio_fm=0 (all MF, alpha>0)'), ('alpha0_all', 'ratio_fm=0 (all MF, alpha=0/JVP)')):
        res = run_case(cfg, net_v, net_s, cond, device, kind)
        report_case_equivalence(label, res)
        report(f'[{label}] batch has no FM samples (as intended)', bool((~res.is_fm).all()))

    # E5: ratio_fm = 1 boundary (all samples on the FM branch).
    res = run_case(cfg, net_v, net_s, cond, device, 'fm')
    report_case_equivalence('ratio_fm=1 (all FM)', res)
    report('[ratio_fm=1 (all FM)] batch has no MF samples (state head unused in forward; see docs for DDP note)',
           bool(res.is_fm.all()))

    # ratio_fm=1 with CFG: guidance is only used for the target velocity of the FM branch.
    res = run_case(cfg, net_v, net_s, cond, device, 'fm',
                   {'cfg_params': EasyDict(omega=0.8, kappa=0.2, t_min=0.0, t_max=1.0)})
    report_case_equivalence('ratio_fm=1 + CFG', res)
    d_guided = rel_diff(res.parts.velocity_cfg, res.parts.velocity)
    report('[ratio_fm=1 + CFG] guidance active on FM targets', d_guided > TOL_DIFF,
           f'guidance rel diff = {d_guided:.3e}')


# ---------------------------------------------------------------------------
# C. Numerical & gradient tests.
# ---------------------------------------------------------------------------

def section_c_gradients(net_v, net_s, cfg, device):
    print('\n=== C. Numerical & gradient tests ===')
    b = 6
    cond = net_v.construct_dummy_cond(b, device)
    eps = float(cfg.loss.adaptive_loss_weight_eps)

    # C1. Prediction conversion (random inputs, including tiny interval).
    torch.manual_seed(1)
    cases = [(0.95, 0.1), (0.7, 0.3), (0.5, 0.499), (0.3, 0.0003)]
    z = torch.randn(b, *net_v.input_shape, device=device)
    for t_val, r_val in cases:
        t = torch.full((b, 1, 1, 1, 1), t_val, device=device)
        r = torch.full((b, 1, 1, 1, 1), r_val, device=device)
        with torch.no_grad():
            u = net_s(z, sigma=t, sigma_next=r, cond=cond, head='velocity')
            f = net_s(z, sigma=t, sigma_next=r, cond=cond, head='state')
            expected_f = z - (t - r) * u
        d = rel_diff(f, expected_f)
        report(f'prediction conversion t={t_val}, r={r_val}', d < TOL_FP32, f'max rel diff = {d:.3e}')

    # C2. Output-gradient relation on the same underlying prediction (u as a leaf):
    #     with f = z - (t-r)u and f_tgt = z - (t-r)u_tgt, L_s(u) == L_v(u) and
    #     dL_s/du == dL_v/du. This is the "output gradient" equivalence.
    torch.manual_seed(2)
    z = torch.randn(b, 1, 2, 2, device=device)
    u_tgt = torch.randn(b, 1, 2, 2, device=device)
    interval = torch.rand(b, 1, 1, 1, device=device) + 0.2
    u = torch.randn(b, 1, 2, 2, device=device, requires_grad=True)
    f = z - interval * u
    f_tgt = z - interval * u_tgt
    Lv = ((u - u_tgt) ** 2).flatten(1).mean(1).sum()
    Ls = ((f - f_tgt) ** 2 / interval ** 2).flatten(1).mean(1).sum()
    report('loss values: L_s(u) == L_v(u)', rel_diff(Ls.detach(), Lv.detach()) < TOL_FP32,
           f'max rel diff = {rel_diff(Ls.detach(), Lv.detach()):.3e}')
    g_u = torch.autograd.grad(Lv, u, retain_graph=True)[0]
    g_s = torch.autograd.grad(Ls, u, retain_graph=True)[0]
    d = rel_diff(g_s, g_u)
    report('output gradients: dL_s/du == dL_v/du (same prediction)', d < TOL_FP32, f'max rel diff = {d:.3e}')

    # C3. Adaptive-weight denominator is detached.
    # With loss = sum_i w_i L_i and w_i = c/(L_i.detach()+eps), d(loss)/dL_i == w_i exactly.
    L_leaf = (torch.randn(b, device=device).abs() + 0.1).requires_grad_(True)
    c = 0.5
    loss_detached = (c / (L_leaf.detach() + eps) * L_leaf).sum()
    g = torch.autograd.grad(loss_detached, L_leaf)[0]
    expected = c / (L_leaf.detach() + eps)
    d = rel_diff(g, expected)
    report('adaptive loss: d(loss)/dL == w.detach() (detached denominator)', d < TOL_FP32, f'max rel diff = {d:.3e}')

    # C4. Parameter gradients: shared trunk identical, state head == -velocity head.
    loss_v = build_loss(cfg, device, 'velocity')
    loss_s = build_loss(cfg, device, 'state')
    batch = make_batch(net_v, device, 'd_all') # consistency branch only => one head per model
    parts = reference_parts(loss_v, net_v, batch, cond)
    is_fm = torch.isclose(batch.t, batch.t_next).flatten()
    interval = batch.t - batch.t_next
    ref_weights = torch.where(torch.isclose(batch.dt.flatten(), torch.zeros(b, device=device)),
                              torch.ones(b, device=device), torch.full((b,), batch.alpha, device=device))

    net_v.zero_grad(set_to_none=True)
    pred_v = net_v(parts.x_t, sigma=batch.t, sigma_next=batch.t_next, cond=cond)
    Lv, wv, _ = per_sample_velocity_loss(pred_v, parts.targets.ref, ref_weights, eps)
    ((wv * Lv).mean()).backward()
    grads_v = {n: p.grad.detach().clone() for n, p in net_v.named_parameters() if p.grad is not None}

    net_s.zero_grad(set_to_none=True)
    # `d_all` puts every sample on the consistency branch, so both models get gradients through
    # exactly one output head and the comparison is meaningful.
    pred_s = net_s(parts.x_t, sigma=batch.t, sigma_next=batch.t_next, cond=cond, head='state')
    Ls, ws, _ = _state_ref_loss(pred_s, parts, is_fm, interval, eps, ref_weights)
    ((ws * Ls).mean()).backward()
    grads_s = {n: p.grad.detach().clone() for n, p in net_s.named_parameters() if p.grad is not None}

    trunk_s = [n for n in grads_v if 'final_layer.linear' not in n]
    trunk_ok = all(torch.allclose(grads_v[n], grads_s[n], rtol=1e-4, atol=1e-6) for n in trunk_s)
    worst = max(float((grads_v[n] - grads_s[n]).abs().max()) for n in trunk_s)
    report(f'parameter gradients: shared trunk identical ({len(trunk_s)} tensors)', trunk_ok,
           f'max abs diff = {worst:.3e}')
    gv = grads_v.get('model.final_layer.linear.weight')
    gs = grads_s.get('model.final_layer.linear_state.weight')
    head_ok = gv is not None and gs is not None and torch.allclose(gs, -gv, rtol=1e-4, atol=1e-6)
    head_diff = float((gs + gv).abs().max()) if (gv is not None and gs is not None) else float('nan')
    report('parameter gradients: state head == -velocity head', head_ok, f'max |g_s + g_v| = {head_diff:.3e}')
    finite_ok = all(bool(torch.isfinite(g).all()) for g in grads_v.values()) and \
                all(bool(torch.isfinite(g).all()) for g in grads_s.values())
    grad_norm = float(torch.cat([g.flatten() for g in grads_v.values()]).norm())
    report('parameter gradients are finite and non-vanishing', finite_ok and grad_norm > 0.0,
           f'grad norm = {grad_norm:.3e}')

    # C5. Division-by-zero safety on a tiny interval for the state path.
    tiny = make_batch(net_v, device, 'd')
    tiny.t = torch.full((b, 1, 1, 1, 1), 0.5, device=device)
    tiny.t_next = torch.full((b, 1, 1, 1, 1), 0.499, device=device)
    tiny.dt = 0.5 * (tiny.t - tiny.t_next)
    loss_s2 = build_loss(cfg, device, 'state')
    out_tiny = production_forward(loss_s2, net_s, tiny, cond)
    report('tiny interval (t-r=1e-3): state loss finite', bool(torch.isfinite(out_tiny['total']).all()),
           f'max |loss| = {float(out_tiny["total"].abs().max()):.3e}')


# ---------------------------------------------------------------------------
# D. Sampling equivalence (no extra NFE).
# ---------------------------------------------------------------------------

def section_d_sampling(net_v, net_s, cfg, device):
    print('\n=== D. Sampling equivalence (state vs velocity), per-step intermediates ===')
    b = 2
    torch.manual_seed(3)
    z = torch.randn(b, *net_v.input_shape, device=device)

    def run(net, pairs, scale):
        """Returns the per-step intermediate states: [(x_next, x_denoised), ...]."""
        sampling_cfg = copy.deepcopy(cfg.sampling)
        sampling_cfg.enable_trajectory_sampling = True
        sampling_cfg.enable_consistency_sampling = False
        sampling_cfg.cfg = EasyDict(scale=scale, t_min=0.0, t_max=1.0)
        x = z
        trajectory = []
        for t_cur, t_next in pairs:
            x, x_denoised, _ = reverse_recflow_step(
                net=net, sampling_cfg=sampling_cfg, x_cur=x,
                # Real sampler passes per-sample t as [b, 1, 1, 1, 1]; broadcasting with a [b]
                # tensor would align the batch dimension with the last spatial dim.
                t_cur=torch.full((b, 1, 1, 1, 1), float(t_cur), device=device),
                t_next=torch.full((b, 1, 1, 1, 1), float(t_next), device=device),
                sc_latents=None, dtype=torch.float32,
            )
            trajectory.append(EasyDict(x_next=x.clone(), x_denoised=x_denoised.clone()))
        return trajectory

    cases = [
        ('NFE-1', [(1.0, 0.0)], 1.0),
        ('NFE-2', [(1.0, 0.5), (0.5, 0.0)], 1.0),
        ('NFE-1 + CFG(2.0)', [(1.0, 0.0)], 2.0),
        ('NFE-2 + CFG(1.5)', [(1.0, 0.5), (0.5, 0.0)], 1.5),
    ]
    for name, pairs, scale in cases:
        traj_v = run(net_v, pairs, scale)
        traj_s = run(net_s, pairs, scale)
        all_steps_ok = True
        # Compare EVERY intermediate state, not only the final one.
        for step_idx, (step_v, step_s) in enumerate(zip(traj_v, traj_s)):
            d_next = rel_diff(step_s.x_next, step_v.x_next)
            d_denoised = rel_diff(step_s.x_denoised, step_v.x_denoised)
            ok = d_next < 1e-4 and d_denoised < 1e-4
            all_steps_ok = all_steps_ok and ok
            report(f'[{name}] step {step_idx + 1}/{len(pairs)} intermediate state == velocity',
                   ok, f'x_next rel diff = {d_next:.3e}, x_denoised rel diff = {d_denoised:.3e}')
        d_final = rel_diff(traj_s[-1].x_next, traj_v[-1].x_next)
        report(f'[{name}] final generated state equal', d_final < 1e-4, f'max rel diff = {d_final:.3e}')
        report(f'[{name}] outputs finite', bool(torch.isfinite(traj_s[-1].x_next).all()))

    # NFE-1 state output should be the analytic z - (t-r)u at (t=1, r=0).
    sampling_cfg = copy.deepcopy(cfg.sampling)
    sampling_cfg.enable_trajectory_sampling = True
    sampling_cfg.cfg = EasyDict(scale=1.0, t_min=0.0, t_max=1.0)
    with torch.no_grad():
        out_s, _, _ = reverse_recflow_step(
            net=net_s, sampling_cfg=sampling_cfg, x_cur=z,
            t_cur=torch.full((b, 1, 1, 1, 1), 1.0, device=device),
            t_next=torch.zeros(b, 1, 1, 1, 1, device=device),
            sc_latents=None, dtype=torch.float32,
        )
        u = net_v(z, sigma=torch.ones(b, 1, 1, 1, 1, device=device), sigma_next=torch.zeros(b, 1, 1, 1, 1, device=device), cond=net_v.construct_dummy_cond(b, device))
        analytic = z - 1.0 * u
    d = rel_diff(out_s, analytic)
    report('NFE-1 state output == z_1 - u(z_1, 0, 1) (analytic)', d < 1e-4, f'max rel diff = {d:.3e}')


# ---------------------------------------------------------------------------

def main():
    disable_trivial_warnings()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        dist.init()
        dist.init_random_state_and_cuda(seed=1, cudnn_benchmark=False, allow_tf32=False)
        misc.force_math_attention() # alpha=0 branch uses torch.autograd.functional.jvp

    torch.manual_seed(0)
    cfg, net_v, net_s, loss_v, loss_s = build_setup(device)
    print(f'State heads initialized from velocity head; params: {sum(p.numel() for p in net_v.parameters()):,} (velocity) / '
          f'{sum(p.numel() for p in net_s.parameters()):,} (state)')

    section_a_alignment(net_v, net_s, loss_v, loss_s, device)
    section_b_branches(net_v, net_s, cfg, device)
    section_c0_full_mapping(net_s, device)
    section_c_gradients(net_v, net_s, cfg, device)
    section_d_sampling(net_v, net_s, cfg, device)
    section_e_combinations(net_v, net_s, cfg, device)

    print()
    failed = [name for name, ok, _ in RESULTS if not ok]
    if failed:
        print(f'RESULT: FAILED ({len(failed)}/{len(RESULTS)} checks)')
        for name in failed:
            print(f'  - {name}')
        return 1
    print(f'RESULT: ALL {len(RESULTS)} CHECKS PASSED')
    print('Stage 1 (verified state parameterization) is ready for the 1-2k step sanity check.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
