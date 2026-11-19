# Stage 1 — Strictly equivalent state parameterization

状态参数化模型直接预测目标时刻状态：

```
f_theta(z_t, r, t) = z_t - (t - r) * u_theta(z_t, r, t)
```

阶段一的目标是：**不引入任何新机制**（无 EMA teacher、无多步一致性），只把输出参数化从
`u` 换成 `z_r`，并且与官方 velocity 模型**严格数学等价**（可逐项核对 loss / 梯度），
作为后续机制消融的干净起点。

---

## 1. 原版 `AlphaFlowLoss` 各分支的准确计算过程（按代码，不按公式）

代码位置：`src/training/loss.py`，`AlphaFlowLoss.forward`（约 L546-640）及其辅助函数。

### 1.1 采样 `sample_traj_params`（L449-464）

- `ratio_fm = get_ratio(cfg.ratio_fm)`；`alpha = get_ratio(cfg.alpha)`
- batch 前 `batch_size_fm = int(batch_size * ratio_fm)` 个样本是 **FM 分支**：
  `t_fm = t_next_fm`（即 `r = t`），`dt_fm = 0`
- 其余为 **MF 分支**：`t_mf > t_next_mf`（`t > r`），`dt_mf = alpha * (t_mf - t_next_mf)`
- 注意 `get_ratio`（L380-427）的 clamp 语义：
  - `current_ratio < clamp_value` → **0.0**（除非 `discrete_training`）
  - `current_ratio > 1 - clamp_value` 或超过 `up_clamp_value` → **1.0**
  - 所以 α-Flow sigmoid 配方后期会**精确出现 α=0**（不是只趋近 0）。

### 1.2 CFG 速度 `_compute_velocity_cfg`（L466-501）

对 `t_min < t < t_max` 的样本，

```
velocity_cfg = omega * v + kappa * u_theta(z_t, t, t | cond) + (1 - omega - kappa) * u_theta(z_t, t, t | uncond)
```

其中 `v = noise - x`；非 cfg 配方 `omega=1, kappa=0` 时 `velocity_cfg = v`。
（label dropout 会在该函数内直接改写 `cond.label`。）

### 1.3 分支划分（L564-569）

```
mask_c = (dt == 0)   # 包含两类样本：FM 分支(r=t) + α=0 的 MF 样本(t>r)
mask_d = (dt > 0)    # 0 < α <= 1 的一致性分支
```

### 1.4 `_compute_mean_velocity_c`（L503-528，`dt == 0` 的样本）

- 若 `t_next == t`（真正的 FM 样本）：目标 `u_tgt = velocity_cfg`（**不 clip**）。
- 若 `t_next < t`（α=0 的 MF 样本）：用 JVP 构造局部展开目标
  ```
  u_tgt = velocity_cfg - (t - r) * d/dt u_theta(z_t, r, t)   # 沿真实轨迹方向 (v, 1)
  ```
  其中 JVP 的切向量是 `(velocity_cfg, 0, 1)`，**也不 clip**。
- 该函数整体在 `@torch.no_grad()` 下（目标分支共享参数 + stop-gradient，**不是 EMA**）。

### 1.5 `_compute_mean_velocity_d`（L530-543，`0 < α <= 1`）

```
z_s = z_t - dt * velocity_cfg                 # 对应 s = t - dt = α r + (1-α) t
u^- = u_theta(z_s, s, r)                       # stop-grad
u_tgt = (dt * velocity_cfg + (s - r) * u^-) / (t - r)
      = α * velocity_cfg + (1 - α) * u^-        # (s - r) = (1-α)(t-r)
u_tgt = clip(u_tgt, -clamp_utgt, +clamp_utgt)  # clamp_utgt = 4.0
```

- `α = 1` 特判：`dt = t - r`，此时 `1 - dt/(t-r) == 0` → `u^-` 不计算，`u_tgt = velocity_cfg`，
  随后同样经过 `clamp_utgt`。
- `velocity_cfg`（含 cfg 修正）参与 `z_s` 构造，也参与组合。

### 1.6 预测与 adaptive loss（L579-644）

```
pred = net(z_t, sigma=t, sigma_next=r)          # velocity head，全 batch 一次前向
L_i  = mean_over_dims((pred_i - u_tgt_i)^2)     # [b]
weight_i = 1                 (mask_c 样本)
         = alpha             (mask_d 样本)
loss_i   = weight_i * L_i / (L_i.detach() + adaptive_loss_weight_eps)   # eps = 0.001
```

另外记录 `trajectory_FM = mean((pred - velocity_cfg)^2)`、
`trajectory_consistency = mean(2*(velocity_cfg - u_tgt)*pred)`（decomposed logging，不反传）。

### 1.7 代码里发现的三个实现事实（不是公式推测）

1. **目标分支是共享参数 + stop-gradient**（`@torch.no_grad()` 直接调用同一个 `net`），
   官方实现里**没有** EMA teacher。EMA teacher 因此是阶段二的一个真实机制差异。
2. **batch 目标拼接与对齐**：原实现把目标按 `cat([c 目标, d 目标])` 重排，而预测按原始
   batch 顺序计算；只有当 `mask_c` 是 batch 前缀（或全为 True）时才逐样本对应。
   **进一步审计结论（修正早期判断）**：`alpha` 是**每步一个标量**，所以
   `dt = alpha·(t-r)` 对同一 batch 的所有 MF 样本相同 → `mask_c` 只可能是
   - `alpha > 0`：只有前缀 FM 样本 `dt=0` → `mask_c` 是前缀；
   - `alpha == 0`：所有 MF 样本 `dt=0` → `mask_c` 为全 True。

   因此**官方采样器能产生的任何 batch 上，原实现都是对齐的**，并不存在实际错配。
   修正版按原始样本顺序回填目标（`mean_velocity[mask_c] = ...`）：在所有真实 batch 上与原版
   **逐位一致**，只对"非均匀 dt 模式"（例如将来改成 per-sample alpha 调度）保持正确。
   审计脚本用**人为构造的 interleaved mask** 证明：原版配对在该模式下错配，
   修正版与"逐样本独立参考实现"一致；真实模式（fm / d / alpha0）下三版本结果相同。
3. **`clamp_utgt` 的位置**：只在 d 分支、且只在**速度空间**裁剪；c 分支（FM + α=0 JVP）
   不裁剪。state 转换必须保留该语义（先得到 `u_tgt` → 裁剪 → `f_tgt = z_t-(t-r)u_tgt`）。

---

## 2. 阶段一实现（代码改动）

| 文件 | 改动 |
| --- | --- |
| `configs/model/dit.yaml` | 新增 `parameterization: velocity\|state`、`state_head_init: negate_velocity` |
| `src/training/dit.py` | `FinalLayer` 增加可选第二输出头 `linear_state`；`SiT.forward(head=...)`；state 头输出 `-u` |
| `src/training/networks.py` | `Diffusion._construct_input/_forward` 支持 `head='state'`：`c_skip=1, c_out=(t-r)`，于是 `diffusion_output = z_t - (t-r)u` |
| `src/training/loss.py` | `AlphaFlowLoss.forward` 支持 state 模式：<br>• `r=t` 的 FM 样本继续用 velocity 头，loss 不变；<br>• 其余样本用 state 头预测 `f`，`L = mean((f - f_tgt)^2/(t-r)^2)`，其中 `f_tgt = z_t-(t-r)u_tgt`（`u_tgt` 仍由官方 velocity 目标构造得到，含 CFG 修正与 `clamp_utgt`）；<br>• adaptive weighting 完全不变（c 样本权重 1，d 样本权重 α，逐样本除以 `L.detach()+eps`）；<br>• 目标按原始样本顺序回填（修复 1.7-2 的错位） |
| `src/inference.py` | `reverse_recflow_step` 支持 state 采样：`z_next = f(z_t, t_next, t)`；velocity 由 `(z_t-z_next)/(t-t_next)` 解析恢复（**不增加 NFE**）；CFG 在 velocity 空间施加后转回 state |
| `src/training/network_utils.py` | `init_state_head_from_velocity_()`：把 velocity 头取负拷贝到 state 头；<br>• 新建模型：初始化即严格等价；<br>• 加载 released ckpt（无 state 头）：加载后从**已加载的 velocity 头**重新取负初始化；<br>• 恢复自己的训练：state 头来自 snapshot，不覆盖 |
| `scripts/check_state_equivalence.py` | 等价性检查（见 §3） |
| `scripts/make_finetune_snapshot.py` | 把 released ckpt（`ema, cfg`）转换成训练用 snapshot（`net, ema, cfg`），用于从官方权重微调 |

### 关键设计点

- **state 头输出 `-u`**：`f = z_t + (t-r)*raw_state`，取 `raw_state = -u` 即得
  `f = z_t - (t-r)u`。这样 velocity 头与 state 头在初始化时**互为负拷贝**，
  `f(z_t,r,t) == z_t-(t-r)u(z_t,r,t)` 精确成立。
- **`r=t` 的 FM 样本必须走 velocity 头**：`f(z_t,t,t) ≡ z_t`，`t-r=0`，state 形式无信号
  且会除零。因此需要 `ratio_fm > 0`（官方 alphaflow/meanflow 配方均为 0.25–0.5），
  velocity 头会持续被 FM 样本训练，CFG 训练用的 `u(z_t,t,t)` 也保持有效。
- **目标分支保持官方构造**：`u_tgt` 仍由 velocity 空间、共享参数 + stop-grad 的官方路径
  计算，再转成 `f_tgt`。这样 stage 1 与官方 loss 完全对应；stage 2 再把目标分支换成
  EMA teacher（真正的机制改动）。
- **alpha=1 / alpha=0 分支**：
  - `α=1`（dt = t-r，`s=r`）：`u_tgt = clip(velocity_cfg)` → `f_tgt = z_t-(t-r)u_tgt`，
    即用真实终点状态监督（等价于原版 `s=r` 情形）。
  - `α=0`（dt=0 但 t>r）：仍走原版 JVP 局部展开目标 `u_tgt = v_cfg-(t-r)JVP`，
    再转成 `f_tgt`；state 头只在主分支使用。
- **数值稳定**：`(t-r)^2` 除以的区间严格 > 0（state 分支断言），α=0 样本走 JVP 分支不除零；
  `clamp_utgt` 仍在 velocity 空间生效后再转 state，等价性不受影响。

---

## 3. 等价性验证套件（不训练、不需要数据/权重）

`scripts/check_state_equivalence.py` 在小 pixel-space SiT（registry 的 smoke preset，64×64，
dummy dataset）上建立**同一函数**的 velocity 模型与 state 模型（net_s 的 state 头 =
net_v 的 velocity 头取负），然后执行四组检查（fp32、确定性、约 1 分钟）：

### A. Batch 对齐审计（三版本对照）

| 版本 | 实现 |
| --- | --- |
| Original Official | 测试脚本内的 legacy 参考：`cat([c 目标, d 目标])` 与预测原序配对 |
| Corrected Velocity | 生产代码的 velocity 路径（按原始顺序回填目标） |
| Corrected State | 生产代码的 state 路径（同上 + state 头） |

对每个模式检查：逐样本目标 == "逐样本独立参考实现"；生产 loss == 参考 loss；
state == velocity；以及 legacy 与 corrected 的差异。
- 真实模式（`fm` / `d` / `alpha0`，对应官方采样器能产生的 mask）：三版本一致；
- 对抗模式（`interleaved`、`prefix_suffix`，人为构造的非均匀 dt）：legacy 出现
  样本错配（报告错配样本数），corrected 仍与逐样本参考一致。

### B. 分支与公式等价（Velocity vs State）

- `0 < alpha <= 1` 的 d 分支；
- `alpha = 0` 的 JVP 局部展开分支；
- `r = t` 的 FM 分支；
- CFG-guided velocity（`omega=0.8, kappa=0.2`，并检查 guidance 确实改变了目标）；
- `clamp_utgt=0.05`（检查裁剪生效、且 state 转换保留裁剪后的目标）；
- adaptive weighting：未加权 loss、权重、detach 分母、最终 loss 逐项相等。

### C. 数值与梯度

- 逐样本预测转换（含 `t-r = 1e-3` 的极小间隔）；
- 输出梯度（同一预测 u 下 `L_s(u) == L_v(u)` 且 `dL_s/du == dL_v/du`）；
- 网络参数梯度：共享 trunk 完全相同、state 头梯度 == −velocity 头梯度、有限且非零；
- adaptive 分母 detach 验证（`d(loss)/dL == w.detach()`）；
- NaN/Inf/除零检查（各分支 + 极小间隔）。

### D. 采样等价（不额外增加 NFE）

- NFE-1、NFE-2、NFE-1+CFG(2.0)、NFE-2+CFG(1.5)：state 采样 == velocity 采样；
- NFE-1 state 输出 == 解析结果 `z_1 - u(z_1,0,1)`。

运行（服务器仓库根目录）：

```bash
cd ~/qiangchen/AFP
git pull
python scripts/check_state_equivalence.py
```

期望末尾：`RESULT: ALL <N> CHECKS PASSED`（FAILED 时列出具体失败项与数值差异）。

> 说明：由于 dual-head 方案本身多了一个输出头，**训练动力学**与单头 velocity 模型不是
> 逐参数完全相同（state 头只从一致性分支拿梯度，velocity 头只从 FM 分支拿梯度）；
> 但"loss 值 / 每分支梯度 / trunk 梯度 / 采样输出"在初始化点严格等价。
> 这是阶段一"严格等价重参数化"的准确含义，也是后续消融要隔离的结构差异。

---

## 4. 阶段一之后的最小 sanity check（1–2k steps）

先训 dummy pixel 版（几分钟，验证双头、state loss、state 采样都能跑通，不需要 ImageNet）：

```bash
# velocity 对照（官方参数化）
bash scripts/train_single_gpu.sh alphaflow-smoke-test-single-gpu \
  desc=stage1-smoke-velocity training.max_steps=1000

# state 版
bash scripts/train_single_gpu.sh alphaflow-smoke-test-single-gpu \
  desc=stage1-smoke-state model.parameterization=state training.max_steps=1000
```

看 `runs/stage1-smoke-{velocity,state}/log.txt` 中 `Loss/total` 曲线：两者应在起始
**几乎重合**（同初始化、同 batch、同 loss 公式），随后因输出参数化/双头结构略有差异；
同时记录 `Loss/trajectory_*`、`Grad/norm`、有无 NaN/Inf、以及最终生成样本
（`runs/stage1-smoke-*/generate/` 或 log 中保存的图）。

再在 latent B/2 上做 1–2k steps 微调（需要先把 `data/imagenet/train` 接好）：

```bash
# 1) 把 released ckpt 转成训练 snapshot
python scripts/make_finetune_snapshot.py \
  snapshots/alphaflow-B-2.pt snapshots/alphaflow-B-2-finetune-init.pt

# 2) velocity 对照（同起点、同步数、同数据）
bash scripts/train_single_gpu.sh alphaflow-sigmoid-latentspace-B-2-single-gpu \
  desc=stage1-ft2k-velocity \
  training.resume.on_start_ckpt_path="$PWD/snapshots/alphaflow-B-2-finetune-init.pt" \
  training.resume.whole_state=false \
  model.ema_rampup_ratio=null \
  training.max_steps=2000 \
  training.freqs.snapshot=null training.freqs.snapshot_latest=null \
  training.freqs.metrics=null training.freqs.save_samples=null

# 3) state 版（唯一差异：parameterization=state + allow_missing_keys）
bash scripts/train_single_gpu.sh alphaflow-sigmoid-latentspace-B-2-single-gpu \
  desc=stage1-ft2k-state \
  model.parameterization=state \
  training.resume.on_start_ckpt_path="$PWD/snapshots/alphaflow-B-2-finetune-init.pt" \
  training.resume.whole_state=false \
  training.resume.allow_missing_keys=true \
  model.ema_rampup_ratio=null \
  training.max_steps=2000 \
  training.freqs.snapshot=null training.freqs.snapshot_latest=null \
  training.freqs.metrics=null training.freqs.save_samples=null
```

---

## 5. 已知限制 / 注意事项

- `ratio_fm > 0` 是 state 模型的硬性要求（velocity 头必须被 FM 样本训练，否则目标分支与
  CFG 的 `u(z_t,t,t)` 会失效；DDP 下 velocity 头无梯度还会触发 unused-parameter 报错）。
- α=0 分支仍依赖 JVP → 训练时仍需 `training.force_math_attention: true`（已配置）。
- α<0 不支持（原版负权重 MSE 无意义）；训练范围保持 `0 < α <= 1`。
- state 采样不增加 NFE；NFE-1 时输出直接就是 `z_0`。
- 阶段二（EMA teacher）与阶段三（多步一致性 / nested consistency）在
  `parameterization` 开关之外实现，避免与本阶段混淆。
