# Stage 1 — Strictly equivalent state parameterization

状态参数化模型直接预测目标时刻状态：

```
f_theta(z_t, r, t) = z_t - (t - r) * u_theta(z_t, r, t)
```

阶段一的目标是：**不引入任何新机制**（无 EMA teacher、无多步一致性），只把输出参数化从
`u` 换成 `z_r`，并且与官方 velocity 模型**严格数学等价**（可逐项核对 loss / 梯度），
作为后续机制消融的干净起点。

## 0. 版本定义（后续 FID 对比的基准口径）

后续所有 FID 对比都必须明确使用哪一版作为基准。三版本定义：

| 版本 | 含义 | 用途 |
| --- | --- | --- |
| **Original Official** | 官方发布代码的原始语义：velocity 参数化 + 原版 `cat([c 目标, d 目标])` 配对 | **仅作审计/数学参照**；不直接用于产出 FID 数字 |
| **Corrected Velocity** | 本仓库生产代码的 velocity 路径：velocity 参数化 + 按原样本顺序回填目标 | **正式 FID 基线**（所有对比实验的基准） |
| **Corrected State** | 本仓库生产代码的 state 路径：state 参数化 + 同样的目标回填 | 候选模型（阶段一验证对象） |

关键事实：**Corrected Velocity 与 Original Official 在官方采样器能产生的所有 batch 上
逐位一致**（见 §1.7 与验证套件 A 组）。因此用 Corrected Velocity 作 FID 基线不会偏离
官方实现；Original Official 只用于展示"如果使用非均匀 mask 的假设实现会错配"。

---

## 1. 原版 `AlphaFlowLoss` 各分支的准确计算过程（按代码，不按公式）

代码位置：`src/training/loss.py`，`AlphaFlowLoss.forward`（约 L546-640）及其辅助函数。

### 1.1 采样 `sample_traj_params`（L449-464）

- `ratio_fm = get_ratio(cfg.ratio_fm)`；`alpha = get_ratio(cfg.alpha)`
- `batch_size_fm = int(batch_size * ratio_fm)`，**FM 样本固定在 batch 前 `batch_size_fm` 位**：
  `t_fm = t_next_fm`（即 `r = t`），`dt_fm = 0`
- 其余为 **MF 分支**：`t_mf > t_next_mf`（`t > r`），`dt_mf = alpha * (t_mf - t_next_mf)`
- 组装方式（代码事实）：`t = cat([t_fm, t_mf])`、`t_next = cat([t_next_fm, t_next_mf])`、
  `dt = cat([dt_fm, dt_mf])`；随后 loss 里这些逐样本张量**按原顺序**作用在 `x`/`cond` 上，
  中间没有任何 permutation。因此 **FM 样本始终是 batch 前缀**（`ratio_fm=0` 时为空，
  `ratio_fm=1` 时为全 batch，都平凡满足前缀性质）。
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

   **精确条件（代码事实）**：FM 样本始终是 batch 前缀，来自 `sample_traj_params` 的组装顺序
   `t = cat([t_fm, t_mf])`、`dt = cat([dt_fm, dt_mf])`（`dt_fm` 全 0），且 loss 中这些逐样本
   张量按原顺序作用于 `x`/`cond`，中间没有 permutation。于是
   `mask_c = (dt == 0)` 是否整体为前缀，只取决于 MF 部分：
   - `alpha > 0` 且对**所有** MF 样本都 `t > r` 严格成立 → `mask_c` 恰好是 FM 前缀；
   - `alpha == 0` → 所有 MF 样本 `dt=0` → `mask_c` 为全 True（前缀性质平凡成立）；
   - `ratio_fm = 0` / `1` → `mask_c` 分别为全 False / 全 True，均平凡满足；
   - **唯一破坏前缀性质的边缘情形**：`alpha > 0` 但个别 MF 样本恰好 `t == r`（`dt` 舍入为 0），
     例如使用离散 `custom_t_steps` 网格，或极端 float32 碰撞。该情形在连续 logit_norm
     采样下概率为 0（正常训练不会出现），但修正版按原序回填目标对该情形也正确，
     而 Original Official 的 `cat` 配对会错位——这正是"A 组 interleaved 对抗测试"验证的内容。

   因此**官方采样器能产生的任何 batch 上，原实现都是对齐的**，不存在实际错配。
   修正版按原始样本顺序回填目标（`mean_velocity[mask_c] = ...`）：在所有真实 batch 上与原版
   **逐位一致**，只对上述边缘/未来非均匀 dt 模式保持正确。
   审计脚本用**人为构造的 interleaved mask** 证明：原版配对在该模式下错配，
   修正版与"逐样本独立参考实现"一致；真实模式（fm / d / alpha0）下三版本结果相同。
3. **`clamp_utgt` 的位置**：只在 d 分支、且只在**速度空间**裁剪；c 分支（FM + α=0 JVP）
   不裁剪。state 转换必须保留该语义（先得到 `u_tgt` → 裁剪 → `f_tgt = z_t-(t-r)u_tgt`）。
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

### 2.1 "梯度等价"的准确含义（区分三层，不能混为一谈）

函数映射等价（`f = z_t-(t-r)u`）**并不意味着**两个不同结构的网络参数梯度逐元素相等。
验证套件分别检查三层，成立条件如下：

| 层级 | 检查内容 | 成立条件 | 是否"相等" |
| --- | --- | --- | --- |
| **输出梯度** | `dL_s/du == dL_v/du`（对同一预测 u） | 只需 `f = z_t-(t-r)u` 在数值上成立；与网络结构无关，是纯代数关系 | 相等 |
| **trunk 参数梯度** | 共享 trunk 的 `∂L/∂θ` 两侧一致 | ①两侧 trunk 参数**完全相同**（net_s 从 net_v 复制）；②**state 头 = 取负的 velocity 头**（`W_s=-W_v, b_s=-b_v`）；③同一 loss/权重。在这些条件下 `∂L_s/∂θ = ∂L_v/∂θ` | 相等（**仅在初始化点**） |
| **head 参数梯度** | `∂L_s/∂W_s` vs `∂L_v/∂W_v` | 同上条件，且由 `raw_state = -raw_velocity` 可知 `∂L_s/∂W_s = -∂L_v/∂W_v` | **互为相反数，不是相等** |

关键限定：

- 上述 trunk 相等是**初始化点**的恒等式（因为 dual-head 模型里 velocity 头只从 FM 样本拿梯度、
  state 头只从一致性样本拿梯度；一步优化后两个头不再保持取负关系，trunk 梯度也不再相等）。
  验证脚本显式在"取负初始化 + 相同输入"的瞬间做检查，这正是"重参数化等价"的精确含义。
- **head 参数不是"等价"，而是镜像关系**：不同 head 结构（单头 vs 双头）不存在逐参数等价，
  不能把函数等价直接写成"参数梯度相同"。
- 采样等价（D 组）属于**函数/数值**层面的等价，不涉及梯度。

### 2.2 全网络输出层面的映射（preconditioning / skip / adaIN）

`f_state == z_t - (t-r)·u_velocity` 的成立**不只是"最后一层权重取负"**，而是完整网络输出层面的
恒等式，来自以下代码路径（`src/training/networks.py` `Diffusion._construct_input/_forward` +
`src/training/dit.py` `FinalLayer`）：

| 环节 | velocity 路径 | state 路径 |
| --- | --- | --- |
| precond `alphaflow` 的 `c_in` | 1 | 1（相同 → 送入 backbone 的输入**逐位相同**） |
| backbone 输入 | `c_in·z_t = z_t` | `c_in·z_t = z_t` |
| 共享 trunk / 时间嵌入 / cond | 相同 | 相同 |
| `FinalLayer.norm_final` + `adaLN_modulation` | 同一模块、同一份 scale/shift 特征 | 同左（**共享**，只有输出 linear 不同） |
| 输出头 | `raw_v`（velocity 头） | `raw_s`（state 头，初始化为 `-`velocity 头，含 bias） |
| `c_skip` / `x_skip` | `None`（无 skip） | `1.0·z_t` |
| `c_out` | `1` | `t-r`（代码中 `noise_labels_next = t-r`） |
| 最终输出 | `u = raw_v` | `f = z_t + (t-r)·raw_s` |

因此，当且仅当 **`raw_s = -raw_v`（在完整头输出上取负，权重与 bias 都取负）** 时：

```
f_state(z_t,r,t) = z_t + (t-r)·(-raw_v) = z_t - (t-r)·u_velocity(z_t,r,t)     (精确成立)
```

**重要限定（阶段一的准确表述）**：

- 该等式的成立条件是"两个头互为负拷贝 + 同一 trunk/输入"。我们在
  `init_state_head_from_velocity_()` 里精确构造这个状态（新建模型；或加载 released ckpt 后
  从**已加载的 velocity 头**取负初始化 state 头）。
- 所以它**只在初始化点（训练第 0 步）成立**（以及每次从 velocity checkpoint 重新初始化时成立）。
- **训练开始后不成立**：dual-head 模型里 velocity 头只从 FM 分支拿梯度、state 头只从一致性
  分支拿梯度，两个头是独立参数，训练后不再互为负拷贝；此时
  `u_velocity(z_t,r,t) ≠ (z_t - f_state)/(t-r)`，不能用 velocity 头反推 state 头。
- 验证套件 C0 组会：① 用 forward hook 证明两个头收到的 backbone 输入与 FinalLayer 的
  norm/adaLN 特征**逐位相同**；② 证明 `raw_s == -raw_v`、`f == z_t+(t-r)raw_s`、
  `f == z_t-(t-r)u`；③ 做**反证测试**：只扰动 state 头权重，等式必须失效（证明这不是架构
  自动性质，而是初始化等价）；④ 覆盖 `r=t` 退化情形（state 头因 `c_out=0` 输出 `z_t`，
  FM 分支不使用 state 头）。
- 在 bf16 autocast 下该等式在 IEEE 对称舍入意义下仍逐位成立（两套头计算对称）；验证脚本在
  fp32 下以 `1e-5` 相对误差容限检查。

---

## 3. 验证报告 Part I — 数学 / 数值 / 代码等价性（硬性容限）

`scripts/check_state_equivalence.py` 在小 pixel-space SiT（registry 的 smoke preset，64×64，
dummy dataset）上建立**同一函数**的 velocity 模型与 state 模型（net_s 的 state 头 =
net_v 的 velocity 头取负），执行 A–D 四组检查（fp32、确定性、约 1 分钟）。
**Part I 要求全部在下列容限内通过**：

| 检查 | 容限 |
| --- | --- |
| 预测转换 `f == z-(t-r)u`（含极小间隔） | max rel diff < 1e-5 |
| **C0 全网络映射**：backbone 输入 / FinalLayer 特征逐位相同（`torch.equal`），`raw_s == -raw_v`、`f == z+(t-r)raw_s`、`f == z-(t-r)u` | 前两项必须精确相等；其余 max rel diff < 1e-5 |
| **C0 反证**：只扰动 state 头后等式失效 | 必须 > 1e-3（证明"仅初始化等价"） |
| 未加权 loss `L_s == L_v`（d/JVP/FM 三分支） | max rel diff < 1e-5 |
| adaptive 权重 / 最终 loss | weight < 1e-5；loss < 1e-3 |
| 输出梯度 `dL_s/du == dL_v/du` | max rel diff < 1e-5 |
| trunk 参数梯度（取负初始化点） | rtol 1e-4 / atol 1e-6 |
| head 参数梯度镜像关系 `g_s == -g_v` | rtol 1e-4 / atol 1e-6 |
| 采样：**每一步中间状态** + 最终状态（NFE-1/2，含 CFG） | max rel diff < 1e-4 |
| **E 组真实分支组合**：`α=0 + CFG`、`α=1` 跳过目标分支、JVP 不裁剪、`ratio_fm=0/1`（含 CFG） | 同上的 loss/权重/梯度容限；外加分支语义断言（见 §3-E） |
| 所有分支的 loss / pred / grad | 有限（无 NaN/Inf），无除零 |

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

### C0. 全网络输出映射（preconditioning / skip / adaIN 层面）

- 用 forward hook 捕获并证明：两个头收到的 **backbone 输入逐位相同**、
  `FinalLayer.norm_final + adaLN_modulation` 产出的**特征逐位相同**；
- `raw_state == -raw_velocity`（完整头输出取负，含 bias）；
- `f == z_t + (t-r)·raw_state`（state precond：`c_skip=1, c_out=t-r, c_in=1`）；
- `f == z_t - (t-r)·u`（完整网络输出层面的目标等式）；
- 多组 `(t, r)`；`r=t` 退化情形验证 state 头以断言拒绝零区间（生产路径在 FM 分支使用
  velocity 头，不调用 state 头）；
- **反证**：只扰动 state 头权重 → 等式必须失效，证明它只是**初始化等价**而非架构性质。

### D. 采样等价（不额外增加 NFE，**逐步骤比较**）

- NFE-1、NFE-2、NFE-1+CFG(2.0)、NFE-2+CFG(1.5)：
  对**每一步**都比较 `x_next`（中间状态）与 `x_denoised`，而不只比较最终结果；
- 再检查最终生成状态；
- NFE-1 state 输出 == 解析结果 `z_1 - u(z_1,0,1)`。

### E. 真实训练分支组合矩阵

- `α=0`（JVP 分支）**+ CFG**：`omega=0.8, kappa=0.2`，检查 guidance 生效、JVP 样本存在，
  且 State/Velocity loss、权重、最终 loss 逐项一致；
- `α=1` **目标分支跳过**：验证代码走 `u_tgt == clip(velocity_cfg)` 的跳过路径
  （state 目标 `f_tgt = z_t-(t-r)u_tgt` 同样成立）；
- **JVP 分支不裁剪**：用 `clamp_utgt=0.05 vs 4.0` 对比，c 分支（FM + JVP）目标完全一致
  （不裁剪、可超过 clamp），d 分支目标被裁剪且满足 clamp 上界；
- **`ratio_fm=0` 边界**：全 MF（`α>0` 的 d 分支；`α=0` 的 JVP 分支）两种子情形；
- **`ratio_fm=1` 边界**：全 FM（无 MF 样本），并测 `ratio_fm=1 + CFG`。

运行（服务器仓库根目录）：

```bash
cd ~/qiangchen/AFP
git pull
python scripts/check_state_equivalence.py
```

期望末尾：`RESULT: ALL <N> CHECKS PASSED`（FAILED 时列出具体失败项与数值差异；所有比较都会打印
max rel diff / max abs diff）。

> 说明：由于 dual-head 方案本身多了一个输出头，**训练动力学**与单头 velocity 模型不是
> 逐参数完全相同（state 头只从一致性分支拿梯度，velocity 头只从 FM 分支拿梯度）；
> 但上述 Part I 的所有量在"取负初始化点 + 相同输入"下严格等价。
> 这是阶段一"严格等价重参数化"的准确含义，也是后续消融要隔离的结构差异。

---

## 4. 验证报告 Part II — 1–2k steps 短训练 sanity（软指标）

Part II **不要求**两种参数化的训练曲线逐点重合（它们结构不同、梯度来源不同），
只检查训练稳定性和基本一致性。**硬性条件**：loss/grad 有限（无 NaN/Inf）、
两种都跑完目标步数/正常结束。**软指标**：初始 loss 接近、grad norm 量级合理、
无爆炸/发散。

先训 dummy pixel 版（几分钟，不需要 ImageNet；`steps_per_tick=10` 让日志足够密）：

```bash
# velocity 对照
bash scripts/train_single_gpu.sh alphaflow-smoke-test-single-gpu \
  desc=stage1-smoke-velocity training.max_steps=1000 training.steps_per_tick=10

# state 版
bash scripts/train_single_gpu.sh alphaflow-smoke-test-single-gpu \
  desc=stage1-smoke-state model.parameterization=state training.max_steps=1000 training.steps_per_tick=10
```

> 本轮起 `log.txt` 的 tick 行会额外输出
> `loss_total=... loss_trajectory_FM=... loss_trajectory_consistency=... grad_norm=...`
> （原先只有 `loss` 汇总值），用于 Part II 报告。

生成 Part II 报告：

```bash
python scripts/summarize_stage1_sanity.py \
  --velocity-dir runs/stage1-smoke-velocity \
  --state-dir runs/stage1-smoke-state \
  --expected-steps 1000
```

报告内容：两边的 ticks、`loss_total` 首/末/极值、`grad_norm` 首/末/最大值、iter/sec、
NaN/Inf 检查、以及初始窗口/全程的 loss 相对差异（仅供参考，不作为"必须重合"的判据）。

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
  training.max_steps=2000 training.steps_per_tick=10 \
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
  training.max_steps=2000 training.steps_per_tick=10 \
  training.freqs.snapshot=null training.freqs.snapshot_latest=null \
  training.freqs.metrics=null training.freqs.save_samples=null
```

同样的 Part II 报告也可用于 latent 微调：

```bash
python scripts/summarize_stage1_sanity.py \
  --velocity-dir runs/stage1-ft2k-velocity \
  --state-dir runs/stage1-ft2k-state \
  --expected-steps 2000
```

---

## 5. 已知限制 / 注意事项

- `f_state == z_t-(t-r)u_velocity` **只在初始化点精确成立**（新建模型，或从 velocity
  checkpoint 取负初始化 state 头之后）。训练开始后两个头独立更新，该等式不再成立，
  不能用于把训练后的 state 模型反解成 velocity 模型。它只作为"同函数初始化"的依据。
- `ratio_fm` 的边界（E 组已测）：
  - `ratio_fm=1`（全 FM）：state 头在 forward 中完全不被使用 → DDP 需要
    `training.dp.ddp_find_unused_parameters=true`（单卡 null 策略无此问题）；
  - `ratio_fm=0`（全 MF）：state 模型里 velocity 头只在 `no_grad` 的目标/CFG 构造中使用，
    backward 无梯度 → 同样需要 `find_unused_parameters`；
  - **官方 alphaflow/meanflow 配方 `ratio_fm=0.25~0.5`**，两个头都能拿到梯度，正常训练无需改动。
- `ratio_fm > 0` 是 state 模型的推荐要求（velocity 头必须被 FM 样本训练，否则目标分支与
  CFG 的 `u(z_t,t,t)` 会失效）。
- α=0 分支仍依赖 JVP → 训练时仍需 `training.force_math_attention: true`（已配置）。
- α<0 不支持（原版负权重 MSE 无意义）；训练范围保持 `0 < α <= 1`。
- state 采样不增加 NFE；NFE-1 时输出直接就是 `z_0`。
- 阶段二（EMA teacher）与阶段三（多步一致性 / nested consistency）在
  `parameterization` 开关之外实现，避免与本阶段混淆。
