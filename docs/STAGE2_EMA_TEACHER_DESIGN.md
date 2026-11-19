# Stage 2 设计文档：State + EMA Teacher（A2）

> 状态：**已实现，F 组 216/216 通过**（见 [STAGE2_EMA_IMPLEMENTATION.md](STAGE2_EMA_IMPLEMENTATION.md)）。
> 训练尚未启动，等待审核批准。实现遵循本设计：A2 相对 A1 **只改目标分支的来源**
> （student shared-stop-grad → EMA teacher），不同时引入任何其他机制。

---

## 0. 三组实验与变量归因

| Arm | 参数化 | 目标分支 | 与上一行的差异 |
| --- | --- | --- | --- |
| **A0** | velocity | shared stop-grad（当前生产代码） | 基线 |
| **A1** | state | shared stop-grad（Stage 1 已验证实现） | 参数化（A0→A1） |
| **A2** | state | **EMA teacher** | 目标来源（A1→A2） |

A0/A1/A2 共享：同一 released ckpt 起点、同一数据、同一优化器/lr/batch、同一 α 与 ratio_fm
调度、同一 seed、同一 eval EMA、同一 NFE/seed/真实分布评测协议。只有上表列出的变量不同。

---

## 1. EMA decay 与更新时机

**decay 公式（与仓库现有 eval EMA 完全一致，不新增超参）**：

```
H_t = min(H, t · r)              if ema_rampup_ratio r is not None
    = H                          otherwise
β_t = 0.5 ** (1 / max(H_t, 1e-8))
θ̄_t = β_t · θ̄_{t-1} + (1 − β_t) · θ_t          (逐元素，对应 lerp_(θ_t, 1−β_t))
```

其中 `H = model.ema_halflife_steps`（B/2 preset 为 6931，即 β≈0.9999），
`t = stats.cur_step`，`r = model.ema_rampup_ratio`。这与 `src/train.py::update_ema`（L464-480）
逐行一致。

**更新时机**：

- 在 student 的 `optimizer.step()` 与 grad clip 之后更新，与现有 eval EMA 同一步；
- 每个 `model.ema_update_freq` 步（默认 1）更新；
- step 0 时 teacher 尚未更新过 → `θ̄_0 = θ_0`，A2 的 target 与 A1 完全一致（见 §8）；
- teacher 从不在 optimizer.step 前被用于生成"当前 batch 的监督"，因此不存在信息泄漏。

**为什么是独立的 teacher 实例、而不是复用 `nets.ema`（eval EMA）**：

1. 若复用，A2 的训练目标会与"最佳 ckpt 选择 / 可视化 / 生成"共用同一个 EMA，语义耦合，
   相当于同时改了评估路径——违反"只改目标分支"；
2. 独立实例可以用**相同超参**（复用 `ema_halflife_steps`/`ema_rampup_ratio`）从而不引入新
   超参数，A0/A1 的评估行为也完全不变；
3. 代价是 B/2 模型多一份参数（约 0.5GB 显存 + 0.5GB ckpt 体积），单卡 5090 可承受。

---

## 2. Student 与 Teacher 的初始化

初始化顺序（A2，从 released ckpt 微调）：

1. `init_networks`：构造 student（`model.parameterization=state`）；
2. `load_states`：加载 released B/2 权重；state 头缺失 → `load_module_state` 中
   `init_state_head_from_velocity_()` 用**已加载的 velocity 头**取负初始化；
3. **加载完成之后** `teacher = copy.deepcopy(student)`，置 `teacher.eval().requires_grad_(False)`；
   此时 `θ̄_0 = θ_0` **逐位相等**（含 trunk、velocity 头、state 头、buffers）。

从 scratch 训练 A2 时同理：teacher = deepcopy(刚初始化的 student)，仍满足 `θ̄_0 = θ_0`。

**恢复训练**：

- A2 的 snapshot 在原有 `net`/`ema` 之外新增 `teacher` 字段，恢复时严格加载（`allow_missing_keys`
  仅允许 state 头缺失的情形）；
- 如果 snapshot 没有 `teacher` 字段（例如误从 A1 的 ckpt 切 A2），则用 `deepcopy(student)`
  初始化 teacher，并打印显式 warning（此时不满足"从同一 teacher 轨迹恢复"，需要重新起跑）。

---

## 3. 两个输出头与共享 trunk 的 EMA 更新规则

- **统一规则**：teacher 的**所有**浮点参数与 buffers 都做 EMA，包括
  trunk（blocks、time embedders、label embedder、pos_embed、adaLN、tokenizer 等）、
  **velocity 头**、**state 头**、以及 LDM 的 normalization buffers；
- 不区分头、不做 per-head 特殊处理、不冻结任何部分；整数/非浮点 buffer 跳过
  （与现有 `update_ema` 一致）；
- 不参与 EMA 的：optimizer state、RNG 状态、训练统计量；
- **训练目标只使用 teacher 的 velocity 头**（与 A1 中"目标始终在 velocity 空间构造"一致）；
  teacher 的 state 头**不参与 loss**，只被 EMA 更新/保存，供完整性与后续分析/可视化使用。
  这一点是设计选择，避免把 A2 悄悄变成"用 teacher 的 state 头做目标"（那会是另一个变量）。

---

## 4. Buffer 与 normalization 的处理

- 代码事实：现有 `update_ema` 对 `named_parameters() + named_buffers()` 中所有浮点张量做
  `lerp_`。teacher 完全复用该规则，因此与 A0/A1 的 eval EMA 语义一致。
- 本代码库的 normalization 是 LayerNorm/adaLN（无 running mean/var，无 BatchNorm），
  EMA buffers 不会引入额外统计量；唯一实际的 float buffers 是 LDM 的
  `autoencoder_mean/var`、`are_ldm_stats_inited` 等——它们在初始化/统计阶段决定后不再变化，
  EMA 等价于固定拷贝（lerp 后数值稳定）。
- **dtype**：teacher 与 student 同 `param_dtype`，EMA 在该 dtype 下按现有实现进行；不额外引入
  fp32 master weights（避免多一个变量）。若后续发现 bf16 EMA 有累积误差，再单独立项。
- **多卡**：单卡 `dp.strategy=null`，teacher 无需 DDP 包裹；多卡时 teacher 不参与 DDP，
  靠"student 已同步 + 相同 β"保持各 rank 一致；设计上预留周期性 broadcast 兜底
  （`_broadcast_teacher_params_()`），本轮不实现。

---

## 5. Teacher 的 no_grad 机制

- `teacher.eval()`（无 dropout 随机性；配置 dropout=0 时与 student 前向一致）+
  `requires_grad_(False)`；
- 所有 teacher 前向都在 `torch.no_grad()` 内（现有 `_compute_velocity_cfg` /
  `_compute_mean_velocity_c/d` 已经是 `@torch.no_grad()`，只需把传入的 `net` 换成 teacher）；
- teacher 不加入 optimizer 参数组（`init_optimizers` 只取 `nets.net`），不参与
  `grad_clip`/`grad_scaler`/DDP 通信；
- 断言级测试：backward 后 teacher 所有参数 `.grad is None`；teacher 前向后 `requires_grad` 属性不变；
- 若 `torch.autograd.functional.jvp` 在"参数 requires_grad=False 的模块"上不可用
  （取决于 torch 版本实现），teacher 的 α=0 JVP 分支改用 `torch.func.jvp` 直接对
  `(z_t, t, r)` 做前向模式求导（数学等价，不建参数梯度图）；实现前先做单测确认。

---

## 6. 各分支在 A2 下的处理

| 分支 | A1（现状，student shared stop-grad） | A2（teacher） |
| --- | --- | --- |
| **0 < α < 1**（d 分支） | `u^- = u_θ(z_s,r,s)`；`u_tgt = α v_cfg + (1−α)u^-`；`clip(u_tgt, ±clamp_utgt)`；`f_tgt = z_t−(t−r)u_tgt` | 只把 `u^-` 换成 `u_θ̄(z_s,r,s)`；其余（`v_cfg`、`z_s`、clip、f_tgt 公式）完全不变 |
| **α = 1**（s=r，目标分支跳过） | `u_tgt = clip(v_cfg)`（非 cfg 即 `clip(v)`），**不调用目标网络** | 与 A1 **完全相同**（不引入 teacher 调用） |
| **α = 0**（JVP 局部展开） | `u_tgt = v_cfg − (t−r)·D_t u_θ`，其中 `D_t u_θ = ∂_t u_θ + J_z u_θ·v_cfg` | `D_t u_θ` 换成 `D_t u_θ̄`（jvp 在 teacher 上；teacher 冻结、无梯度） |
| **FM（r=t）** | velocity 头预测 + `u_tgt = clip(v_cfg)`，无目标网络调用 | 与 A1 **完全相同** |
| **CFG（仅 cfg 训练：κ≠0）** | `velocity_cfg = ω v + κ u_θ(cond) + (1−ω−κ)u_θ(uncond)`，网络调用在 student 上 | 这些**目标构造**用的网络调用换成 teacher 的 velocity 头（`u_θ̄`）。非 cfg 主矩阵（`ω=1,κ=0`）不触发 |
| **clamp_utgt** | d 分支、速度空间裁剪 | 位置不变；只影响 teacher 构造出的 `u_tgt` |
| **adaptive weighting** | `w=1`（c）/`α`（d），`w·L/(L.detach()+eps)` | **完全不变** |
| **FM 主预测 / state 头预测** | student | student（不涉及 teacher） |

> 结论：A2 与 A1 的差异可以精确描述为"**所有 `no_grad` 目标构造中的网络评估从 θ 换成 θ̄**"，
> 不新增/删除任何一次前向调用（数量与 A1 相同），不改变 loss 结构。

---

## 7. State Teacher 目标的数学推导

A1 的 d 分支（0<α<1）：

```
z_s = z_t − α(t−r)·v_cfg,      s = t − α(t−r),      s−r = (1−α)(t−r)
u_tgt^A1 = α·v_cfg + (1−α)·u_θ(z_s, r, s)
f_tgt^A1 = z_t − (t−r)·u_tgt^A1
```

A2 只替换目标来源 `u_θ → u_θ̄`：

```
u_tgt^A2 = α·v_cfg + (1−α)·u_θ̄(z_s, r, s)
f_tgt^A2 = z_t − (t−r)·[α·v_cfg + (1−α)·u_θ̄(z_s, r, s)]
         = z_t − α(t−r)·v_cfg − (1−α)(t−r)·u_θ̄(z_s, r, s)
         = z_s − (s−r)·u_θ̄(z_s, r, s)
```

即：**A2 的 state 目标 = teacher 在中间状态 `z_s`、目标时刻 `r` 处隐含的终点状态**。
注意它是"teacher 的 velocity 头在 velocity 空间内插后转成 state"，**不是** teacher 的
state 头输出（teacher state 头只在保存/分析中使用）。

α=1：`f_tgt = z_t−(t−r)·clip(v_cfg) = z_r`（真实终点状态，无 teacher 调用）。

α=0（局部展开）：

```
u_tgt^A2 = v_cfg − (t−r)·D_t u_θ̄,    D_t u_θ̄ = ∂_t u_θ̄(z_t,r,t) + J_z u_θ̄(z_t,r,t)·v_cfg
f_tgt^A2 = z_t − (t−r)·u_tgt^A2
```

FM（r=t）：state 分支不使用，velocity 头主预测 + `u_tgt = clip(v_cfg)`。

由于 `f_tgt` 与 `u_tgt` 之间是固定的线性变换（`f_tgt = z_t−(t−r)u_tgt`），A2 的 loss 表达式与
A1 **完全同构**，唯一区别是 target 中出现的网络参数是 `θ̄` 而非 `θ`。

---

## 8. 初始化时 EMA 与 Shared stop-grad 的等价性测试（实现后、训练前必须全过）

计划在 `scripts/check_state_equivalence.py` 增加 **F 组**（新增 `target_branch=ema` 的构造开关）：

1. **参数逐位一致**：`teacher = deepcopy(student)` 后，teacher 与 student 的所有
   `named_parameters()`/`named_buffers()` `torch.equal`；
2. **loss/target 等价**：固定同一 batch、`t/t_next/dt/α/cond/noise`，A1（shared）与
   A2（teacher）的逐样本 un-scaled loss、weights、最终 loss 相等（≤1e-6）；
   覆盖 d 分支、α=0 JVP、FM、α=1 skip、CFG（κ≠0）、clamp 生效、ratio_fm=0/1；
3. **student 梯度等价**：同一固定 batch 下，A1 与 A2 的 student 参数梯度逐元素一致
   （≤1e-4 rtol），trunk/velocity head/state head 分别检查；
4. **teacher 无梯度**：backward 后 teacher 全部 `.grad is None`；teacher 参数在
   `optimizer.param_groups` 中不存在（object identity）；
5. **EMA 单步规则**：手动 `student += δ` 后执行一次 EMA 更新，验证
   `teacher_new == β·teacher_old + (1−β)·student_new`（按 §1 公式逐元素，β 用同一函数计算）；
6. **frozen 语义**：更新 student 后再前向 teacher，输出不随 student 当前值变化（只随 teacher 值）；
7. **jvp 分支**：A2 的 α=0 JVP 在 teacher 上可计算，且与 A1（同参数时）数值一致；
8. **snapshot 往返**：保存含 `teacher` 的 snapshot → 新进程加载 → teacher 逐位一致；
9. **step-0 训练日志**：真实训练第 1 个 tick 的 `loss_total` 与 A1 同配置下在容限内一致。

任何一项失败都先修实现，不进入训练。

---

## 9. 后续实验的对照变量与评价指标

### 对照变量

| 变量 | A0 | A1 | A2 |
| --- | --- | --- | --- |
| 参数化 | velocity | state | state |
| 目标来源 | student shared stop-grad | student shared stop-grad | **EMA teacher** |
| 其余（数据/起点/步数/lr/α/ratio_fm/clamp/CFG/eval EMA/评测协议） | 相同 | 相同 | 相同 |

### 评价指标

- **主指标**：`fid50k_full`（与 Stage 0 相同协议：真实分布用 val 50k、seed 0-49999、NFE-1 与
  NFE-2 分别报告）。**FID 的基线是 A0（Corrected Velocity）**；
- **辅助指标**（环境允许时）：`dinofid50k_full`（≈FDD）、`fcd50k_full`；
- **机制诊断（不作"更好"的判据）**：
  - `loss_total` / `trajectory_FM` / `trajectory_consistency` 曲线；
  - `grad_norm`、`grad_net_num_nans/infs`；
  - **teacher–student 参数距离**（每 N 步记录相对 L2），用来确认 teacher 是否在移动；
  - `Progress/ema_beta`（已有）与 teacher β（相同公式）；
- **稳定性硬指标**：无 NaN/Inf、跑完目标步数、checkpoint 可加载/生成 4 张图非退化。

### 预算与归因注意

- 2k sanity 中 `halflife=6931` 使 teacher 几乎不动（β≈0.9999）→ **这一步只验证"接入正确 +
  训练稳定"，不能用来判断 EMA 是否有效**；
- 若要观察 EMA 效果，需要足够长的正式预算（例如 20k 步量级，待你批准）或者单独做
  "短 halflife" 消融——**短 halflife 必须作为独立变量单独标注，不混入主矩阵**；
- 归因规则：只有 A2 vs A1 在同预算同协议下的 FID 差异才能归因于 EMA teacher；A1 vs A0
  的差异归因于参数化；任何额外机制不得同时引入。

---

## 10. 实现计划（批准后才执行，本轮只设计）

| 文件 | 计划改动 |
| --- | --- |
| `configs/loss/alphaflow.yaml`（或新 config group） | 新增 `target_branch: shared`（默认，行为不变）；A2 用 `ema` |
| `src/training/network_utils.py` | `init_modules` 中创建 teacher（在 load_states 之后 deepcopy）；`save_snapshot`/`load_states` 支持 `teacher` 字段；resume 时校验 |
| `src/train.py` | 每个优化步后：eval EMA 更新（不变）+ teacher EMA 更新（复用 `update_ema` 逻辑）；记录 teacher–student 距离 |
| `src/training/loss.py` | `AlphaFlowLoss.forward(..., teacher=None)`；`_compute_velocity_cfg` / `_compute_mean_velocity_c/d` / JVP 改为可指定 `target_net`（默认 student）；主预测仍用 student |
| `src/inference.py`、`scripts/generate.py`、`scripts/evaluate.py` | **不改**（teacher 不参与推理/评测；所有 arm 生成统一用 `nets.ema`） |
| `scripts/check_state_equivalence.py` | 增加 F 组（§8 的 9 项） |
| 文档 | 本文件 + 实验 registry（待批准后加 A2 preset） |

设计选择（请审核时确认）：

1. teacher 用**独立实例**（不复用 eval EMA），超参与 eval EMA 相同；
2. **不新增超参数**（复用 `ema_halflife_steps`/`ema_rampup_ratio`/`ema_update_freq`）；
3. teacher 目标只用 velocity 头；state 头仅 EMA/保存；
4. CFG 训练的 target 构造也切到 teacher（仅影响未来 cfg 实验；非 cfg 主矩阵不触发）；
5. 多卡 teacher 同步广播本轮不实现（单卡无影响）。

---

## 11. 风险与缓解

| 风险 | 缓解 |
| --- | --- |
| teacher 在 2k sanity 内几乎冻结，看不到机制效果 | 明确只验证接入/稳定性；正式预算或短-halflife 独立消融 |
| 显存 +0.5GB（teacher）+ snapshot 变大 | 50 步测速实测；必要时 batch_gpu 32→16 |
| `torch.autograd.functional.jvp` 对 `requires_grad=False` 参数的兼容性 | 实现前单测；不行则 teacher JVP 改用 `torch.func.jvp` |
| 多卡 teacher 漂移（未来） | 单卡无风险；预留 periodic broadcast |
| 误用 teacher 参与推理导致评测口径变化 | 文档+代码约束：所有 arm 生成只用 `nets.ema` |
| 额外机制混入 | A2 只允许 `target_branch` 一个变量；其他改动单独 PR/单独消融 |

---

## 12. 本轮不做

- 不写 EMA 实现代码；
- 不启动 EMA 训练；
- 不引入 Nested Consistency 或其他机制；
- 不启动 20k 步或更长训练。
