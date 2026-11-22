# 正式训练前计划：α schedule、resume 分析、空闲 profiling、A0/A1/A2 3k 配置

> 状态：**仅供审查，未启动任何训练**。本文件是正式 20k / 3k 阶段的设计依据。

---

## 1. 当前短测使用的完整 α schedule（A0/A1/A2 完全一致）

所有 1k sanity 与 50-step profiling 的 override：

```yaml
loss:
  alpha: {scheduler: sigmoid, initial_value: 1.0, end_value: 0.0,
          change_init_steps: 0, change_end_steps: 1200,
          clamp_value: 0.005, gamma: 25.0}
  ratio_fm: {scheduler: constant, initial_value: 0.5}   # 来自预设，未改
  clamp_utgt: 4.0                                       # 来自预设，未改
```

`get_ratio`（`src/training/loss.py:390`）对 sigmoid 的精确行为（`change_init_steps=0`）：

```
middle = change_end / 2 = 600
progress = (cur_step - 600) / 1200
alpha_raw = 1 / (1 + exp(25 * progress))
若 alpha_raw > 1 - 0.005 -> alpha = 1.0
若 alpha_raw < 0.005     -> alpha = 0.0
```

即 1k 短测中：

| step | 0–345 | 346 | 400 | 500 | 600 | 700 | 800 | 854 | 855–1000 |
|---|---|---|---|---|---|---|---|---|---|
| α | **1.0**（clamp） | 0.995 | 0.985 | 0.889 | **0.500** | 0.111 | 0.0153 | 0.0050 | **0.0**（clamp） |

- α 在 step≈600 处才过半，step≈855 起为 0；**最后 14.6% 的步数完全走 α=0 的 JVP 分支**；
- 由于 `middle=600` 是预算 1000 的 60%（而非 50%），短测 schedule 只是"压缩扫一遍"，**不是官方 schedule 的等比缩放**；
- α=1 时 MF 样本不调用目标网络（A2 的 teacher 此时不被使用）；α<1 后 d 分支才用 teacher，α=0 后全走 JVP（teacher 求导）。因此 A2 的 teacher 只在 step≳346 之后才真正参与目标构造。

### 1.1 重要发现：短测的 LR warmup 实际未被走完，训练接近"空转"

继承链：`alphaflow-sigmoid-latentspace-B-2-single-gpu` → `alphaflow-latentspace-B-2-single-gpu`
→ `alphaflow-256x256-b256-ditb2-sdvae-base-folder`，其中：

- optimizer：**Adam**, `betas=[0.9,0.95]`, `weight_decay=0.0`, `lr=1e-4`；
- `model.lr_scheduler.final_lr=1e-4`（= lr，因此 warmup 后为**恒定 LR**）；
- `model.lr_scheduler.num_warmup_steps` **未覆盖**，沿用 `configs/model/lr_scheduler/composite.yaml` 的 **10,000**；
- `model.lr_scheduler.max_steps = training.max_steps`。

`create_composite_lr_scheduler` 用 `max_steps` 构造 `T_max = max_steps - 10000`：

- 1k 短测：`T_max = 1000-10000 = -9000`（负数，warmup 永远走不完），
  `LinearLR(start_factor=1/10000)` ⇒ 实际 LR ≈ `1e-4 × (1e-4 + 0.9999·step/10000)`，
  即 step 0 ≈ **1e-8**、step 1000 ≈ **1e-5**，**从未到达 1e-4**；
- 这解释了为什么 1k 短测的 `loss_total` 几乎不动、A2 的 `teacher_student_rel_dist` 极小、生成样本与 step 0 权重几乎相同；
- **结论：现有 1k checkpoint 是"管线/稳定性验证产物"，不是有意义的训练权重，不能当作正式训练的起点或预热。**

正式 3k/20k 必须覆盖：

```yaml
model.lr_scheduler.num_warmup_steps: 75     # 3k：官方比例 10000/400000 = 2.5% => 75
model.lr_scheduler.num_warmup_steps: 500    # 20k：2.5% => 500
```

（`final_lr` 保持 = lr = 1e-4，与官方一致：warmup 后恒 LR。）

---

## 2. 正式 20k 建议的 α schedule

官方 B/2 预设（`alphaflow-sigmoid-latentspace-B-2`）：`max_steps=400,000`、
`change_end_steps=400,000`、`gamma=25`、`clamp=0.005`、`ratio_fm=0.5`。
`change_end_steps == max_steps` 意味着 α 在**整个预算内**扫过 1→0。

20k 采用**等比缩放**（形状与官方完全一致，因为 sigmoid 公式只依赖
`progress = (step - middle) / change_end`，预算缩放等价于时间轴缩放）：

```yaml
loss:
  alpha: {scheduler: sigmoid, initial_value: 1.0, end_value: 0.0,
          change_init_steps: 0, change_end_steps: 20000,
          clamp_value: 0.005, gamma: 25.0}
  ratio_fm: {scheduler: constant, initial_value: 0.5}
model:
  lr_scheduler: {num_warmup_steps: 500}
training:
  max_steps: 20000
```

对应分段（与官方 400k 的比例一致）：

| 阶段 | 20k 中的 step | 占预算 |
|---|---|---|
| α=1（clamp，纯 FM/d-branch α=1） | 0 – 5766 | 28.8% |
| α 下降（d 分支首次用 teacher） | 5766 – 14234 | 42.3% |
| α=0（JVP 分支） | 14234 – 20000 | 28.8% |

**与 1k 短测的区别**：① 预算 20×；② `middle` 回到预算的 50%（10k），而不是 60%；
③ α 在全程铺开，每个阶段占预算比例与官方一致；④ LR warmup 按比例设为 500 步（短测实际未走完 warmup）。

---

## 3. 能否从现有 1k checkpoint 继续训练？

**机制上可以，科学上不可以；正式实验必须从 released checkpoint 重新开始。**

技术上：1k snapshot 含 `net/ema/optimizer/lr_scheduler/stats/dataloader/vis`，
A2 的 snapshot 还含 `teacher=True`；resume 基础设施能加载它们，A2 的 teacher 也会被恢复。

**不能继续的原因**：

1. **α schedule 不连续**：1k 结束点 α=0；若在 `cur_step=1000` 之上用 20k schedule（`change_end=20000`），
   `alpha(1000) = σ(-25·(1000-10000)/20000) = σ(11.25) ≈ 1 - 1.3e-5 > 1-0.005` ⇒ **α 从 0 跳回 1.0**，
   目标构造/权重瞬间跳变；
2. **LR scheduler 状态不可移植**：`create_composite_lr_scheduler` 在构造时用 `training.max_steps` 计算
   `T_max = max_steps - num_warmup_steps`；snapshot 保存并恢复 scheduler 的 `T_max/last_epoch/base_lrs`。
   1k 短测的 `T_max=-9000`、warmup 未完成；resume 到 3k/20k 会继承这个错误状态（且 `load_state_dict`
   不会按新 `max_steps` 重建），LR 轨迹与官方不符；
3. **1k 权重本身近乎 init**（见 §1.1：LR 最高仅 ~1e-5），继续训练等于在"假预热"之上叠加新 schedule，
   归因会混在一起；
4. **公平性**：正式对照要求 A0/A1/A2 从同一 released 起点、同一 schedule、同一步数预算开始，
   唯一变量是 parameterization / target_branch。

> 若只是想验证 resume 机制（不用于正式实验），可以用**完全相同**的 1k 配置续跑 50 步
> （例如 `training.max_steps=1050` 从 `snapshot-00001000` 继续），确认 A2 的 teacher 正确恢复；
> 这一步可选，不阻塞正式计划。

---

## 4. 空闲 GPU profiling 设计（batch_gpu=16 vs 32）

### 4.1 前置条件

- GPU 必须**空闲**：`nvidia-smi` 中 `memory.free ≥ 24 GiB` 且 `utilization ≤ 20%`（可多次确认）；
- 与同事任务隔离期间进行；一旦开始不要并发其他 GPU 任务；
- 使用与正式阶段一致的 `alphaflow-sigmoid-latentspace-B-2-single-gpu` 预设。

### 4.2 测试矩阵（50 步/项，`steps_per_tick=5`）

| run | 参数化 | target_branch | batch_gpu | 目的 |
|---|---|---|---|---|
| prof-a0-b16 | velocity | shared | 16 | A0 基线吞吐/显存 |
| prof-a0-b32 | velocity | shared | 32 | batch 放大收益 |
| prof-a2-b16 | state | ema | 16 | A2（最重）吞吐/显存 |
| prof-a2-b32 | state | ema | 32 | 最重配置下 b32 是否可用 |

（可选：每项重复 2 次估计方差。）

### 4.3 记录指标

- **吞吐**：`iter/sec`（排除 startup 后取 tick 均值/中位数）、`samples/sec`、`sec/kimg/gpu`；
- **启动开销**：从启动到 `step 1` 的 wall time（数据集索引 + VAE/DiT 加载）；
- **显存**：`gpumem` 稳态最大值、`reserved` 最大值、step 1 峰值；
- **稳定性**：loss 有限、`grad_net_num_nans/infs=0`、`grad_norm` 量级、是否 OOM（退出码）；
- **有效性**：b32 的 `reserved` 必须 < 28 GiB（给系统/碎片留余量），明显超过则放弃 b32。

### 4.4 Profiling 命令（空闲 GPU，50 步/项；先确认 free≥24GiB、util≤20%）

```bash
cd ~/qiangchen/AFP
export HF_HUB_OFFLINE=1
INIT=$PWD/snapshots/alphaflow-B-2-finetune-init.pt
nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv

# 公共: 50 步 profiling；α/LR 与正式 3k 同形（只是预算缩短，不影响吞吐）
base=(
  training.resume.on_start_ckpt_path=$INIT training.resume.whole_state=false
  model.ema_rampup_ratio=null model.lr_scheduler.num_warmup_steps=10
  training.max_steps=50 training.steps_per_tick=5
  training.freqs.metrics=null training.freqs.metrics_rare=null
  training.metrics_final=[] training.freqs.loss_val=null training.freqs.loss_per_sigma=null
  training.freqs.snapshot=null training.freqs.snapshot_latest=null
  training.freqs.save_samples=null
  loss.alpha.scheduler=sigmoid loss.alpha.initial_value=1.0 loss.alpha.end_value=0
  loss.alpha.change_init_steps=0 loss.alpha.change_end_steps=50
  loss.alpha.clamp_value=0.005 loss.alpha.gamma=25.0
)

# a0-b16
bash scripts/train_single_gpu.sh alphaflow-sigmoid-latentspace-B-2-single-gpu \
  desc=prof-a0-b16 dataset.batch_gpu=16 "${base[@]}" > runs/_prof_a0_b16.out 2>&1
# a0-b32
bash scripts/train_single_gpu.sh alphaflow-sigmoid-latentspace-B-2-single-gpu \
  desc=prof-a0-b32 dataset.batch_gpu=32 "${base[@]}" > runs/_prof_a0_b32.out 2>&1
# a2-b16
bash scripts/train_single_gpu.sh alphaflow-sigmoid-latentspace-B-2-single-gpu \
  desc=prof-a2-b16 dataset.batch_gpu=16 model.parameterization=state \
  loss.target_branch=ema training.resume.allow_missing_keys=true \
  "${base[@]}" > runs/_prof_a2_b16.out 2>&1
# a2-b32
bash scripts/train_single_gpu.sh alphaflow-sigmoid-latentspace-B-2-single-gpu \
  desc=prof-a2-b32 dataset.batch_gpu=32 model.parameterization=state \
  loss.target_branch=ema training.resume.allow_missing_keys=true \
  "${base[@]}" > runs/_prof_a2_b32.out 2>&1

# 汇总
for d in prof-a0-b16 prof-a0-b32 prof-a2-b16 prof-a2-b32; do
  echo "=== $d ==="
  grep 'step ' runs/$d/log.txt | tail -n 3
  grep -E 'Peak GPU memory' runs/_prof_*.out | tail -n 4
done
```

> 注意：50 步 prof 的 `num_warmup_steps=10`、`change_end_steps=50` 只是为了保持分支覆盖，
> **吞吐/显存与 schedule 无关**；正式 3k 仍用 §5 的 75/3000。

### 4.4.1 最终决策（2026-11-22）：跳过 profiling，统一 `batch_gpu=16`

- 依据：`batch_gpu=16` 已在真实 ImageNet latent 上完成 A0/A1/A2 各 3000 步，稳定显存
  A0/A1 ~7.5GB、A2 ~8.45GB reserved（峰值 11.13GB），无 OOM、无 NaN；
- 20k 的首要目标是 **A0/A1/A2 的可比、可复现、可信对照**，而不是吞吐最优；
  global batch=256、优化器、lr、数据策略、α schedule 在 b16 下保持一致即可满足要求；
- 因此 **20k 不采用 b32**（避免引入未验证的 batch 变量与额外 profiling 等待）；
  若将来 20k 墙钟时间不可接受，再单独做吞吐优化（不影响本组实验结论）；
- 该决策不影响 §4 的判定规则本身，仅表示本阶段不执行 profiling。

### 4.5 判定（以 samples/sec 为主）

由于全局 batch 固定为 256（`num_accum_rounds = 256/batch_gpu`），
`samples/sec = 256 × iter/sec`，因此比较 b16/b32 时应以 **`samples/sec`** 为准（等价于
optimizer 步吞吐，但比 `iter/sec` 更直观地反映"每张卡每秒处理多少样本"）。

b32 通过条件（全部满足才统一使用 b32，否则统一 b16）：

1. **A2 b32 峰值 `reserved` < 28 GiB**（给系统/碎片留余量）；
2. 无 OOM、无 NaN/Inf；
3. **`samples/sec` 相对 b16 提升 ≥ 10%**；
4. 连续运行稳定（loss/grad 正常，无显存增长趋势）。

---

## 5. A0/A1/A2 统一训练到 3k steps（配置与命令）

> 目的：在正式 20k 之前，用统一条件把 A0/A1/A2 各训练 3k 步，验证
> 真实 latent 下的协议一致性、梯度/显存稳定性、A2 的 teacher 更新行为。
> **3k 是独立阶段**（自带 3k 等比 α schedule），完成后正式 20k 从 released snapshot 重启，
> 不从 3k checkpoint 续跑（理由同 §3：换 schedule 会不连续）。

### 5.1 三组共同配置（唯一变量 = parameterization / target_branch）

```yaml
training:
  max_steps: 3000
  steps_per_tick: 10
  resume: {on_start_ckpt_path: snapshots/alphaflow-B-2-finetune-init.pt, whole_state: false}
  freqs: {metrics: null, metrics_rare: null, loss_val: null, loss_per_sigma: null,
          snapshot: null, snapshot_latest: null, save_samples: 1000, metrics_final: []}
dataset:
  batch_gpu: 16          # 或 profiling 判定后的 32；三组必须一致
  batch_size: 256
  num_workers: 8
model:
  ema_rampup_ratio: null            # 恒定 EMA/teacher 衰减（β=0.9999，halflife 6931）
  lr_scheduler: {num_warmup_steps: 75}   # 官方比例 2.5%
loss:
  alpha: {scheduler: sigmoid, initial_value: 1.0, end_value: 0.0,
          change_init_steps: 0, change_end_steps: 3000,
          clamp_value: 0.005, gamma: 25.0}
  ratio_fm: {scheduler: constant, initial_value: 0.5}
  clamp_utgt: 4.0
```

- 3k 的 α 分段：α=1 直到 step≈865，α=0.5 @1500，step≈2135 后 α=0；
  A2 的 teacher 从 step≈865 起真正参与目标；
- 保持 `model.checkpointing: true`（预设值），`ema_halflife_steps=6931`（预设值）；
- `resume.allow_missing_keys=true` 仅 A1/A2 需要（state head 在 released snapshot 中缺失，
  按 `state_head_init: negate_velocity` 从 velocity head 初始化）；
- 训练结束（`done`）自动保存 `snapshot-00003000.pt`。

### 5.2 命令（待 GPU 空闲并经审查后执行）

```bash
cd ~/qiangchen/AFP
export HF_HUB_OFFLINE=1
INIT=$PWD/snapshots/alphaflow-B-2-finetune-init.pt
BG=16   # profiling 判定后用 32（三组必须相同）

COMMON=(
  training.resume.on_start_ckpt_path=$INIT training.resume.whole_state=false
  model.ema_rampup_ratio=null model.lr_scheduler.num_warmup_steps=75
  training.max_steps=3000 training.steps_per_tick=10
  training.freqs.metrics=null training.freqs.metrics_rare=null
  training.metrics_final=[] training.freqs.loss_val=null training.freqs.loss_per_sigma=null
  training.freqs.snapshot=null training.freqs.snapshot_latest=null
  training.freqs.save_samples=1000 dataset.batch_gpu=$BG
  loss.alpha.scheduler=sigmoid loss.alpha.initial_value=1.0 loss.alpha.end_value=0
  loss.alpha.change_init_steps=0 loss.alpha.change_end_steps=3000
  loss.alpha.clamp_value=0.005 loss.alpha.gamma=25.0
)

# A0: velocity + shared
bash scripts/train_single_gpu.sh alphaflow-sigmoid-latentspace-B-2-single-gpu \
  desc=stage3-a0-latent-3k "${COMMON[@]}" \
  > runs/_stage3_a0_3k.out 2>&1 &

# A1: state + shared
bash scripts/train_single_gpu.sh alphaflow-sigmoid-latentspace-B-2-single-gpu \
  desc=stage3-a1-latent-3k model.parameterization=state \
  training.resume.allow_missing_keys=true "${COMMON[@]}" \
  > runs/_stage3_a1_3k.out 2>&1 &

# A2: state + EMA teacher（唯一额外变量）
bash scripts/train_single_gpu.sh alphaflow-sigmoid-latentspace-B-2-single-gpu \
  desc=stage3-a2-latent-3k model.parameterization=state loss.target_branch=ema \
  training.resume.allow_missing_keys=true "${COMMON[@]}" \
  > runs/_stage3_a2_3k.out 2>&1 &
```

（三组应**顺序**执行还是并行？单卡上必须顺序；正式执行时用脚本串行。）

### 5.3 预期资源（基于当前短测实测，正式数字以 §4 profiling 为准）

| | A0 | A1 | A2 |
|---|---|---|---|
| 显存（b16，实测） | ~5.9/7.5GB | ~6.1/7.7GB | ~6.9/8.5GB |
| 显存（b32，待测） | ~10–13GB（估） | ~10–14GB（估） | ~11–15GB（估） |
| 运行时间（共享 GPU 实测 0.14–0.18 it/s） | 4.6–6h | 4.6–6h | 4.6–6h |
| 运行时间（空闲 GPU，待 profiling） | 预计 1–2h/arm（0.4–0.8 it/s 量级） | 同左 | 同左 |
| 快照大小（done） | ~2.6GB | ~2.6GB | ~3.4GB |
| 生成的 fakes | step 0/1000/2000 网格 | 同左 | 同左 |

> 若 GPU 仍被同事占用（99%），**不要跑 3k**：预计总时长 14–18h 且数字不可用；
> 应在空闲 profiling 后再决定。

### 5.4 3k 阶段完成标准

- 三组均 3000/3000 步、无 NaN/Inf、无 OOM、`grad_norm` 稳定（<16，不触发裁剪）；
- A0/A1 loss 曲线接近（非逐点重合）；A2 loss 同量级、teacher 字段正常
  （`teacher_ema_updates_total≈3000`、`teacher_ema_beta=0.9999`、`teacher_student_rel_dist` 单调缓增）；
- 各自 `snapshot-00003000.pt`（A2 `teacher=True`）可加载、可生成；
- 任何不满足 → 先修配置/环境，不进 20k。

---

## 6. A1 初始化转换审计（3k 前置，必须通过）

问题：从 released **velocity** checkpoint 加载 state 模型时，是否真的把**完整输出头**
（weight 和 bias）取负？`allow_missing_keys=true` 到底忽略了哪些 key？

代码事实（`src/training/network_utils.py`）：

- `init_networks` 在**加载之前**先用 `init_state_head_from_velocity_` 把 state 头初始化为
  `linear_state.load_state_dict(linear.state_dict())` 再 `neg_()` weight/bias（line 105–127）；
- `load_module_state` 在**加载之后**检查 missing keys：若含 `linear_state`，对**已加载的 velocity 头**
  再执行一次同样的取负（line 359–368）——因此 released ckpt 的 velocity 权重会正确映射到 state 头；
- `allow_missing_keys=true` 允许的 missing keys 应当**恰好是**每个模块的
  `model.final_layer.linear_state.weight` 与 `.bias` 两个 key；unexpected keys 应为 0；
  其他任何被忽略的参数都属于审计失败（脚本会 assert）。

### 审计命令（只读，不训练；需要 snapshots/`alphaflow-B-2-finetune-init.pt`）

```bash
cd ~/qiangchen/AFP
HF_HUB_OFFLINE=1 python scripts/check_released_state_init.py
```

脚本 `scripts/check_released_state_init.py` 会：

1. 用与 A1 相同的方式构建 state 模型（`model.parameterization=state`）；
2. 打印 `[net]/[ema]` 的 **exact missing / unexpected key 列表**；
3. 走生产加载路径（`load_module_state`）后逐元素检查
   `W_state == -W_velocity`、`b_state == -b_velocity`（bitwise）；
4. 用 released 权重做全网络等价检查：`f_state(z_t,r,t) == z_t-(t-r)·u_velocity`（≤1e-5）；
5. 打印 `RESULT: ALL ... CHECKS PASSED`。

期望输出要点：

```text
[net] missing keys (2): ['model.final_layer.linear_state.weight', 'model.final_layer.linear_state.bias']
[net] unexpected keys (0): []
[ema] missing keys (2): [...linear_state.weight, ...linear_state.bias]
[ema] unexpected keys (0): []
Re-initialized 1 state head(s) from the loaded velocity head (the checkpoint has no state head).
...
RESULT: ALL ... CHECKS PASSED — released velocity ckpt -> state head is exact.
```

---

## 7. A2 resume 连续性测试（3k 前置，批准执行）

目标：验证"连续训练 1050 步"与"1000 步保存 → 恢复 → 再训 50 步"在 **EMA/optimizer 状态**上连续。
只验证恢复机制，不作为训练质量结论（旧 1k 实验的 LR schedule 本身有问题）。

要求：**`whole_state=true`**（恢复 net/ema/optimizer/grad_scaler/lr_scheduler/stats/teacher），
复用现有 `runs/stage2-a2-latent-1k/snapshot-00001000.pt`。

```bash
cd ~/qiangchen/AFP
export HF_HUB_OFFLINE=1
nohup bash scripts/train_single_gpu.sh alphaflow-sigmoid-latentspace-B-2-single-gpu \
  desc=stage2-a2-latent-1k \
  model.parameterization=state loss.target_branch=ema \
  training.resume.whole_state=true \
  training.resume.allow_missing_keys=true \
  model.ema_rampup_ratio=null \
  training.max_steps=1050 training.steps_per_tick=5 \
  training.freqs.metrics=null training.freqs.metrics_rare=null \
  training.metrics_final=[] training.freqs.loss_val=null training.freqs.loss_per_sigma=null \
  training.freqs.snapshot=null training.freqs.snapshot_latest=null training.freqs.save_samples=null \
  dataset.batch_gpu=16 \
  loss.alpha.scheduler=sigmoid loss.alpha.initial_value=1.0 loss.alpha.end_value=0 \
  loss.alpha.change_init_steps=0 loss.alpha.change_end_steps=1200 \
  loss.alpha.clamp_value=0.005 loss.alpha.gamma=25.0 \
  > runs/_a2_resume_1050.out 2>&1 &
tail -f runs/_a2_resume_1050.out
```

（不传 `on_start_ckpt_path`；训练器会在输出目录发现 `snapshot-00001000.pt` 并自动恢复。）

期望日志/指标（对照用户验收表）：

| 检查项 | 期望 |
|---|---|
| 恢复日志 | `Found 1 snapshots ... resume ... snapshot-00001000.pt`；`Loaded the EMA target teacher state from the snapshot.` |
| `cur_step` | 首 tick 从 ~1005 开始，结束 1050 |
| `teacher_ema_updates`（本进程） | 1 → 50（每次 optimizer step 一次） |
| `teacher_ema_updates_total` | **1000 → 1050**（新日志字段，续上历史计数） |
| teacher 参数 | 每个 resumed step 都在变化（相邻 tick 的 teacher 不同） |
| `teacher_ema_beta` | 0.9999，与恢复前一致 |
| `eval_ema_beta` | 0.9999，与 teacher beta 一致 |
| teacher 梯度 | 始终不存在（由 F 组与实现保证） |
| optimizer/scheduler | 能成功 resume 即证明已恢复（`allow_missing_optim=false`，缺 key 会直接报错） |
| `teacher_student_rel_dist` | 从 ~0.00231 起平滑变化，**无跳变**（不应回到 ~0 或突变） |
| NaN/Inf | 0 |

通过后再进行 3k。
