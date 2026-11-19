# Stage 1 → Stage 2 之间：真实 ImageNet latent 小规模验证计划

> 状态：**计划阶段**。本文件不启动训练，只定义要做什么、怎么验收。
> 边界：本阶段**不追求 FID**，只验证 Stage 1 实现（Corrected Velocity / Corrected State）
> 在真实 latent 数据管线 + 真实模型规模下能稳定运行；不引入 EMA / Nested Consistency；
> 不启动 20k 步或更长训练。

---

## 0. 目标与验收边界

| 项目 | 是否验证 |
| --- | --- |
| 真实 ImageNet train 数据管线（软链 + meta + latent 编码） | ✅ |
| 真实 B/2 规模模型的显存 / 吞吐 / iter-sec | ✅ |
| 训练日志（loss 分量、grad norm、NaN 计数） | ✅ |
| checkpoint 保存/恢复（含 dual-head 的 state head） | ✅ |
| state 采样器在真实 latent + SD-VAE 下出图 | ✅ |
| FID / FDD / FCD | ❌ 不在本阶段 |
| 生成质量结论 | ❌ 不作为本阶段结论 |

验收通过的判据（Part II 式的硬性条件）：
1. 50 步测速跑通、无 OOM、无 NaN/Inf，`iter/sec` 与 `gpumem` 记录在案；
2. 两个 arm 各 1k 步跑完：loss 有限、`grad_net_num_nans/infs=0`、两边都正常结束；
3. state arm 的最终 snapshot 能被 `generate_single_gpu.sh` 正常加载并生成图像
   （验证 dual-head checkpoint 往返）；
4. 初始 loss 两条曲线接近（软指标，不要求逐点一致）。

---

## 1. 数据准备（ImageNet train split，只读引用 `/mnt/win`）

val 已在 Stage 0 准备好（`data/imagenet/val`，50k + meta）。训练需要 train split：

```bash
cd ~/qiangchen/AFP

# 1) 用硬链接/软链方式把 train 类目录引进来（不复制 138GB 图片）
mkdir -p data/imagenet/train
cp -rs /mnt/win/data/imagenet/train/. data/imagenet/train/

# 2) 展平 + 生成 .meta.json（后台跑，约 20-40 分钟、几 GB 元数据）
nohup python scripts/data_scripts/prepare_imagenet.py \
  --target_directory ./data/imagenet --skip-download --splits train --mode symlink \
  > runs/_prepare_train.log 2>&1 &
tail -f runs/_prepare_train.log

# 3) 校验
python scripts/check_setup.py
#    期望看到 ImageNet train: 1281167 images / 1281167 meta files
df -h .    # 仓库盘剩余空间（生成图/快照需要 ~5-10GB）
```

> 若 `cp -rs` 因 `/mnt/win` 权限失败，改用 `cp -as`（archive + symlink）；若 `prepare_imagenet.py`
> 报告 "already prepared" 但 meta 数不对，先看 `runs/_prepare_train.log` 的 count 行再决定是否
> 加 `--force`（脚本是非破坏式的，会先把类目录移到 `data/imagenet/_raw/train/`）。

---

## 2. 公共设置：从 released ckpt 微调

两个 arm 使用**同一个起点**：released 非 cfg α-Flow B/2 权重（EMA）。

```bash
cd ~/qiangchen/AFP
python scripts/make_finetune_snapshot.py \
  snapshots/alphaflow-B-2.pt snapshots/alphaflow-B-2-finetune-init.pt
```

公共 overrides（两个 arm 完全一致）：

| override | 值 | 原因 |
| --- | --- | --- |
| `training.resume.on_start_ckpt_path` | `$PWD/snapshots/alphaflow-B-2-finetune-init.pt` | 同起点 |
| `training.resume.whole_state` | `false` | 只加载权重，不恢复 optimizer/stats |
| `training.resume.allow_missing_keys` | `true`（仅 state arm 需要） | state 头不在 released ckpt 里 |
| `model.ema_rampup_ratio` | `null` | 微调时关闭 EMA rampup（仓库推荐） |
| `training.freqs.metrics` / `metrics_rare` | `null` | **不跑 FID**（否则 step 1 就会触发 metrics） |
| `training.freqs.loss_val` / `loss_per_sigma` | `null` | 省时间 |
| `training.freqs.snapshot` | `null` | 只在 `done` 时保存最终 snapshot（自动） |
| `training.max_steps` | 50（测速）/ 1000（sanity） | |
| `training.steps_per_tick` | 5（测速）/ 10（sanity） | 日志密度 |

**短 α 调度（本阶段专用）**：官方 α 调度按 400k 步设计，1000 步内 α 恒为 1（只覆盖
α=1 skip 分支）。为了让本阶段覆盖 `0<α<1` 与 `α=0` JVP 分支，两个 arm 都使用同一个短调度
（改动对两 arm 相同，不影响对照公平性）：

```
loss.alpha={scheduler: sigmoid, initial_value: 1.0, end_value: 0,
            change_init_steps: 0, change_end_steps: 1200, clamp_value: 0.005, gamma: 25.0}
```

作用：1000 步内 α 从 1 降到 0（约 850 步后 clamp 到 0），d 分支、α=1 skip、α=0 JVP 全覆盖。
**正式 Stage 2 实验仍使用官方调度**，此短调度只用于本次管线验证。

其余保持 preset 默认：`alphaflow-sigmoid-latentspace-B-2-single-gpu`
（B/2 latent、SD-VAE、batch_gpu 32 / global batch 256、lr 1e-4、Adam、4 亿步公式中的
`max_steps` 被覆盖）。

---

## 3. 第一步：50 步测速（两条命令，约 2–5 分钟/条）

目标：确认数据管线、显存、吞吐、日志、checkpoint 能用。**不改任何模型/损失设置。**

```bash
cd ~/qiangchen/AFP
export HF_ENDPOINT=https://hf-mirror.com

INIT=$PWD/snapshots/alphaflow-B-2-finetune-init.pt

# ---- A. Corrected Velocity（50 步） ----
bash scripts/train_single_gpu.sh alphaflow-sigmoid-latentspace-B-2-single-gpu \
  desc=stage1-latent-speed-velocity \
  training.resume.on_start_ckpt_path="$INIT" \
  training.resume.whole_state=false \
  model.ema_rampup_ratio=null \
  training.max_steps=50 training.steps_per_tick=5 \
  training.freqs.metrics=null training.freqs.metrics_rare=null \
  training.freqs.metrics_final=[] training.freqs.loss_val=null training.freqs.loss_per_sigma=null \
  training.freqs.snapshot=null training.freqs.snapshot_latest=null training.freqs.save_samples=null \
  loss.alpha.scheduler=sigmoid loss.alpha.initial_value=1.0 loss.alpha.end_value=0 \
  loss.alpha.change_init_steps=0 loss.alpha.change_end_steps=1200 \
  loss.alpha.clamp_value=0.005 loss.alpha.gamma=25.0

# ---- B. Corrected State（50 步，唯一差异 = parameterization + allow_missing_keys） ----
bash scripts/train_single_gpu.sh alphaflow-sigmoid-latentspace-B-2-single-gpu \
  desc=stage1-latent-speed-state \
  model.parameterization=state \
  training.resume.on_start_ckpt_path="$INIT" \
  training.resume.whole_state=false \
  training.resume.allow_missing_keys=true \
  model.ema_rampup_ratio=null \
  training.max_steps=50 training.steps_per_tick=5 \
  training.freqs.metrics=null training.freqs.metrics_rare=null \
  training.freqs.metrics_final=[] training.freqs.loss_val=null training.freqs.loss_per_sigma=null \
  training.freqs.snapshot=null training.freqs.snapshot_latest=null training.freqs.save_samples=null \
  loss.alpha.scheduler=sigmoid loss.alpha.initial_value=1.0 loss.alpha.end_value=0 \
  loss.alpha.change_init_steps=0 loss.alpha.change_end_steps=1200 \
  loss.alpha.clamp_value=0.005 loss.alpha.gamma=25.0
```

采样观察（每 tick 一行，`steps_per_tick=5` → 50 步约 10 行）：

```bash
grep 'step ' runs/stage1-latent-speed-velocity/log.txt | tail -n 12
grep 'step ' runs/stage1-latent-speed-state/log.txt    | tail -n 12
```

记录并回传：

| 指标 | 期望 |
| --- | --- |
| 是否跑通 / 无异常栈 | 必须 |
| `iter/sec`（tick 行） | 记录；B/2 latent 参考量级 **1–5 it/s**（每 optimizer step 含 8 次梯度累积） |
| `gpumem` / `reserved`（tick 行） | 记录；必须 < 31.4GB，建议 < 28GB |
| `loss_total`、`loss_trajectory_FM`、`loss_trajectory_consistency`、`grad_norm` | 有限、量级稳定 |
| `grad_net_num_nans/infs` | 必须为 0 |

**若 OOM**：`dataset.batch_gpu=16`（global batch 仍 256 → 16 次累积），其余不变；
再不行 `batch_gpu=8`。**若数据吞吐是瓶颈**（`samples/sec` 远低于 GPU 吞吐）：`dataset.num_workers=12`。

---

## 4. 第二步：1k 步 sanity（两条命令，合计约 1–4 小时，取决于测速）

在 50 步通过后执行；两个 arm 除 parameterization 外完全一致：

```bash
cd ~/qiangchen/AFP
INIT=$PWD/snapshots/alphaflow-B-2-finetune-init.pt

# A. Corrected Velocity
bash scripts/train_single_gpu.sh alphaflow-sigmoid-latentspace-B-2-single-gpu \
  desc=stage1-latent-1k-velocity \
  training.resume.on_start_ckpt_path="$INIT" training.resume.whole_state=false \
  model.ema_rampup_ratio=null \
  training.max_steps=1000 training.steps_per_tick=10 \
  training.freqs.metrics=null training.freqs.metrics_rare=null \
  training.freqs.metrics_final=[] training.freqs.loss_val=null training.freqs.loss_per_sigma=null \
  training.freqs.snapshot=null training.freqs.snapshot_latest=null \
  training.freqs.save_samples=1000 \
  loss.alpha.scheduler=sigmoid loss.alpha.initial_value=1.0 loss.alpha.end_value=0 \
  loss.alpha.change_init_steps=0 loss.alpha.change_end_steps=1200 \
  loss.alpha.clamp_value=0.005 loss.alpha.gamma=25.0

# B. Corrected State（唯一差异）
bash scripts/train_single_gpu.sh alphaflow-sigmoid-latentspace-B-2-single-gpu \
  desc=stage1-latent-1k-state \
  model.parameterization=state \
  training.resume.on_start_ckpt_path="$INIT" training.resume.whole_state=false \
  training.resume.allow_missing_keys=true \
  model.ema_rampup_ratio=null \
  training.max_steps=1000 training.steps_per_tick=10 \
  training.freqs.metrics=null training.freqs.metrics_rare=null \
  training.freqs.metrics_final=[] training.freqs.loss_val=null training.freqs.loss_per_sigma=null \
  training.freqs.snapshot=null training.freqs.snapshot_latest=null \
  training.freqs.save_samples=1000 \
  loss.alpha.scheduler=sigmoid loss.alpha.initial_value=1.0 loss.alpha.end_value=0 \
  loss.alpha.change_init_steps=0 loss.alpha.change_end_steps=1200 \
  loss.alpha.clamp_value=0.005 loss.alpha.gamma=25.0
```

说明：

- `save_samples=1000`：只在 `done` 时生成 8 张图（`runs/<desc>/fakes_00001000_*.png`），
  用于确认 state 采样器在真实 latent 下能出图；不跑任何 metric。
- `done` 时会自动保存最终 snapshot（`runs/<desc>/snapshot-*.pt`，含 net/ema/optimizer，
  约 2–3GB/个）；这就是 state 版 checkpoint 往返测试的对象。
- 两个 arm 的日志用 Part II 工具比较（不要求曲线一致）：

```bash
python scripts/summarize_stage1_sanity.py \
  --velocity-dir runs/stage1-latent-1k-velocity \
  --state-dir runs/stage1-latent-1k-state --expected-steps 1000
```

- 抽查生成图：

```bash
python - <<'PY'
import glob, numpy as np
from PIL import Image
for d in ['stage1-latent-1k-velocity', 'stage1-latent-1k-state']:
    fs = sorted(glob.glob(f'runs/{d}/fakes_*.png'))
    print(d, len(fs))
    for f in fs[:4]:
        a = np.array(Image.open(f).convert('RGB'))
        print(' ', f.split('/')[-1], a.shape, 'mean=%.1f std=%.1f' % (a.mean(), a.std()))
PY
```

---

## 5. 第三步（可选但推荐）：state snapshot 往返 + 4 张 latent 生成

验证 dual-head checkpoint 在真实 generate 流程（不是训练内 save_samples）中可加载、可出图：

```bash
cd ~/qiangchen/AFP
for arm in velocity state; do
  CKPT=$(ls runs/stage1-latent-1k-$arm/snapshot-*.pt | tail -1)
  echo "== $arm: $CKPT"
  CKPT="$CKPT" SEEDS=0-3 NFE=1 BATCH_SIZE=2 \
    bash scripts/generate_single_gpu.sh || break
done
# 期望输出:
#   runs/stage1-latent-1k-velocity/generate/nfe1/sample_00000{0..3}.png
#   runs/stage1-latent-1k-state/generate/nfe1/sample_00000{0..3}.png
```

验收：两个 arm 都能加载、生成 4 张 256×256 图，图像 mean/std 正常（非黑/白）。

---

## 6. 工程建议（可选，减少命令出错）

为减少长命令行出错，建议在 `infra/experiments/experiments-alphaflow.yaml` 增加一个**临时**
验证 preset（待批准后再加，不影响正式实验）：

```yaml
stage1-latent-validation-common:
  _inherit_: alphaflow-sigmoid-latentspace-B-2-single-gpu
  overrides:
    training:
      max_steps: 1000
      steps_per_tick: 10
      freqs:
        metrics: null
        metrics_rare: null
        metrics_final: []
        loss_val: null
        loss_per_sigma: null
        snapshot: null
        snapshot_latest: null
        save_samples: 1000
    model:
      ema_rampup_ratio: ~
    loss:
      alpha: {scheduler: sigmoid, initial_value: 1.0, end_value: 0,
              change_init_steps: 0, change_end_steps: 1200,
              clamp_value: 0.005, gamma: 25.0}
```

然后只需 `bash scripts/train_single_gpu.sh <preset> desc=... training.resume...`。
（若你同意，我可以在下一轮把 preset 加进去，本轮先按命令行 override 执行。）

---

## 7. 本阶段不做的事

- 不跑 FID / FDD / FCD；不根据生成质量下结论；
- 不启动 20k 步或更长训练；
- 不引入 EMA Teacher / Nested Consistency；
- 不对 α 调度、CFG、clamp、采样参数做任何额外改动（短 α 调度仅用于本验证，两 arm 相同）。
