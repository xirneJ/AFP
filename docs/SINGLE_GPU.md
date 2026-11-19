# AlphaFlow 单卡（RTX 5090 / Blackwell）运行手册

> 目标：在这台 **1×RTX 5090（32 GiB）**、`venv` 环境的 Ubuntu 服务器上，
> 拉取本仓库后能完成 **推理 → 训练 smoke test → 真实训练 → 评测** 的完整流程。
> 本仓库原始代码是按 **8×H100 + conda + 多卡 FSDP/DDP** 写的，本手册对应的改动
> 让它在单卡上也能跑；多卡流程不受影响。

---

## 0. 先跑体检（最重要，1 分钟）

```bash
cd ~/qiangchen/AFP
python scripts/check_setup.py
```

它会检查并直接告诉你：

| 检查项 | 说明 |
|---|---|
| PyTorch / CUDA | 5090 是 sm_120，**必须** CUDA 12.8+ 的 torch，否则会报 `no kernel image is available` 或静默回退 CPU |
| 当前空闲显存 | 现在别人的 4 个任务占了 ~30/32 GiB，脚本会给出“能推理 / 能训练 / 都不能”的结论 |
| Python 依赖 | 缺哪个包、是否要装 |
| ImageNet / dummy 数据 | `data/imagenet` 是否为仓库要求的 `folder + .meta.json` 格式 |
| 权重文件 | `snapshots/` 里有没有 checkpoint、缺哪个 |
| HF 缓存 | LDM（latent-space）模型需要 `stabilityai/sd-vae-ft-ema` |

**结论会写在最后一段 “5. Recommendation”。** 如果它说 free VRAM < 3 GiB，
那只能做单张推理，训练要等显卡空出来。

---

## 1. 安装环境（Blackwell 专用）

原 `environment.yml` 是 conda + torch 2.6/cu118，**在 5090 上跑不了**。推荐用 venv：

```bash
cd ~/qiangchen/AFP
bash scripts/setup_env.sh          # 默认创建 ./venv，装 torch 2.7.1+cu128 和依赖
source venv/bin/activate
python scripts/check_setup.py
```

如果系统里没有 Python 3.11：

```bash
PYTHON=python3.12 bash scripts/setup_env.sh   # 能用但 numpy/scipy 版本会变，FID128 可能有偏差
```

也可以手动装（等价）：

```bash
python -m venv venv && source venv/bin/activate
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-blackwell.txt
```

> 你的 venv 放在哪里都可以：单卡入口脚本用的是**当前已激活的 `python`**，
> 只要 `source` 过对应 venv 就行。`configs/env/user.yaml` 里的 `project_path`
> 由 `ALPHAFLOW_PROJECT_PATH` 环境变量提供，入口脚本会自动设成仓库根目录，
> 所以不需要手改路径。若要把 data/snapshots/output 放到别的盘，先
> `export ALPHAFLOW_PROJECT_PATH=/your/path` 再运行。

---

## 2. 下载预训练权重（推理用）

把两个文件放到 `./snapshots/`：

```bash
mkdir -p snapshots
# α-Flow-B/2（带 CFG，推荐先跑这个：NFE-1 FID 5.40）
curl -L -o snapshots/alphaflow-B-2-cfg.pt \
  "https://huggingface.co/snap-research/alphaflow/resolve/main/alphaflow-B-2-cfg.pt?download=true"
# SD-VAE（latent-space 模型出图必须）
curl -L -o snapshots/sd_vae_ft_ema.pt \
  "https://huggingface.co/snap-research/alphaflow/resolve/main/sd_vae_ft_ema.pt?download=true"

# （可选）做对照实验时再下这两个：
# curl -L -o snapshots/alphaflow-B-2.pt \
#   "https://huggingface.co/snap-research/alphaflow/resolve/main/alphaflow-B-2.pt?download=true"
# curl -L -o snapshots/meanflow-B-2-cfg.pt \
#   "https://huggingface.co/snap-research/alphaflow/resolve/main/meanflow-B-2-cfg.pt?download=true"
```

> 训练也需要 `sd_vae_ft_ema.pt`（模型初始化时加载 VAE），所以这一步不能跳过。
> 第一次运行还会从 HuggingFace 拉 `stabilityai/sd-vae-ft-ema` 的配置/权重作为
> diffusers 初始化，之后会缓存在 `./.cache/huggingface`。

---

## 3. 推理 / 生成（现在就能做，~2–3 GiB 显存）

```bash
# NFE=1，生成 8 张（seed 0-7），batch=1，显存最省
bash scripts/generate_single_gpu.sh

# 2-NFE + 更多张 + 稍微大点 batch
NFE=2 SEEDS=0-31 BATCH_SIZE=2 bash scripts/generate_single_gpu.sh

# 显存非常紧（比如只剩 2 GiB）：把 DiT 主干转 bf16
DIT_DTYPE=bfloat16 bash scripts/generate_single_gpu.sh
```

输出在 `data/generate/alphaflow-B-2-cfg-NFE1/`。

常见问题：

* **OOM**：先 `DIT_DTYPE=bfloat16 BATCH_SIZE=1`；还是不行就等显存。
  脚本会打印 `Peak GPU memory ...`，可以据此判断实际占用。
* **找不到 sd_vae**：把 `sd_vae_ft_ema.pt` 放到 `snapshots/`；加载时会自动把
  checkpoint 里记录的旧机器路径重定向到本地 `./snapshots/`。
* **算子不支持 / 全黑图**：几乎一定是 torch 不是 cu128 版本，回第 1 步。

---

## 4. 训练 smoke test（不需要 ImageNet，1–2 分钟，验证整条训练链路）

```bash
python scripts/make_dummy_dataset.py       # 生成 data/dummy 合成小数据集（64×64, 8 类）
bash scripts/train_single_gpu.sh alphaflow-smoke-test-single-gpu
```

预期：打印模型结构 / 参数量 / batch 设置，训练 30 步，最后保存一个
`output/alphaflow-smoke-test-single-gpu/snapshot-00000030.pt`（单卡 `direct_launch`
模式下每个实验有独立输出目录 `output/<实验名>/`）。
显存占用约 1–2 GiB，这也能让你先测出**你这块卡上每步大概多少秒**。

latent-space（VAE + DiT）链路的 smoke test（需要 sd_vae 权重）：

```bash
bash scripts/train_single_gpu.sh alphaflow-smoke-test-ldm-single-gpu
```

> 想从头重跑 smoke test：先删掉对应的输出目录，例如
> `rm -rf output/alphaflow-smoke-test-single-gpu`（否则会自动从上次的
> checkpoint 续跑，直接“秒结束”）。

---

## 5. 真实训练（ImageNet-1K，需要整卡/≥20 GiB 空闲）

### 5.1 准备数据

仓库要求的数据格式是**扁平的图片目录 + 每张图一个 `.meta.json`**（字段 `class` 为 0..999）：

```
data/imagenet/
  train/ n01440764_10026.jpeg  n01440764_10026.meta.json ...
  val/   ...
```

如果你已经有原始 ImageNet（tar 包或解压好的 `train/`、`val/` 类别子目录）：

```bash
# 检测 + 非破坏性重组（默认用软链接，不搬动/删除你的原始数据）
python scripts/data_scripts/prepare_imagenet.py \
  --target_directory ./data/imagenet --skip-download
```

如果服务器上还没有 ImageNet：

```bash
# 注意：image-net.org 需要登录，脚本会尝试直接下载；失败时按提示手动下载两个 tar
# 放到 ./data/imagenet/ 下，再执行上面的 --skip-download 命令。
python scripts/data_scripts/prepare_imagenet.py --target_directory ./data/imagenet
```

`check_setup.py` 会检查它是否准备好。

### 5.2 启动训练（三种对照实验已内置）

```bash
# α-Flow（本文方法，sigmoid alpha 调度）
bash scripts/train_single_gpu.sh alphaflow-sigmoid-latentspace-B-2-single-gpu

# MeanFlow 对照
bash scripts/train_single_gpu.sh alphaflow-meanflow-latentspace-B-2-single-gpu

# 普通 Flow Matching 对照
bash scripts/train_single_gpu.sh alphaflow-latentspace-B-2-single-gpu
```

默认配置：**1 卡、B/2、256×256 latent-space、全局 batch 256（batch_gpu=32 × 8 次梯度累积）、
400k steps、推理端 NFE-1、每 25k 步算一次 FID-2048、W&B 关闭**。
latent 由 VAE 在每个 micro-batch 上即时编码（不需要事先提取全量 latent）。

按显存/时间调整（都可以直接追加在命令行）：

```bash
# 显存不够：batch_gpu 减半，并把 batch_size 也减半（保持 8 次累积）
bash scripts/train_single_gpu.sh alphaflow-sigmoid-latentspace-B-2-single-gpu \
  dataset.batch_gpu=16 dataset.batch_size=128
# 先短跑验证：5k 步
bash scripts/train_single_gpu.sh alphaflow-sigmoid-latentspace-B-2-single-gpu \
  training.max_steps=5000 training.freqs.metrics=null
# 想开 W&B：删掉脚本里强制关闭的 wandb.enabled=false，或直接
#   python infra/launch.py <exp> num_gpus=1 direct_launch=true wandb.enabled=true
```

训练输出在 `output/<实验名>/`（checkpoint、sample 图、日志）。**断点续训**：直接重新执行
同一条命令即可，程序会自动从该目录里最新的 `snapshot-*.pt` 恢复。

### 5.3 显存 / 速度参考

| 配置 | 训练显存（估） | 说明 |
|---|---|---|
| B/2 latent，batch_gpu=32 | ~10–16 GiB | 默认，需要整卡 |
| B/2 latent，batch_gpu=16 | ~7–10 GiB | 显存紧张时 |
| B/2 latent，batch_gpu=8 | ~5–7 GiB | 更慢 |
| 2–3 GiB 空闲 | **不可训练** | 只能推理，等显卡空出来 |

> 32 GiB 卡上 B/2 预计 ~20–40k steps / 小时（还取决于 CPU/数据读取），400k 步大约 1–2 天。

---

## 6. 评测（生成 + FID）

训练过程中会自动算 FID；也可以单独评测：

```bash
# 1) 先按第 3 步用训练好的 checkpoint 生成 50k 张（分批跑，可以多次续跑）
CKPT=output/alphaflow-sigmoid-latentspace-B-2-single-gpu/snapshot-0400000.pt \
OUT=data/generate/my-B-2 SEEDS=0-49999 BATCH_SIZE=4 \
  bash scripts/generate_single_gpu.sh

# 2) 对生成目录算 FID / FDD / FCD
python scripts/evaluate.py \
  dataset.resolution=[1,256,256] \
  gen_dataset.src=data/generate/my-B-2/ gen_dataset.resolution=[1,256,256] gen_dataset.label_shape=[] \
  metrics=[fid50k_full]
```

第一次算 FID 需要先统计真实数据特征，会缓存到 `data/stats/`，之后就是增量加载。

---

## 7. 做“对照实验”时改哪里

你要修改模型/损失做对比，主要看这几个文件：

| 位置 | 内容 |
|---|---|
| `configs/loss/alphaflow.yaml` | α-Flow 的核心超参：`alpha` 调度（sigmoid/linear/constant）、`ratio_fm`、`cfg_params(omega/kappa/t_min/t_max)`、时间采样分布 |
| `src/training/loss.py` 的 `AlphaFlowLoss`（约 361 行起） | α-Flow 损失实现：`sample_traj_params`、`get_ratio`、`_compute_velocity_cfg`、`_compute_mean_velocity_c` |
| `configs/sampling/recflow.yaml` | 推理端采样：`num_steps`、`sigma_noise`、`cfg.scale`、`time_shifting` |
| `src/inference.py` | 单步/多步采样实现（`reverse_recflow_step`） |
| `infra/experiments/experiments-alphaflow.yaml` | 实验注册表：新增一条 `xxx-single-gpu` 实验 = 一组消融 |

最小工作流：复制一条现有 `*-single-gpu` 实验 → 改 `loss:` 覆盖项或直接改
`AlphaFlowLoss` → `bash scripts/train_single_gpu.sh 你的实验名` → 用第 6 步评测。
因为三种 baseline（FM / MeanFlow / α-Flow）用的是**同一个** `alphaflow-latentspace-B-2-single-gpu`
父实验、同样的数据/batch/步数，所以对比结果是可控的。

---

## 8. FAQ

**Q: 为什么原来 README 的命令要 8 卡？**
A: 原始实验在 8×H100 上用 `torchrun` + FSDP/DDP，batch 256、400k~1.2M 步。
单卡代码路径已经保留，只是把并行策略设为 `dp.strategy: null`、用 `direct_launch`。

**Q: `direct_launch=true` 是什么意思？**
A: 不创建 `experiments/<id>-...` 目录、不拼 `torchrun` 命令，直接在当前仓库里调用
`train(cfg)`；输出统一写到 `./output`。单卡实验都在注册表里设置好了。

**Q: 报 `no kernel image is available for execution on the device`**
A: torch 版本不对（不是 cu128）。重装第 1 步。

**Q: 报 HF 下载失败（`stabilityai/sd-vae-ft-ema`）**
A: 需要联网或预先把模型放进 `$HF_HOME`（默认 `./.cache/huggingface`）。
纯离线环境：在能联网的机器上 `huggingface-cli download stabilityai/sd-vae-ft-ema`，
把整个目录拷到 `./.cache/huggingface/hub/` 下。

**Q: 只想先看效果，不想训练？**
A: 直接第 2、3 步，1 张 256×256 图，NFE-1，几秒钟。

**Q: 现在只有 2–3 GiB 空闲显存，能训练吗？**
A: 不能。单卡 B/2 训练最低约 5–7 GiB（batch_gpu=8），且需要等别人的任务结束。
推理可以做到 batch=1（见第 3 步的 `DIT_DTYPE=bfloat16`）。
