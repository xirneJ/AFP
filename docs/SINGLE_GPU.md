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
| 当前空闲显存 | 脚本会按当前 free VRAM 给出“只能推理 / 可以训练”的结论（单卡 B/2 训练通常需要 ≥10–16 GiB 空闲） |
| Python 依赖 | 缺哪个包、是否要装 |
| ImageNet / dummy 数据 | `data/imagenet` 是否为仓库要求的 `folder + .meta.json` 格式 |
| 权重文件 | `snapshots/` 里有没有 checkpoint、缺哪个 |
| HF 缓存 | LDM（latent-space）模型需要 `stabilityai/sd-vae-ft-ema` |

**结论会写在最后一段 “5. Recommendation”。** 如果它说 free VRAM < 8 GiB，
那只适合推理；训练建议等显存更宽裕时再跑。

---

## 0.5 这些单卡配置 vs. 原版 α-Flow：改了什么、没改什么

**没有改的东西（算法/模型本体）**：
`configs/loss/alphaflow.yaml`、`src/training/loss.py` 里的 `AlphaFlowLoss`、
`configs/model/*` 的 DiT/SiT 结构、τ 采样与 α 调度公式、`src/inference.py` 的采样器——
全部保持原样；官方 8 卡实验（`experiments-alphaflow.yaml` 里原有的那些条目）也一行没动。

**为了单卡能跑，改了的东西**：

| 项目 | 官方 8 卡 | 本单卡预设 | 是否影响算法/可比性 |
|---|---|---|---|
| 并行 | `torchrun` + FSDP/DDP | 1 卡、`dp.strategy=null`、`direct_launch` | 不影响数学 |
| latent 提取 | 官方 B/2 用**预提取** latent 加快训练 | **即时 VAE 编码**（数学等价，只是每步更慢） | 不影响（预提取路径也保留） |
| 全局 batch | 256（batch_gpu 32 × 8 卡） | 256（batch_gpu 32 × 8 次梯度累积 / 1 卡） | 一致 |
| 训练步数 | 400k（非 cfg）/ 1.2M（cfg） | 相同 | 一致 |
| 训练中评测 | joint_image_fd512_full / fd50k_full 等 | **FID-2048**，去掉 NFE-2 一致性评测 | ⚠️ 论文数字不可直接横比；你的各变体之间可比 |
| W&B / 画图频率 | 开 | 默认关、频率降低 | 不影响 |

所以注册表里的这些条目**是原版 B/2 配方在单卡上的“同协议运行”**，不是把模型改小：

- `alphaflow-*-latentspace-B-2-single-gpu`（FM / MeanFlow / α-Flow）：**12 blocks, dim 768, 12 heads,
  lr 1e-4, batch 256, 400k 步，α 调度与原版一致**。单卡跑完 400k 步约 **2–4 天/个**。
- `alphaflow-*-latentspace-B-2-cfg-single-gpu`：**官方 cfg 训练配方**（label_dropout=0.1,
  kappa=0.5, 1.2M 步），更接近论文主表/`*-cfg` 权重；单卡跑完约 **1–2 周/个**。
  算力紧时可用 `training.max_steps=100000` 做同预算对比。
- `alphaflow-smoke-test-*`：**只有这两个是“缩小版”**——6 blocks/dim 256（像素空间）或
  2 blocks/dim 128（LDM），64/128 分辨率，几十步，纯粹用来 1 分钟验证数据/损失/优化器/
  checkpoint 链路是否通，**不代表 α-Flow 效果**。

如果要做“改模型 vs 原版”的对照：建议 3 个变体都用
`alphaflow-*-latentspace-B-2-single-gpu`（同样的 400k 步预算）或全部用
`-cfg-single-gpu` + 相同的更短步数，保证预算一致即可。

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
> 所以不需要手改路径。若要把 data/snapshots 放到别的盘，先
> `export ALPHAFLOW_PROJECT_PATH=/your/path` 再运行。

---

## 1.5 所有实验产物统一放在 `runs/` 下

`configs/env/user.yaml` 里的 `experiments_root`（默认 `<仓库根目录>/runs`）是**唯一**的
实验产物目录，所有脚本都会写到这里，不会散落在别处：

```
runs/
├── <run_name>/                     # 一次训练 = 一个 run（run_name 由 desc=... 指定）
│   ├── log.txt                     # 训练日志（自动写入）
│   ├── snapshot-XXXXXXXX.pt        # 8 位步数的 checkpoint（断点续训也在这里找）
│   ├── metric-*.jsonl              # 训练中 FID 等指标历史
│   ├── fakes_*.png / real_*.png    # 训练中可视化样本
│   ├── generate/
│   │   └── nfe1/ nfe2/             # 从该 run 的 checkpoint 生成的 50k 图片
│   └── metrics/
│       ├── eval-nfe1.json          # 最终评测汇总（FID/FDD/FCD）
│       └── metric-fid50k_full.jsonl
├── _released/
│   └── <ckpt名>/
│       ├── nfe1/ nfe2/             # 官方发布权重的生成结果
│       └── metrics/                # 对应的评测结果
└── _dataset_stats/                 # FID 真实数据特征缓存（所有 run 共享，首次生成后复用）
```

- 想改存放位置：改 `configs/env/user.yaml` 的 `experiments_root` 一行即可
  （生成脚本默认仍用 `<仓库>/runs`，改过目录后生成时显式传 `OUT=<新目录>/...`）。
- 所有路径都在 `.gitignore` 里，不会被误提交。

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

输出在 `runs/_released/alphaflow-B-2-cfg/nfe1/`（官方权重的生成结果统一放 `_released/`）。

常见问题：

* **OOM**：先 `DIT_DTYPE=bfloat16 BATCH_SIZE=1`；还是不行就等占显存的进程结束。
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
`runs/alphaflow-smoke-test-single-gpu/snapshot-00000030.pt`（单卡 `direct_launch`
模式下每个实验有独立目录 `runs/<desc>/`，`desc` 默认等于实验名）。
显存占用约 1–2 GiB，这也能让你先测出**你这块卡上每步大概多少秒**。

latent-space（VAE + DiT）链路的 smoke test（需要 sd_vae 权重）：

```bash
bash scripts/train_single_gpu.sh alphaflow-smoke-test-ldm-single-gpu
```

> 想从头重跑 smoke test：先删掉对应的输出目录，例如
> `rm -rf runs/alphaflow-smoke-test-single-gpu`（否则会自动从上次的
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

以上是**非 cfg** 配方（原版 400k 步、batch 256）。如果你要对标论文主表/`*-cfg` 权重，
用官方 cfg 训练配方的单卡版（1.2M 步，batch 256，batch_gpu=16 累积 16 次）：

```bash
bash scripts/train_single_gpu.sh alphaflow-sigmoid-latentspace-B-2-cfg-single-gpu    # α-Flow cfg
bash scripts/train_single_gpu.sh alphaflow-meanflow-latentspace-B-2-cfg-single-gpu    # MeanFlow cfg
# 预算紧：加 training.max_steps=100000，三个变体用同样的步数即可公平对比
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

训练输出在 `runs/<desc>/`（`log.txt`、checkpoint、sample 图、训练中指标）。
**断点续训**：直接重新执行同一条命令即可（同一个 `desc`），程序会自动从该目录里最新的
`snapshot-*.pt` 恢复。用不同的 `desc=` 可以得到互不干扰的多次运行。

### 5.3 显存 / 速度参考

| 配置 | 训练显存（估） | 说明 |
|---|---|---|
| B/2 latent，batch_gpu=32 | ~10–16 GiB | 默认，需要 ≥16 GiB 空闲 |
| B/2 latent，batch_gpu=16 | ~7–10 GiB | 显存紧张时（cfg 预设默认用它） |
| B/2 latent，batch_gpu=8 | ~5–7 GiB | 更慢 |
| < 8 GiB 空闲 | 不建议训练 | 先跑推理 |

> 单卡 5090 上 B/2、batch 256（8 次累积）预计 **~5–15k optimizer steps / 小时**
> （受数据读取和同卡其他进程影响）。400k 步约 **2–4 天**；cfg 配方 1.2M 步约 **1–2 周**。
> 做算法迭代时，建议先用 `training.max_steps=20000` 左右短跑看相对趋势。

---

## 6. 评测（生成 + FID / FDD / FCD）

以“训练好的 checkpoint → 生成 50k → 算指标”为例。checkpoint 文件名是 **8 位步数**，
例如第 400000 步是 `snapshot-00400000.pt`（`ls runs/<desc>/` 看实际名字）。

```bash
# 1) 生成 50k 张。只传 CKPT 即可：脚本会按规矩写进
#    runs/<desc>/generate/nfe<NFE>/（官方权重则写 runs/_released/...）
#    按 5000 一组分批，中断了只重跑对应分段；文件名按 seed，不会冲突。
CKPT=runs/ft-alphaflow-base/snapshot-00020000.pt
for s in 0 5000 10000 15000 20000 25000 30000 35000 40000 45000; do
  e=$((s + 4999))
  CKPT="$CKPT" SEEDS="$s-$e" NFE=1 BATCH_SIZE=8 \
    bash scripts/generate_single_gpu.sh || break
done

# 2) 算指标（结果自动保存到 runs/<desc>/metrics/）
bash scripts/evaluate_single_gpu.sh runs/ft-alphaflow-base/generate/nfe1
```

`evaluate_single_gpu.sh` 默认：
- `metrics=fid50k_full,dinofid50k_full,fcd50k_full`（可用 `METRICS=` 覆盖，建议先只跑 FID）；
- 用 **ImageNet val 50k** 作为真实分布（标准 FID，`VAL_SPLIT=1` 默认）；
- 把汇总写到 **生成目录所属 run 的 `metrics/`**：
  `runs/<desc>/metrics/eval-nfe1.json`（官方权重则是 `runs/_released/<ckpt>/metrics/...`），
  并追加 `metric-*.jsonl`；
- 第一次运行会下载 detector 权重（Inception / DINOv2 / CLIP）并把**真实数据特征**缓存到
  `runs/_dataset_stats/`，之后的变体评测直接复用，只算生成数据的特征。
- `fcd50k_full` 需要先 `pip install git+https://github.com/openai/CLIP.git`。

> 只有 val 数据时：`DATA_SRC=$PWD/data/imagenet/val/ bash scripts/evaluate_single_gpu.sh ...`
> （训练集目录不存在时用它顶替真实数据来源）。
>
> 想和论文 `evaluate.sh` 的默认协议完全一致（真实分布用 1.28M 训练图，首次统计要几小时）：
> `VAL_SPLIT=0 bash scripts/evaluate_single_gpu.sh <gen_dir>`。
> **公平对比**：所有变体用同一个 step、同一 NFE、同一套采样参数
> （默认 `sampling.sigma_noise=1 sampling.cfg.scale=1.0 sampling.enable_trajectory_sampling=true`），
> 生成时 seed 范围一致（0-49999）。NFE-2 把 `NFE=2` 再跑一遍即可（输出到 `nfe2/`）。

> 磁盘：50k 张 256×256 PNG 约 10–15 GiB/个目录，先 `df -h ~` 确认空间。
> 训练过程中的 FID-2048（写在 `runs/<desc>/metric-*.jsonl`）只看趋势，
> 最终对照用这里的 50k FID。

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
`train(cfg)`；所有实验产物统一写到 `runs/<desc>/`。单卡实验都在注册表里设置好了。

**Q: 实验产物都存哪？**
A: 全部在 `configs/env/user.yaml` 的 `experiments_root`（默认 `<仓库>/runs`）下：
`runs/<desc>/`（训练/checkpoint/日志/训练中指标）、`runs/<desc>/generate/`（50k 生成图）、
`runs/<desc>/metrics/`（最终 FID 结果）、`runs/_dataset_stats/`（真实数据特征缓存）。

**Q: 报 `no kernel image is available for execution on the device`**
A: torch 版本不对（不是 cu128）。重装第 1 步。

**Q: 报 HF 下载失败（`stabilityai/sd-vae-ft-ema`）**
A: 需要联网或预先把模型放进 `$HF_HOME`（默认 `./.cache/huggingface`）。
纯离线环境：在能联网的机器上 `huggingface-cli download stabilityai/sd-vae-ft-ema`，
把整个目录拷到 `./.cache/huggingface/hub/` 下。

**Q: 只想先看效果，不想训练？**
A: 直接第 2、3 步，1 张 256×256 图，NFE-1，几秒钟。

**Q: 现在只有 2–3 GiB 空闲显存，能训练吗？**
A: 不能。单卡 B/2 训练最低约 5–7 GiB（batch_gpu=8）；一般需要 ≥16 GiB 空闲才能用默认
batch。推理可以做到 batch=1（见第 3 步的 `DIT_DTYPE=bfloat16`）。
