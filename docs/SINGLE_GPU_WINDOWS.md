# AlphaFlow 在 Windows 笔记本上运行（RTX 5060 Laptop / Blackwell）

> 适用：Windows 10/11 + NVIDIA RTX 50 系笔记本显卡（sm_120）+ Python 3.12。
> 本手册针对“**用自己电脑跑推理 / smoke test / 小规模对照实验**”，
> 完整的 ImageNet-1K 训练仍然建议在服务器上做（原因见第 1 节）。

---

## 1. 你这台机器能做什么

实测配置：**RTX 5060 Laptop，8 GiB 显存（当前桌面占用约 3.3 GiB，空闲约 4.6 GiB）**，
sm_120（Blackwell，必须 CUDA 12.8+ 的 PyTorch），Python 3.12，E 盘剩余 ~87 GiB。

| 任务 | 本地可行性 | 说明 |
|---|---|---|
| 单张/少量推理（α-Flow-B/2-cfg，256×256，NFE-1） | ✅ | batch=1 峰值约 2–3 GiB |
| 训练链路 smoke test（合成小数据，几十步） | ✅ | 约 1.5–2 GiB，1–2 分钟 |
| B/2 latent 小规模训练（batch_gpu=4~8） | ⚠️ 勉强 | 约 6–8 GiB；只适合看 loss/流程/小数据消融，速度较慢 |
| 完整 ImageNet-1K 400k 步训练 | ❌ 不现实 | 8 GiB 显存 + 笔记本算力 + 没有本地数据；服务器更合适 |

结论：**电脑上先跑通推理 + smoke test；真正的对照实验（改模型后训练对比）建议仍在服务器**
（等服务器显卡空出来），本地可以跑小规模消融来快速验证代码。

---

## 2. 安装环境（PowerShell，一条命令）

在仓库根目录（`E:\root\alphaflow`）打开 PowerShell：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
```

它会：
1. 用 `python` 创建 `.\venv`；
2. 从**上海交大镜像**安装 `torch 2.7.1 + torchvision 0.22.1 (cu128)`（Windows 有官方
   Blackwell wheel，已验证该镜像包含 `cp312-win_amd64` 包）；
3. 安装 `requirements-blackwell.txt`；
4. 自动运行 `scripts\check_setup.py` 自检。

注意：本机 pip 配置了代理 `127.0.0.1:7897`。如果安装卡住/失败，依次尝试：

```powershell
# 忽略 pip 的全局代理配置（推荐）
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1 -Isolated
# 或改用 PyTorch 官方源（代理能访问境外时）
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1 -UseOfficialIndex
```

安装完成后：

```powershell
.\venv\Scripts\Activate.ps1
python scripts\check_setup.py
```

看到 `Blackwell support: torch built with CUDA 12.8` 和 `GPU kernel test ... succeeded` 就说明
环境正确。**如果显示 CUDA not available 或 “no kernel image”，就是 torch 装错了**，重跑第 2 步。

> 安装体积约 6–7 GiB（torch wheel 约 2.6 GiB），E 盘空间足够。

---

## 3. 下载推理权重（放 `.\snapshots\`）

国内网络直接用 HuggingFace 镜像 `hf-mirror.com`：

```powershell
New-Item -ItemType Directory -Force snapshots | Out-Null
curl.exe -L -o snapshots\alphaflow-B-2-cfg.pt "https://hf-mirror.com/snap-research/alphaflow/resolve/main/alphaflow-B-2-cfg.pt?download=true"
curl.exe -L -o snapshots\sd_vae_ft_ema.pt     "https://hf-mirror.com/snap-research/alphaflow/resolve/main/sd_vae_ft_ema.pt?download=true"
```

可选（做对照实验时再下）：

```powershell
curl.exe -L -o snapshots\meanflow-B-2-cfg.pt "https://hf-mirror.com/snap-research/alphaflow/resolve/main/meanflow-B-2-cfg.pt?download=true"
curl.exe -L -o snapshots\alphaflow-B-2.pt     "https://hf-mirror.com/snap-research/alphaflow/resolve/main/alphaflow-B-2.pt?download=true"
```

另外 latent-space 模型第一次初始化时 diffusers 会拉 `stabilityai/sd-vae-ft-ema` 的配置，
建议先设置镜像（每个新 PowerShell 窗口都要设，或写进系统环境变量）：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

---

## 4. 推理（先验证环境 + 看效果）

```powershell
powershell -ExecutionPolicy Bypass -File scripts\generate_single_gpu.ps1

# 2-NFE + 更多张 + 再省显存
$env:NFE="2"; $env:SEEDS="0-31"; $env:DIT_DTYPE="bfloat16"
powershell -ExecutionPolicy Bypass -File scripts\generate_single_gpu.ps1
```

输出在 `runs\_released\alphaflow-B-2-cfg\nfe1\`（所有实验产物统一在仓库根目录的
`runs\` 下：训练/生成/评测结果各有子目录）。脚本会打印峰值显存。

**OOM 处理顺序**：`DIT_DTYPE=bfloat16` → `BATCH_SIZE=1` → 关掉浏览器/游戏/桌面动画再试。
桌面本身约占 1–3 GiB，笔记本 8 GiB 卡要留余量。

---

## 5. 训练链路 smoke test（不需要 ImageNet，1–2 分钟）

```powershell
python scripts\make_dummy_dataset.py
powershell -ExecutionPolicy Bypass -File scripts\train_single_gpu.ps1 alphaflow-smoke-test-single-gpu
```

预期：打印模型结构/batch 设置，训练 30 步，最后在
`runs\alphaflow-smoke-test-single-gpu\snapshot-00000030.pt` 保存 checkpoint。
重跑前先删输出目录（否则会从上次 checkpoint 续跑）：

```powershell
Remove-Item -Recurse -Force runs\alphaflow-smoke-test-single-gpu
```

再测 latent-space（VAE + DiT）链路（需要第 3 步的 sd_vae 权重）：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\train_single_gpu.ps1 alphaflow-smoke-test-ldm-single-gpu
```

---

## 6. 本地小规模训练（可选）

前提：需要 ImageNet 格式数据（可以只拷一个子集，比如 5 万张，约 7–10 GiB）。
把服务器上处理好的 `data\imagenet\train|val` 目录拷到本仓库 `data\imagenet\` 下即可
（repo 要求的格式是**扁平图片 + 每张一个 `.meta.json`**）。

然后按 8 GiB 显存压小 batch：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\train_single_gpu.ps1 `
  alphaflow-sigmoid-latentspace-B-2-single-gpu `
  dataset.batch_gpu=4 dataset.batch_size=32 dataset.num_workers=2 `
  training.max_steps=2000 training.freqs.metrics=null
```

显存参考（B/2 latent，256×256）：

| batch_gpu | 训练显存（估） | 备注 |
|---|---|---|
| 8 | ~7–8 GiB | 8 GiB 卡基本到顶，建议先关桌面程序 |
| 4 | ~6–7 GiB | 本地小规模实验推荐 |
| 2 | ~5–6 GiB | 最稳 |

> 本机只适合跑 1k–10k 步看 loss 曲线 / 验证你改的模型能跑通；
`400k` 步的正式对照实验请放服务器（服务器脚本见 `docs/SINGLE_GPU.md`）。

如果想在 Windows 上准备 ImageNet（需要原始数据），可用：

```powershell
python scripts\data_scripts\prepare_imagenet.py --target_directory .\data\imagenet --skip-download --mode copy
```

Windows 普通账户默认没有创建软链接的权限，所以 `--mode` 用 `copy`（或 `move`），不要用默认的 `symlink`。

---

## 7. Windows 上已经适配好的点 / 注意事项

| 问题 | 说明 |
|---|---|
| Windows 没有 NCCL | `src/utils/distributed.py` 在 Windows 用 gloo；单进程下所有 CUDA 集合通信已跳过（`ControlSignal`、tick 同步、gather 等） |
| gloo 不能对 CUDA tensor 做 all_reduce | 同上，单进程路径已提前返回 |
| DataLoader 多进程用 spawn | 默认 `num_workers=2`；如果卡住/报 pickling 错误，命令行加 `dataset.num_workers=0` |
| 软链接需要管理员/开发者模式 | `prepare_imagenet.py` 用 `--mode copy` |
| 中文用户名 / 缓存路径 | 入口脚本把 `HF_HOME`、`TORCH_HOME` 指到仓库内 `.\cache`，避免中文路径问题；仓库路径本身建议**不要有空格/中文** |
| `torch.compile` | Windows 不支持 Triton，预设里没有开启 |
| 显存被桌面占用 | 用 `nvidia-smi` 看；关浏览器/游戏/多显示器可释放 1–3 GiB |
| 分页文件 | 8 GiB 内存的笔记本建议保证系统分页文件 >= 16 GiB（默认“自动管理”即可） |
| 多卡 / 多进程训练 | 不支持，本仓库单卡路径足够 |

---

## 8. 常见错误对照

| 报错 | 原因 / 解决 |
|---|---|
| `no kernel image is available for execution on the device` | torch 不是 cu128（装了 cu118/cu124）。重装 `scripts\setup_env.ps1` |
| `CUDA is not available` | 同上，或驱动太旧（5090/5060 需要 ≥ 570 驱动） |
| `Could not find a version that satisfies torch==2.7.1` | 镜像/网络问题，试 `-UseOfficialIndex` 或 `-Isolated` |
| `OSError: [WinError 1314] 客户端没有所需的特权`（软链接） | 用 `prepare_imagenet.py --mode copy` |
| `DataLoader worker ... BrokenPipe` / 卡住 | 加 `dataset.num_workers=0` |
| 推理 OOM | 见第 4 节；先 bf16 + batch 1 |
| 下载 HF 权重卡住 | 设 `$env:HF_ENDPOINT="https://hf-mirror.com"`；或手动把模型目录放进 `.\cache\huggingface\hub` |
