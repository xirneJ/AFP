# AlphaFlow

This repo will contain the official pytorch code for "[AlphaFlow: Understanding and Improving MeanFlow models](https://arxiv.org/abs/2510.20771)". Our proposed method for understanding and improving MeanFlow models.

![Comparion.](./figures/teaser.jpg)
*Uncurated* samples (seeds 1-8) from the DiT-XL model for MeanFlow [1] and $\alpha$-Flow (our proposed method) produced with 1 (upper) and 2 (lower) sampling steps for ImageNet-1K $256^2$.

## Installation

#### Step 1: Clone and set up environment
Clone the repo, install the conda environment and the packages in environment.yml:
```bash
git clone git@github.com:snap-research/alphaflow.git
cd alphaflow
conda env create -f environment.yml -p ./env
conda activate ./env
```

#### Step 2: Configure user settings
This fork ships a ready-to-use `configs/env/user.yaml` for a single-GPU / `venv`
setup. `project_path` is read from the `ALPHAFLOW_PROJECT_PATH` environment variable,
which every entrypoint (`infra/launch.py`, `scripts/*.py`, `scripts/*_single_gpu.sh|.ps1`)
sets to the repository root automatically — so normally you do not need to edit anything.
For the original multi-GPU/conda setup you can instead start from the example:
```bash
cp configs/env/user-example.yaml configs/env/user.yaml
```
Then set the correct paths to the data, the environment and Wandb config (if required). You need to at least define `project_path` and `conda_init_bin_path` in this file.

## Single-GPU quickstart (RTX 5090 / Blackwell, Linux)

The official instructions above assume 8×H100 + conda. This fork also supports a
**single consumer GPU (e.g. RTX 5090, sm_120) with a plain `venv`**, which is enough
to run inference and to train the B/2 model for controlled comparison experiments.

```bash
# 1) Blackwell-compatible environment (torch cu128) + dependency check
bash scripts/setup_env.sh
source venv/bin/activate
python scripts/check_setup.py          # reports GPU/CUDA/data/checkpoint status

# 2) Single-image inference with the released alpha-Flow-B/2-cfg checkpoint
#    (download the *.pt files into ./snapshots/ first)
bash scripts/generate_single_gpu.sh

# 3) Offline smoke test of the training loop (no ImageNet needed, ~1 min)
python scripts/make_dummy_dataset.py
bash scripts/train_single_gpu.sh alphaflow-smoke-test-single-gpu

# 4) Real single-GPU training presets (FM / MeanFlow / alpha-Flow) on ImageNet-1K
bash scripts/train_single_gpu.sh alphaflow-sigmoid-latentspace-B-2-single-gpu
```

Full runbook (data preparation, VRAM/time estimates, evaluation, how to set up
your own comparison experiments): **[docs/SINGLE_GPU.md](docs/SINGLE_GPU.md)**.

All single-GPU experiment artifacts (training logs, checkpoints, in-training metrics,
generated samples, final FID results, cached dataset statistics) are written under a
single folder: `<repo>/runs/` (configurable via `experiments_root` in
`configs/env/user.yaml`) — see the layout in `docs/SINGLE_GPU.md`.

### Windows (RTX 50-series laptop, PowerShell)

The same single-GPU presets run natively on Windows (PyTorch cu128, gloo backend):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
.\venv\Scripts\Activate.ps1
python scripts\check_setup.py
python scripts\make_dummy_dataset.py
powershell -ExecutionPolicy Bypass -File scripts\train_single_gpu.ps1 alphaflow-smoke-test-single-gpu
```

See **[docs/SINGLE_GPU_WINDOWS.md](docs/SINGLE_GPU_WINDOWS.md)** for the Windows runbook
(inference, data, 8-GiB VRAM limits, HF mirror, troubleshooting).



## Inference

#### Download checkpoints pretrained on ImageNet 256

| Model | FID-NFE-1 | FDD-NFE-1 | FID-NFE-2 | FDD-NFE-2 |
|:-----|:-----:|:---:|:--:|:---------------:|
| [MeanFlow-B/2](https://huggingface.co/snap-research/alphaflow/resolve/main/meanflow-B-2.pt?download=true)            | 43.1 | 819.2 | 38.5 | 787.6 |
| [α-Flow-B/2](https://huggingface.co/snap-research/alphaflow/resolve/main/alphaflow-B-2.pt?download=true)       | 40.2 | 781.0 | 37.1 | 775.0 |
| [MeanFlow-B/2-cfg](https://huggingface.co/snap-research/alphaflow/resolve/main/alphaflow-B-2-cfg.pt?download=true)        | 6.04 | 312.3 | 5.17 | 232.1 |
| [α-Flow-B/2-cfg](https://huggingface.co/snap-research/alphaflow/resolve/main/alphaflow-B-2-cfg.pt?download=true)   | 5.40 | 287.1 | 5.01 | 231.8 |
| [MeanFlow-XL/2-cfg](https://huggingface.co/snap-research/alphaflow/resolve/main/meanflow-XL-2-cfg.pt?download=true)       | 3.47 | 185.8 | 2.46 | 108.7 |
| [α-Flow-XL/2-cfg](https://huggingface.co/snap-research/alphaflow/resolve/main/alphaflow-XL-2-cfg.pt?download=true)  | 2.95 | 164.6 | 2.34 | 105.7 |
| [α-Flow-XL/2+-cfg](https://huggingface.co/snap-research/alphaflow/resolve/main/alphaflow-XL-2-plus-cfg.pt?download=true) | **2.58** | **148.4** | **2.15** | **96.8** |

Download the checkpoints and VAE encoder checkpoint `sd_vae_ft_ema.pt` from [our Hugging Face repository](https://huggingface.co/snap-research/alphaflow/tree/main) into the `./snapshots` folder.
```bash
mkdir snapshots
```

#### Generating samples
Use the following commands to generate samples.
```bash
# AlphaFlow-B/2 NFE-1
torchrun --max-restarts=0 --standalone --nproc-per-node=8 scripts/generate.py ckpt.snapshot_path=./snapshots/alphaflow-B-2.pt output_dir=data/generate/alphaflow-B-2-1-NFE/ seeds=0-49999 batch_size=32 sampling=recflow sampling.sigma_noise=1 sampling.num_steps=1 sampling.enable_trajectory_sampling=true

# AlphaFlow-XL-2-plus-cfg NFE-2
torchrun --max-restarts=0 --standalone --nproc-per-node=8 scripts/generate.py ckpt.snapshot_path=./snapshots/alphaflow-XL-2-plus-cfg.pt output_dir=data/generate/alphaflow-XL-2-plus-cfg-2-NFE/ seeds=0-49999 batch_size=32 sampling=recflow sampling.sigma_noise=1 sampling.num_steps=2 sampling.enable_trajectory_sampling=true sampling.enable_consistency_sampling=true
```
We provide a script `./scripts/evaluate.sh` to generate samples and compute metrics for all released models.

#### Computing metrics

To evaluate a specific directory of generated samples (for FID, FDD, FCD):
```bash
# AlphaFlow-B/2 NFE-1
torchrun --max-restarts=0 --standalone --nproc-per-node=8 scripts/evaluate.py dataset.resolution=\[1,256,256\] gen_dataset.src=data/generate/alphaflow-B-2-1-NFE/ gen_dataset.resolution=\[1,256,256\] gen_dataset.label_shape=\[\]
```

## Training

#### Preparing dataset

Download and reorganize the ImageNet dataset using the provided script. This script prepares the data structure required by our custom dataloader.

```bash
python scripts/data_scripts/prepare_imagenet.py --target_directory ./data/imagenet
```

#### Create an experiment config
We use [hydra](https://hydra.cc/docs/intro/) for experiment configuration. All primary experiment settings are stored in `./infra/experiments/experiments-alphaflow.yaml`.

#### Launching an experiment

You can launch experiments in two ways:

Run the experiment directly without cloning the repository internally.
```bash
torchrun --max-restarts=0 --standalone --nproc-per-node=8 infra/launch.py alphaflow-sigmoid-latentspace-B-2 direct_launch=true wandb.enabled=false
```
Clone the repository into an `./experiments` directory first, then run the experiment from the cloned version.
```bash
mkdir experiments
python infra/launch.py alphaflow-sigmoid-latentspace-B-2 wandb.enabled=false
```

A full list of possible arguments for the launch script can be found in `./configs/infra.yaml`.

## Citation
<details open>
<summary> bibtex </summary>

```latex
@article{zhang2025alphaflow,
  title={AlphaFlow: Understanding and Improving MeanFlow Models},
  author={Zhang, Huijie and Siarohin, Aliaksandr and Menapace, Willi and Vasilkovsky, Michael and Tulyakov, Sergey and Qu, Qing and Skorokhodov, Ivan},
  journal={arXiv preprint arXiv:2510.20771},
  year={2025}
}
```



