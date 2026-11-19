# ---------------------------------------------------------------------------
# Generate samples from a released AlphaFlow checkpoint on a single GPU
# (Windows / RTX 5060 Laptop etc.).
#
# 1) Download the checkpoints into .\snapshots\ (see docs/SINGLE_GPU_WINDOWS.md):
#      snapshots\alphaflow-B-2-cfg.pt
#      snapshots\sd_vae_ft_ema.pt
#
# 2) Run:
#      powershell -ExecutionPolicy Bypass -File scripts\generate_single_gpu.ps1
#
# Environment variables (all optional):
#   CKPT        checkpoint path           (default: snapshots\alphaflow-B-2-cfg.pt)
#   OUT         output directory          (default: runs\<run>\generate\nfe<NFE> when the
#                                          checkpoint lives under runs\,
#                                          else runs\_released\<ckpt>\nfe<NFE>)
#   SEEDS       seeds to generate         (default: 0-7)
#   NFE         number of sampling steps  (default: 1; use 2 for the 2-NFE variant)
#   BATCH_SIZE  generation batch size     (default: 1; raise to 2-4 if VRAM allows)
#   CFG_SCALE   classifier-free guidance  (default: unset; released -cfg checkpoints
#                                          are normally sampled with scale 1.0)
#   DIT_DTYPE   cast the DiT backbone to bfloat16 to save VRAM (default: unset)
# Any extra CLI args are forwarded to scripts\generate.py.
# ---------------------------------------------------------------------------
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Extra
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$env:PYTHONPATH = "$repoRoot;$env:PYTHONPATH"
if (-not $env:ALPHAFLOW_PROJECT_PATH) { $env:ALPHAFLOW_PROJECT_PATH = $repoRoot }
if (-not $env:PYTORCH_CUDA_ALLOC_CONF) { $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True" }
if (-not $env:HF_HOME) { $env:HF_HOME = Join-Path $repoRoot ".cache\huggingface" }
if (-not $env:TORCH_HOME) { $env:TORCH_HOME = Join-Path $repoRoot ".cache\torch" }
if (-not $env:XDG_CACHE_HOME) { $env:XDG_CACHE_HOME = Join-Path $repoRoot ".cache" }
$env:HYDRA_FULL_ERROR = "1"

$venvPy = Join-Path $repoRoot "venv\Scripts\python.exe"
if (Test-Path $venvPy) { $python = $venvPy } else { $python = "python" }

if (-not $env:CKPT) { $env:CKPT = "snapshots\alphaflow-B-2-cfg.pt" }
if (-not $env:SEEDS) { $env:SEEDS = "0-7" }
if (-not $env:NFE) { $env:NFE = "1" }
if (-not $env:BATCH_SIZE) { $env:BATCH_SIZE = "1" }

# --- Where to write generated images (under the unified experiments root) ------
#   * checkpoint inside <repo>\runs\<run>\...  ->  runs\<run>\generate\nfe<NFE>\
#   * released checkpoint in .\snapshots\...    ->  runs\_released\<ckpt>\nfe<NFE>\
# Override explicitly with $env:OUT.
$experimentsRoot = Join-Path $repoRoot "runs"
if (-not $env:OUT) {
    $ckptName = [System.IO.Path]::GetFileNameWithoutExtension($env:CKPT)
    $ckptDir = ""
    if (Test-Path $env:CKPT) { $ckptDir = (Resolve-Path (Split-Path -Parent $env:CKPT)).Path }
    if ($ckptDir -and $ckptDir.ToLower().StartsWith($experimentsRoot.ToLower())) {
        $rel = $ckptDir.Substring($experimentsRoot.Length).TrimStart([char[]]@('\', '/'))
        $runName = ($rel -split '[\\/]')[0]
        if (-not $runName) { $runName = "_released" }
        $env:OUT = Join-Path $experimentsRoot "$runName\generate\nfe$($env:NFE)"
    } else {
        $env:OUT = Join-Path $experimentsRoot "_released\$ckptName\nfe$($env:NFE)"
    }
}
New-Item -ItemType Directory -Force -Path $env:OUT | Out-Null

if (-not (Test-Path $env:CKPT)) {
    Write-Host "[ERROR] checkpoint not found: $($env:CKPT)"
    Write-Host "        Download it first, see docs/SINGLE_GPU_WINDOWS.md."
    exit 1
}
if (-not (Test-Path "snapshots\sd_vae_ft_ema.pt")) {
    Write-Host "[WARN] snapshots\sd_vae_ft_ema.pt is missing."
    Write-Host "       Latent-space (LDM) checkpoints need it to decode images."
}

$genArgs = @(
    (Join-Path $repoRoot "scripts\generate.py"),
    "ckpt.snapshot_path=$($env:CKPT)",
    "output_dir=$($env:OUT)",
    "seeds=$($env:SEEDS)",
    "batch_size=$($env:BATCH_SIZE)",
    "distributed=false",
    "sampling=recflow",
    "sampling.sigma_noise=1",
    "sampling.num_steps=$($env:NFE)",
    "sampling.enable_trajectory_sampling=true",
    "sampling.enable_consistency_sampling=false"
)
if ($env:CFG_SCALE) { $genArgs += "sampling.cfg.scale=$($env:CFG_SCALE)" }
if ($env:DIT_DTYPE) { $genArgs += "inference_dtype=$($env:DIT_DTYPE)" }
if ($Extra) { $genArgs += $Extra }

Write-Host "=============================================================="
Write-Host " Checkpoint : $($env:CKPT)"
Write-Host " Output     : $($env:OUT)"
Write-Host " Seeds      : $($env:SEEDS)  (batch_size=$($env:BATCH_SIZE), NFE=$($env:NFE))"
Write-Host " Extra      : $($Extra -join ' ')"
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host " GPU:"
    & nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader | ForEach-Object { Write-Host "   $_" }
}
Write-Host "=============================================================="
Write-Host ""

& $python @genArgs
exit $LASTEXITCODE
