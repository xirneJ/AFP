# ---------------------------------------------------------------------------
# Launch an AlphaFlow experiment on a single GPU on Windows
# (RTX 5060/5090 = Blackwell; Windows has no NCCL, the code automatically uses
#  the gloo backend and skips multi-process collectives).
#
# Usage (from PowerShell, repo root):
#   powershell -ExecutionPolicy Bypass -File scripts\train_single_gpu.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\train_single_gpu.ps1 alphaflow-smoke-test-single-gpu
#   powershell -ExecutionPolicy Bypass -File scripts\train_single_gpu.ps1 `
#       alphaflow-sigmoid-latentspace-B-2-single-gpu dataset.batch_gpu=4 dataset.batch_size=32
#
# Everything after the experiment name is passed through to Hydra.
# ---------------------------------------------------------------------------
param(
    [string]$Experiment = "alphaflow-smoke-test-single-gpu",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$HydraArgs
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

# --- Environment ---------------------------------------------------------------
$env:PYTHONPATH = "$repoRoot;$env:PYTHONPATH"
if (-not $env:ALPHAFLOW_PROJECT_PATH) { $env:ALPHAFLOW_PROJECT_PATH = $repoRoot }
if (-not $env:PYTORCH_CUDA_ALLOC_CONF) { $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True" }
if (-not $env:HF_HOME) { $env:HF_HOME = Join-Path $repoRoot ".cache\huggingface" }
if (-not $env:TORCH_HOME) { $env:TORCH_HOME = Join-Path $repoRoot ".cache\torch" }
if (-not $env:XDG_CACHE_HOME) { $env:XDG_CACHE_HOME = Join-Path $repoRoot ".cache" }
if (-not $env:OMP_NUM_THREADS) { $env:OMP_NUM_THREADS = "4" }
$env:HYDRA_FULL_ERROR = "1"
$env:TOKENIZERS_PARALLELISM = "false"

New-Item -ItemType Directory -Force -Path (Join-Path $repoRoot "experiments") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $repoRoot "data") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $repoRoot "runs") | Out-Null

# --- Pick the interpreter (prefer the repo venv) -------------------------------
$venvPy = Join-Path $repoRoot "venv\Scripts\python.exe"
if (Test-Path $venvPy) { $python = $venvPy } else { $python = "python" }

Write-Host "=============================================================="
Write-Host " Experiment : $Experiment"
Write-Host " Repo       : $repoRoot"
Write-Host " Output dir : $repoRoot\runs\<desc>\   (desc=... override; defaults to the experiment name)"
Write-Host " Python     : $python"
Write-Host " Extra args : $($HydraArgs -join ' ')"
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host " GPU:"
    & nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader | ForEach-Object { Write-Host "   $_" }
}
Write-Host "=============================================================="
Write-Host ""

$launchArgs = @(
    (Join-Path $repoRoot "infra\launch.py"),
    $Experiment,
    "num_gpus=1",
    "direct_launch=true",
    "wandb.enabled=false"
)
if ($HydraArgs) { $launchArgs += $HydraArgs }

& $python @launchArgs
exit $LASTEXITCODE
