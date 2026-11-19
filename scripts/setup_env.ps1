# ---------------------------------------------------------------------------
# Create a Windows `venv` with PyTorch built for Blackwell (sm_120, e.g. RTX
# 5060/5090) and install the dependencies required by AlphaFlow.
#
# Usage (from PowerShell, repo root):
#   powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
#
# Options:
#   -VenvDir venv            venv directory (default: venv)
#   -UseOfficialIndex        use download.pytorch.org instead of the SJTU mirror
#   -Isolated                ignore pip config / env vars (use when a proxy in
#                            pip.conf makes downloads hang)
#
# The default uses the SJTU mirrors because they are fast from mainland China:
#   * torch/torchvision cu128 wheels: https://mirror.sjtu.edu.cn/pytorch-wheels/cu128
#   * regular PyPI packages:          https://mirror.sjtu.edu.cn/pypi/web/simple
# ---------------------------------------------------------------------------
param(
    [string]$VenvDir = "venv",
    [string]$TorchVersion = "2.7.1",
    [string]$TorchvisionVersion = "0.22.1",
    [switch]$UseOfficialIndex,
    [switch]$Isolated
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$PyPiIndex = "https://mirror.sjtu.edu.cn/pypi/web/simple"
if ($UseOfficialIndex) {
    $TorchIndex = "https://download.pytorch.org/whl/cu128"
} else {
    $TorchIndex = "https://mirror.sjtu.edu.cn/pytorch-wheels/cu128"
}

$py = "python"
if ($env:ALPHAFLOW_PYTHON) { $py = $env:ALPHAFLOW_PYTHON }

# --- Pick a Python interpreter -------------------------------------------------
Write-Host "==> Using interpreter: $py"
& $py -c "import sys, platform; print('Python', sys.version.split()[0], platform.architecture()[0])"
if ($LASTEXITCODE -ne 0) { throw "Python not found. Install Python 3.10-3.13 (64-bit) first." }

# --- Create / reuse the venv ---------------------------------------------------
if (-not (Test-Path $VenvDir)) {
    Write-Host "==> Creating venv at $VenvDir"
    & $py -m venv $VenvDir
} else {
    Write-Host "==> Reusing existing venv at $VenvDir"
}
$venvPy = Join-Path $repoRoot "$VenvDir\Scripts\python.exe"
if (-not (Test-Path $venvPy)) { throw "venv python not found: $venvPy" }

$pipCommon = @()
if ($Isolated) { $pipCommon += "--isolated" }

# --- Upgrade pip ---------------------------------------------------------------
Write-Host "==> Upgrading pip"
& $venvPy -m pip install @pipCommon --upgrade pip wheel setuptools -i $PyPiIndex
if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip." }

# --- PyTorch for Blackwell -----------------------------------------------------
Write-Host "==> Installing torch==$TorchVersion torchvision==$TorchvisionVersion (cu128)"
& $venvPy -m pip install @pipCommon `
    "torch==$TorchVersion" "torchvision==$TorchvisionVersion" `
    --index-url $TorchIndex --extra-index-url $PyPiIndex
if ($LASTEXITCODE -ne 0) { throw "Failed to install torch. Try -UseOfficialIndex, or update your GPU driver (>= 570)." }

# --- Remaining requirements ----------------------------------------------------
Write-Host "==> Installing requirements-blackwell.txt"
& $venvPy -m pip install @pipCommon -r requirements-blackwell.txt -i $PyPiIndex
if ($LASTEXITCODE -ne 0) { throw "Failed to install requirements-blackwell.txt." }

# --- Sanity check --------------------------------------------------------------
Write-Host "==> Sanity check"
& $venvPy scripts\check_setup.py

Write-Host ""
Write-Host "==> Done. Activate the environment with:"
Write-Host "      .\$VenvDir\Scripts\Activate.ps1"
Write-Host "    Then run:"
Write-Host "      python scripts/check_setup.py"
