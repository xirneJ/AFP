#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Create a `venv` with PyTorch built for Blackwell (sm_120, e.g. RTX 5090)
# and install the dependencies required by AlphaFlow.
#
# Usage:
#   bash scripts/setup_env.sh                # creates ./venv with python3.11
#   bash scripts/setup_env.sh .venv          # custom venv directory
#   PYTHON=python3.11 bash scripts/setup_env.sh
#
# Notes:
#   * RTX 50-series GPUs need CUDA >= 12.8 kernels.  The default
#     `pytorch=2.6` conda environment from environment.yml only ships CUDA
#     12.4 and will NOT run on a 5090 ("no kernel image is available for
#     execution on the device" / silent CPU fallback).
#   * The cu128 wheel index is used explicitly, so no conda is required.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR="${1:-venv}"
TORCH_VERSION="${TORCH_VERSION:-2.7.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.1}"
CUDA_TAG="${CUDA_TAG:-cu128}"

# --- Pick a Python interpreter -------------------------------------------------
if [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
elif command -v python3.11 >/dev/null 2>&1; then
  PY="python3.11"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
else
  echo "[ERROR] No python3 found. Please install Python 3.11 first." >&2
  exit 1
fi

PY_VER="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "==> Using $PY (Python $PY_VER) to create venv at $VENV_DIR"
if [[ "$PY_VER" != "3.11" && "$PY_VER" != "3.10" ]]; then
  echo "[WARN] Python $PY_VER detected. The original environment pins Python 3.11."
  echo "       numpy<2 / scipy==1.10.1 wheels may not be available for this version."
  echo "       Prefer: PYTHON=python3.11 bash scripts/setup_env.sh"
fi

# --- Create / reuse the venv ---------------------------------------------------
if [[ ! -d "$VENV_DIR" ]]; then
  "$PY" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# --- PyTorch for Blackwell -----------------------------------------------------
echo "==> Installing torch==${TORCH_VERSION} / torchvision==${TORCHVISION_VERSION} (${CUDA_TAG})"
pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

# --- Remaining requirements ----------------------------------------------------
echo "==> Installing requirements-blackwell.txt"
pip install -r requirements-blackwell.txt

# --- Sanity check --------------------------------------------------------------
echo
echo "==> Sanity check"
python - <<'PY'
import torch
print("torch          :", torch.__version__)
print("torch CUDA     :", torch.version.cuda)
print("cuda available :", torch.cuda.is_available())
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    print(f"GPU            : {name} (compute capability {cap[0]}.{cap[1]})")
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    y = x @ x
    torch.cuda.synchronize()
    print("bf16 matmul on GPU: OK")
    if cap[0] * 10 + cap[1] > 90 and "12.8" not in str(torch.version.cuda):
        print("[WARN] Blackwell GPU but torch was not built with CUDA 12.8+!")
else:
    print("[ERROR] CUDA is not available. If you have a GPU, the torch build is wrong.")
PY

echo
echo "==> Done. Activate the environment with:"
echo "      source ${VENV_DIR}/bin/activate"
echo "    Then run the health check:"
echo "      python scripts/check_setup.py"
