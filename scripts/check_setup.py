#!/usr/bin/env python3
"""AlphaFlow single-GPU health check.

Run this on the server *before* launching anything:

    python scripts/check_setup.py

It reports:
  * whether PyTorch can actually use the GPU (RTX 5090 = sm_120 needs CUDA 12.8+);
  * how much free VRAM is available right now (decides inference vs training);
  * whether the Python dependencies are installed;
  * whether ImageNet / dummy data / pretrained checkpoints are present;
  * concrete next-step recommendations.

Exit code is 0 unless --strict is passed and a hard failure is found.
"""
import argparse
import os
import shutil
import sys
from dataclasses import dataclass, field

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OK, WARN, FAIL, INFO = "OK", "WARN", "FAIL", "INFO"


@dataclass
class Check:
    status: str
    name: str
    detail: str = ""


results: list[Check] = []


def add(status: str, name: str, detail: str = ""):
    results.append(Check(status, name, detail))
    line = f"[{status:<4}] {name}"
    if detail:
        line += f": {detail}"
    print(line)


def section(title: str):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def human_bytes(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024.0:
            return f"{num:.1f}{unit}"
        num /= 1024.0
    return f"{num:.1f}PB"


# ---------------------------------------------------------------------------
# 1. System
# ---------------------------------------------------------------------------
def check_system():
    section("1. System")
    add(INFO, "Python", sys.version.split()[0] + f" ({sys.executable})")
    add(INFO, "Platform", f"{sys.platform} / {os.uname().machine if hasattr(os, 'uname') else 'n/a'}")
    add(INFO, "CPU cores", str(os.cpu_count()))

    try:
        import psutil
        vm = psutil.virtual_memory()
        add(INFO, "RAM", f"{human_bytes(vm.available)} available / {human_bytes(vm.total)} total")
        swap = psutil.swap_memory()
        add(INFO, "Swap", f"{human_bytes(swap.free)} free / {human_bytes(swap.total)} total")
    except Exception:
        pass

    total, used, free = shutil.disk_usage(REPO_ROOT)
    add(INFO, "Disk (repo)", f"{human_bytes(free)} free / {human_bytes(total)} total at {REPO_ROOT}")
    if free < 60 * 1024**3:
        add(WARN, "Disk space", "less than 60 GB free: full ImageNet-1K (~150 GB) will not fit")


# ---------------------------------------------------------------------------
# 2. PyTorch / GPU
# ---------------------------------------------------------------------------
def check_torch() -> dict:
    section("2. PyTorch / GPU")
    info = {"cuda_ok": False, "free_vram_gb": 0.0, "capability": None}
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        add(FAIL, "import torch", f"{type(e).__name__}: {e}")
        add(FAIL, "environment", "run `bash scripts/setup_env.sh` (or install torch cu128 manually)")
        return info

    add(INFO, "torch", torch.__version__)
    add(INFO, "torch CUDA build", str(torch.version.cuda))
    add(INFO, "cuda available", str(torch.cuda.is_available()))
    if not torch.cuda.is_available():
        add(FAIL, "CUDA", "torch does not see any GPU. Reinstall torch from the cu128 wheel index.")
        return info

    try:
        props = torch.cuda.get_device_properties(0)
        cap = torch.cuda.get_device_capability(0)
        info["capability"] = cap
        add(INFO, "GPU", f"{props.name} ({human_bytes(props.total_memory)})")
        add(INFO, "compute capability", f"sm_{cap[0]}{cap[1]}")

        # RTX 50 series / Blackwell needs sm_120 kernels.
        if cap[0] >= 12:
            cuda_str = str(torch.version.cuda)
            try:
                cuda_ver = tuple(int(x) for x in cuda_str.split(".")[:2])
            except (ValueError, AttributeError):
                cuda_ver = (0, 0)
            if cuda_ver < (12, 8):
                add(FAIL, "Blackwell support", f"torch was built with CUDA {cuda_str}, need >= 12.8 for {props.name}")
            else:
                add(OK, "Blackwell support", f"torch built with CUDA {cuda_str}")

        # Actually execute a tiny kernel so PTX-JIT / missing-arch errors surface.
        try:
            x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
            _ = x @ x
            torch.cuda.synchronize()
            info["cuda_ok"] = True
            add(OK, "GPU kernel test", "bf16 matmul on GPU succeeded")
        except Exception as e:  # noqa: BLE001
            add(FAIL, "GPU kernel test", f"{type(e).__name__}: {e}")
            add(FAIL, "Blackwell support", "torch cannot execute on this GPU. Install torch from cu128 wheels.")

        if not torch.cuda.is_bf16_supported():
            add(WARN, "bf16 support", "GPU does not report bf16 support; training will be slow/unsupported")

        free, total = torch.cuda.mem_get_info(0)
        info["free_vram_gb"] = free / 1024**3
        add(INFO, "free VRAM now", f"{human_bytes(free)} / {human_bytes(total)}")
        add(INFO, "torch threads", str(torch.get_num_threads()))
    except Exception as e:  # noqa: BLE001
        add(WARN, "GPU query", f"{type(e).__name__}: {e}")
    return info


# ---------------------------------------------------------------------------
# 3. Python packages
# ---------------------------------------------------------------------------
REQUIRED_PACKAGES = [
    "numpy", "scipy", "PIL", "tqdm", "einops", "beartype", "loguru",
    "hydra", "omegaconf", "click", "yaml", "requests", "git", "wandb",
    "psutil", "termcolor", "torchmetrics", "lpips", "timm", "fvcore",
    "torch_dct", "av", "diffusers", "accelerate",
]
OPTIONAL_PACKAGES = ["clip", "matplotlib"]


def check_packages():
    section("3. Python packages")
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
        except Exception as e:  # noqa: BLE001
            missing.append(pkg)
            add(FAIL, pkg, f"{type(e).__name__}: {e}")
    if not missing:
        add(OK, "required packages", f"{len(REQUIRED_PACKAGES)} imports OK")
    else:
        add(FAIL, "missing packages", f"{missing}; run `pip install -r requirements-blackwell.txt`")
    for pkg in OPTIONAL_PACKAGES:
        try:
            __import__(pkg)
        except Exception:  # noqa: BLE001
            add(INFO, pkg, "not installed (optional)")
    return missing


# ---------------------------------------------------------------------------
# 4. Repository layout / data / checkpoints
# ---------------------------------------------------------------------------
def find_images(directory: str, max_entries: int = 3_000_000):
    """Returns (num_images, num_meta, has_class_subfolders). Best-effort counting."""
    if not os.path.isdir(directory):
        return None
    meta_exts = (".json",)
    img_exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    num_imgs = num_meta = num_dirs = 0
    for entry in os.scandir(directory):
        try:
            if entry.is_dir(follow_symlinks=False):
                num_dirs += 1
            elif entry.name.lower().endswith(meta_exts):
                num_meta += 1
            elif entry.name.lower().endswith(img_exts):
                num_imgs += 1
            if num_imgs + num_meta > max_entries:
                break
        except OSError:
            continue
    return num_imgs, num_meta, num_dirs


def check_layout(data_root: str):
    section("4. Data / checkpoints")
    user_cfg = os.path.join(REPO_ROOT, "configs", "env", "user.yaml")
    if os.path.isfile(user_cfg):
        add(OK, "configs/env/user.yaml", "present")
    else:
        add(FAIL, "configs/env/user.yaml", "missing (required by Hydra env config)")

    snap_dir = os.path.join(REPO_ROOT, "snapshots")
    if os.path.isdir(snap_dir):
        pts = sorted(f for f in os.listdir(snap_dir) if f.endswith((".pt", ".pth")))
        if pts:
            for f in pts:
                p = os.path.join(snap_dir, f)
                add(INFO, f"checkpoint {f}", human_bytes(os.path.getsize(p)))
        else:
            add(WARN, "snapshots/", "directory exists but contains no *.pt checkpoint")
        expected = {
            "alphaflow-B-2.pt": "α-Flow-B/2 (pixel/latent-space, FID 40.2 @ NFE-1)",
            "alphaflow-B-2-cfg.pt": "α-Flow-B/2-cfg (FID 5.40 @ NFE-1) <- recommended for a quick try",
            "sd_vae_ft_ema.pt": "SD-VAE encoder needed by latent-space (LDM) checkpoints",
        }
        for name, desc in expected.items():
            if not os.path.isfile(os.path.join(snap_dir, name)):
                status = WARN if name.startswith("sd_vae") else INFO
                add(status, name, f"not found -- {desc}")
    else:
        add(WARN, "snapshots/", "missing; download released checkpoints there for inference")

    # Datasets inside the repo (`${env.project_path}/data`, see configs/env/user.yaml).
    imagenet_train = os.path.join(data_root, "imagenet", "train")
    imagenet_val = os.path.join(data_root, "imagenet", "val")
    for split, path in (("train", imagenet_train), ("val", imagenet_val)):
        count = find_images(path)
        if count is None:
            add(INFO, f"ImageNet {split}", f"not found at {path}")
            continue
        num_imgs, num_meta, num_dirs = count
        if num_dirs > 0:
            add(
                WARN,
                f"ImageNet {split}",
                f"found {num_dirs} class subfolders (raw layout). Run "
                f"`python scripts/data_scripts/prepare_imagenet.py --target_directory {os.path.join(data_root, 'imagenet')} --skip-download`",
            )
        elif num_imgs > 0 and num_meta == 0:
            add(WARN, f"ImageNet {split}", f"{num_imgs} images but no *.meta.json files")
        elif num_imgs > 0:
            add(OK, f"ImageNet {split}", f"{num_imgs} images / {num_meta} meta files")
        else:
            add(WARN, f"ImageNet {split}", "empty or unreadable")

    dummy_train = os.path.join(data_root, "dummy", "train")
    if os.path.isdir(dummy_train):
        count = find_images(dummy_train)
        add(OK, "dummy dataset", f"{count[0]} images at {dummy_train}" if count else "present")
    else:
        add(INFO, "dummy dataset", "not found (only needed for the offline smoke test)")

    stats_dir = os.path.join(data_root, "stats")
    add(INFO if os.path.isdir(stats_dir) else WARN, "dataset stats (FID)", stats_dir)

    for name in ("experiments", "output"):
        p = os.path.join(REPO_ROOT, name)
        add(INFO, name, "present" if os.path.isdir(p) else "absent (will be created)")

    # HuggingFace cache: latent-space checkpoints instantiate the sd-vae through diffusers.
    hf_home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    add(INFO, "HF_HOME", hf_home)
    hit = os.path.isdir(hf_home) and any(
        "sd-vae-ft-ema" in name or "sd_vae" in name
        for name in (os.listdir(hf_home) if os.path.isdir(hf_home) else [])
    )
    if not hit:
        add(WARN, "sd-vae-ft-ema", "not found in HF cache; latent-space models need `stabilityai/sd-vae-ft-ema` (network or pre-populated cache)")


# ---------------------------------------------------------------------------
# 5. Recommendation
# ---------------------------------------------------------------------------
def print_recommendation(torch_info: dict, missing_packages: list[str]):
    section("5. Recommendation")
    free_gb = torch_info.get("free_vram_gb", 0.0)
    cap = torch_info.get("capability")
    cuda_ok = torch_info.get("cuda_ok", False)
    is_windows = sys.platform == "win32"

    if is_windows:
        setup_cmd = "powershell -ExecutionPolicy Bypass -File scripts\\setup_env.ps1"
        gen_cmd = "powershell -ExecutionPolicy Bypass -File scripts\\generate_single_gpu.ps1"
        train_cmd = "powershell -ExecutionPolicy Bypass -File scripts\\train_single_gpu.ps1"
        runbook = "docs/SINGLE_GPU_WINDOWS.md"
    else:
        setup_cmd = "bash scripts/setup_env.sh"
        gen_cmd = "bash scripts/generate_single_gpu.sh"
        train_cmd = "bash scripts/train_single_gpu.sh"
        runbook = "docs/SINGLE_GPU.md"

    if not cuda_ok:
        print("  * PyTorch cannot run on the GPU yet.")
        print(f"    Fix: {setup_cmd}   (installs torch cu128)")
        return
    if missing_packages:
        print("  * Missing Python packages:", ", ".join(missing_packages))
        print("    Fix: pip install -r requirements-blackwell.txt")
        return

    if cap and cap[0] >= 12:
        print(f"  * GPU is a Blackwell card (sm_{cap[0]}{cap[1]}), PyTorch sees it correctly.")

    if free_gb < 3:
        print(f"  * Only {free_gb:.1f} GiB free VRAM -> training is NOT possible right now.")
        print("    You can try single-image inference (batch_size=1), but it may still OOM.")
        print("    Recommended: free the GPU (close desktop apps / shared jobs) and rerun.")
    elif free_gb < 8:
        print(f"  * {free_gb:.1f} GiB free VRAM -> single-image inference is fine.")
        print("    Training a DiT-B/2 latent model needs ~10-16 GiB at batch 32;")
        print("    with batch_gpu=4-8 it can fit in ~6-8 GiB (slow, small-scale only).")
    elif free_gb < 16:
        print(f"  * {free_gb:.1f} GiB free VRAM -> inference + small-scale training (small batch) possible.")
        print("    Use `alphaflow-latentspace-B-2-single-gpu` with reduced dataset.batch_gpu if OOM.")
    else:
        print(f"  * {free_gb:.1f} GiB free VRAM -> full single-GPU inference and B/2 training possible.")

    print()
    print(f"  Suggested order (see {runbook} for the full runbook):")
    print(f"    1) {setup_cmd} && python scripts/check_setup.py")
    print("    2) download released checkpoints into ./snapshots/ and run")
    print(f"       {gen_cmd}")
    print("    3) offline smoke test of the training loop:")
    print("       python scripts/make_dummy_dataset.py")
    print(f"       {train_cmd} alphaflow-smoke-test-single-gpu")
    print("    4) prepare ImageNet and launch a real single-GPU experiment.")


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="AlphaFlow single-GPU health check")
    parser.add_argument("--data-root", default=os.path.join(REPO_ROOT, "data"),
                        help="dataset root directory (default: <repo>/data)")
    parser.add_argument("--strict", action="store_true", help="exit with code 1 on hard failures")
    args = parser.parse_args()

    print(f"AlphaFlow health check for: {REPO_ROOT}")
    check_system()
    torch_info = check_torch()
    missing = check_packages()
    check_layout(args.data_root)
    print_recommendation(torch_info, missing)

    section("Summary")
    n_fail = sum(1 for c in results if c.status == FAIL)
    n_warn = sum(1 for c in results if c.status == WARN)
    print(f"  {n_fail} failure(s), {n_warn} warning(s), {len(results)} checks total")
    if n_fail and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
