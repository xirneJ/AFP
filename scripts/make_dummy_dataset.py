#!/usr/bin/env python3
"""Create a tiny synthetic image dataset in the folder + `.meta.json` format
expected by AlphaFlow's FolderDataset.

Use it to verify the whole training pipeline (dataloader, VAE, loss, optimizer,
snapshotting) on a GPU without downloading ImageNet:

    python scripts/make_dummy_dataset.py                 # -> data/dummy
    bash scripts/train_single_gpu.sh alphaflow-smoke-test-single-gpu

Layout created:
    data/dummy/train/000000.png + 000000.meta.json   (class: 0)
    data/dummy/val/...
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image


def make_split(out_dir: str, num_images: int, num_classes: int, resolution: int, seed: int, prefix: str):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.RandomState(seed)
    for i in range(num_images):
        cls = i % num_classes
        # Class-dependent color + smooth random noise, so images are not all identical.
        base = rng.uniform(0.0, 1.0, size=(1, 1, 3)).astype(np.float32)
        noise = rng.uniform(-0.5, 0.5, size=(resolution, resolution, 3)).astype(np.float32)
        img = np.clip((base + 0.25 * noise) * 255.0, 0, 255).astype(np.uint8)
        if cls % 2 == 1:
            img = 255 - img  # Give odd classes a distinct look.
        name = f"{prefix}{i:06d}"
        Image.fromarray(img).save(os.path.join(out_dir, f"{name}.png"))
        with open(os.path.join(out_dir, f"{name}.meta.json"), "w") as f:
            json.dump({"class": cls}, f)
    print(f"  wrote {num_images} images to {out_dir}")


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="Create a synthetic folder dataset for AlphaFlow smoke tests")
    parser.add_argument("--out", default=os.path.join(repo_root, "data", "dummy"), help="output dataset directory")
    parser.add_argument("--num-train", type=int, default=256)
    parser.add_argument("--num-val", type=int, default=32)
    parser.add_argument("--classes", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=64, help="square image size in pixels")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Creating dummy dataset at: {args.out}")
    make_split(os.path.join(args.out, "train"), args.num_train, args.classes, args.resolution, args.seed, "train")
    make_split(os.path.join(args.out, "val"), args.num_val, args.classes, args.resolution, args.seed + 1, "val")
    print("Done.")
    print()
    print("Next step (offline training smoke test):")
    print("  bash scripts/train_single_gpu.sh alphaflow-smoke-test-single-gpu")


if __name__ == "__main__":
    sys.exit(main())
