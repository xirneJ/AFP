#!/usr/bin/env python3
"""Download / extract / reorganize ImageNet-1K into the layout expected by
AlphaFlow's ``FolderDataset`` dataloader.

Expected layout (flat files + one ``.meta.json`` per image)::

    <target>/train/ n01440764_10026.jpeg  n01440764_10026.meta.json  ...
    <target>/val/   ILSVRC2012_val_00000001.jpeg  ILSVRC2012_val_00000001.meta.json ...

``.meta.json`` contains ``{"class": <int in [0, 999]>}`` where the class index is
the alphabetical index of the synset folder (the standard ImageNet ordering).

The script is **non-destructive**: the original class folders are *moved aside*
into ``<target>/_raw/<split>/<class>/`` (a rename on the same filesystem, so it is
instant) and images are symlinked into the flat layout by default.
Use ``--mode copy`` for real copies or ``--mode move`` to move the files.
Note: ``<target>/_raw`` is outside ``train/``/``val/``, so the dataloader never
sees duplicated samples.

Typical usage on a server that already has ImageNet::

    # case A: extracted class folders -> <target>/train/n01440764/*.JPEG
    python scripts/data_scripts/prepare_imagenet.py \
        --target_directory ./data/imagenet --skip-download

    # case B: the two official tar files are in <target>/
    python scripts/data_scripts/prepare_imagenet.py \
        --target_directory ./data/imagenet --skip-download

    # case C: nothing yet (downloads from image-net.org; most networks need a
    # login/cookie -- see the error message if it fails)
    python scripts/data_scripts/prepare_imagenet.py --target_directory ./data/imagenet
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile

TRAIN_URL = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar"
VAL_URL = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"
VALPREP_URL = "https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh"

IMAGE_EXTS = (".jpeg", ".jpg", ".png", ".bmp", ".webp")
SPLITS = ("train", "val")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def safe_extract(tar: tarfile.TarFile, path: str):
    """`tarfile.extractall(filter=...)` exists on Python >= 3.11.4 / 3.12."""
    try:
        tar.extractall(path=path, filter="data")  # type: ignore[call-arg]
    except TypeError:
        tar.extractall(path=path)


def check_is_tar(path: str) -> bool:
    try:
        return os.path.isfile(path) and tarfile.is_tarfile(path)
    except OSError:
        return False


def count_images(directory: str) -> int:
    if not os.path.isdir(directory):
        return 0
    return sum(1 for f in os.listdir(directory) if f.lower().endswith(IMAGE_EXTS))


def has_class_dirs(directory: str) -> bool:
    if not os.path.isdir(directory):
        return False
    return any(os.path.isdir(os.path.join(directory, name)) for name in os.listdir(directory))


def download(url: str, dst: str) -> bool:
    print(f"--- Downloading {url}\n    -> {dst}")
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    if os.path.isfile(dst) and os.path.getsize(dst) > 0:
        print("    already exists, skipping download.")
        return True
    if shutil.which("wget"):
        rc = subprocess.call(["wget", "-c", "-O", dst, url])
    else:
        import urllib.request
        try:
            urllib.request.urlretrieve(url, dst)
            rc = 0
        except Exception as e:  # noqa: BLE001
            print(f"    urllib failed: {e}")
            rc = 1
    if rc != 0:
        print(f"    [ERROR] download failed (rc={rc}).")
        return False
    return True


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def extract_train(target: str, train_archive: str):
    train_dir = os.path.join(target, "train")
    os.makedirs(train_dir, exist_ok=True)
    print("\n--- Extracting training data")
    with tarfile.open(train_archive, "r") as tar:
        safe_extract(tar, train_dir)

    print("--- Extracting per-class training archives")
    for fname in sorted(os.listdir(train_dir)):
        if not fname.endswith(".tar"):
            continue
        class_name = fname.split(".")[0]
        class_dir = os.path.join(train_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        with tarfile.open(os.path.join(train_dir, fname), "r") as tar:
            safe_extract(tar, class_dir)
        os.remove(os.path.join(train_dir, fname))  # nested archives are useless after this


def sort_val_with_valprep(val_dir: str) -> bool:
    """Sorts raw ILSVRC2012_val_*.JPEG files into synset class folders. Returns success."""
    valprep_path = os.path.join(val_dir, "valprep.sh")
    if not os.path.isfile(valprep_path):
        print("--- Downloading valprep.sh")
        if not download(VALPREP_URL, valprep_path):
            print("    [WARN] Could not download valprep.sh. Validation images stay unsorted; "
                  "training still works (validation is only used for FID statistics).")
            return False
    print("--- Sorting validation images into class folders (valprep.sh)")
    subprocess.run(["bash", "valprep.sh"], cwd=val_dir, check=True)
    os.remove(valprep_path)
    return True


def extract_val(target: str, val_archive: str):
    val_dir = os.path.join(target, "val")
    os.makedirs(val_dir, exist_ok=True)
    print("\n--- Extracting validation data")
    with tarfile.open(val_archive, "r") as tar:
        safe_extract(tar, val_dir)
    sort_val_with_valprep(val_dir)


# ---------------------------------------------------------------------------
# Reorganization
# ---------------------------------------------------------------------------
def build_class_ids(target: str) -> dict[str, int]:
    names = set()
    for split in SPLITS:
        split_dir = os.path.join(target, split)
        if not os.path.isdir(split_dir):
            continue
        for name in os.listdir(split_dir):
            if os.path.isdir(os.path.join(split_dir, name)) and name not in ("_raw",):
                names.add(name)
    return {name: i for i, name in enumerate(sorted(names))}


def relocate_raw_classes(target: str, split: str) -> str:
    """Moves ``<target>/<split>/<class>/`` dirs to ``<target>/_raw/<split>/<class>/``.

    This keeps the original files intact (they are only renamed) while making the
    ``<split>/`` directory flat, so the dataloader does not see each image twice.
    """
    split_dir = os.path.join(target, split)
    raw_split_dir = os.path.join(target, "_raw", split)
    os.makedirs(raw_split_dir, exist_ok=True)
    for name in sorted(os.listdir(split_dir)):
        src = os.path.join(split_dir, name)
        if not os.path.isdir(src):
            continue
        dst = os.path.join(raw_split_dir, name)
        if os.path.exists(dst):
            # Interrupted earlier: merge the not-yet-relocated files instead of
            # deleting anything.
            for fname in os.listdir(src):
                s = os.path.join(src, fname)
                d = os.path.join(dst, fname)
                if not os.path.exists(d):
                    shutil.move(s, d)
            shutil.rmtree(src)
        else:
            os.rename(src, dst)
    return raw_split_dir


def prepare_split(target: str, split: str, mode: str, class_ids: dict[str, int]) -> int:
    """Creates flat ``<split>/<file>`` + ``<split>/<file>.meta.json`` entries."""
    split_dir = os.path.join(target, split)
    if not has_class_dirs(split_dir):
        print(f"    [{split}] already flat, nothing to do.")
        return 0

    raw_split_dir = relocate_raw_classes(target, split)
    num_done = 0
    for class_name in sorted(os.listdir(raw_split_dir)):
        class_dir = os.path.join(raw_split_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        class_id = class_ids[class_name]
        for fname in sorted(os.listdir(class_dir)):
            if not fname.lower().endswith(IMAGE_EXTS):
                continue
            new_name = fname.lower()
            dst_img = os.path.join(split_dir, new_name)
            dst_meta = os.path.join(split_dir, os.path.splitext(new_name)[0] + ".meta.json")
            src_img = os.path.join(class_dir, fname)

            if os.path.isfile(dst_img) and os.path.isfile(dst_meta):
                continue
            if os.path.lexists(dst_img):
                os.remove(dst_img)
            if mode == "symlink":
                os.symlink(os.path.abspath(src_img), dst_img)
            elif mode == "copy":
                shutil.copy2(src_img, dst_img)
            elif mode == "move":
                shutil.move(src_img, dst_img)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            with open(dst_meta, "w") as f:
                json.dump({"class": class_id}, f)
            num_done += 1

    print(f"    [{split}] created {num_done} flat entries (mode={mode}, raw kept in {raw_split_dir}).")
    return num_done


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Prepare ImageNet-1K for AlphaFlow")
    parser.add_argument("--target_directory", default="./data/imagenet", help="where ImageNet lives / will live")
    parser.add_argument("--skip-download", action="store_true", help="do not download; use existing files/folders")
    parser.add_argument("--mode", choices=["symlink", "copy", "move"], default="symlink",
                        help="how to place images in the flat layout (default: symlink)")
    parser.add_argument("--delete-archives", action="store_true", help="delete the official .tar files after extraction")
    args = parser.parse_args()

    target = os.path.abspath(args.target_directory)
    os.makedirs(target, exist_ok=True)

    train_archive = os.path.join(target, "ILSVRC2012_img_train.tar")
    val_archive = os.path.join(target, "ILSVRC2012_img_val.tar")
    train_dir = os.path.join(target, "train")
    val_dir = os.path.join(target, "val")

    # --- 0. already prepared? --------------------------------------------------
    if count_images(train_dir) > 0 and not has_class_dirs(train_dir):
        print(f"Dataset already prepared at {target} (train: {count_images(train_dir)} images). Nothing to do.")
        print("Run: python scripts/check_setup.py")
        return 0

    # --- 1. downloads (only if needed) -----------------------------------------
    need_train_archive = count_images(train_dir) == 0 and not has_class_dirs(train_dir)
    if need_train_archive and not os.path.isfile(train_archive) and not args.skip_download:
        print("=" * 78)
        print("NOTE: image-net.org requires a (free) login. If the download below fails,")
        print("      manually download these two files on a machine with browser access")
        print(f"        {TRAIN_URL}")
        print(f"        {VAL_URL}")
        print(f"      put them into {target}/ and rerun with --skip-download.")
        print("=" * 78)
        download(TRAIN_URL, train_archive)

    need_val = count_images(val_dir) == 0 and not has_class_dirs(val_dir)
    if need_val and not os.path.isfile(val_archive) and not args.skip_download:
        download(VAL_URL, val_archive)

    # --- 2. extraction ----------------------------------------------------------
    if need_train_archive:
        if not check_is_tar(train_archive):
            print(f"[ERROR] {train_archive} is missing or is not a tar archive.")
            print("        Place ILSVRC2012_img_train.tar there, then rerun with --skip-download.")
            return 1
        extract_train(target, train_archive)
        if args.delete_archives:
            os.remove(train_archive)
    elif has_class_dirs(train_dir):
        print(f"--- Using already-extracted training data at {train_dir}")

    if need_val:
        if check_is_tar(val_archive):
            extract_val(target, val_archive)
            if args.delete_archives:
                os.remove(val_archive)
        elif os.path.isdir(val_dir):
            pass  # raw val images may already be extracted; handled below
        else:
            print(f"[WARN] Validation data not found at {val_dir}; training will skip validation/FID.")

    if os.path.isdir(val_dir) and not has_class_dirs(val_dir) and count_images(val_dir) > 0:
        sort_val_with_valprep(val_dir)

    # --- 3. reorganize ----------------------------------------------------------
    print(f"\n--- Reorganizing into the folder + .meta.json layout (mode={args.mode})")
    class_ids = build_class_ids(target)
    if not class_ids:
        print("[ERROR] No class folders found; cannot build ImageNet class ids.")
        return 1
    if len(class_ids) != 1000:
        print(f"[WARN] Found {len(class_ids)} classes (expected 1000).")
    prepare_split(target, "train", args.mode, class_ids)
    if os.path.isdir(val_dir):
        prepare_split(target, "val", args.mode, class_ids)

    print("\n--- Done")
    print(f"    train: {count_images(train_dir)} images")
    print(f"    val:   {count_images(val_dir)} images")
    print("    Next: python scripts/check_setup.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
