#!/usr/bin/env python3
"""
Convert the provided 'carpet/' folder into a binary ImageFolder dataset.

Your input layout (from your tree):
carpet/
  train/
    good/
  test/
    good/
    color/
    cut/
    hole/
    metal_contamination/
    thread/
  ground_truth/   (ignored by default)

Output layout:
output_root/
  train/
    accept/
    reject/
  val/
    accept/
    reject/
  test/
    accept/
    reject/

Rules:
- accept images come from carpet/train/good AND carpet/test/good
- reject images come from carpet/test/{color,cut,hole,metal_contamination,thread}
- val is created by splitting the ACCEPT training images and (optionally) some reject images
  (since carpet/ has no val folder)
"""

import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".ppm", ".pgm"}


def iter_images(folder: Path):
    if not folder.is_dir():
        return []
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]


def safe_copy(src: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists():
        stem, suf = src.stem, src.suffix
        i = 1
        while True:
            cand = dst_dir / f"{stem}_{i}{suf}"
            if not cand.exists():
                dst = cand
                break
            i += 1
    shutil.copy2(src, dst)


def split_list(items, train_ratio=0.7, val_ratio=0.15):
    # remaining is test_ratio
    random.shuffle(items)
    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    return train, val, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True, help="Path to carpet/ (the provided dataset folder)")
    parser.add_argument("--output_root", required=True, help="Where to write binary dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    args = parser.parse_args()

    random.seed(args.seed)

    in_root = Path(args.input_root)
    out_root = Path(args.output_root)

    # -----------------------
    # Gather ACCEPT images
    # -----------------------
    accept_train_src = iter_images(in_root / "train" / "good")
    accept_test_src = iter_images(in_root / "test" / "good")

    # We'll split the train/good into train/val (since no val in dataset)
    a_train, a_val, a_holdout = split_list(accept_train_src, args.train_ratio, args.val_ratio)

    # Use dataset-provided test/good as test accept
    a_test = accept_test_src

    # -----------------------
    # Gather REJECT images
    # -----------------------
    defect_classes = ["color", "cut", "hole", "metal_contamination", "thread"]
    reject_test_src = []
    for cls in defect_classes:
        reject_test_src.extend(iter_images(in_root / "test" / cls))

    # There is no reject in carpet/train in your structure, so:
    # We'll create train/val reject by splitting the reject_test_src.
    # (This is a compromise, but it creates all required splits.)
    r_train, r_val, r_test = split_list(reject_test_src, args.train_ratio, args.val_ratio)

    # -----------------------
    # Copy to output
    # -----------------------
    for p in a_train:
        safe_copy(p, out_root / "train" / "accept")
    for p in a_val:
        safe_copy(p, out_root / "val" / "accept")
    for p in a_test:
        safe_copy(p, out_root / "test" / "accept")

    for p in r_train:
        safe_copy(p, out_root / "train" / "reject")
    for p in r_val:
        safe_copy(p, out_root / "val" / "reject")
    for p in r_test:
        safe_copy(p, out_root / "test" / "reject")

    print(f"Done. Wrote dataset to: {out_root}")
    print(f"ACCEPT  train={len(a_train)} val={len(a_val)} test={len(a_test)}")
    print(f"REJECT  train={len(r_train)} val={len(r_val)} test={len(r_test)}")


if __name__ == "__main__":
    main()