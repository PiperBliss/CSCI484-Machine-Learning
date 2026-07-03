#!/usr/bin/env python3
"""
Orchestrate training/evaluation across multiple datasets and compare results.

Usage (example):
  python3 run_experiments.py --datasets carpet metal_nut toothbrush

This script:
 - Converts each raw dataset into a binary ImageFolder (train/val/test with accept/reject)
 - Runs `carpet_train.py` for each converted dataset and saves results under `experiments/<name>`
 - Merges all converted datasets into `experiments/merged` and trains on combined data
 - Collects `metrics.csv` from each run and plots comparison curves

It calls `carpet_train.py` as a subprocess so no code changes to that file are required.
"""

import argparse
import csv
import os
import shutil
import subprocess
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".ppm", ".pgm"}


def iter_images(folder: Path):
    if not folder.is_dir():
        return []
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def convert_generic(data_root: Path, out_root: Path, seed: int = 42, train_ratio: float = 0.7, val_ratio: float = 0.15):
    """Create a binary accept/reject dataset under out_root using conventions in the repo.

    - accept: images from train/good + test/good
    - reject: images from other subfolders under test
    """
    from random import Random

    rng = Random(seed)

    ensure_dir(out_root)

    accept_train_src = iter_images(data_root / "train" / "good")
    accept_test_src = iter_images(data_root / "test" / "good")

    # collect reject classes as any subfolder under test except 'good'
    reject_test_src = []
    test_dir = data_root / "test"
    if test_dir.exists():
        for p in test_dir.iterdir():
            if p.is_dir() and p.name != "good":
                reject_test_src.extend(iter_images(p))

    # split accept_train_src into train/val/test
    def split_list(items):
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train = items[:n_train]
        val = items[n_train:n_train + n_val]
        test = items[n_train + n_val:]
        return train, val, test

    a_train, a_val, a_holdout = split_list(accept_train_src)
    a_test = accept_test_src
    r_train, r_val, r_test = split_list(reject_test_src)

    # copy files
    def safe_copy(src: Path, dst_dir: Path):
        ensure_dir(dst_dir)
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

    return {
        "accept": {"train": len(a_train), "val": len(a_val), "test": len(a_test)},
        "reject": {"train": len(r_train), "val": len(r_val), "test": len(r_test)},
    }


def run_training(converted_root: Path, results_dir: Path, extra_args: List[str]):
    ensure_dir(results_dir)
    # write logs into a central logs folder under the results_dir parent
    log_dir = results_dir.parent / "logs"
    ensure_dir(log_dir)

    cmd = ["python3", "carpet_train.py", "--data_root", str(converted_root), "--results_dir", str(results_dir)] + extra_args
    print("Running:", " ".join(cmd))

    # capture stdout/stderr to a per-run log file
    run_name = results_dir.name
    log_path = log_dir / f"{run_name}.log"
    with open(log_path, "w") as lf:
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True)

    # also print a short summary pointer
    print(f"Wrote run log: {log_path}")
    return r.returncode == 0


def read_metrics_csv(results_dir: Path):
    csvp = results_dir / "metrics.csv"
    if not csvp.exists():
        return None
    epochs, tr_l, tr_a, va_l, va_a = [], [], [], [], []
    with open(csvp, newline="") as f:
        r = csv.reader(f)
        hdr = next(r, None)
        for row in r:
            epochs.append(int(row[0]))
            tr_l.append(float(row[1]))
            tr_a.append(float(row[2]))
            va_l.append(float(row[3]))
            va_a.append(float(row[4]))
    return {"epochs": epochs, "train_loss": tr_l, "train_acc": tr_a, "val_loss": va_l, "val_acc": va_a}


def merge_converted(dirs: List[Path], out_root: Path):

    if out_root.exists():
        shutil.rmtree(out_root)
    for split in ["train", "val", "test"]:
        for cls in ["accept", "reject"]:
            ensure_dir(out_root / split / cls)

    for d in dirs:
        for split in ["train", "val", "test"]:
            for cls in ["accept", "reject"]:
                src = d / split / cls
                if not src.exists():
                    continue
                for p in src.iterdir():
                    if p.is_file():
                        dst = out_root / split / cls / p.name
                        if dst.exists():
                            # avoid name collision
                            dst = out_root / split / cls / (p.stem + "_" + d.name + p.suffix)
                        shutil.copy2(p, dst)


def plot_comparison(all_metrics: dict, out_dir: Path):
    ensure_dir(out_dir)

    # Summary CSV of final metrics
    summary_csv = out_dir / "summary_metrics.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "final_epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        for name, m in all_metrics.items():
            if m is None or len(m.get("epochs", [])) == 0:
                w.writerow([name, "", "", "", "", ""])
                continue
            i = len(m["epochs"]) - 1
            w.writerow([name, m["epochs"][i], m["train_loss"][i], m["train_acc"][i], m["val_loss"][i], m["val_acc"][i]])

    # Pretty plot: Val Accuracy comparison
    plt.figure(figsize=(8, 5))
    for name, m in all_metrics.items():
        if m is None:
            continue
        plt.plot(m["epochs"], m["val_acc"], marker="o", linewidth=2, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    ds_list = ", ".join([k for k in all_metrics.keys()])
    plt.title(f"Validation Accuracy Comparison (datasets: {ds_list})")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "val_accuracy_comparison.png", dpi=200)
    plt.close()

    # Pretty plot: Val Loss comparison
    plt.figure(figsize=(8, 5))
    for name, m in all_metrics.items():
        if m is None:
            continue
        plt.plot(m["epochs"], m["val_loss"], marker="o", linewidth=2, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    ds_list = ", ".join([k for k in all_metrics.keys()])
    plt.title(f"Validation Loss Comparison (datasets: {ds_list})")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "val_loss_comparison.png", dpi=200)
    plt.close()

    # Per-dataset subplots: train vs val (loss & acc)
    plt.figure(figsize=(12, 6))
    for idx, (name, m) in enumerate(all_metrics.items(), start=1):
        if m is None:
            continue
        # loss subplot
        plt.subplot(2, max(1, len(all_metrics)), idx)
        plt.plot(m["epochs"], m["train_loss"], label="train loss")
        plt.plot(m["epochs"], m["val_loss"], label="val loss")
        plt.title(f"Loss: {name}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(alpha=0.2)
        plt.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "per_dataset_loss_subplots.png", dpi=200)
    plt.close()

    plt.figure(figsize=(12, 6))
    for idx, (name, m) in enumerate(all_metrics.items(), start=1):
        if m is None:
            continue
        plt.subplot(2, max(1, len(all_metrics)), idx)
        plt.plot(m["epochs"], m["train_acc"], label="train acc")
        plt.plot(m["epochs"], m["val_acc"], label="val acc")
        plt.title(f"Accuracy: {name}")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid(alpha=0.2)
        plt.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "per_dataset_accuracy_subplots.png", dpi=200)
    plt.close()




def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["carpet", "metal_nut", "toothbrush"], help="Paths (relative) to dataset roots")
    p.add_argument("--work_dir", default="experiments", help="Where to write converted datasets and results")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    base = Path(args.work_dir)
    ensure_dir(base)

    converted_dirs = []
    metrics_by_name = {}

    for ds in args.datasets:
        ds_path = Path(ds)
        name = ds_path.name
        converted = base / f"converted_{name}"
        print(f"Converting {ds} -> {converted}")
        stats = convert_generic(ds_path, converted, seed=args.seed)
        print("Converted stats:", stats)
        converted_dirs.append(converted)

        # run training
        results_dir = base / f"results_{name}"
        extra = ["--epochs", str(args.epochs), "--batch_size", str(args.batch_size), "--num_workers", str(args.num_workers), "--image_size", str(args.image_size), "--results_dir", str(results_dir)]
        ok = run_training(converted, results_dir, extra_args=["--epochs", str(args.epochs), "--batch_size", str(args.batch_size), "--num_workers", str(args.num_workers), "--image_size", str(args.image_size), "--results_dir", str(results_dir)])
        if not ok:
            print(f"Training failed for {name}")
        metrics = read_metrics_csv(results_dir)
        metrics_by_name[name] = metrics

    # Merge datasets
    merged = base / "converted_merged"
    print("Merging converted datasets ->", merged)
    merge_converted(converted_dirs, merged)

    merged_results = base / "results_merged"
    ok = run_training(merged, merged_results, extra_args=["--epochs", str(args.epochs), "--batch_size", str(args.batch_size), "--num_workers", str(args.num_workers), "--image_size", str(args.image_size), "--results_dir", str(merged_results)])
    if not ok:
        print("Merged training failed")
    metrics_by_name["merged"] = read_metrics_csv(merged_results)

    # Plot comparisons
    plot_comparison(metrics_by_name, base / "summary_plots")

    print("Done. Summary plots and results in:", base)


if __name__ == "__main__":
    main()
