#!/usr/bin/env python3
"""
ImageFolder classifier using PyTorch + ResNet18.

(Stdout-compatible version: preserves original print output exactly.)
"""

import argparse
import csv
import os
from collections import Counter
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import datasets, models, transforms
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt


# -----------------------------
# Device selection
# -----------------------------
def pick_device() -> torch.device:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -----------------------------
# Data utilities
# -----------------------------
def check_folder_structure(data_root: str) -> None:
    for split in ["train", "val", "test"]:
        split_path = os.path.join(data_root, split)
        if not os.path.isdir(split_path):
            raise FileNotFoundError(f"Missing folder: {split_path}")
        if not any(os.path.isdir(os.path.join(split_path, d)) for d in os.listdir(split_path)):
            raise FileNotFoundError(f"No class subfolders found under: {split_path}")


def build_transforms(image_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


def build_dataloaders(data_root, batch_size, num_workers, image_size):
    train_tf, eval_tf = build_transforms(image_size)

    train_ds = datasets.ImageFolder(os.path.join(data_root, "train"), train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(data_root, "val"), eval_tf)
    test_ds  = datasets.ImageFolder(os.path.join(data_root, "test"), eval_tf)

    return (
        DataLoader(train_ds, batch_size, shuffle=True,  num_workers=num_workers),
        DataLoader(val_ds,   batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds,  batch_size, shuffle=False, num_workers=num_workers),
        train_ds.classes,
        train_ds.targets,
    )


# -----------------------------
# Model
# -----------------------------
def build_model(num_classes, pretrained):
    model = models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# -----------------------------
# Train / Eval loops
# -----------------------------
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    y_true, y_pred = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)

        preds = logits.argmax(1)
        loss_sum += loss.item() * y.size(0)
        correct += (preds == y).sum().item()
        total += y.size(0)

        y_true.extend(y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    return loss_sum / total, correct / total, y_true, y_pred


def print_metrics(y_true, y_pred, class_names, title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    ))


# -----------------------------
# Silent saving utilities (NO PRINTS)
# -----------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_curves(results_dir, epochs, tr_l, tr_a, va_l, va_a):
    ensure_dir(results_dir)

    # CSV
    with open(os.path.join(results_dir, "metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        for i in range(len(epochs)):
            w.writerow([epochs[i], tr_l[i], tr_a[i], va_l[i], va_a[i]])

    # Loss plot
    plt.figure()
    plt.plot(epochs, tr_l, label="train")
    plt.plot(epochs, va_l, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "loss_curves.png"), dpi=200)
    plt.close()

    # Accuracy plot
    plt.figure()
    plt.plot(epochs, tr_a, label="train")
    plt.plot(epochs, va_a, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "accuracy_curves.png"), dpi=200)
    plt.close()


def save_confusion(results_dir, name, cm, class_names):
    ensure_dir(results_dir)
    plt.figure()
    plt.imshow(cm)
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.yticks(range(len(class_names)), class_names)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, name), dpi=200)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--results_dir", default="./results")
    args = parser.parse_args()

    check_folder_structure(args.data_root)

    device = pick_device()
    print(f"\nUsing device: {device}")

    train_loader, val_loader, test_loader, class_names, train_targets = \
        build_dataloaders(args.data_root, args.batch_size, args.num_workers, args.image_size)

    print(f"Classes (folder order): {class_names}")

    model = build_model(len(class_names), args.pretrained).to(device)

    counts = Counter(train_targets)
    weights = torch.tensor(
        [sum(counts.values()) / (counts[i] + 1e-6) for i in range(len(class_names))],
        dtype=torch.float,
        device=device,
    )

    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    epochs_hist, tr_l, tr_a, va_l, va_a = [], [], [], [], []
    best_state, best_val = None, -1.0

    for e in range(1, args.epochs + 1):
        tl, ta = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        vl, va, _, _ = evaluate(model, val_loader, loss_fn, device)

        epochs_hist.append(e)
        tr_l.append(tl)
        tr_a.append(ta)
        va_l.append(vl)
        va_a.append(va)

        print("\n" + "-" * 70)
        print(f"Epoch {e}/{args.epochs}")
        print(f"Train: loss={tl:.4f}, acc={ta:.4f}")
        print(f"Val:   loss={vl:.4f}, acc={va:.4f}")

        if va > best_val:
            best_val = va
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)
        print("\nLoaded best model based on validation accuracy.")

    _, _, yv_t, yv_p = evaluate(model, val_loader, loss_fn, device)
    print_metrics(yv_t, yv_p, class_names, "VALIDATION RESULTS")
    save_confusion(args.results_dir, "confusion_matrix_val.png",
                   confusion_matrix(yv_t, yv_p), class_names)

    _, _, yt_t, yt_p = evaluate(model, test_loader, loss_fn, device)
    print_metrics(yt_t, yt_p, class_names, "TEST RESULTS")
    save_confusion(args.results_dir, "confusion_matrix_test.png",
                   confusion_matrix(yt_t, yt_p), class_names)

    save_curves(args.results_dir, epochs_hist, tr_l, tr_a, va_l, va_a)


if __name__ == "__main__":
    main()