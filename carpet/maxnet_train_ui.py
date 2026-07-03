#!/usr/bin/env python3
"""
Train a 4-layer NN to compute y=max(x1,x2) and dump artifacts to out_dir.

TEACHING FEATURES (kept from original):
  1) Artifacts folder with CSV splits, logs, plots, predictions, report, config
  2) trace_samples.txt:
       - Picks N fixed training inputs (default 10)
       - Every K epochs (default 10, plus epoch 1), records:
           * Layer weights + biases (fc1..fc4)
           * Per-sample layer-by-layer activations:
               z1, a1, z2, a2, z3, a3, z4 (output)
  3) Progress / status UI:
       - Shows a status line for major phases
       - Shows progress bars for:
           * Dataset generation
           * CSV writing
           * Training (epochs + batch progress)
           * Saving artifacts
       - Shows total bytes written and average write speed at the end

NEW FEATURES (requested):
  A) "Pretty" terminal UI that does NOT scroll:
       - Uses ANSI cursor control to redraw a fixed dashboard region.
       - Falls back to normal printing when stdout is not a TTY.
  B) Colorized text (ANSI):
       - Falls back to no-color when stdout is not a TTY.
  C) Output folder name automatically appends current date-time:
       - if --out_dir is "run/out", actual dir becomes "run/out_YYYY-MM-DD_HH-MM-SS"
  D) network_overview.txt:
       - ASCII visualization of the network
       - Parameter counts per layer and total
       - Formulas: affine transform / inner product, ReLU
       - Shape annotations

NEW FEATURES (requested in this prompt):
  E) Graphviz artifacts (auto-generate .dot files + PNGs if 'dot' exists):
       - Creates a subfolder inside out_dir (graphviz/)
       - Writes:
           * maxnet_collapsed.dot  (layer-level nodes)
           * maxnet_expanded.dot   (neuron-level nodes)
       - If Graphviz 'dot' executable is found on PATH, also writes:
           * maxnet_collapsed.png
           * maxnet_expanded.png
         using high DPI for crisp images.

Notes:
  - No external progress-bar dependencies (no tqdm).
  - Works on cpu / mps / cuda.
"""

import argparse
import csv
import json
import os
import random
import sys
import time
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import List

import torch
import torch.nn as nn

# matplotlib is only for saving plots (no fancy styling)
import matplotlib.pyplot as plt


# ============================
# Terminal UI (fixed, non-scrolling)
# ============================
CSI = "\033["


def _isatty() -> bool:
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def _supports_ansi() -> bool:
    # In most macOS/Linux terminals, TTY implies ANSI works.
    # If you're piping output to a file, ANSI should be disabled.
    return _isatty()


class Ansi:
    def __init__(self, enable: bool):
        self.enable = enable

    def c(self, code: str) -> str:
        return f"{CSI}{code}m" if self.enable else ""

    @property
    def reset(self) -> str:
        return self.c("0")

    @property
    def bold(self) -> str:
        return self.c("1")

    @property
    def dim(self) -> str:
        return self.c("2")

    @property
    def red(self) -> str:
        return self.c("31")

    @property
    def green(self) -> str:
        return self.c("32")

    @property
    def yellow(self) -> str:
        return self.c("33")

    @property
    def blue(self) -> str:
        return self.c("34")

    @property
    def magenta(self) -> str:
        return self.c("35")

    @property
    def cyan(self) -> str:
        return self.c("36")

    def clear_screen(self) -> str:
        return (CSI + "2J" + CSI + "H") if self.enable else ""

    def move(self, row: int, col: int = 1) -> str:
        return f"{CSI}{row};{col}H" if self.enable else ""

    def clear_line(self) -> str:
        return (CSI + "2K") if self.enable else ""


class FixedDashboard:
    """
    A fixed "dashboard" at the top of the terminal.
    We redraw the same lines in-place so output doesn't scroll.

    If ANSI isn't available, we fall back to plain prints.
    """

    def __init__(self, enable: bool, height: int = 12):
        self.enable = enable
        self.ansi = Ansi(enable)
        self.height = height
        self._initialized = False
        self.lines = [""] * height  # cached last lines

    def init(self, title: str):
        if not self.enable or self._initialized:
            return
        # Clear screen and print blank dashboard area
        sys.stdout.write(self.ansi.clear_screen())
        sys.stdout.write(title + "\n")
        for _ in range(self.height - 1):
            sys.stdout.write("\n")
        sys.stdout.flush()
        self._initialized = True

    def set_line(self, idx: int, text: str):
        if idx < 0 or idx >= self.height:
            return

        if not self.enable:
            # fallback: just print (scrolling)
            print(text, flush=True)
            return

        if self.lines[idx] == text:
            return  # no need to redraw identical line
        self.lines[idx] = text

        # The dashboard starts at line 1 of terminal (after clear_screen),
        # so idx=0 -> row=1, idx=1 -> row=2, etc.
        row = idx + 1
        sys.stdout.write(self.ansi.move(row, 1) + self.ansi.clear_line() + text)
        sys.stdout.flush()

    def finish(self):
        if not self.enable:
            return
        # Move cursor just below dashboard so any later prints won't overwrite it
        sys.stdout.write(self.ansi.move(self.height + 1, 1))
        sys.stdout.flush()


# ============================
# Human-friendly formatting
# ============================
def _human_bytes(n: int) -> str:
    n = int(n)
    units = ["B", "KB", "MB", "GB", "TB"]
    f = float(n)
    for u in units:
        if f < 1024.0 or u == units[-1]:
            return f"{int(f)} {u}" if u == "B" else f"{f:.2f} {u}"
        f /= 1024.0
    return f"{f:.2f} TB"


def _human_seconds(sec: float) -> str:
    sec = max(0.0, float(sec))
    if sec < 60:
        return f"{sec:.1f}s"
    m = int(sec // 60)
    s = sec - 60 * m
    if m < 60:
        return f"{m}m {s:.0f}s"
    h = m // 60
    m2 = m % 60
    return f"{h}h {m2}m"


def fmt_float(x: float, width: int = 10, prec: int = 5) -> str:
    return f"{x:{width}.{prec}f}"


def fmt_int(x: int, width: int = 6) -> str:
    return f"{x:{width}d}"


def tensor_to_nested_list(t: torch.Tensor) -> list:
    return t.detach().cpu().numpy().tolist()


def format_matrix(mat: List[List[float]], row_prefix: str = "  ", width: int = 10, prec: int = 5) -> str:
    lines = []
    for r in mat:
        lines.append(row_prefix + " ".join(fmt_float(float(v), width=width, prec=prec) for v in r))
    return "\n".join(lines)


def format_vector(vec: List[float], prefix: str = "  ", width: int = 10, prec: int = 5) -> str:
    return prefix + " ".join(fmt_float(float(v), width=width, prec=prec) for v in vec)


# ============================
# Progress bars (renderable into dashboard lines)
# ============================
class ProgressBar:
    """
    Progress bar that can render as a string.
    We don't print directly inside update(); the caller decides where to display.
    """

    def __init__(self, total: int, label: str = "", width: int = 26):
        self.total = max(1, int(total))
        self.label = label
        self.width = max(10, int(width))
        self.start = time.time()
        self.i = 0
        self.extra = ""

    def update(self, i: int, extra: str = ""):
        self.i = max(0, min(int(i), self.total))
        self.extra = extra

    def render(self) -> str:
        frac = self.i / self.total
        filled = int(self.width * frac)
        bar = "█" * filled + "░" * (self.width - filled)
        elapsed = time.time() - self.start
        rate = self.i / elapsed if elapsed > 1e-9 else 0.0
        eta = (self.total - self.i) / rate if rate > 1e-9 else 0.0
        pct = 100.0 * frac
        base = (
            f"{self.label} [{bar}] {pct:6.2f}% ({self.i}/{self.total}) "
            f"elapsed {_human_seconds(elapsed)} eta {_human_seconds(eta)}"
        )
        if self.extra:
            base += f" | {self.extra}"
        return base


# ============================
# Utilities
# ============================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def save_plot(path, x, ys, labels, title, xlabel, ylabel):
    plt.figure()
    for y, lab in zip(ys, labels):
        plt.plot(x, y, label=lab)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_correctness_grid(model: nn.Module, device: torch.device, out_path: str, x_min: int, x_max: int, step: int = 1):
    import numpy as np
    import matplotlib.pyplot as plt
    model.eval()
    xs = np.arange(int(x_min), int(x_max) + 1, int(step))
    ys = np.arange(int(x_min), int(x_max) + 1, int(step))
    X1, X2 = np.meshgrid(xs, ys)
    X_flat = np.stack([X1.ravel(), X2.ravel()], axis=1).astype(np.float32)
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_flat).to(device)
        pred = model(X_tensor).cpu().numpy().reshape(X1.shape)
    y_true = np.maximum(X1, X2)
    correct = (np.round(pred) == y_true).astype(np.float32)
    overall_acc = float(correct.mean()) if correct.size > 0 else 0.0
    plt.figure(figsize=(10, 7))
    im = plt.imshow(
        correct,
        origin="lower",
        extent=[x_min, x_max, x_min, x_max],
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    plt.colorbar(im, label="correct (0/1)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Exact integer match: round(y_hat) == max(x1,x2)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return overall_acc


def _file_size_or_zero(path: str) -> int:
    try:
        return int(os.path.getsize(path))
    except OSError:
        return 0


# ============================
# Data
# ============================
def make_dataset(n_samples: int, max_int: int, seed: int, pb: ProgressBar = None):
    rng = random.Random(seed)
    X, Y = [], []
    for i in range(n_samples):
        x1 = rng.randint(0, max_int)
        x2 = rng.randint(0, max_int)
        y = max(x1, x2)
        X.append([float(x1), float(x2)])
        Y.append([float(y)])
        if pb:
            pb.update(i + 1, extra=f"range [0..{max_int}]")
    return X, Y


def split_dataset(X, Y, train_frac: float, val_frac: float, seed: int):
    n = len(X)
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    def take(idxs):
        return [X[i] for i in idxs], [Y[i] for i in idxs]

    return take(train_idx), take(val_idx), take(test_idx)


def to_tensor(X, Y, device):
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    Yt = torch.tensor(Y, dtype=torch.float32, device=device)
    return Xt, Yt


@dataclass
class Metrics:
    mse: float
    mae: float
    exact_int_acc: float


@torch.no_grad()
def evaluate(model, X, Y) -> Metrics:
    model.eval()
    pred = model(X)
    err = pred - Y
    mse = float(torch.mean(err * err).item())
    mae = float(torch.mean(torch.abs(err)).item())
    pred_int = torch.round(pred).to(torch.int64)
    y_int = Y.to(torch.int64)
    exact = float(torch.mean((pred_int == y_int).float()).item())
    return Metrics(mse=mse, mae=mae, exact_int_acc=exact)


def write_csv_with_progress(path: str, header: List[str], rows: List[List[object]], pb: ProgressBar = None) -> int:
    total = len(rows)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, r in enumerate(rows):
            w.writerow(r)
            if pb:
                pb.update(i + 1)
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


# ============================
# Model: 2 -> H1 -> H2 -> H3 -> 1
# ============================
class MaxNet4Layer(nn.Module):
    def __init__(self, h1: int, h2: int, h3: int):
        super().__init__()
        self.fc1 = nn.Linear(2, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, 1)
        self.act = nn.ReLU()

        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)

        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc2.bias)

        nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc3.bias)

        nn.init.kaiming_uniform_(self.fc4.weight, nonlinearity="linear")
        nn.init.zeros_(self.fc4.bias)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        return self.fc4(x)

    @torch.no_grad()
    def forward_with_trace(self, x: torch.Tensor):
        self.eval()
        z1 = self.fc1(x)
        a1 = self.act(z1)
        z2 = self.fc2(a1)
        a2 = self.act(z2)
        z3 = self.fc3(a2)
        a3 = self.act(z3)
        z4 = self.fc4(a3)
        return {"z1": z1, "a1": a1, "z2": z2, "a2": a2, "z3": z3, "a3": a3, "z4": z4}


# ============================
# Tracing output
# ============================
def append_trace_block(
    trace_path: str,
    epoch: int,
    model: MaxNet4Layer,
    X_trace_cpu: List[List[float]],
    Y_trace_cpu: List[float],
    device: torch.device,
    max_int: int,
    width: int = 10,
    prec: int = 5,
):
    X_trace_t = torch.tensor(X_trace_cpu, dtype=torch.float32, device=device)
    trace = model.forward_with_trace(X_trace_t)

    z1 = trace["z1"].detach().cpu().numpy().tolist()
    a1 = trace["a1"].detach().cpu().numpy().tolist()
    z2 = trace["z2"].detach().cpu().numpy().tolist()
    a2 = trace["a2"].detach().cpu().numpy().tolist()
    z3 = trace["z3"].detach().cpu().numpy().tolist()
    a3 = trace["a3"].detach().cpu().numpy().tolist()
    z4 = trace["z4"].detach().cpu().view(-1).numpy().tolist()

    fc1_w = tensor_to_nested_list(model.fc1.weight)
    fc1_b = tensor_to_nested_list(model.fc1.bias)
    fc2_w = tensor_to_nested_list(model.fc2.weight)
    fc2_b = tensor_to_nested_list(model.fc2.bias)
    fc3_w = tensor_to_nested_list(model.fc3.weight)
    fc3_b = tensor_to_nested_list(model.fc3.bias)
    fc4_w = tensor_to_nested_list(model.fc4.weight)
    fc4_b = tensor_to_nested_list(model.fc4.bias)

    with open(trace_path, "a") as f:
        f.write("\n" + "=" * 100 + "\n")
        f.write(f"EPOCH {epoch}\n")
        f.write("=" * 100 + "\n\n")

        f.write("PARAMETERS (weights and biases)\n")
        f.write("-" * 100 + "\n")

        f.write("fc1.weight (shape: [h1, 2])\n")
        f.write(format_matrix(fc1_w, row_prefix="  ", width=width, prec=prec) + "\n")
        f.write("fc1.bias (shape: [h1])\n")
        f.write(format_vector(fc1_b, prefix="  ", width=width, prec=prec) + "\n\n")

        f.write("fc2.weight (shape: [h2, h1])\n")
        f.write(format_matrix(fc2_w, row_prefix="  ", width=width, prec=prec) + "\n")
        f.write("fc2.bias (shape: [h2])\n")
        f.write(format_vector(fc2_b, prefix="  ", width=width, prec=prec) + "\n\n")

        f.write("fc3.weight (shape: [h3, h2])\n")
        f.write(format_matrix(fc3_w, row_prefix="  ", width=width, prec=prec) + "\n")
        f.write("fc3.bias (shape: [h3])\n")
        f.write(format_vector(fc3_b, prefix="  ", width=width, prec=prec) + "\n\n")

        f.write("fc4.weight (shape: [1, h3])\n")
        f.write(format_matrix(fc4_w, row_prefix="  ", width=width, prec=prec) + "\n")
        f.write("fc4.bias (shape: [1])\n")
        f.write(format_vector(fc4_b, prefix="  ", width=width, prec=prec) + "\n\n")

        f.write("TRACE SAMPLES (layer-by-layer activations)\n")
        f.write("-" * 100 + "\n")
        f.write(
            "Legend:\n"
            "  x = [x1, x2]\n"
            "  z1 = fc1(x), a1 = ReLU(z1)\n"
            "  z2 = fc2(a1), a2 = ReLU(z2)\n"
            "  z3 = fc3(a2), a3 = ReLU(z3)\n"
            "  z4 = fc4(a3)  (final output)\n\n"
        )

        f.write("SUMMARY (one row per sample)\n")
        f.write(
            " idx |   x1   x2 | y_true |   y_pred | y_round | exact\n"
            "-----+-----------+--------+----------+---------+------\n"
        )
        for i, (x, y_true, y_pred) in enumerate(zip(X_trace_cpu, Y_trace_cpu, z4)):
            y_round = int(round(float(y_pred)))
            exact = 1 if y_round == int(y_true) else 0
            f.write(
                f"{fmt_int(i, 4)} |"
                f"{fmt_int(int(x[0]), 5)}{fmt_int(int(x[1]), 5)} |"
                f"{fmt_int(int(y_true), 6)} |"
                f"{fmt_float(float(y_pred), width=8, prec=4)} |"
                f"{fmt_int(y_round, 7)} |"
                f"{fmt_int(exact, 5)}\n"
            )
        f.write("\n")

        for i, x in enumerate(X_trace_cpu):
            f.write(f"SAMPLE {i}: x=[{int(x[0])}, {int(x[1])}]  y_true={int(Y_trace_cpu[i])}\n")
            f.write("-" * 100 + "\n")

            f.write("z1:\n")
            f.write(format_vector(z1[i], prefix="  ", width=width, prec=prec) + "\n")
            f.write("a1 (ReLU):\n")
            f.write(format_vector(a1[i], prefix="  ", width=width, prec=prec) + "\n\n")

            f.write("z2:\n")
            f.write(format_vector(z2[i], prefix="  ", width=width, prec=prec) + "\n")
            f.write("a2 (ReLU):\n")
            f.write(format_vector(a2[i], prefix="  ", width=width, prec=prec) + "\n\n")

            f.write("z3:\n")
            f.write(format_vector(z3[i], prefix="  ", width=width, prec=prec) + "\n")
            f.write("a3 (ReLU):\n")
            f.write(format_vector(a3[i], prefix="  ", width=width, prec=prec) + "\n\n")

            f.write(f"z4 (output): {fmt_float(float(z4[i]), width=width, prec=prec)}\n")
            f.write(f"rounded(z4): {int(round(float(z4[i])))}\n")
            f.write(f"max(x1,x2):  {max(int(x[0]), int(x[1]))}  (note: should match y_true)\n")
            f.write("\n")


# ============================
# network_overview.txt (ASCII + formulas + param counts)
# ============================
def _layer_param_count(layer: nn.Linear) -> int:
    return int(layer.weight.numel() + layer.bias.numel())


def write_network_overview(path: str, model: MaxNet4Layer):
    h1 = model.fc1.out_features
    h2 = model.fc2.out_features
    h3 = model.fc3.out_features

    p1 = _layer_param_count(model.fc1)
    p2 = _layer_param_count(model.fc2)
    p3 = _layer_param_count(model.fc3)
    p4 = _layer_param_count(model.fc4)
    total = p1 + p2 + p3 + p4

    with open(path, "w") as f:
        f.write("NETWORK OVERVIEW: 4-layer (2 -> h1 -> h2 -> h3 -> 1) with ReLU activations\n")
        f.write("=" * 92 + "\n\n")

        f.write("ASCII DIAGRAM\n")
        f.write("-" * 92 + "\n")
        f.write(
            "  Inputs                    Hidden Layers (ReLU)                          Output\n"
            "  ------                    ---------------------                        ------\n"
            f"  x = [x1,x2]   ->   fc1: 2 -> {h1}   ->   fc2: {h1} -> {h2}   ->   fc3: {h2} -> {h3}   ->   fc4: {h3} -> 1\n"
            "\n"
        )

        f.write("TENSOR SHAPES (single sample)\n")
        f.write("-" * 92 + "\n")
        f.write(
            "  x   : (2,)\n"
            f"  z1  : ({h1},)   a1 = ReLU(z1)\n"
            f"  z2  : ({h2},)   a2 = ReLU(z2)\n"
            f"  z3  : ({h3},)   a3 = ReLU(z3)\n"
            "  z4  : (1,)      output (no activation)\n\n"
        )

        f.write("PARAMETER COUNTS (where do the weights and biases come from?)\n")
        f.write("-" * 92 + "\n")
        f.write(
            f"  fc1: W1 is [{h1} x 2],  b1 is [{h1}]   -> params = {h1}*2 + {h1} = {p1}\n"
            f"  fc2: W2 is [{h2} x {h1}], b2 is [{h2}] -> params = {h2}*{h1} + {h2} = {p2}\n"
            f"  fc3: W3 is [{h3} x {h2}], b3 is [{h3}] -> params = {h3}*{h2} + {h3} = {p3}\n"
            f"  fc4: W4 is [1 x {h3}],  b4 is [1]      -> params = 1*{h3} + 1 = {p4}\n"
            f"\n  TOTAL PARAMETERS: {total}\n\n"
        )

        f.write("THE MATH (general form)\n")
        f.write("-" * 92 + "\n")
        f.write(
            "Affine (a.k.a. linear layer with bias):\n"
            "  z = W x + b\n\n"
            "Inner product view (for one output neuron j):\n"
            "  z_j = sum_i (W_{j,i} * x_i) + b_j\n\n"
            "Activation (ReLU):\n"
            "  ReLU(t) = max(0, t)\n\n"
            "Layer-by-layer for this network:\n"
            "  z1 = W1 x + b1        a1 = ReLU(z1)\n"
            "  z2 = W2 a1 + b2       a2 = ReLU(z2)\n"
            "  z3 = W3 a2 + b3       a3 = ReLU(z3)\n"
            "  z4 = W4 a3 + b4       (final output)\n\n"
            "In PyTorch terms:\n"
            "  z1 = fc1(x); a1 = relu(z1)\n"
            "  z2 = fc2(a1); a2 = relu(z2)\n"
            "  z3 = fc3(a2); a3 = relu(z3)\n"
            "  z4 = fc4(a3)\n"
        )


# ============================
# Graphviz (.dot) + PNG generation
# ============================
def _gv_write_text(path: str, text: str):
    with open(path, "w") as f:
        f.write(text)


def _gv_collapsed_dot(model: MaxNet4Layer) -> str:
    h1 = model.fc1.out_features
    h2 = model.fc2.out_features
    h3 = model.fc3.out_features

    # Collapsed: one node per layer, annotated with sizes and transforms.
    return f"""digraph MaxNet4Layer_Collapsed {{
  rankdir=LR;
  splines=false;
  nodesep=0.6;
  ranksep=1.0;
  fontname="Helvetica";

  node [shape=record, fontname="Helvetica", fontsize=12];

  in  [label="{{Input|2: x=[x1,x2]}}"];
  l1  [label="{{fc1 + ReLU|2 → {h1}}}"];
  l2  [label="{{fc2 + ReLU|{h1} → {h2}}}"];
  l3  [label="{{fc3 + ReLU|{h2} → {h3}}}"];
  out [label="{{fc4 (linear)|{h3} → 1: y_hat}}"];

  in  -> l1  [label="W1,b1"];
  l1  -> l2  [label="W2,b2"];
  l2  -> l3  [label="W3,b3"];
  l3  -> out [label="W4,b4"];
}}
"""


def _gv_expanded_dot(model: MaxNet4Layer, max_nodes_per_layer: int = 200) -> str:
    """
    Expanded: one node per neuron (plus x1/x2 and y).
    NOTE: If someone sets huge hidden sizes, this can explode. We cap each layer.
    """
    h1 = int(model.fc1.out_features)
    h2 = int(model.fc2.out_features)
    h3 = int(model.fc3.out_features)

    # Cap to avoid accidental "thousands of nodes" graphs in class.
    h1c = min(h1, max_nodes_per_layer)
    h2c = min(h2, max_nodes_per_layer)
    h3c = min(h3, max_nodes_per_layer)

    warn = ""
    if (h1c, h2c, h3c) != (h1, h2, h3):
        warn = f'  // NOTE: CAPPED NODES for readability: h1={h1c}/{h1}, h2={h2c}/{h2}, h3={h3c}/{h3}\n'

    lines = []
    lines.append('digraph MaxNet4Layer_Expanded {')
    lines.append('  rankdir=LR;')
    lines.append('  splines=false;')
    lines.append('  nodesep=0.35;')
    lines.append('  ranksep=0.9;')
    lines.append('  fontname="Helvetica";')
    lines.append('  node [shape=circle, fontname="Helvetica", fontsize=10, width=0.35, height=0.35, fixedsize=true];')
    lines.append('  edge [penwidth=0.8];')
    if warn:
        lines.append(warn.rstrip("\n"))

    # Inputs
    lines.append('  subgraph cluster_in {')
    lines.append('    label="Inputs";')
    lines.append('    style="rounded";')
    lines.append('    color="#999999";')
    lines.append('    x1 [shape=box, label="x1", width=0.5, height=0.35];')
    lines.append('    x2 [shape=box, label="x2", width=0.5, height=0.35];')
    lines.append('  }')

    # Hidden layers
    def layer_cluster(name: str, n: int, label: str):
        lines.append(f'  subgraph cluster_{name} {{')
        lines.append(f'    label="{label}";')
        lines.append('    style="rounded";')
        lines.append('    color="#999999";')
        for i in range(n):
            lines.append(f'    {name}_{i} [label="{i}"];')
        if n < 1:
            lines.append(f'    {name}_empty [label="(empty)", shape=plaintext];')
        lines.append('  }')

    layer_cluster("h1", h1c, f"Hidden 1 (ReLU) 2→{h1}")
    layer_cluster("h2", h2c, f"Hidden 2 (ReLU) {h1}→{h2}")
    layer_cluster("h3", h3c, f"Hidden 3 (ReLU) {h2}→{h3}")

    # Output
    lines.append('  subgraph cluster_out {')
    lines.append('    label="Output";')
    lines.append('    style="rounded";')
    lines.append('    color="#999999";')
    lines.append('    y [shape=box, label="ŷ", width=0.55, height=0.35];')
    lines.append('  }')

    # Edges: fully connect adjacent layers
    # Inputs -> h1
    for i in range(h1c):
        lines.append(f'  x1 -> h1_{i};')
        lines.append(f'  x2 -> h1_{i};')

    # h1 -> h2
    for i in range(h1c):
        for j in range(h2c):
            lines.append(f'  h1_{i} -> h2_{j};')

    # h2 -> h3
    for i in range(h2c):
        for j in range(h3c):
            lines.append(f'  h2_{i} -> h3_{j};')

    # h3 -> y
    for i in range(h3c):
        lines.append(f'  h3_{i} -> y;')

    # Helpful note if capped
    if warn:
        lines.append('  // If you want the full expanded graph for large layers, raise max_nodes_per_layer in code.')

    lines.append('}')
    return "\n".join(lines) + "\n"


def _graphviz_dot_to_png(dot_exe: str, dot_path: str, png_path: str, dpi: int = 600) -> bool:
    """
    Run: dot -Tpng -Gdpi=<dpi> -o <png_path> <dot_path>
    Returns True on success, False otherwise.
    """
    try:
        cmd = [dot_exe, "-Tpng", f"-Gdpi={int(dpi)}", "-o", png_path, dot_path]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return r.returncode == 0
    except Exception:
        return False


def generate_graphviz_artifacts(base_out_dir: str, model: MaxNet4Layer) -> List[str]:
    """
    Creates graphviz subfolder and writes dot/png files.
    Returns a list of created file paths (best-effort).
    """
    created = []
    gv_dir = os.path.join(base_out_dir, "graphviz")
    ensure_dir(gv_dir)

    collapsed_dot = os.path.join(gv_dir, "maxnet_collapsed.dot")
    expanded_dot = os.path.join(gv_dir, "maxnet_expanded.dot")

    _gv_write_text(collapsed_dot, _gv_collapsed_dot(model))
    created.append(collapsed_dot)

    _gv_write_text(expanded_dot, _gv_expanded_dot(model))
    created.append(expanded_dot)

    dot_exe = shutil.which("dot")
    if dot_exe:
        collapsed_png = os.path.join(gv_dir, "maxnet_collapsed.png")
        expanded_png = os.path.join(gv_dir, "maxnet_expanded.png")

        if _graphviz_dot_to_png(dot_exe, collapsed_dot, collapsed_png, dpi=600):
            created.append(collapsed_png)

        if _graphviz_dot_to_png(dot_exe, expanded_dot, expanded_png, dpi=600):
            created.append(expanded_png)

    return created


# ============================
# Main
# ============================
def main():
    p = argparse.ArgumentParser(
        description="Train a 4-layer NN to compute y=max(x1,x2) and dump artifacts to out_dir."
    )

    # Output
    p.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Base folder name. Program will append _YYYY-MM-DD_HH-MM-SS automatically.",
    )

    # Data
    p.add_argument("--n_samples", type=int, default=40000)
    p.add_argument("--max_int", type=int, default=50)
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=123)

    # Model
    p.add_argument("--h1", type=int, default=8)
    p.add_argument("--h2", type=int, default=8)
    p.add_argument("--h3", type=int, default=8)

    # Training
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--patience", type=int, default=25, help="Early stopping patience on val MSE.")
    p.add_argument("--print_every", type=int, default=10)

    # Device
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])

    # Tracing
    p.add_argument("--trace_n", type=int, default=10, help="How many fixed samples to trace.")
    p.add_argument("--trace_every", type=int, default=10, help="Record a trace block every N epochs (+epoch 1).")
    p.add_argument("--trace_seed", type=int, default=999, help="Seed used to choose fixed trace samples.")
    p.add_argument("--trace_precision", type=int, default=5, help="Decimal precision in trace_samples.txt")

    # UI
    p.add_argument("--no_ui", action="store_true", help="Disable fixed UI dashboard (use plain prints).")
    p.add_argument("--no_color", action="store_true", help="Disable ANSI colors (even if TTY).")

    args = p.parse_args()

    # Resolve output directory with timestamp suffix
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = f"{args.out_dir}_{timestamp}"
    ensure_dir(out_dir)

    # ANSI + dashboard
    enable_ansi = _supports_ansi() and (not args.no_color)
    ansi = Ansi(enable_ansi)

    enable_ui = _supports_ansi() and (not args.no_ui)
    dash = FixedDashboard(enable=enable_ui, height=12)

    # Initialize dashboard
    title = f"{ansi.cyan}{ansi.bold}MAX(x1,x2) — 4-LAYER NN TRAINING DASHBOARD{ansi.reset}"
    dash.init(title)

    def dash_line(i: int, s: str):
        dash.set_line(i, s)

    # If UI disabled, fallback printing helper
    def log_plain(msg: str):
        if not enable_ui:
            print(msg, flush=True)

    # Tracking disk writes (approx)
    total_bytes_written = 0
    bytes_start_time = time.time()

    # Seeds & device
    set_seed(args.seed)
    device = torch.device(args.device)

    # Header info
    dash_line(1, f"{ansi.dim}Output dir:{ansi.reset} {ansi.green}{out_dir}{ansi.reset}")
    dash_line(2, f"{ansi.dim}Device:{ansi.reset} {ansi.yellow}{device}{ansi.reset}")
    dash_line(3, f"{ansi.dim}Model sizes:{ansi.reset} h1={args.h1} h2={args.h2} h3={args.h3}")
    dash_line(4, f"{ansi.dim}Data:{ansi.reset} n={args.n_samples} max_int={args.max_int} train/val={args.train_frac}/{args.val_frac}")
    dash_line(5, f"{ansi.dim}Status:{ansi.reset} starting...")

    # Save config (include resolved out_dir)
    dash_line(5, f"{ansi.dim}Status:{ansi.reset} saving config.json")
    cfg_path = os.path.join(out_dir, "config.json")
    cfg_obj = vars(args).copy()
    cfg_obj["resolved_out_dir"] = out_dir
    cfg_obj["timestamp"] = timestamp
    save_json(cfg_path, cfg_obj)
    try:
        total_bytes_written += os.path.getsize(cfg_path)
    except OSError:
        pass

    # -------------------------
    # Data generation
    # -------------------------
    dash_line(5, f"{ansi.dim}Status:{ansi.reset} generating dataset")
    pb_data = ProgressBar(args.n_samples, label="Data gen", width=26)
    dash_line(6, ansi.cyan + pb_data.render() + ansi.reset)

    X, Y = make_dataset(args.n_samples, args.max_int, seed=args.seed, pb=pb_data)
    dash_line(6, ansi.green + pb_data.render() + ansi.reset)

    # Split
    dash_line(5, f"{ansi.dim}Status:{ansi.reset} splitting dataset train/val/test")
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = split_dataset(X, Y, args.train_frac, args.val_frac, seed=args.seed)
    dash_line(7, f"{ansi.dim}Split sizes:{ansi.reset} train={len(Xtr)} val={len(Xva)} test={len(Xte)}")

    # -------------------------
    # Write data CSVs
    # -------------------------
    dash_line(5, f"{ansi.dim}Status:{ansi.reset} writing CSV splits")
    train_csv = os.path.join(out_dir, "data_train.csv")
    val_csv = os.path.join(out_dir, "data_val.csv")
    test_csv = os.path.join(out_dir, "data_test.csv")

    # Train CSV
    pb_csv = ProgressBar(len(Xtr), label="CSV train", width=26)
    dash_line(6, ansi.cyan + pb_csv.render() + ansi.reset)
    t_csv0 = time.time()
    total_bytes_written += write_csv_with_progress(
        train_csv,
        ["x1", "x2", "y"],
        [[x[0], x[1], y[0]] for x, y in zip(Xtr, Ytr)],
        pb=pb_csv,
    )
    dt = time.time() - t_csv0
    speed = (os.path.getsize(train_csv) / dt) if dt > 1e-9 else 0.0
    pb_csv.update(pb_csv.total, extra=f"{_human_bytes(os.path.getsize(train_csv))} @ {_human_bytes(speed)}/s")
    dash_line(6, ansi.green + pb_csv.render() + ansi.reset)

    # Val CSV
    pb_csv = ProgressBar(len(Xva), label="CSV val  ", width=26)
    dash_line(6, ansi.cyan + pb_csv.render() + ansi.reset)
    t_csv0 = time.time()
    total_bytes_written += write_csv_with_progress(
        val_csv,
        ["x1", "x2", "y"],
        [[x[0], x[1], y[0]] for x, y in zip(Xva, Yva)],
        pb=pb_csv,
    )
    dt = time.time() - t_csv0
    speed = (os.path.getsize(val_csv) / dt) if dt > 1e-9 else 0.0
    pb_csv.update(pb_csv.total, extra=f"{_human_bytes(os.path.getsize(val_csv))} @ {_human_bytes(speed)}/s")
    dash_line(6, ansi.green + pb_csv.render() + ansi.reset)

    # Test CSV
    pb_csv = ProgressBar(len(Xte), label="CSV test ", width=26)
    dash_line(6, ansi.cyan + pb_csv.render() + ansi.reset)
    t_csv0 = time.time()
    total_bytes_written += write_csv_with_progress(
        test_csv,
        ["x1", "x2", "y"],
        [[x[0], x[1], y[0]] for x, y in zip(Xte, Yte)],
        pb=pb_csv,
    )
    dt = time.time() - t_csv0
    speed = (os.path.getsize(test_csv) / dt) if dt > 1e-9 else 0.0
    pb_csv.update(pb_csv.total, extra=f"{_human_bytes(os.path.getsize(test_csv))} @ {_human_bytes(speed)}/s")
    dash_line(6, ansi.green + pb_csv.render() + ansi.reset)

    # -------------------------
    # Tensors
    # -------------------------
    dash_line(5, f"{ansi.dim}Status:{ansi.reset} converting to tensors")
    Xtr_t, Ytr_t = to_tensor(Xtr, Ytr, device)
    Xva_t, Yva_t = to_tensor(Xva, Yva, device)
    Xte_t, Yte_t = to_tensor(Xte, Yte, device)

    # -------------------------
    # Trace samples init
    # -------------------------
    dash_line(5, f"{ansi.dim}Status:{ansi.reset} selecting trace samples")
    trace_rng = random.Random(args.trace_seed)
    n_trace = min(args.trace_n, len(Xtr))
    trace_indices = trace_rng.sample(range(len(Xtr)), k=n_trace)
    X_trace = [Xtr[i] for i in trace_indices]
    Y_trace = [Ytr[i][0] for i in trace_indices]

    trace_path = os.path.join(out_dir, "trace_samples.txt")
    with open(trace_path, "w") as f:
        f.write("TRACE FILE: layer-by-layer activations over training\n")
        f.write(f"Device: {device}\n")
        f.write(f"Task: y = max(x1, x2) for integer inputs in [0, {args.max_int}]\n")
        f.write(f"Trace samples picked from TRAIN split with trace_seed={args.trace_seed}\n")
        f.write(f"Trace sample indices (within train split): {trace_indices}\n")
        f.write(f"Trace every {args.trace_every} epochs (plus epoch 1)\n\n")

    # -------------------------
    # Model / optimizer
    # -------------------------
    dash_line(5, f"{ansi.dim}Status:{ansi.reset} initializing model/optimizer")
    model = MaxNet4Layer(args.h1, args.h2, args.h3).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    # network_overview.txt
    dash_line(5, f"{ansi.dim}Status:{ansi.reset} writing network_overview.txt")
    net_path = os.path.join(out_dir, "network_overview.txt")
    write_network_overview(net_path, model)
    try:
        total_bytes_written += os.path.getsize(net_path)
    except OSError:
        pass

    # -------------------------
    # Graphviz dot + PNG artifacts (NEW)
    # -------------------------
    dash_line(5, f"{ansi.dim}Status:{ansi.reset} generating Graphviz (.dot/.png) artifacts")
    gv_created = generate_graphviz_artifacts(out_dir, model)
    # Track sizes
    for fp in gv_created:
        total_bytes_written += _file_size_or_zero(fp)

    # Show a helpful UI note (dot may or may not exist)
    dot_found = shutil.which("dot") is not None
    gv_dir = os.path.join(out_dir, "graphviz")
    if dot_found:
        dash_line(6, f"{ansi.green}Graphviz:{ansi.reset} wrote .dot and .png in {ansi.green}{gv_dir}{ansi.reset}")
    else:
        dash_line(6, f"{ansi.yellow}Graphviz:{ansi.reset} wrote .dot in {ansi.green}{gv_dir}{ansi.reset} (dot not found; PNGs skipped)")

    # -------------------------
    # Training loop
    # -------------------------
    dash_line(5, f"{ansi.dim}Status:{ansi.reset} training (early stop on val MSE)")
    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    n_train = Xtr_t.shape[0]
    indices = torch.arange(n_train, device=device)

    epochs_list = []
    train_mse_list, val_mse_list = [], []
    train_mae_list, val_mae_list = [], []
    train_exact_list, val_exact_list = [], []
    log_rows = [["epoch", "train_mse", "train_mae", "train_exact_int", "val_mse", "val_mae", "val_exact_int"]]

    pb_epoch = ProgressBar(args.epochs, label="Epochs  ", width=26)
    dash_line(8, ansi.cyan + pb_epoch.render() + ansi.reset)

    # We'll reuse line 9 for per-epoch batch progress
    pb_batch = None

    t0_train = time.time()
    stopped_epoch = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = indices[torch.randperm(n_train, device=device)]

        n_batches = (n_train + args.batch_size - 1) // args.batch_size
        pb_batch = ProgressBar(n_batches, label=f"Batches e{epoch:03d}", width=26)
        dash_line(9, ansi.cyan + pb_batch.render() + ansi.reset)

        for b, start in enumerate(range(0, n_train, args.batch_size), start=1):
            batch_idx = perm[start : start + args.batch_size]
            xb = Xtr_t[batch_idx]
            yb = Ytr_t[batch_idx]

            pred = model(xb)
            loss = loss_fn(pred, yb)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            # update batch bar (fixed UI)
            pb_batch.update(b, extra=f"loss {float(loss.item()):.5f}")
            dash_line(9, ansi.cyan + pb_batch.render() + ansi.reset)

        # evaluate
        train_metrics = evaluate(model, Xtr_t, Ytr_t)
        val_metrics = evaluate(model, Xva_t, Yva_t)

        # early stopping
        improved = False
        if val_metrics.mse < best_val - 1e-12:
            best_val = val_metrics.mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
            improved = True
        else:
            bad_epochs += 1

        # log
        epochs_list.append(epoch)
        train_mse_list.append(train_metrics.mse)
        val_mse_list.append(val_metrics.mse)
        train_mae_list.append(train_metrics.mae)
        val_mae_list.append(val_metrics.mae)
        train_exact_list.append(train_metrics.exact_int_acc)
        val_exact_list.append(val_metrics.exact_int_acc)

        log_rows.append(
            [
                epoch,
                train_metrics.mse,
                train_metrics.mae,
                train_metrics.exact_int_acc,
                val_metrics.mse,
                val_metrics.mae,
                val_metrics.exact_int_acc,
            ]
        )

        # update epoch bar + metrics line
        pb_epoch.update(epoch, extra=f"valMSE {val_metrics.mse:.6f} best {best_val:.6f}")
        dash_line(8, ansi.cyan + pb_epoch.render() + ansi.reset)

        star = "★" if improved else " "
        dash_line(
            10,
            f"{ansi.yellow}{star}{ansi.reset} "
            f"epoch {epoch:4d} | "
            f"train MSE {train_metrics.mse:.6f} MAE {train_metrics.mae:.6f} exact {train_metrics.exact_int_acc:.3f} | "
            f"val MSE {val_metrics.mse:.6f} MAE {val_metrics.mae:.6f} exact {val_metrics.exact_int_acc:.3f}",
        )

        # occasional plain print (if UI disabled or for logs)
        if (epoch % args.print_every == 0) or (epoch == 1) or (epoch == args.epochs):
            log_plain(
                f"Epoch {epoch:4d} | train MSE {train_metrics.mse:.6f} MAE {train_metrics.mae:.6f} exact {train_metrics.exact_int_acc:.3f} | "
                f"val MSE {val_metrics.mse:.6f} MAE {val_metrics.mae:.6f} exact {val_metrics.exact_int_acc:.3f}"
                + (" (new best)" if improved else "")
            )

        # trace
        if epoch == 1 or (args.trace_every > 0 and epoch % args.trace_every == 0):
            dash_line(5, f"{ansi.dim}Status:{ansi.reset} tracing activations (epoch {epoch}) -> trace_samples.txt")
            append_trace_block(
                trace_path=trace_path,
                epoch=epoch,
                model=model,
                X_trace_cpu=X_trace,
                Y_trace_cpu=Y_trace,
                device=device,
                max_int=args.max_int,
                width=10,
                prec=int(args.trace_precision),
            )
            dash_line(5, f"{ansi.dim}Status:{ansi.reset} training (early stop on val MSE)")

        if bad_epochs >= args.patience:
            stopped_epoch = epoch
            dash_line(11, f"{ansi.red}Early stopping at epoch {epoch} (best val MSE={best_val:.6f}){ansi.reset}")
            log_plain(f"Early stopping at epoch {epoch} (best val MSE={best_val:.6f})")
            break

    elapsed_train = time.time() - t0_train

    # restore best
    dash_line(5, f"{ansi.dim}Status:{ansi.reset} restoring best model (lowest val MSE)")
    if best_state is not None:
        model.load_state_dict(best_state)

    # -------------------------
    # Save model checkpoint (NEW)
    # -------------------------
    dash_line(5, f"{ansi.dim}Status:{ansi.reset} saving model checkpoint")

    model_path = os.path.join(out_dir, "model_state_dict.pt")
    torch.save(model.state_dict(), model_path)

    # Optional: also save a full checkpoint bundle (more complete)
    checkpoint_path = os.path.join(out_dir, "model_checkpoint.pt")
    torch.save(
        {
            "model_class": "MaxNet4Layer",
            "h1": args.h1,
            "h2": args.h2,
            "h3": args.h3,
            "state_dict": model.state_dict(),
            "best_val_mse": best_val,
            "seed": args.seed,
            "device": str(device),
            "timestamp": timestamp,
        },
        checkpoint_path,
    )

    # Track disk bytes
    try:
        total_bytes_written += os.path.getsize(model_path)
    except OSError:
        pass
    try:
        total_bytes_written += os.path.getsize(checkpoint_path)
    except OSError:
        pass


    # -------------------------
    # Save train_log.csv
    # -------------------------
    dash_line(5, f"{ansi.dim}Status:{ansi.reset} saving train_log.csv")
    log_path = os.path.join(out_dir, "train_log.csv")
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerows(log_rows)
    try:
        total_bytes_written += os.path.getsize(log_path)
    except OSError:
        pass

    # -------------------------
    # Save plots
    # -------------------------
    dash_line(5, f"{ansi.dim}Status:{ansi.reset} saving plots")
    title_suffix = f"(n={args.n_samples}, max_int={args.max_int}, h=[{args.h1},{args.h2},{args.h3}], bs={args.batch_size}, lr={args.lr})"
    plots = [
        ("loss_curve.png", [train_mse_list, val_mse_list], ["train MSE", "val MSE"], f"MSE vs Epoch {title_suffix}", "MSE"),
        ("mae_curve.png", [train_mae_list, val_mae_list], ["train MAE", "val MAE"], f"MAE vs Epoch {title_suffix}", "MAE"),
        ("exact_int_curve.png", [train_exact_list, val_exact_list], ["train exact_int", "val exact_int"], f"Exact Integer Accuracy vs Epoch {title_suffix}", "Accuracy"),
    ]

    pb_plots = ProgressBar(len(plots), label="Plots   ", width=26)
    dash_line(6, ansi.cyan + pb_plots.render() + ansi.reset)

    for i, (fname, ys, labels, title_txt, ylabel) in enumerate(plots, start=1):
        save_plot(os.path.join(out_dir, fname), epochs_list, ys, labels, title_txt, "Epoch", ylabel)
        try:
            total_bytes_written += os.path.getsize(os.path.join(out_dir, fname))
        except OSError:
            pass
        pb_plots.update(i, extra=fname)
        dash_line(6, ansi.cyan + pb_plots.render() + ansi.reset)

    dash_line(6, ansi.green + pb_plots.render() + ansi.reset)

    # -------------------------
    # Final evaluation
    # -------------------------
    dash_line(5, f"{ansi.dim}Status:{ansi.reset} final evaluation (train/val/test)")
    train_final = evaluate(model, Xtr_t, Ytr_t)
    val_final = evaluate(model, Xva_t, Yva_t)
    test_final = evaluate(model, Xte_t, Yte_t)

    dash_line(
        7,
        f"{ansi.dim}Final:{ansi.reset} "
        f"Train MSE {train_final.mse:.6f} exact {train_final.exact_int_acc:.3f} | "
        f"Val MSE {val_final.mse:.6f} exact {val_final.exact_int_acc:.3f} | "
        f"Test MSE {test_final.mse:.6f} exact {test_final.exact_int_acc:.3f}"
    )

    # -------------------------
    # predictions_test.csv
    # -------------------------
    dash_line(5, f"{ansi.dim}Status:{ansi.reset} writing predictions_test.csv")
    model.eval()
    with torch.no_grad():
        pred_test = model(Xte_t).detach().cpu().view(-1).tolist()

    pred_rows = []
    for x, y, yhat in zip(Xte, [yy[0] for yy in Yte], pred_test):
        pred_rows.append([x[0], x[1], y, yhat, round(yhat), int(round(yhat) == int(y))])

    preds_path = os.path.join(out_dir, "predictions_test.csv")
    pb_preds = ProgressBar(len(pred_rows), label="Pred CSV", width=26)
    dash_line(6, ansi.cyan + pb_preds.render() + ansi.reset)
    total_bytes_written += write_csv_with_progress(
        preds_path,
        ["x1", "x2", "y_true", "y_pred", "y_pred_rounded", "exact_match"],
        pred_rows,
        pb=pb_preds,
    )
    dash_line(6, ansi.green + pb_preds.render() + ansi.reset)

    # -------------------------
    # Correctness heatmap (model accuracy over grid)
    # -------------------------
    dash_line(5, f"{ansi.dim}Status:{ansi.reset} saving correctness heatmap")
    grid_path = os.path.join(out_dir, "correctness_grid.png")
    acc_over_grid = save_correctness_grid(
        model=model,
        device=device,
        out_path=grid_path,
        x_min=0,
        x_max=int(args.max_int),
        step=1,
    )
    try:
        total_bytes_written += os.path.getsize(grid_path)
    except OSError:
        pass

    # -------------------------
    # report.txt (metrics + learned params + sanity checks)
    # -------------------------
    dash_line(5, f"{ansi.dim}Status:{ansi.reset} writing report.txt (params + sanity checks)")
    report_path = os.path.join(out_dir, "report.txt")

    def tlist(t: torch.Tensor):
        return t.detach().cpu().numpy().tolist()

    with open(report_path, "w") as f:
        f.write("4-layer max(x1,x2) regression\n")
        f.write(f"Resolved output dir: {out_dir}\n")
        f.write(f"Elapsed training time: {elapsed_train:.2f} sec\n\n")

        f.write("Final metrics:\n")
        f.write(f"Train: MSE {train_final.mse:.6f} MAE {train_final.mae:.6f} exact_int {train_final.exact_int_acc:.3f}\n")
        f.write(f"Val:   MSE {val_final.mse:.6f} MAE {val_final.mae:.6f} exact_int {val_final.exact_int_acc:.3f}\n")
        f.write(f"Test:  MSE {test_final.mse:.6f} MAE {test_final.mae:.6f} exact_int {test_final.exact_int_acc:.3f}\n\n")

        f.write("Learned parameters (PyTorch):\n")
        f.write("fc1.weight:\n" + json.dumps(tlist(model.fc1.weight), indent=2) + "\n")
        f.write("fc1.bias:\n" + json.dumps(tlist(model.fc1.bias), indent=2) + "\n")
        f.write("fc2.weight:\n" + json.dumps(tlist(model.fc2.weight), indent=2) + "\n")
        f.write("fc2.bias:\n" + json.dumps(tlist(model.fc2.bias), indent=2) + "\n")
        f.write("fc3.weight:\n" + json.dumps(tlist(model.fc3.weight), indent=2) + "\n")
        f.write("fc3.bias:\n" + json.dumps(tlist(model.fc3.bias), indent=2) + "\n")
        f.write("fc4.weight:\n" + json.dumps(tlist(model.fc4.weight), indent=2) + "\n")
        f.write("fc4.bias:\n" + json.dumps(tlist(model.fc4.bias), indent=2) + "\n")

        f.write("\nSanity checks:\n")
        for (x1, x2) in [(3, 5), (7, 2), (4, 4), (0, args.max_int)]:
            x = torch.tensor([[float(x1), float(x2)]], device=device)
            yhat = float(model(x).item())
            f.write(f"x=({x1},{x2}) pred={yhat:.4f} rounded={round(yhat)} true={max(x1,x2)}\n")

    try:
        total_bytes_written += os.path.getsize(report_path)
    except OSError:
        pass

    # Count trace size once at end (more honest than repeated adds)
    try:
        total_bytes_written += os.path.getsize(trace_path)
    except OSError:
        pass

    # -------------------------
    # Disk write summary
    # -------------------------
    elapsed_write = time.time() - bytes_start_time
    avg_write_speed = total_bytes_written / elapsed_write if elapsed_write > 1e-9 else 0.0

    dash_line(5, f"{ansi.dim}Status:{ansi.reset} done")
    dash_line(
        11,
        f"{ansi.green}DONE{ansi.reset}  "
        f"wrote ~{_human_bytes(total_bytes_written)} in {_human_seconds(elapsed_write)} "
        f"({ _human_bytes(avg_write_speed) }/s avg)"
    )

    # Put cursor below dashboard so your prompt is clean
    dash.finish()

    # Final plain summary (always)
    print(f"\nWrote artifacts to: {out_dir}")
    print(f"  - config:  {cfg_path}")
    print(f"  - data:    data_train.csv, data_val.csv, data_test.csv")
    print(f"  - logs:    train_log.csv")
    print(f"  - plots:   loss_curve.png, mae_curve.png, exact_int_curve.png")
    print(f"  - preds:   predictions_test.csv")
    print(f"  - report:  report.txt")
    print(f"  - trace:   trace_samples.txt")
    print(f"  - net:     network_overview.txt")
    print(f"  - graphviz:{os.path.join(out_dir, 'graphviz')}")
    print(f"            (maxnet_collapsed.dot / maxnet_expanded.dot and PNGs if 'dot' exists)")
    print(f"  - model:   model_state_dict.pt")
    print(f"  - ckpt:    model_checkpoint.pt")


if __name__ == "__main__":
    main()

