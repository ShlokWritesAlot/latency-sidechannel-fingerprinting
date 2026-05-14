"""
EXPERIMENT 2 — LATENCY DISTRIBUTION + VISUAL REPORT
=====================================================
Extends Experiment 1 by visualizing latency distributions as histograms.
Reads latency_results.csv produced by Experiment 1, or re-runs inference
if the CSV is not found.

Outputs:
  latency_distributions.png  — grid of per-model histograms
"""

import io
import os
import sys
import csv
import time
import numpy as np
from scipy.stats import skew, kurtosis, gaussian_kde
import matplotlib
matplotlib.use("Agg")          # headless-safe backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torchvision.models as models

# Force UTF-8 output on Windows cp1252 terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Configuration ──────────────────────────────────────────────────────────────
MODELS_CONFIG = {
    "resnet18":    models.resnet18,
    "resnet50":    models.resnet50,
    "vgg16":       models.vgg16,
    "inception_v3": models.inception_v3,
    "mobilenet_v2": models.mobilenet_v2,
}
NUM_RUNS   = 100
DEVICE     = torch.device("cpu")
INPUT_SIZE = (1, 3, 224, 224)

MODEL_TITLES = {
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "vgg16": "VGG-16",
    "mobilenet_v2": "MobileNetV2",
    "inception_v3": "InceptionV3",
}
# ───────────────────────────────────────────────────────────────────────────────


# ── Data helpers ───────────────────────────────────────────────────────────────

def load_csv(path: str) -> dict[str, list[float]]:
    """Load latency data from a CSV produced by Experiment 1."""
    data: dict[str, list[float]] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row["model"]
            data.setdefault(model, []).append(float(row["latency_ms"]))
    return data


def run_inference(model: torch.nn.Module, input_tensor: torch.Tensor, runs: int) -> list[float]:
    model.eval()
    latencies = []
    with torch.no_grad():
        _ = model(input_tensor)          # warm-up
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(input_tensor)
            latencies.append((time.perf_counter() - t0) * 1000)
    return latencies


def collect_latencies() -> dict[str, list[float]]:
    """Try to load from CSV; fall back to live inference."""
    csv_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "EXPERIMENT 1 - LATENCY MEASUREMENT + REPORT",
        "latency_results.csv",
    )
    if os.path.exists(csv_path):
        print(f"[OK] Loaded latency data from {csv_path}")
        return load_csv(csv_path)

    print("[!] CSV not found — running inference now ...")
    input_tensor = torch.randn(*INPUT_SIZE).to(DEVICE)
    results = {}
    for name, fn in MODELS_CONFIG.items():
        print(f"  [->] {name}")
        if name == "inception_v3":
            model = fn(weights=None, aux_logits=False).to(DEVICE)
        else:
            model = fn(weights=None).to(DEVICE)
        results[name] = run_inference(model, input_tensor, NUM_RUNS)
    return results


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_distributions(data: dict[str, list[float]], out_path: str) -> None:
    model_names = list(data.keys())
    n = len(model_names)

    # Use white background, default scientific aesthetics
    plt.style.use("default")
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "Inference Latency Distributions per Model",
        fontsize=18, fontweight="bold", y=0.98,
    )

    # 5 subplots: 2 rows, 3 columns
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    axes = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(n)]

    for ax, name in zip(axes, model_names):
        arr   = np.array(data[name])
        title = MODEL_TITLES.get(name, name)
        
        mean  = arr.mean()
        std   = arr.std()
        p95   = np.percentile(arr, 95)
        skew_val = skew(arr)
        kurt_val = kurtosis(arr)

        # Histogram (density=True for KDE)
        ax.hist(
            arr, bins=30, alpha=0.6, density=True,
            edgecolor="black", linewidth=1.0, zorder=2, color="lightgray"
        )
        
        # KDE Curve
        kde = gaussian_kde(arr)
        x_vals = np.linspace(arr.min(), arr.max(), 200)
        ax.plot(x_vals, kde(x_vals), color="black", linewidth=1.5, zorder=3, label="KDE")

        # Mean & P95 lines
        ax.axvline(mean, color="red", linestyle="--", linewidth=1.5,
                   label=f"Mean = {mean:.1f} ms")
        ax.axvline(p95,  color="blue", linestyle=":",  linewidth=1.5,
                   label=f"P95  = {p95:.1f} ms")

        # Styling
        ax.set_title(title, fontsize=14, fontweight="bold", pad=8)
        ax.set_xlabel("Latency (ms)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.tick_params(which="both")
        ax.grid(axis="y", linestyle="--", alpha=0.7, zorder=1)

        # Stats annotation inside plot
        stats_text = f"μ = {mean:.2f}\nσ = {std:.2f}\nP95 = {p95:.2f}\nSkew = {skew_val:.2f}\nKurt = {kurt_val:.2f}"
        ax.text(
            0.95, 0.95, stats_text,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="black", alpha=0.9),
            zorder=4
        )

        ax.legend(fontsize=10, loc="center right")

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Plot saved -> {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  EXPERIMENT 2 — LATENCY DISTRIBUTION + VISUAL REPORT")
    print("=" * 55)

    data = collect_latencies()

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Results"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "latency_distributions_paper.png")
    plot_distributions(data, out_path)

    # Quick console summary
    print(f"\n{'Model':<15} {'Mean':>10} {'Std':>10} {'P95':>10}")
    print("-" * 50)
    for name, latencies in data.items():
        arr = np.array(latencies)
        print(f"{name:<15} {arr.mean():>9.3f}ms {arr.std():>9.3f}ms {np.percentile(arr,95):>9.3f}ms")

    print("\n[OK] Experiment 2 complete.")


if __name__ == "__main__":
    main()
