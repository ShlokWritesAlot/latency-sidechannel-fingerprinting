"""
EXPERIMENT 4 — QUERY COUNT vs ATTACK SUCCESS
=============================================
Evaluates how classification accuracy changes as a function of the number
of latency queries (N) used per prediction.

Context (Side-Channel / Model-Fingerprinting scenario):
  An attacker observes N latency measurements per "probe" and must infer
  which ML model is running behind an API. Larger N → better features →
  higher attack success rate.

Pipeline:
  1. Load (or generate) raw per-run latency values.
  2. For each N in [1, 3, 5, 10]:
       a. Group consecutive latency values into windows of size N.
       b. Compute features: [mean, std] of each window.
       c. Train a Logistic Regression classifier (80/20 split).
       d. Record test accuracy.
  3. Plot accuracy vs N.

Outputs:
  accuracy_vs_query_count.png
"""

import io
import os
import csv
import sys
import time
import warnings
import numpy as np
import torch
import torchvision.models as models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

# Force UTF-8 output on Windows cp1252 terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Configuration ──────────────────────────────────────────────────────────────
MODELS_CONFIG = {
    "resnet18":    models.resnet18,
    "resnet50":    models.resnet50,
    "vgg16":       models.vgg16,
    "inception_v3": models.inception_v3,
}
NUM_RUNS     = 100          # latency measurements per model
QUERY_SIZES  = [1, 3, 5, 10]
DEVICE       = torch.device("cpu")
INPUT_SIZE   = (1, 3, 224, 224)
RANDOM_STATE = 42
# ───────────────────────────────────────────────────────────────────────────────


# ── Data helpers ───────────────────────────────────────────────────────────────

def load_csv(path: str) -> dict[str, list[float]]:
    data: dict[str, list[float]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            data.setdefault(row["model"], []).append(float(row["latency_ms"]))
    return data


def run_inference(runs: int) -> dict[str, list[float]]:
    input_tensor = torch.randn(*INPUT_SIZE).to(DEVICE)
    results = {}
    for name, fn in MODELS_CONFIG.items():
        print(f"  [->] {name} ({runs} runs) ...")
        if name == "inception_v3":
            model = fn(weights=None, aux_logits=False).to(DEVICE)
        else:
            model = fn(weights=None).to(DEVICE)
        model.eval()
        latencies = []
        with torch.no_grad():
            _ = model(input_tensor)
            for _ in range(runs):
                t0 = time.perf_counter()
                _ = model(input_tensor)
                latencies.append((time.perf_counter() - t0) * 1000)
        results[name] = latencies
    return results


def collect_data() -> dict[str, list[float]]:
    csv_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "EXPERIMENT 1 - LATENCY MEASUREMENT + REPORT",
        "latency_results.csv",
    )
    if os.path.exists(csv_path):
        print(f"[OK] Loaded data from {csv_path}")
        return load_csv(csv_path)
    print("[!] CSV not found — running inference ...")
    return run_inference(NUM_RUNS)


# ── Feature engineering ────────────────────────────────────────────────────────

def build_dataset_for_n(data: dict[str, list[float]], n: int):
    """
    Group each model's latency list into non-overlapping windows of size N.
    Feature per window: [mean, std]   (if N==1, std is 0 → kept for uniformity)
    """
    X, y = [], []
    for name, latencies in data.items():
        arr = np.array(latencies)
        # Discard tail if not divisible
        num_windows = len(arr) // n
        for w in range(num_windows):
            window = arr[w * n : (w + 1) * n]
            mean_val = window.mean()
            std_val  = window.std() if n > 1 else 0.0
            X.append([mean_val, std_val])
            y.append(name)
    return np.array(X), np.array(y)


# ── Experiment loop ────────────────────────────────────────────────────────────

def run_experiment(data: dict[str, list[float]]) -> dict[int, float]:
    results = {}

    print("-" * 55)
    print(f"  {'N (queries)':>12}  {'Train samples':>14}  {'Test samples':>12}  {'Accuracy':>10}")
    print("-" * 55)

    for n in QUERY_SIZES:
        X, y = build_dataset_for_n(data, n)

        if len(X) < 4:
            print(f"  {n:>12}   Insufficient samples — skipping")
            results[n] = 0.0
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        clf.fit(X_train_s, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test_s))

        results[n] = acc
        print(f"  {n:>12}  {len(X_train):>14}  {len(X_test):>12}  {acc*100:>9.2f}%")

    return results


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_accuracy_vs_n(results: dict[int, float], out_path: str) -> None:
    Ns   = sorted(results.keys())
    accs = [results[n] * 100 for n in Ns]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#0D1117")
    ax.set_facecolor("#161B22")

    # Main line + markers
    ax.plot(Ns, accs, color="#4FC3F7", linewidth=2.5,
            marker="o", markersize=9, markerfacecolor="#81D4FA",
            markeredgecolor="white", markeredgewidth=1.2, zorder=3, label="LR Accuracy")

    # Shaded area under curve
    ax.fill_between(Ns, accs, alpha=0.15, color="#4FC3F7", zorder=2)

    # Annotate each point
    for n, acc in zip(Ns, accs):
        ax.annotate(
            f"{acc:.1f}%",
            xy=(n, acc), xytext=(0, 12), textcoords="offset points",
            ha="center", fontsize=10, color="white",
            fontweight="bold",
        )

    # Axes / labels
    ax.set_xlabel("Number of Queries (N) per Prediction", color="#B0BEC5", fontsize=12)
    ax.set_ylabel("Classification Accuracy (%)",          color="#B0BEC5", fontsize=12)
    ax.set_title(
        "Query Count vs Attack Success Rate\n(Model Fingerprinting via Latency Side-Channel)",
        color="white", fontsize=13, fontweight="bold", pad=14,
    )

    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_ylim(0, 110)
    ax.tick_params(colors="#B0BEC5")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363D")
    ax.grid(color="#30363D", linewidth=0.6, linestyle="--", zorder=1)

    legend = ax.legend(fontsize=10, facecolor="#1F2937",
                       edgecolor="#30363D", labelcolor="white")

    # Insight annotation
    insight = (
        "↑ More queries → richer features\n"
        "  → higher attack success"
    )
    ax.text(
        0.98, 0.10, insight, transform=ax.transAxes,
        ha="right", va="bottom", fontsize=9, color="#90CAF9",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#1F2937", alpha=0.9),
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[OK] Plot saved -> {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  EXPERIMENT 4 — QUERY COUNT vs ATTACK SUCCESS")
    print("=" * 55)
    print(f"  Query sizes tested : {QUERY_SIZES}")
    print(f"  Classifier         : Logistic Regression")
    print(f"  Features           : [mean, std] per window")

    data    = collect_data()
    results = run_experiment(data)

    out_path = os.path.join(os.path.dirname(__file__), "accuracy_vs_query_count.png")
    plot_accuracy_vs_n(results, out_path)

    print("\n[OK] Experiment 4 complete.")


if __name__ == "__main__":
    main()
