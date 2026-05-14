"""
EXPERIMENT 4 — QUERY COUNT vs ATTACK SUCCESS (UPGRADED)
=======================================================
Evaluates how classification accuracy changes as a function of the number
of latency queries (N) used per prediction.

Pipeline:
  1. Load raw per-run latency values.
  2. For each N in [1, 3, 5, 10, 20]:
       a. Group consecutive latency values into non-overlapping windows of size N.
       b. Compute 8 statistical features per window.
       c. 5-Fold Stratified CV with Logistic Regression.
       d. Record mean accuracy, std accuracy, macro-F1.
  3. Plot accuracy vs N with confidence interval shading.
"""

import os
import csv
import sys
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torchvision.models as models

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy.stats import skew, kurtosis

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
MODELS_CONFIG = {
    "resnet18":    models.resnet18,
    "resnet50":    models.resnet50,
    "vgg16":       models.vgg16,
    "inception_v3": models.inception_v3,
    "mobilenet_v2": models.mobilenet_v2,
}
NUM_RUNS     = 100
QUERY_SIZES  = [1, 3, 5, 10, 20]
DEVICE       = torch.device("cpu")
INPUT_SIZE   = (1, 3, 224, 224)
RANDOM_STATE = 42

def load_csv(path: str) -> dict[str, list[float]]:
    data: dict[str, list[float]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("hardware_type", "cpu") != "cpu":
                continue
            data.setdefault(row["model"], []).append(float(row["latency_ms"]))
    return data

def run_inference(runs: int) -> dict[str, list[float]]:
    input_tensor = torch.randn(*INPUT_SIZE).to(DEVICE)
    results = {}
    for name, fn in MODELS_CONFIG.items():
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
    csv_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "Results", "latency_results.csv"
    ))
    if os.path.exists(csv_path):
        print(f"[OK] Loaded data from {csv_path}")
        return load_csv(csv_path)
    return run_inference(NUM_RUNS)

def extract_features(window: np.ndarray) -> list[float]:
    mean_val = window.mean()
    std_val = window.std() if len(window) > 1 else 0.0
    min_val = window.min()
    max_val = window.max()
    p95_val = np.percentile(window, 95) if len(window) > 1 else window[0]
    skew_val = skew(window) if len(window) > 1 else 0.0
    kurt_val = kurtosis(window) if len(window) > 1 else 0.0
    
    if len(window) > 1 and std_val > 1e-6:
        lag1 = np.corrcoef(window[:-1], window[1:])[0, 1]
        if np.isnan(lag1): lag1 = 0.0
    else:
        lag1 = 0.0

    return [mean_val, std_val, min_val, max_val, p95_val, skew_val, kurt_val, lag1]

def build_dataset_for_n(data: dict[str, list[float]], n: int):
    X, y = [], []
    for name, latencies in data.items():
        arr = np.array(latencies)
        for i in range(0, len(arr) - n + 1, n):
            window = arr[i:i+n]
            features = extract_features(window)
            X.append(features)
            y.append(name)
    return np.array(X), np.array(y)

def run_experiment(data: dict[str, list[float]]) -> list[dict]:
    results = []

    print("-" * 70)
    print(f"  {'N (queries)':>12}  {'Samples':>10}  {'Mean Acc':>12}  {'Std Acc':>10}  {'Macro F1':>10}")
    print("-" * 70)

    for n in QUERY_SIZES:
        X, y = build_dataset_for_n(data, n)
        print(f"[INFO] N={n} generated {len(X)} independent samples")

        unique, counts = np.unique(y, return_counts=True)
        if len(counts) > 0 and min(counts) < 5:
            print(f"  {n:>12}   Insufficient samples for 5-fold CV ({min(counts)} per class)")
            results.append({
                "n_queries": n, "mean_accuracy": 0.0, "std_accuracy": 0.0, "macro_f1": 0.0
            })
            continue

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        
        acc_list = []
        f1_list = []
        
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s  = scaler.transform(X_test)
            
            clf.fit(X_train_s, y_train)
            y_pred = clf.predict(X_test_s)
            
            acc_list.append(accuracy_score(y_test, y_pred))
            _, _, f, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
            f1_list.append(f)

        mean_acc = np.mean(acc_list)
        std_acc = np.std(acc_list)
        mean_f1 = np.mean(f1_list)

        results.append({
            "n_queries": n,
            "mean_accuracy": round(mean_acc, 4),
            "std_accuracy": round(std_acc, 4),
            "macro_f1": round(mean_f1, 4)
        })
        print(f"  {n:>12}  {len(X):>10}  {mean_acc*100:>11.2f}%  {std_acc*100:>9.2f}%  {mean_f1:>10.4f}")

    return results

def plot_accuracy_vs_n(results: list[dict], out_path: str) -> None:
    Ns   = [r["n_queries"] for r in results if r["mean_accuracy"] > 0]
    accs = [r["mean_accuracy"] * 100 for r in results if r["mean_accuracy"] > 0]
    stds = [r["std_accuracy"] * 100 for r in results if r["mean_accuracy"] > 0]

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(9, 5))

    accs = np.array(accs)
    stds = np.array(stds)

    ax.plot(Ns, accs, color="black", linewidth=2.0, marker="o", markersize=8, zorder=3, label="Mean Accuracy")
    ax.fill_between(Ns, accs - stds, accs + stds, color="lightgray", alpha=0.5, zorder=2, label="±1 Std. Dev.")

    for n, acc in zip(Ns, accs):
        ax.annotate(
            f"{acc:.1f}%",
            xy=(n, acc), xytext=(0, 10), textcoords="offset points",
            ha="center", fontsize=10, fontweight="bold", zorder=4
        )

    ax.set_xlabel("Number of Queries (N) per Prediction", fontsize=12)
    ax.set_ylabel("Classification Accuracy (%)", fontsize=12)
    ax.set_title(
        "Query Count vs Attack Success Rate\n(Model Fingerprinting via Latency Side-Channel)",
        fontsize=14, fontweight="bold", pad=14,
    )

    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_ylim(max(0, min(accs-stds)-10), 110)
    ax.grid(color="gray", linewidth=0.5, linestyle="--", alpha=0.5, zorder=1)

    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n[OK] Plot saved -> {out_path}")

def main():
    print("=" * 65)
    print("  EXPERIMENT 4 — QUERY COUNT vs ATTACK SUCCESS (UPGRADED)")
    print("=" * 65)
    print(f"  Query sizes tested : {QUERY_SIZES}")

    data    = collect_data()
    results = run_experiment(data)

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Results"))
    os.makedirs(out_dir, exist_ok=True)
    
    csv_path = os.path.join(out_dir, "query_window_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n_queries", "mean_accuracy", "std_accuracy", "macro_f1"])
        writer.writeheader()
        writer.writerows(results)
    print(f"[OK] Results saved -> {csv_path}")

    out_path = os.path.join(out_dir, "accuracy_vs_query_count_paper.png")
    plot_accuracy_vs_n(results, out_path)

    print("\n[OK] Experiment 4 complete.")

if __name__ == "__main__":
    main()
