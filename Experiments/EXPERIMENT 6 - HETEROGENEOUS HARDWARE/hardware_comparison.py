"""
EXPERIMENT 6 — HETEROGENEOUS HARDWARE
======================================
Evaluates model classification separability across CPU and GPU hardware.
Compares CPU-only, GPU-only, and mixed hardware environments.
"""

import os
import csv
import sys
import io
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import skew, kurtosis

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace") if hasattr(sys.stdout, 'buffer') else sys.stdout

WINDOW_SIZE = 10
STRIDE = 10
RANDOM_STATE = 42
FEATURE_NAMES = ["mean", "std", "min", "max", "p95", "skewness", "kurtosis", "lag1_autocorr"]

def load_csv(path: str) -> dict[tuple[str, str], list[float]]:
    # Returns (model, hw) -> latencies
    data = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            hw = row.get("hardware_type", "cpu")
            data.setdefault((row["model"], hw), []).append(float(row["latency_ms"]))
    return data

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

def build_dataset(data: dict[tuple[str, str], list[float]]):
    X, y, hw_labels = [], [], []
    for (name, hw), latencies in data.items():
        arr = np.array(latencies)
        n_samples = len(arr)
        for i in range(0, n_samples - WINDOW_SIZE + 1, STRIDE):
            window = arr[i : i + WINDOW_SIZE]
            features = extract_features(window)
            X.append(features)
            y.append(name)
            hw_labels.append(hw)
    return np.array(X), np.array(y), np.array(hw_labels)

def evaluate_cv(clf, X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    acc_list = []
    f1_list = []
    
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)
        
        acc_list.append(accuracy_score(y_test, y_pred))
        _, _, f, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
        f1_list.append(f)
        
    return np.mean(acc_list), np.std(acc_list), np.mean(f1_list)

def plot_distributions(data, out_path):
    plt.figure(figsize=(12, 6), facecolor="white")
    
    models = sorted(list(set([k[0] for k in data.keys()])))
    colors = plt.cm.tab10.colors
    
    for i, model in enumerate(models):
        cpu_lats = data.get((model, "cpu"), [])
        gpu_lats = data.get((model, "gpu"), [])
        
        if len(cpu_lats) > 0:
            plt.hist(cpu_lats, bins=30, alpha=0.5, color=colors[i % len(colors)], histtype='stepfilled', label=f"{model} (CPU)")
        if len(gpu_lats) > 0:
            plt.hist(gpu_lats, bins=30, alpha=0.8, color=colors[i % len(colors)], histtype='step', linewidth=2, linestyle='--', label=f"{model} (GPU)")

    plt.title("Latency Distributions: CPU vs GPU", fontsize=15, fontweight="bold")
    plt.xlabel("Latency (ms)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_pca(X, y, hw_labels, out_path):
    X_s = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_s)
    
    plt.figure(figsize=(10, 8), facecolor="white")
    
    models = sorted(list(set(y)))
    colors = plt.cm.tab10.colors
    markers = {"cpu": "o", "gpu": "X"}
    
    for i, model in enumerate(models):
        for hw in ["cpu", "gpu"]:
            idx = np.where((y == model) & (hw_labels == hw))[0]
            if len(idx) > 0:
                plt.scatter(X_pca[idx, 0], X_pca[idx, 1], 
                            label=f"{model} ({hw})", 
                            color=colors[i % len(colors)], 
                            marker=markers[hw],
                            alpha=0.7, edgecolors='k', s=60)
                
    plt.title("PCA Feature Space: Heterogeneous Hardware", fontsize=15, fontweight="bold")
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=12)
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_accuracy(results, out_path):
    classifiers = ["KNN", "Logistic Regression"]
    hws = ["cpu", "gpu", "mixed"]
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    
    bar_width = 0.25
    x = np.arange(len(classifiers))
    
    for i, hw in enumerate(hws):
        means = [r["mean_accuracy"] * 100 for r in results if r["hardware_type"] == hw]
        stds = [r["std_accuracy"] * 100 for r in results if r["hardware_type"] == hw]
        
        if len(means) == len(classifiers):
            ax.bar(x + (i - 1) * bar_width, means, bar_width, yerr=stds, capsize=5, label=hw.upper(), alpha=0.8)

    ax.set_title("Classification Accuracy: CPU vs GPU vs Mixed", fontsize=15, fontweight="bold")
    ax.set_xlabel("Classifier", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(classifiers)
    ax.legend(title="Hardware Dataset", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', linestyle="--", alpha=0.5)
    
    plt.ylim(0, 110)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def main():
    print("=" * 65)
    print("  EXPERIMENT 6 — HETEROGENEOUS HARDWARE COMPARISON")
    print("=" * 65)
    
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Results", "latency_results.csv"))
    
    if not os.path.exists(csv_path):
        print(f"[!] Error: Raw latency data not found at {csv_path}")
        return
        
    print(f"[OK] Loaded raw latency data from {csv_path}")
    data = load_csv(csv_path)
    
    X_all, y_all, hw_all = build_dataset(data)
    
    print(f"\n  Total samples : {len(X_all)}")
    
    subsets = {
        "cpu": (X_all[hw_all == "cpu"], y_all[hw_all == "cpu"]),
        "gpu": (X_all[hw_all == "gpu"], y_all[hw_all == "gpu"]),
        "mixed": (X_all, y_all)
    }
    
    classifiers = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    }
    
    results = []
    
    print("\n" + "-" * 75)
    print(f"  {'Hardware':>10}  {'Classifier':>20}  {'Mean Acc':>10}  {'Std Acc':>10}  {'Macro F1':>10}")
    print("-" * 75)
    
    for hw_name, (X_sub, y_sub) in subsets.items():
        if len(X_sub) == 0:
            print(f"  {hw_name:>10} dataset is empty (skipping).")
            continue
            
        unique, counts = np.unique(y_sub, return_counts=True)
        if len(counts) > 0 and min(counts) < 5:
            print(f"  {hw_name:>10} has insufficient samples for 5-fold CV (skipping).")
            continue
            
        for clf_name, clf in classifiers.items():
            mean_acc, std_acc, mean_f1 = evaluate_cv(clf, X_sub, y_sub)
            results.append({
                "hardware_type": hw_name,
                "classifier": clf_name,
                "mean_accuracy": round(mean_acc, 4),
                "std_accuracy": round(std_acc, 4),
                "macro_f1": round(mean_f1, 4)
            })
            print(f"  {hw_name:>10}  {clf_name:>20}  {mean_acc*100:>9.2f}%  {std_acc*100:>9.2f}%  {mean_f1:>10.4f}")
            
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Results"))
    
    csv_out = os.path.join(out_dir, "hardware_comparison_results.csv")
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["hardware_type", "classifier", "mean_accuracy", "std_accuracy", "macro_f1"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[OK] Results saved -> {csv_out}")
    
    dist_plot = os.path.join(out_dir, "cpu_vs_gpu_latency_distributions.png")
    plot_distributions(data, dist_plot)
    print(f"[OK] Plot saved -> {dist_plot}")
    
    pca_plot = os.path.join(out_dir, "hardware_pca_clusters.png")
    plot_pca(X_all, y_all, hw_all, pca_plot)
    print(f"[OK] Plot saved -> {pca_plot}")
    
    acc_plot = os.path.join(out_dir, "hardware_accuracy_comparison.png")
    plot_accuracy(results, acc_plot)
    print(f"[OK] Plot saved -> {acc_plot}")
    
    print("\n[OK] Experiment 6 complete.")

if __name__ == "__main__":
    main()
