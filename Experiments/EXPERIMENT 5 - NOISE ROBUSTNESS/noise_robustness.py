"""
EXPERIMENT 5 — NOISE ROBUSTNESS
================================
Evaluates how latency-side-channel classification degrades under noisy 
timing conditions (timing jitter).
"""

import os
import csv
import sys
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy.stats import skew, kurtosis

# Force UTF-8 output on Windows cp1252 terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace") if hasattr(sys.stdout, 'buffer') else sys.stdout

# ── Configuration ──────────────────────────────────────────────────────────────
SIGMAS       = [0, 1, 5, 15, 30, 50]
WINDOW_SIZE  = 10
STRIDE       = WINDOW_SIZE
RANDOM_STATE = 42

def load_raw_latencies(path: str) -> dict[str, list[float]]:
    data = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("hardware_type", "cpu") != "cpu":
                continue
            data.setdefault(row["model"], []).append(float(row["latency_ms"]))
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

def apply_noise_and_extract_features(data: dict[str, list[float]], sigma: float):
    np.random.seed(RANDOM_STATE) # Reproducibility per sigma run
    X, y = [], []
    for name, latencies in data.items():
        arr = np.array(latencies)
        
        # Add Gaussian noise and clip
        noise = np.random.normal(0, sigma, size=len(arr))
        noisy_arr = np.clip(arr + noise, a_min=0.0, a_max=None)
        
        n_samples = len(noisy_arr)
        for i in range(0, n_samples - WINDOW_SIZE + 1, STRIDE):
            window = noisy_arr[i : i + WINDOW_SIZE]
            features = extract_features(window)
            X.append(features)
            y.append(name)
            
    return np.array(X), np.array(y)

def evaluate_classifier(clf, X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    acc_list = []
    f1_list = []
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)
        
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)
        
        acc_list.append(accuracy_score(y_test, y_pred))
        _, _, f, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
        f1_list.append(f)
        
    return np.mean(acc_list), np.std(acc_list), np.mean(f1_list)

def plot_robustness(results: list[dict], out_path: str):
    sigmas = sorted(list(set(r["sigma"] for r in results)))
    
    knn_accs = [r["mean_accuracy"] * 100 for r in results if r["classifier"] == "KNN"]
    knn_stds = [r["std_accuracy"] * 100 for r in results if r["classifier"] == "KNN"]
    
    lr_accs  = [r["mean_accuracy"] * 100 for r in results if r["classifier"] == "Logistic Regression"]
    lr_stds  = [r["std_accuracy"] * 100 for r in results if r["classifier"] == "Logistic Regression"]

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(9, 5))
    
    knn_accs = np.array(knn_accs)
    knn_stds = np.array(knn_stds)
    lr_accs = np.array(lr_accs)
    lr_stds = np.array(lr_stds)
    
    ax.plot(sigmas, knn_accs, color="#1f77b4", linewidth=2.0, marker="o", label="KNN Accuracy")
    ax.fill_between(sigmas, knn_accs - knn_stds, knn_accs + knn_stds, color="#1f77b4", alpha=0.2)
    
    ax.plot(sigmas, lr_accs, color="#ff7f0e", linewidth=2.0, marker="s", label="LR Accuracy")
    ax.fill_between(sigmas, lr_accs - lr_stds, lr_accs + lr_stds, color="#ff7f0e", alpha=0.2)

    ax.set_xlabel("Noise Sigma (ms)", fontsize=12)
    ax.set_ylabel("Classification Accuracy (%)", fontsize=12)
    ax.set_title("Robustness of Architecture Fingerprinting to Timing Noise", fontsize=14, fontweight="bold", pad=14)
    
    ax.set_xticks(sigmas)
    ax.set_ylim(max(0, min(min(knn_accs - knn_stds), min(lr_accs - lr_stds)) - 5), 105)
    ax.grid(color="gray", linewidth=0.5, linestyle="--", alpha=0.5, zorder=1)
    ax.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Plot saved -> {out_path}")

def main():
    print("=" * 75)
    print("  EXPERIMENT 5 — NOISE ROBUSTNESS")
    print("=" * 75)
    
    import io
    # For robust stdout handling
    
    raw_csv_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "Results", "latency_results.csv"
    ))
    
    if not os.path.exists(raw_csv_path):
        print(f"[!] Error: Raw latency data not found at {raw_csv_path}")
        return
        
    print(f"[OK] Loaded raw latency data from {raw_csv_path}")
    raw_data = load_raw_latencies(raw_csv_path)
    
    results = []
    
    print("-" * 75)
    print(f"  {'Sigma (ms)':>10}  {'Classifier':>20}  {'Mean Acc':>10}  {'Std Acc':>10}  {'Macro F1':>10}")
    print("-" * 75)

    knn = KNeighborsClassifier(n_neighbors=5)
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    
    for sigma in SIGMAS:
        X, y = apply_noise_and_extract_features(raw_data, sigma)
        
        # Evaluate KNN
        mean_acc, std_acc, mean_f1 = evaluate_classifier(knn, X, y)
        results.append({
            "sigma": sigma,
            "classifier": "KNN",
            "mean_accuracy": round(mean_acc, 4),
            "std_accuracy": round(std_acc, 4),
            "macro_f1": round(mean_f1, 4)
        })
        print(f"  {sigma:>10}  {'KNN':>20}  {mean_acc*100:>9.2f}%  {std_acc*100:>9.2f}%  {mean_f1:>10.4f}")
        
        # Evaluate LR
        mean_acc, std_acc, mean_f1 = evaluate_classifier(lr, X, y)
        results.append({
            "sigma": sigma,
            "classifier": "Logistic Regression",
            "mean_accuracy": round(mean_acc, 4),
            "std_accuracy": round(std_acc, 4),
            "macro_f1": round(mean_f1, 4)
        })
        print(f"  {sigma:>10}  {'Logistic Regression':>20}  {mean_acc*100:>9.2f}%  {std_acc*100:>9.2f}%  {mean_f1:>10.4f}")

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Results"))
    
    csv_path = os.path.join(out_dir, "noise_robustness.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sigma", "classifier", "mean_accuracy", "std_accuracy", "macro_f1"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[OK] Results saved -> {csv_path}")
    
    plot_path = os.path.join(out_dir, "noise_robustness_plot.png")
    plot_robustness(results, plot_path)
    
    print("\n[OK] Experiment 5 complete.")

if __name__ == "__main__":
    main()
