"""
EXPERIMENT 3 — MODEL CLASSIFICATION FROM LATENCY (UPGRADED)
============================================================
Trains KNN and Logistic Regression classifiers to identify which model
produced a given latency sample, using statistical feature vectors extracted
from latency windows.

Pipeline:
  1. Load raw per-run latency values from Experiment 1.
  2. Sliding window feature extraction (mean, std, min, max, p95, skew, kurtosis, lag1).
  3. Stratified 5-Fold Cross Validation.
  4. Evaluate KNN and Logistic Regression (mean acc, std acc, macro-F1).
  5. Print averaged confusion matrix and classification report.
  6. PCA Visualization of feature space.
"""

import io
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

from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import skew, kurtosis

warnings.filterwarnings("ignore")

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
NUM_RUNS     = 100
DEVICE       = torch.device("cpu")
INPUT_SIZE   = (1, 3, 224, 224)
RANDOM_STATE = 42

WINDOW_SIZE  = 10
STRIDE       = WINDOW_SIZE
FEATURE_NAMES = ["mean", "std", "min", "max", "p95", "skewness", "kurtosis", "lag1_autocorr"]
# ───────────────────────────────────────────────────────────────────────────────

def load_csv(path: str) -> dict[str, list[float]]:
    data: dict[str, list[float]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("hardware_type", "cpu") != "cpu":
                continue
            data.setdefault(row["model"], []).append(float(row["latency_ms"]))
    return data

def run_inference(runs: int) -> dict[str, list[float]]:
    """Generate latency data via live inference if CSV is missing."""
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
            _ = model(input_tensor)                          # warm-up
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
    print("[!] CSV not found — running inference ...")
    return run_inference(NUM_RUNS)

def extract_features(window: np.ndarray) -> list[float]:
    """Extract 8 statistical features from a window of latencies."""
    mean_val = window.mean()
    std_val = window.std()
    min_val = window.min()
    max_val = window.max()
    p95_val = np.percentile(window, 95)
    skew_val = skew(window)
    kurt_val = kurtosis(window)
    
    if len(window) > 1 and std_val > 1e-6:
        lag1 = np.corrcoef(window[:-1], window[1:])[0, 1]
        if np.isnan(lag1): lag1 = 0.0
    else:
        lag1 = 0.0

    return [mean_val, std_val, min_val, max_val, p95_val, skew_val, kurt_val, lag1]

def build_dataset(data: dict[str, list[float]], out_csv_path: str):
    """Convert dict → windowed (X, y) arrays and save feature vectors."""
    X, y = [], []
    
    with open(out_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + FEATURE_NAMES)
        
        for name, latencies in data.items():
            arr = np.array(latencies)
            n_samples = len(arr)
            
            for i in range(0, n_samples - WINDOW_SIZE + 1, STRIDE):
                window = arr[i : i + WINDOW_SIZE]
                features = extract_features(window)
                X.append(features)
                y.append(name)
                
                # save to CSV
                writer.writerow([name] + [round(val, 6) for val in features])
                
    return np.array(X), np.array(y)

def print_confusion(cm: np.ndarray, labels: list[str], title: str) -> None:
    print(f"\n  Averaged Confusion Matrix — {title}")
    header = f"  {'':>15}" + "".join(f"{l:>14}" for l in labels)
    print(header)
    print("  " + "-" * (15 + 14 * len(labels)))
    for i, row_label in enumerate(labels):
        row = f"  {row_label:>15}" + "".join(f"{cm[i, j]:>14.1f}" for j in range(len(labels)))
        print(row)

def evaluate_cv(clf, X, y, label_names: list[str], clf_name: str) -> dict:
    """Evaluate using 5-Fold Stratified CV."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    acc_list = []
    prec_list = []
    rec_list = []
    f1_list = []
    
    # Accumulate confusion matrix
    total_cm = np.zeros((len(label_names), len(label_names)))
    
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Scale features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)
        
        acc_list.append(accuracy_score(y_test, y_pred))
        
        # macro metrics
        p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
        prec_list.append(p)
        rec_list.append(r)
        f1_list.append(f)
        
        cm = confusion_matrix(y_test, y_pred, labels=label_names)
        total_cm += cm
        
    avg_cm = total_cm / 5.0
    
    mean_acc = np.mean(acc_list)
    std_acc = np.std(acc_list)
    mean_prec = np.mean(prec_list)
    mean_rec = np.mean(rec_list)
    mean_f1 = np.mean(f1_list)
    
    print(f"\n{'='*65}")
    print(f"  Classifier : {clf_name}")
    print("=" * 65)
    print(f"  Accuracy   : {mean_acc * 100:.2f}% (± {std_acc * 100:.2f}%)")
    print(f"  Macro F1   : {mean_f1:.4f}")
    
    print_confusion(avg_cm, label_names, clf_name)
    
    print(f"\n  Macro Classification Report (Averaged over 5 Folds):")
    print(f"  Precision : {mean_prec:.4f}")
    print(f"  Recall    : {mean_rec:.4f}")
    print(f"  F1-Score  : {mean_f1:.4f}")
    
    # Output format for CSV
    return {
        "classifier": clf_name,
        "mean_accuracy": round(mean_acc, 4),
        "std_accuracy": round(std_acc, 4),
        "macro_precision": round(mean_prec, 4),
        "macro_recall": round(mean_rec, 4),
        "macro_f1": round(mean_f1, 4)
    }

def plot_pca(X, y, label_names, out_path):
    # Scale entirely for PCA visualization
    X_s = StandardScaler().fit_transform(X)
    
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_s)
    
    plt.figure(figsize=(10, 8), facecolor="white")
    
    colors = plt.cm.tab10.colors
    for i, label in enumerate(label_names):
        idx = np.where(y == label)
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=label, color=colors[i % len(colors)], alpha=0.7, edgecolors='k')
        
    plt.title("PCA of Latency Feature Vectors", fontsize=15, fontweight="bold")
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=12)
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=12)
    plt.legend(title="Architecture")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n[OK] PCA Visualization saved -> {out_path}")


def main():
    print("=" * 65)
    print("  EXPERIMENT 3 — MODEL CLASSIFICATION FROM LATENCY (UPGRADED)")
    print("=" * 65)
    
    # Metadata logging
    print(f"[INFO] Using NON-OVERLAPPING windows")
    print(f"[INFO] Window Size = {WINDOW_SIZE}")
    print(f"[INFO] Effective Stride = {STRIDE}")
    print(f"  Features    : {', '.join(FEATURE_NAMES)}")
    
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Results"))
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Collect data
    data = collect_data()

    # 2. Build dataset
    features_csv = os.path.join(out_dir, "feature_vectors.csv")
    X, y = build_dataset(data, features_csv)
    label_names = list(MODELS_CONFIG.keys()) # Ensure consistent ordering

    print(f"\n[OK] Feature vectors saved -> {features_csv}")
    print(f"\n  Dataset size : {len(X)} samples × {X.shape[1]} features")
    print(f"  Classes      : {label_names}")

    # 3. Stratified CV Evaluation
    cv_results = []
    
    knn = KNeighborsClassifier(n_neighbors=5)
    cv_results.append(evaluate_cv(knn, X, y, label_names, "KNN (k=5)"))
    
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    cv_results.append(evaluate_cv(lr, X, y, label_names, "Logistic Regression"))
    
    # 4. Save CV results CSV
    clf_csv = os.path.join(out_dir, "classification_cv_results.csv")
    fieldnames = ["classifier", "mean_accuracy", "std_accuracy", "macro_precision", "macro_recall", "macro_f1"]
    with open(clf_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cv_results)
    print(f"\n[OK] Classification CV report saved -> {clf_csv}")
    
    # 5. PCA Visualization
    pca_img = os.path.join(out_dir, "pca_architecture_clusters.png")
    plot_pca(X, y, label_names, pca_img)

    print("\n[OK] Experiment 3 complete.")

if __name__ == "__main__":
    main()
