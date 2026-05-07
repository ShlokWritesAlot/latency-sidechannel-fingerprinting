"""
EXPERIMENT 3 — MODEL CLASSIFICATION FROM LATENCY
==================================================
Trains KNN and Logistic Regression classifiers to identify which model
produced a given latency sample.

Pipeline:
  1. Load (or generate) raw per-run latency values from Experiment 1.
  2. Each sample  → one latency value (ms)
     Each label   → model name (resnet18, resnet50, vgg16, inception_v3)
  3. 80/20 train-test split
  4. Evaluate KNN and Logistic Regression
  5. Print accuracy + confusion matrix for both classifiers
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

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

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
NUM_RUNS   = 100
DEVICE     = torch.device("cpu")
INPUT_SIZE = (1, 3, 224, 224)
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
    """Generate latency data via live inference."""
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


def build_dataset(data: dict[str, list[float]]):
    """Convert dict → (X, y) arrays."""
    X, y = [], []
    for name, latencies in data.items():
        for lat in latencies:
            X.append([lat])
            y.append(name)
    return np.array(X), np.array(y)


# ── Reporting helpers ──────────────────────────────────────────────────────────

def print_confusion(cm: np.ndarray, labels: list[str], title: str) -> None:
    print(f"\n  Confusion Matrix — {title}")
    header = f"  {'':>15}" + "".join(f"{l:>14}" for l in labels)
    print(header)
    print("  " + "-" * (15 + 14 * len(labels)))
    for i, row_label in enumerate(labels):
        row = f"  {row_label:>15}" + "".join(f"{cm[i, j]:>14}" for j in range(len(labels)))
        print(row)


def evaluate(clf, X_train, X_test, y_train, y_test, label_names: list[str], clf_name: str) -> dict:
    """Fit classifier, print report, and return metrics dict for CSV export."""
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred, labels=label_names)

    print(f"  Classifier : {clf_name}")
    print("=" * 55)
    print(f"  Accuracy   : {acc * 100:.2f}%")
    print_confusion(cm, label_names, clf_name)
    print(f"\n  Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=label_names, zero_division=0))

    # Build per-class metrics dict for CSV
    from sklearn.metrics import precision_recall_fscore_support
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_test, y_pred, labels=label_names, zero_division=0
    )
    rows = []
    for i, lbl in enumerate(label_names):
        rows.append({
            "classifier": clf_name,
            "class": lbl,
            "precision": round(float(prec[i]), 4),
            "recall": round(float(rec[i]), 4),
            "f1_score": round(float(f1[i]), 4),
            "support": int(sup[i]),
            "overall_accuracy": "",
        })
    # Summary row
    rows.append({
        "classifier": clf_name,
        "class": "OVERALL",
        "precision": "",
        "recall": "",
        "f1_score": "",
        "support": len(y_test),
        "overall_accuracy": round(acc * 100, 2),
    })
    return rows


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  EXPERIMENT 3 — MODEL CLASSIFICATION FROM LATENCY")
    print("=" * 55)

    # 1. Collect data
    data = collect_data()

    # 2. Build dataset
    X, y = build_dataset(data)
    label_names = list(data.keys())

    print(f"\n  Dataset size : {len(X)} samples × {X.shape[1]} feature(s)")
    print(f"  Classes      : {label_names}")

    # 3. Encode labels
    le = LabelEncoder()
    le.fit(label_names)

    # 4. Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train samples: {len(X_train)}")
    print(f"  Test  samples: {len(X_test)}")

    # 5. Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # 6. KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn_rows = evaluate(knn, X_train_s, X_test_s, y_train, y_test, label_names, "KNN (k=5)")

    # 7. Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr_rows = evaluate(lr, X_train_s, X_test_s, y_train, y_test, label_names, "Logistic Regression")

    # 8. Per-model latency stats (for context)
    print("=" * 55)
    print("  Per-Model Latency Summary (context for classifier)")
    print("=" * 55)
    print(f"  {'Model':<16} {'Mean (ms)':>12} {'Std (ms)':>12}")
    print("  " + "-" * 42)
    latency_rows = []
    for name, latencies in data.items():
        arr = np.array(latencies)
        print(f"  {name:<16} {arr.mean():>12.3f} {arr.std():>12.3f}")
        latency_rows.append({
            "model": name,
            "mean_ms": round(float(arr.mean()), 4),
            "std_ms": round(float(arr.std()), 4),
            "min_ms": round(float(arr.min()), 4),
            "max_ms": round(float(arr.max()), 4),
            "p95_ms": round(float(np.percentile(arr, 95)), 4),
        })

    # 9. Save CSV report
    out_dir = os.path.dirname(__file__)

    # 9a. Classification metrics
    clf_csv = os.path.join(out_dir, "classification_report.csv")
    fieldnames = ["classifier", "class", "precision", "recall", "f1_score", "support", "overall_accuracy"]
    with open(clf_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(knn_rows + lr_rows)
    print(f"\n[OK] Classification report saved -> {clf_csv}")

    # 9b. Latency summary
    lat_csv = os.path.join(out_dir, "latency_summary.csv")
    lat_fields = ["model", "mean_ms", "std_ms", "min_ms", "max_ms", "p95_ms"]
    with open(lat_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=lat_fields)
        writer.writeheader()
        writer.writerows(latency_rows)
    print(f"[OK] Latency summary saved      -> {lat_csv}")

    print("\n[OK] Experiment 3 complete.")


if __name__ == "__main__":
    main()
