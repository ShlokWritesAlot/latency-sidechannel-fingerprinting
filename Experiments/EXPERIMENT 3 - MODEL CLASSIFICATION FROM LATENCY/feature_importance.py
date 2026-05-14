import os
import sys
import csv
import io
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Force UTF-8 output on Windows cp1252 terminals
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

FEATURE_NAMES = ["mean", "std", "min", "max", "p95", "skewness", "kurtosis", "lag1_autocorr"]

def load_features(path):
    X, y = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            y.append(row["model"])
            features = [float(row[feat]) for feat in FEATURE_NAMES]
            X.append(features)
    return np.array(X), np.array(y)

def main():
    print("=" * 65)
    print("  FEATURE IMPORTANCE ANALYSIS")
    print("=" * 65)

    csv_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "Results", "feature_vectors.csv"
    ))

    if not os.path.exists(csv_path):
        print(f"[!] Error: Feature vectors not found at {csv_path}")
        print("    Please run Experiment 3 first.")
        return

    X, y = load_features(csv_path)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_s, y)

    # For multi-class Logistic Regression, coef_ is (n_classes, n_features)
    # We take the mean absolute coefficient across all classes for an overall importance
    importance = np.mean(np.abs(clf.coef_), axis=0)
    
    # Normalize to sum to 1.0 (or just present raw, but normalized is nicer)
    importance = importance / np.sum(importance)

    # Sort
    indices = np.argsort(importance)[::-1]

    print("-" * 45)
    print(f"  {'Feature':<20}  {'Importance':>15}")
    print("-" * 45)

    for idx in indices:
        print(f"  {FEATURE_NAMES[idx]:<20}  {importance[idx]:>15.4f}")

    print("-" * 45)

    # Save to CSV
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Results"))
    out_csv = os.path.join(out_dir, "feature_importance.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Feature", "Importance"])
        for idx in indices:
            writer.writerow([FEATURE_NAMES[idx], round(float(importance[idx]), 4)])
            
    print(f"\n[OK] Feature importance saved -> {out_csv}")

if __name__ == "__main__":
    main()
