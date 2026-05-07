"""
EXPERIMENT 1 — LATENCY MEASUREMENT + REPORT
============================================
Measures inference latency of multiple torchvision models on CPU.
Models: resnet18, resnet50, vgg16, inception_v3
Runs 100 inference passes per model and computes mean, std, and p95.
"""

import io
import sys
import time
import numpy as np
import torch
import torchvision.models as models

# Force UTF-8 output so Unicode chars work on Windows cp1252 terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Configuration ──────────────────────────────────────────────────────────────
MODELS = {
    "resnet18":    models.resnet18,
    "resnet50":    models.resnet50,
    "vgg16":       models.vgg16,
    "inception_v3": models.inception_v3,
}
NUM_RUNS   = 100
DEVICE     = torch.device("cpu")
INPUT_SIZE = (1, 3, 224, 224)
# ───────────────────────────────────────────────────────────────────────────────


def measure_latency(model: torch.nn.Module, input_tensor: torch.Tensor, runs: int) -> list[float]:
    """Run `runs` forward passes and return per-run latency in milliseconds."""
    model.eval()
    latencies = []

    with torch.no_grad():
        # Warm-up pass (not counted)
        _ = model(input_tensor)

        for _ in range(runs):
            start = time.perf_counter()
            _ = model(input_tensor)
            end   = time.perf_counter()
            latencies.append((end - start) * 1000)  # convert to ms

    return latencies


def print_report(name: str, latencies: list[float]) -> None:
    arr  = np.array(latencies)
    mean = arr.mean()
    std  = arr.std()
    p95  = np.percentile(arr, 95)
    pmin = arr.min()
    pmax = arr.max()

    print(f"\n{'─'*50}")
    print(f"  Model : {name}")
    print(f"{'─'*50}")
    print(f"  Runs  : {len(latencies)}")
    print(f"  Mean  : {mean:.3f} ms")
    print(f"  Std   : {std:.3f} ms")
    print(f"  P95   : {p95:.3f} ms")
    print(f"  Min   : {pmin:.3f} ms")
    print(f"  Max   : {pmax:.3f} ms")
    print(f"{'─'*50}")


def main():
    print("=" * 50)
    print("  EXPERIMENT 1 — LATENCY MEASUREMENT REPORT")
    print("=" * 50)
    print(f"  Device     : {DEVICE}")
    print(f"  Runs/model : {NUM_RUNS}")
    print(f"  Input size : {INPUT_SIZE}")

    # Standard input tensor for all models
    input_tensor = torch.randn(*INPUT_SIZE).to(DEVICE)

    all_results = {}

    for name, model_fn in MODELS.items():
        print(f"\n[-] Loading {name} ...")

        # inception_v3 requires aux_logits=False for simple forward pass
        if name == "inception_v3":
            model = model_fn(weights=None, aux_logits=False).to(DEVICE)
        else:
            model = model_fn(weights=None).to(DEVICE)

        print(f"[->] Running {NUM_RUNS} inference passes ...")
        latencies = measure_latency(model, input_tensor, NUM_RUNS)
        all_results[name] = latencies

        print_report(name, latencies)

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  {'Model':<15} {'Mean (ms)':>12} {'Std (ms)':>12} {'P95 (ms)':>12}")
    print(f"{'='*60}")
    for name, latencies in all_results.items():
        arr = np.array(latencies)
        print(f"  {name:<15} {arr.mean():>12.3f} {arr.std():>12.3f} {np.percentile(arr,95):>12.3f}")
    print(f"{'='*60}")

    # ── Save raw data ──────────────────────────────────────────────────────────
    import csv, os
    out_path = os.path.join(os.path.dirname(__file__), "latency_results.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "run", "latency_ms"])
        for name, latencies in all_results.items():
            for i, lat in enumerate(latencies):
                writer.writerow([name, i + 1, round(lat, 6)])
    print(f"\n[OK] Raw results saved -> {out_path}")


if __name__ == "__main__":
    main()
