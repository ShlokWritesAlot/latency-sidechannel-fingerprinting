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
from scipy.stats import skew, kurtosis

# Force UTF-8 output so Unicode chars work on Windows cp1252 terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Reproducibility and CPU thread stabilization
torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(1)

# ── Configuration ──────────────────────────────────────────────────────────────
MODELS = {
    "resnet18":    models.resnet18,
    "resnet50":    models.resnet50,
    "vgg16":       models.vgg16,
    "inception_v3": models.inception_v3,
    "mobilenet_v2": models.mobilenet_v2,
}
NUM_RUNS   = 100
AVAILABLE_DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
INPUT_SIZE = (1, 3, 224, 224)
# ───────────────────────────────────────────────────────────────────────────────


def measure_latency(model: torch.nn.Module, input_tensor: torch.Tensor, runs: int) -> list[float]:
    """Run `runs` forward passes and return per-run latency in milliseconds."""
    model.eval()
    latencies = []

    with torch.no_grad():
        # Warm-up pass (not counted)
        for _ in range(3):
            _ = model(input_tensor)

        for _ in range(runs):
            if str(model.device_str) == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(input_tensor)
            if str(model.device_str) == "cuda":
                torch.cuda.synchronize()
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
    print(f"  Runs/model : {NUM_RUNS}")
    print(f"  Input size : {INPUT_SIZE}")

    all_results = {}

    for device_str in AVAILABLE_DEVICES:
        device = torch.device(device_str)
        hw_type = "gpu" if device_str == "cuda" else "cpu"
        print(f"\n[{hw_type.upper()}] Running measurements on {device}...")
        
        input_tensor = torch.randn(*INPUT_SIZE).to(device)

        for name, model_fn in MODELS.items():
            print(f"\n[-] Loading {name} ...")

            if name == "inception_v3":
                model = model_fn(weights=None, aux_logits=False).to(device)
            else:
                model = model_fn(weights=None).to(device)
            
            # Monkey-patch device_str for the timing function
            model.device_str = device_str

            print(f"[->] Running {NUM_RUNS} inference passes ...")
            latencies = measure_latency(model, input_tensor, NUM_RUNS)
            all_results[(name, hw_type)] = latencies

            print_report(f"{name} ({hw_type})", latencies)

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"  {'Model':<15} {'HW':<5} {'Mean (ms)':>12} {'Std (ms)':>12} {'P95 (ms)':>12}")
    print(f"{'='*75}")
    for (name, hw), latencies in all_results.items():
        arr = np.array(latencies)
        print(f"  {name:<15} {hw:<5} {arr.mean():>12.3f} {arr.std():>12.3f} {np.percentile(arr,95):>12.3f}")
    print(f"{'='*75}")

    # ── Save raw data ──────────────────────────────────────────────────────────
    import csv, os
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Results"))
    os.makedirs(out_dir, exist_ok=True)
    
    out_path = os.path.join(out_dir, "latency_results.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "hardware_type", "run", "latency_ms"])
        for (name, hw), latencies in all_results.items():
            for i, lat in enumerate(latencies):
                writer.writerow([name, hw, i + 1, round(lat, 6)])
    print(f"\n[OK] Raw results saved -> {out_path}")

    # ── Save extended summary ──────────────────────────────────────────────────
    summary_path = os.path.join(out_dir, "latency_summary_extended.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "hardware_type", "mean_ms", "std_ms", "min_ms", "max_ms", "p95_ms", "skewness", "kurtosis", "lag1_autocorr"])
        for (name, hw), latencies in all_results.items():
            arr = np.array(latencies)
            mean_ms = arr.mean()
            std_ms = arr.std()
            min_ms = arr.min()
            max_ms = arr.max()
            p95_ms = np.percentile(arr, 95)
            skew_val = skew(arr)
            kurt_val = kurtosis(arr)
            if len(arr) > 1:
                lag1 = np.corrcoef(arr[:-1], arr[1:])[0, 1]
            else:
                lag1 = 0.0
            writer.writerow([
                name, hw,
                round(mean_ms, 6), round(std_ms, 6), 
                round(min_ms, 6), round(max_ms, 6), round(p95_ms, 6),
                round(skew_val, 6), round(kurt_val, 6), round(lag1, 6)
            ])
    print(f"[OK] Extended summary saved -> {summary_path}")


if __name__ == "__main__":
    main()
