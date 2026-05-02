#!/usr/bin/env python3
"""
Comprehensive analysis script for paper upgrade.
Runs on Raspberry Pi 5 to gather all needed data.
"""
import os
import json
import csv
import numpy as np
from scipy import stats
import time

BASE = os.path.expanduser("~/Desktop/researchpaper3/exports")
PAPER_RESULTS = os.path.expanduser("~/paper_results")
OUT = os.path.expanduser("~/paper_comprehensive_results.json")

results = {}

# ─── 1. Accuracy from per_class_accuracy.csv ────────────────────────────────
accuracy_data = {}
export_dirs = {
    "mv3s_cifar100": "CIFAR100_mobilenet_v3_small_CIFAR100_mobilenet_v3_small_20251015_155545",
    "r18_cifar100":  "CIFAR100_resnet18_CIFAR100_resnet18_20251015_162556",
    "mv3s_svhn":     "SVHN_mobilenet_v3_small_SVHN_mobilenet_v3_small_20251015_145710",
    "r18_svhn":      "SVHN_resnet18_SVHN_resnet18_20251015_153705",
}

for key, dirname in export_dirs.items():
    path = os.path.join(BASE, dirname)
    # read timing csv for accuracy proxy - check if accuracy file exists
    acc_file = os.path.join(path, "analysis", "per_class_accuracy.csv")
    timing_file = os.path.join(path, "timing.csv")
    
    if os.path.exists(acc_file):
        accs = []
        with open(acc_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                accs.append(float(row['acc']))
        # This is per-class accuracy, average = top-1 accuracy
        accuracy_data[key] = {"mean_per_class_acc": float(np.mean(accs)), "n_classes": len(accs)}
        print(f"{key}: mean per-class acc = {np.mean(accs):.2f}%")
    else:
        accuracy_data[key] = {"mean_per_class_acc": None}
        print(f"{key}: no accuracy file found")

results["accuracy"] = accuracy_data

# ─── 2. Calibration ablation ─────────────────────────────────────────────────
# Look for int8 models with different calibration sizes
calib_data = {}
for key, dirname in export_dirs.items():
    path = os.path.join(BASE, dirname)
    calib_variants = {}
    for n in [128, 512, 1024]:
        model_path = os.path.join(path, f"model_int8_calib{n}.onnx")
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / 1e6
            calib_variants[n] = {"size_mb": size_mb, "exists": True}
            print(f"{key} calib{n}: {size_mb:.2f} MB")
        else:
            calib_variants[n] = {"exists": False}
    calib_data[key] = calib_variants
results["calibration"] = calib_data

# ─── 3. Model sizes ───────────────────────────────────────────────────────────
model_sizes = {}
for key, dirname in export_dirs.items():
    path = os.path.join(BASE, dirname)
    sizes = {}
    for prec in ["fp32", "int8"]:
        model_path = os.path.join(path, f"model_{prec}.onnx")
        if os.path.exists(model_path):
            sizes[prec] = os.path.getsize(model_path) / 1e6
    if "fp32" in sizes and "int8" in sizes:
        sizes["ratio"] = sizes["fp32"] / sizes["int8"]
    model_sizes[key] = sizes
    print(f"{key}: FP32={sizes.get('fp32','?'):.2f}MB INT8={sizes.get('int8','?'):.2f}MB ratio={sizes.get('ratio','?'):.2f}x")
results["model_sizes"] = model_sizes

# ─── 4. Thread-scaling analysis from pi_results.csv ─────────────────────────
pi_csv = os.path.join(BASE, "../../../paper_results/static_benchmark_results.csv")
# Also check direct path
alt_csv = os.path.expanduser("~/paper_results/static_benchmark_results.csv")
if os.path.exists(alt_csv):
    pi_csv = alt_csv

thread_data = []
if os.path.exists(pi_csv):
    with open(pi_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            thread_data.append({
                "model": row["model"],
                "batch": int(row["batch"]),
                "threads": int(row["threads"]),
                "precision": row["precision"],
                "p50_ms": float(row["p50_latency_ms"]),
                "p90_ms": float(row["p90_latency_ms"]),
                "cv_pct": float(row["cv_percent"]),
                "temp_rise": float(row["temp_rise_c"]),
            })
    print(f"Loaded {len(thread_data)} rows from thread-scaling data")
    results["thread_scaling"] = thread_data
else:
    print(f"WARNING: thread CSV not found at {pi_csv}")

# ─── 5. Main pi_results latency data ─────────────────────────────────────────
main_csv = os.path.expanduser("~/paper_results/static_benchmark_results.csv")
# Also try the timing.csv from all_timing
all_timing = os.path.join(BASE, "all_timing.csv")
if os.path.exists(all_timing):
    timing_rows = []
    with open(all_timing) as f:
        reader = csv.DictReader(f)
        for row in reader:
            timing_rows.append(dict(row))
    results["all_timing"] = timing_rows[:50]  # first 50
    print(f"Loaded {len(timing_rows)} rows from all_timing.csv")

# ─── 6. Statistical analysis on main pi_results.csv ─────────────────────────
# Load the original pi_results.csv
orig_csv = os.path.expanduser("~/paper_results/../Desktop/researchpaper3/pi_results.csv")
# Try alternate locations
for cpath in [
    os.path.expanduser("~/Desktop/researchpaper3/pi_results.csv"),
    os.path.expanduser("~/pi_results.csv"),
]:
    if os.path.exists(cpath):
        orig_csv = cpath
        break

# ─── 7. Batch scaling analysis from pi_results.csv (our main data) ───────────
# We already loaded this - use the thread_data for batch scaling too
if thread_data:
    # Group by model, precision, threads=4 for batch scaling
    batch_scaling = {}
    for row in thread_data:
        if row["threads"] == 4:
            key = f"{row['model']}_{row['precision']}"
            if key not in batch_scaling:
                batch_scaling[key] = []
            batch_scaling[key].append({"batch": row["batch"], "p50_ms": row["p50_ms"], "p90_ms": row["p90_ms"]})
    results["batch_scaling_threads4"] = batch_scaling

    # Thread scaling (batch=1)
    thread_scaling = {}
    for row in thread_data:
        if row["batch"] == 1:
            key = f"{row['model']}_{row['precision']}"
            if key not in thread_scaling:
                thread_scaling[key] = []
            thread_scaling[key].append({"threads": row["threads"], "p50_ms": row["p50_ms"], "cv_pct": row["cv_pct"]})
    results["thread_scaling_batch1"] = thread_scaling

# ─── 8. INT8 vs FP32 batch=1 threads=4 speedup stats ────────────────────────
speedups = {}
if thread_data:
    for model in ["mobilenet_v3_small", "resnet18"]:
        fp32_row = next((r for r in thread_data if r["model"]==model and r["precision"]=="fp32" and r["batch"]==1 and r["threads"]==4), None)
        int8_row = next((r for r in thread_data if r["model"]==model and r["precision"]=="int8" and r["batch"]==1 and r["threads"]==4), None)
        if fp32_row and int8_row:
            speedup = fp32_row["p50_ms"] / int8_row["p50_ms"]
            p90_ratio = int8_row["p90_ms"] / int8_row["p50_ms"]
            speedups[model] = {
                "fp32_p50": fp32_row["p50_ms"],
                "int8_p50": int8_row["p50_ms"],
                "speedup": speedup,
                "int8_p90_p50_ratio": p90_ratio,
                "fp32_temp_rise": fp32_row["temp_rise"],
                "int8_temp_rise": int8_row["temp_rise"],
            }
            print(f"{model}: speedup={speedup:.2f}x, P90/P50={p90_ratio:.2f}")
results["speedups_batch1_t4"] = speedups

# ─── 9. Thermal analysis ─────────────────────────────────────────────────────
thermal_csv = os.path.expanduser("~/paper_results/thermal_profile.csv")
if os.path.exists(thermal_csv):
    thermal_rows = []
    with open(thermal_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            thermal_rows.append(dict(row))
    results["thermal"] = thermal_rows[:20]
    print(f"Loaded {len(thermal_rows)} thermal rows")

# ─── Save all results ─────────────────────────────────────────────────────────
with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {OUT}")
print("DONE")
