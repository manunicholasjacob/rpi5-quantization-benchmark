#!/usr/bin/env python3
"""Get all INT8 tail-latency and accuracy data needed for paper v3."""
import csv, os, json

# Pi results CSV (from original experiments, 300 iters, 4 threads)
pi_csv = os.path.expanduser("~/Desktop/researchpaper3/pi_results.csv")
alt_csv = os.path.expanduser("~/pi_results.csv")

for cpath in [pi_csv, alt_csv]:
    if os.path.exists(cpath):
        rows = list(csv.DictReader(open(cpath)))
        print(f"=== {cpath} ({len(rows)} rows) ===")
        for r in rows:
            model = "MV3-S" if "mobilenet" in r.get("model","") else "R18"
            ds    = r.get("dataset","?")
            prec  = r.get("precision","?")
            batch = r.get("batch","?")
            threads = r.get("threads","?")
            p50  = float(r.get("p50_ms","0"))
            p90  = float(r.get("p90_ms","0"))
            mean = float(r.get("mean_ms","0"))
            ratio = p90/p50 if p50>0 else 0
            imgs  = r.get("img_s","?")
            print(f"  {model} {ds} {prec} b={batch} t={threads}: "
                  f"p50={p50:.3f} p90={p90:.3f} ratio={ratio:.3f} mean={mean:.3f} img/s={imgs}")
        break

# Also check if there's an int8 version in static_benchmark_results
static_csv = os.path.expanduser("~/paper_results/static_benchmark_results.csv")
if os.path.exists(static_csv):
    rows2 = list(csv.DictReader(open(static_csv)))
    int8_rows = [r for r in rows2 if r.get("precision","") == "int8"]
    fp32_rows = [r for r in rows2 if r.get("precision","") == "fp32"]
    print(f"\n=== static_benchmark_results.csv: {len(fp32_rows)} fp32, {len(int8_rows)} int8 rows ===")

# Get accuracy from confusion matrices
BASE = os.path.expanduser("~/Desktop/researchpaper3/exports")
configs = {
    "MV3-S_CIFAR100": "CIFAR100_mobilenet_v3_small_CIFAR100_mobilenet_v3_small_20251015_155545",
    "R18_CIFAR100":   "CIFAR100_resnet18_CIFAR100_resnet18_20251015_162556",
    "MV3-S_SVHN":     "SVHN_mobilenet_v3_small_SVHN_mobilenet_v3_small_20251015_145710",
    "R18_SVHN":       "SVHN_resnet18_SVHN_resnet18_20251015_153705",
}

print("\n=== Accuracy from confusion matrices ===")
import numpy as np
for key, dirname in configs.items():
    cm_path = os.path.join(BASE, dirname, "analysis", "confusion_matrix.csv")
    if os.path.exists(cm_path):
        try:
            cm_rows = []
            with open(cm_path) as f:
                for line in f:
                    cm_rows.append([int(x) for x in line.strip().split(",")])
            cm = np.array(cm_rows)
            if cm.shape[0] == cm.shape[1]:
                top1 = 100.0 * np.trace(cm) / cm.sum()
                print(f"  {key}: top-1 = {top1:.2f}%")
        except Exception as e:
            print(f"  {key}: ERROR {e}")

# Check if there's int8 confusion matrix too
print("\n=== Checking for INT8 accuracy ===")
for key, dirname in configs.items():
    path = os.path.join(BASE, dirname)
    int8_cm = os.path.join(path, "analysis", "confusion_matrix_int8.csv")
    models  = [f for f in os.listdir(path) if "int8" in f and f.endswith(".onnx")]
    print(f"  {key}: int8 models: {models}")
    print(f"    int8 cm exists: {os.path.exists(int8_cm)}")

# Calibration ablation latency (already gathered from pi_qdq_analysis.py)
print("\n=== Summary of calibration ablation ===")
qdq_json = os.path.expanduser("~/pi_qdq_results.json")
if os.path.exists(qdq_json):
    data = json.load(open(qdq_json))
    if "calibration_ablation" in data:
        for model, calib_data in data["calibration_ablation"].items():
            print(f"  {model}: {calib_data}")

print("\nDONE")
