#!/usr/bin/env python3
"""
Verify Q/DQ node counts, accuracy numbers, and calibration ablation.
Run on Raspberry Pi 5.
"""
import os, json, csv, time
import numpy as np

BASE = os.path.expanduser("~/Desktop/researchpaper3/exports")
OUT  = os.path.expanduser("~/pi_qdq_results.json")
results = {}

# ─── 1. Q/DQ node counts from ONNX graphs ────────────────────────────────────
try:
    import onnx
    model_files = {
        "mv3s_int8":  os.path.join(BASE, "CIFAR100_mobilenet_v3_small_CIFAR100_mobilenet_v3_small_20251015_155545/model_int8.onnx"),
        "r18_int8":   os.path.join(BASE, "CIFAR100_resnet18_CIFAR100_resnet18_20251015_162556/model_int8.onnx"),
        "mv3s_fp32":  os.path.join(BASE, "CIFAR100_mobilenet_v3_small_CIFAR100_mobilenet_v3_small_20251015_155545/model_fp32.onnx"),
        "r18_fp32":   os.path.join(BASE, "CIFAR100_resnet18_CIFAR100_resnet18_20251015_162556/model_fp32.onnx"),
    }
    qdq_counts = {}
    for name, path in model_files.items():
        if os.path.exists(path):
            m = onnx.load(path)
            ops = [n.op_type for n in m.graph.node]
            q_count  = ops.count("QuantizeLinear")
            dq_count = ops.count("DequantizeLinear")
            total    = len(ops)
            qdq_counts[name] = {
                "QuantizeLinear": q_count,
                "DequantizeLinear": dq_count,
                "qdq_pairs": min(q_count, dq_count),
                "total_ops": total,
                "op_types": list(set(ops)),
            }
            print(f"{name}: Q={q_count} DQ={dq_count} pairs={min(q_count,dq_count)} total={total}")
        else:
            print(f"MISSING: {path}")
    results["qdq_counts"] = qdq_counts
except ImportError:
    print("onnx not installed, skipping Q/DQ analysis")
    results["qdq_counts"] = {"error": "onnx not installed"}

# ─── 2. Accuracy from per_class_accuracy.csv ─────────────────────────────────
# These are per-class accuracy values; average = mean per-class accuracy
# Top-1 accuracy = fraction of test samples classified correctly
# We need to compute top-1 from confusion_matrix.csv instead
acc_data = {}
export_dirs = {
    "mv3s_cifar100": "CIFAR100_mobilenet_v3_small_CIFAR100_mobilenet_v3_small_20251015_155545",
    "r18_cifar100":  "CIFAR100_resnet18_CIFAR100_resnet18_20251015_162556",
    "mv3s_svhn":     "SVHN_mobilenet_v3_small_SVHN_mobilenet_v3_small_20251015_145710",
    "r18_svhn":      "SVHN_resnet18_SVHN_resnet18_20251015_153705",
}

for key, dirname in export_dirs.items():
    path = os.path.join(BASE, dirname)
    cm_file = os.path.join(path, "analysis", "confusion_matrix.csv")
    pca_file = os.path.join(path, "analysis", "per_class_accuracy.csv")
    
    entry = {"key": key}
    
    # Per-class accuracy (fp32 only - check if precision is tracked)
    if os.path.exists(pca_file):
        accs = []
        with open(pca_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                accs.append(float(row['acc']))
        # Mean per-class accuracy
        entry["mean_per_class_acc_pct"] = float(np.mean(accs))
        # Compute macro top-1: since each class has equal weight in per_class_acc
        # this IS the balanced accuracy. For CIFAR-100 this is close to top-1 for balanced datasets
        entry["n_classes"] = len(accs)
        print(f"{key}: mean_per_class={np.mean(accs):.2f}% n_classes={len(accs)}")
    
    # Try to get top-1 from confusion matrix (diagonal sum / total)
    if os.path.exists(cm_file):
        try:
            cm_rows = []
            with open(cm_file) as f:
                reader = csv.reader(f)
                for row in reader:
                    cm_rows.append([int(x) for x in row])
            cm = np.array(cm_rows)
            if cm.shape[0] == cm.shape[1]:  # square matrix
                top1 = 100.0 * np.trace(cm) / cm.sum()
                entry["top1_acc_pct"] = float(top1)
                entry["cm_shape"] = cm.shape[0]
                print(f"{key}: top-1 from CM = {top1:.2f}%")
        except Exception as e:
            print(f"{key}: CM error: {e}")
    
    acc_data[key] = entry
results["accuracy"] = acc_data

# ─── 3. Calibration ablation (128 vs 512 vs 1024 samples) ────────────────────
# We have model_int8_calib128.onnx, model_int8_calib512.onnx, model_int8_calib1024.onnx
# Run benchmark for each

try:
    import onnxruntime as ort
    
    calib_results = {}
    test_input = np.random.randn(1, 3, 32, 32).astype(np.float32)
    
    calib_dirs = ["CIFAR100_mobilenet_v3_small_CIFAR100_mobilenet_v3_small_20251015_155545",
                  "CIFAR100_resnet18_CIFAR100_resnet18_20251015_162556"]
    
    for dirname in calib_dirs:
        model_key = "mv3s" if "mobilenet" in dirname else "r18"
        path = os.path.join(BASE, dirname)
        calib_entry = {}
        
        for n in [128, 512, 1024]:
            model_path = os.path.join(path, f"model_int8_calib{n}.onnx")
            if os.path.exists(model_path):
                # Benchmark 50 iters
                sess = ort.InferenceSession(model_path,
                    providers=["CPUExecutionProvider"])
                input_name = sess.get_inputs()[0].name
                # warmup
                for _ in range(10):
                    sess.run(None, {input_name: test_input})
                # measure
                times = []
                for _ in range(50):
                    t0 = time.perf_counter()
                    sess.run(None, {input_name: test_input})
                    times.append((time.perf_counter()-t0)*1000)
                p50 = float(np.percentile(times, 50))
                p90 = float(np.percentile(times, 90))
                calib_entry[f"calib{n}"] = {"p50_ms": p50, "p90_ms": p90}
                print(f"{model_key} calib{n}: p50={p50:.3f}ms p90={p90:.3f}ms")
        
        # Also get baseline int8
        base_path = os.path.join(path, "model_int8.onnx")
        if os.path.exists(base_path):
            sess = ort.InferenceSession(base_path, providers=["CPUExecutionProvider"])
            input_name = sess.get_inputs()[0].name
            for _ in range(10):
                sess.run(None, {input_name: test_input})
            times = []
            for _ in range(50):
                t0 = time.perf_counter()
                sess.run(None, {input_name: test_input})
                times.append((time.perf_counter()-t0)*1000)
            p50 = float(np.percentile(times, 50))
            calib_entry["base_int8"] = {"p50_ms": p50}
            print(f"{model_key} base_int8: p50={p50:.3f}ms")
        
        calib_results[model_key] = calib_entry
    
    results["calibration_ablation"] = calib_results
    
except ImportError as e:
    print(f"onnxruntime not available: {e}")
    results["calibration_ablation"] = {"error": str(e)}

# ─── Save ─────────────────────────────────────────────────────────────────────
with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUT}")
print("DONE")
