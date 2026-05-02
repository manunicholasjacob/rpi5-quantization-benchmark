#!/usr/bin/env python3
"""Analyze actual mechanism of R18 INT8 tail-latency inflation."""
import os, json
import numpy as np

BASE = os.path.expanduser("~/Desktop/researchpaper3/exports")

# ── 1. Detailed ONNX graph analysis ──────────────────────────────────────────
import onnx
from collections import Counter, OrderedDict

results = {}

configs = {
    "mv3s_fp32": BASE + "/CIFAR100_mobilenet_v3_small_CIFAR100_mobilenet_v3_small_20251015_155545/model_fp32.onnx",
    "mv3s_int8": BASE + "/CIFAR100_mobilenet_v3_small_CIFAR100_mobilenet_v3_small_20251015_155545/model_int8.onnx",
    "r18_fp32":  BASE + "/CIFAR100_resnet18_CIFAR100_resnet18_20251015_162556/model_fp32.onnx",
    "r18_int8":  BASE + "/CIFAR100_resnet18_CIFAR100_resnet18_20251015_162556/model_int8.onnx",
}

for name, path in configs.items():
    if not os.path.exists(path):
        print(f"MISSING: {path}")
        continue
    m = onnx.load(path)
    ops = [n.op_type for n in m.graph.node]
    cnt = Counter(ops)
    
    # Key distinction: QLinearConv = fused int8 conv (good, fast)
    # QuantizeLinear/DequantizeLinear = explicit cast nodes (potential overhead at boundaries)
    q_nodes   = cnt.get("QuantizeLinear", 0)
    dq_nodes  = cnt.get("DequantizeLinear", 0)
    ql_conv   = cnt.get("QLinearConv", 0)
    ql_mat    = cnt.get("QLinearMatMul", 0)
    ql_add    = cnt.get("QLinearAdd", 0)
    conv_fp   = cnt.get("Conv", 0)
    
    # Compute initializer (weight tensor) sizes to estimate memory pressure
    total_params = sum(
        np.prod([d for d in init.dims]) 
        for init in m.graph.initializer
        if len(init.dims) > 0
    )
    
    # Model file size
    model_size_mb = os.path.getsize(path) / 1e6
    
    entry = {
        "QuantizeLinear":    q_nodes,
        "DequantizeLinear":  dq_nodes,
        "QLinearConv":       ql_conv,
        "QLinearMatMul":     ql_mat,
        "QLinearAdd":        ql_add,
        "Conv_fp32":         conv_fp,
        "total_nodes":       len(ops),
        "total_params":      int(total_params),
        "model_size_mb":     model_size_mb,
        "top_ops":           dict(cnt.most_common(12)),
    }
    results[name] = entry
    print(f"\n{name}:")
    print(f"  Nodes: {len(ops)}, Params: {total_params/1e6:.2f}M, Size: {model_size_mb:.2f}MB")
    print(f"  Q={q_nodes}, DQ={dq_nodes}, QLinearConv={ql_conv}, Conv(fp)={conv_fp}")
    print(f"  Top ops: {dict(cnt.most_common(8))}")

# ── 2. The KEY insight: ratio of fused vs unfused ops ──────────────────────
print("\n=== MECHANISM ANALYSIS ===")
for name in ["mv3s_int8", "r18_int8"]:
    if name not in results:
        continue
    r = results[name]
    fused   = r["QLinearConv"] + r["QLinearMatMul"] + r["QLinearAdd"]
    unfused = r["QuantizeLinear"] + r["DequantizeLinear"]
    total   = r["total_nodes"]
    print(f"{name}: fused={fused}, unfused_castops={unfused}, ratio={unfused/max(fused,1):.2f}, total={total}")
    # Unfused/fused ratio: high ratio = more data-type conversion overhead

# ── 3. Inspect per-node sizes for R18 to find memory bottleneck ──────────────
print("\n=== R18 INT8 large initializers (potential cache misses) ===")
path = configs["r18_int8"]
if os.path.exists(path):
    m = onnx.load(path)
    inits = sorted(
        [(init.name, np.prod([d for d in init.dims]), init.data_type) 
         for init in m.graph.initializer if len(init.dims) > 1],
        key=lambda x: -x[1]
    )
    for name_i, size, dtype in inits[:10]:
        print(f"  {name_i[:40]}: {size/1000:.1f}K elements, dtype={dtype}")

# ── 4. Look at P90/P50 from the original CSV more carefully ──────────────────
print("\n=== Tail latency from pi_results.csv ===")
import csv
csv_path = os.path.expanduser("~/paper_results/static_benchmark_results.csv")
if os.path.exists(csv_path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        ratio = float(r["p90_latency_ms"]) / float(r["p50_latency_ms"])
        print(f"  {r['model']:25s} b={r['batch']} t={r['threads']} {r['precision']:4s}: "
              f"p50={float(r['p50_latency_ms']):.3f} p90={float(r['p90_latency_ms']):.3f} ratio={ratio:.3f}")

# ── 5. Save results ──────────────────────────────────────────────────────────
with open(os.path.expanduser("~/pi_mechanism_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to ~/pi_mechanism_results.json")
