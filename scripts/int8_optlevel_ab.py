#!/usr/bin/env python3
"""Reconcile Paper 7 vs Paper 3 INT8 contradiction.
Hypothesis: INT8 speed depends on ORT graph_optimization_level. The original
harness used the default (ENABLE_ALL); my accuracy script forced ENABLE_BASIC.
Test fp32 vs int8 under DISABLE_ALL / BASIC / EXTENDED / ALL for one model."""
import numpy as np, time, onnxruntime as ort
from pathlib import Path

D = Path.home()/"Desktop/researchpaper3/exports/CIFAR100_mobilenet_v3_small_CIFAR100_mobilenet_v3_small_20251015_155545"
LEVELS = {
    "DISABLE_ALL": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "BASIC":       ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "EXTENDED":    ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "ALL":         ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}

def bench(path, level, threads=4, warmup=40, iters=200):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads; so.inter_op_num_threads = max(1, threads//2)
    so.graph_optimization_level = level
    sess = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
    i0 = sess.get_inputs()[0].name
    x = np.random.randn(1,3,32,32).astype("float32")
    for _ in range(warmup): sess.run(None,{i0:x})
    ts=[]
    for _ in range(iters):
        t0=time.perf_counter(); sess.run(None,{i0:x}); ts.append((time.perf_counter()-t0)*1000)
    return float(np.mean(ts)), float(np.percentile(ts,50))

print(f"ORT {ort.__version__}  (device: Raspberry Pi 5)\n")
print(f"{'level':12s} {'fp32 mean':>10s} {'int8 mean':>10s} {'int8/fp32':>10s} {'verdict':>14s}")
print("-"*60)
for name, lvl in LEVELS.items():
    fp = bench(D/"model_fp32.onnx", lvl)
    iq = bench(D/"model_int8.onnx", lvl)
    ratio = iq[0]/fp[0]
    v = "INT8 faster" if ratio<0.95 else ("~equal" if ratio<1.1 else "INT8 slower")
    print(f"{name:12s} {fp[0]:9.3f}ms {iq[0]:9.3f}ms {ratio:9.3f}x {v:>14s}", flush=True)

print("\nOriginal latency_logs.csv reported int8/fp32 = 0.264x (INT8 ~4x faster) for this model.")
print("If ALL/EXTENDED reproduce that, the speedup is real but optimization-dependent;")
print("if every level shows INT8 slower, the original 0.524ms was a measurement bug.")
