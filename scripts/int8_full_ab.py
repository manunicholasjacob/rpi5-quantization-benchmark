#!/usr/bin/env python3
"""Full INT8 opt-level A/B across all 4 Paper-7 model pairs.
Confirms the QDQ-fusion mechanism and gives real speedups at ENABLE_ALL."""
import numpy as np, time, onnxruntime as ort, json
from pathlib import Path

EXPORTS = Path.home()/"Desktop/researchpaper3/exports"
PAIRS=[("CIFAR100","mobilenet_v3_small"),("CIFAR100","resnet18"),
       ("SVHN","mobilenet_v3_small"),("SVHN","resnet18")]
LEVELS = {"BASIC":ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
          "ALL":  ort.GraphOptimizationLevel.ORT_ENABLE_ALL}

def latest(ds,m):
    c=[p for p in EXPORTS.glob(f"{ds}_{m}_*") if p.is_dir()]
    return sorted(c,key=lambda p:p.stat().st_mtime)[-1] if c else None

def bench(path, level, threads=4, warmup=40, iters=200):
    so=ort.SessionOptions(); so.intra_op_num_threads=threads
    so.inter_op_num_threads=max(1,threads//2); so.graph_optimization_level=level
    s=ort.InferenceSession(str(path),sess_options=so,providers=["CPUExecutionProvider"])
    i0=s.get_inputs()[0].name; x=np.random.randn(1,3,32,32).astype("float32")
    for _ in range(warmup): s.run(None,{i0:x})
    ts=[]
    for _ in range(iters):
        t0=time.perf_counter(); s.run(None,{i0:x}); ts.append((time.perf_counter()-t0)*1000)
    return float(np.mean(ts))

print(f"ORT {ort.__version__}\n")
print(f"{'pair':30s} {'fp32':>8s} | {'int8 BASIC':>10s} {'spd':>6s} | {'int8 ALL':>9s} {'spd':>6s}")
print("-"*82)
out=[]
for ds,m in PAIRS:
    d=latest(ds,m)
    if not d: continue
    fp=bench(d/"model_fp32.onnx",LEVELS["ALL"])
    ib=bench(d/"model_int8.onnx",LEVELS["BASIC"])
    ia=bench(d/"model_int8.onnx",LEVELS["ALL"])
    print(f"{ds+'/'+m:30s} {fp:7.3f}ms | {ib:9.3f}ms {fp/ib:5.2f}x | {ia:8.3f}ms {fp/ia:5.2f}x",flush=True)
    out.append(dict(dataset=ds,model=m,fp32_all_ms=fp,int8_basic_ms=ib,int8_all_ms=ia,
                    speedup_basic=fp/ib,speedup_all=fp/ia))
json.dump(out,open(Path.home()/"int8_full_ab.json","w"),indent=2)
print("\nspeedup>1 = INT8 faster than FP32. Compare ALL column to original latency_logs claims.")
