#!/usr/bin/env python3
"""Verify Paper 7's central claim: R18 INT8 CIFAR-100 shows ~1.94x P90/P50 tail
inflation, absent on SVHN and on MobileNetV3. Measure at ENABLE_ALL (the setting
that gives the real INT8 speedup), ORT 1.24.4, 500 iters, for all 4 configs +
FP32 baselines. This is the paper's headline finding — reproduce or refute it."""
import numpy as np, time, onnxruntime as ort, json
from pathlib import Path

EXPORTS = Path.home()/"Desktop/researchpaper3/exports"
PAIRS=[("CIFAR100","mobilenet_v3_small"),("SVHN","mobilenet_v3_small"),
       ("CIFAR100","resnet18"),("SVHN","resnet18")]

def latest(ds,m):
    c=[p for p in EXPORTS.glob(f"{ds}_{m}_*") if p.is_dir()]
    return sorted(c,key=lambda p:p.stat().st_mtime)[-1] if c else None

def bench(path, iters=500, warmup=60):
    so=ort.SessionOptions(); so.intra_op_num_threads=4; so.inter_op_num_threads=2
    so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    s=ort.InferenceSession(str(path),sess_options=so,providers=["CPUExecutionProvider"])
    i0=s.get_inputs()[0].name; x=np.random.randn(1,3,32,32).astype("float32")
    for _ in range(warmup): s.run(None,{i0:x})
    t=[]
    for _ in range(iters):
        t0=time.perf_counter(); s.run(None,{i0:x}); t.append((time.perf_counter()-t0)*1000)
    a=np.array(t)
    return dict(p50=float(np.percentile(a,50)),p90=float(np.percentile(a,90)),
                p99=float(np.percentile(a,99)),mean=float(a.mean()),
                cv=float(a.std()/a.mean()),ratio=float(np.percentile(a,90)/np.percentile(a,50)))

print(f"ORT {ort.__version__}  ENABLE_ALL  500 iters/config\n")
print(f"{'config':28s} {'prec':5s} {'P50':>7s} {'P90':>7s} {'P99':>7s} {'CV':>6s} {'P90/P50':>8s}")
print("-"*76)
out=[]
for ds,m in PAIRS:
    d=latest(ds,m)
    for prec,fn in [("fp32","model_fp32.onnx"),("int8","model_int8.onnx")]:
        r=bench(d/fn)
        flag="  <-- TAIL" if r["ratio"]>1.5 else ""
        print(f"{ds+'/'+m:28s} {prec:5s} {r['p50']:6.3f}ms {r['p90']:6.3f}ms {r['p99']:6.3f}ms "
              f"{r['cv']*100:5.1f}% {r['ratio']:7.2f}x{flag}",flush=True)
        out.append(dict(dataset=ds,model=m,precision=prec,**r))
json.dump(out,open(Path.home()/"paper7_tail_verify.json","w"),indent=2)
print("\nPaper 7 (ORT 1.18.0) claimed R18/INT8/CIFAR100 P90/P50 = 1.94x, others ~1.0-1.1x.")
print("Does it reproduce on ORT 1.24.4?")
