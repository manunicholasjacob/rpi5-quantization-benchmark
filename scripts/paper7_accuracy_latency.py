#!/usr/bin/env python3
"""
Paper 7 — the decisive check: do the INT8 models actually compute correct outputs?

latency_logs.csv has a latency column but NO accuracy column, so the reported
~8.7x median INT8 speedup could be a quantisation that silently degraded layers
(latency collapses, accuracy collapses with it). This measures BOTH top-1
accuracy and CPU latency for every precision variant of every model, on the
correct held-out test set, using ONNX Runtime only.

Reuses the exact data loaders / normalisation from researchpaper3/cm_eval.py.
Run on the Pi:  python3 paper7_accuracy_latency.py
Output:         ~/paper7_accuracy_latency.json  (incremental)
"""
import os, sys, json, time, gc, tarfile, hashlib, pickle
from pathlib import Path
import numpy as np
import onnxruntime as ort
from scipy.io import loadmat

ROOT = Path.home() / "Desktop" / "researchpaper3"
EXPORTS = ROOT / "exports"
DATA = ROOT / "data"
OUT = Path.home() / "paper7_accuracy_latency.json"

CIFAR100_MEAN=(0.5071,0.4865,0.4409); CIFAR100_STD=(0.2673,0.2564,0.2762)
SVHN_MEAN=(0.4377,0.4438,0.4728);     SVHN_STD=(0.1980,0.2010,0.1970)
THREADS = 4
BS = 128
TIMING_ITERS = 200   # single-sample latency iterations
TIMING_WARMUP = 40

def md5(p):
    h=hashlib.md5()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(1<<20),b""): h.update(c)
    return h.hexdigest()

def _key(d,*names):
    for n in names:
        if n in d: return d[n]
        if isinstance(n,str) and n.encode() in d: return d[n.encode()]
        if isinstance(n,bytes) and n.decode() in d: return d[n.decode()]
    raise KeyError(f"none of {names} in {list(d.keys())[:6]}")

def load_cifar100(root):
    tgz=root/"cifar-100-python.tar.gz"
    with tarfile.open(tgz,"r:gz") as tar:
        data=pickle.loads(tar.extractfile("cifar-100-python/test").read(),encoding="latin1")
    raw=np.asarray(_key(data,"data",b"data"),dtype=np.uint8)
    y=np.array(_key(data,"fine_labels",b"fine_labels","labels",b"labels"),dtype=np.int64)
    return raw,y

def cifar_batch(raw,sl):
    mean=np.array(CIFAR100_MEAN,np.float32)[:,None,None]; std=np.array(CIFAR100_STD,np.float32)[:,None,None]
    x=raw[sl].reshape(-1,3,32,32).astype(np.float32)/255.0
    return (x-mean)/std

def load_svhn(root):
    mat=loadmat(root/"test_32x32.mat"); X=mat["X"]; y=mat["y"].reshape(-1); y[y==10]=0
    return X,y.astype(np.int64)

def svhn_batch(X,sl):
    mean=np.array(SVHN_MEAN,np.float32)[:,None,None]; std=np.array(SVHN_STD,np.float32)[:,None,None]
    Xb=X[...,sl]; x=np.transpose(Xb,(3,2,0,1)).astype(np.float32)/255.0
    return (x-mean)/std

def make_session(onnx_path):
    so=ort.SessionOptions()
    so.intra_op_num_threads=THREADS; so.inter_op_num_threads=max(1,THREADS//2)
    so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    return ort.InferenceSession(str(onnx_path),sess_options=so,providers=["CPUExecutionProvider"])

def eval_accuracy(sess, dataset, raw, y):
    iname=sess.get_inputs()[0].name; n=len(y) if dataset=="CIFAR100" else y.shape[0]
    correct=0
    N = raw.shape[0] if dataset=="CIFAR100" else raw.shape[-1]
    for i in range(0,N,BS):
        x = cifar_batch(raw,slice(i,i+BS)) if dataset=="CIFAR100" else svhn_batch(raw,slice(i,i+BS))
        logits=sess.run(None,{iname:x})[0]
        pred=logits.argmax(axis=1); yb=y[i:i+BS]
        correct+=int((pred==yb).sum())
        del x,logits,pred
    return 100.0*correct/N

def measure_latency(sess, input_shape):
    iname=sess.get_inputs()[0].name
    x=np.random.randn(1,*input_shape).astype(np.float32)
    for _ in range(TIMING_WARMUP): sess.run(None,{iname:x})
    ts=[]
    for _ in range(TIMING_ITERS):
        t0=time.perf_counter(); sess.run(None,{iname:x}); ts.append((time.perf_counter()-t0)*1000)
    ts=np.array(ts)
    return dict(mean_ms=float(ts.mean()),p50_ms=float(np.percentile(ts,50)),
                p90_ms=float(np.percentile(ts,90)),p99_ms=float(np.percentile(ts,99)))

PAIRS=[("CIFAR100","mobilenet_v3_small"),("CIFAR100","resnet18"),
       ("SVHN","mobilenet_v3_small"),("SVHN","resnet18")]

def latest_dir(ds,model):
    c=[p for p in EXPORTS.glob(f"{ds}_{model}_*") if p.is_dir()]
    return sorted(c,key=lambda p:p.stat().st_mtime)[-1] if c else None

def main():
    results=[]
    print(f"ORT {ort.__version__}, threads={THREADS}, bs={BS}",flush=True)
    for ds,model in PAIRS:
        d=latest_dir(ds,model)
        if not d: print(f"skip {ds}/{model}: no export"); continue
        print(f"\n=== {ds}/{model}  ({d.name}) ===",flush=True)
        # load dataset once per pair
        if ds=="CIFAR100": raw,y=load_cifar100(DATA)
        else: raw,y=load_svhn(DATA)
        input_shape=(3,32,32)
        variants=sorted(d.glob("model_*.onnx"))
        fp32_acc=None
        for onnx in variants:
            prec=onnx.stem.replace("model_","")
            sz=onnx.stat().st_size/1e6
            try:
                sess=make_session(onnx)
                t0=time.time(); acc=eval_accuracy(sess,ds,raw,y); eval_s=time.time()-t0
                lat=measure_latency(sess,input_shape)
                if prec=="fp32": fp32_acc=acc
                row=dict(dataset=ds,model=model,precision=prec,size_mb=round(sz,3),
                         top1_acc=round(acc,3),eval_s=round(eval_s,1),**{k:round(v,4) for k,v in lat.items()})
                results.append(row)
                print(f"  {prec:16s} size={sz:5.2f}MB  acc={acc:6.2f}%  "
                      f"p50={lat['p50_ms']:.3f}ms mean={lat['mean_ms']:.3f}ms",flush=True)
                del sess; gc.collect()
            except Exception as e:
                print(f"  {prec:16s} FAILED: {e}",flush=True)
                results.append(dict(dataset=ds,model=model,precision=prec,error=str(e)))
            json.dump(results,open(OUT,"w"),indent=2)
        del raw,y; gc.collect()

    # summary: int8 vs fp32 accuracy drop + speedup
    print("\n"+"="*72); print("SUMMARY: is the INT8 speedup real (accuracy preserved)?"); print("="*72,flush=True)
    by=lambda ds,m,p: next((r for r in results if r.get("dataset")==ds and r.get("model")==m and r.get("precision")==p and "top1_acc" in r),None)
    for ds,model in PAIRS:
        f=by(ds,model,"fp32"); q=by(ds,model,"int8")
        if f and q:
            drop=f["top1_acc"]-q["top1_acc"]; spd=f["mean_ms"]/q["mean_ms"]
            verdict="REAL" if drop<3 else ("DEGRADED" if drop<15 else "BROKEN")
            print(f"  {ds}/{model:18s} fp32={f['top1_acc']:.1f}% int8={q['top1_acc']:.1f}% "
                  f"drop={drop:+.1f}pt speedup={spd:.2f}x  -> {verdict}",flush=True)
    json.dump(results,open(OUT,"w"),indent=2)
    print(f"\nSaved {OUT}")

if __name__=="__main__":
    main()
