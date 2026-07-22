# The INT8 Configuration Cliff (Raspberry Pi 5 / Arm Cortex-A76)

Reproducible artifact for the paper **"The INT8 Configuration Cliff: Export Format and
Graph Optimization, Not Weights, Govern a 40× Latency Swing on Arm Cortex-A76"**
(submitted to IEEE Embedded Systems Letters).

## Summary
On the same Raspberry Pi 5 and the same INT8 weights, inference latency spans a ~40× range in
the INT8/FP32 ratio — from a 10.7× speedup to a 2.2× slowdown — controlled entirely by two
configuration choices most benchmarks never report:

1. **Export format** — ONNX QDQ (`QuantizeLinear`/`DequantizeLinear`) vs. QOperator (`QLinearConv`).
2. **Runtime graph-optimization level** — `ORT_ENABLE_BASIC` vs. `ORT_ENABLE_ALL`.

QDQ at `ENABLE_ALL` gives 2.6–10.7× speedups; the same QDQ model at `BASIC` is ~2× *slower*
than FP32; QOperator models are 1.8–4× slower at every level. Top-1 accuracy is preserved to
within 0.5 points throughout. We also report that a previously observed tail-latency anomaly on
this platform did not reproduce across a single minor ONNX Runtime update — a caution on the
fragility of configuration- and version-blind edge benchmarks.

## Contents
```
paper/main_v3_reframed.pdf   the paper
figures/fig_config_cliff.pdf the figure  (+ gen_fig_cliff.py to regenerate it)
scripts/
  int8_full_ab.py            fp32 vs int8 (QDQ) at BASIC vs ALL, all 4 model/dataset pairs
  int8_optlevel_ab.py        optimization-level sweep (DISABLE/BASIC/EXTENDED/ALL)
  paper3_int8_ab.py          QDQ vs QOperator format comparison
  paper7_accuracy_latency.py top-1 accuracy + latency per precision variant
  paper7_tail_verify.py      P90/P50 tail-latency check across configs
data/latency_logs.csv        raw on-device latency measurements
LICENSE                      MIT
```

## Environment
- Raspberry Pi 5, 64-bit Raspberry Pi OS, Arm Cortex-A76 @ 2.4 GHz
- Python 3.11, ONNX Runtime 1.24.4, `CPUExecutionProvider`
- Install: `pip install onnxruntime numpy scipy scikit-learn`

## Reproducing the headline numbers
Place the FP32/INT8 ONNX exports under `exports/` (paths configurable at the top of each
script) and run, e.g.:
```bash
python scripts/int8_full_ab.py        # the configuration cliff table
python scripts/paper7_accuracy_latency.py   # accuracy is preserved across precisions
```
Every number in the paper's tables and figure is produced by these scripts on the hardware
above. Absolute latencies vary slightly with thermal state; the qualitative cliff (format ×
optimization level determining the sign of the INT8 effect) is stable.

## Citation
If you use this artifact, please cite the paper (IEEE Embedded Systems Letters, 2026).
Author: Manu Nicholas Jacob. Released under the MIT License.
