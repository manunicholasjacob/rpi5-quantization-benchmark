> ## Superseded: the headline result of this artifact is a measurement artifact
>
> **Updated August 2026.** This repository accompanied a manuscript arguing that the ONNX
> **export format** (QOperator vs QDQ) governs a large INT8 latency swing on Arm Cortex-A76,
> and that QOperator models run 1.8 to 4x *slower* than FP32.
>
> **That attribution is wrong. It is retracted, and the manuscript is not being pursued.**
>
> The arm labelled `QOperator` here was never a QOperator export. The model files contain
> `DynamicQuantizeLinear` and `ConvInteger` operators, which are the signature of *dynamic*
> quantization, rather than the `QLinearConv` that a genuine static QOperator export produces.
> The label was assumed rather than verified against the operator graph.
>
> Re-quantizing the same weights statically in both representations reverses the finding. On the
> Cortex-A76, static QOperator runs **1.25 to 2.70x faster** than FP32. The real determinant is
> **dynamic versus static quantization**, worth more than 4x. The choice between the two *static*
> representations is worth under 5%.
>
> The raw measurements in this repository are unchanged and remain valid as measurements. It is
> the interpretation that was wrong. They are kept public rather than deleted so that the
> correction is checkable.
>
> **If you are reusing anything here, verify the operator graph of any quantized artifact before
> labelling it.** `DynamicQuantizeLinear` / `ConvInteger` means dynamic, `QLinearConv` means
> static QOperator, and `QuantizeLinear` / `DequantizeLinear` around ops means QDQ.

# INT8 configuration study (Raspberry Pi 5 / Arm Cortex-A76)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21844863.svg)](https://doi.org/10.5281/zenodo.21844863)

Measurement artifact from a retracted manuscript. The manuscript argued that the ONNX export
format governs the sign of the INT8 effect on a Cortex-A76. That argument does not hold, for
the reason given in the notice above, and the paper is not being pursued. The measurements
themselves are unaffected and are kept here so the correction is checkable.

The corrected study is "When Does INT8 Actually Pay on an Edge CPU? A Cross-Platform Audit of
Post-Training Quantization", which re-measures this across three microarchitectures with
byte-identical models and one runtime version.

## What this artifact contains, and what it does not show

The measurements are latency of the same ImageNet models on a Raspberry Pi 5 across quantized
variants and two ONNX Runtime graph-optimization levels. Those numbers are real and reproducible
with the scripts here.

What the original manuscript concluded from them is not. It read the spread as a consequence of
the ONNX export format, QDQ against QOperator, and reported that QOperator models run slower
than FP32 at every optimization level. The arm labelled QOperator was never a QOperator export.
Its operator graph contains DynamicQuantizeLinear and ConvInteger, which is dynamic
quantization, not the QLinearConv that a static QOperator export produces.

Re-quantizing the same weights statically in both representations reverses it. On the
Cortex-A76 static QOperator runs 1.25 to 2.70 times faster than FP32. What actually governs the
sign is dynamic against static quantization, worth more than 4x. The choice between the two
static representations is worth under 20 percent.

Graph-optimization level does matter, and that part survives: QDQ at ORT_ENABLE_ALL is much
faster than the same file at ORT_ENABLE_BASIC, because fusion is what makes the quantized graph
pay. Top-1 accuracy is preserved to within 0.5 points across the variants.

One secondary result in the manuscript, a tail-latency anomaly, did not reproduce across a
minor ONNX Runtime update and was retracted separately.

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

## Reproducing the measurements
Place the FP32/INT8 ONNX exports under `exports/` (paths configurable at the top of each
script) and run, e.g.:
```bash
python scripts/int8_full_ab.py              # the latency table across configurations
python scripts/paper7_accuracy_latency.py   # accuracy is preserved across precisions
```
These scripts produce every number the manuscript reported, on the hardware above. Absolute
latencies vary slightly with thermal state. What the numbers were taken to mean is retracted,
for the reason in the notice at the top of this page; the numbers themselves reproduce.

## Citation

There is no paper to cite for this artifact. The manuscript it accompanied is retracted and is
not being pursued.

Cite the archived snapshot if you use the measurements:

    Jacob, M. N. INT8 configuration study on Raspberry Pi 5 (Arm Cortex-A76).
    Zenodo. https://doi.org/10.5281/zenodo.21844863

For the corrected result on when INT8 pays on an edge CPU, cite that work instead once it is
available.

Author: Manu Nicholas Jacob. Released under the MIT License.

## Archived version

This artifact is archived on Zenodo. The concept DOI
[10.5281/zenodo.21844863](https://doi.org/10.5281/zenodo.21844863)
always resolves to the latest release, and `CITATION.cff` carries the full metadata,
which is what GitHub's "Cite this repository" button renders.
