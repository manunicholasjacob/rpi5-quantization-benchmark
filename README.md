# INT8 Quantization on Raspberry Pi 5

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](paper/quantization_paper.pdf)

Reproducible benchmark for FP32 vs INT8 inference on Raspberry Pi 5 with ONNX Runtime.

**Paper**: "INT8 Quantization on Raspberry Pi 5: A Reproducible Latency–Energy Benchmark with Statistical Validation"

## Key Results

- **3.80–11.73× speedup** with INT8 quantization (p < 0.001, Cohen's d > 100)
- **1.37–3.69× energy reduction** (surrogate model, ±20–60% uncertainty)
- **5.2–43.3× EDP improvement** (energy-delay product)
- **<1% accuracy loss** at 512 calibration samples
- **Failure mode identified**: ResNet-18 INT8 shows 1.94× P90/P50 latency inflation due to Q/DQ overhead

## Hardware

- **Device**: Raspberry Pi 5 (8 GB)
- **CPU**: ARM Cortex-A76, 2.4 GHz
- **OS**: Raspberry Pi OS 64-bit (kernel 6.6)
- **Thermal**: No throttling (monitored via `vcgencmd`)

## Software

- **Runtime**: ONNX Runtime 1.18.0 (CPUExecutionProvider)
- **Python**: 3.11
- **Framework**: PyTorch 2.1.0 (for model training)
- **Quantization**: Static PTQ, MinMax calibration, 512 samples

## Models

- **MobileNetV3-Small**: 2.54M parameters, depthwise separable convolutions
- **ResNet-18**: 11.69M parameters, standard 3×3 convolutions

Trained on CIFAR-100 and SVHN (32×32 RGB images).

## Repository Structure

```
rpi5-quantization-benchmark/
├── paper/
│   └── quantization_paper.pdf          # Published paper
├── models/
│   ├── mobilenetv3_fp32.onnx           # FP32 models (placeholder)
│   ├── mobilenetv3_int8.onnx           # INT8 models (placeholder)
│   ├── resnet18_fp32.onnx
│   └── resnet18_int8.onnx
├── data/
│   ├── raw_logs/
│   │   └── latency_logs.csv            # Raw experimental data (9,600 measurements)
│   └── enhanced_analysis.json          # Statistical analysis results
├── scripts/
│   ├── run_enhanced_experiments.py     # Statistical analysis script
│   └── generate_enhanced_figures.py    # Figure generation script
├── figures/
│   ├── fig_latency.png                 # Latency comparison with 95% CI
│   ├── fig_energy.png                  # Energy comparison with 95% CI
│   └── fig_edp.png                     # Energy-delay product comparison
├── environment/
│   ├── requirements.txt                # Python dependencies
│   └── environment.yml                 # Conda environment (optional)
├── README.md                           # This file
├── reproduce.sh                        # One-command reproduction script
└── LICENSE                             # MIT License
```

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/manunicholasjacob/rpi5-quantization-benchmark.git
cd rpi5-quantization-benchmark
```

### 2. Install Dependencies

```bash
pip install -r environment/requirements.txt
```

Or using conda:

```bash
conda env create -f environment/environment.yml
conda activate rpi5-quant
```

### 3. Run Analysis

```bash
# Run statistical analysis on existing data
python scripts/run_enhanced_experiments.py

# Generate publication-quality figures
python scripts/generate_enhanced_figures.py
```

Or use the one-command script:

```bash
bash reproduce.sh
```

## Reproducing Experiments

### On Raspberry Pi 5

To reproduce the full experimental pipeline on your own Raspberry Pi 5:

1. **Set up environment**:
   ```bash
   # Install ONNX Runtime
   pip install onnxruntime==1.18.0
   
   # Set CPU governor to performance
   echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
   ```

2. **Run inference benchmarks** (requires ONNX models):
   ```bash
   # Models should be placed in models/ directory
   # Run 300-iteration trials with batch=1, threads=4
   python scripts/run_inference_benchmark.py
   ```

3. **Analyze results**:
   ```bash
   python scripts/run_enhanced_experiments.py
   ```

### Expected Outputs

- **Latency statistics**: P50, P90, 95% confidence intervals
- **Energy estimates**: Via validated surrogate model (±20–60% uncertainty)
- **Effect sizes**: Cohen's d for FP32 vs INT8 comparisons
- **Publication figures**: Latency, energy, and EDP plots with error bars

## Key Findings

### Speedup (Batch=1, 4 Threads)

| Model | Dataset | FP32 (ms) | INT8 (ms) | Speedup |
|-------|---------|-----------|-----------|---------|
| MobileNetV3-Small | CIFAR-100 | 1.98 | 0.52 | **3.80×** |
| MobileNetV3-Small | SVHN | 1.96 | 0.49 | **3.99×** |
| ResNet-18 | CIFAR-100 | 13.25 | 1.13 | **11.73×** |
| ResNet-18 | SVHN | 12.75 | 1.11 | **11.53×** |

### Statistical Significance

- **All comparisons**: p < 0.001
- **Effect sizes**: Cohen's d = 100.8–128.9 (very large)
- **Measurement precision**: CV = 0.8–1.2%

### Failure Mode: Tail Latency Inflation

ResNet-18 INT8 on CIFAR-100 exhibits **1.94× P90/P50 ratio**:
- **Median latency**: 1.13 ms
- **P90 latency**: 2.20 ms (95% slowdown at tail)
- **Cause**: 18 Q/DQ operation pairs causing cache thrashing

MobileNetV3-Small shows **<10% P90 inflation** due to fused depthwise kernels.

## Practical Recommendations

For Raspberry Pi 5 + ONNX Runtime 1.18.0:

1. **Use INT8** with n≥512 calibration samples for <0.5% accuracy loss
2. **Prefer batch=1–2, threads=2–4** for latency-critical tasks
3. **Audit INT8 operator coverage** (target >95%) and enable kernel fusion
4. **Monitor P90 latency** for models with >20 layers; consider mixed precision if P90/P50 > 1.5

## Scope and Limitations

**Findings are scoped to**:
- Raspberry Pi 5 (ARM Cortex-A76, 2.4 GHz) with ONNX Runtime 1.18.0
- MobileNetV3-Small and ResNet-18 only
- CIFAR-100 and SVHN (32×32 RGB classification)
- Batch=1, threads=4, static PTQ (no QAT or mixed precision)
- Energy via surrogate model (absolute values ±20–60% uncertain; relative rankings more reliable)

**May not generalize to**: Other backends (TFLite, TVM), hardware (x86, GPU), models (transformers, detection), or quantization schemes (QAT, W4A8).

## Citation

If you use this work, please cite:

```bibtex
@article{jacob2026int8quantization,
  title={INT8 Quantization on Raspberry Pi 5: A Reproducible Latency–Energy Benchmark with Statistical Validation},
  author={Jacob, Manu Nicholas},
  journal={IEEE Embedded Systems Letters},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**Manu Nicholas Jacob**  
Email: manunicholasjacob@gmail.com  
GitHub: [@manunicholasjacob](https://github.com/manunicholasjacob)

## Acknowledgments

- ONNX Runtime team for the inference framework
- PyTorch team for model training tools
- Raspberry Pi Foundation for the hardware platform
