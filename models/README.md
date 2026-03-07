# ONNX Models

This directory contains the ONNX models used in the benchmark.

## Models

- `mobilenetv3_fp32.onnx` - MobileNetV3-Small FP32 (2.54M params)
- `mobilenetv3_int8.onnx` - MobileNetV3-Small INT8 quantized
- `resnet18_fp32.onnx` - ResNet-18 FP32 (11.69M params)
- `resnet18_int8.onnx` - ResNet-18 INT8 quantized

## Model Details

### MobileNetV3-Small
- **Parameters**: 2.54M
- **Architecture**: Depthwise separable convolutions, inverted residuals, hard-swish activations
- **Input**: 32×32 RGB images
- **Datasets**: CIFAR-100 (100 classes), SVHN (10 classes)

### ResNet-18
- **Parameters**: 11.69M
- **Architecture**: Standard 3×3 convolutions, residual connections
- **Input**: 32×32 RGB images
- **Datasets**: CIFAR-100 (100 classes), SVHN (10 classes)

## Quantization

- **Method**: Static Post-Training Quantization (PTQ)
- **Calibration**: MinMax, 512 samples
- **Precision**: INT8 (per-channel weights, per-tensor activations)
- **Accuracy loss**: <1% for both models

## Download

Models are not included in this repository due to size constraints.

To obtain the models:
1. Train using PyTorch 2.1.0 with the specifications in the paper
2. Export to ONNX (opset 17)
3. Quantize using ONNX Runtime quantization tools

Or contact: manunicholasjacob@gmail.com

## SHA-256 Checksums

For reproducibility, verify model checksums:

```
# Checksums will be added after model upload
mobilenetv3_fp32.onnx: <TBD>
mobilenetv3_int8.onnx: <TBD>
resnet18_fp32.onnx: <TBD>
resnet18_int8.onnx: <TBD>
```
