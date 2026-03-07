#!/bin/bash
# Reproduce INT8 Quantization Benchmark Results
# One-command script to regenerate all analysis and figures

set -e  # Exit on error

echo "========================================="
echo "INT8 Quantization Benchmark Reproduction"
echo "========================================="
echo ""

# Check if running on Raspberry Pi 5
if [ -f /proc/device-tree/model ]; then
    MODEL=$(cat /proc/device-tree/model)
    echo "Detected hardware: $MODEL"
else
    echo "Warning: Could not detect hardware model"
fi

echo ""
echo "Step 1: Installing dependencies..."
pip install -r environment/requirements.txt

echo ""
echo "Step 2: Running statistical analysis..."
python scripts/run_enhanced_experiments.py

echo ""
echo "Step 3: Generating publication figures..."
python scripts/generate_enhanced_figures.py

echo ""
echo "========================================="
echo "Reproduction complete!"
echo "========================================="
echo ""
echo "Outputs:"
echo "  - enhanced_analysis.json (statistical results)"
echo "  - fig_latency_comparison_ci.png"
echo "  - fig_energy_comparison_ci.png"
echo "  - fig_edp_comparison_ci.png"
echo ""
echo "Expected variance: <2% across Raspberry Pi 5 boards"
