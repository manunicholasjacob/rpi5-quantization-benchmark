#!/usr/bin/env python3
"""Generate publication-quality figures with error bars and statistical annotations."""

import matplotlib.pyplot as plt
import numpy as np
import json

# Publication quality settings
plt.rcParams['font.size'] = 9
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 7.5
plt.rcParams['figure.dpi'] = 300

# Load enhanced analysis
with open('enhanced_analysis.json', 'r') as f:
    data = json.load(f)

# Extract data
models = ['MV3-S\nCIFAR', 'MV3-S\nSVHN', 'R18\nCIFAR', 'R18\nSVHN']
keys = [
    'mobilenet_v3_small_CIFAR100_mobilenet_v3_small_CIFAR100',
    'mobilenet_v3_small_SVHN_mobilenet_v3_small_SVHN',
    'resnet18_CIFAR100_resnet18_CIFAR100',
    'resnet18_SVHN_resnet18_SVHN'
]

# Figure 1: Latency comparison with error bars
fig, ax = plt.subplots(figsize=(3.5, 2.8))

fp32_lats = []
int8_lats = []
fp32_errs = []
int8_errs = []

for key in keys:
    stats = data['statistical_tests'][key]
    ci = data['confidence_intervals'][key]
    
    fp32_lats.append(stats['fp32_ms'])
    int8_lats.append(stats['int8_ms'])
    
    # Error bars (half of CI width)
    fp32_errs.append((ci['fp32_ci_upper'] - ci['fp32_ci_lower']) / 2)
    int8_errs.append((ci['int8_ci_upper'] - ci['int8_ci_lower']) / 2)

x = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x - width/2, fp32_lats, width, yerr=fp32_errs, 
               label='FP32', color='#d62728', alpha=0.8, capsize=3)
bars2 = ax.bar(x + width/2, int8_lats, width, yerr=int8_errs,
               label='INT8', color='#2ca02c', alpha=0.8, capsize=3)

# Add speedup annotations
for i, key in enumerate(keys):
    speedup = data['statistical_tests'][key]['speedup']
    y_pos = max(fp32_lats[i], int8_lats[i]) * 1.15
    ax.text(i, y_pos, f'{speedup:.1f}×', ha='center', va='bottom', 
            fontsize=7, fontweight='bold')

ax.set_ylabel('Latency (ms/image)')
ax.set_xlabel('Model / Dataset')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=7)
ax.legend(loc='upper left', framealpha=0.9)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, max(fp32_lats) * 1.3)
ax.set_title('Latency Comparison (batch=1, 4 threads)', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('fig_latency_comparison_ci.png', dpi=300, bbox_inches='tight')
print("Generated: fig_latency_comparison_ci.png")
plt.close()

# Figure 2: Energy comparison with error bars
fig, ax = plt.subplots(figsize=(3.5, 2.8))

def energy_mj(lat_ms):
    return 0.8 + 0.02 * 95 + 0.8 * lat_ms

def energy_err(lat_err):
    return 0.8 * lat_err

fp32_energies = [energy_mj(lat) for lat in fp32_lats]
int8_energies = [energy_mj(lat) for lat in int8_lats]
fp32_e_errs = [energy_err(err) for err in fp32_errs]
int8_e_errs = [energy_err(err) for err in int8_errs]

bars1 = ax.bar(x - width/2, fp32_energies, width, yerr=fp32_e_errs,
               label='FP32', color='#d62728', alpha=0.8, capsize=3)
bars2 = ax.bar(x + width/2, int8_energies, width, yerr=int8_e_errs,
               label='INT8', color='#2ca02c', alpha=0.8, capsize=3)

# Add reduction factors
for i in range(len(models)):
    reduction = fp32_energies[i] / int8_energies[i]
    y_pos = max(fp32_energies[i], int8_energies[i]) * 1.12
    ax.text(i, y_pos, f'{reduction:.2f}×', ha='center', va='bottom',
            fontsize=7, fontweight='bold')

ax.set_ylabel('Energy (mJ/image)')
ax.set_xlabel('Model / Dataset')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=7)
ax.legend(loc='upper left', framealpha=0.9)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, max(fp32_energies) * 1.25)
ax.set_title('Energy Comparison (batch=1, 4 threads)', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('fig_energy_comparison_ci.png', dpi=300, bbox_inches='tight')
print("Generated: fig_energy_comparison_ci.png")
plt.close()

# Figure 3: EDP comparison with error bars
fig, ax = plt.subplots(figsize=(3.5, 2.5))

fp32_edp = [e * l for e, l in zip(fp32_energies, fp32_lats)]
int8_edp = [e * l for e, l in zip(int8_energies, int8_lats)]

# Error propagation for EDP: σ(E*L) ≈ sqrt((L*σE)^2 + (E*σL)^2)
fp32_edp_errs = [np.sqrt((l*e_err)**2 + (e*l_err)**2) 
                 for l, e, l_err, e_err in zip(fp32_lats, fp32_energies, fp32_errs, fp32_e_errs)]
int8_edp_errs = [np.sqrt((l*e_err)**2 + (e*l_err)**2)
                 for l, e, l_err, e_err in zip(int8_lats, int8_energies, int8_errs, int8_e_errs)]

bars1 = ax.bar(x - width/2, fp32_edp, width, yerr=fp32_edp_errs,
               label='FP32', color='#d62728', alpha=0.8, capsize=3)
bars2 = ax.bar(x + width/2, int8_edp, width, yerr=int8_edp_errs,
               label='INT8', color='#2ca02c', alpha=0.8, capsize=3)

# Add reduction factors
for i in range(len(models)):
    reduction = fp32_edp[i] / int8_edp[i]
    y_pos = max(fp32_edp[i], int8_edp[i]) * 1.08
    ax.text(i, y_pos, f'{reduction:.1f}×', ha='center', va='bottom',
            fontsize=7, fontweight='bold')

ax.set_ylabel('Energy-Delay Product (mJ·ms)')
ax.set_xlabel('Model / Dataset')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=7)
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, max(fp32_edp) * 1.2)
ax.set_title('Energy-Delay Product', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('fig_edp_comparison_ci.png', dpi=300, bbox_inches='tight')
print("Generated: fig_edp_comparison_ci.png")
plt.close()

print("\n✓ All enhanced figures generated with error bars!")
print("Files created:")
print("  - fig_latency_comparison_ci.png")
print("  - fig_energy_comparison_ci.png")
print("  - fig_edp_comparison_ci.png")
