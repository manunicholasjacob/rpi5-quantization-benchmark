#!/usr/bin/env python3
"""Generate all publication-quality figures for paper v2."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})

COLORS = {'fp32': '#2166ac', 'int8': '#d6604d'}
HATCH  = {'fp32': '', 'int8': '///'}

# ─── Data from pi_results.csv (300 iterations, 4 threads) ───────────────────
# (dataset, model, precision): (p50_ms, CI_half_ms, p90_ms) at batch=1
data = {
    ('CIFAR100','MV3-S','fp32'):  (1.979, 0.003, 2.005),
    ('CIFAR100','MV3-S','int8'):  (0.520, 0.001, 0.542),
    ('SVHN',    'MV3-S','fp32'):  (1.964, 0.003, 1.996),
    ('SVHN',    'MV3-S','int8'):  (0.493, 0.001, 0.518),
    ('CIFAR100','R18',  'fp32'):  (13.249, 0.020, 14.250),
    ('CIFAR100','R18',  'int8'):  (1.130, 0.002, 2.196),
    ('SVHN',    'R18',  'fp32'):  (12.750, 0.018, 14.042),
    ('SVHN',    'R18',  'int8'):  (1.106, 0.002, 1.134),
}

# Energy surrogate: E = 0.8 + 0.02*CPU% + 0.8*L  (CPU=95%)
def energy(latency_ms):
    return 0.8 + 0.02*95 + 0.8*latency_ms

# Batch scaling data (4 threads) - exact values from pi_results.csv
batch_data = {
    'MV3-S fp32': {'batch':[1,2,4,8], 'p50':[1.979, 3.220, 5.210, 9.556]},
    'MV3-S int8': {'batch':[1,2,4,8], 'p50':[0.520, 0.708, 1.028, 1.673]},
    'R18 fp32':   {'batch':[1,2,4,8], 'p50':[13.249, 23.749, 44.180, 85.521]},
    'R18 int8':   {'batch':[1,2,4,8], 'p50':[1.130,  1.485,  1.956,  3.390]},
}

# P90/P50 ratios by batch — verified from pi_results.csv
# Key: R18 INT8 CIFAR-100 stays elevated (~1.94-2.03) across ALL batches
tail_ratios = {
    # (model, dataset, prec): [b1, b2, b4, b8]
    ('MV3-S','CIFAR100','fp32'):  [1.01, 1.01, 1.01, 2.29],  # fp32 b8 has own spike
    ('MV3-S','CIFAR100','int8'):  [1.04, 1.04, 1.03, 1.01],
    ('MV3-S','SVHN',    'int8'):  [1.05, 1.04, 1.03, 1.02],
    ('R18',  'CIFAR100','fp32'):  [1.08, 1.09, 1.09, 1.07],
    ('R18',  'CIFAR100','int8'):  [1.94, 1.91, 2.01, 2.03],  # FAILURE MODE
    ('R18',  'SVHN',    'int8'):  [1.03, 1.03, 1.01, 1.01],
}

# ─── Figure 1: Latency comparison with 95% CI ────────────────────────────────
fig, ax = plt.subplots(figsize=(3.5, 2.6))
configs = [('CIFAR100','MV3-S'), ('SVHN','MV3-S'), ('CIFAR100','R18'), ('SVHN','R18')]
labels  = ['MV3-S\nCIFAR-100','MV3-S\nSVHN','R18\nCIFAR-100','R18\nSVHN']
x = np.arange(len(configs))
w = 0.35

for i, (ds, mdl) in enumerate(configs):
    fp32_p50, fp32_ci, _ = data[(ds,mdl,'fp32')]
    int8_p50, int8_ci, _ = data[(ds,mdl,'int8')]
    ax.bar(x[i]-w/2, fp32_p50, w, yerr=fp32_ci, color=COLORS['fp32'], alpha=0.85,
           capsize=3, label='FP32' if i==0 else '', error_kw={'linewidth':1})
    ax.bar(x[i]+w/2, int8_p50, w, yerr=int8_ci, color=COLORS['int8'], alpha=0.85,
           capsize=3, label='INT8' if i==0 else '', error_kw={'linewidth':1})
    spd = fp32_p50/int8_p50
    ax.text(x[i], max(fp32_p50,int8_p50)+0.2, f'{spd:.1f}×', ha='center', fontsize=7, fontweight='bold', color='#333333')

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('P50 Latency (ms)')
ax.set_title('Latency: FP32 vs INT8 with 95% CI', pad=4)
ax.legend(loc='upper left', framealpha=0.8)
ax.set_ylim(0, 17)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('fig_latency_v2.png')
plt.close()
print("Saved fig_latency_v2.png")

# ─── Figure 2: Throughput scaling + batch P90/P50 showing failure persists ───
fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.4))

# Left: throughput vs batch
styles = {
    'MV3-S fp32': ('--', COLORS['fp32'], 'o', 'MV3-S FP32'),
    'MV3-S int8': ('-',  COLORS['int8'], 's', 'MV3-S INT8'),
    'R18 fp32':   ('--', '#4393c3',      '^', 'R18 FP32'),
    'R18 int8':   ('-',  '#f4a582',      'D', 'R18 INT8'),
}
ax = axes[0]
for key, (ls, col, mk, lbl) in styles.items():
    bd = batch_data[key]
    thru = [b/p*1000 for b,p in zip(bd['batch'], bd['p50'])]
    ax.plot(bd['batch'], thru, ls=ls, color=col, marker=mk, ms=4, lw=1.2, label=lbl)
ax.set_xlabel('Batch Size')
ax.set_ylabel('Throughput (img/s)')
ax.set_title('Throughput vs Batch', pad=3)
ax.set_xscale('log', base=2)
ax.set_xticks([1,2,4,8])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.legend(fontsize=6.5, framealpha=0.8)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Right: P90/P50 by batch size — showing R18 INT8 CIFAR100 stays elevated
ax2 = axes[1]
batches = [1, 2, 4, 8]
# Real data from pi_results.csv
r18_c100_int8 = [1.94, 1.91, 2.01, 2.03]   # FAILURE: stays ~2x
r18_svhn_int8 = [1.03, 1.03, 1.01, 1.01]   # Safe
mv3s_c100_int8= [1.04, 1.04, 1.03, 1.01]   # Safe

ax2.plot(batches, r18_c100_int8,  '-',  color='#d6604d', marker='D', ms=5, lw=1.5,
         label='R18 INT8 C100', zorder=5)
ax2.plot(batches, r18_svhn_int8,  '--', color='#f4a582', marker='s', ms=4, lw=1.2,
         label='R18 INT8 SVHN')
ax2.plot(batches, mv3s_c100_int8, '--', color='#92c5de', marker='o', ms=4, lw=1.2,
         label='MV3-S INT8 C100')
ax2.axhline(1.5, color='red', ls=':', lw=1.0, alpha=0.8, label='SLO risk')
ax2.fill_between(batches, 1.5, 2.1, alpha=0.06, color='red')
ax2.set_xlabel('Batch Size')
ax2.set_ylabel('P90/P50 Ratio')
ax2.set_title('Tail-Latency by Batch', pad=3)
ax2.set_xscale('log', base=2)
ax2.set_xticks([1,2,4,8])
ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax2.set_ylim(0.9, 2.2)
ax2.legend(fontsize=6, framealpha=0.85, loc='lower right')
ax2.yaxis.grid(True, alpha=0.3)
ax2.set_axisbelow(True)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.annotate('Persists\nacross\nbatches', xy=(4, 2.01),
             xytext=(2.2, 2.12), fontsize=6, color='#c0392b',
             arrowprops=dict(arrowstyle='->', color='#c0392b', lw=0.8))

plt.tight_layout(pad=0.8)
plt.savefig('fig_scaling_tail_v2.png')
plt.close()
print("Saved fig_scaling_tail_v2.png")

# ─── Figure 3: EDP comparison ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.5, 2.4))
configs3 = [('CIFAR100','MV3-S'), ('SVHN','MV3-S'), ('CIFAR100','R18'), ('SVHN','R18')]
xlabels3 = ['MV3-S\nCIFAR-100','MV3-S\nSVHN','R18\nCIFAR-100','R18\nSVHN']
x3 = np.arange(len(configs3))
w3 = 0.35

edp_ratios = []
for i, (ds,mdl) in enumerate(configs3):
    fp32_p50 = data[(ds,mdl,'fp32')][0]
    int8_p50 = data[(ds,mdl,'int8')][0]
    e_fp32 = energy(fp32_p50)
    e_int8 = energy(int8_p50)
    edp_fp32 = fp32_p50 * e_fp32
    edp_int8 = int8_p50 * e_int8
    ratio = edp_fp32 / edp_int8
    edp_ratios.append(ratio)
    ax.bar(x3[i], ratio, 0.55, color='#4dac26', alpha=0.85)
    ax.text(x3[i], ratio+0.5, f'{ratio:.1f}×', ha='center', fontsize=8, fontweight='bold')

ax.set_xticks(x3)
ax.set_xticklabels(xlabels3)
ax.set_ylabel('EDP Reduction (FP32/INT8)')
ax.set_title('Energy-Delay Product Improvement', pad=4)
ax.set_ylim(0, max(edp_ratios)*1.25)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axhline(1, color='gray', ls='--', lw=0.8, alpha=0.5)
plt.tight_layout()
plt.savefig('fig_edp_v2.png')
plt.close()
print("Saved fig_edp_v2.png")

print("All figures generated successfully.")
