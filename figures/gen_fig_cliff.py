#!/usr/bin/env python3
"""Publication figure for Paper 7 (config cliff). Two panels:
(a) INT8/FP32 latency ratio vs ONNX Runtime graph-optimization level (the cliff).
(b) INT8 speedup at ENABLE_ALL: QDQ (fast) vs QOperator (slow) — format decides sign.
All values measured on Raspberry Pi 5 / Cortex-A76 / ORT 1.24.4."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 8, "font.family": "serif", "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "figure.dpi": 300,
})

INK = "#1a1a1a"; MUTED = "#8a8a8a"; ACC = "#2b6cb0"; WARN = "#c05621"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.5))

# --- Panel (a): the optimization-level cliff (MV3/CIFAR-100, QDQ) ---
levels = ["DISABLE", "BASIC", "EXTENDED", "ALL"]
ratio = [2.085, 2.084, 0.590, 0.390]          # INT8/FP32 latency
x = np.arange(len(levels))
colors = [WARN if r > 1 else ACC for r in ratio]
ax1.bar(x, ratio, color=colors, width=0.62, edgecolor=INK, linewidth=0.5)
ax1.axhline(1.0, color=INK, lw=0.8, ls="--")
ax1.text(0.05, 1.06, "FP32 parity", transform=ax1.get_yaxis_transform(),
         fontsize=6.5, color=INK)
for xi, r in zip(x, ratio):
    lab = f"{1/r:.1f}× faster" if r < 1 else f"{r:.1f}× slower"
    ax1.text(xi, r + 0.05, lab, ha="center", va="bottom", fontsize=6,
             color=(ACC if r < 1 else WARN))
ax1.set_xticks(x); ax1.set_xticklabels(levels, fontsize=7)
ax1.set_ylabel("INT8 / FP32 latency ratio")
ax1.set_title("(a) The optimization-level cliff (QDQ)", fontsize=8)
ax1.set_ylim(0, 2.5)
ax1.spines[["top", "right"]].set_visible(False)

# --- Panel (b): format decides the sign at ENABLE_ALL ---
cfgs = ["QDQ\nMV3", "QDQ\nR18", "QOp\nsmall", "QOp\nmed", "QOp\nlarge"]
spd = [2.55, 10.71, 0.25, 0.26, 0.57]          # FP32/INT8 (>1 faster)
colors2 = [ACC if s > 1 else WARN for s in spd]
xb = np.arange(len(cfgs))
ax2.bar(xb, spd, color=colors2, width=0.62, edgecolor=INK, linewidth=0.5)
ax2.axhline(1.0, color=INK, lw=0.8, ls="--")
for xi, s in zip(xb, spd):
    ax2.text(xi, s + 0.2, f"{s:.1f}×", ha="center", va="bottom", fontsize=6,
             color=(ACC if s > 1 else WARN))
ax2.set_xticks(xb); ax2.set_xticklabels(cfgs, fontsize=6.5)
ax2.set_ylabel("INT8 speedup (FP32/INT8)")
ax2.set_title("(b) Format decides the sign (ENABLE\\_ALL)", fontsize=8)
ax2.set_yscale("log"); ax2.set_ylim(0.15, 15)
ax2.set_yticks([0.25, 0.5, 1, 2, 5, 10])
ax2.set_yticklabels(["0.25", "0.5", "1", "2", "5", "10"], fontsize=6.5)
ax2.spines[["top", "right"]].set_visible(False)

fig.tight_layout(pad=0.5)
fig.savefig("fig_config_cliff.pdf", bbox_inches="tight")
fig.savefig("fig_config_cliff.png", bbox_inches="tight", dpi=200)
print("wrote fig_config_cliff.pdf / .png")
