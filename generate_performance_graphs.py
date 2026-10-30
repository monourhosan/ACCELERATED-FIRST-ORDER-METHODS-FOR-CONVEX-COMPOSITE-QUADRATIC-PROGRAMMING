"""
Generate publication-quality performance graphs comparing Python, C, C++, and Rust
across CPU Baseline, Multi-Core Accelerated, and GPU Acceleration on CVXQP1_L.
"""

import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

# Colors
color_py = "#3776AB"    # Python Blue
color_c = "#555555"     # Classic C Slate
color_cpp = "#00599C"   # C++ Navy
color_rust = "#CE412B"  # Rust Orange-Red
color_gpu = "#76B900"   # NVIDIA Green

# -------------------------------------------------------------
# GRAPH 1: 300 Iterations Execution Time on CVXQP1_L (n=10,000)
# -------------------------------------------------------------
categories = [
    "Python\n(SciPy CSR)",
    "Rust\n(Single-Core)",
    "C++20\n(Single-Core)",
    "C99\n(Single-Core)",
    "Accelerated\nC99 (8-Thread)",
    "Accelerated\nC++20 (8-Thread)",
    "Accelerated\nPython Engine",
    "Rust GPU\n(NVIDIA MX250)",
    "C++20 GPU\n(NVIDIA MX250)",
    "C99 GPU\n(NVIDIA MX250)"
]

times = [
    178.48, # Python
    136.31, # Rust CPU
    92.23,  # C++ CPU
    75.28,  # C CPU
    52.92,  # C 8T
    42.66,  # C++ 8T
    40.22,  # Python Accelerated Bridge
    80.05,  # Rust GPU
    80.45,  # C++ GPU
    81.16   # C GPU
]

bar_colors = [
    color_py,
    color_rust,
    color_cpp,
    color_c,
    "#888888",
    "#2E86AB",
    "#1E88E5",
    "#E65100",
    "#00897B",
    color_gpu
]

bars1 = ax1.bar(categories, times, color=bar_colors, width=0.62, edgecolor='black', linewidth=0.8)

# Add values on top of bars
for bar, t in zip(bars1, times):
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + 2.5, f"{t:.1f} ms",
             ha='center', va='bottom', fontsize=9, fontweight='bold')

ax1.set_title("Runtime Comparison on CVXQP1_L (300 Iterations)\nLower is Faster", fontsize=13, fontweight='bold', pad=12)
ax1.set_ylabel("Execution Time (milliseconds)", fontsize=11, fontweight='bold')
ax1.set_ylim(0, 205)
ax1.tick_params(axis='x', rotation=35, labelsize=9)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Highlight fastest CPU and GPU
ax1.annotate('Fastest CPU:\nAccelerated Engine (40.2 ms)',
             xy=(6, 40.22), xytext=(5.2, 115),
             arrowprops=dict(facecolor='#1E88E5', shrink=0.08, width=1.5, headwidth=6),
             fontsize=9, fontweight='bold', color='#0D47A1',
             bbox=dict(boxstyle="round,pad=0.3", fc="#E3F2FD", ec="#1E88E5", lw=1))

ax1.annotate('Hardware GPU Parity:\n~80 ms across C/C++/Rust',
             xy=(8, 80.45), xytext=(6.5, 155),
             arrowprops=dict(facecolor=color_gpu, shrink=0.08, width=1.5, headwidth=6),
             fontsize=9, fontweight='bold', color='#2E7D32',
             bbox=dict(boxstyle="round,pad=0.3", fc="#E8F5E9", ec=color_gpu, lw=1))


# -------------------------------------------------------------
# GRAPH 2: Speedup Factor Relative to Single-Core Python Baseline
# -------------------------------------------------------------
speedups = [178.48 / t for t in times]

bars2 = ax2.bar(categories, speedups, color=bar_colors, width=0.62, edgecolor='black', linewidth=0.8)

for bar, s in zip(bars2, speedups):
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.08, f"{s:.2f}x",
             ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2.axhline(1.0, color='red', linestyle='--', linewidth=1.2, label='Python Baseline (1.0x)')
ax2.set_title("Speedup vs. Python Baseline (300 Iterations)\nHigher is Faster", fontsize=13, fontweight='bold', pad=12)
ax2.set_ylabel("Speedup Multiplier (x times faster)", fontsize=11, fontweight='bold')
ax2.set_ylim(0, 5.2)
ax2.tick_params(axis='x', rotation=35, labelsize=9)
ax2.grid(axis='y', linestyle='--', alpha=0.7)
ax2.legend(loc='upper left', frameon=True)

plt.tight_layout()
output_path = "assets/performance_comparison.png"
plt.savefig(output_path, dpi=300)
print(f"Performance graph successfully saved to: {output_path}")

# -------------------------------------------------------------
# GRAPH 3: Small-Problem Regime (Microsecond Latency on Ex 5.1 & HS21)
# -------------------------------------------------------------
fig_small, ax_small = plt.subplots(figsize=(10, 5), dpi=300)

prob_names = ["Example 5.1 (n=3)", "Example 5.2 (n=2)", "HS21 (n=2, m=1)"]
py_times_us = [9850, 7420, 9000] # microsecond (us)
c_times_us  = [12, 7, 9]         # microsecond (us)
cpp_times_us = [11.7, 6.6, 8.4]  # microsecond (us)
rust_times_us = [11.0, 6.0, 8.0] # microsecond (us)

x = np.arange(len(prob_names))
width = 0.2

rects1 = ax_small.bar(x - 1.5*width, py_times_us, width, label='Python 3.12 (CPython)', color=color_py)
rects2 = ax_small.bar(x - 0.5*width, c_times_us, width, label='C99 (/O2)', color=color_c)
rects3 = ax_small.bar(x + 0.5*width, cpp_times_us, width, label='C++20 (/O2)', color=color_cpp)
rects4 = ax_small.bar(x + 1.5*width, rust_times_us, width, label='Rust 2021 (--release)', color=color_rust)

ax_small.set_yscale('log')
ax_small.set_title("Small/Medium Problems (Latency-Bound Regime): C/C++/Rust vs Python\nLog Scale (Microseconds)", fontsize=12, fontweight='bold')
ax_small.set_ylabel("Execution Time (µs, Log Scale)", fontsize=10, fontweight='bold')
ax_small.set_xticks(x)
ax_small.set_xticklabels(prob_names, fontsize=10, fontweight='bold')
ax_small.legend(loc='upper right', frameon=True)
ax_small.grid(True, which="both", ls="--", alpha=0.5)

# Annotation for ~1000x speedup
ax_small.text(1, 100, "C, C++, and Rust are ~1,000x to 1,200x FASTER\n(Nanoseconds vs Microseconds)",
              ha='center', va='center', fontsize=10, fontweight='bold', color='#B71C1C',
              bbox=dict(boxstyle="round,pad=0.4", fc="#FFEBEE", ec="#D32F2F", lw=1))

small_path = "assets/latency_small_problems.png"
fig_small.tight_layout()
fig_small.savefig(small_path, dpi=300)
print(f"Small problems graph successfully saved to: {small_path}")
