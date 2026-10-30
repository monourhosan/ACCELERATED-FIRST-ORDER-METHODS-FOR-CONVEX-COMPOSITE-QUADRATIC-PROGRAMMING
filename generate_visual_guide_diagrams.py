"""
Generate visual architectural diagrams:
1. Algorithm workflow: HPR-QP with sGS decomposition and Halpern acceleration.
2. CPU vs GPU Language Performance Matrix across Python, C, C++, and Rust.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ----------------------------------------------------------------------
# 1. ALGORITHM ARCHITECTURE & WORKFLOW DIAGRAM
# ----------------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(15, 8.5), dpi=300)
ax1.set_xlim(0, 15)
ax1.set_ylim(0, 10)
ax1.axis('off')

# Title
ax1.text(7.5, 9.6, "HPR-QP Algorithm Architecture & Computational Pipeline", 
         ha='center', va='center', fontsize=16, fontweight='bold', color='#1A237E')
ax1.text(7.5, 9.15, "Accelerated First-Order Method with Block Symmetric Gauss-Seidel (sGS) & Halpern Anchor", 
         ha='center', va='center', fontsize=11, style='italic', color='#37474F')

def draw_box(ax, x, y, w, h, title, subtitle, color_edge, color_bg, tag=None):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.2",
                                  linewidth=2, edgecolor=color_edge, facecolor=color_bg)
    ax.add_patch(rect)
    if tag:
        tag_box = patches.FancyBboxPatch((x + 0.15, y + h - 0.35), 1.2, 0.3,
                                         boxstyle="round,pad=0.05", fc=color_edge, ec="none")
        ax.add_patch(tag_box)
        ax.text(x + 0.75, y + h - 0.2, tag, ha='center', va='center',
                fontsize=8, fontweight='bold', color='white')
    ax.text(x + w/2, y + h/2 + 0.1, title, ha='center', va='center',
            fontsize=10.5, fontweight='bold', color='#212121')
    ax.text(x + w/2, y + h/2 - 0.35, subtitle, ha='center', va='center',
            fontsize=8.5, color='#424242')

# Draw Problem Formulation Box
draw_box(ax1, 0.5, 7.3, 14.0, 1.4,
         r"Primal Convex Composite Problem:   $\min_{x} \frac{1}{2} \langle x, Qx \rangle + \langle c, x \rangle + \phi(x) \quad \text{s.t.} \quad Ax \in \mathcal{K}$",
         r"Restricted Wolfe Dual on $\mathcal{W} = \mathrm{Range}(Q)$: Eliminates singular null space divergence ($O(1/k)$ rate)",
         "#3F51B5", "#E8EAF6", tag="FORMULATION")

# Step 1: Forward Proximal & Box Projection
draw_box(ax1, 0.5, 5.0, 4.2, 1.7,
         "Step 1: Forward Proximal Step",
         r"$\nabla f(x) = Qw - A^T y + c$" + "\n" + r"$r_z = x - \sigma \nabla f(x)$" + "\n" + r"$\bar{x} = \Pi_{\mathcal{C}}(r_z) = \mathrm{clip}(r_z, l_c, u_c)$",
         "#00897B", "#E0F2F1", tag="STAGE 1")

# Step 2: Peaceman-Rachford Reflection
draw_box(ax1, 5.4, 5.0, 4.2, 1.7,
         "Step 2: Peaceman-Rachford Reflection",
         r"$\hat{x} = 2\bar{x} - x \quad \text{(Reflection)}$" + "\n" +
         r"$\bar{z} = \frac{1}{\sigma}(\bar{x} - r_z)$" + "\n" +
         r"$\bar{w}^{k+1/2} = \frac{\sigma \lambda_Q w + \hat{x}}{1 + \sigma \lambda_Q}$",
         "#FB8C00", "#FFF3E0", tag="STAGE 2")

# Step 3: Cyclic Block sGS
draw_box(ax1, 10.3, 5.0, 4.2, 1.7,
         "Step 3: Symmetric Gauss-Seidel (sGS)",
         r"$\bar{w}^{k+1/2} \;\to\; \bar{y}^{k+1} \;\to\; \bar{w}^{k+1}$" + "\n" +
         r"$\bar{y} = \Pi_{\mathcal{K}^*}(y - \sigma \lambda_A A \mathrm{temp}_n)$" + "\n" +
         r"$\bar{w} = \bar{w}^{1/2} + \frac{\sigma}{1+\sigma\lambda_Q} A^T \Delta y$",
         "#E53935", "#FFEBEE", tag="STAGE 3")

# Step 4: Halpern Anchor Averaging
draw_box(ax1, 2.5, 2.2, 4.8, 1.7,
         "Step 4: Halpern Anchor Averaging",
         r"Anchor to initial point $u^0 = (x^0, w^0, z^0, y^0)$:" + "\n" +
         r"$u^{k+1} = \frac{1}{t+2} u^0 + \frac{t+1}{t+2} (2\bar{u}^{k+1} - u^k)$" + "\n" +
         r"Guarantees optimal nonergodic $O(1/k)$ rate",
         "#8E24AA", "#F3E5F5", tag="STAGE 4")

# Step 5: Adaptive Restarts
draw_box(ax1, 8.2, 2.2, 4.8, 1.7,
         "Step 5: Adaptive Restart Trigger",
         r"Surrogate merit residual $\tilde{R}_{r,t} = \|u_Q^{r,t} - \bar{u}_Q^{r,t+1}\|_M$" + "\n" +
         r"If $\tilde{R}_{r,t} > \gamma \tilde{R}_{r,0}$ or $t \geq \tau_r$:" + "\n" +
         r"Reset anchor $u^0 \leftarrow \bar{u}^{k+1}$, restart counter $t \leftarrow 0$",
         "#3949AB", "#E8EAF6", tag="STAGE 5")

# Output: Optimal Solution
draw_box(ax1, 3.5, 0.3, 8.0, 1.2,
         r"Optimal Solution Output:  $x^* = \bar{x}^{K}, \quad \text{Objective: } \frac{1}{2}\langle x^*, Qx^* \rangle + \langle c, x^* \rangle$",
         r"Bitwise verified identical across Python, C, C++, and Rust on CPU and GPU",
         "#2E7D32", "#E8F5E9", tag="CONVERGENCE")

# Connect with arrows
arrow_props = dict(arrowstyle="->", lw=2, color="#455A64")

# Flow: Stage 1 -> Stage 2 -> Stage 3
ax1.annotate("", xy=(5.4, 5.85), xytext=(4.7, 5.85), arrowprops=arrow_props)
ax1.annotate("", xy=(10.3, 5.85), xytext=(9.6, 5.85), arrowprops=arrow_props)

# Flow: Stage 3 -> Stage 4
ax1.annotate("", xy=(4.9, 3.9), xytext=(12.4, 5.0),
             arrowprops=dict(arrowstyle="->", lw=2, color="#455A64", connectionstyle="arc3,rad=0.25"))

# Flow: Stage 4 -> Stage 5
ax1.annotate("", xy=(8.2, 3.05), xytext=(7.3, 3.05), arrowprops=arrow_props)

# Flow: Stage 5 -> Output
ax1.annotate("", xy=(7.5, 1.5), xytext=(10.6, 2.2),
             arrowprops=dict(arrowstyle="->", lw=2, color="#2E7D32", connectionstyle="arc3,rad=-0.15"))

# Flow: Loop back to Stage 1
ax1.annotate("", xy=(0.5, 5.85), xytext=(2.5, 3.05),
             arrowprops=dict(arrowstyle="->", lw=2, color="#1976D2", linestyle="--", connectionstyle="arc3,rad=0.35"))
ax1.text(0.6, 4.3, r"Next Iterate" + "\n" + r"$(k \leftarrow k+1)$", fontsize=8.5, fontweight='bold', color="#1976D2")

plt.tight_layout()
fig1.savefig("assets/algorithm_architecture.png", dpi=300)
print("Algorithm architecture diagram saved: assets/algorithm_architecture.png")


# ----------------------------------------------------------------------
# 2. CPU vs GPU PERFORMANCE MATRIX & HARDWARE BOTTLENECK ANALYSIS
# ----------------------------------------------------------------------
fig2, ((ax_cpu1, ax_cpu2), (ax_gpu1, ax_gpu2)) = plt.subplots(2, 2, figsize=(15, 10), dpi=300)

# Colors
c_py = "#3776AB"
c_c = "#555555"
c_cpp = "#00599C"
c_rust = "#CE412B"
c_gpu = "#76B900"

# Panel 1: Small Problems (Log Scale Latency)
probs = ["Example 5.1\n(Singular Q)", "Example 5.2\n(Ill-Conditioned)", "HS21\n(Constrained QP)"]
x_pos = np.arange(len(probs))
width = 0.2

py_us = [9850, 7420, 9000]
c_us  = [12.0, 7.0, 9.0]
cpp_us = [11.7, 6.6, 8.4]
rust_us = [11.0, 6.0, 8.0]

ax_cpu1.bar(x_pos - 1.5*width, py_us, width, label='Python 3.12 (CPython)', color=c_py)
ax_cpu1.bar(x_pos - 0.5*width, c_us, width, label='C99 (/O2)', color=c_c)
ax_cpu1.bar(x_pos + 0.5*width, cpp_us, width, label='C++20 (/O2)', color=c_cpp)
ax_cpu1.bar(x_pos + 1.5*width, rust_us, width, label='Rust 2021 (--release)', color=c_rust)

ax_cpu1.set_yscale('log')
ax_cpu1.set_title("Panel A: Small Problem Latency (Microseconds, Log Scale)\nWinner: Rust & C/C++ (~1,000x Faster)", fontsize=11, fontweight='bold')
ax_cpu1.set_ylabel("Execution Time (µs, Log Scale)", fontsize=9, fontweight='bold')
ax_cpu1.set_xticks(x_pos)
ax_cpu1.set_xticklabels(probs, fontsize=9)
ax_cpu1.legend(loc='upper right', fontsize=8)
ax_cpu1.grid(True, which="both", ls="--", alpha=0.5)

# Panel 2: Large CPU Sparse Problems (CVXQP1_L 300 iters)
cpu_labels = [
    "Python\n(SciPy CSR)",
    "Rust\n(Single-Core)",
    "C++20\n(Single-Core)",
    "C99\n(Single-Core)",
    "Accelerated\nC99 (8T)",
    "Accelerated\nC++20 (8T)",
    "Accelerated\nPython C-Bridge"
]
cpu_times = [178.48, 136.31, 92.23, 75.28, 52.92, 42.66, 40.22]
colors_panel2 = [c_py, c_rust, c_cpp, c_c, "#888888", "#2E86AB", "#1E88E5"]

bars2 = ax_cpu2.bar(cpu_labels, cpu_times, color=colors_panel2, edgecolor='black', linewidth=0.8)
for bar, t in zip(bars2, cpu_times):
    y = bar.get_height()
    ax_cpu2.text(bar.get_x() + bar.get_width()/2.0, y + 2.5, f"{t:.1f}ms",
                 ha='center', va='bottom', fontsize=8, fontweight='bold')

ax_cpu2.set_title("Panel B: Large Problem on CPU (CVXQP1_L, 300 Iters)\nWinner: Compiled C/C++ with 8-Thread OpenMP & AVX2", fontsize=11, fontweight='bold')
ax_cpu2.set_ylabel("Execution Time (ms)", fontsize=9, fontweight='bold')
ax_cpu2.set_ylim(0, 205)
ax_cpu2.tick_params(axis='x', rotation=30, labelsize=8)
ax_cpu2.grid(axis='y', linestyle='--', alpha=0.6)

# Panel 3: GPU OpenCL Architecture (VRAM Residency)
ax_gpu1.axis('off')
ax_gpu1.set_title("Panel C: GPU VRAM Memory Residency Architecture\nZero PCIe Overhead during Iterations", fontsize=11, fontweight='bold')

# Host RAM box
rect_host = patches.FancyBboxPatch((0.05, 0.55), 0.35, 0.38, boxstyle="round,pad=0.03", fc="#ECEFF1", ec="#455A64", lw=1.5)
ax_gpu1.add_patch(rect_host)
ax_gpu1.text(0.225, 0.85, "Host CPU & RAM", ha='center', va='center', fontweight='bold', fontsize=10, color="#263238")
ax_gpu1.text(0.225, 0.70, "Python / C / C++ / Rust Runtime\nInitial Matrix Loader\nFinal Solution Collector", ha='center', va='center', fontsize=8, color="#37474F")

# PCIe Bus Box
rect_pcie = patches.FancyBboxPatch((0.44, 0.65), 0.12, 0.18, boxstyle="round,pad=0.02", fc="#FFF9C4", ec="#FBC02D", lw=1.5)
ax_gpu1.add_patch(rect_pcie)
ax_gpu1.text(0.50, 0.77, "PCIe Bus", ha='center', va='center', fontweight='bold', fontsize=8.5, color="#F57F17")
ax_gpu1.text(0.50, 0.69, "0 Bytes / Iter\n(Bypassed!)", ha='center', va='center', fontsize=7.5, fontweight='bold', color="#D84315")

# GPU VRAM Box
rect_vram = patches.FancyBboxPatch((0.60, 0.15), 0.38, 0.78, boxstyle="round,pad=0.03", fc="#E8F5E9", ec=c_gpu, lw=2.0)
ax_gpu1.add_patch(rect_vram)
ax_gpu1.text(0.79, 0.86, "NVIDIA GeForce MX250 VRAM", ha='center', va='center', fontweight='bold', fontsize=10, color="#1B5E20")
ax_gpu1.text(0.79, 0.76, "Persistent CSR Buffers: d_P, d_A, d_AT\n24 Working Iterate Buffers: d_x, d_w, d_z, d_y\nZero Host-Device Round-Trip", ha='center', va='center', fontsize=7.5, color="#2E7D32")

# GPU Kernels
rect_k1 = patches.FancyBboxPatch((0.62, 0.48), 0.34, 0.18, boxstyle="round,pad=0.02", fc="#C8E6C9", ec="#388E3C", lw=1.0)
ax_gpu1.add_patch(rect_k1)
ax_gpu1.text(0.79, 0.58, "Massively Parallel SpMV Kernel\nRow-partitioned compute across 384 CUDA cores", ha='center', va='center', fontsize=7.5, color="#1B5E20")

rect_k2 = patches.FancyBboxPatch((0.62, 0.22), 0.34, 0.20, boxstyle="round,pad=0.02", fc="#C8E6C9", ec="#388E3C", lw=1.0)
ax_gpu1.add_patch(rect_k2)
ax_gpu1.text(0.79, 0.32, "Fused Projection & sGS Kernels\nBox Clipping, Reflection & Halpern Average\nDirectly in GPU Register Space", ha='center', va='center', fontsize=7.5, color="#1B5E20")

# Panel 4: Multi-Language GPU Hardware Parity
gpu_labels = ["Old CPU Python", "Old CPU Rust", "Old CPU C++", "Old CPU C", "GPU C (OpenCL)", "GPU C++20", "GPU Rust (OpenCL)"]
gpu_times = [178.48, 136.31, 92.23, 75.28, 81.16, 80.45, 80.05]
colors_panel4 = [c_py, c_rust, c_cpp, c_c, c_gpu, "#00897B", "#E65100"]

bars4 = ax_gpu2.bar(gpu_labels, gpu_times, color=colors_panel4, edgecolor='black', linewidth=0.8)
for bar, t in zip(bars4, gpu_times):
    y = bar.get_height()
    ax_gpu2.text(bar.get_x() + bar.get_width()/2.0, y + 2.5, f"{t:.1f}ms",
                 ha='center', va='bottom', fontsize=8, fontweight='bold')

ax_gpu2.set_title("Panel D: GPU Hardware Silicon Parity (CVXQP1_L, 300 Iters)\nWinner: Parity across Rust, C++, and C (~80 ms / 0.267 ms/iter)", fontsize=11, fontweight='bold')
ax_gpu2.set_ylabel("Execution Time (ms)", fontsize=9, fontweight='bold')
ax_gpu2.set_ylim(0, 205)
ax_gpu2.tick_params(axis='x', rotation=35, labelsize=8)
ax_gpu2.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
fig2.savefig("assets/cpu_vs_gpu_performance_matrix.png", dpi=300)
print("Performance matrix diagram saved: assets/cpu_vs_gpu_performance_matrix.png")
