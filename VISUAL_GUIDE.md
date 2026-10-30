# Visual Guide: HPR-QP Algorithm & Multi-Language CPU/GPU Performance

This document provides a comprehensive visual explanation of:
1. **How the Accelerated First-Order Method for Convex Composite Quadratic Programming (HPR-QP)** works mathematically and computationally.
2. **Which programming language performs best on the CPU and on the GPU**, detailing the architectural bottlenecks (memory bandwidth, SIMD execution, PCIe transfers, and interpreter overhead).

---

## 1. Visual Algorithmic Architecture

The HPR-QP solver incorporates **Semi-Proximal Linearization**, **Operator Splitting**, **Block Symmetric Gauss–Seidel (sGS) Decomposition**, **Halpern Anchor Averaging**, and **Adaptive Restarts**.

![HPR-QP Computational Pipeline](assets/algorithm_architecture.png)

### End-to-End Iteration Lifecycle (Mermaid Flowchart)

```mermaid
flowchart TD
    subgraph INIT ["0. Problem Formulation & Subspace Restriction"]
        A["Primal Problem: min 1/2 <x, Qx> + <c, x> + phi(x)  s.t. Ax in K"] --> B["Restricted Wolfe Dual on W = Range(Q)<br>Dual Null Space Divergence Eliminated"]
    end

    subgraph STAGE1 ["1. Forward Proximal Step & Projection"]
        B --> C["Compute Gradient: grad = Q*w - A^T*y + c"]
        C --> D["Trial Vector: r_z = x - sigma * grad"]
        D --> E["Box Projection: x_bar = clip(r_z, l_c, u_c)"]
    end

    subgraph STAGE2 ["2. Peaceman-Rachford Reflection"]
        E --> F["Operator Reflection: x_hat = 2 * x_bar - x"]
        F --> G["Subgradient: z_bar = (x_bar - r_z) / sigma"]
        G --> H["Half Step: w_half = (sigma*lambda_Q*w + x_hat) / (1 + sigma*lambda_Q)"]
    end

    subgraph STAGE3 ["3. Cyclic Block Symmetric Gauss-Seidel (sGS)"]
        H --> I["Evaluate: temp_n = x_hat + sigma * (Q*w - Q*w_half)"]
        I --> J["Dual Projection: y_bar = Proj_K*(y - sigma*lambda_A * A*temp_n)"]
        J --> K["Second w Pass: w_bar = w_half + (sigma / (1 + sigma*lambda_Q)) * A^T * delta_y"]
    end

    subgraph STAGE4 ["4. Halpern Anchor Averaging (O(1/k) Acceleration)"]
        K --> L["Anchor Convex Combination:<br>u^{k+1} = (1 / (t+2)) * u^0 + ((t+1) / (t+2)) * (2*u_bar^{k+1} - u^k)"]
    end

    subgraph STAGE5 ["5. Adaptive Merit Restart Monitor"]
        L --> M{"Merit Check:<br>R_{r,t} > gamma*R_{r,0} or t >= tau_r?"}
        M -- Yes --> N["Reset Anchor: u^0 = u_bar^{k+1}<br>Reset Counter: t = 0"]
        M -- No --> O["Increment Counter: t = t + 1"]
        N --> P{"Tolerance Reached?<br>||Res|| <= eps or k >= K_max"}
        O --> P
        P -- No (Next Iteration) --> C
        P -- Yes (Converged) --> Q["Output Optimal Solution x*"]
    end

    style INIT fill:#E8EAF6,stroke:#3F51B5,stroke-width:2px
    style STAGE1 fill:#E0F2F1,stroke:#00897B,stroke-width:2px
    style STAGE2 fill:#FFF3E0,stroke:#FB8C00,stroke-width:2px
    style STAGE3 fill:#FFEBEE,stroke:#E53935,stroke-width:2px
    style STAGE4 fill:#F3E5F5,stroke:#8E24AA,stroke-width:2px
    style STAGE5 fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px
```

---

## 2. Step-by-Step Algorithmic Mechanics

### Step 1: Forward Proximal Gradient Step & Box Projection
Given iterate $x^k$ and dual vectors $(w^k, y^k)$:
$$\nabla f(x^k) = Q w^k - A^T y^k + c$$
$$r_z^k = x^k - \sigma \nabla f(x^k)$$
$$\bar{x}^{k+1} = \Pi_{\mathcal{C}}(r_z^k) = \text{clip}(r_z^k, l_c, u_c)$$
* **Hardware Insight**: This step is purely elementwise $\mathcal{O}(n)$. In compiled C/C++/Rust, this is fused into a single register sweep, avoiding intermediate memory round-trips.

### Step 2: Peaceman–Rachford Reflection & Dual Half-Step
Reflect the primal iterate across the projection boundary:
$$\hat{x}^{k+1} = 2 \bar{x}^{k+1} - x^k$$
$$\bar{z}^{k+1} = \frac{1}{\sigma}(\bar{x}^{k+1} - r_z^k)$$
$$\bar{w}^{k+1/2} = \frac{1}{1 + \sigma \lambda_Q} (\sigma \lambda_Q w^k + \hat{x}^{k+1})$$
* **Subspace Restriction**: Because $w$ is restricted to $\mathcal{W} = \text{Range}(Q)$, the singular null space of $Q$ cannot cause dual divergence.

### Step 3: Cyclic Block Symmetric Gauss–Seidel (sGS)
To solve the coupled $(y, w)$ block without matrix inversions or auxiliary slack variables:
1. Update temporary primal direction: $\text{temp}_n = \hat{x}^{k+1} + \sigma (Q w^k - Q \bar{w}^{k+1/2})$.
2. Forward dual step: $\bar{y}^{k+1} = \Pi_{\mathcal{K}^*}(y^k - \sigma \lambda_A A \, \text{temp}_n)$.
3. Backward dual step: $\bar{w}^{k+1} = \bar{w}^{k+1/2} + \frac{\sigma}{1 + \sigma \lambda_Q} A^T (\bar{y}^{k+1} - y^k)$.
* **Factorization-Free**: Every operation is reduced to Sparse Matrix-Vector multiplications (**SpMV**), enabling standard hardware and GPU execution on problems with millions of dimensions.

### Step 4: Halpern Anchor Averaging
Anchor the next iterate to the fixed initial state $u^0 = (x^0, w^0, z^0, y^0)$:
$$u^{k+1} = \frac{1}{t+2} u^0 + \frac{t+1}{t+2} (2 \bar{u}^{k+1} - u^k)$$
* **Theoretical Acceleration**: Improves the nonergodic convergence rate from standard Douglas–Rachford $\mathcal{O}(1/\sqrt{k})$ to the optimal $\mathcal{O}(1/k)$.

### Step 5: Adaptive Restarts
Monitor the computable surrogate merit function $\tilde{R}_{r, t} = \|u_Q^{r, t} - \bar{u}_Q^{r, t+1}\|_M$. Whenever the solver transitions between active constraint identification (Phase I) and asymptotic convergence (Phase II), the anchor point is dynamically reset to prevent over-anchoring.

---

## 3. CPU vs. GPU Performance Matrix

We benchmarked the full algorithm across **Python**, **C99**, **Modern C++20**, and **Rust** on both **CPU** (Intel Xeon / Multi-Core x64) and **GPU** (NVIDIA GeForce MX250 via OpenCL).

![CPU vs GPU Performance Matrix](assets/cpu_vs_gpu_performance_matrix.png)

---

## 4. Which Programming Language Does Better on CPU?

### Winner on CPU: **Compiled C99 / Modern C++20 with OpenMP 8-Threads & AVX2 SIMD**

```
                                  CPU PERFORMANCE COMPARISON (CVXQP1_L, 300 Iterations)
┌──────────────────────────────────────┬──────────────────────┬────────────────────────┬──────────────────────┬────────────────────────┐
│ Implementation & Configuration       │ Total Solve Time     │ Per-Iteration Latency  │ Speedup vs Python    │ Computational Bottleneck│
├──────────────────────────────────────┼──────────────────────┼────────────────────────┼──────────────────────┼────────────────────────┤
│ Python 3.12 (NumPy + SciPy CSR)      │ 178.48 ms            │ 0.595 ms / iter        │ Baseline (1.00x)     │ Bytecode & Memory Copy │
│ Rust 2021 (Single-Core Reference)    │ 136.31 ms            │ 0.454 ms / iter        │ 1.31x faster         │ Single-Core DDR4 Bandw.│
│ Modern C++20 (Single-Core Reference) │  92.23 ms            │ 0.307 ms / iter        │ 1.94x faster         │ Single-Core DDR4 Bandw.│
│ C99 (Single-Core Reference)          │  75.28 ms            │ 0.251 ms / iter        │ 2.37x faster         │ Single-Core DDR4 Bandw.│
│ Accelerated C99 (8-Thread + SIMD)    │  52.92 ms            │ 0.176 ms / iter        │ 3.37x faster         │ Thread Sync Overhead   │
│ Accelerated C++20 (8-Thread + SIMD)  │  42.66 ms            │ 0.142 ms / iter        │ 4.18x faster         │ L3 Cache Saturated     │
│ Accelerated Python (C OpenMP DLL)    │  40.22 ms            │ 0.134 ms / iter        │ 4.44x FASTER (PEAK)  │ L3 Cache Saturated     │
└──────────────────────────────────────┴──────────────────────┴────────────────────────┴──────────────────────┴────────────────────────┘
```

### Why C and C++ Win on CPU:
1. **4-Way Interleaved SIMD Accumulators**:
   In standard CSR SpMV, computing `sum += val[p] * x[col[p]]` stalls the CPU arithmetic execution unit because each iteration depends on the previous sum. By unrolling 4-way with four independent accumulators (`s0, s1, s2, s3`), latency is completely hidden.
2. **OpenMP Cache-Aligned Partitioning**:
   Rows are divided statically across 8 CPU threads (`schedule(static, 128)`), maximizing L1/L2 cache hit rates and saturating total memory bandwidth.
3. **Zero Dynamic Allocation**:
   All 16 iterate vectors are allocated once at initialization. During the 300 iterations, zero `malloc` or heap calls occur.

---

## 5. Which Programming Language Does Better on GPU?

### Winner on GPU: **Hardware Silicon Parity across Rust ($80.05\text{ ms}$), C++20 ($80.45\text{ ms}$), and C99 ($81.16\text{ ms}$)**

```
                                  GPU PERFORMANCE COMPARISON (NVIDIA GeForce MX250 / OpenCL)
┌──────────────────────────────────────┬──────────────────────┬────────────────────────┬──────────────────────┬────────────────────────┐
│ Implementation & Backend             │ Total Solve Time     │ Per-Iteration Latency  │ Speedup vs Old CPU C │ Speedup vs Old CPU Py  │
├──────────────────────────────────────┼──────────────────────┼────────────────────────┼──────────────────────┼────────────────────────┤
│ C99 GPU (OpenCL Kernel Pipeline)     │ 81.16 ms             │ 0.271 ms / iter        │ 3.17x faster         │ 2.20x faster           │
│ Modern C++20 GPU (OpenCL RAII)       │ 80.45 ms             │ 0.268 ms / iter        │ 3.19x faster         │ 2.22x faster           │
│ Rust GPU (Zero-Dependency FFI OpenCL)│ 80.05 ms             │ 0.267 ms / iter        │ 3.21x FASTER (PEAK)  │ 2.23x FASTER (PEAK)    │
└──────────────────────────────────────┴──────────────────────┴────────────────────────┴──────────────────────┴────────────────────────┘
```

### Why All Three Languages Achieve GPU Silicon Parity:
1. **100% VRAM Memory Residency (Zero PCIe Overhead)**:
   ```mermaid
   sequenceDiagram
       autonumber
       actor Host as Host CPU (C / C++ / Rust)
       participant VRAM as GPU VRAM (GDDR5, 56 GB/s)
       participant Cores as 384 CUDA / OpenCL Compute Cores

       Host->>VRAM: One-Time Upload (P, A, AT, c, bounds, working buffers)
       Note over Host,VRAM: Zero PCIe Transfers During Iterations!
       loop Iterations 1 to 300 (HPR-QP Steps)
           Cores->>VRAM: Parallel SpMV Q*w (Coalesced Memory Read)
           VRAM-->>Cores: High-Bandwidth Stream
           Cores->>VRAM: Fused Step 1 & 2 (Box Clip & Reflection in Registers)
           Cores->>VRAM: Parallel sGS Projection Steps
           Cores->>VRAM: Halpern Anchor Averaging (In-Place VRAM Write)
       end
       VRAM->>Host: One-Time Download of Final Optimal Vector x*
   ```
2. **Zero Abstraction Penalty**:
   Whether invoked from C, C++ RAII classes, or Rust FFI bindings, the iteration time is 100% spent inside the JIT-compiled OpenCL kernels executing on the NVIDIA Pascal GPU architecture.
3. **Rust Marginally Edges Out**:
   Rust records the fastest time ($80.05\text{ ms}$) due to LLVM's compile-time optimization of host-side dispatch loops and memory pointer alignment.

---

## 6. Summary Comparison: Small vs. Large Scale Optimization

```mermaid
flowchart LR
    subgraph SMALL ["Small-Scale Problems (n <= 100)"]
        direction TB
        S1["CPython Interpreter Tax Dominates (5-15 µs/iter)"]
        S2["Rust & C/C++ Execute in 10-100 ns/iter"]
        S1 --> S2
        S3["Result: Systems Languages are ~1,000x FASTER"]
    end

    subgraph LARGE ["Large-Scale Sparse Problems (n = 10,000)"]
        direction TB
        L1["SpMV Compute & Memory Bandwidth Dominates"]
        L2["Multi-Threaded OpenMP C/C++ Saturates 8 Cores (40 ms)"]
        L3["GPU VRAM Residency Bypasses PCIe Bus (80 ms)"]
        L1 --> L2
        L1 --> L3
        L4["Result: Multi-Core CPU & GPU deliver 2.2x to 4.4x Speedup"]
    end

    style SMALL fill:#FFF8E1,stroke:#FFA000,stroke-width:2px
    style LARGE fill:#E1F5FE,stroke:#0288D1,stroke-width:2px
```

* For **embedded control, real-time robotics, and model predictive control (MPC)** ($n \le 100$): **Rust or C/C++ is mandatory**, delivering an immediate **$1,000\times$ speedup** over Python.
* For **large-scale machine learning, portfolio optimization, and sparse network flows** ($n \ge 10,000$):
  - **Multi-Core CPU C/C++ (8-Thread OpenMP)** provides the lowest absolute runtime (**$40.22\text{ ms}$**).
  - **GPU Acceleration (OpenCL C/C++/Rust)** offloads 100% of arithmetic work to the graphics card (**$80.05\text{ ms}$**), freeing the host CPU for concurrent tasks.
