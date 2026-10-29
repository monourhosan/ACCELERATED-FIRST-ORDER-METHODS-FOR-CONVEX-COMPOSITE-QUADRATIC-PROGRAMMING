# Result and Finding

This document compiles the empirical results, experimental evaluations, and quantitative findings across all tests conducted for the **Accelerated First-Order Methods for Convex Composite Quadratic Programming (HPR-QP)** and the **"Attention Is All You Need" Transformer** implementations.

---

## 1. HPR-QP Numerical Verification on Thesis Examples

We replicated and tested the synthetic verification instances from Section 5.2.3 of the thesis (*Chen, 2026*): **Example 5.1** (singular Hessian $Q$) and **Example 5.2** (strongly convex but severely ill-conditioned $Q$).

### Test Specifications
* **Example 5.1 (Singular $Q$ with null space)**:
  $$Q = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 2 & 0 \\ 0 & 0 & 0 \end{pmatrix}, \quad c = \begin{pmatrix} -1 \\ -1 \\ -1 \end{pmatrix}, \quad \mathcal{K} = \mathbb{R}^3, \quad C = (-\infty, 1]^3$$
  *Analytical Solution*: $x^* = (1.0, 0.5, 1.0)^T$, with active box constraints on $x_1$ and $x_3$.
* **Example 5.2 (Ill-conditioned $Q$ with condition number $\approx 100$)**:
  $$Q = \begin{pmatrix} 1 & 0.1 \\ 0.1 & 10 \end{pmatrix}, \quad c = \begin{pmatrix} -1 \\ -1 \end{pmatrix}, \quad \mathcal{K} = \mathbb{R}^2, \quad C = \mathbb{R}^2$$
  *Analytical Solution*: $x^* = -Q^{-1} c = (0.990991, 0.090090)^T$.

### Quantitative Results
| Test Instance | Problem Dimension | Iterations to Converge | Final KKT Residual | Computed Solution $x^*$ | Expected Solution $x^*$ | Test Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Example 5.1 (Singular $Q$)** | $n=3, m=3$ | **80 iters** | **$2.32 \times 10^{-10}$** | `[1.000000, 0.500000, 1.000000]` | `[1.0, 0.5, 1.0]` | **PASSED (100% Match)** |
| **Example 5.2 (Ill-conditioned)** | $n=2, m=2$ | **100 iters** | **$8.17 \times 10^{-11}$** | `[0.990991, 0.090090]` | `[0.990991, 0.090090]` | **PASSED (100% Match)** |

*Observation*: By restricting updates to $\mathcal{W} = \text{Range}(Q)$, the singular null space was rendered completely benign, matching the rapid convergence reported in Figure 5.5 of the thesis.

---

## 2. Real-World Maros–Mészáros Benchmark Results

We benchmarked HPR-QP against diverse problem classes from the industry-standard **Maros–Mészáros QP dataset** located in `Data/maros_meszaros_qpbenchmark-main/data/`.

### Experimental Setup
* **Solver**: Dual HPR-QP with sGS decomposition and adaptive restart ([`hpr_qp.py`](hpr_qp.py))
* **Tolerance**: $\epsilon = 10^{-4}$ on maximal KKT residual $\|\mathcal{R}(u)\|_\infty$
* **Maximum Iterations**: 3,000
* **Restart Frequency**: $\tau_r = 32$ iterations
* **Hardware**: Intel Xeon Platinum CPU @ 2.60 GHz

### Full Benchmark Performance Table

| Instance | Problem Class / Origin | Vars ($n$) | Constraints ($m_{\text{lin}}$) | Nonzeros ($Q+A$) | Status | Iters | Runtime (s) | Final KKT Residual | Optimal Objective Value |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TAME** | Simple QP test | 2 | 1 | 4 | `OPTIMAL` | **20** | **0.0036** | $4.77 \times 10^{-15}$ | `0.000000` |
| **HS21** | Hock-Schittkowski #21 | 2 | 1 | 4 | `OPTIMAL` | **40** | **0.0090** | $3.91 \times 10^{-5}$ | `0.040000` |
| **HS35** | Hock-Schittkowski #35 | 3 | 1 | 7 | `OPTIMAL` | **80** | **0.0153** | $8.05 \times 10^{-6}$ | `-8.888886` |
| **HS76** | Hock-Schittkowski #76 | 4 | 3 | 14 | `OPTIMAL` | **80** | **0.0164** | $2.25 \times 10^{-5}$ | `-4.681463` |
| **GENHS28** | Generalized HS28 | 10 | 8 | 32 | `OPTIMAL` | **200** | **0.0388** | $6.65 \times 10^{-5}$ | `0.927155` |
| **ZECEVIC2** | Chemical equilibrium | 2 | 2 | 6 | `OPTIMAL` | **140** | **0.0284** | $5.55 \times 10^{-5}$ | `-4.125111` |
| **DUAL2** | Dual structural program | 96 | 1 | 192 | `OPTIMAL` | **320** | **0.0629** | $5.84 \times 10^{-5}$ | `0.033747` |
| **DUAL1** | Dual structural program | 85 | 1 | 170 | `OPTIMAL` | **340** | **0.0666** | $8.43 \times 10^{-5}$ | `0.035042` |
| **S268** | Structural optimization | 5 | 5 | 25 | `OPTIMAL` | **340** | **0.0685** | $1.55 \times 10^{-5}$ | `-13565.491050` |
| **HS51** | Hock-Schittkowski #51 | 5 | 3 | 18 | `OPTIMAL` | **460** | **0.0910** | $9.39 \times 10^{-5}$ | `-6.000000` |
| **HS52** | Hock-Schittkowski #52 | 5 | 3 | 18 | `OPTIMAL` | **1020** | **0.2078** | $9.01 \times 10^{-5}$ | `-0.674237` |
| **DUAL3** | Dual structural program | 111 | 1 | 222 | `OPTIMAL` | **2200** | **0.5985** | $9.82 \times 10^{-5}$ | `0.135742` |
| **DUAL4** | Dual structural program | 75 | 1 | 150 | `NEAR_CONV` | 3000 | 0.8152 | $1.75 \times 10^{-4}$ | `0.745944` |
| **QSCSD1** | Dynamic structural design | 760 | 77 | 3,148 | `IN_PROGRESS`| 3000 | 2.7067 | $3.69 \times 10^{-2}$ | `4.348337` |

---

## 3. "Attention Is All You Need" Transformer Copy Task Results

We tested the from-scratch PyTorch Transformer architecture on an autoregressive integer sequence copy task.

### Model Configuration
* **Architecture**: 2 Encoder layers, 2 Decoder layers, 4 Attention heads ($d_k = 16$).
* **Hidden Dimension ($d_{\text{model}}$)**: 64.
* **Feed-Forward Dimension ($d_{\text{ff}}$)**: 256.
* **Vocabulary Size**: 11 tokens (`0: <PAD>`, `1: <SOS>`, `2..10: Tokens`).
* **Sequence Length**: 10 tokens.
* **Total Parameters**: 235,851.

### Training Progression Over 15 Epochs
| Epoch | Training Loss | Validation Loss | Throughput (Tokens/s) | Note |
| :---: | :---: | :---: | :---: | :--- |
| **Pre-Train** | — | — | — | Random outputs (gibberish) |
| **1** | 0.8421 | 0.0801 | 10,352 | Rapid initial alignment |
| **2** | 0.0474 | 0.0143 | 9,750 | Sharp error drop |
| **3** | 0.0265 | 0.0067 | 7,333 | Sub-0.01 validation loss |
| **4** | 0.0194 | 0.0094 | 8,496 | Stable attention weights |
| **5** | 0.0135 | 0.0429 | 9,576 | Minor transient fluctuation |
| **6** | 0.0139 | 0.0016 | 10,101 | Re-stabilization |
| **7** | 0.0067 | 0.0000 | 9,614 | Near-perfect accuracy |
| **8 – 15** | **0.0000** | **0.0000** | **10,633** | **Exact zero cross-entropy loss** |

### Evaluation on Unseen Test Sequences (Greedy Autoregressive Decoding)
Each unseen test sequence was decoded token-by-token:
1. `Input : [3, 8, 5, 2, 9, 4, 7, 6, 10, 2]`  
   `Output: [3, 8, 5, 2, 9, 4, 7, 6, 10, 2]` $\rightarrow$ **MATCH (100%)**
2. `Input : [7, 7, 2, 9, 5, 3, 10, 4, 8, 6]`  
   `Output: [7, 7, 2, 9, 5, 3, 10, 4, 8, 6]` $\rightarrow$ **MATCH (100%)**
3. `Input : [10, 9, 8, 7, 6, 5, 4, 3, 2, 10]`  
   `Output: [10, 9, 8, 7, 6, 5, 4, 3, 2, 10]` $\rightarrow$ **MATCH (100%)**
4. `Input : [2, 3, 4, 5, 6, 7, 8, 9, 10, 2]`  
   `Output: [2, 3, 4, 5, 6, 7, 8, 9, 10, 2]` $\rightarrow$ **MATCH (100%)**
5. `Input : [9, 2, 4, 8, 10, 7, 3, 5, 6, 4]`  
   `Output: [9, 2, 4, 8, 10, 7, 3, 5, 6, 4]` $\rightarrow$ **MATCH (100%)**

**Final Copy Accuracy**: **5/5 (100.0%)**.
