# Accelerated First-Order Methods for CCQP & Transformer Architecture from Scratch

This repository contains clean, from-scratch implementations of two cutting-edge algorithmic foundations in machine learning and optimization:

1. **Accelerated First-Order Methods for Convex Composite Quadratic Programming (CCQP / HPR-QP)**  
   *Based on Kaihuang Chen (2026), "Accelerated First Order Methods for Convex Composite Quadratic Programming: Theory, Algorithms, and GPU Implementation"* ([`8892.pdf`](8892.pdf)).
2. **"Attention Is All You Need" (Transformer) from Scratch in PyTorch**  
   *Based on Vaswani et al. (2017), "Attention Is All You Need"* with full Encoder-Decoder architecture, sinusoidal positional encodings, multi-head self-attention, and an autoregressive toy copy task.

---

## Repository Structure

```
├── 8892.pdf                 # Reference doctoral thesis on CCQP & HPR methods
├── hpr_qp.py                # Accelerated First-Order Solvers for CCQP (Primal & Dual HPR-QP)
├── test_hpr_qp.py           # Unit tests reproducing paper examples (Ex 5.1 & 5.2)
├── transformer_scratch.py   # Complete Transformer from scratch in pure PyTorch
├── train_copy_task.py       # Autoregressive sequence copy task benchmark & training loop
├── README.md                # Project documentation
└── .gitignore
```

---

## Part 1: Accelerated First-Order Methods for CCQP (HPR-QP)

### Problem Formulation
Convex Composite Quadratic Programming (CCQP):
$$\min_{x \in \mathbb{R}^n} \frac{1}{2} \langle x, Qx \rangle + \langle c, x \rangle + \phi(x) \quad \text{s.t.} \quad Ax \in \mathcal{K}$$
where:
- $Q \succeq 0$ is a self-adjoint positive semidefinite linear operator.
- $\phi(x)$ is a closed, proper, convex function (e.g., indicator $\delta_C(x)$ for box constraints, or $\lambda \|x\|_1$ for LASSO).
- $\mathcal{K} = \{ y \in \mathbb{R}^m \mid l \le y \le u \}$ is a polyhedral set.

### Implemented Methods in `hpr_qp.py`
- **Primal HPR (Algorithm 3)**: Linearized semi-proximal Halpern Peaceman-Rachford method enjoying an $O(1/k)$ KKT residual convergence rate.
- **Dual HPR-QP with sGS (Algorithms 7 & 8)**: Operates on the *restricted Wolfe dual*:
  $$\min_{y, w \in \text{Range}(Q), z} \frac{1}{2} \langle w, Qw \rangle + \delta_{\mathcal{K}}^*(-y) + \phi^*(-z) \quad \text{s.t.} \quad -Qw + A^T y + z = c$$
  incorporating symmetric Gauss-Seidel (sGS) block decomposition and adaptive restart.
- **Factorization-Free Updates**: Relies exclusively on matrix-vector multiplications and proximal/projection operations, eliminating costly matrix inversions and enabling seamless GPU acceleration.

### Quick Start: Running HPR-QP Tests
```bash
python test_hpr_qp.py
```
**Verification Output:**
```text
============================================================
Testing Example 5.1 from Chapter 5.2.3 (Singular Q):
Dual HPR-QP solved in 80 iters | KKT residual: 2.32e-10
Optimal x: [1.0, 0.5, 1.0]
Expected x: [1.0, 0.5, 1.0]
Example 5.1 PASSED!
============================================================
Testing Example 5.2 from Chapter 5.2.3 (Ill-conditioned Q):
Dual HPR-QP solved in 100 iters | KKT residual: 8.17e-11
Optimal x: [0.99099, 0.09009]
Expected x: [0.99099, 0.09009]
Example 5.2 PASSED!
============================================================
ALL HPR-QP TESTS COMPLETED SUCCESSFULLY!
```

---

## Part 2: "Attention Is All You Need" (Transformer) from Scratch

A ground-up implementation of Vaswani et al. (2017) built without using `nn.Transformer` or `nn.MultiheadAttention`.

### Key Components in `transformer_scratch.py`
1. **Sinusoidal Positional Encoding**:
   $$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)$$
2. **Scaled Dot-Product Attention**:
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
3. **Multi-Head Attention**:
   $$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$$
4. **Position-wise Feed-Forward Networks**:
   $$\text{FFN}(x) = \max(0, x W_1 + b_1)W_2 + b_2$$
5. **Residual Connections & Layer Normalization**:
   Post-LN sublayer connections $\text{LayerNorm}(x + \text{Dropout}(\text{Sublayer}(x)))$.
6. **Masked Decoder**: Look-ahead causal masking combined with padding mask.

### Quick Start: Running Copy Task Training
```bash
python train_copy_task.py
```

### In-Action Demonstration & Training Progress
A 2-layer, 4-head Transformer ($d_{\text{model}} = 64$) was trained on random integer sequences of length 10:

```text
======================================================================
TRAINING 'ATTENTION IS ALL YOU NEED' TRANSFORMER ON TOY COPY TASK
======================================================================
Constructed Transformer model from scratch (235,851 parameters).
Testing model BEFORE training (random weights):
  Input sequence : [2, 4, 6, 8, 10, 3, 5, 7, 9, 2]
  Untrained pred : [6, 6, 7, 6,  3, 6, 3, 6, 3, 6]  (Random)
----------------------------------------------------------------------
Starting training for 15 epochs...
Epoch  1/15 | Train Loss: 0.8421 | Val Loss: 0.0801 | Tokens/sec: 10352
Epoch  2/15 | Train Loss: 0.0474 | Val Loss: 0.0143 | Tokens/sec: 9750
Epoch  3/15 | Train Loss: 0.0265 | Val Loss: 0.0067 | Tokens/sec: 7333
Epoch  5/15 | Train Loss: 0.0135 | Val Loss: 0.0429 | Tokens/sec: 9576
Epoch  7/15 | Train Loss: 0.0067 | Val Loss: 0.0000 | Tokens/sec: 9614
Epoch  8/15 | Train Loss: 0.0000 | Val Loss: 0.0000 | Tokens/sec: 9770
...
Epoch 15/15 | Train Loss: 0.0000 | Val Loss: 0.0000 | Tokens/sec: 10633
----------------------------------------------------------------------
Evaluating on 5 unseen test sequences (Autoregressive Greedy Decoding):
Sample 1:
  Input  : [3, 8, 5, 2, 9, 4, 7, 6, 10, 2]
  Output : [3, 8, 5, 2, 9, 4, 7, 6, 10, 2]  -> PASSED (100% Exact Match)
Sample 2:
  Input  : [7, 7, 2, 9, 5, 3, 10, 4, 8, 6]
  Output : [7, 7, 2, 9, 5, 3, 10, 4, 8, 6]  -> PASSED (100% Exact Match)
Sample 3:
  Input  : [10, 9, 8, 7, 6, 5, 4, 3, 2, 10]
  Output : [10, 9, 8, 7, 6, 5, 4, 3, 2, 10] -> PASSED (100% Exact Match)
Sample 4:
  Input  : [2, 3, 4, 5, 6, 7, 8, 9, 10, 2]
  Output : [2, 3, 4, 5, 6, 7, 8, 9, 10, 2]  -> PASSED (100% Exact Match)
Sample 5:
  Input  : [9, 2, 4, 8, 10, 7, 3, 5, 6, 4]
  Output : [9, 2, 4, 8, 10, 7, 3, 5, 6, 4]  -> PASSED (100% Exact Match)
======================================================================
Overall Accuracy: 5/5 (100.0%)
```

---

## Requirements
- Python 3.8+
- PyTorch 2.0+
- NumPy
