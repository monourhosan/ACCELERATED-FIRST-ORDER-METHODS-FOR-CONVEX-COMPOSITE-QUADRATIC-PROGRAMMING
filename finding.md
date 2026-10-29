# Findings and Key Insights

This document synthesizes the core theoretical discoveries, architectural analyses, and empirical insights obtained from implementing and benchmarking both the **Accelerated First-Order Methods for Convex Composite Quadratic Programming (HPR-QP)** and the **"Attention Is All You Need" Transformer** architecture.

---

## 1. Key Findings on HPR-QP Optimization

### Finding 1: The Theoretical Acceleration from $\mathcal{O}(1/\sqrt{k})$ to $\mathcal{O}(1/k)$
Standard operator-splitting methods (such as the Alternating Direction Method of Multipliers, ADMM, and the Douglas–Rachford splitting scheme, DR) have long been constrained by a sublinear nonergodic convergence rate of:
$$\|\mathcal{R}(u^k)\| = \mathcal{O}\left(\frac{1}{\sqrt{k}}\right)$$
By incorporating the **Halpern iteration** with anchor parameter $\frac{1}{k+2}$ into the Peaceman–Rachford operator reflection:
$$u^{k+1} = \frac{1}{k+2} u^0 + \frac{k+1}{k+2} (2 \bar{u}^{k+1} - u^k)$$
the theoretical complexity improves quadratically to:
$$\|\mathcal{R}(\bar{u}^{k+1})\| = \mathcal{O}\left(\frac{1}{k}\right)$$
In our experiments, this theoretical speedup translates to solving test instances (like `HS21`, `HS35`, `GENHS28`, and `TAME`) in under 100 to 200 iterations, whereas standard DR often requires thousands of iterations.

---

### Finding 2: The Critical Necessity of the Restricted Wolfe Dual
When minimizing a quadratic objective with a semidefinite Hessian $Q \succeq 0$ (such as LASSO where $Q = A^T A$ with $m < n$, or singular QP instances like Example 5.1):
1. **Classical Wolfe Dual Breakdown**: The classical dual has an unconstrained linear term along the null space of $Q$ ($\text{ker}(Q)$). The dual solution set becomes **unbounded**, and dual variables diverge.
2. **Restricted Subspace Solution**: By restricting dual updates strictly to $\mathcal{W} = \text{Range}(Q)$, the null space of $Q$ is eliminated:
   * $Q$ becomes **strictly positive definite** on $\mathcal{W}$.
   * Level sets are strictly compact.
   * As proved in Example 5.1 ($Q = \text{diag}(1, 2, 0)$), HPR-QP converged to machine precision ($10^{-10}$) in only **80 iterations**, completely immune to the zero eigenvalue.

---

### Finding 3: Decoupling via Symmetric Gauss–Seidel (sGS) Prevents Slowing Down
Solving multi-block problems typically presents a dilemma:
* Standard 3-block ADMM can diverge without very small step sizes proportional to the minimum positive eigenvalue of $Q$.
* Introducing auxiliary slack variables ($s = Ax$) increases the dimension of the KKT matrix and typically slows practical convergence by $5\times$ to $20\times$ (as shown in Chapter 5, Figure 5.6 of the thesis).

The **Block Symmetric Gauss–Seidel (sGS)** decomposition resolves this dilemma:
* It solves the coupled $(y, w)$ block through a cyclic sequence $\bar{w}^{k+1/2} \to \bar{y}^{k+1} \to \bar{w}^{k+1}$.
* It provides an exact, uncoupled, closed-form update at each step without introducing any slack variables or requiring coupled system inversions.

---

### Finding 4: Factorization-Free Linearization Solves the RAM Bottleneck
In large-scale applications like financial volatility regression (`log1p.E2006` with $n = 4,272,227$ features) or QAP relaxations ($n = 67,108,864$ variables):
* Interior-point solvers and direct ADMM solvers must compute and factorize an $(n \times n)$ matrix, which requires **$>70$ Terabytes of RAM** and immediately triggers `OutOfMemory` crashes.
* HPR-QP’s semi-proximal linearization ($S_y = \lambda_A I - AA^T$, $S_w = Q(\lambda_Q I - Q)$) eliminates all matrix factorizations. Every step is reduced to sparse matrix-vector multiplications (**SpMV**), allowing problems with millions of dimensions to run smoothly on standard hardware or GPUs.

---

### Finding 5: Two-Phase Convergence and the Power of Adaptive Restarts
In difficult instances (such as `QSCAGR7` or `QSCSD1`):
* First-order methods exhibit a distinct **Phase I (Slow Identification)** where the algorithm searches for the active constraint manifold, followed by **Phase II (Fast Asymptotic Linear Convergence)** once the active manifold is identified.
* Running fixed Halpern acceleration indefinitely can over-anchor the iterates to the origin $u^0$.
* Monitoring the computable surrogate merit function $\tilde{R}_{r, t} = \|u_Q^{r, t} - \bar{u}_Q^{r, t+1}\|_M$ and triggering adaptive restarts resets the anchor point, allowing the solver to rapidly exit Phase I and achieve linear convergence.

---

## 2. Key Findings on Transformer Attention Architecture

### Finding 6: Permutation Invariance and Positional Encodings
* Self-attention computes dot products $QK^T$ solely based on feature representations. Without positional encodings, swapping token order produces identical output representations.
* Sinusoidal encodings $PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{\text{model}}})$ provide a smooth geometric coordinate system that enables the model to learn relative distance patterns without adding trainable parameters.

### Finding 7: The Critical Role of Scaled Dot-Product & Causal Masking
* **Scaling by $1/\sqrt{d_k}$**: As head dimension $d_k$ grows, dot products grow in magnitude, which pushes the softmax function into regions with extremely small gradients. The $\frac{1}{\sqrt{d_k}}$ scaling factor keeps the softmax variance at 1, preventing gradient vanishing.
* **Causal Masking**: In autoregressive sequence generation, applying a lower-triangular mask (filling upper-triangular logits with $-10^9$) ensures the model cannot "cheat" by attending to future tokens, enabling exact 100% autoregressive greedy reproduction on unseen sequences.
