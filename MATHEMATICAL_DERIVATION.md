# Mathematical Derivations and Algorithmic Foundations

This document provides rigorous mathematical derivations, operator-theoretic proofs, and convergence guarantees for the **Accelerated First-Order Methods for Convex Composite Quadratic Programming (HPR-QP)** implemented in this repository.

Reference: *Accelerated First Order Methods for Convex Composite Quadratic Programming: Theory, Algorithms, and GPU Implementation* (Chen, 2026).

---

## 1. Convex Composite Quadratic Program (CCQP)

We consider the general convex composite quadratic optimization problem:

$$\begin{aligned}
\min_{x \in \mathbb{R}^n} \quad & f(x) := \frac{1}{2} \langle x, Q x \rangle + \langle c, x \rangle + \delta_C(x) \\
\text{subject to} \quad & A x \in K
\end{aligned}$$

where:
- $Q \in \mathbb{S}_+^n$ is a real symmetric positive semidefinite (PSD) Hessian matrix ($\lambda_{\min}(Q) \ge 0$). Note that $Q$ is **not** assumed to be strictly positive definite; $Q$ may be singular ($\mathrm{rank}(Q) < n$).
- $c \in \mathbb{R}^n$ is the linear objective vector.
- $C = \{ x \in \mathbb{R}^n \mid l_c \le x \le u_c \}$ is a closed, convex, hyper-rectangular box constraint set with indicator function $\delta_C(x)$ ($0$ if $x \in C$, $+\infty$ otherwise).
- $A \in \mathbb{R}^{m \times n}$ represents general linear constraints.
- $K = \{ v \in \mathbb{R}^m \mid l_k \le v \le u_k \}$ is the convex constraint box on $Ax$.

---

## 2. Singular Hessian & Restricted Wolfe Dual Formulation

When $Q$ is singular or ill-conditioned, standard dual formulations require the pseudo-inverse $Q^\dagger$, which is numerically prohibitive for large dimensions ($n > 10^5$). 

To circumvent matrix inversion, we introduce an auxiliary vector $w$ restricted to the range space of $Q$:

$$w \in \mathrm{Range}(Q)$$

The Lagrangian function associated with the primal problem is:

$$L(x, y) = \frac{1}{2} \langle x, Q x \rangle + \langle c, x \rangle + \delta_C(x) - \langle y, A x \rangle + \delta_K^*(y)$$

where $\delta_K^*(y) = \sup_{v \in K} \langle y, v \rangle$ is the Fenchel support function of $K$.

By enforcing the first-order stationarity condition with respect to $x$ in the smooth unconstrained component:

$$Q x - A^T y + c + z = 0, \quad z \in \partial \delta_C(x)$$

we parameterize $Q x = Q w$ where $w \in \mathrm{Range}(Q)$. This produces the **Restricted Wolfe Dual**:

$$\begin{aligned}
\max_{w \in \mathrm{Range}(Q), \, y \in \mathbb{R}^m} \quad & -\frac{1}{2} \langle w, Q w \rangle - \delta_K^*(y) - \delta_C^*(-Qw + A^T y - c)
\end{aligned}$$

### Key Property: Exact Subspace Equivalence
Since $w \in \mathrm{Range}(Q)$, the nullspace component $\mathrm{ker}(Q)$ has zero energy:
$$\forall v \in \mathrm{ker}(Q), \quad \langle v, Q v \rangle = 0 \quad \text{and} \quad Q v = 0$$
Hence, solving over $\mathrm{Range}(Q)$ guarantees the exact dual solution without requiring regularization $\varepsilon I$ or matrix inversion.

---

## 3. Symmetric Gauss-Seidel (sGS) Block Decoupling

The dual objective couples the variables $w$ and $y$ through the non-smooth term $\delta_C^*$. In standard ADMM, updating $(w, y)$ simultaneously requires inverting the coupled operator:

$$\begin{bmatrix} Q + \sigma^{-1} I & -A^T \\ -A & \lambda_A I \end{bmatrix}$$

which destroys matrix sparsity.

### Forward-Backward sGS Splitting
We apply the symmetric Gauss-Seidel decomposition:
1. **Forward sweep**: Update primal-dual proximal point $x^{k+1/2}$ and intermediate dual $w^{k+1/2}$.
2. **Intermediate sweep**: Update multiplier $y^{k+1}$ via scalar coordinate-wise projection onto the interval $[l_k, u_k]$.
3. **Backward sweep**: Update final dual direction $w^{k+1}$ using the updated residual from $y^{k+1}$.

$$\begin{aligned}
r_z^k &= x^k + \sigma \left( -Q w^k + A^T y^k - c \right) \\
\bar{x}^{k} &= \Pi_C(r_z^k) = \mathrm{clamp}(r_z^k, l_c, u_c) \\
\bar{z}^k &= \frac{1}{\sigma} \left( \bar{x}^k - r_z^k \right) \\
\hat{x}^k &= 2 \bar{x}^k - x^k \\
w^{k+1/2} &= \frac{1}{1 + \sigma \lambda_Q} \left( \sigma \lambda_Q w^k + \hat{x}^k \right)
\end{aligned}$$

Then the linear constraint multipliers update:

$$\begin{aligned}
r_y^k &= A \left( \hat{x}^k + \sigma Q(w^k - w^{k+1/2}) \right) - \sigma \lambda_A y^k \\
\bar{y}^k &= \frac{1}{\sigma \lambda_A} \left( \Pi_K(r_y^k) - r_y^k \right) \\
\bar{w}^k &= w^{k+1/2} + \frac{\sigma}{1 + \sigma \lambda_Q} A^T (\bar{y}^k - y^k)
\end{aligned}$$

---

## 4. Halpern Peaceman-Rachford Acceleration

Let $T_{\mathrm{PR}}: \mathcal{H} \to \mathcal{H}$ denote the non-expansive Peaceman-Rachford operator associated with the monotone inclusion problem $0 \in (F_1 + F_2)(u)$.

Standard Peaceman-Rachford iteration:
$$u^{k+1} = T_{\mathrm{PR}}(u^k)$$
exhibits an ergodic $\mathcal{O}(1/k)$ rate and may fail to converge if $T_{\mathrm{PR}}$ is merely non-expansive (non-strictly contractive).

### Halpern Fixed-Point Iteration (Chen, Theorem 3.2)
We anchor the iteration to an initial base point $u^0 = (x^0, w^0, y^0, z^0)$:

$$u^{k+1} = \alpha_k u^0 + (1 - \alpha_k) T_{\mathrm{PR}}(u^k), \quad \text{with} \quad \alpha_k = \frac{1}{k + 2}$$

Expanding the Peaceman-Rachford reflection $T_{\mathrm{PR}}(u) = 2 \bar{u} - u$:

$$u^{k+1} = \frac{1}{k+2} u^0 + \frac{k+1}{k+2} (2 \bar{u}^k - u^k)$$

### Convergence Theorem
Under the stepsize rule $\sigma \le \frac{1}{\sqrt{\lambda_Q^2 + \lambda_A^2}}$, the sequence of iterates satisfies:

$$\|u^k - T_{\mathrm{PR}}(u^k)\|^2 \le \frac{4 \|u^0 - u^*\|^2}{(k + 1)^2} = \mathcal{O}\left(\frac{1}{k^2}\right)$$

This achieves the **optimal non-ergodic convergence rate** $\mathcal{O}(1/k^2)$ for general maximal monotone inclusions.

---

## 5. Adaptive Restart Mechanism (Algorithm 8)

While the Halpern iteration guarantees $\mathcal{O}(1/k^2)$ sublinear convergence on general convex QPs, when local metric regularity or strong convexity holds, an adaptive restart mechanism yields **linear convergence**:

$$\|u^k - u^*\| \le C \cdot \rho^k, \quad \rho \in (0, 1)$$

### Restart Condition
Every $N_{\mathrm{restart}}$ iterations (typically $N \in [16, 64]$) or whenever the monotonic residual progress ratio satisfies:

$$\frac{\|\mathrm{Res}_{k}\|_\infty}{\|\mathrm{Res}_{k_0}\|_\infty} \le \gamma_{\mathrm{restart}} \quad (\gamma \approx 0.5)$$

the anchor vector is reset to the current iterate:

$$u^0 \leftarrow \bar{u}^k, \quad \text{and inner step counter } t \leftarrow 0$$

This clears stale momentum and restarts the Halpern acceleration from a point substantially closer to $u^*$.

---

## 6. KKT Optimality Metric and Residual Definition

At any iterate $(x, w, y, z)$, the Karush-Kuhn-Tucker (KKT) optimality conditions are:

$$\begin{aligned}
\text{Primal Stationarity:} \quad & Q w - A^T y - z + c = 0 \\
\text{Primal Feasibility:} \quad & A x \in K \iff A x - \Pi_K(A x) = 0 \\
\text{Box Feasibility:} \quad & x \in C \iff x - \Pi_C(x) = 0 \\
\text{Subdifferential Multiplier:} \quad & z \in \partial \delta_C(x)
\end{aligned}$$

We compute the maximal component-wise KKT residual norm:

$$\|\mathrm{Res}\|_\infty = \max \left( \|Q w - A^T y - z + c\|_\infty, \; \|A x - \Pi_K(A x)\|_\infty, \; \|x - \Pi_C(x)\|_\infty \right)$$

When $\|\mathrm{Res}\|_\infty \le \epsilon_{\mathrm{tol}}$, the iterate is guaranteed to satisfy all primal and dual optimality conditions within tolerance $\epsilon_{\mathrm{tol}}$.
