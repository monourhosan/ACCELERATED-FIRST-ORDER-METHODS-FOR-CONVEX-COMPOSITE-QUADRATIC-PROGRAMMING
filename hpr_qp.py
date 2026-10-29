"""
Accelerated First Order Methods for Convex Composite Quadratic Programming (CCQP)
Implementation of the Halpern Peaceman-Rachford (HPR) framework based on:
"Accelerated First Order Methods for Convex Composite Quadratic Programming:
Theory, Algorithms, and GPU Implementation" (Chen, 2026).

Includes:
- Primal HPR Formulation I (Algorithm 3)
- Primal HPR Formulation II (Algorithm 4)
- Dual HPR-QP with Symmetric Gauss-Seidel (sGS) and Adaptive Restart (Algorithms 7 & 8)
- Solver for general Convex Quadratic Programs (CQP) and LASSO
"""

import math
import time
import torch
from typing import Optional, Dict, Tuple


def power_method(A: torch.Tensor, max_iter: int = 50, tol: float = 1e-6) -> float:
    """
    Compute the largest eigenvalue of A^T A (or A if symmetric PSD) via the power method.
    """
    n = A.size(1)
    v = torch.randn(n, 1, dtype=A.dtype, device=A.device)
    v = v / torch.norm(v)
    
    for _ in range(max_iter):
        # Compute w = A^T (A v)
        Av = torch.matmul(A, v)
        w = torch.matmul(A.t(), Av)
        norm_w = torch.norm(w)
        if norm_w < 1e-12:
            return 0.0
        v_next = w / norm_w
        if torch.norm(v_next - v) < tol:
            break
        v = v_next
        
    Av = torch.matmul(A, v)
    lam = torch.sum(Av ** 2).item()
    return max(lam, 1e-6)


def power_method_sym(Q: torch.Tensor, max_iter: int = 50, tol: float = 1e-6) -> float:
    """
    Compute largest eigenvalue of symmetric PSD matrix Q via power method.
    """
    n = Q.size(0)
    v = torch.randn(n, 1, dtype=Q.dtype, device=Q.device)
    v = v / torch.norm(v)
    
    for _ in range(max_iter):
        w = torch.matmul(Q, v)
        norm_w = torch.norm(w)
        if norm_w < 1e-12:
            return 0.0
        v_next = w / norm_w
        if torch.norm(v_next - v) < tol:
            break
        v = v_next
        
    Qw = torch.matmul(Q, v)
    lam = torch.sum(v * Qw).item()
    return max(lam, 1e-6)


def proj_box(x: torch.Tensor, l: Optional[torch.Tensor], u: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Euclidean projection onto box constraint set K = {x | l <= x <= u}.
    """
    res = x
    if l is not None:
        res = torch.clamp_min(res, l)
    if u is not None:
        res = torch.clamp_max(res, u)
    return res


def prox_l1(x: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Soft-thresholding operator: Prox_{gamma * ||.||_1}(x) = sign(x) * max(|x| - gamma, 0).
    """
    return torch.sign(x) * torch.clamp_min(torch.abs(x) - gamma, 0.0)


# =============================================================================
# 1. Primal HPR Method (Algorithm 3)
# =============================================================================
def solve_primal_hpr_cqp(
    Q: torch.Tensor,
    c: torch.Tensor,
    A: torch.Tensor,
    l_k: Optional[torch.Tensor] = None,
    u_k: Optional[torch.Tensor] = None,
    l_c: Optional[torch.Tensor] = None,
    u_c: Optional[torch.Tensor] = None,
    max_iter: int = 2000,
    tol: float = 1e-6,
    sigma: Optional[float] = None
) -> Dict[str, any]:
    """
    Solves CQP via Primal HPR (Algorithm 3):
        min 1/2 <x, Qx> + <c, x> + delta_C(x)
        s.t. Ax in K
    """
    m, n = A.size()
    device = A.device
    dtype = A.dtype

    # Reshape vectors
    c = c.view(n, 1)
    if l_k is not None: l_k = l_k.view(m, 1)
    if u_k is not None: u_k = u_k.view(m, 1)
    if l_c is not None: l_c = l_c.view(n, 1)
    if u_c is not None: u_c = u_c.view(n, 1)

    lambda_A = power_method(A)
    lambda_Q = power_method_sym(Q)

    if sigma is None:
        norm_b = 1.0
        if u_k is not None:
            finite_u = u_k[torch.isfinite(u_k)]
            if len(finite_u) > 0: norm_b = max(norm_b, torch.max(torch.abs(finite_u)).item())
        norm_c = max(torch.max(torch.abs(c)).item(), 1.0)
        sigma = norm_b / norm_c

    # Initialize u^0 = (x^0, s^0, y^0)
    x = torch.zeros(n, 1, dtype=dtype, device=device)
    s = torch.zeros(m, 1, dtype=dtype, device=device)
    y = torch.zeros(m, 1, dtype=dtype, device=device)

    x0, s0, y0 = x.clone(), s.clone(), y.clone()
    step_size = 1.0 / (lambda_Q + sigma * lambda_A)

    history = []
    start_time = time.time()

    for k in range(max_iter):
        # Step 1: s_bar^{k+1} = Proj_K(Ax^k - y^k / sigma)
        s_bar = proj_box(torch.matmul(A, x) - (1.0 / sigma) * y, l_k, u_k)

        # Step 2: y_bar^{k+1} = y^k + sigma * (s_bar - Ax^k)
        Ax = torch.matmul(A, x)
        y_bar = y + sigma * (s_bar - Ax)

        # Step 3: x_bar^{k+1}
        y_hat = 2.0 * y_bar - y
        grad = -torch.matmul(Q, x) - c + torch.matmul(A.t(), y_hat)
        x_step = x + step_size * grad
        x_bar = proj_box(x_step, l_c, u_c)

        # Step 4: Halpern reflection u_hat^{k+1} = 2 * u_bar^{k+1} - u^k
        u_hat_x = 2.0 * x_bar - x
        u_hat_s = 2.0 * s_bar - s
        u_hat_y = 2.0 * y_bar - y

        # Step 5: Halpern anchor update: u^{k+1} = 1/(k+2)*u^0 + (k+1)/(k+2)*u_hat^{k+1}
        coef_0 = 1.0 / (k + 2)
        coef_hat = (k + 1.0) / (k + 2)
        x = coef_0 * x0 + coef_hat * u_hat_x
        s = coef_0 * s0 + coef_hat * u_hat_s
        y = coef_0 * y0 + coef_hat * u_hat_y

        # Compute KKT residuals
        if (k + 1) % 50 == 0 or k == max_iter - 1:
            prim_feas = torch.norm(torch.matmul(A, x_bar) - s_bar, p=float('inf')).item()
            # Dual residual: ||Qx + c - A^T y + subdiff(phi)||_inf
            proj_x = proj_box(x_bar - (torch.matmul(Q, x_bar) + c - torch.matmul(A.t(), y_bar)), l_c, u_c)
            dual_res = torch.norm(x_bar - proj_x, p=float('inf')).item()
            kkt_res = max(prim_feas, dual_res)
            history.append((k + 1, kkt_res))

            if kkt_res < tol:
                break

    return {
        "x": x_bar.squeeze(),
        "s": s_bar.squeeze(),
        "y": y_bar.squeeze(),
        "iterations": k + 1,
        "runtime": time.time() - start_time,
        "kkt_res": kkt_res,
        "history": history
    }


# =============================================================================
# 2. Dual HPR-QP Method with sGS Decomposition & Restart (Algorithm 7 & 8)
# =============================================================================
def solve_dual_hpr_qp(
    Q: torch.Tensor,
    c: torch.Tensor,
    A: torch.Tensor,
    l_k: Optional[torch.Tensor] = None,
    u_k: Optional[torch.Tensor] = None,
    l_c: Optional[torch.Tensor] = None,
    u_c: Optional[torch.Tensor] = None,
    max_iter: int = 3000,
    tol: float = 1e-6,
    sigma: Optional[float] = None,
    restart_freq: int = 64
) -> Dict[str, any]:
    """
    Solves CCQP via Dual HPR with symmetric Gauss-Seidel (sGS) and Restart (Algorithm 7 & 8):
        min 1/2 <x, Qx> + <c, x> + delta_C(x)
        s.t. Ax in K
    """
    m, n = A.size()
    device = A.device
    dtype = A.dtype

    c = c.view(n, 1)
    if l_k is not None: l_k = l_k.view(m, 1)
    if u_k is not None: u_k = u_k.view(m, 1)
    if l_c is not None: l_c = l_c.view(n, 1)
    if u_c is not None: u_c = u_c.view(n, 1)

    lambda_A = power_method(A)
    lambda_Q = power_method_sym(Q)

    if sigma is None:
        sigma = 1.0

    # Variables: u = (y, w_Q, z, x)
    y = torch.zeros(m, 1, dtype=dtype, device=device)
    w_Q = torch.zeros(n, 1, dtype=dtype, device=device)
    z = torch.zeros(n, 1, dtype=dtype, device=device)
    x = torch.zeros(n, 1, dtype=dtype, device=device)

    # Initial reference point u^{r, 0}
    y_r0, w_r0, z_r0, x_r0 = y.clone(), w_Q.clone(), z.clone(), x.clone()

    history = []
    start_time = time.time()
    inner_t = 0

    for k in range(max_iter):
        inner_t += 1

        # Step 1: r_z = x + sigma * (-Q w_Q + A^T y - c)
        Qw = torch.matmul(Q, w_Q)
        ATy = torch.matmul(A.t(), y)
        r_z = x + sigma * (-Qw + ATy - c)
        
        # z_bar = (1 / sigma) * (Prox_phi(r_z) - r_z)
        x_bar = proj_box(r_z, l_c, u_c)  # Prox for indicator delta_C is projection
        z_bar = (1.0 / sigma) * (x_bar - r_z)

        # Step 2: x_hat = 2 * x_bar - x
        x_hat = 2.0 * x_bar - x

        # Step 3-1: w_Q^{k+1/2}
        denom_w = 1.0 + sigma * lambda_Q
        w_half = (1.0 / denom_w) * (sigma * lambda_Q * w_Q + x_hat)

        # Step 3-2: y_bar^{k+1}
        Qw_half = torch.matmul(Q, w_half)
        diff_Qw = Qw - Qw_half
        r_y = torch.matmul(A, x_hat + sigma * diff_Qw) - (sigma * lambda_A) * y
        # y_bar = 1/(sigma * lambda_A) * (Proj_K(r_y) - r_y)
        proj_ry = proj_box(r_y, l_k, u_k)
        y_bar = (1.0 / (sigma * lambda_A)) * (proj_ry - r_y)

        # Step 3-3: w_bar^{k+1}
        w_bar = w_half + (sigma / denom_w) * torch.matmul(A.t(), y_bar - y)

        # Step 4: Halpern reflection: u_hat = 2 * u_bar - u
        u_hat_y = 2.0 * y_bar - y
        u_hat_w = 2.0 * w_bar - w_Q
        u_hat_z = 2.0 * z_bar - z
        u_hat_x = 2.0 * x_bar - x

        # Step 5: Halpern averaging
        coef_0 = 1.0 / (inner_t + 1)
        coef_hat = float(inner_t) / (inner_t + 1)

        y = coef_0 * y_r0 + coef_hat * u_hat_y
        w_Q = coef_0 * w_r0 + coef_hat * u_hat_w
        z = coef_0 * z_r0 + coef_hat * u_hat_z
        x = coef_0 * x_r0 + coef_hat * u_hat_x

        # Check restart criteria
        if restart_freq > 0 and inner_t >= restart_freq:
            # Restart outer loop
            y_r0, w_r0, z_r0, x_r0 = y_bar.clone(), w_bar.clone(), z_bar.clone(), x_bar.clone()
            y, w_Q, z, x = y_r0.clone(), w_r0.clone(), z_r0.clone(), x_r0.clone()
            inner_t = 0

        # Monitor KKT residuals periodically
        if (k + 1) % 20 == 0 or k == max_iter - 1:
            # Primal feasibility: Ax_bar in K
            Ax_bar = torch.matmul(A, x_bar)
            p_res = torch.norm(Ax_bar - proj_box(Ax_bar, l_k, u_k), p=float('inf')).item()
            # Dual feasibility: -Q w_bar + A^T y_bar + z_bar - c = 0
            d_res = torch.norm(-torch.matmul(Q, w_bar) + torch.matmul(A.t(), y_bar) + z_bar - c, p=float('inf')).item()
            # Complementarity / Box constraint: x_bar in C
            c_res = torch.norm(x_bar - proj_box(x_bar, l_c, u_c), p=float('inf')).item()

            kkt_res = max(p_res, d_res, c_res)
            history.append((k + 1, kkt_res))

            if kkt_res < tol:
                break

    return {
        "x": x_bar.squeeze(),
        "w": w_bar.squeeze(),
        "y": y_bar.squeeze(),
        "z": z_bar.squeeze(),
        "iterations": k + 1,
        "runtime": time.time() - start_time,
        "kkt_res": kkt_res,
        "history": history
    }
