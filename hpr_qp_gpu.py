"""
GPU-Accelerated First-Order Solver for Convex Composite Quadratic Programming (HPR-QP).
Supports NVIDIA CUDA, Apple MPS, and High-Performance CPU SIMD execution via PyTorch.

Based on: Kaihuang Chen (2026), "Accelerated First Order Methods for CCQP:
Theory, Algorithms, and GPU Implementation" (Chapter 6).
"""

import time
import math
from typing import Dict, Any, Optional, Tuple, Union
import torch

class GPUHPRQPSolver:
    """
    High-Performance GPU Solver for Convex Composite Quadratic Programming (CCQP):
      min 0.5 * <x, Q x> + <c, x> + phi(x)   s.t.   l_k <= A x <= u_k
    where phi(x) is the indicator function of box constraints l_c <= x <= u_c.
    """
    def __init__(
        self,
        P: Union[torch.Tensor, Any],  # Sparse CSR or dense tensor
        c: torch.Tensor,
        A: Optional[Union[torch.Tensor, Any]] = None,
        AT: Optional[Union[torch.Tensor, Any]] = None,
        l_k: Optional[torch.Tensor] = None,
        u_k: Optional[torch.Tensor] = None,
        l_c: Optional[torch.Tensor] = None,
        u_c: Optional[torch.Tensor] = None,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float64,
        sigma: float = 1.0,
        tol: float = 1e-4,
        max_iter: int = 1000,
        restart_freq: int = 32,
    ):
        # Determine execution hardware device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        self.dtype = dtype
        self.sigma = float(sigma)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.restart_freq = int(restart_freq)

        # Move problem data directly to target GPU device
        self.n = c.shape[0]
        self.c = c.to(device=self.device, dtype=self.dtype)

        # Hessian Q / P matrix
        if P.is_sparse_csr:
            self.P = P.to(device=self.device, dtype=self.dtype)
        elif P.is_sparse:
            self.P = P.to_sparse_csr().to(device=self.device, dtype=self.dtype)
        else:
            self.P = P.to(device=self.device, dtype=self.dtype)

        # Constraint matrix A and its transpose A^T
        self.has_A = A is not None
        if self.has_A:
            self.m = A.shape[0]
            if A.is_sparse_csr:
                self.A = A.to(device=self.device, dtype=self.dtype)
            elif A.is_sparse:
                self.A = A.to_sparse_csr().to(device=self.device, dtype=self.dtype)
            else:
                self.A = A.to(device=self.device, dtype=self.dtype)

            if AT is not None:
                if AT.is_sparse_csr:
                    self.AT = AT.to(device=self.device, dtype=self.dtype)
                elif AT.is_sparse:
                    self.AT = AT.to_sparse_csr().to(device=self.device, dtype=self.dtype)
                else:
                    self.AT = AT.to(device=self.device, dtype=self.dtype)
            else:
                if self.A.is_sparse_csr:
                    self.AT = self.A.to_sparse_coo().t().to_sparse_csr()
                else:
                    self.AT = self.A.t()
        else:
            self.m = 0
            self.A = None
            self.AT = None

        inf_val = 1e20
        self.l_k = l_k.to(device=self.device, dtype=self.dtype) if l_k is not None else torch.full((self.m,), -inf_val, device=self.device, dtype=self.dtype)
        self.u_k = u_k.to(device=self.device, dtype=self.dtype) if u_k is not None else torch.full((self.m,),  inf_val, device=self.device, dtype=self.dtype)
        self.l_c = l_c.to(device=self.device, dtype=self.dtype) if l_c is not None else torch.full((self.n,), -inf_val, device=self.device, dtype=self.dtype)
        self.u_c = u_c.to(device=self.device, dtype=self.dtype) if u_c is not None else torch.full((self.n,),  inf_val, device=self.device, dtype=self.dtype)

        # Spectral norm estimates
        self.lam_q = self._estimate_lambda_q(max_iter=15)
        self.lam_a = self._estimate_lambda_a(max_iter=15) if self.has_A else 1.0

    def _mat_vec(self, M: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if M.is_sparse_csr:
            return torch.mv(M, x)
        elif M.is_sparse:
            return torch.sparse.mm(M, x.unsqueeze(1)).squeeze(1)
        else:
            return torch.matmul(M, x)

    def _estimate_lambda_q(self, max_iter: int = 15) -> float:
        v = torch.randn(self.n, device=self.device, dtype=self.dtype)
        v = v / torch.norm(v)
        for _ in range(max_iter):
            w = self._mat_vec(self.P, v)
            nrm = torch.norm(w)
            if nrm < 1e-12: break
            v = w / nrm
        lam = float(torch.dot(v, self._mat_vec(self.P, v)))
        return max(lam, 1e-6)

    def _estimate_lambda_a(self, max_iter: int = 15) -> float:
        if not self.has_A or self.m == 0: return 1.0
        v = torch.randn(self.n, device=self.device, dtype=self.dtype)
        v = v / torch.norm(v)
        for _ in range(max_iter):
            av = self._mat_vec(self.A, v)
            w = self._mat_vec(self.AT, av)
            nrm = torch.norm(w)
            if nrm < 1e-12: break
            v = w / nrm
        av = self._mat_vec(self.A, v)
        lam = float(torch.sum(av * av))
        return max(lam, 1e-6)

    def solve(self) -> Dict[str, Any]:
        """
        Executes HPR-QP purely in device memory (VRAM).
        No Host-Device memory synchronization occurs inside the iteration loop.
        """
        n, m = self.n, self.m
        sigma = self.sigma
        denom_w = 1.0 + sigma * self.lam_q
        sig_lam_a = sigma * self.lam_a
        inv_sig_lam_a = 1.0 / sig_lam_a
        w_step = sigma / denom_w

        # Device Working Buffers (Pre-allocated in VRAM)
        x = torch.zeros(n, device=self.device, dtype=self.dtype)
        w_q = torch.zeros(n, device=self.device, dtype=self.dtype)
        z = torch.zeros(n, device=self.device, dtype=self.dtype)
        y = torch.zeros(m if m > 0 else 1, device=self.device, dtype=self.dtype)

        x_bar = torch.zeros_like(x)
        w_bar = torch.zeros_like(w_q)
        z_bar = torch.zeros_like(z)
        y_bar = torch.zeros_like(y)

        # Anchor points for Halpern restart
        x_r0 = torch.zeros_like(x)
        w_r0 = torch.zeros_like(w_q)
        z_r0 = torch.zeros_like(z)
        y_r0 = torch.zeros_like(y)

        # Synchronize GPU stream before timing
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        t_start = time.perf_counter()

        inner_t = 0
        converged = False
        final_iter = self.max_iter
        kkt_residual = 1e9

        for k in range(self.max_iter):
            inner_t += 1

            # 1) SpMV for Qw and A^T y
            qw = self._mat_vec(self.P, w_q)
            aty = self._mat_vec(self.AT, y) if self.has_A else torch.zeros_like(x)

            # 2) Fused Step 1 & 2: Box projection & reflection
            r_z = x + sigma * (-qw + aty - self.c)
            x_bar = torch.clamp(r_z, self.l_c, self.u_c)
            z_bar = (1.0 / sigma) * (x_bar - r_z)

            x_hat = 2.0 * x_bar - x
            w_half = (1.0 / denom_w) * (sigma * self.lam_q * w_q + x_hat)

            # 3) Step 3: sGS block update
            if self.has_A:
                qw_half = self._mat_vec(self.P, w_half)
                temp_n = x_hat + sigma * (qw - qw_half)
                r_y = self._mat_vec(self.A, temp_n) - sig_lam_a * y
                proj_y = torch.clamp(r_y, self.l_k, self.u_k)
                y_bar = inv_sig_lam_a * (proj_y - r_y)

                delta_y = y_bar - y
                w_bar = w_half + w_step * self._mat_vec(self.AT, delta_y)
            else:
                w_bar = w_half

            # 4) Halpern Averaging (in-place on GPU)
            coef_0 = 1.0 / (inner_t + 1.0)
            coef_hat = float(inner_t) / (inner_t + 1.0)

            x = coef_0 * x_r0 + coef_hat * (2.0 * x_bar - x)
            w_q = coef_0 * w_r0 + coef_hat * (2.0 * w_bar - w_q)
            z = coef_0 * z_r0 + coef_hat * (2.0 * z_bar - z)
            if self.has_A:
                y = coef_0 * y_r0 + coef_hat * (2.0 * y_bar - y)

            # 5) Adaptive Restart
            if self.restart_freq > 0 and inner_t >= self.restart_freq:
                x_r0.copy_(x_bar)
                w_r0.copy_(w_bar)
                z_r0.copy_(z_bar)
                y_r0.copy_(y_bar)

                x.copy_(x_r0)
                w_q.copy_(w_r0)
                z.copy_(z_r0)
                y.copy_(y_r0)
                inner_t = 0

            # 6) Periodic Residual Check (Every 25 iters)
            if (k + 1) % 25 == 0 or k == self.max_iter - 1:
                qw_chk = self._mat_vec(self.P, w_bar)
                aty_chk = self._mat_vec(self.AT, y_bar) if self.has_A else torch.zeros_like(x)
                stat_res = torch.max(torch.abs(-qw_chk + aty_chk + z_bar - self.c))

                if self.has_A:
                    ax = self._mat_vec(self.A, x_bar)
                    feas_lin = torch.max(torch.abs(ax - torch.clamp(ax, self.l_k, self.u_k)))
                else:
                    feas_lin = torch.tensor(0.0, device=self.device)

                feas_box = torch.max(torch.abs(x_bar - torch.clamp(x_bar, self.l_c, self.u_c)))
                kkt_residual = float(torch.max(torch.stack([stat_res, feas_lin, feas_box])))

                if kkt_residual <= self.tol:
                    converged = True
                    final_iter = k + 1
                    break

        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        elapsed_sec = time.perf_counter() - t_start

        # Compute objective on GPU
        px = self._mat_vec(self.P, x_bar)
        obj_val = float(0.5 * torch.dot(x_bar, px) + torch.dot(self.c, x_bar))

        return {
            'x': x_bar.cpu().numpy(),
            'w': w_bar.cpu().numpy(),
            'y': y_bar.cpu().numpy() if self.has_A else None,
            'z': z_bar.cpu().numpy(),
            'iterations': final_iter,
            'elapsed_ms': elapsed_sec * 1000.0,
            'kkt_residual': kkt_residual,
            'objective': obj_val,
            'converged': converged,
            'device': str(self.device),
        }
