"""
Sparse Feature Selection via LASSO (L1-Regularized Least Squares) using HPR-QP.

Problem Formulation:
    min_{x} 1/2 ||A x - b||_2^2 + lambda * ||x||_1
Equivalently formulated as a Quadratic Program via variable split x = u - v with u, v >= 0:
    min_{u, v >= 0} 1/2 [u; v]^T [A^T A, -A^T A; -A^T A, A^T A] [u; v] + [-A^T b + lambda; A^T b + lambda]^T [u; v]

When m < n (fat matrix with more features than samples), the Gram matrix Q = A^T A has a
significant null space. HPR-QP's restricted Wolfe dual guarantees convergence without divergence along ker(Q).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from hpr_qp import solve_dual_hpr_qp


def run_lasso_benchmark():
    print("=" * 70)
    print("  REAL-WORLD APPLICATION: HIGH-DIMENSIONAL SPARSE LASSO REGRESSION")
    print("=" * 70)

    torch.manual_seed(101)
    num_samples = 150    # m
    num_features = 300   # n (m < n, singular Gram matrix)
    true_sparsity = 15   # Only 15 features are non-zero

    print(f"Sample Count (m)       : {num_samples}")
    print(f"Feature Dimension (n)  : {num_features} (Underdetermined: m < n)")
    print(f"Ground Truth Nonzeros  : {true_sparsity} features")

    # Generate design matrix A and sparse ground truth beta*
    A = torch.randn(num_samples, num_features, dtype=torch.float64)
    beta_star = torch.zeros(num_features, dtype=torch.float64)
    active_indices = torch.randperm(num_features)[:true_sparsity]
    beta_star[active_indices] = torch.randn(true_sparsity, dtype=torch.float64) * 3.0

    # Add Gaussian observation noise
    noise = torch.randn(num_samples, dtype=torch.float64) * 0.05
    b = A @ beta_star + noise

    AtA = A.t() @ A
    Atb = A.t() @ b

    # Regularization parameter
    lam = 0.05 * torch.norm(Atb, p=float('inf')).item()
    print(f"L1 Regularization (lam): {lam:.4f}")

    # Build 2n variable split: z = [u; v] >= 0
    Q_block = torch.cat([
        torch.cat([AtA, -AtA], dim=1),
        torch.cat([-AtA, AtA], dim=1)
    ], dim=0)

    ones_n = torch.full((num_features,), lam, dtype=torch.float64)
    c_block = torch.cat([-Atb + ones_n, Atb + ones_n])

    # Box bounds: u >= 0, v >= 0
    l_c = torch.zeros(2 * num_features, dtype=torch.float64)
    inf_val = 1e9
    u_c = torch.full((2 * num_features,), inf_val, dtype=torch.float64)

    # Identity linear constraints with unbounded bounds
    A_dummy = torch.eye(2 * num_features, dtype=torch.float64)
    l_k = torch.full((2 * num_features,), -inf_val, dtype=torch.float64)
    u_k = torch.full((2 * num_features,), inf_val, dtype=torch.float64)

    res = solve_dual_hpr_qp(
        Q=Q_block, c=c_block, A=A_dummy,
        l_k=l_k, u_k=u_k, l_c=l_c, u_c=u_c,
        max_iter=600, tol=1e-6, restart_freq=32
    )

    z = res['x']
    u = z[:num_features]
    v = z[num_features:]
    beta_hat = u - v

    identified_nonzeros = (torch.abs(beta_hat) > 1e-2).sum().item()
    est_error = torch.norm(beta_hat - beta_star).item() / torch.norm(beta_star).item()

    print(f"\nSolver Status          : Solved in {res['iterations']} iterations")
    print(f"Final KKT Residual     : {res['kkt_res']:.2e}")
    print(f"Recovered Nonzeros     : {identified_nonzeros} / {true_sparsity} true active features")
    print(f"Relative Recovery Error: {est_error:.4f} ({(1 - est_error) * 100:.2f}% accuracy)")

    assert est_error < 0.25, "LASSO parameter recovery error is unexpectedly high"
    print("\nSparse LASSO feature selection successfully verified!")
    print("=" * 70)


if __name__ == "__main__":
    run_lasso_benchmark()
