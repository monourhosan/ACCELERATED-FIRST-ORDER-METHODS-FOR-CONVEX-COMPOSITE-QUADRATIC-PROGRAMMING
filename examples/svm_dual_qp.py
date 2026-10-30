"""
Support Vector Machine (SVM) Dual Quadratic Programming using HPR-QP.

Dual Formulation of Soft-Margin SVM:
    min_{alpha} 1/2 alpha^T (Y K Y) alpha - sum(alpha_i)
    s.t.        y^T alpha = 0          (Hyperplane bias balance)
                0 <= alpha_i <= C       (Box regularizer)
where K_ij = x_i^T x_j (or RBF kernel) and Y = diag(y).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from hpr_qp import solve_dual_hpr_qp


def run_svm_benchmark():
    print("=" * 70)
    print("  REAL-WORLD APPLICATION: SUPPORT VECTOR MACHINE (SVM) DUAL QP")
    print("=" * 70)

    torch.manual_seed(2024)
    num_samples = 120
    num_features = 10
    C_param = 1.0

    # Generate synthetic binary classification dataset
    X_pos = torch.randn(num_samples // 2, num_features, dtype=torch.float64) + 1.2
    X_neg = torch.randn(num_samples // 2, num_features, dtype=torch.float64) - 1.2
    X = torch.cat([X_pos, X_neg], dim=0)

    y = torch.cat([
        torch.ones(num_samples // 2, dtype=torch.float64),
        -torch.ones(num_samples // 2, dtype=torch.float64)
    ])

    # Compute Gram Matrix: Q_ij = y_i * y_j * (x_i^T x_j)
    Kernel = X @ X.t()
    Q = (y.unsqueeze(1) @ y.unsqueeze(0)) * Kernel + 1e-5 * torch.eye(num_samples, dtype=torch.float64)
    c = -torch.ones(num_samples, dtype=torch.float64)

    # Constraint: y^T alpha = 0
    A = y.unsqueeze(0)  # (1, num_samples)
    l_k = torch.tensor([0.0], dtype=torch.float64)
    u_k = torch.tensor([0.0], dtype=torch.float64)

    # Box bounds: 0 <= alpha_i <= C
    l_c = torch.zeros(num_samples, dtype=torch.float64)
    u_c = torch.full((num_samples,), C_param, dtype=torch.float64)

    print(f"Sample Count (N)       : {num_samples}")
    print(f"Feature Dimension (D)  : {num_features}")
    print(f"Penalty Parameter (C)  : {C_param}\n")

    res = solve_dual_hpr_qp(
        Q=Q, c=c, A=A,
        l_k=l_k, u_k=u_k,
        l_c=l_c, u_c=u_c,
        max_iter=500, tol=1e-6,
        restart_freq=16
    )

    alpha = res['x']
    support_vectors = (alpha > 1e-3).sum().item()
    margin_support_vectors = ((alpha > 1e-3) & (alpha < C_param - 1e-3)).sum().item()

    # Reconstruct optimal weight vector w* = sum(alpha_i * y_i * x_i)
    w_star = torch.sum((alpha * y).unsqueeze(1) * X, dim=0)

    # Calculate training accuracy
    predictions = torch.sign(X @ w_star)
    accuracy = (predictions == y).float().mean().item() * 100.0

    print(f"Solver Status          : Solved in {res['iterations']} iterations")
    print(f"Final KKT Residual     : {res['kkt_res']:.2e}")
    print(f"Support Vectors Found  : {support_vectors} / {num_samples} ({margin_support_vectors} on margin)")
    print(f"Linear SVM Training Acc: {accuracy:.2f}%")

    assert accuracy >= 95.0, f"SVM accuracy unexpectedly low ({accuracy:.2f}%)"
    assert (alpha >= -1e-4).all() and (alpha <= C_param + 1e-4).all()
    print("\nSVM Dual QP solver successfully verified!")
    print("=" * 70)


if __name__ == "__main__":
    run_svm_benchmark()
