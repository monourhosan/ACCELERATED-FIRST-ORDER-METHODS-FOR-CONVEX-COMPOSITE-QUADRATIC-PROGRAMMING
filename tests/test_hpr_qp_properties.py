"""
Property-based and mathematical invariance tests for HPR-QP solver.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from hpr_qp import solve_dual_hpr_qp


def test_positive_definite_unconstrained():
    """Verify that strongly convex quadratic unconstrained problems converge to -Q^{-1}c."""
    n = 3
    Q = torch.tensor([[4.0, 1.0, 0.0],
                      [1.0, 3.0, 0.5],
                      [0.0, 0.5, 2.0]], dtype=torch.float64)
    c = torch.tensor([-1.0, 2.0, -3.0], dtype=torch.float64)
    A = torch.eye(n, dtype=torch.float64)
    inf_val = 1e9
    l_k = torch.full((n,), -inf_val, dtype=torch.float64)
    u_k = torch.full((n,), inf_val, dtype=torch.float64)
    l_c = torch.full((n,), -inf_val, dtype=torch.float64)
    u_c = torch.full((n,), inf_val, dtype=torch.float64)

    expected_x = torch.linalg.solve(Q, -c)
    res = solve_dual_hpr_qp(Q, c, A, l_k=l_k, u_k=u_k, l_c=l_c, u_c=u_c, max_iter=300, tol=1e-7)

    assert torch.allclose(res['x'], expected_x, rtol=1e-3, atol=1e-3)


def test_box_constraint_clipping():
    """Verify that box constraints are strictly respected at convergence."""
    n = 3
    Q = torch.eye(n, dtype=torch.float64)
    c = torch.tensor([-5.0, 5.0, -10.0], dtype=torch.float64)
    A = torch.eye(n, dtype=torch.float64)
    inf_val = 1e9
    l_k = torch.full((n,), -inf_val, dtype=torch.float64)
    u_k = torch.full((n,), inf_val, dtype=torch.float64)
    l_c = torch.tensor([-1.0, -1.0, -2.0], dtype=torch.float64)
    u_c = torch.tensor([1.0, 1.0, 2.0], dtype=torch.float64)

    res = solve_dual_hpr_qp(Q, c, A, l_k=l_k, u_k=u_k, l_c=l_c, u_c=u_c, max_iter=300, tol=1e-7)

    assert (res['x'] >= l_c - 1e-4).all()
    assert (res['x'] <= u_c + 1e-4).all()
    expected_x = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float64)
    assert torch.allclose(res['x'], expected_x, atol=1e-3)


if __name__ == "__main__":
    test_positive_definite_unconstrained()
    test_box_constraint_clipping()
    print("All HPR-QP property tests passed successfully!")
