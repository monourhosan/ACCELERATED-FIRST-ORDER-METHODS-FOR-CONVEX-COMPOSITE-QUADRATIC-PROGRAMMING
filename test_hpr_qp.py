import torch
from hpr_qp import solve_primal_hpr_cqp, solve_dual_hpr_qp

def test_paper_example_5_1():
    print("=" * 60)
    print("Testing Example 5.1 from Chapter 5.2.3 (Singular Q):")
    # Q = diag([1, 2, 0])
    Q = torch.tensor([[1.0, 0.0, 0.0],
                      [0.0, 2.0, 0.0],
                      [0.0, 0.0, 0.0]], dtype=torch.float64)
    c = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float64)
    # A = identity, unbounded K = [-inf, inf]
    A = torch.eye(3, dtype=torch.float64)
    inf_val = 1e9
    l_k = torch.tensor([-inf_val, -inf_val, -inf_val], dtype=torch.float64)
    u_k = torch.tensor([inf_val, inf_val, inf_val], dtype=torch.float64)
    
    # C = (-inf, 1] x (-inf, 1] x (-inf, 1]
    l_c = torch.tensor([-inf_val, -inf_val, -inf_val], dtype=torch.float64)
    u_c = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)

    res_dual = solve_dual_hpr_qp(Q, c, A, l_k=l_k, u_k=u_k, l_c=l_c, u_c=u_c, 
                                 max_iter=300, tol=1e-8, sigma=1.0, restart_freq=8)
    print(f"Dual HPR-QP solved in {res_dual['iterations']} iters | KKT residual: {res_dual['kkt_res']:.2e}")
    print(f"Optimal x: {res_dual['x'].tolist()}")
    
    # Expected analytical solution:
    # min 1/2 x1^2 - x1 + x2^2 - x2 - x3 subject to x1, x2, x3 <= 1
    # x1* = 1.0, x2* = 0.5, x3* = 1.0 (boundary)
    print(f"Expected x: [1.0, 0.5, 1.0]")
    assert abs(res_dual['x'][0].item() - 1.0) < 1e-4
    assert abs(res_dual['x'][1].item() - 0.5) < 1e-4
    assert abs(res_dual['x'][2].item() - 1.0) < 1e-4
    print("Example 5.1 PASSED!")

def test_paper_example_5_2():
    print("=" * 60)
    print("Testing Example 5.2 from Chapter 5.2.3 (Ill-conditioned Q):")
    # Q = [[1, 0.1], [0.1, 10]]
    Q = torch.tensor([[1.0, 0.1],
                      [0.1, 10.0]], dtype=torch.float64)
    c = torch.tensor([-1.0, -1.0], dtype=torch.float64)
    A = torch.eye(2, dtype=torch.float64)
    inf_val = 1e9
    l_k = torch.tensor([-inf_val, -inf_val], dtype=torch.float64)
    u_k = torch.tensor([inf_val, inf_val], dtype=torch.float64)
    l_c = torch.tensor([-inf_val, -inf_val], dtype=torch.float64)
    u_c = torch.tensor([inf_val, inf_val], dtype=torch.float64)

    res_dual = solve_dual_hpr_qp(Q, c, A, l_k=l_k, u_k=u_k, l_c=l_c, u_c=u_c,
                                 max_iter=500, tol=1e-8, sigma=1.0, restart_freq=8)
    print(f"Dual HPR-QP solved in {res_dual['iterations']} iters | KKT residual: {res_dual['kkt_res']:.2e}")
    print(f"Optimal x: {res_dual['x'].tolist()}")
    
    # Analytical: Q x + c = 0 => x = -Q^-1 c
    expected_x = torch.linalg.solve(Q, -c)
    print(f"Expected x: {expected_x.tolist()}")
    assert torch.norm(res_dual['x'] - expected_x) < 1e-4
    print("Example 5.2 PASSED!")

if __name__ == "__main__":
    test_paper_example_5_1()
    test_paper_example_5_2()
    print("=" * 60)
    print("ALL HPR-QP TESTS COMPLETED SUCCESSFULLY!")
