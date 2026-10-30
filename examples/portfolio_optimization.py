"""
Markowitz Mean-Variance Portfolio Optimization using HPR-QP.

Problem Formulation:
    min_{x} 1/2 x^T Sigma x - gamma * mu^T x
    s.t.    sum(x_i) = 1           (100% budget invested)
            0 <= x_i <= max_weight  (Long-only, diversification bound)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from hpr_qp import solve_dual_hpr_qp


def run_portfolio_optimization():
    print("=" * 70)
    print("  REAL-WORLD APPLICATION: MARKOWITZ MEAN-VARIANCE PORTFOLIO ALLOCATION")
    print("=" * 70)

    torch.manual_seed(42)
    num_assets = 50
    max_weight = 0.10  # Max 10% in any single asset
    risk_aversion = 2.5

    # Generate synthetic asset returns and positive semidefinite covariance matrix
    returns_mean = torch.empty(num_assets, dtype=torch.float64).uniform_(0.04, 0.20)
    factor_loadings = torch.randn(num_assets, 5, dtype=torch.float64)
    Sigma = factor_loadings @ factor_loadings.t() + torch.diag(torch.empty(num_assets, dtype=torch.float64).uniform_(0.01, 0.05))

    # Quadratic objective: Q = Sigma, c = -gamma * mu
    Q = Sigma
    c = -risk_aversion * returns_mean

    # Budget constraint: sum(x_i) = 1  => 1 <= [1 ... 1] x <= 1
    A = torch.ones(1, num_assets, dtype=torch.float64)
    l_k = torch.tensor([1.0], dtype=torch.float64)
    u_k = torch.tensor([1.0], dtype=torch.float64)

    # Box bounds: 0 <= x_i <= max_weight
    l_c = torch.zeros(num_assets, dtype=torch.float64)
    u_c = torch.full((num_assets,), max_weight, dtype=torch.float64)

    print(f"Number of Assets: {num_assets}")
    print(f"Risk Aversion Coefficient (gamma): {risk_aversion}")
    print(f"Max Weight per Asset: {max_weight * 100:.1f}%\n")

    res = solve_dual_hpr_qp(
        Q=Q, c=c, A=A,
        l_k=l_k, u_k=u_k,
        l_c=l_c, u_c=u_c,
        max_iter=500, tol=1e-7,
        sigma=1.0, restart_freq=16
    )

    weights = res['x']
    total_allocated = torch.sum(weights).item()
    exp_return = torch.dot(returns_mean, weights).item()
    portfolio_vol = torch.sqrt(weights @ Sigma @ weights).item()

    print(f"Solver Status       : Solved in {res['iterations']} iterations")
    print(f"Final KKT Residual  : {res['kkt_res']:.2e}")
    print(f"Total Weight Sum    : {total_allocated:.6f} (Constraint: 1.000000)")
    print(f"Expected Return     : {exp_return * 100:.2f}%")
    print(f"Portfolio Volatility: {portfolio_vol * 100:.2f}%")
    print(f"Sharpe Ratio (Rf=2%): {(exp_return - 0.02) / portfolio_vol:.2f}")

    top_assets = torch.topk(weights, 5)
    print("\nTop 5 Holdings:")
    for idx, weight in zip(top_assets.indices.tolist(), top_assets.values.tolist()):
        print(f"  Asset #{idx:02d}: {weight * 100:6.2f}% (Expected Return: {returns_mean[idx] * 100:5.2f}%)")

    assert abs(total_allocated - 1.0) < 1e-4, "Budget constraint violated!"
    assert (weights >= -1e-5).all(), "Short positions detected!"
    assert (weights <= max_weight + 1e-5).all(), "Max weight bound exceeded!"
    print("\nPortfolio allocation successfully verified!")
    print("=" * 70)


if __name__ == "__main__":
    run_portfolio_optimization()
