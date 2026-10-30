#!/usr/bin/env python3
"""
Interactive Command-Line Interface (CLI) for HPR-QP Solver Suite
Unified entry point to solve benchmark problems across CPU, GPU, and C-accelerated backends.
"""

import sys
import time
import argparse
import numpy as np
import torch

# Import core solvers
from hpr_qp import solve_dual_hpr_qp


def generate_benchmark_problem(name: str, device: str = "cpu"):
    """Generates problem data for known benchmark instances as PyTorch tensors."""
    name = name.lower()
    dev = torch.device(device)
    dtype = torch.float64

    if name == "ex51":
        # Example 5.1 from paper: Singular Hessian (3 variables, unconstrained box)
        Q = torch.diag(torch.tensor([1.0, 2.0, 0.0], dtype=dtype, device=dev))
        c = torch.tensor([-1.0, -1.0, -1.0], dtype=dtype, device=dev)
        A = torch.zeros((1, 3), dtype=dtype, device=dev)
        l_k = torch.tensor([-1e20], dtype=dtype, device=dev)
        u_k = torch.tensor([1e20], dtype=dtype, device=dev)
        l_c = torch.full((3,), -1e20, dtype=dtype, device=dev)
        u_c = torch.tensor([1.0, 1.0, 1.0], dtype=dtype, device=dev)
        return {"n": 3, "m": 1, "Q": Q, "c": c, "A": A, "l_k": l_k, "u_k": u_k, "l_c": l_c, "u_c": u_c}

    elif name == "ex52":
        # Example 5.2 from paper: Ill-conditioned Hessian
        Q = torch.tensor([[1.0, 0.1], [0.1, 10.0]], dtype=dtype, device=dev)
        c = torch.tensor([-1.0, -1.0], dtype=dtype, device=dev)
        A = torch.zeros((1, 2), dtype=dtype, device=dev)
        l_k = torch.tensor([-1e20], dtype=dtype, device=dev)
        u_k = torch.tensor([1e20], dtype=dtype, device=dev)
        l_c = torch.full((2,), -1e20, dtype=dtype, device=dev)
        u_c = torch.full((2,), 1e20, dtype=dtype, device=dev)
        return {"n": 2, "m": 1, "Q": Q, "c": c, "A": A, "l_k": l_k, "u_k": u_k, "l_c": l_c, "u_c": u_c}

    elif name == "hs21":
        # Hock-Schittkowski Problem 21
        Q = torch.diag(torch.tensor([0.02, 2.0], dtype=dtype, device=dev))
        c = torch.tensor([0.0, 0.0], dtype=dtype, device=dev)
        A = torch.tensor([[10.0, -1.0]], dtype=dtype, device=dev)
        l_k = torch.tensor([10.0], dtype=dtype, device=dev)
        u_k = torch.tensor([1e20], dtype=dtype, device=dev)
        l_c = torch.tensor([2.0, -1e20], dtype=dtype, device=dev)
        u_c = torch.tensor([50.0, 50.0], dtype=dtype, device=dev)
        return {"n": 2, "m": 1, "Q": Q, "c": c, "A": A, "l_k": l_k, "u_k": u_k, "l_c": l_c, "u_c": u_c}

    elif name == "cvxqp1_l":
        # Synthetic large-scale convex quadratic program
        torch.manual_seed(42)
        n = 1000
        m = 250
        diag = torch.empty(n, dtype=dtype, device=dev).uniform_(1.0, 10.0)
        Q = torch.diag(diag)
        c = torch.randn(n, dtype=dtype, device=dev)
        A = torch.randn(m, n, dtype=dtype, device=dev) * (1.0 / np.sqrt(n))
        l_k = torch.full((m,), -2.0, dtype=dtype, device=dev)
        u_k = torch.full((m,), 2.0, dtype=dtype, device=dev)
        l_c = torch.full((n,), -1.5, dtype=dtype, device=dev)
        u_c = torch.full((n,), 1.5, dtype=dtype, device=dev)
        return {"n": n, "m": m, "Q": Q, "c": c, "A": A, "l_k": l_k, "u_k": u_k, "l_c": l_c, "u_c": u_c}

    else:
        raise ValueError(f"Unknown problem name '{name}'. Choose from: ex51, ex52, hs21, cvxqp1_l")


def main():
    parser = argparse.ArgumentParser(
        description="Unified HPR-QP Solver Command-Line Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="hs21",
        choices=["ex51", "ex52", "hs21", "cvxqp1_l"],
        help="Benchmark instance name to solve",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="cpu",
        choices=["cpu", "gpu", "cuda"],
        help="Execution computational device/backend",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=2000,
        help="Maximum iterations allowed",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="Termination tolerance on maximal KKT residual",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Penalty stepsize scaling factor",
    )

    args = parser.parse_args()

    # Device selection
    if args.backend in ("gpu", "cuda"):
        if torch.cuda.is_available():
            dev_str = "cuda"
        else:
            print("CUDA device not detected; falling back to CPU.")
            dev_str = "cpu"
    else:
        dev_str = "cpu"

    print("======================================================================")
    print("             HPR-QP ACCELERATED SOLVER CLI INTERFACE                 ")
    print("======================================================================")
    print(f"Problem Name   : {args.problem.upper()}")
    print(f"Backend Target : {dev_str.upper()}")
    print(f"Max Iterations : {args.max_iter}")
    print(f"Tolerance      : {args.tol:.1e}")
    print("----------------------------------------------------------------------")

    prob = generate_benchmark_problem(args.problem, device=dev_str)
    n = prob["n"]
    m = prob["m"]
    print(f"Dimensions     : Primal n={n}, Dual m={m}")

    t0 = time.perf_counter()
    res = solve_dual_hpr_qp(
        Q=prob["Q"],
        c=prob["c"],
        A=prob["A"],
        l_k=prob["l_k"],
        u_k=prob["u_k"],
        l_c=prob["l_c"],
        u_c=prob["u_c"],
        max_iter=args.max_iter,
        tol=args.tol,
        sigma=args.sigma,
    )
    t_elapsed = (time.perf_counter() - t0) * 1000.0

    x_sol = res["x"]
    obj_val = (0.5 * (x_sol @ prob["Q"] @ x_sol) + (prob["c"] @ x_sol)).item()

    kkt_residual = res["kkt_res"]
    print("----------------------------------------------------------------------")
    print("                      SOLVER EXECUTION SUMMARY                        ")
    print("----------------------------------------------------------------------")
    status = "CONVERGED" if kkt_residual <= args.tol else "REACHED_MAX_ITER"
    print(f"Status         : {status}")
    print(f"Iterations     : {res['iterations']}")
    print(f"Elapsed Time   : {t_elapsed:.3f} ms")
    print(f"Final Residual : {kkt_residual:.3e}")
    print(f"Objective Val  : {obj_val:.6f}")
    if n <= 5:
        print(f"Primal Sol (x) : {np.round(x_sol.detach().cpu().numpy(), 6)}")
    print("======================================================================")


if __name__ == "__main__":
    main()
