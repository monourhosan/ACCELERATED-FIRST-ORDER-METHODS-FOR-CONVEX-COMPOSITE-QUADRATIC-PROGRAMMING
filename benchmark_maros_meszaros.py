"""
Maros-Meszaros QP Benchmark Runner for HPR-QP Solver.
Benchmarks HPR-QP against real-world quadratic programming instances
from the Maros-Meszaros dataset located in D:\\paper project\\T1\\Data.
"""

import os
import sys
import time
import argparse
import scipy.io as spio
import scipy.sparse as spa
import numpy as np
import torch

from hpr_qp import solve_dual_hpr_qp, solve_primal_hpr_cqp


def load_mat_problem(mat_path: str):
    """
    Load a QP problem from a Maros-Meszaros .mat file.
    Format adheres to the standard Maros-Meszaros benchmark:
        min  0.5 * x^T P x + q^T x
        s.t. l_k <= A_lin x <= u_k
             l_c <= x <= u_c
    """
    mat = spio.loadmat(mat_path)
    n = int(mat['n'][0, 0])
    m = int(mat['m'][0, 0])
    m_lin = m - n

    P_mat = mat['P']
    q_vec = mat['q'].T.flatten()
    A_mat = mat['A']
    l_vec = mat['l'].T.flatten()
    u_vec = mat['u'].T.flatten()

    # Convert to float64 numpy
    P_dense = P_mat.toarray().astype(np.float64) if spa.issparse(P_mat) else P_mat.astype(np.float64)
    A_dense = A_mat.toarray().astype(np.float64) if spa.issparse(A_mat) else A_mat.astype(np.float64)
    q = q_vec.astype(np.float64)
    l = l_vec.astype(np.float64)
    u = u_vec.astype(np.float64)

    # Clean infinite values (threshold 9e19)
    inf_bound = 1e20
    l[l < -9e19] = -inf_bound
    l[l > 9e19] = inf_bound
    u[u > 9e19] = inf_bound
    u[u < -9e19] = -inf_bound

    # Split A into general linear constraints and variable bounds
    A_lin = A_dense[:m_lin]
    l_k = l[:m_lin]
    u_k = u[:m_lin]

    l_c = l[m_lin:]
    u_c = u[m_lin:]

    return {
        "n": n,
        "m_lin": m_lin,
        "nnz_P": int(np.count_nonzero(P_dense)),
        "nnz_A": int(np.count_nonzero(A_lin)),
        "P": torch.tensor(P_dense, dtype=torch.float64),
        "q": torch.tensor(q, dtype=torch.float64),
        "A_lin": torch.tensor(A_lin, dtype=torch.float64),
        "l_k": torch.tensor(l_k, dtype=torch.float64),
        "u_k": torch.tensor(u_k, dtype=torch.float64),
        "l_c": torch.tensor(l_c, dtype=torch.float64),
        "u_c": torch.tensor(u_c, dtype=torch.float64),
    }


def run_benchmark(data_dir: str, instances: list, max_iter: int = 3000, tol: float = 1e-4, restart_freq: int = 32):
    """
    Run HPR-QP across a list of test instances and print a structured benchmark table.
    """
    print("=" * 95)
    print("HPR-QP SOLVER: REAL-WORLD MAROS-MESZAROS BENCHMARK EVALUATION")
    print("=" * 95)
    print(f"{'Instance':<12} | {'n':<5} | {'m_lin':<5} | {'Status':<10} | {'Iters':<6} | {'Time (s)':<8} | {'KKT Res':<10} | {'Objective':<14}")
    print("-" * 95)

    results = []

    for name in instances:
        file_path = os.path.join(data_dir, f"{name}.mat")
        if not os.path.exists(file_path):
            print(f"Skipping {name}: file not found at {file_path}")
            continue

        prob = load_mat_problem(file_path)

        res = solve_dual_hpr_qp(
            Q=prob["P"],
            c=prob["q"],
            A=prob["A_lin"],
            l_k=prob["l_k"],
            u_k=prob["u_k"],
            l_c=prob["l_c"],
            u_c=prob["u_c"],
            max_iter=max_iter,
            tol=tol,
            restart_freq=restart_freq
        )

        x = res["x"]
        P = prob["P"]
        q = prob["q"]
        obj = (0.5 * torch.dot(x, torch.matmul(P, x)) + torch.dot(q, x)).item()
        solved = res["kkt_res"] <= tol
        status_str = "OPTIMAL" if solved else "MAX_ITER"

        print(f"{name:<12} | {prob['n']:<5} | {prob['m_lin']:<5} | {status_str:<10} | {res['iterations']:<6} | {res['runtime']:<8.4f} | {res['kkt_res']:<10.2e} | {obj:<14.6f}")

        results.append({
            "name": name,
            "n": prob["n"],
            "m_lin": prob["m_lin"],
            "status": status_str,
            "iters": res["iterations"],
            "runtime": res["runtime"],
            "kkt_res": res["kkt_res"],
            "obj": obj
        })

    print("=" * 95)
    return results


if __name__ == "__main__":
    default_dir = r"D:\paper project\T1\Data\maros_meszaros_qpbenchmark-main\data"
    
    # Representative test suite spanning diverse real-world problem sizes and structures
    selected_instances = [
        "HS21",
        "HS35",
        "HS51",
        "HS52",
        "HS76",
        "GENHS28",
        "ZECEVIC2",
        "TAME",
        "S268",
        "DUAL1",
        "DUAL2",
        "DUAL3",
        "DUAL4",
        "CVXQP1_S",
        "CVXQP2_S",
        "CVXQP3_S",
        "QSCAGR7",
        "QSCSD1"
    ]

    run_benchmark(default_dir, selected_instances, max_iter=3000, tol=1e-4, restart_freq=32)
