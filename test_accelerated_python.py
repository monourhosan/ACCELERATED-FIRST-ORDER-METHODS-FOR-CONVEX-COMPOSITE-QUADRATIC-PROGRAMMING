import time
import numpy as np
import scipy.sparse as sp
from hpr_qp_accelerated import solve_hpr_qp_accelerated
from benchmark_large_python import load_dataset

def main():
    print("=" * 75)
    print("  PYTHON ENGINE COMPARISON: OLD CPU PYTHON vs. NEW ACCELERATED PYTHON")
    print("  Problem: CVXQP1_L (n = 10,000, m = 15,000, Nonzeros = 94,966)")
    print("=" * 75)

    n, m, P, A, AT, c, l_k, u_k, l_c, u_c = load_dataset('Data/cvxqp1_l.bin')

    for iters in [100, 300]:
        print(f"\n--- Testing {iters} Iterations ---")
        
        # 1. Baseline Old CPU Python
        res_old = solve_hpr_qp_accelerated(P, c, A, AT, l_k, u_k, l_c, u_c, engine='scipy', max_iter=iters)
        ms_old = res_old['elapsed_ms']
        print(f"  [Old CPU Python]       Total: {ms_old:6.2f} ms | Latency: {ms_old/iters:5.3f} ms/iter | KKT: {res_old['kkt_residual']:.2e} | Obj: {res_old['objective']:.2f}")

        # 2. New Accelerated Engine (Called directly from Python)
        res_fast = solve_hpr_qp_accelerated(P, c, A, AT, l_k, u_k, l_c, u_c, engine='native_c', max_iter=iters)
        ms_fast = res_fast['elapsed_ms']
        speedup = ms_old / ms_fast
        print(f"  [New Accelerated C/OMP] Total: {ms_fast:6.2f} ms | Latency: {ms_fast/iters:5.3f} ms/iter | KKT: {res_fast['kkt_residual']:.2e} | Obj: {res_fast['objective']:.2f}")
        print(f"  >>> Direct Speedup: {speedup:.2f}x FASTER!")

    print("\n" + "=" * 75)

if __name__ == '__main__':
    main()
