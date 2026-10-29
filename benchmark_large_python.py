"""
Benchmark HPR-QP on CVXQP1_L (n=10,000, m=5,000) in Python
"""

import time
import struct
import numpy as np
import scipy.sparse as sp

def load_dataset(bin_path: str):
    with open(bin_path, 'rb') as f:
        header = f.read(4 + 4 + 8 + 8 + 8)
        n, m, nnz_P, nnz_A, nnz_AT = struct.unpack('<IIQQQ', header)
        
        P_row_ptr = np.fromfile(f, dtype=np.int32, count=n + 1)
        P_col_idx = np.fromfile(f, dtype=np.int32, count=nnz_P)
        P_val = np.fromfile(f, dtype=np.float64, count=nnz_P)
        P = sp.csr_matrix((P_val, P_col_idx, P_row_ptr), shape=(n, n))
        
        A_row_ptr = np.fromfile(f, dtype=np.int32, count=m + 1)
        A_col_idx = np.fromfile(f, dtype=np.int32, count=nnz_A)
        A_val = np.fromfile(f, dtype=np.float64, count=nnz_A)
        A = sp.csr_matrix((A_val, A_col_idx, A_row_ptr), shape=(m, n))
        
        AT_row_ptr = np.fromfile(f, dtype=np.int32, count=n + 1)
        AT_col_idx = np.fromfile(f, dtype=np.int32, count=nnz_AT)
        AT_val = np.fromfile(f, dtype=np.float64, count=nnz_AT)
        AT = sp.csr_matrix((AT_val, AT_col_idx, AT_row_ptr), shape=(n, m))
        
        c = np.fromfile(f, dtype=np.float64, count=n)
        l_k = np.fromfile(f, dtype=np.float64, count=m)
        u_k = np.fromfile(f, dtype=np.float64, count=m)
        l_c = np.fromfile(f, dtype=np.float64, count=n)
        u_c = np.fromfile(f, dtype=np.float64, count=n)
        
    return n, m, P, A, AT, c, l_k, u_k, l_c, u_c

def run_hpr_qp_python(n, m, P, A, AT, c, l_k, u_k, l_c, u_c, max_iter=200, restart_freq=32):
    # Estimate spectral norms via power method
    np.random.seed(42)
    v = np.random.randn(n)
    v /= np.linalg.norm(v)
    for _ in range(15):
        w = P.dot(v)
        nw = np.linalg.norm(w)
        if nw < 1e-12: break
        v = w / nw
    lam_q = float(np.dot(v, P.dot(v)))
    lam_q = max(lam_q, 1e-6)
    
    v_a = np.random.randn(n)
    v_a /= np.linalg.norm(v_a)
    for _ in range(15):
        w = AT.dot(A.dot(v_a))
        nw = np.linalg.norm(w)
        if nw < 1e-12: break
        v_a = w / nw
    lam_a = float(np.dot(v_a, AT.dot(A.dot(v_a))))
    lam_a = max(lam_a, 1e-6)
    
    sigma = 1.0
    denom_w = 1.0 + sigma * lam_q
    
    x = np.zeros(n)
    w_q = np.zeros(n)
    z = np.zeros(n)
    y = np.zeros(m)
    
    x_r0 = np.zeros(n)
    w_r0 = np.zeros(n)
    z_r0 = np.zeros(n)
    y_r0 = np.zeros(m)
    
    inner_t = 0
    t0 = time.perf_counter()
    
    for k in range(max_iter):
        inner_t += 1
        
        # 1) Qw and A^T y
        qw = P.dot(w_q)
        aty = AT.dot(y)
        
        # 2) Step 1 & 2
        r_z = x + sigma * (-qw + aty - c)
        x_bar = np.clip(r_z, l_c, u_c)
        z_bar = (1.0 / sigma) * (x_bar - r_z)
        
        x_hat = 2.0 * x_bar - x
        w_half = (1.0 / denom_w) * (sigma * lam_q * w_q + x_hat)
        
        # 3) Step 3: sGS
        qw_half = P.dot(w_half)
        temp_n = x_hat + sigma * (qw - qw_half)
        r_y = A.dot(temp_n) - (sigma * lam_a) * y
        proj_y = np.clip(r_y, l_k, u_k)
        y_bar = (1.0 / (sigma * lam_a)) * (proj_y - r_y)
        
        delta_y = y_bar - y
        w_bar = w_half + (sigma / denom_w) * AT.dot(delta_y)
        
        # 4) Halpern Averaging
        coef_0 = 1.0 / (inner_t + 1.0)
        coef_hat = float(inner_t) / (inner_t + 1.0)
        
        x = coef_0 * x_r0 + coef_hat * (2.0 * x_bar - x)
        w_q = coef_0 * w_r0 + coef_hat * (2.0 * w_bar - w_q)
        z = coef_0 * z_r0 + coef_hat * (2.0 * z_bar - z)
        y = coef_0 * y_r0 + coef_hat * (2.0 * y_bar - y)
        
        # 5) Adaptive restart
        if restart_freq > 0 and inner_t >= restart_freq:
            x_r0 = x_bar.copy()
            w_r0 = w_bar.copy()
            z_r0 = z_bar.copy()
            y_r0 = y_bar.copy()
            
            x = x_r0.copy()
            w_q = w_r0.copy()
            z = z_r0.copy()
            y = y_r0.copy()
            inner_t = 0
            
    t1 = time.perf_counter()
    elapsed = t1 - t0
    
    # KKT residual and objective
    qw = P.dot(w_bar)
    aty = AT.dot(y_bar)
    stat_res = np.max(np.abs(-qw + aty + z_bar - c))
    ax = A.dot(x_bar)
    feas_lin = np.max(np.abs(ax - np.clip(ax, l_k, u_k)))
    feas_box = np.max(np.abs(x_bar - np.clip(x_bar, l_c, u_c)))
    kkt_res = max(stat_res, feas_lin, feas_box)
    
    obj = 0.5 * float(np.dot(x_bar, P.dot(x_bar))) + float(np.dot(c, x_bar))
    
    return elapsed, kkt_res, obj

if __name__ == '__main__':
    print("=" * 65)
    print("  PYTHON BENCHMARK ON CVXQP1_L (n=10,000, m=5,000)")
    print("=" * 65)
    n, m, P, A, AT, c, l_k, u_k, l_c, u_c = load_dataset('Data/cvxqp1_l.bin')
    print(f"Loaded dataset: n={n}, m={m}")
    
    for iters in [100, 300]:
        elapsed, kkt, obj = run_hpr_qp_python(n, m, P, A, AT, c, l_k, u_k, l_c, u_c, max_iter=iters)
        ms_per_iter = (elapsed / iters) * 1000.0
        print(f"  [{iters:3d} iters] Total: {elapsed*1000.0:8.2f} ms | {ms_per_iter:.3f} ms/iter | KKT Res: {kkt:.2e} | Obj: {obj:.4f}")
    print("=" * 65)
