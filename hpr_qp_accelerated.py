"""
Accelerated Unified HPR-QP Solver in Python.
Seamlessly routes execution to:
  1. 'gpu'      : GPU-Accelerated PyTorch Solver (NVIDIA CUDA / Apple MPS via cuSPARSE & VRAM residency)
  2. 'native_c' : Ultra-High-Throughput Native C99 Engine (4-way SIMD FMA unrolling + OpenMP 8-threads)
  3. 'scipy'    : Baseline CPU Python Engine (NumPy + SciPy CSR)
"""

import os
import time
import ctypes
import numpy as np
import scipy.sparse as sp
import torch
from typing import Dict, Any, Optional, Union
from hpr_qp_gpu import GPUHPRQPSolver

# Load native compiled DLL if available (or compile on the fly)
_C_DIR = os.path.join(os.path.dirname(__file__), 'c')
_DLL_PATH = os.path.join(_C_DIR, 'hpr_qp_fast.dll')

def _try_compile_native():
    if os.path.exists(_DLL_PATH):
        return True
    c_src = os.path.join(_C_DIR, 'benchmark_fast_c.c')
    if not os.path.exists(c_src):
        return False
    # Attempt compilation with gcc or clang or cl
    import subprocess
    # Try gcc (Linux/MinGW)
    try:
        cmd = f'gcc -O3 -shared -fopenmp -fPIC -DHPR_QP_DLL_ONLY "{c_src}" -o "{_DLL_PATH}"'
        res = subprocess.run(cmd, shell=True, capture_output=True)
        if res.returncode == 0 and os.path.exists(_DLL_PATH):
            return True
    except Exception:
        pass
    return os.path.exists(_DLL_PATH)

_HAS_NATIVE_C = os.path.exists(_DLL_PATH) or _try_compile_native()
_c_lib = None

if _HAS_NATIVE_C and os.path.exists(_DLL_PATH):
    try:
        _c_lib = ctypes.CDLL(_DLL_PATH)
        _c_lib.solve_hpr_qp_fast_c_api.restype = ctypes.c_int
        _c_lib.solve_hpr_qp_fast_c_api.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_double), ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_double), ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_double), ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.c_int, ctypes.c_int, ctypes.c_double,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)
        ]
    except Exception as e:
        _c_lib = None
        _HAS_NATIVE_C = False

def solve_hpr_qp_accelerated(
    P_csr: sp.csr_matrix,
    c_vec: np.ndarray,
    A_csr: Optional[sp.csr_matrix] = None,
    AT_csr: Optional[sp.csr_matrix] = None,
    l_k: Optional[np.ndarray] = None,
    u_k: Optional[np.ndarray] = None,
    l_c: Optional[np.ndarray] = None,
    u_c: Optional[np.ndarray] = None,
    engine: str = 'auto',  # 'gpu', 'native_c', 'scipy', 'auto'
    max_iter: int = 300,
    tol: float = 1e-4,
    restart_freq: int = 32,
) -> Dict[str, Any]:
    n = P_csr.shape[0]
    m = A_csr.shape[0] if A_csr is not None else 0

    if AT_csr is None and A_csr is not None:
        AT_csr = A_csr.T.tocsr()

    inf_bound = 1e20
    l_k = np.clip(l_k, -inf_bound, inf_bound) if l_k is not None else np.full(m, -inf_bound)
    u_k = np.clip(u_k, -inf_bound, inf_bound) if u_k is not None else np.full(m, inf_bound)
    l_c = np.clip(l_c, -inf_bound, inf_bound) if l_c is not None else np.full(n, -inf_bound)
    u_c = np.clip(u_c, -inf_bound, inf_bound) if u_c is not None else np.full(n, inf_bound)

    # Resolve automatic engine selection
    if engine == 'auto':
        if torch.cuda.is_available():
            engine = 'gpu'
        elif _HAS_NATIVE_C:
            engine = 'native_c'
        else:
            engine = 'scipy'

    if engine == 'native_c' and _HAS_NATIVE_C:
        # High-Speed Native C Engine via ctypes
        stats = np.zeros(3, dtype=np.float64)
        x_out = np.zeros(n, dtype=np.float64)

        P_row_ptr = P_csr.indptr.astype(np.int32)
        P_col_idx = P_csr.indices.astype(np.int32)
        P_val = P_csr.data.astype(np.float64)

        if A_csr is not None:
            A_row_ptr = A_csr.indptr.astype(np.int32)
            A_col_idx = A_csr.indices.astype(np.int32)
            A_val = A_csr.data.astype(np.float64)

            AT_row_ptr = AT_csr.indptr.astype(np.int32)
            AT_col_idx = AT_csr.indices.astype(np.int32)
            AT_val = AT_csr.data.astype(np.float64)
            nnz_A = int(A_csr.nnz)
            nnz_AT = int(AT_csr.nnz)
        else:
            dummy_i = np.zeros(1, dtype=np.int32)
            dummy_f = np.zeros(1, dtype=np.float64)
            A_row_ptr, A_col_idx, A_val = dummy_i, dummy_i, dummy_f
            AT_row_ptr, AT_col_idx, AT_val = dummy_i, dummy_i, dummy_f
            nnz_A, nnz_AT = 0, 0

        c_arr = np.ascontiguousarray(c_vec, dtype=np.float64)
        lk_arr = np.ascontiguousarray(l_k, dtype=np.float64)
        uk_arr = np.ascontiguousarray(u_k, dtype=np.float64)
        lc_arr = np.ascontiguousarray(l_c, dtype=np.float64)
        uc_arr = np.ascontiguousarray(u_c, dtype=np.float64)

        _c_lib.solve_hpr_qp_fast_c_api(
            ctypes.c_uint32(n), ctypes.c_uint32(m),
            P_row_ptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            P_col_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            P_val.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_uint64(int(P_csr.nnz)),
            A_row_ptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            A_col_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            A_val.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_uint64(nnz_A),
            AT_row_ptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            AT_col_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            AT_val.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_uint64(nnz_AT),
            c_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            lk_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            uk_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            lc_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            uc_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(max_iter), ctypes.c_int(restart_freq), ctypes.c_double(tol),
            x_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            stats.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )

        return {
            'engine': 'Native C99 (4-way SIMD + OpenMP 8-Threads)',
            'elapsed_ms': stats[0],
            'kkt_residual': stats[1],
            'objective': stats[2],
            'iterations': max_iter,
        }

    elif engine == 'gpu':
        # GPU PyTorch CSR Solver
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        P_t = torch.sparse_csr_tensor(
            torch.from_numpy(P_csr.indptr.astype(np.int64)),
            torch.from_numpy(P_csr.indices.astype(np.int64)),
            torch.from_numpy(P_csr.data.astype(np.float64)),
            size=(n, n)
        )
        A_t = torch.sparse_csr_tensor(
            torch.from_numpy(A_csr.indptr.astype(np.int64)),
            torch.from_numpy(A_csr.indices.astype(np.int64)),
            torch.from_numpy(A_csr.data.astype(np.float64)),
            size=(m, n)
        ) if A_csr is not None else None

        AT_t = torch.sparse_csr_tensor(
            torch.from_numpy(AT_csr.indptr.astype(np.int64)),
            torch.from_numpy(AT_csr.indices.astype(np.int64)),
            torch.from_numpy(AT_csr.data.astype(np.float64)),
            size=(n, m)
        ) if AT_csr is not None else None

        solver = GPUHPRQPSolver(
            P=P_t, c=torch.from_numpy(c_vec),
            A=A_t, AT=AT_t,
            l_k=torch.from_numpy(l_k), u_k=torch.from_numpy(u_k),
            l_c=torch.from_numpy(l_c), u_c=torch.from_numpy(u_c),
            device=device, dtype=torch.float64, max_iter=max_iter, restart_freq=restart_freq, tol=tol
        )
        res = solver.solve()
        res['engine'] = f'PyTorch GPU ({device})'
        return res

    else:
        # Baseline SciPy Engine
        from benchmark_large_python import run_hpr_qp_python
        t0 = time.perf_counter()
        elapsed, kkt, obj = run_hpr_qp_python(
            n, m, P_csr, A_csr, AT_csr, c_vec, l_k, u_k, l_c, u_c, max_iter=max_iter, restart_freq=restart_freq
        )
        return {
            'engine': 'Baseline CPU Python (NumPy + SciPy CSR)',
            'elapsed_ms': elapsed * 1000.0,
            'kkt_residual': kkt,
            'objective': obj,
            'iterations': max_iter,
        }
