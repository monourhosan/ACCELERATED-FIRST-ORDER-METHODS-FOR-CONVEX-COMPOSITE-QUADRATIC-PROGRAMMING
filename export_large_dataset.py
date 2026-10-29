"""
Export CVXQP1_L from Maros-Meszaros dataset into a high-performance binary format
for cross-language benchmarking across Python, C, C++, and Rust.
"""

import struct
import numpy as np
import scipy.io as sio
import scipy.sparse as sp

def export_binary(mat_path: str, bin_path: str):
    print(f"Loading {mat_path}...")
    mat = sio.loadmat(mat_path)
    
    n = int(mat['P'].shape[0])
    m_total = int(mat['A'].shape[0])
    m = m_total - n  # general linear constraints (excluding box bounds)
    
    P_csr = mat['P'].tocsr().astype(np.float64)
    A_csr = mat['A'][:m].tocsr().astype(np.float64)
    AT_csr = A_csr.T.tocsr().astype(np.float64)
    
    c = mat['q'].flatten().astype(np.float64)
    l_raw = mat['l'].flatten().astype(np.float64)
    u_raw = mat['u'].flatten().astype(np.float64)
    
    inf_bound = 1e20
    l_raw = np.clip(l_raw, -inf_bound, inf_bound)
    u_raw = np.clip(u_raw, -inf_bound, inf_bound)
    
    l_k = l_raw[:m]
    u_k = u_raw[:m]
    l_c = l_raw[m:]
    u_c = u_raw[m:]
    
    nnz_P = int(P_csr.nnz)
    nnz_A = int(A_csr.nnz)
    nnz_AT = int(AT_csr.nnz)
    
    print(f"Dataset stats:")
    print(f"  Variables (n): {n}")
    print(f"  Linear constraints (m): {m}")
    print(f"  nnz(P): {nnz_P}")
    print(f"  nnz(A): {nnz_A}")
    print(f"  nnz(A^T): {nnz_AT}")
    
    with open(bin_path, 'wb') as f:
        # Header: n (u32), m (u32), nnz_P (u64), nnz_A (u64), nnz_AT (u64)
        f.write(struct.pack('<IIQQQ', n, m, nnz_P, nnz_A, nnz_AT))
        
        # P matrix (CSR)
        P_csr.indptr.astype(np.int32).tofile(f)
        P_csr.indices.astype(np.int32).tofile(f)
        P_csr.data.astype(np.float64).tofile(f)
        
        # A matrix (CSR)
        A_csr.indptr.astype(np.int32).tofile(f)
        A_csr.indices.astype(np.int32).tofile(f)
        A_csr.data.astype(np.float64).tofile(f)
        
        # A^T matrix (CSR)
        AT_csr.indptr.astype(np.int32).tofile(f)
        AT_csr.indices.astype(np.int32).tofile(f)
        AT_csr.data.astype(np.float64).tofile(f)
        
        # Vectors
        c.astype(np.float64).tofile(f)
        l_k.astype(np.float64).tofile(f)
        u_k.astype(np.float64).tofile(f)
        l_c.astype(np.float64).tofile(f)
        u_c.astype(np.float64).tofile(f)
        
    print(f"Exported successfully to {bin_path}!")

if __name__ == '__main__':
    export_binary(
        'Data/maros_meszaros_qpbenchmark-main/data/CVXQP1_L.mat',
        'Data/cvxqp1_l.bin'
    )
