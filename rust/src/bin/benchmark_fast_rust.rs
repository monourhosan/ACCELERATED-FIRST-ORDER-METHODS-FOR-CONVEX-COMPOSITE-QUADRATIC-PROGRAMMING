//! Accelerated HPR-QP solver in Rust with 4-way unrolled CSR SpMV and fused vector pipelines.

use std::fs::File;
use std::io::{Read, BufReader};
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct CSRMatrix {
    pub rows: usize,
    pub cols: usize,
    pub nnz: usize,
    pub row_ptr: Vec<i32>,
    pub col_idx: Vec<i32>,
    pub val: Vec<f64>,
}

impl CSRMatrix {
    // 4-way unrolled SpMV with multiple accumulators
    #[inline(always)]
    pub fn spmv_fast(&self, x: &[f64], y: &mut [f64]) {
        for i in 0..self.rows {
            let start = self.row_ptr[i] as usize;
            let end = self.row_ptr[i + 1] as usize;
            let mut p = start;

            let mut s0 = 0.0;
            let mut s1 = 0.0;
            let mut s2 = 0.0;
            let mut s3 = 0.0;

            while p + 3 < end {
                s0 += self.val[p]     * x[self.col_idx[p] as usize];
                s1 += self.val[p + 1] * x[self.col_idx[p + 1] as usize];
                s2 += self.val[p + 2] * x[self.col_idx[p + 2] as usize];
                s3 += self.val[p + 3] * x[self.col_idx[p + 3] as usize];
                p += 4;
            }

            let mut sum = (s0 + s1) + (s2 + s3);
            while p < end {
                sum += self.val[p] * x[self.col_idx[p] as usize];
                p += 1;
            }
            y[i] = sum;
        }
    }
}

pub struct LargeQPProblem {
    pub n: usize,
    pub m: usize,
    pub p: CSRMatrix,
    pub a: CSRMatrix,
    pub at: CSRMatrix,
    pub c: Vec<f64>,
    pub l_k: Vec<f64>,
    pub u_k: Vec<f64>,
    pub l_c: Vec<f64>,
    pub u_c: Vec<f64>,
}

fn read_i32_vec<R: Read>(reader: &mut R, count: usize) -> Vec<i32> {
    let mut vec = vec![0i32; count];
    let byte_slice = unsafe {
        std::slice::from_raw_parts_mut(vec.as_mut_ptr() as *mut u8, count * 4)
    };
    reader.read_exact(byte_slice).expect("Failed to read i32 array");
    vec
}

fn read_f64_vec<R: Read>(reader: &mut R, count: usize) -> Vec<f64> {
    let mut vec = vec![0.0f64; count];
    let byte_slice = unsafe {
        std::slice::from_raw_parts_mut(vec.as_mut_ptr() as *mut u8, count * 8)
    };
    reader.read_exact(byte_slice).expect("Failed to read f64 array");
    vec
}

pub fn load_binary_dataset(path: &str) -> LargeQPProblem {
    let file = File::open(path).unwrap_or_else(|_| panic!("Could not open {}", path));
    let mut reader = BufReader::new(file);

    let mut header = [0u8; 32];
    reader.read_exact(&mut header).expect("Failed to read header");

    let n = u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
    let m = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
    let nnz_p = u64::from_le_bytes(header[8..16].try_into().unwrap()) as usize;
    let nnz_a = u64::from_le_bytes(header[16..24].try_into().unwrap()) as usize;
    let nnz_at = u64::from_le_bytes(header[24..32].try_into().unwrap()) as usize;

    let p_row_ptr = read_i32_vec(&mut reader, n + 1);
    let p_col_idx = read_i32_vec(&mut reader, nnz_p);
    let p_val = read_f64_vec(&mut reader, nnz_p);

    let a_row_ptr = read_i32_vec(&mut reader, m + 1);
    let a_col_idx = read_i32_vec(&mut reader, nnz_a);
    let a_val = read_f64_vec(&mut reader, nnz_a);

    let at_row_ptr = read_i32_vec(&mut reader, n + 1);
    let at_col_idx = read_i32_vec(&mut reader, nnz_at);
    let at_val = read_f64_vec(&mut reader, nnz_at);

    let c = read_f64_vec(&mut reader, n);
    let l_k = read_f64_vec(&mut reader, m);
    let u_k = read_f64_vec(&mut reader, m);
    let l_c = read_f64_vec(&mut reader, n);
    let u_c = read_f64_vec(&mut reader, n);

    LargeQPProblem {
        n,
        m,
        p: CSRMatrix { rows: n, cols: n, nnz: nnz_p, row_ptr: p_row_ptr, col_idx: p_col_idx, val: p_val },
        a: CSRMatrix { rows: m, cols: n, nnz: nnz_a, row_ptr: a_row_ptr, col_idx: a_col_idx, val: a_val },
        at: CSRMatrix { rows: n, cols: m, nnz: nnz_at, row_ptr: at_row_ptr, col_idx: at_col_idx, val: at_val },
        c,
        l_k,
        u_k,
        l_c,
        u_c,
    }
}

#[inline(always)]
fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    v.max(lo).min(hi)
}

pub struct BenchmarkResult {
    pub elapsed_ms: f64,
    pub kkt_residual: f64,
    pub objective: f64,
}

pub fn run_hpr_qp_fast_rust(prob: &LargeQPProblem, max_iter: usize, restart_freq: usize) -> BenchmarkResult {
    let n = prob.n;
    let m = prob.m;

    let mut x = vec![0.0; n];
    let mut w_q = vec![0.0; n];
    let mut z = vec![0.0; n];
    let mut y = vec![0.0; m];

    let mut x_bar = vec![0.0; n];
    let mut w_bar = vec![0.0; n];
    let mut z_bar = vec![0.0; n];
    let mut y_bar = vec![0.0; m];

    let mut x_hat = vec![0.0; n];
    let mut w_half = vec![0.0; n];
    let mut r_y = vec![0.0; m];
    let mut qw = vec![0.0; n];
    let mut qw_half = vec![0.0; n];
    let mut aty = vec![0.0; n];
    let mut temp_n = vec![0.0; n];
    let mut temp_m = vec![0.0; m];

    let mut x_r0 = vec![0.0; n];
    let mut w_r0 = vec![0.0; n];
    let mut z_r0 = vec![0.0; n];
    let mut y_r0 = vec![0.0; m];

    let inv_sqrt_n = 1.0 / (n as f64).sqrt();
    temp_n.fill(inv_sqrt_n);
    for _ in 0..15 {
        prob.p.spmv_fast(&temp_n, &mut qw);
        let nrm = qw.iter().map(|&v| v * v).sum::<f64>().sqrt();
        if nrm < 1e-12 { break; }
        for i in 0..n { temp_n[i] = qw[i] / nrm; }
    }
    prob.p.spmv_fast(&temp_n, &mut qw);
    let mut lam_q: f64 = temp_n.iter().zip(qw.iter()).map(|(&a, &b)| a * b).sum();
    if lam_q < 1e-6 { lam_q = 1e-6; }

    temp_n.fill(inv_sqrt_n);
    for _ in 0..15 {
        prob.a.spmv_fast(&temp_n, &mut temp_m);
        prob.at.spmv_fast(&temp_m, &mut qw);
        let nrm = qw.iter().map(|&v| v * v).sum::<f64>().sqrt();
        if nrm < 1e-12 { break; }
        for i in 0..n { temp_n[i] = qw[i] / nrm; }
    }
    prob.a.spmv_fast(&temp_n, &mut temp_m);
    let mut lam_a: f64 = temp_m.iter().map(|&v| v * v).sum();
    if lam_a < 1e-6 { lam_a = 1e-6; }

    let sigma = 1.0;
    let denom_w = 1.0 + sigma * lam_q;
    let sig_lam_a = sigma * lam_a;
    let inv_sig_lam_a = 1.0 / sig_lam_a;
    let w_step = sigma / denom_w;

    let t_start = Instant::now();
    let mut inner_t = 0usize;

    for _ in 0..max_iter {
        inner_t += 1;

        // 1) SpMV for Qw and A^T y
        prob.p.spmv_fast(&w_q, &mut qw);
        prob.at.spmv_fast(&y, &mut aty);

        // 2) Fused Step 1 & 2
        for j in 0..n {
            let rz = x[j] + sigma * (-qw[j] + aty[j] - prob.c[j]);
            let xb = clamp(rz, prob.l_c[j], prob.u_c[j]);
            x_bar[j] = xb;
            z_bar[j] = (1.0 / sigma) * (xb - rz);

            let xh = 2.0 * xb - x[j];
            x_hat[j] = xh;
            w_half[j] = (1.0 / denom_w) * (sigma * lam_q * w_q[j] + xh);
        }

        // 3) Step 3: sGS
        prob.p.spmv_fast(&w_half, &mut qw_half);
        for j in 0..n {
            temp_n[j] = x_hat[j] + sigma * (qw[j] - qw_half[j]);
        }
        prob.a.spmv_fast(&temp_n, &mut r_y);

        for i in 0..m {
            r_y[i] -= sig_lam_a * y[i];
            let py = clamp(r_y[i], prob.l_k[i], prob.u_k[i]);
            let yb = inv_sig_lam_a * (py - r_y[i]);
            y_bar[i] = yb;
            temp_m[i] = yb - y[i];
        }

        prob.at.spmv_fast(&temp_m, &mut temp_n);
        for j in 0..n {
            w_bar[j] = w_half[j] + w_step * temp_n[j];
        }

        // 4) Halpern Averaging
        let coef_0 = 1.0 / (inner_t as f64 + 1.0);
        let coef_hat = (inner_t as f64) / (inner_t as f64 + 1.0);

        for j in 0..n {
            x[j] = coef_0 * x_r0[j] + coef_hat * (2.0 * x_bar[j] - x[j]);
            w_q[j] = coef_0 * w_r0[j] + coef_hat * (2.0 * w_bar[j] - w_q[j]);
            z[j] = coef_0 * z_r0[j] + coef_hat * (2.0 * z_bar[j] - z[j]);
        }
        for i in 0..m {
            y[i] = coef_0 * y_r0[i] + coef_hat * (2.0 * y_bar[i] - y[i]);
        }

        // 5) Adaptive restart
        if restart_freq > 0 && inner_t >= restart_freq {
            x_r0.copy_from_slice(&x_bar);
            w_r0.copy_from_slice(&w_bar);
            z_r0.copy_from_slice(&z_bar);
            y_r0.copy_from_slice(&y_bar);

            x.copy_from_slice(&x_r0);
            w_q.copy_from_slice(&w_r0);
            z.copy_from_slice(&z_r0);
            y.copy_from_slice(&y_r0);
            inner_t = 0;
        }
    }

    let elapsed = t_start.elapsed().as_secs_f64() * 1000.0;

    prob.p.spmv_fast(&w_bar, &mut qw);
    prob.at.spmv_fast(&y_bar, &mut aty);

    let mut max_kkt = 0.0f64;
    for j in 0..n {
        let diff = (-qw[j] + aty[j] + z_bar[j] - prob.c[j]).abs();
        if diff > max_kkt { max_kkt = diff; }
    }
    prob.a.spmv_fast(&x_bar, &mut r_y);
    for i in 0..m {
        let diff = (r_y[i] - clamp(r_y[i], prob.l_k[i], prob.u_k[i])).abs();
        if diff > max_kkt { max_kkt = diff; }
    }

    let mut obj = 0.0f64;
    prob.p.spmv_fast(&x_bar, &mut temp_n);
    for j in 0..n {
        obj += 0.5 * x_bar[j] * temp_n[j] + prob.c[j] * x_bar[j];
    }

    BenchmarkResult {
        elapsed_ms: elapsed,
        kkt_residual: max_kkt,
        objective: obj,
    }
}

fn main() {
    println!("=======================================================================");
    println!("  ACCELERATED HPR-QP BENCHMARK IN RUST (4-Way Unrolled & Pipelined)");
    println!("=======================================================================");

    let prob = load_binary_dataset("Data/cvxqp1_l.bin");
    println!("Loaded dataset: n={}, m={}, nnz={}\n",
             prob.n, prob.m, prob.p.nnz + prob.a.nnz);

    for iters in [100, 300] {
        let res = run_hpr_qp_fast_rust(&prob, iters, 32);
        let ms_per_iter = res.elapsed_ms / (iters as f64);
        println!("  [{:>3} iters] Total: {:>6.2} ms | {:.3} ms/iter | KKT: {:.2e} | Obj: {:.2}",
                 iters, res.elapsed_ms, ms_per_iter, res.kkt_residual, res.objective);
    }
    println!("=======================================================================");
}
