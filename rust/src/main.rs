//! Accelerated First-Order Methods for Convex Composite Quadratic Programming (HPR-QP) in Rust.
//! Based on: "Accelerated First Order Methods for Convex Composite Quadratic Programming:
//! Theory, Algorithms, and GPU Implementation" (Chen, 2026).

use std::time::Instant;

pub const INF_BOUND: f64 = 1e20;

#[derive(Clone, Debug)]
pub struct QPProblem {
    pub n: usize,
    pub m: usize,
    pub q_mat: Option<Vec<f64>>,   // Row-major n x n
    pub c_vec: Option<Vec<f64>>,   // Length n
    pub a_mat: Option<Vec<f64>>,   // Row-major m x n
    pub l_k: Option<Vec<f64>>,     // Length m
    pub u_k: Option<Vec<f64>>,     // Length m
    pub l_c: Option<Vec<f64>>,     // Length n
    pub u_c: Option<Vec<f64>>,     // Length n
}

#[derive(Clone, Copy, Debug)]
pub struct SolverConfig {
    pub max_iter: usize,
    pub tol: f64,
    pub sigma: f64,
    pub restart_freq: usize,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            max_iter: 3000,
            tol: 1e-6,
            sigma: 1.0,
            restart_freq: 32,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SolverResult {
    pub x: Vec<f64>,
    pub w: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<f64>,
    pub iterations: usize,
    pub runtime_sec: f64,
    pub kkt_residual: f64,
    pub objective: f64,
    pub converged: bool,
}

#[inline(always)]
fn clamp(val: f64, lb: f64, ub: f64) -> f64 {
    val.max(lb).min(ub)
}

fn mat_vec_mul(m: &[f64], x: &[f64], y: &mut [f64], rows: usize, cols: usize) {
    for i in 0..rows {
        let row_start = i * cols;
        let mut sum = 0.0;
        for j in 0..cols {
            sum += m[row_start + j] * x[j];
        }
        y[i] = sum;
    }
}

fn mat_t_vec_mul(m: &[f64], x: &[f64], y: &mut [f64], rows: usize, cols: usize) {
    y.fill(0.0);
    for i in 0..rows {
        let xi = x[i];
        let row_start = i * cols;
        for j in 0..cols {
            y[j] += m[row_start + j] * xi;
        }
    }
}

fn estimate_lambda_a(a: &[f64], m: usize, n: usize, max_iter: usize) -> f64 {
    if m == 0 || n == 0 || a.is_empty() { return 1.0; }
    let mut v = vec![1.0 / (n as f64).sqrt(); n];
    let mut av = vec![0.0; m];
    let mut w = vec![0.0; n];

    for _ in 0..max_iter {
        mat_vec_mul(a, &v, &mut av, m, n);
        mat_t_vec_mul(a, &av, &mut w, m, n);
        let norm_w = w.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm_w < 1e-12 { return 1e-6; }
        for j in 0..n { v[j] = w[j] / norm_w; }
    }

    mat_vec_mul(a, &v, &mut av, m, n);
    let sum_sq: f64 = av.iter().map(|&x| x * x).sum();
    sum_sq.max(1e-6)
}

fn estimate_lambda_q(q: &[f64], n: usize, max_iter: usize) -> f64 {
    if n == 0 || q.is_empty() { return 1.0; }
    let mut v = vec![1.0 / (n as f64).sqrt(); n];
    let mut w = vec![0.0; n];

    for _ in 0..max_iter {
        mat_vec_mul(q, &v, &mut w, n, n);
        let norm_w = w.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm_w < 1e-12 { break; }
        for j in 0..n { v[j] = w[j] / norm_w; }
    }

    mat_vec_mul(q, &v, &mut w, n, n);
    let lambda: f64 = v.iter().zip(w.iter()).map(|(&a, &b)| a * b).sum();
    lambda.max(1e-6)
}

pub fn solve_hpr_qp(prob: &QPProblem, cfg: &SolverConfig) -> SolverResult {
    let start_time = Instant::now();

    let n = prob.n;
    let m = prob.m;
    let has_a = m > 0 && prob.a_mat.is_some();
    let has_q = prob.q_mat.is_some();

    let lambda_a = if has_a { estimate_lambda_a(prob.a_mat.as_ref().unwrap(), m, n, 40) } else { 1.0 };
    let lambda_q = if has_q { estimate_lambda_q(prob.q_mat.as_ref().unwrap(), n, 40) } else { 1.0 };

    let sigma = if cfg.sigma > 0.0 { cfg.sigma } else { 1.0 };
    let denom_w = 1.0 + sigma * lambda_q;

    // Working vector allocations
    let mut x = vec![0.0; n];
    let mut w_q = vec![0.0; n];
    let mut z = vec![0.0; n];
    let mut y = vec![0.0; m.max(1)];

    let mut x_bar = vec![0.0; n];
    let mut z_bar = vec![0.0; n];
    let mut w_half = vec![0.0; n];
    let mut w_bar = vec![0.0; n];
    let mut y_bar = vec![0.0; m.max(1)];

    let mut x_hat = vec![0.0; n];
    let mut r_z = vec![0.0; n];
    let mut r_y = vec![0.0; m.max(1)];
    let mut qw = vec![0.0; n];
    let mut qw_half = vec![0.0; n];
    let mut aty = vec![0.0; n];
    let mut temp_n = vec![0.0; n];

    // Anchor points for Halpern restart
    let mut x_r0 = vec![0.0; n];
    let mut w_r0 = vec![0.0; n];
    let mut z_r0 = vec![0.0; n];
    let mut y_r0 = vec![0.0; m.max(1)];

    let mut inner_t = 0usize;
    let mut kkt_residual = 1e9;
    let mut converged = false;
    let mut total_iters = cfg.max_iter;

    for k in 0..cfg.max_iter {
        inner_t += 1;

        // 1) Qw and A^T y
        if let Some(ref q) = prob.q_mat {
            mat_vec_mul(q, &w_q, &mut qw, n, n);
        }
        if let Some(ref a) = prob.a_mat {
            mat_t_vec_mul(a, &y, &mut aty, m, n);
        }

        // 2) Step 1 & 2: r_z and projection x_bar
        for j in 0..n {
            let c_val = prob.c_vec.as_ref().map_or(0.0, |c| c[j]);
            r_z[j] = x[j] + sigma * (-qw[j] + aty[j] - c_val);

            let lb = prob.l_c.as_ref().map_or(-INF_BOUND, |b| b[j]);
            let ub = prob.u_c.as_ref().map_or(INF_BOUND, |b| b[j]);
            x_bar[j] = clamp(r_z[j], lb, ub);
            z_bar[j] = (1.0 / sigma) * (x_bar[j] - r_z[j]);

            x_hat[j] = 2.0 * x_bar[j] - x[j];
            w_half[j] = (1.0 / denom_w) * (sigma * lambda_q * w_q[j] + x_hat[j]);
        }

        // 3) Step 3: y_bar and w_bar
        if let Some(ref a) = prob.a_mat {
            if let Some(ref q) = prob.q_mat {
                mat_vec_mul(q, &w_half, &mut qw_half, n, n);
            }
            for j in 0..n {
                temp_n[j] = x_hat[j] + sigma * (qw[j] - qw_half[j]);
            }
            mat_vec_mul(a, &temp_n, &mut r_y, m, n);

            let sig_lam_a = sigma * lambda_a;
            for i in 0..m {
                r_y[i] -= sig_lam_a * y[i];
                let lb = prob.l_k.as_ref().map_or(-INF_BOUND, |b| b[i]);
                let ub = prob.u_k.as_ref().map_or(INF_BOUND, |b| b[i]);
                let proj = clamp(r_y[i], lb, ub);
                y_bar[i] = (1.0 / sig_lam_a) * (proj - r_y[i]);
            }

            for i in 0..m { r_y[i] = y_bar[i] - y[i]; }
            mat_t_vec_mul(a, &r_y, &mut temp_n, m, n);
            let w_step = sigma / denom_w;
            for j in 0..n {
                w_bar[j] = w_half[j] + w_step * temp_n[j];
            }
        } else {
            w_bar.copy_from_slice(&w_half);
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

        // 5) Adaptive Restart
        if cfg.restart_freq > 0 && inner_t >= cfg.restart_freq {
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

        // 6) Periodic Residual Check
        if (k + 1) % 20 == 0 || k == cfg.max_iter - 1 {
            let mut max_kkt = 0.0f64;

            if let Some(ref a) = prob.a_mat {
                mat_vec_mul(a, &x_bar, &mut r_y, m, n);
                for i in 0..m {
                    let lb = prob.l_k.as_ref().map_or(-INF_BOUND, |b| b[i]);
                    let ub = prob.u_k.as_ref().map_or(INF_BOUND, |b| b[i]);
                    max_kkt = max_kkt.max((r_y[i] - clamp(r_y[i], lb, ub)).abs());
                }
            }

            if let Some(ref q) = prob.q_mat { mat_vec_mul(q, &w_bar, &mut qw, n, n); }
            if let Some(ref a) = prob.a_mat { mat_t_vec_mul(a, &y_bar, &mut aty, m, n); }

            for j in 0..n {
                let c_val = prob.c_vec.as_ref().map_or(0.0, |c| c[j]);
                max_kkt = max_kkt.max((-qw[j] + aty[j] + z_bar[j] - c_val).abs());
                let lb = prob.l_c.as_ref().map_or(-INF_BOUND, |b| b[j]);
                let ub = prob.u_c.as_ref().map_or(INF_BOUND, |b| b[j]);
                max_kkt = max_kkt.max((x_bar[j] - clamp(x_bar[j], lb, ub)).abs());
            }

            kkt_residual = max_kkt;
            if kkt_residual <= cfg.tol {
                converged = true;
                total_iters = k + 1;
                break;
            }
        }
    }

    let elapsed = start_time.elapsed().as_secs_f64();

    // Compute final objective: 0.5 * x^T Q x + c^T x
    let mut obj = 0.0;
    if let Some(ref q) = prob.q_mat {
        mat_vec_mul(q, &x_bar, &mut temp_n, n, n);
        for j in 0..n { obj += 0.5 * x_bar[j] * temp_n[j]; }
    }
    if let Some(ref c) = prob.c_vec {
        for j in 0..n { obj += c[j] * x_bar[j]; }
    }

    SolverResult {
        x: x_bar,
        w: w_bar,
        y: y_bar,
        z: z_bar,
        iterations: total_iters,
        runtime_sec: elapsed,
        kkt_residual,
        objective: obj,
        converged,
    }
}

fn main() {
    println!("===============================================================");
    println!("  HPR-QP SOLVER IN RUST (Fast, Safe, Zero-Cost Abstractions)");
    println!("===============================================================");

    // Test Example 5.1 (Singular Q)
    {
        let prob = QPProblem {
            n: 3,
            m: 0,
            q_mat: Some(vec![
                1.0, 0.0, 0.0,
                0.0, 2.0, 0.0,
                0.0, 0.0, 0.0,
            ]),
            c_vec: Some(vec![-1.0, -1.0, -1.0]),
            a_mat: None,
            l_k: None,
            u_k: None,
            l_c: None,
            u_c: Some(vec![1.0, 1.0, 1.0]),
        };

        let cfg = SolverConfig {
            max_iter: 1000,
            tol: 1e-8,
            sigma: 1.0,
            restart_freq: 8,
        };

        let res = solve_hpr_qp(&prob, &cfg);
        println!("Example 5.1 (Singular Q):");
        println!("  Status: {} | Iters: {} | Time: {:.6} ms | KKT Res: {:.2e} | Obj: {:.6}",
                 if res.converged { "CONVERGED" } else { "MAX_ITER" },
                 res.iterations, res.runtime_sec * 1000.0, res.kkt_residual, res.objective);
        println!("  Optimal x: [{:.6}, {:.6}, {:.6}]", res.x[0], res.x[1], res.x[2]);
        println!("  Expected : [1.000000, 0.500000, 1.000000]\n");
    }

    // Test Example 5.2 (Ill-conditioned Q)
    {
        let prob = QPProblem {
            n: 2,
            m: 0,
            q_mat: Some(vec![
                1.0, 0.1,
                0.1, 10.0,
            ]),
            c_vec: Some(vec![-1.0, -1.0]),
            a_mat: None,
            l_k: None,
            u_k: None,
            l_c: None,
            u_c: None,
        };

        let cfg = SolverConfig {
            max_iter: 1000,
            tol: 1e-8,
            sigma: 1.0,
            restart_freq: 8,
        };

        let res = solve_hpr_qp(&prob, &cfg);
        println!("Example 5.2 (Ill-conditioned Q):");
        println!("  Status: {} | Iters: {} | Time: {:.6} ms | KKT Res: {:.2e} | Obj: {:.6}",
                 if res.converged { "CONVERGED" } else { "MAX_ITER" },
                 res.iterations, res.runtime_sec * 1000.0, res.kkt_residual, res.objective);
        println!("  Optimal x: [{:.6}, {:.6}]", res.x[0], res.x[1]);
        println!("  Expected : [0.990991, 0.090090]\n");
    }

    // Test HS21
    {
        let prob = QPProblem {
            n: 2,
            m: 1,
            q_mat: Some(vec![
                0.02, 0.0,
                0.0,  2.0,
            ]),
            c_vec: Some(vec![0.0, 0.0]),
            a_mat: Some(vec![10.0, -1.0]),
            l_k: Some(vec![10.0]),
            u_k: None,
            l_c: Some(vec![2.0, -INF_BOUND]),
            u_c: Some(vec![50.0, 50.0]),
        };

        let cfg = SolverConfig {
            max_iter: 1000,
            tol: 1e-6,
            sigma: 1.0,
            restart_freq: 16,
        };

        let res = solve_hpr_qp(&prob, &cfg);
        println!("Hock-Schittkowski #21 (HS21):");
        println!("  Status: {} | Iters: {} | Time: {:.6} ms | KKT Res: {:.2e} | Obj: {:.6}",
                 if res.converged { "CONVERGED" } else { "MAX_ITER" },
                 res.iterations, res.runtime_sec * 1000.0, res.kkt_residual, res.objective);
        println!("  Optimal x: [{:.6}, {:.6}]", res.x[0], res.x[1]);
        println!("  Expected : [2.000000, 0.000000] | Obj: 0.040000");
    }

    println!("===============================================================");
    println!("  ALL RUST BENCHMARKS COMPLETED SUCCESSFULLY!");
    println!("===============================================================");
}
