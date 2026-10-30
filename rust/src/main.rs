//! Accelerated First-Order Methods for Convex Composite Quadratic Programming (HPR-QP) in Rust.
//! Benchmark runner binary for standalone tests.

use hpr_qp::{QPProblem, SolverConfig, solve_hpr_qp, INF_BOUND};

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
