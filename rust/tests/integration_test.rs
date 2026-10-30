//! Integration tests for the Rust HPR-QP solver.

use hpr_qp::{QPProblem, SolverConfig, solve_hpr_qp};

#[test]
fn test_default_config() {
    let cfg = SolverConfig::default();
    assert!(cfg.max_iter > 0);
    assert!(cfg.tol > 0.0 && cfg.tol < 1.0);
    assert!(cfg.sigma > 0.0);
}

#[test]
fn test_unconstrained_1d() {
    // min 0.5 * 4 * x^2 - 8 * x => x* = 2.0, f(x*) = -8.0
    let prob = QPProblem {
        n: 1,
        m: 0,
        q_mat: Some(vec![4.0]),
        c_vec: Some(vec![-8.0]),
        a_mat: None,
        l_k: None,
        u_k: None,
        l_c: None,
        u_c: None,
    };
    let cfg = SolverConfig {
        max_iter: 500,
        tol: 1e-6,
        sigma: 1.0,
        restart_freq: 16,
    };
    let res = solve_hpr_qp(&prob, &cfg);
    assert!(res.converged, "1D solver must converge");
    assert!((res.x[0] - 2.0).abs() < 1e-3, "Optimal x must be 2.0");
    assert!((res.objective - (-8.0)).abs() < 1e-3, "Optimal objective must be -8.0");
}

#[test]
fn test_box_constrained_2d() {
    // min 0.5 * (x0^2 + x1^2) - 3*x0 - 3*x1
    // s.t. 0 <= x0 <= 1.0, 0 <= x1 <= 1.0
    // Unconstrained min is at (3, 3). Active upper bound => x* = (1, 1).
    // obj* = 0.5 * 2 - 6 = -5.0.
    let prob = QPProblem {
        n: 2,
        m: 0,
        q_mat: Some(vec![
            1.0, 0.0,
            0.0, 1.0,
        ]),
        c_vec: Some(vec![-3.0, -3.0]),
        a_mat: None,
        l_k: None,
        u_k: None,
        l_c: Some(vec![0.0, 0.0]),
        u_c: Some(vec![1.0, 1.0]),
    };
    let cfg = SolverConfig {
        max_iter: 1000,
        tol: 1e-4,
        sigma: 1.0,
        restart_freq: 32,
    };
    let res = solve_hpr_qp(&prob, &cfg);
    assert!(res.converged, "2D box solver must converge");
    assert!((res.x[0] - 1.0).abs() < 1e-2, "x0 must be on boundary 1.0");
    assert!((res.x[1] - 1.0).abs() < 1e-2, "x1 must be on boundary 1.0");
    assert!((res.objective - (-5.0)).abs() < 2e-2, "Objective must be -5.0");
}

#[test]
fn test_singular_hessian() {
    // Example 5.1 from paper: singular Q with zero eigenvalue in 3rd dimension
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
        tol: 1e-6,
        sigma: 1.0,
        restart_freq: 16,
    };
    let res = solve_hpr_qp(&prob, &cfg);
    assert!(res.converged, "Singular Hessian QP must converge in restricted Wolfe dual");
    assert!((res.x[0] - 1.0).abs() < 1e-2);
    assert!((res.x[1] - 0.5).abs() < 1e-2);
    assert!((res.x[2] - 1.0).abs() < 1e-2);
}
