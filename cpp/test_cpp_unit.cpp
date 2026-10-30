/**
 * Unit Test Harness for Modern C++20 HPR-QP Solver Implementation
 */
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include "hpr_qp.hpp"

#define ASSERT_TRUE(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "  [FAIL] Line " << __LINE__ << ": " << msg << std::endl; \
        return 1; \
    } \
} while(0)

#define ASSERT_NEAR(a, b, tol, msg) do { \
    if (std::abs((a) - (b)) > (tol)) { \
        std::cerr << "  [FAIL] Line " << __LINE__ << ": " << msg \
                  << " (expected " << (b) << ", got " << (a) \
                  << ", diff " << std::abs((a) - (b)) << ")" << std::endl; \
        return 1; \
    } \
} while(0)

int test_solver_config() {
    std::cout << "[TEST 1] SolverConfig default initialization..." << std::endl;
    hpr::SolverConfig cfg;
    ASSERT_TRUE(cfg.max_iter > 0, "max_iter must be positive");
    ASSERT_TRUE(cfg.tol > 0.0 && cfg.tol < 1.0, "tol must be in (0, 1)");
    ASSERT_TRUE(cfg.sigma > 0.0, "sigma must be positive");
    ASSERT_TRUE(cfg.restart_freq >= 0, "restart_freq must be non-negative");
    return 0;
}

int test_1d_unconstrained() {
    std::cout << "[TEST 2] 1D Unconstrained Strictly Convex QP..." << std::endl;
    // min 0.5 * 2 * x^2 - 4 * x
    // x* = 2.0, obj* = -4.0
    hpr::QPProblem prob;
    prob.n = 1;
    prob.m = 0;
    prob.Q = {2.0};
    prob.c = {-4.0};

    hpr::SolverConfig cfg;
    cfg.max_iter = 500;
    cfg.tol = 1e-4;

    hpr::HPRQPSolver solver(cfg);
    hpr::SolverResult res = solver.solve(prob);

    ASSERT_TRUE(res.converged, "1D QP must converge");
    ASSERT_NEAR(res.x[0], 2.0, 1e-2, "x[0] must be near 2.0");
    ASSERT_NEAR(res.objective, -4.0, 1e-2, "objective must be near -4.0");
    return 0;
}

int test_2d_equality_constrained() {
    std::cout << "[TEST 3] 2D Simplex Projection QP (Equality Constraint)..." << std::endl;
    // min 0.5 * (x0^2 + x1^2)
    // s.t. x0 + x1 = 1, x >= 0
    // x0* = 0.5, x1* = 0.5, obj* = 0.25
    hpr::QPProblem prob;
    prob.n = 2;
    prob.m = 1;
    prob.Q = {1.0, 0.0, 0.0, 1.0};
    prob.c = {0.0, 0.0};
    prob.A = {1.0, 1.0};
    prob.l_k = {1.0};
    prob.u_k = {1.0};
    prob.l_c = {0.0, 0.0};
    prob.u_c = {10.0, 10.0};

    hpr::SolverConfig cfg;
    cfg.max_iter = 1000;
    cfg.tol = 1e-4;

    hpr::HPRQPSolver solver(cfg);
    hpr::SolverResult res = solver.solve(prob);

    ASSERT_TRUE(res.converged, "2D Simplex QP must converge");
    ASSERT_NEAR(res.x[0], 0.5, 2e-2, "x[0] must be near 0.5");
    ASSERT_NEAR(res.x[1], 0.5, 2e-2, "x[1] must be near 0.5");
    ASSERT_NEAR(res.objective, 0.25, 2e-2, "objective must be near 0.25");
    return 0;
}

int main() {
    std::cout << "====================================================" << std::endl;
    std::cout << "  HPR-QP Modern C++20 Core Unit Test Suite" << std::endl;
    std::cout << "====================================================" << std::endl;

    if (test_solver_config() != 0) return 1;
    if (test_1d_unconstrained() != 0) return 1;
    if (test_2d_equality_constrained() != 0) return 1;

    std::cout << "====================================================" << std::endl;
    std::cout << "  All C++20 tests passed successfully!" << std::endl;
    std::cout << "====================================================" << std::endl;
    return 0;
}
