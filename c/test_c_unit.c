/**
 * Unit Test Harness for C99 HPR-QP Solver Implementation
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "hpr_qp.h"

#define TEST_PASS 0
#define TEST_FAIL 1

static int tests_run = 0;
static int tests_passed = 0;

#define ASSERT_TRUE(cond, msg) do { \
    tests_run++; \
    if (!(cond)) { \
        printf("  [FAIL] Line %d: %s\n", __LINE__, msg); \
        return TEST_FAIL; \
    } else { \
        tests_passed++; \
    } \
} while (0)

#define ASSERT_NEAR(a, b, tol, msg) do { \
    tests_run++; \
    if (fabs((a) - (b)) > (tol)) { \
        printf("  [FAIL] Line %d: %s (expected %.6f, got %.6f, diff %.6e)\n", \
               __LINE__, msg, (double)(b), (double)(a), fabs((a) - (b))); \
        return TEST_FAIL; \
    } else { \
        tests_passed++; \
    } \
} while (0)

/* Test 1: Configuration defaults */
static int test_default_config(void) {
    printf("[TEST 1] Default configuration parameters...\n");
    HPRQPConfig cfg = hpr_qp_default_config();
    ASSERT_TRUE(cfg.max_iter > 0, "max_iter must be positive");
    ASSERT_TRUE(cfg.tol > 0.0 && cfg.tol < 1.0, "tol must be in (0, 1)");
    ASSERT_TRUE(cfg.restart_freq >= 0, "restart_freq must be non-negative");
    return TEST_PASS;
}

/* Test 2: 2D Box-Constrained QP */
static int test_2d_box_qp(void) {
    printf("[TEST 2] 2D Box-Constrained Quadratic Program...\n");
    /*
     * Problem:
     *   min  0.5 * (2*x0^2 + 2*x1^2) - 4*x0 - 6*x1
     *   s.t. 0 <= x0 <= 1.5
     *        0 <= x1 <= 2.0
     *
     * Unconstrained minimum is at x0 = 4/2 = 2, x1 = 6/2 = 3.
     * Both exceed upper bounds, so projected solution must hit bounds:
     *   x0* = 1.5, x1* = 2.0
     *
     * Optimal objective:
     *   0.5 * (2*(1.5)^2 + 2*(2.0)^2) - 4*(1.5) - 6*(2.0)
     *   = 0.5 * (4.5 + 8.0) - 6.0 - 12.0
     *   = 6.25 - 18.0 = -11.75
     */
    const int n = 2;
    const double Q[4] = { 2.0, 0.0, 0.0, 2.0 };
    const double c[2] = { -4.0, -6.0 };
    const double l_c[2] = { 0.0, 0.0 };
    const double u_c[2] = { 1.5, 2.0 };

    HPRQPProblem prob = {0};
    prob.n = n;
    prob.m = 0;
    prob.Q = Q;
    prob.c = c;
    prob.A = NULL;
    prob.l_k = NULL;
    prob.u_k = NULL;
    prob.l_c = l_c;
    prob.u_c = u_c;

    HPRQPConfig cfg = hpr_qp_default_config();
    cfg.max_iter = 500;
    cfg.tol = 1e-4;

    HPRQPResult res = hpr_qp_solve(&prob, &cfg);

    ASSERT_TRUE(res.converged, "Solver must converge on 2D box QP");
    ASSERT_NEAR(res.x[0], 1.5, 1e-2, "x0 must be near upper bound 1.5");
    ASSERT_NEAR(res.x[1], 2.0, 1e-2, "x1 must be near upper bound 2.0");
    ASSERT_NEAR(res.objective, -11.75, 1e-2, "Objective must match analytical minimum");

    hpr_qp_free_result(&res);
    return TEST_PASS;
}

/* Test 3: Unconstrained 1D QP */
static int test_1d_unconstrained_qp(void) {
    printf("[TEST 3] 1D Strictly Convex Unconstrained QP...\n");
    /*
     * min 0.5 * 4 * x^2 - 8 * x
     * x* = 2.0, f(x*) = 0.5 * 4 * 4 - 16 = -8.0
     */
    const int n = 1;
    const double Q[1] = { 4.0 };
    const double c[1] = { -8.0 };

    HPRQPProblem prob = {0};
    prob.n = n;
    prob.m = 0;
    prob.Q = Q;
    prob.c = c;
    prob.l_c = NULL;
    prob.u_c = NULL;

    HPRQPConfig cfg = hpr_qp_default_config();
    cfg.max_iter = 300;
    cfg.tol = 1e-5;

    HPRQPResult res = hpr_qp_solve(&prob, &cfg);

    ASSERT_TRUE(res.converged, "Solver must converge on 1D QP");
    ASSERT_NEAR(res.x[0], 2.0, 1e-3, "x0 must be 2.0");
    ASSERT_NEAR(res.objective, -8.0, 1e-3, "Objective must be -8.0");

    hpr_qp_free_result(&res);
    return TEST_PASS;
}

int main(void) {
    printf("====================================================\n");
    printf("  HPR-QP C99 Core Unit Test Suite\n");
    printf("====================================================\n");

    if (test_default_config() != TEST_PASS) return 1;
    if (test_2d_box_qp() != TEST_PASS) return 1;
    if (test_1d_unconstrained_qp() != TEST_PASS) return 1;

    printf("====================================================\n");
    printf("  All tests passed! (%d checks)\n", tests_passed);
    printf("====================================================\n");
    return 0;
}
