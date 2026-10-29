#ifndef HPR_QP_H
#define HPR_QP_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdbool.h>

/**
 * Convex Composite Quadratic Program (CCQP):
 *   min  0.5 * x^T Q x + c^T x
 *   s.t. l_k <= A x <= u_k
 *        l_c <= x   <= u_c
 */
typedef struct {
    int n;                 /* Number of primal variables */
    int m;                 /* Number of general linear constraints */
    const double *Q;       /* Symmetric PSD matrix (n x n, row-major) */
    const double *c;       /* Linear cost vector (n) */
    const double *A;       /* Linear constraint matrix (m x n, row-major) */
    const double *l_k;     /* Lower bounds on Ax (m), NULL for -inf */
    const double *u_k;     /* Upper bounds on Ax (m), NULL for +inf */
    const double *l_c;     /* Lower bounds on x (n), NULL for -inf */
    const double *u_c;     /* Upper bounds on x (n), NULL for +inf */
} HPRQPProblem;

typedef struct {
    int max_iter;          /* Maximum iterations */
    double tol;            /* Tolerance on maximal KKT residual */
    double sigma;          /* Initial penalty parameter (<=0 for auto) */
    int restart_freq;      /* Adaptive restart frequency (steps) */
} HPRQPConfig;

typedef struct {
    double *x;             /* Primal solution (n) */
    double *w;             /* Dual subspace variable w in Range(Q) (n) */
    double *y;             /* Multipliers for Ax in K (m) */
    double *z;             /* Subdifferential multiplier for x in C (n) */
    int iterations;        /* Total iterations executed */
    double runtime_sec;    /* Total solve time in seconds */
    double kkt_residual;   /* Final maximal KKT residual */
    double objective;      /* Optimal objective value */
    bool converged;        /* True if kkt_residual <= tol */
} HPRQPResult;

/* Default configuration */
HPRQPConfig hpr_qp_default_config(void);

/* Main HPR-QP Solver (Algorithm 7 & 8) */
HPRQPResult hpr_qp_solve(const HPRQPProblem *prob, const HPRQPConfig *cfg);

/* Free allocated result memory */
void hpr_qp_free_result(HPRQPResult *res);

#ifdef __cplusplus
}
#endif

#endif /* HPR_QP_H */
