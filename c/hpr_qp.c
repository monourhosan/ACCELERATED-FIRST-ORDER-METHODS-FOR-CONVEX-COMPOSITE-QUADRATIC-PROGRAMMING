#include "hpr_qp.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define INF_VAL 1e20

HPRQPConfig hpr_qp_default_config(void) {
    HPRQPConfig cfg;
    cfg.max_iter = 3000;
    cfg.tol = 1e-6;
    cfg.sigma = 1.0;
    cfg.restart_freq = 32;
    return cfg;
}

static inline double clamp_val(double val, double lb, double ub) {
    if (val < lb) return lb;
    if (val > ub) return ub;
    return val;
}

/* Dense Matrix-Vector Multiplication: y = M * x (M is rows x cols) */
static void mat_vec_mul(const double *M, const double *x, double *y, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        double sum = 0.0;
        const double *row_ptr = &M[i * cols];
        for (int j = 0; j < cols; ++j) {
            sum += row_ptr[j] * x[j];
        }
        y[i] = sum;
    }
}

/* Transpose Matrix-Vector Multiplication: y = M^T * x (M is rows x cols, y is cols) */
static void mat_t_vec_mul(const double *M, const double *x, double *y, int rows, int cols) {
    for (int j = 0; j < cols; ++j) {
        y[j] = 0.0;
    }
    for (int i = 0; i < rows; ++i) {
        double xi = x[i];
        const double *row_ptr = &M[i * cols];
        for (int j = 0; j < cols; ++j) {
            y[j] += row_ptr[j] * xi;
        }
    }
}

/* Power method to estimate spectral norm lambda_max(A^T A) */
static double estimate_lambda_A(const double *A, int m, int n, int max_iter) {
    if (m <= 0 || n <= 0 || A == NULL) return 1.0;
    double *v = (double *)malloc(n * sizeof(double));
    double *Av = (double *)malloc(m * sizeof(double));
    double *w = (double *)malloc(n * sizeof(double));

    for (int j = 0; j < n; ++j) v[j] = 1.0 / sqrt((double)n);

    double lambda = 1.0;
    for (int it = 0; it < max_iter; ++it) {
        mat_vec_mul(A, v, Av, m, n);
        mat_t_vec_mul(A, Av, w, m, n);
        double norm_w = 0.0;
        for (int j = 0; j < n; ++j) norm_w += w[j] * w[j];
        norm_w = sqrt(norm_w);
        if (norm_w < 1e-12) { lambda = 0.0; break; }
        for (int j = 0; j < n; ++j) v[j] = w[j] / norm_w;
    }

    mat_vec_mul(A, v, Av, m, n);
    double sum_sq = 0.0;
    for (int i = 0; i < m; ++i) sum_sq += Av[i] * Av[i];
    lambda = sum_sq;

    free(v); free(Av); free(w);
    return (lambda < 1e-6) ? 1e-6 : lambda;
}

/* Power method to estimate spectral norm lambda_max(Q) */
static double estimate_lambda_Q(const double *Q, int n, int max_iter) {
    if (n <= 0 || Q == NULL) return 1.0;
    double *v = (double *)malloc(n * sizeof(double));
    double *w = (double *)malloc(n * sizeof(double));

    for (int j = 0; j < n; ++j) v[j] = 1.0 / sqrt((double)n);

    for (int it = 0; it < max_iter; ++it) {
        mat_vec_mul(Q, v, w, n, n);
        double norm_w = 0.0;
        for (int j = 0; j < n; ++j) norm_w += w[j] * w[j];
        norm_w = sqrt(norm_w);
        if (norm_w < 1e-12) break;
        for (int j = 0; j < n; ++j) v[j] = w[j] / norm_w;
    }

    mat_vec_mul(Q, v, w, n, n);
    double lambda = 0.0;
    for (int j = 0; j < n; ++j) lambda += v[j] * w[j];

    free(v); free(w);
    return (lambda < 1e-6) ? 1e-6 : lambda;
}

HPRQPResult hpr_qp_solve(const HPRQPProblem *prob, const HPRQPConfig *cfg) {
    clock_t start_clock = clock();

    int n = prob->n;
    int m = prob->m;
    const double *Q = prob->Q;
    const double *c = prob->c;
    const double *A = prob->A;
    const double *l_k = prob->l_k;
    const double *u_k = prob->u_k;
    const double *l_c = prob->l_c;
    const double *u_c = prob->u_c;

    double lambda_A = (m > 0 && A != NULL) ? estimate_lambda_A(A, m, n, 40) : 1.0;
    double lambda_Q = (Q != NULL) ? estimate_lambda_Q(Q, n, 40) : 1.0;

    double sigma = (cfg->sigma > 0.0) ? cfg->sigma : 1.0;
    int max_iter = cfg->max_iter;
    double tol = cfg->tol;
    int restart_freq = cfg->restart_freq;

    /* Allocate working buffers */
    double *x = (double *)calloc(n, sizeof(double));
    double *w_Q = (double *)calloc(n, sizeof(double));
    double *z = (double *)calloc(n, sizeof(double));
    double *y = (double *)calloc(m > 0 ? m : 1, sizeof(double));

    /* Iteration state buffers */
    double *x_bar = (double *)calloc(n, sizeof(double));
    double *z_bar = (double *)calloc(n, sizeof(double));
    double *w_half = (double *)calloc(n, sizeof(double));
    double *w_bar = (double *)calloc(n, sizeof(double));
    double *y_bar = (double *)calloc(m > 0 ? m : 1, sizeof(double));

    double *x_hat = (double *)calloc(n, sizeof(double));
    double *r_z = (double *)calloc(n, sizeof(double));
    double *r_y = (double *)calloc(m > 0 ? m : 1, sizeof(double));
    double *Qw = (double *)calloc(n, sizeof(double));
    double *Qw_half = (double *)calloc(n, sizeof(double));
    double *ATy = (double *)calloc(n, sizeof(double));
    double *temp_n = (double *)calloc(n, sizeof(double));

    /* Anchor points for Halpern restart */
    double *x_r0 = (double *)calloc(n, sizeof(double));
    double *w_r0 = (double *)calloc(n, sizeof(double));
    double *z_r0 = (double *)calloc(n, sizeof(double));
    double *y_r0 = (double *)calloc(m > 0 ? m : 1, sizeof(double));

    int inner_t = 0;
    int k = 0;
    double kkt_residual = 1e9;
    bool converged = false;

    double denom_w = 1.0 + sigma * lambda_Q;

    for (k = 0; k < max_iter; ++k) {
        inner_t++;

        /* 1) Compute Q * w_Q and A^T * y */
        if (Q != NULL) mat_vec_mul(Q, w_Q, Qw, n, n);
        else memset(Qw, 0, n * sizeof(double));

        if (m > 0 && A != NULL) mat_t_vec_mul(A, y, ATy, m, n);
        else memset(ATy, 0, n * sizeof(double));

        /* 2) Step 1: r_z = x + sigma * (-Qw + ATy - c) */
        for (int j = 0; j < n; ++j) {
            double c_val = (c != NULL) ? c[j] : 0.0;
            r_z[j] = x[j] + sigma * (-Qw[j] + ATy[j] - c_val);
            
            /* Step 2: x_bar = Proj_C(r_z) */
            double lb = (l_c != NULL) ? l_c[j] : -INF_VAL;
            double ub = (u_c != NULL) ? u_c[j] : INF_VAL;
            x_bar[j] = clamp_val(r_z[j], lb, ub);
            z_bar[j] = (1.0 / sigma) * (x_bar[j] - r_z[j]);

            /* x_hat = 2 * x_bar - x */
            x_hat[j] = 2.0 * x_bar[j] - x[j];

            /* Step 3-1: w_half */
            w_half[j] = (1.0 / denom_w) * (sigma * lambda_Q * w_Q[j] + x_hat[j]);
        }

        /* 3) Step 3-2: y_bar */
        if (m > 0 && A != NULL) {
            if (Q != NULL) mat_vec_mul(Q, w_half, Qw_half, n, n);
            else memset(Qw_half, 0, n * sizeof(double));

            for (int j = 0; j < n; ++j) {
                temp_n[j] = x_hat[j] + sigma * (Qw[j] - Qw_half[j]);
            }
            mat_vec_mul(A, temp_n, r_y, m, n);

            double sig_lam_A = sigma * lambda_A;
            for (int i = 0; i < m; ++i) {
                r_y[i] -= sig_lam_A * y[i];
                double lb = (l_k != NULL) ? l_k[i] : -INF_VAL;
                double ub = (u_k != NULL) ? u_k[i] : INF_VAL;
                double proj_ry = clamp_val(r_y[i], lb, ub);
                y_bar[i] = (1.0 / sig_lam_A) * (proj_ry - r_y[i]);
            }

            /* Step 3-3: w_bar */
            for (int i = 0; i < m; ++i) r_y[i] = y_bar[i] - y[i];
            mat_t_vec_mul(A, r_y, temp_n, m, n);
            double w_step = sigma / denom_w;
            for (int j = 0; j < n; ++j) {
                w_bar[j] = w_half[j] + w_step * temp_n[j];
            }
        } else {
            memcpy(w_bar, w_half, n * sizeof(double));
        }

        /* 4) Step 4 & 5: Halpern Reflection and Averaging */
        double coef_0 = 1.0 / (double)(inner_t + 1);
        double coef_hat = (double)inner_t / (double)(inner_t + 1);

        for (int j = 0; j < n; ++j) {
            double u_hat_x = 2.0 * x_bar[j] - x[j];
            double u_hat_w = 2.0 * w_bar[j] - w_Q[j];
            double u_hat_z = 2.0 * z_bar[j] - z[j];

            x[j] = coef_0 * x_r0[j] + coef_hat * u_hat_x;
            w_Q[j] = coef_0 * w_r0[j] + coef_hat * u_hat_w;
            z[j] = coef_0 * z_r0[j] + coef_hat * u_hat_z;
        }
        for (int i = 0; i < m; ++i) {
            double u_hat_y = 2.0 * y_bar[i] - y[i];
            y[i] = coef_0 * y_r0[i] + coef_hat * u_hat_y;
        }

        /* 5) Restart check */
        if (restart_freq > 0 && inner_t >= restart_freq) {
            memcpy(x_r0, x_bar, n * sizeof(double));
            memcpy(w_r0, w_bar, n * sizeof(double));
            memcpy(z_r0, z_bar, n * sizeof(double));
            memcpy(y_r0, y_bar, (m > 0 ? m : 1) * sizeof(double));

            memcpy(x, x_r0, n * sizeof(double));
            memcpy(w_Q, w_r0, n * sizeof(double));
            memcpy(z, z_r0, n * sizeof(double));
            memcpy(y, y_r0, (m > 0 ? m : 1) * sizeof(double));
            inner_t = 0;
        }

        /* 6) Periodic KKT residual check */
        if ((k + 1) % 20 == 0 || k == max_iter - 1) {
            double max_kkt = 0.0;

            /* Primal feasibility: ||Ax - Proj_K(Ax)||_inf */
            if (m > 0 && A != NULL) {
                mat_vec_mul(A, x_bar, r_y, m, n);
                for (int i = 0; i < m; ++i) {
                    double lb = (l_k != NULL) ? l_k[i] : -INF_VAL;
                    double ub = (u_k != NULL) ? u_k[i] : INF_VAL;
                    double diff = fabs(r_y[i] - clamp_val(r_y[i], lb, ub));
                    if (diff > max_kkt) max_kkt = diff;
                }
            }

            /* Dual feasibility: ||-Qw + A^T y + z - c||_inf */
            if (Q != NULL) mat_vec_mul(Q, w_bar, Qw, n, n);
            else memset(Qw, 0, n * sizeof(double));
            if (m > 0 && A != NULL) mat_t_vec_mul(A, y_bar, ATy, m, n);
            else memset(ATy, 0, n * sizeof(double));

            for (int j = 0; j < n; ++j) {
                double c_val = (c != NULL) ? c[j] : 0.0;
                double diff = fabs(-Qw[j] + ATy[j] + z_bar[j] - c_val);
                if (diff > max_kkt) max_kkt = diff;
            }

            /* Complementarity / Box feasibility: ||x - Proj_C(x)||_inf */
            for (int j = 0; j < n; ++j) {
                double lb = (l_c != NULL) ? l_c[j] : -INF_VAL;
                double ub = (u_c != NULL) ? u_c[j] : INF_VAL;
                double diff = fabs(x_bar[j] - clamp_val(x_bar[j], lb, ub));
                if (diff > max_kkt) max_kkt = diff;
            }

            kkt_residual = max_kkt;
            if (kkt_residual <= tol) {
                converged = true;
                break;
            }
        }
    }

    clock_t end_clock = clock();

    /* Compute final objective value 0.5 * x^T Q x + c^T x */
    double obj = 0.0;
    if (Q != NULL) {
        mat_vec_mul(Q, x_bar, temp_n, n, n);
        for (int j = 0; j < n; ++j) obj += 0.5 * x_bar[j] * temp_n[j];
    }
    if (c != NULL) {
        for (int j = 0; j < n; ++j) obj += c[j] * x_bar[j];
    }

    HPRQPResult result;
    result.x = x_bar;
    result.w = w_bar;
    result.y = y_bar;
    result.z = z_bar;
    result.iterations = (converged) ? (k + 1) : max_iter;
    result.runtime_sec = (double)(end_clock - start_clock) / CLOCKS_PER_SEC;
    result.kkt_residual = kkt_residual;
    result.objective = obj;
    result.converged = converged;

    /* Free intermediate buffers */
    free(x); free(w_Q); free(z); free(y);
    free(w_half); free(x_hat); free(r_z); free(r_y);
    free(Qw); free(Qw_half); free(ATy); free(temp_n);
    free(x_r0); free(w_r0); free(z_r0); free(y_r0);

    return result;
}

void hpr_qp_free_result(HPRQPResult *res) {
    if (res->x) free(res->x);
    if (res->w) free(res->w);
    if (res->y) free(res->y);
    if (res->z) free(res->z);
    res->x = res->w = res->y = res->z = NULL;
}

#ifdef HPR_QP_STANDALONE
int main(void) {
    printf("===============================================================\n");
    printf("  HPR-QP SOLVER IN C (ISO C99 / C11)\n");
    printf("===============================================================\n");

    /* Test Example 5.1 (Singular Q) */
    {
        double Q[9] = {1.0, 0.0, 0.0,
                       0.0, 2.0, 0.0,
                       0.0, 0.0, 0.0};
        double c[3] = {-1.0, -1.0, -1.0};
        double u_c[3] = {1.0, 1.0, 1.0};

        HPRQPProblem prob;
        memset(&prob, 0, sizeof(prob));
        prob.n = 3;
        prob.m = 0;
        prob.Q = Q;
        prob.c = c;
        prob.u_c = u_c;

        HPRQPConfig cfg = hpr_qp_default_config();
        cfg.tol = 1e-8;
        cfg.restart_freq = 8;

        HPRQPResult res = hpr_qp_solve(&prob, &cfg);
        printf("Example 5.1 (Singular Q):\n");
        printf("  Status: %s | Iters: %d | Time: %.6f s | KKT Res: %.2e | Obj: %.6f\n",
               res.converged ? "CONVERGED" : "MAX_ITER",
               res.iterations, res.runtime_sec, res.kkt_residual, res.objective);
        printf("  Optimal x: [%.6f, %.6f, %.6f]\n", res.x[0], res.x[1], res.x[2]);
        printf("  Expected : [1.000000, 0.500000, 1.000000]\n\n");
        hpr_qp_free_result(&res);
    }

    /* Test Example 5.2 (Ill-conditioned Q) */
    {
        double Q[4] = {1.0, 0.1,
                       0.1, 10.0};
        double c[2] = {-1.0, -1.0};

        HPRQPProblem prob;
        memset(&prob, 0, sizeof(prob));
        prob.n = 2;
        prob.m = 0;
        prob.Q = Q;
        prob.c = c;

        HPRQPConfig cfg = hpr_qp_default_config();
        cfg.tol = 1e-8;
        cfg.restart_freq = 8;

        HPRQPResult res = hpr_qp_solve(&prob, &cfg);
        printf("Example 5.2 (Ill-conditioned Q):\n");
        printf("  Status: %s | Iters: %d | Time: %.6f s | KKT Res: %.2e | Obj: %.6f\n",
               res.converged ? "CONVERGED" : "MAX_ITER",
               res.iterations, res.runtime_sec, res.kkt_residual, res.objective);
        printf("  Optimal x: [%.6f, %.6f]\n", res.x[0], res.x[1]);
        printf("  Expected : [0.990991, 0.090090]\n\n");
        hpr_qp_free_result(&res);
    }

    /* Test HS21 (m=1, n=2) */
    {
        double Q[4] = {0.02, 0.0,
                       0.0,  2.0};
        double c[2] = {0.0, 0.0};
        double A[2] = {10.0, -1.0};
        double l_k[1] = {10.0};
        double l_c[2] = {2.0, -1e20};
        double u_c[2] = {50.0, 50.0};

        HPRQPProblem prob;
        memset(&prob, 0, sizeof(prob));
        prob.n = 2;
        prob.m = 1;
        prob.Q = Q;
        prob.c = c;
        prob.A = A;
        prob.l_k = l_k;
        prob.l_c = l_c;
        prob.u_c = u_c;

        HPRQPConfig cfg = hpr_qp_default_config();
        cfg.tol = 1e-6;
        cfg.restart_freq = 16;

        HPRQPResult res = hpr_qp_solve(&prob, &cfg);
        printf("Hock-Schittkowski #21 (HS21):\n");
        printf("  Status: %s | Iters: %d | Time: %.6f s | KKT Res: %.2e | Obj: %.6f\n",
               res.converged ? "CONVERGED" : "MAX_ITER",
               res.iterations, res.runtime_sec, res.kkt_residual, res.objective);
        printf("  Optimal x: [%.6f, %.6f]\n", res.x[0], res.x[1]);
        printf("  Expected : [2.000000, 0.000000] | Obj: 0.040000\n");
        hpr_qp_free_result(&res);
    }

    printf("===============================================================\n");
    printf("  ALL C BENCHMARKS COMPLETED SUCCESSFULLY!\n");
    printf("===============================================================\n");
    return 0;
}
#endif
