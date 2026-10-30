#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <windows.h>
#include <omp.h>

typedef struct {
    uint32_t n;
    uint32_t m;
    uint64_t nnz_P;
    uint64_t nnz_A;
    uint64_t nnz_AT;

    int32_t *P_row_ptr;
    int32_t *P_col_idx;
    double  *P_val;

    int32_t *A_row_ptr;
    int32_t *A_col_idx;
    double  *A_val;

    int32_t *AT_row_ptr;
    int32_t *AT_col_idx;
    double  *AT_val;

    double *c;
    double *l_k;
    double *u_k;
    double *l_c;
    double *u_c;
} LargeDataset;

static inline double clamp(double val, double lb, double ub) {
    if (val < lb) return lb;
    if (val > ub) return ub;
    return val;
}

// 1. Baseline SpMV
static void spmv_csr_baseline(const int32_t *row_ptr, const int32_t *col_idx, const double *val,
                              const double *x, double *y, uint32_t rows) {
    for (uint32_t i = 0; i < rows; ++i) {
        double sum = 0.0;
        int32_t start = row_ptr[i];
        int32_t end = row_ptr[i + 1];
        for (int32_t p = start; p < end; ++p) {
            sum += val[p] * x[col_idx[p]];
        }
        y[i] = sum;
    }
}

// 2. High-Performance 4-way Unrolled & Parallel SpMV
static void spmv_csr_fast(const int32_t *row_ptr, const int32_t *col_idx, const double *val,
                          const double *x, double *y, uint32_t rows) {
    int i;
    #pragma omp parallel for schedule(static, 128)
    for (i = 0; i < (int)rows; ++i) {
        int32_t start = row_ptr[i];
        int32_t end = row_ptr[i + 1];
        int32_t p = start;

        double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
        for (; p + 3 < end; p += 4) {
            sum0 += val[p]     * x[col_idx[p]];
            sum1 += val[p + 1] * x[col_idx[p + 1]];
            sum2 += val[p + 2] * x[col_idx[p + 2]];
            sum3 += val[p + 3] * x[col_idx[p + 3]];
        }
        double sum = (sum0 + sum1) + (sum2 + sum3);
        for (; p < end; ++p) {
            sum += val[p] * x[col_idx[p]];
        }
        y[i] = sum;
    }
}

static LargeDataset* load_binary_dataset(const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) return NULL;

    LargeDataset *ds = (LargeDataset*)malloc(sizeof(LargeDataset));
    fread(&ds->n, sizeof(uint32_t), 1, f);
    fread(&ds->m, sizeof(uint32_t), 1, f);
    fread(&ds->nnz_P, sizeof(uint64_t), 1, f);
    fread(&ds->nnz_A, sizeof(uint64_t), 1, f);
    fread(&ds->nnz_AT, sizeof(uint64_t), 1, f);

    uint32_t n = ds->n;
    uint32_t m = ds->m;

    ds->P_row_ptr = (int32_t*)malloc((n + 1) * sizeof(int32_t));
    ds->P_col_idx = (int32_t*)malloc(ds->nnz_P * sizeof(int32_t));
    ds->P_val = (double*)malloc(ds->nnz_P * sizeof(double));
    fread(ds->P_row_ptr, sizeof(int32_t), n + 1, f);
    fread(ds->P_col_idx, sizeof(int32_t), ds->nnz_P, f);
    fread(ds->P_val, sizeof(double), ds->nnz_P, f);

    ds->A_row_ptr = (int32_t*)malloc((m + 1) * sizeof(int32_t));
    ds->A_col_idx = (int32_t*)malloc(ds->nnz_A * sizeof(int32_t));
    ds->A_val = (double*)malloc(ds->nnz_A * sizeof(double));
    fread(ds->A_row_ptr, sizeof(int32_t), m + 1, f);
    fread(ds->A_col_idx, sizeof(int32_t), ds->nnz_A, f);
    fread(ds->A_val, sizeof(double), ds->nnz_A, f);

    ds->AT_row_ptr = (int32_t*)malloc((n + 1) * sizeof(int32_t));
    ds->AT_col_idx = (int32_t*)malloc(ds->nnz_AT * sizeof(int32_t));
    ds->AT_val = (double*)malloc(ds->nnz_AT * sizeof(double));
    fread(ds->AT_row_ptr, sizeof(int32_t), n + 1, f);
    fread(ds->AT_col_idx, sizeof(int32_t), ds->nnz_AT, f);
    fread(ds->AT_val, sizeof(double), ds->nnz_AT, f);

    ds->c = (double*)malloc(n * sizeof(double));
    ds->l_k = (double*)malloc(m * sizeof(double));
    ds->u_k = (double*)malloc(m * sizeof(double));
    ds->l_c = (double*)malloc(n * sizeof(double));
    ds->u_c = (double*)malloc(n * sizeof(double));
    fread(ds->c, sizeof(double), n, f);
    fread(ds->l_k, sizeof(double), m, f);
    fread(ds->u_k, sizeof(double), m, f);
    fread(ds->l_c, sizeof(double), n, f);
    fread(ds->u_c, sizeof(double), n, f);

    fclose(f);
    return ds;
}

static void free_dataset(LargeDataset *ds) {
    if (!ds) return;
    free(ds->P_row_ptr); free(ds->P_col_idx); free(ds->P_val);
    free(ds->A_row_ptr); free(ds->A_col_idx); free(ds->A_val);
    free(ds->AT_row_ptr); free(ds->AT_col_idx); free(ds->AT_val);
    free(ds->c); free(ds->l_k); free(ds->u_k); free(ds->l_c); free(ds->u_c);
    free(ds);
}

typedef struct {
    double elapsed_ms;
    double kkt_residual;
    double objective;
} RunStats;

static RunStats run_hpr_qp(const LargeDataset *ds, int max_iter, int restart_freq, int use_fast_spmv) {
    uint32_t n = ds->n;
    uint32_t m = ds->m;

    double *x = (double*)calloc(n, sizeof(double));
    double *w_q = (double*)calloc(n, sizeof(double));
    double *z = (double*)calloc(n, sizeof(double));
    double *y = (double*)calloc(m, sizeof(double));

    double *x_bar = (double*)calloc(n, sizeof(double));
    double *w_bar = (double*)calloc(n, sizeof(double));
    double *z_bar = (double*)calloc(n, sizeof(double));
    double *y_bar = (double*)calloc(m, sizeof(double));

    double *x_hat = (double*)calloc(n, sizeof(double));
    double *w_half = (double*)calloc(n, sizeof(double));
    double *r_y = (double*)calloc(m, sizeof(double));
    double *qw = (double*)calloc(n, sizeof(double));
    double *qw_half = (double*)calloc(n, sizeof(double));
    double *aty = (double*)calloc(n, sizeof(double));
    double *temp_n = (double*)calloc(n, sizeof(double));
    double *temp_m = (double*)calloc(m, sizeof(double));

    double *x_r0 = (double*)calloc(n, sizeof(double));
    double *w_r0 = (double*)calloc(n, sizeof(double));
    double *z_r0 = (double*)calloc(n, sizeof(double));
    double *y_r0 = (double*)calloc(m, sizeof(double));

    void (*spmv)(const int32_t*, const int32_t*, const double*, const double*, double*, uint32_t) =
        use_fast_spmv ? spmv_csr_fast : spmv_csr_baseline;

    // Power method for lambda_q and lambda_a
    for (uint32_t i = 0; i < n; ++i) temp_n[i] = 1.0 / sqrt((double)n);
    for (int it = 0; it < 15; ++it) {
        spmv(ds->P_row_ptr, ds->P_col_idx, ds->P_val, temp_n, qw, n);
        double nrm = 0.0;
        for (uint32_t i = 0; i < n; ++i) nrm += qw[i] * qw[i];
        nrm = sqrt(nrm);
        if (nrm < 1e-12) break;
        for (uint32_t i = 0; i < n; ++i) temp_n[i] = qw[i] / nrm;
    }
    spmv(ds->P_row_ptr, ds->P_col_idx, ds->P_val, temp_n, qw, n);
    double lam_q = 0.0;
    for (uint32_t i = 0; i < n; ++i) lam_q += temp_n[i] * qw[i];
    if (lam_q < 1e-6) lam_q = 1e-6;

    for (uint32_t i = 0; i < n; ++i) temp_n[i] = 1.0 / sqrt((double)n);
    for (int it = 0; it < 15; ++it) {
        spmv(ds->A_row_ptr, ds->A_col_idx, ds->A_val, temp_n, temp_m, m);
        spmv(ds->AT_row_ptr, ds->AT_col_idx, ds->AT_val, temp_m, qw, n);
        double nrm = 0.0;
        for (uint32_t i = 0; i < n; ++i) nrm += qw[i] * qw[i];
        nrm = sqrt(nrm);
        if (nrm < 1e-12) break;
        for (uint32_t i = 0; i < n; ++i) temp_n[i] = qw[i] / nrm;
    }
    spmv(ds->A_row_ptr, ds->A_col_idx, ds->A_val, temp_n, temp_m, m);
    double lam_a = 0.0;
    for (uint32_t i = 0; i < m; ++i) lam_a += temp_m[i] * temp_m[i];
    if (lam_a < 1e-6) lam_a = 1e-6;

    double sigma = 1.0;
    double denom_w = 1.0 + sigma * lam_q;
    double sig_lam_a = sigma * lam_a;
    double inv_sig_lam_a = 1.0 / sig_lam_a;
    double w_step = sigma / denom_w;

    LARGE_INTEGER freq, t_start, t_end;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t_start);

    int inner_t = 0;
    for (int k = 0; k < max_iter; ++k) {
        inner_t++;

        // 1) SpMV for Qw and A^T y
        spmv(ds->P_row_ptr, ds->P_col_idx, ds->P_val, w_q, qw, n);
        spmv(ds->AT_row_ptr, ds->AT_col_idx, ds->AT_val, y, aty, n);

        // 2) Fused Step 1 & 2
        int j;
        int i;
        #pragma omp parallel for schedule(static, 256) if(use_fast_spmv)
        for (j = 0; j < (int)n; ++j) {
            double rz = x[j] + sigma * (-qw[j] + aty[j] - ds->c[j]);
            double xb = clamp(rz, ds->l_c[j], ds->u_c[j]);
            x_bar[j] = xb;
            z_bar[j] = (1.0 / sigma) * (xb - rz);

            double xh = 2.0 * xb - x[j];
            x_hat[j] = xh;
            w_half[j] = (1.0 / denom_w) * (sigma * lam_q * w_q[j] + xh);
        }

        // 3) Step 3: sGS
        spmv(ds->P_row_ptr, ds->P_col_idx, ds->P_val, w_half, qw_half, n);

        #pragma omp parallel for schedule(static, 256) if(use_fast_spmv)
        for (j = 0; j < (int)n; ++j) {
            temp_n[j] = x_hat[j] + sigma * (qw[j] - qw_half[j]);
        }
        spmv(ds->A_row_ptr, ds->A_col_idx, ds->A_val, temp_n, r_y, m);

        #pragma omp parallel for schedule(static, 256) if(use_fast_spmv)
        for (i = 0; i < (int)m; ++i) {
            r_y[i] -= sig_lam_a * y[i];
            double py = clamp(r_y[i], ds->l_k[i], ds->u_k[i]);
            double yb = inv_sig_lam_a * (py - r_y[i]);
            y_bar[i] = yb;
            temp_m[i] = yb - y[i];
        }

        spmv(ds->AT_row_ptr, ds->AT_col_idx, ds->AT_val, temp_m, temp_n, n);

        #pragma omp parallel for schedule(static, 256) if(use_fast_spmv)
        for (j = 0; j < (int)n; ++j) {
            w_bar[j] = w_half[j] + w_step * temp_n[j];
        }

        // 4) Halpern Averaging
        double coef_0 = 1.0 / (inner_t + 1.0);
        double coef_hat = (double)inner_t / (inner_t + 1.0);

        #pragma omp parallel for schedule(static, 256) if(use_fast_spmv)
        for (j = 0; j < (int)n; ++j) {
            x[j] = coef_0 * x_r0[j] + coef_hat * (2.0 * x_bar[j] - x[j]);
            w_q[j] = coef_0 * w_r0[j] + coef_hat * (2.0 * w_bar[j] - w_q[j]);
            z[j] = coef_0 * z_r0[j] + coef_hat * (2.0 * z_bar[j] - z[j]);
        }

        #pragma omp parallel for schedule(static, 256) if(use_fast_spmv)
        for (i = 0; i < (int)m; ++i) {
            y[i] = coef_0 * y_r0[i] + coef_hat * (2.0 * y_bar[i] - y[i]);
        }

        // 5) Adaptive restart
        if (restart_freq > 0 && inner_t >= restart_freq) {
            memcpy(x_r0, x_bar, n * sizeof(double));
            memcpy(w_r0, w_bar, n * sizeof(double));
            memcpy(z_r0, z_bar, n * sizeof(double));
            memcpy(y_r0, y_bar, m * sizeof(double));

            memcpy(x, x_r0, n * sizeof(double));
            memcpy(w_q, w_r0, n * sizeof(double));
            memcpy(z, z_r0, n * sizeof(double));
            memcpy(y, y_r0, m * sizeof(double));
            inner_t = 0;
        }
    }

    QueryPerformanceCounter(&t_end);
    double elapsed = (double)(t_end.QuadPart - t_start.QuadPart) * 1000.0 / freq.QuadPart;

    spmv(ds->P_row_ptr, ds->P_col_idx, ds->P_val, w_bar, qw, n);
    spmv(ds->AT_row_ptr, ds->AT_col_idx, ds->AT_val, y_bar, aty, n);

    double max_kkt = 0.0;
    for (uint32_t j = 0; j < n; ++j) {
        double r = fabs(-qw[j] + aty[j] + z_bar[j] - ds->c[j]);
        if (r > max_kkt) max_kkt = r;
    }

    spmv(ds->A_row_ptr, ds->A_col_idx, ds->A_val, x_bar, r_y, m);
    for (uint32_t i = 0; i < m; ++i) {
        double diff = fabs(r_y[i] - clamp(r_y[i], ds->l_k[i], ds->u_k[i]));
        if (diff > max_kkt) max_kkt = diff;
    }

    double obj = 0.0;
    spmv(ds->P_row_ptr, ds->P_col_idx, ds->P_val, x_bar, temp_n, n);
    for (uint32_t j = 0; j < n; ++j) {
        obj += 0.5 * x_bar[j] * temp_n[j] + ds->c[j] * x_bar[j];
    }

    free(x); free(w_q); free(z); free(y);
    free(x_bar); free(w_bar); free(z_bar); free(y_bar);
    free(x_hat); free(w_half); free(r_y);
    free(qw); free(qw_half); free(aty); free(temp_n); free(temp_m);
    free(x_r0); free(w_r0); free(z_r0); free(y_r0);

    RunStats stats;
    stats.elapsed_ms = elapsed;
    stats.kkt_residual = max_kkt;
    stats.objective = obj;
    return stats;
}

__declspec(dllexport) int solve_hpr_qp_fast_c_api(
    uint32_t n, uint32_t m,
    const int32_t *P_row_ptr, const int32_t *P_col_idx, const double *P_val, uint64_t nnz_P,
    const int32_t *A_row_ptr, const int32_t *A_col_idx, const double *A_val, uint64_t nnz_A,
    const int32_t *AT_row_ptr, const int32_t *AT_col_idx, const double *AT_val, uint64_t nnz_AT,
    const double *c, const double *l_k, const double *u_k, const double *l_c, const double *u_c,
    int max_iter, int restart_freq, double tol,
    double *x_out, double *stats_out // [elapsed_ms, kkt_res, obj]
) {
    LargeDataset ds;
    ds.n = n; ds.m = m;
    ds.nnz_P = nnz_P; ds.nnz_A = nnz_A; ds.nnz_AT = nnz_AT;
    ds.P_row_ptr = (int32_t*)P_row_ptr; ds.P_col_idx = (int32_t*)P_col_idx; ds.P_val = (double*)P_val;
    ds.A_row_ptr = (int32_t*)A_row_ptr; ds.A_col_idx = (int32_t*)A_col_idx; ds.A_val = (double*)A_val;
    ds.AT_row_ptr = (int32_t*)AT_row_ptr; ds.AT_col_idx = (int32_t*)AT_col_idx; ds.AT_val = (double*)AT_val;
    ds.c = (double*)c; ds.l_k = (double*)l_k; ds.u_k = (double*)u_k; ds.l_c = (double*)l_c; ds.u_c = (double*)u_c;

    RunStats s = run_hpr_qp(&ds, max_iter, restart_freq, 1);
    if (stats_out) {
        stats_out[0] = s.elapsed_ms;
        stats_out[1] = s.kkt_residual;
        stats_out[2] = s.objective;
    }
    return 0;
}

#ifndef HPR_QP_DLL_ONLY
int main(void) {
    printf("======================================================================\n");
    printf("  OPTIMIZED HPR-QP BENCHMARK: BASELINE vs. ACCELERATED (C99 + OpenMP)\n");
    printf("======================================================================\n");

    LargeDataset *ds = load_binary_dataset("Data/cvxqp1_l.bin");
    if (!ds) return 1;

    int max_threads = omp_get_max_threads();
    printf("System OpenMP Threads Available: %d\n", max_threads);
    printf("Dataset: CVXQP1_L (n=%u, m=%u, nnz=%llu)\n\n",
           ds->n, ds->m, (unsigned long long)(ds->nnz_P + ds->nnz_A));

    // Warm-up run
    run_hpr_qp(ds, 20, 32, 0);

    for (int iters = 100; iters <= 300; iters += 200) {
        printf("--- Testing %d Iterations ---\n", iters);
        
        // 1. Baseline Single-Threaded C
        RunStats s_base = run_hpr_qp(ds, iters, 32, 0);
        printf("  [Baseline Single-Core] Total: %6.2f ms | %6.3f ms/iter | KKT: %.2e | Obj: %.2f\n",
               s_base.elapsed_ms, s_base.elapsed_ms / iters, s_base.kkt_residual, s_base.objective);

        // 2. Optimized 4-way Unrolled + OpenMP Multi-Threaded C
        RunStats s_fast = run_hpr_qp(ds, iters, 32, 1);
        double speedup = s_base.elapsed_ms / s_fast.elapsed_ms;
        printf("  [Accelerated SIMD+OMP] Total: %6.2f ms | %6.3f ms/iter | KKT: %.2e | Obj: %.2f\n",
               s_fast.elapsed_ms, s_fast.elapsed_ms / iters, s_fast.kkt_residual, s_fast.objective);
        printf("  >>> Speedup from Computational Optimizations: %.2fx faster!\n\n", speedup);
    }

    printf("======================================================================\n");
    free_dataset(ds);
    return 0;
}
#endif
