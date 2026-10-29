#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <algorithm>
#include <omp.h>

struct CSRMatrix {
    uint32_t rows{0};
    uint32_t cols{0};
    uint64_t nnz{0};
    std::vector<int32_t> row_ptr;
    std::vector<int32_t> col_idx;
    std::vector<double>  val;

    // Baseline single-threaded SpMV
    void spmv_baseline(const double* x, double* y) const {
        for (uint32_t i = 0; i < rows; ++i) {
            double sum = 0.0;
            const int32_t start = row_ptr[i];
            const int32_t end = row_ptr[i + 1];
            for (int32_t p = start; p < end; ++p) {
                sum += val[p] * x[col_idx[p]];
            }
            y[i] = sum;
        }
    }

    // High-performance 4-way unrolled & OpenMP parallelized SpMV
    void spmv_fast(const double* x, double* y) const {
        int r = static_cast<int>(rows);
        #pragma omp parallel for schedule(static, 128)
        for (int i = 0; i < r; ++i) {
            const int32_t start = row_ptr[i];
            const int32_t end = row_ptr[i + 1];
            int32_t p = start;

            double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
            for (; p + 3 < end; p += 4) {
                s0 += val[p]     * x[col_idx[p]];
                s1 += val[p + 1] * x[col_idx[p + 1]];
                s2 += val[p + 2] * x[col_idx[p + 2]];
                s3 += val[p + 3] * x[col_idx[p + 3]];
            }
            double sum = (s0 + s1) + (s2 + s3);
            for (; p < end; ++p) {
                sum += val[p] * x[col_idx[p]];
            }
            y[i] = sum;
        }
    }
};

struct LargeQPProblem {
    uint32_t n{0};
    uint32_t m{0};
    CSRMatrix P;
    CSRMatrix A;
    CSRMatrix AT;
    std::vector<double> c;
    std::vector<double> l_k;
    std::vector<double> u_k;
    std::vector<double> l_c;
    std::vector<double> u_c;
};

static LargeQPProblem load_binary_dataset(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Could not open file: " + path);

    LargeQPProblem prob;
    uint64_t nnz_P = 0, nnz_A = 0, nnz_AT = 0;

    f.read(reinterpret_cast<char*>(&prob.n), sizeof(uint32_t));
    f.read(reinterpret_cast<char*>(&prob.m), sizeof(uint32_t));
    f.read(reinterpret_cast<char*>(&nnz_P), sizeof(uint64_t));
    f.read(reinterpret_cast<char*>(&nnz_A), sizeof(uint64_t));
    f.read(reinterpret_cast<char*>(&nnz_AT), sizeof(uint64_t));

    prob.P.rows = prob.n; prob.P.cols = prob.n; prob.P.nnz = nnz_P;
    prob.P.row_ptr.resize(prob.n + 1); prob.P.col_idx.resize(nnz_P); prob.P.val.resize(nnz_P);
    f.read(reinterpret_cast<char*>(prob.P.row_ptr.data()), (prob.n + 1) * sizeof(int32_t));
    f.read(reinterpret_cast<char*>(prob.P.col_idx.data()), nnz_P * sizeof(int32_t));
    f.read(reinterpret_cast<char*>(prob.P.val.data()), nnz_P * sizeof(double));

    prob.A.rows = prob.m; prob.A.cols = prob.n; prob.A.nnz = nnz_A;
    prob.A.row_ptr.resize(prob.m + 1); prob.A.col_idx.resize(nnz_A); prob.A.val.resize(nnz_A);
    f.read(reinterpret_cast<char*>(prob.A.row_ptr.data()), (prob.m + 1) * sizeof(int32_t));
    f.read(reinterpret_cast<char*>(prob.A.col_idx.data()), nnz_A * sizeof(int32_t));
    f.read(reinterpret_cast<char*>(prob.A.val.data()), nnz_A * sizeof(double));

    prob.AT.rows = prob.n; prob.AT.cols = prob.m; prob.AT.nnz = nnz_AT;
    prob.AT.row_ptr.resize(prob.n + 1); prob.AT.col_idx.resize(nnz_AT); prob.AT.val.resize(nnz_AT);
    f.read(reinterpret_cast<char*>(prob.AT.row_ptr.data()), (prob.n + 1) * sizeof(int32_t));
    f.read(reinterpret_cast<char*>(prob.AT.col_idx.data()), nnz_AT * sizeof(int32_t));
    f.read(reinterpret_cast<char*>(prob.AT.val.data()), nnz_AT * sizeof(double));

    prob.c.resize(prob.n);
    prob.l_k.resize(prob.m);
    prob.u_k.resize(prob.m);
    prob.l_c.resize(prob.n);
    prob.u_c.resize(prob.n);

    f.read(reinterpret_cast<char*>(prob.c.data()), prob.n * sizeof(double));
    f.read(reinterpret_cast<char*>(prob.l_k.data()), prob.m * sizeof(double));
    f.read(reinterpret_cast<char*>(prob.u_k.data()), prob.m * sizeof(double));
    f.read(reinterpret_cast<char*>(prob.l_c.data()), prob.n * sizeof(double));
    f.read(reinterpret_cast<char*>(prob.u_c.data()), prob.n * sizeof(double));

    return prob;
}

inline double clamp(double v, double lo, double hi) noexcept {
    return std::max(lo, std::min(hi, v));
}

struct BenchmarkResult {
    double elapsed_ms;
    double kkt_residual;
    double objective;
};

BenchmarkResult run_hpr_qp_cpp(const LargeQPProblem& prob, int max_iter, int restart_freq, bool fast_mode) {
    const uint32_t n = prob.n;
    const uint32_t m = prob.m;

    std::vector<double> x(n, 0.0), w_q(n, 0.0), z(n, 0.0), y(m, 0.0);
    std::vector<double> x_bar(n, 0.0), w_bar(n, 0.0), z_bar(n, 0.0), y_bar(m, 0.0);
    std::vector<double> x_hat(n, 0.0), w_half(n, 0.0);
    std::vector<double> r_y(m, 0.0), qw(n, 0.0), qw_half(n, 0.0), aty(n, 0.0);
    std::vector<double> temp_n(n, 0.0), temp_m(m, 0.0);

    std::vector<double> x_r0(n, 0.0), w_r0(n, 0.0), z_r0(n, 0.0), y_r0(m, 0.0);

    auto spmv_P = [&](const double* in, double* out) {
        if (fast_mode) prob.P.spmv_fast(in, out);
        else prob.P.spmv_baseline(in, out);
    };
    auto spmv_A = [&](const double* in, double* out) {
        if (fast_mode) prob.A.spmv_fast(in, out);
        else prob.A.spmv_baseline(in, out);
    };
    auto spmv_AT = [&](const double* in, double* out) {
        if (fast_mode) prob.AT.spmv_fast(in, out);
        else prob.AT.spmv_baseline(in, out);
    };

    // Power method for lambda_q
    const double inv_sqrt_n = 1.0 / std::sqrt(static_cast<double>(n));
    std::fill(temp_n.begin(), temp_n.end(), inv_sqrt_n);
    for (int it = 0; it < 15; ++it) {
        spmv_P(temp_n.data(), qw.data());
        double nrm = 0.0;
        for (uint32_t i = 0; i < n; ++i) nrm += qw[i] * qw[i];
        nrm = std::sqrt(nrm);
        if (nrm < 1e-12) break;
        for (uint32_t i = 0; i < n; ++i) temp_n[i] = qw[i] / nrm;
    }
    spmv_P(temp_n.data(), qw.data());
    double lam_q = 0.0;
    for (uint32_t i = 0; i < n; ++i) lam_q += temp_n[i] * qw[i];
    lam_q = std::max(lam_q, 1e-6);

    // Power method for lambda_a
    std::fill(temp_n.begin(), temp_n.end(), inv_sqrt_n);
    for (int it = 0; it < 15; ++it) {
        spmv_A(temp_n.data(), temp_m.data());
        spmv_AT(temp_m.data(), qw.data());
        double nrm = 0.0;
        for (uint32_t i = 0; i < n; ++i) nrm += qw[i] * qw[i];
        nrm = std::sqrt(nrm);
        if (nrm < 1e-12) break;
        for (uint32_t i = 0; i < n; ++i) temp_n[i] = qw[i] / nrm;
    }
    spmv_A(temp_n.data(), temp_m.data());
    double lam_a = 0.0;
    for (uint32_t i = 0; i < m; ++i) lam_a += temp_m[i] * temp_m[i];
    lam_a = std::max(lam_a, 1e-6);

    const double sigma = 1.0;
    const double denom_w = 1.0 + sigma * lam_q;
    const double sig_lam_a = sigma * lam_a;
    const double inv_sig_lam_a = 1.0 / sig_lam_a;
    const double w_step = sigma / denom_w;

    auto t_start = std::chrono::high_resolution_clock::now();
    int inner_t = 0;

    for (int k = 0; k < max_iter; ++k) {
        inner_t++;

        // 1) SpMV for Qw and A^T y
        spmv_P(w_q.data(), qw.data());
        spmv_AT(y.data(), aty.data());

        // 2) Fused Step 1 & 2
        int j_len = static_cast<int>(n);
        int i_len = static_cast<int>(m);

        #pragma omp parallel for schedule(static, 256) if(fast_mode)
        for (int j = 0; j < j_len; ++j) {
            const double rz = x[j] + sigma * (-qw[j] + aty[j] - prob.c[j]);
            const double xb = clamp(rz, prob.l_c[j], prob.u_c[j]);
            x_bar[j] = xb;
            z_bar[j] = (1.0 / sigma) * (xb - rz);

            const double xh = 2.0 * xb - x[j];
            x_hat[j] = xh;
            w_half[j] = (1.0 / denom_w) * (sigma * lam_q * w_q[j] + xh);
        }

        // 3) Step 3: sGS
        spmv_P(w_half.data(), qw_half.data());

        #pragma omp parallel for schedule(static, 256) if(fast_mode)
        for (int j = 0; j < j_len; ++j) {
            temp_n[j] = x_hat[j] + sigma * (qw[j] - qw_half[j]);
        }
        spmv_A(temp_n.data(), r_y.data());

        #pragma omp parallel for schedule(static, 256) if(fast_mode)
        for (int i = 0; i < i_len; ++i) {
            r_y[i] -= sig_lam_a * y[i];
            const double py = clamp(r_y[i], prob.l_k[i], prob.u_k[i]);
            const double yb = inv_sig_lam_a * (py - r_y[i]);
            y_bar[i] = yb;
            temp_m[i] = yb - y[i];
        }

        spmv_AT(temp_m.data(), temp_n.data());

        #pragma omp parallel for schedule(static, 256) if(fast_mode)
        for (int j = 0; j < j_len; ++j) {
            w_bar[j] = w_half[j] + w_step * temp_n[j];
        }

        // 4) Halpern Averaging
        const double coef_0 = 1.0 / (inner_t + 1.0);
        const double coef_hat = static_cast<double>(inner_t) / (inner_t + 1.0);

        #pragma omp parallel for schedule(static, 256) if(fast_mode)
        for (int j = 0; j < j_len; ++j) {
            x[j] = coef_0 * x_r0[j] + coef_hat * (2.0 * x_bar[j] - x[j]);
            w_q[j] = coef_0 * w_r0[j] + coef_hat * (2.0 * w_bar[j] - w_q[j]);
            z[j] = coef_0 * z_r0[j] + coef_hat * (2.0 * z_bar[j] - z[j]);
        }

        #pragma omp parallel for schedule(static, 256) if(fast_mode)
        for (int i = 0; i < i_len; ++i) {
            y[i] = coef_0 * y_r0[i] + coef_hat * (2.0 * y_bar[i] - y[i]);
        }

        // 5) Adaptive restart
        if (restart_freq > 0 && inner_t >= restart_freq) {
            x_r0 = x_bar;
            w_r0 = w_bar;
            z_r0 = z_bar;
            y_r0 = y_bar;

            x = x_r0;
            w_q = w_r0;
            z = z_r0;
            y = y_r0;
            inner_t = 0;
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    // KKT residual & objective
    spmv_P(w_bar.data(), qw.data());
    spmv_AT(y_bar.data(), aty.data());

    double max_kkt = 0.0;
    for (uint32_t j = 0; j < n; ++j) {
        max_kkt = std::max(max_kkt, std::abs(-qw[j] + aty[j] + z_bar[j] - prob.c[j]));
    }
    spmv_A(x_bar.data(), r_y.data());
    for (uint32_t i = 0; i < m; ++i) {
        max_kkt = std::max(max_kkt, std::abs(r_y[i] - clamp(r_y[i], prob.l_k[i], prob.u_k[i])));
    }

    double obj = 0.0;
    spmv_P(x_bar.data(), temp_n.data());
    for (uint32_t j = 0; j < n; ++j) {
        obj += 0.5 * x_bar[j] * temp_n[j] + prob.c[j] * x_bar[j];
    }

    return {elapsed_ms, max_kkt, obj};
}

int main() {
    std::cout << "======================================================================\n";
    std::cout << "  OPTIMIZED HPR-QP BENCHMARK: BASELINE vs. ACCELERATED (C++20 + OpenMP)\n";
    std::cout << "======================================================================\n";

    LargeQPProblem prob;
    try {
        prob = load_binary_dataset("Data/cvxqp1_l.bin");
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    int max_threads = omp_get_max_threads();
    std::cout << "System OpenMP Threads Available: " << max_threads << "\n";
    std::cout << "Dataset: CVXQP1_L (n=" << prob.n << ", m=" << prob.m
              << ", nnz=" << (prob.P.nnz + prob.A.nnz) << ")\n\n";

    // Warmup
    run_hpr_qp_cpp(prob, 20, 32, false);

    for (int iters : {100, 300}) {
        std::cout << "--- Testing " << iters << " Iterations ---\n";
        auto res_base = run_hpr_qp_cpp(prob, iters, 32, false);
        std::cout << "  [Baseline Single-Core] Total: "
                  << std::setw(6) << std::fixed << std::setprecision(2) << res_base.elapsed_ms << " ms | "
                  << std::setprecision(3) << (res_base.elapsed_ms / iters) << " ms/iter | KKT: "
                  << std::scientific << std::setprecision(2) << res_base.kkt_residual << " | Obj: "
                  << std::fixed << std::setprecision(2) << res_base.objective << "\n";

        auto res_fast = run_hpr_qp_cpp(prob, iters, 32, true);
        double speedup = res_base.elapsed_ms / res_fast.elapsed_ms;
        std::cout << "  [Accelerated SIMD+OMP] Total: "
                  << std::setw(6) << std::fixed << std::setprecision(2) << res_fast.elapsed_ms << " ms | "
                  << std::setprecision(3) << (res_fast.elapsed_ms / iters) << " ms/iter | KKT: "
                  << std::scientific << std::setprecision(2) << res_fast.kkt_residual << " | Obj: "
                  << std::fixed << std::setprecision(2) << res_fast.objective << "\n";
        std::cout << "  >>> Speedup from Computational Optimizations: "
                  << std::setprecision(2) << speedup << "x faster!\n\n";
    }

    std::cout << "======================================================================\n";
    return 0;
}
