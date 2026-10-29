#include "hpr_qp.hpp"
#include <iomanip>
#include <numeric>
#include <algorithm>

namespace hpr {

inline double clamp(double val, double lb, double ub) {
    return std::max(lb, std::min(val, ub));
}

void HPRQPSolver::mat_vec_mul(const std::vector<double>& M, const double* x, double* y, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        double sum = 0.0;
        const double* row_ptr = &M[i * cols];
        for (int j = 0; j < cols; ++j) {
            sum += row_ptr[j] * x[j];
        }
        y[i] = sum;
    }
}

void HPRQPSolver::mat_t_vec_mul(const std::vector<double>& M, const double* x, double* y, int rows, int cols) {
    std::fill(y, y + cols, 0.0);
    for (int i = 0; i < rows; ++i) {
        double xi = x[i];
        const double* row_ptr = &M[i * cols];
        for (int j = 0; j < cols; ++j) {
            y[j] += row_ptr[j] * xi;
        }
    }
}

double HPRQPSolver::estimate_lambda_A(const std::vector<double>& A, int m, int n, int max_iter) {
    if (m <= 0 || n <= 0 || A.empty()) return 1.0;
    std::vector<double> v(n, 1.0 / std::sqrt(static_cast<double>(n)));
    std::vector<double> Av(m, 0.0);
    std::vector<double> w(n, 0.0);

    for (int it = 0; it < max_iter; ++it) {
        mat_vec_mul(A, v.data(), Av.data(), m, n);
        mat_t_vec_mul(A, Av.data(), w.data(), m, n);
        double norm_w = 0.0;
        for (double val : w) norm_w += val * val;
        norm_w = std::sqrt(norm_w);
        if (norm_w < 1e-12) return 1e-6;
        for (int j = 0; j < n; ++j) v[j] = w[j] / norm_w;
    }

    mat_vec_mul(A, v.data(), Av.data(), m, n);
    double sum_sq = 0.0;
    for (double val : Av) sum_sq += val * val;
    return std::max(sum_sq, 1e-6);
}

double HPRQPSolver::estimate_lambda_Q(const std::vector<double>& Q, int n, int max_iter) {
    if (n <= 0 || Q.empty()) return 1.0;
    std::vector<double> v(n, 1.0 / std::sqrt(static_cast<double>(n)));
    std::vector<double> w(n, 0.0);

    for (int it = 0; it < max_iter; ++it) {
        mat_vec_mul(Q, v.data(), w.data(), n, n);
        double norm_w = 0.0;
        for (double val : w) norm_w += val * val;
        norm_w = std::sqrt(norm_w);
        if (norm_w < 1e-12) break;
        for (int j = 0; j < n; ++j) v[j] = w[j] / norm_w;
    }

    mat_vec_mul(Q, v.data(), w.data(), n, n);
    double lambda = 0.0;
    for (int j = 0; j < n; ++j) lambda += v[j] * w[j];
    return std::max(lambda, 1e-6);
}

SolverResult HPRQPSolver::solve(const QPProblem& prob) {
    auto start_time = std::chrono::high_resolution_clock::now();

    const int n = prob.n;
    const int m = prob.m;
    const bool has_A = (m > 0 && !prob.A.empty());
    const bool has_Q = (!prob.Q.empty());

    double lambda_A = has_A ? estimate_lambda_A(prob.A, m, n) : 1.0;
    double lambda_Q = has_Q ? estimate_lambda_Q(prob.Q, n) : 1.0;

    double sigma = (config_.sigma > 0.0) ? config_.sigma : 1.0;
    const double denom_w = 1.0 + sigma * lambda_Q;

    // Allocate all buffers contiguously once
    std::vector<double> x(n, 0.0), w_Q(n, 0.0), z(n, 0.0), y(std::max(m, 1), 0.0);
    std::vector<double> x_bar(n, 0.0), z_bar(n, 0.0), w_half(n, 0.0), w_bar(n, 0.0);
    std::vector<double> y_bar(std::max(m, 1), 0.0);
    std::vector<double> x_hat(n, 0.0), r_z(n, 0.0), r_y(std::max(m, 1), 0.0);
    std::vector<double> Qw(n, 0.0), Qw_half(n, 0.0), ATy(n, 0.0), temp_n(n, 0.0);

    // Anchor points for Halpern acceleration & adaptive restart
    std::vector<double> x_r0(n, 0.0), w_r0(n, 0.0), z_r0(n, 0.0), y_r0(std::max(m, 1), 0.0);

    int inner_t = 0;
    int k = 0;
    double kkt_residual = 1e9;
    bool converged = false;

    for (k = 0; k < config_.max_iter; ++k) {
        inner_t++;

        // 1) Q * w_Q and A^T * y
        if (has_Q) mat_vec_mul(prob.Q, w_Q.data(), Qw.data(), n, n);
        if (has_A) mat_t_vec_mul(prob.A, y.data(), ATy.data(), m, n);

        // 2) Step 1 & 2: r_z and projection x_bar
        for (int j = 0; j < n; ++j) {
            double c_val = (!prob.c.empty()) ? prob.c[j] : 0.0;
            r_z[j] = x[j] + sigma * (-Qw[j] + ATy[j] - c_val);

            double lb = (!prob.l_c.empty()) ? prob.l_c[j] : -INF_BOUND;
            double ub = (!prob.u_c.empty()) ? prob.u_c[j] : INF_BOUND;
            x_bar[j] = clamp(r_z[j], lb, ub);
            z_bar[j] = (1.0 / sigma) * (x_bar[j] - r_z[j]);

            x_hat[j] = 2.0 * x_bar[j] - x[j];
            w_half[j] = (1.0 / denom_w) * (sigma * lambda_Q * w_Q[j] + x_hat[j]);
        }

        // 3) Step 3: y_bar and w_bar
        if (has_A) {
            if (has_Q) mat_vec_mul(prob.Q, w_half.data(), Qw_half.data(), n, n);

            for (int j = 0; j < n; ++j) {
                temp_n[j] = x_hat[j] + sigma * (Qw[j] - Qw_half[j]);
            }
            mat_vec_mul(prob.A, temp_n.data(), r_y.data(), m, n);

            const double sig_lam_A = sigma * lambda_A;
            for (int i = 0; i < m; ++i) {
                r_y[i] -= sig_lam_A * y[i];
                double lb = (!prob.l_k.empty()) ? prob.l_k[i] : -INF_BOUND;
                double ub = (!prob.u_k.empty()) ? prob.u_k[i] : INF_BOUND;
                double proj_ry = clamp(r_y[i], lb, ub);
                y_bar[i] = (1.0 / sig_lam_A) * (proj_ry - r_y[i]);
            }

            for (int i = 0; i < m; ++i) r_y[i] = y_bar[i] - y[i];
            mat_t_vec_mul(prob.A, r_y.data(), temp_n.data(), m, n);
            const double w_step = sigma / denom_w;
            for (int j = 0; j < n; ++j) {
                w_bar[j] = w_half[j] + w_step * temp_n[j];
            }
        } else {
            w_bar = w_half;
        }

        // 4) Halpern Averaging
        const double coef_0 = 1.0 / static_cast<double>(inner_t + 1);
        const double coef_hat = static_cast<double>(inner_t) / static_cast<double>(inner_t + 1);

        for (int j = 0; j < n; ++j) {
            x[j] = coef_0 * x_r0[j] + coef_hat * (2.0 * x_bar[j] - x[j]);
            w_Q[j] = coef_0 * w_r0[j] + coef_hat * (2.0 * w_bar[j] - w_Q[j]);
            z[j] = coef_0 * z_r0[j] + coef_hat * (2.0 * z_bar[j] - z[j]);
        }
        for (int i = 0; i < m; ++i) {
            y[i] = coef_0 * y_r0[i] + coef_hat * (2.0 * y_bar[i] - y[i]);
        }

        // 5) Adaptive Restart
        if (config_.restart_freq > 0 && inner_t >= config_.restart_freq) {
            x_r0 = x_bar;
            w_r0 = w_bar;
            z_r0 = z_bar;
            y_r0 = y_bar;
            x = x_r0;
            w_Q = w_r0;
            z = z_r0;
            y = y_r0;
            inner_t = 0;
        }

        // 6) Periodic Residual Check
        if ((k + 1) % 20 == 0 || k == config_.max_iter - 1) {
            double max_kkt = 0.0;

            if (has_A) {
                mat_vec_mul(prob.A, x_bar.data(), r_y.data(), m, n);
                for (int i = 0; i < m; ++i) {
                    double lb = (!prob.l_k.empty()) ? prob.l_k[i] : -INF_BOUND;
                    double ub = (!prob.u_k.empty()) ? prob.u_k[i] : INF_BOUND;
                    max_kkt = std::max(max_kkt, std::abs(r_y[i] - clamp(r_y[i], lb, ub)));
                }
            }

            if (has_Q) mat_vec_mul(prob.Q, w_bar.data(), Qw.data(), n, n);
            if (has_A) mat_t_vec_mul(prob.A, y_bar.data(), ATy.data(), m, n);

            for (int j = 0; j < n; ++j) {
                double c_val = (!prob.c.empty()) ? prob.c[j] : 0.0;
                max_kkt = std::max(max_kkt, std::abs(-Qw[j] + ATy[j] + z_bar[j] - c_val));
                double lb = (!prob.l_c.empty()) ? prob.l_c[j] : -INF_BOUND;
                double ub = (!prob.u_c.empty()) ? prob.u_c[j] : INF_BOUND;
                max_kkt = std::max(max_kkt, std::abs(x_bar[j] - clamp(x_bar[j], lb, ub)));
            }

            kkt_residual = max_kkt;
            if (kkt_residual <= config_.tol) {
                converged = true;
                break;
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double runtime_sec = std::chrono::duration<double>(end_time - start_time).count();

    // Objective value: 0.5 * x^T Q x + c^T x
    double obj = 0.0;
    if (has_Q) {
        mat_vec_mul(prob.Q, x_bar.data(), temp_n.data(), n, n);
        for (int j = 0; j < n; ++j) obj += 0.5 * x_bar[j] * temp_n[j];
    }
    if (!prob.c.empty()) {
        for (int j = 0; j < n; ++j) obj += prob.c[j] * x_bar[j];
    }

    SolverResult res;
    res.x = std::move(x_bar);
    res.w = std::move(w_bar);
    res.y = std::move(y_bar);
    res.z = std::move(z_bar);
    res.iterations = converged ? (k + 1) : config_.max_iter;
    res.runtime_sec = runtime_sec;
    res.kkt_residual = kkt_residual;
    res.objective = obj;
    res.converged = converged;

    return res;
}

} // namespace hpr

#ifdef HPR_QP_CPP_STANDALONE
int main() {
    std::cout << "===============================================================\n";
    std::cout << "  HPR-QP SOLVER IN C++ (Modern C++20)\n";
    std::cout << "===============================================================\n";

    // Example 5.1 (Singular Q)
    {
        hpr::QPProblem prob;
        prob.n = 3;
        prob.m = 0;
        prob.Q = {1.0, 0.0, 0.0,
                  0.0, 2.0, 0.0,
                  0.0, 0.0, 0.0};
        prob.c = {-1.0, -1.0, -1.0};
        prob.u_c = {1.0, 1.0, 1.0};

        hpr::SolverConfig cfg;
        cfg.tol = 1e-8;
        cfg.restart_freq = 8;

        hpr::HPRQPSolver solver(cfg);
        auto res = solver.solve(prob);

        std::cout << "Example 5.1 (Singular Q):\n";
        std::cout << "  Status: " << (res.converged ? "CONVERGED" : "MAX_ITER")
                  << " | Iters: " << res.iterations
                  << " | Time: " << std::fixed << std::setprecision(6) << res.runtime_sec * 1000.0 << " ms"
                  << " | KKT Res: " << std::scientific << res.kkt_residual
                  << " | Obj: " << std::fixed << res.objective << "\n";
        std::cout << "  Optimal x: [" << res.x[0] << ", " << res.x[1] << ", " << res.x[2] << "]\n\n";
    }

    // Example 5.2 (Ill-conditioned Q)
    {
        hpr::QPProblem prob;
        prob.n = 2;
        prob.m = 0;
        prob.Q = {1.0, 0.1,
                  0.1, 10.0};
        prob.c = {-1.0, -1.0};

        hpr::SolverConfig cfg;
        cfg.tol = 1e-8;
        cfg.restart_freq = 8;

        hpr::HPRQPSolver solver(cfg);
        auto res = solver.solve(prob);

        std::cout << "Example 5.2 (Ill-conditioned Q):\n";
        std::cout << "  Status: " << (res.converged ? "CONVERGED" : "MAX_ITER")
                  << " | Iters: " << res.iterations
                  << " | Time: " << std::fixed << std::setprecision(6) << res.runtime_sec * 1000.0 << " ms"
                  << " | KKT Res: " << std::scientific << res.kkt_residual
                  << " | Obj: " << std::fixed << res.objective << "\n";
        std::cout << "  Optimal x: [" << res.x[0] << ", " << res.x[1] << "]\n\n";
    }

    // HS21
    {
        hpr::QPProblem prob;
        prob.n = 2;
        prob.m = 1;
        prob.Q = {0.02, 0.0,
                  0.0,  2.0};
        prob.c = {0.0, 0.0};
        prob.A = {10.0, -1.0};
        prob.l_k = {10.0};
        prob.l_c = {2.0, -hpr::INF_BOUND};
        prob.u_c = {50.0, 50.0};

        hpr::SolverConfig cfg;
        cfg.tol = 1e-6;
        cfg.restart_freq = 16;

        hpr::HPRQPSolver solver(cfg);
        auto res = solver.solve(prob);

        std::cout << "Hock-Schittkowski #21 (HS21):\n";
        std::cout << "  Status: " << (res.converged ? "CONVERGED" : "MAX_ITER")
                  << " | Iters: " << res.iterations
                  << " | Time: " << std::fixed << std::setprecision(6) << res.runtime_sec * 1000.0 << " ms"
                  << " | KKT Res: " << std::scientific << res.kkt_residual
                  << " | Obj: " << std::fixed << res.objective << "\n";
        std::cout << "  Optimal x: [" << res.x[0] << ", " << res.x[1] << "]\n";
    }

    std::cout << "===============================================================\n";
    std::cout << "  ALL C++ BENCHMARKS COMPLETED SUCCESSFULLY!\n";
    std::cout << "===============================================================\n";
    return 0;
}
#endif
