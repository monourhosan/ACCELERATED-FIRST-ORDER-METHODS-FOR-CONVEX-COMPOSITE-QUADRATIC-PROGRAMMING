#pragma once

#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <limits>
#include <iostream>

namespace hpr {

constexpr double INF_BOUND = 1e20;

struct QPProblem {
    int n{0};                        // Number of variables
    int m{0};                        // Number of linear constraints
    std::vector<double> Q;           // Symmetric PSD matrix (n x n, row-major)
    std::vector<double> c;           // Linear cost vector (n)
    std::vector<double> A;           // Linear constraints matrix (m x n, row-major)
    std::vector<double> l_k;         // Lower bounds on Ax (m)
    std::vector<double> u_k;         // Upper bounds on Ax (m)
    std::vector<double> l_c;         // Lower bounds on x (n)
    std::vector<double> u_c;         // Upper bounds on x (n)
};

struct SolverConfig {
    int max_iter{3000};
    double tol{1e-6};
    double sigma{1.0};
    int restart_freq{32};
};

struct SolverResult {
    std::vector<double> x;
    std::vector<double> w;
    std::vector<double> y;
    std::vector<double> z;
    int iterations{0};
    double runtime_sec{0.0};
    double kkt_residual{1e9};
    double objective{0.0};
    bool converged{false};
};

class HPRQPSolver {
public:
    explicit HPRQPSolver(SolverConfig config = SolverConfig{}) : config_(config) {}

    SolverResult solve(const QPProblem& prob);

private:
    SolverConfig config_;

    static double estimate_lambda_A(const std::vector<double>& A, int m, int n, int max_iter = 40);
    static double estimate_lambda_Q(const std::vector<double>& Q, int n, int max_iter = 40);
    static void mat_vec_mul(const std::vector<double>& M, const double* x, double* y, int rows, int cols);
    static void mat_t_vec_mul(const std::vector<double>& M, const double* x, double* y, int rows, int cols);
};

} // namespace hpr
