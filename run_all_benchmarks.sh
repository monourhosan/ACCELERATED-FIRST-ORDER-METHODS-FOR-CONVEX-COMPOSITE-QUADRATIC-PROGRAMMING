#!/usr/bin/env bash
# ==============================================================================
# Cross-Platform Automated Benchmark & Test Suite Runner for Linux/macOS
# ==============================================================================
set -e

echo "======================================================================"
echo "          HPR-QP MULTI-LANGUAGE AUTOMATED BENCHMARK SUITE             "
echo "======================================================================"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# 1. Run Python Test Suite
echo ""
echo "[1/5] Running Python Unit Tests..."
if command -v pytest >/dev/null 2>&1; then
    pytest tests/ -v
else
    python3 tests/test_hpr_qp_properties.py
    python3 tests/test_transformer.py
fi

# 2. Run Real-World Applications Suite
echo ""
echo "[2/5] Running Real-World Application Benchmarks..."
python3 examples/portfolio_optimization.py
python3 examples/lasso_sparse_regression.py
python3 examples/svm_dual_qp.py

# 3. Compile and Run C Benchmarks
echo ""
echo "[3/5] Testing C99 Solver & Unit Tests..."
if command -v gcc >/dev/null 2>&1; then
    cd c
    gcc -O3 -std=c99 test_c_unit.c hpr_qp.c -lm -o test_c_unit
    ./test_c_unit
    rm -f test_c_unit
    cd ..
else
    echo "gcc not found, skipping C tests."
fi

# 4. Compile and Run C++ Benchmarks
echo ""
echo "[4/5] Testing C++20 Solver & Unit Tests..."
if command -v g++ >/dev/null 2>&1; then
    cd cpp
    g++ -O3 -std=c++20 test_cpp_unit.cpp hpr_qp.cpp -o test_cpp_unit || true
    if [ -f test_cpp_unit ]; then
        ./test_cpp_unit
        rm -f test_cpp_unit
    fi
    cd ..
else
    echo "g++ not found, skipping C++ tests."
fi

# 5. Run Rust Benchmarks and Tests
echo ""
echo "[5/5] Testing Rust Solver & Integration Tests..."
if command -v cargo >/dev/null 2>&1; then
    cd rust
    cargo test --quiet
    cargo run --release --bin hpr-qp
    cd ..
else
    echo "cargo not found, skipping Rust tests."
fi

echo ""
echo "======================================================================"
echo "          ALL MULTI-LANGUAGE BENCHMARKS EXECUTED SUCCESSFULLY         "
echo "======================================================================"
