# Contributing to HPR-QP Solver Suite

Thank you for your interest in contributing to the **Accelerated First-Order Methods for Convex Composite Quadratic Programming (HPR-QP)** project!

This guide outlines our development workflow, coding standards, and procedures for submitting improvements.

---

## 1. Development Environment Setup

### Prerequisites
- **Python**: 3.10+ with `numpy`, `torch`, `matplotlib`
- **C/C++ Compiler**: GCC 11+, Clang 14+, or MSVC 2022+ with C++20 and OpenMP support
- **Rust**: 1.70+ with `cargo`
- **OpenCL SDK**: Any vendor OpenCL 1.2+ runtime (NVIDIA CUDA Toolkit, AMD ROCm, Intel oneAPI)
- **Git**

### Installation
Clone the repository:
```bash
git clone https://github.com/monourhosan/ACCELERATED-FIRST-ORDER-METHODS-FOR-CONVEX-COMPOSITE-QUADRATIC-PROGRAMMING.git
cd ACCELERATED-FIRST-ORDER-METHODS-FOR-CONVEX-COMPOSITE-QUADRATIC-PROGRAMMING
```

Install Python dependencies:
```bash
pip install -e ".[dev]"
```

---

## 2. Running Test Suites

Before opening a pull request, verify that all test suites pass across languages:

### Python Unit Tests
```bash
pytest tests/ -v
# Or run standalone:
python tests/test_hpr_qp_properties.py
python tests/test_transformer.py
```

### C99 Tests
```bash
cd c
# With GCC/Clang:
make test
# With MSVC:
cl /O2 /W3 test_c_unit.c hpr_qp.c && ./test_c_unit.exe
```

### Modern C++20 Tests
```bash
cd cpp
# With MSVC:
cl /O2 /std:c++20 /EHsc test_cpp_unit.cpp hpr_qp.cpp && ./test_cpp_unit.exe
```

### Rust Integration Tests
```bash
cd rust
cargo test --all-targets
```

---

## 3. Code Style and Quality Standards

### Python
- Follow PEP 8 style guidelines.
- Use explicit type annotations (PEP 484 / PEP 585).
- Format code using `black` and check with `flake8`.

### C & C++
- Follow modern idioms: RAII, `const`-correctness, no uninitialized pointers.
- C code must conform to strict ISO C99.
- C++ code must conform to ISO C++20 (`std::vector`, `std::unique_ptr`, concepts where appropriate).
- Format with `clang-format` (LLVM or Google style).

### Rust
- Enforce strict `clippy` checks:
  ```bash
  cargo clippy --all-targets -- -D warnings
  ```
- Format with `cargo fmt`.

---

## 4. Conventional Commits

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:
- `feat(scope)`: A new feature or solver implementation
- `fix(scope)`: A bug fix
- `test(scope)`: Adding or refactoring unit/integration tests
- `docs(scope)`: Documentation, derivations, or README improvements
- `refactor(scope)`: Code refactoring without behavioral changes
- `chore(scope)`: Build tools, CI/CD, or maintenance

---

## 5. Submitting Pull Requests

1. Fork the repository and create your branch from `main`:
   ```bash
   git checkout -b feat/my-new-feature
   ```
2. Commit your changes with informative messages.
3. Push to your fork:
   ```bash
   git push origin feat/my-new-feature
   ```
4. Open a Pull Request against `main`. Ensure all CI checks pass.
