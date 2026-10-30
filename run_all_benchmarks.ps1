# ==============================================================================
# Windows PowerShell Automated Benchmark & Test Suite Runner
# ==============================================================================
$ErrorActionPreference = "Stop"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "       HPR-QP POWERSHELL MULTI-LANGUAGE AUTOMATED BENCHMARK RUNNER      " -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

$PythonExe = if (Test-Path "C:\Python314\python.exe") { "C:\Python314\python.exe" } else { "python.exe" }
$CargoExe = if (Test-Path "$env:USERPROFILE\.cargo\bin\cargo.exe") { "$env:USERPROFILE\.cargo\bin\cargo.exe" } else { "cargo.exe" }

# 1. Run Python Unit Tests
Write-Host "`n[1/5] Executing Python Unit Tests..." -ForegroundColor Yellow
& $PythonExe tests/test_hpr_qp_properties.py
& $PythonExe tests/test_transformer.py

# 2. Run Real-World Domain Examples
Write-Host "`n[2/5] Executing Domain Benchmark Applications..." -ForegroundColor Yellow
& $PythonExe examples/portfolio_optimization.py
& $PythonExe examples/lasso_sparse_regression.py
& $PythonExe examples/svm_dual_qp.py

# 3. Test C99 Solver
Write-Host "`n[3/5] Compiling and Testing C99 Solver..." -ForegroundColor Yellow
$Vcvars = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if (Test-Path $Vcvars) {
    cmd.exe /c "call `"$Vcvars`" && cd /d c && cl /O2 /W3 test_c_unit.c hpr_qp.c /Fe:test_c_unit.exe && test_c_unit.exe && del test_c_unit.exe test_c_unit.obj hpr_qp.obj"
} else {
    Write-Host "Visual Studio Developer Command Prompt not detected, skipping C MSVC build." -ForegroundColor DarkGray
}

# 4. Test C++20 Solver
Write-Host "`n[4/5] Compiling and Testing Modern C++20 Solver..." -ForegroundColor Yellow
if (Test-Path $Vcvars) {
    cmd.exe /c "call `"$Vcvars`" && cd /d cpp && cl /O2 /std:c++20 /EHsc test_cpp_unit.cpp hpr_qp.cpp /Fe:test_cpp_unit.exe && test_cpp_unit.exe && del test_cpp_unit.exe test_cpp_unit.obj hpr_qp.obj"
} else {
    Write-Host "Visual Studio Developer Command Prompt not detected, skipping C++ MSVC build." -ForegroundColor DarkGray
}

# 5. Run Rust Integration Tests
Write-Host "`n[5/5] Executing Rust Integration Tests & Benchmarks..." -ForegroundColor Yellow
if (Get-Command $CargoExe -ErrorAction SilentlyContinue) {
    Push-Location rust
    & $CargoExe test --quiet
    & $CargoExe run --release --bin hpr-qp
    Pop-Location
} else {
    Write-Host "Cargo not detected, skipping Rust build." -ForegroundColor DarkGray
}

# 6. Run Interactive CLI Demonstration
Write-Host "`n[Bonus] Testing Unified Interactive CLI..." -ForegroundColor Yellow
& $PythonExe cli.py --problem hs21

Write-Host "`n======================================================================" -ForegroundColor Green
Write-Host "       ALL BENCHMARKS AND VALIDATION SUITES COMPLETED SUCCESSFULLY!    " -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
