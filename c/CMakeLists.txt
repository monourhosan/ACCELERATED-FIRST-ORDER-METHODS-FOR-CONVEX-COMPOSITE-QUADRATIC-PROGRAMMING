cmake_minimum_required(VERSION 3.16)
project(hpr_qp_c LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Optimization flags
if(MSVC)
    add_compile_options(/O2 /fp:fast /W3)
else()
    add_compile_options(-O3 -march=native -Wall -Wextra)
endif()

find_package(OpenMP)
find_package(OpenCL)

# Core HPR-QP static library
add_library(hpr_qp STATIC
    hpr_qp.c
    hpr_qp.h
)
target_include_directories(hpr_qp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Fast OpenMP shared library
add_library(hpr_qp_fast SHARED
    benchmark_fast_c.c
)
if(OpenMP_C_FOUND)
    target_link_libraries(hpr_qp_fast PRIVATE OpenMP::OpenMP_C)
endif()

# CPU Benchmark Executable
add_executable(benchmark_cvxqp1_l
    benchmark_cvxqp1_l.c
)
target_link_libraries(benchmark_cvxqp1_l PRIVATE hpr_qp)

# GPU Benchmark Executable (if OpenCL is available)
if(OpenCL_FOUND)
    add_executable(benchmark_gpu_c
        benchmark_gpu_c.c
    )
    target_link_libraries(benchmark_gpu_c PRIVATE OpenCL::OpenCL)
endif()
