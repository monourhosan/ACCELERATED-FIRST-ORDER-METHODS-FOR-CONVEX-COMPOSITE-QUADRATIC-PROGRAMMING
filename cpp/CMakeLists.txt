cmake_minimum_required(VERSION 3.16)
project(hpr_qp_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags
if(MSVC)
    add_compile_options(/O2 /std:c++20 /fp:fast /W3)
else()
    add_compile_options(-O3 -std=c++20 -march=native -Wall -Wextra)
endif()

find_package(OpenMP)
find_package(OpenCL)

# Core HPR-QP C++ static library
add_library(hpr_qp_cpp STATIC
    hpr_qp.cpp
    hpr_qp.hpp
)
target_include_directories(hpr_qp_cpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# C++ Benchmark Executable
add_executable(benchmark_cvxqp1_l_cpp
    benchmark_cvxqp1_l.cpp
)
target_link_libraries(benchmark_cvxqp1_l_cpp PRIVATE hpr_qp_cpp)

# Fast C++ Benchmark with OpenMP
add_executable(benchmark_fast_cpp
    benchmark_fast_cpp.cpp
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(benchmark_fast_cpp PRIVATE OpenMP::OpenMP_CXX)
endif()

# GPU C++ Benchmark (if OpenCL is available)
if(OpenCL_FOUND)
    add_executable(benchmark_gpu_cpp
        benchmark_gpu_cpp.cpp
    )
    target_link_libraries(benchmark_gpu_cpp PRIVATE OpenCL::OpenCL)
endif()
