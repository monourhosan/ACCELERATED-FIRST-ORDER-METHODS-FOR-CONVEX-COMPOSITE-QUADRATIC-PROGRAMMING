#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <cstdint>
#include <windows.h>

// OpenCL API Types and Constants
typedef int cl_int;
typedef unsigned int cl_uint;
typedef unsigned long long cl_ulong;
typedef void* cl_platform_id;
typedef void* cl_device_id;
typedef void* cl_context;
typedef void* cl_command_queue;
typedef void* cl_mem;
typedef void* cl_program;
typedef void* cl_kernel;

#define CL_SUCCESS                                  0
#define CL_TRUE                                     1
#define CL_FALSE                                    0
#define CL_PLATFORM_NAME                            0x0903
#define CL_DEVICE_NAME                              0x102B
#define CL_DEVICE_TYPE_GPU                          (1 << 2)
#define CL_MEM_READ_ONLY                            (1 << 2)
#define CL_MEM_WRITE_ONLY                           (1 << 1)
#define CL_MEM_READ_WRITE                           (1 << 0)
#define CL_MEM_COPY_HOST_PTR                        (1 << 5)
#define CL_PROGRAM_BUILD_LOG                        0x1183

// Function Pointer Signatures
typedef cl_int (__stdcall *pfn_clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int (__stdcall *pfn_clGetPlatformInfo)(cl_platform_id, cl_uint, size_t, void*, size_t*);
typedef cl_int (__stdcall *pfn_clGetDeviceIDs)(cl_platform_id, cl_ulong, cl_uint, cl_device_id*, cl_uint*);
typedef cl_int (__stdcall *pfn_clGetDeviceInfo)(cl_device_id, cl_uint, size_t, void*, size_t*);
typedef cl_context (__stdcall *pfn_clCreateContext)(const void*, cl_uint, const cl_device_id*, void*, void*, cl_int*);
typedef cl_command_queue (__stdcall *pfn_clCreateCommandQueue)(cl_context, cl_device_id, cl_ulong, cl_int*);
typedef cl_mem (__stdcall *pfn_clCreateBuffer)(cl_context, cl_ulong, size_t, void*, cl_int*);
typedef cl_program (__stdcall *pfn_clCreateProgramWithSource)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
typedef cl_int (__stdcall *pfn_clBuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*, void*, void*);
typedef cl_int (__stdcall *pfn_clGetProgramBuildInfo)(cl_program, cl_device_id, cl_uint, size_t, void*, size_t*);
typedef cl_kernel (__stdcall *pfn_clCreateKernel)(cl_program, const char*, cl_int*);
typedef cl_int (__stdcall *pfn_clSetKernelArg)(cl_kernel, cl_uint, size_t, const void*);
typedef cl_int (__stdcall *pfn_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const void*, void*);
typedef cl_int (__stdcall *pfn_clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_uint, size_t, size_t, void*, cl_uint, const void*, void*);
typedef cl_int (__stdcall *pfn_clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_uint, size_t, size_t, const void*, cl_uint, const void*, void*);
typedef cl_int (__stdcall *pfn_clFinish)(cl_command_queue);
typedef cl_int (__stdcall *pfn_clReleaseMemObject)(cl_mem);
typedef cl_int (__stdcall *pfn_clReleaseKernel)(cl_kernel);
typedef cl_int (__stdcall *pfn_clReleaseProgram)(cl_program);
typedef cl_int (__stdcall *pfn_clReleaseCommandQueue)(cl_command_queue);
typedef cl_int (__stdcall *pfn_clReleaseContext)(cl_context);

// Global Function Pointers
static pfn_clGetPlatformIDs clGetPlatformIDs = nullptr;
static pfn_clGetPlatformInfo clGetPlatformInfo = nullptr;
static pfn_clGetDeviceIDs clGetDeviceIDs = nullptr;
static pfn_clGetDeviceInfo clGetDeviceInfo = nullptr;
static pfn_clCreateContext clCreateContext = nullptr;
static pfn_clCreateCommandQueue clCreateCommandQueue = nullptr;
static pfn_clCreateBuffer clCreateBuffer = nullptr;
static pfn_clCreateProgramWithSource clCreateProgramWithSource = nullptr;
static pfn_clBuildProgram clBuildProgram = nullptr;
static pfn_clGetProgramBuildInfo clGetProgramBuildInfo = nullptr;
static pfn_clCreateKernel clCreateKernel = nullptr;
static pfn_clSetKernelArg clSetKernelArg = nullptr;
static pfn_clEnqueueNDRangeKernel clEnqueueNDRangeKernel = nullptr;
static pfn_clEnqueueReadBuffer clEnqueueReadBuffer = nullptr;
static pfn_clEnqueueWriteBuffer clEnqueueWriteBuffer = nullptr;
static pfn_clFinish clFinish = nullptr;
static pfn_clReleaseMemObject clReleaseMemObject = nullptr;
static pfn_clReleaseKernel clReleaseKernel = nullptr;
static pfn_clReleaseProgram clReleaseProgram = nullptr;
static pfn_clReleaseCommandQueue clReleaseCommandQueue = nullptr;
static pfn_clReleaseContext clReleaseContext = nullptr;

// OpenCL Dynamic Library Loader
class OpenCLLoader {
    HMODULE hLib_{nullptr};
public:
    OpenCLLoader() {
        hLib_ = LoadLibraryA("OpenCL.dll");
        if (!hLib_) {
            throw std::runtime_error("Failed to load OpenCL.dll from system.");
        }
        #define LOAD_PROC(fn) fn = reinterpret_cast<pfn_##fn>(GetProcAddress(hLib_, #fn)); \
            if (!fn) throw std::runtime_error("Failed to load OpenCL symbol: " #fn);

        LOAD_PROC(clGetPlatformIDs);
        LOAD_PROC(clGetPlatformInfo);
        LOAD_PROC(clGetDeviceIDs);
        LOAD_PROC(clGetDeviceInfo);
        LOAD_PROC(clCreateContext);
        LOAD_PROC(clCreateCommandQueue);
        LOAD_PROC(clCreateBuffer);
        LOAD_PROC(clCreateProgramWithSource);
        LOAD_PROC(clBuildProgram);
        LOAD_PROC(clGetProgramBuildInfo);
        LOAD_PROC(clCreateKernel);
        LOAD_PROC(clSetKernelArg);
        LOAD_PROC(clEnqueueNDRangeKernel);
        LOAD_PROC(clEnqueueReadBuffer);
        LOAD_PROC(clEnqueueWriteBuffer);
        LOAD_PROC(clFinish);
        LOAD_PROC(clReleaseMemObject);
        LOAD_PROC(clReleaseKernel);
        LOAD_PROC(clReleaseProgram);
        LOAD_PROC(clReleaseCommandQueue);
        LOAD_PROC(clReleaseContext);
        #undef LOAD_PROC
    }
    ~OpenCLLoader() {
        if (hLib_) FreeLibrary(hLib_);
    }
};

// RAII Wrapper for OpenCL Device & Context
class OpenCLContext {
public:
    cl_platform_id platform{nullptr};
    cl_device_id device{nullptr};
    cl_context context{nullptr};
    cl_command_queue queue{nullptr};
    std::string device_name;

    OpenCLContext() {
        cl_uint num_platforms = 0;
        cl_int err = clGetPlatformIDs(0, nullptr, &num_platforms);
        if (err != CL_SUCCESS || num_platforms == 0) {
            throw std::runtime_error("No OpenCL platforms found.");
        }

        std::vector<cl_platform_id> platforms(num_platforms);
        clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

        bool found = false;
        // Prefer NVIDIA GPU
        for (cl_uint p = 0; p < num_platforms; ++p) {
            cl_uint num_devices = 0;
            if (clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices) == CL_SUCCESS && num_devices > 0) {
                std::vector<cl_device_id> devices(num_devices);
                clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);
                for (cl_uint d = 0; d < num_devices; ++d) {
                    char name_buf[256] = {0};
                    clGetDeviceInfo(devices[d], CL_DEVICE_NAME, sizeof(name_buf), name_buf, nullptr);
                    std::string dname(name_buf);
                    if (dname.find("NVIDIA") != std::string::npos || dname.find("GeForce") != std::string::npos) {
                        platform = platforms[p];
                        device = devices[d];
                        device_name = dname;
                        found = true;
                        break;
                    }
                }
            }
            if (found) break;
        }

        // Fallback to first GPU device if NVIDIA wasn't explicitly matched
        if (!found) {
            for (cl_uint p = 0; p < num_platforms && !found; ++p) {
                cl_uint num_devices = 0;
                if (clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices) == CL_SUCCESS && num_devices > 0) {
                    platform = platforms[p];
                    clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
                    char name_buf[256] = {0};
                    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name_buf), name_buf, nullptr);
                    device_name = std::string(name_buf);
                    found = true;
                }
            }
        }

        if (!found) {
            throw std::runtime_error("No suitable OpenCL GPU device found.");
        }

        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("clCreateContext failed: " + std::to_string(err));

        queue = clCreateCommandQueue(context, device, 0, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("clCreateCommandQueue failed: " + std::to_string(err));
    }

    ~OpenCLContext() {
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }
};

// RAII Wrapper for OpenCL VRAM Buffer
class GPUBuffer {
public:
    cl_mem mem{nullptr};
    size_t size_bytes{0};

    GPUBuffer() = default;
    GPUBuffer(cl_context ctx, cl_ulong flags, size_t size, void* host_ptr = nullptr)
        : size_bytes(size) {
        cl_int err = 0;
        mem = clCreateBuffer(ctx, flags, size, host_ptr, &err);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("clCreateBuffer failed with error: " + std::to_string(err));
        }
    }

    ~GPUBuffer() {
        if (mem) {
            clReleaseMemObject(mem);
            mem = nullptr;
        }
    }

    // Move semantics for RAII
    GPUBuffer(GPUBuffer&& other) noexcept : mem(other.mem), size_bytes(other.size_bytes) {
        other.mem = nullptr;
        other.size_bytes = 0;
    }

    GPUBuffer& operator=(GPUBuffer&& other) noexcept {
        if (this != &other) {
            if (mem) clReleaseMemObject(mem);
            mem = other.mem;
            size_bytes = other.size_bytes;
            other.mem = nullptr;
            other.size_bytes = 0;
        }
        return *this;
    }

    GPUBuffer(const GPUBuffer&) = delete;
    GPUBuffer& operator=(const GPUBuffer&) = delete;

    void write(cl_command_queue queue, const void* host_ptr, size_t bytes) {
        cl_int err = clEnqueueWriteBuffer(queue, mem, CL_TRUE, 0, bytes, host_ptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) throw std::runtime_error("clEnqueueWriteBuffer failed: " + std::to_string(err));
    }

    void read(cl_command_queue queue, void* host_ptr, size_t bytes) {
        cl_int err = clEnqueueReadBuffer(queue, mem, CL_TRUE, 0, bytes, host_ptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) throw std::runtime_error("clEnqueueReadBuffer failed: " + std::to_string(err));
    }
};

// OpenCL Kernels Source
static const char* GPU_OPENCL_SOURCE = R"(
__kernel void spmv_csr_kernel(
    const uint num_rows,
    __global const int* row_ptr,
    __global const int* col_idx,
    __global const double* val,
    __global const double* x,
    __global double* y)
{
    uint i = get_global_id(0);
    if (i < num_rows) {
        int start = row_ptr[i];
        int end = row_ptr[i + 1];
        double sum = 0.0;
        for (int p = start; p < end; ++p) {
            sum += val[p] * x[col_idx[p]];
        }
        y[i] = sum;
    }
}

__kernel void fused_step1_2_kernel(
    const uint n,
    const double sigma,
    const double denom_w,
    const double lam_q,
    __global const double* x,
    __global const double* qw,
    __global const double* aty,
    __global const double* c,
    __global const double* l_c,
    __global const double* u_c,
    __global const double* w_q,
    __global double* x_bar,
    __global double* z_bar,
    __global double* x_hat,
    __global double* w_half)
{
    uint j = get_global_id(0);
    if (j < n) {
        double rz = x[j] + sigma * (-qw[j] + aty[j] - c[j]);
        double xb = rz;
        if (xb < l_c[j]) xb = l_c[j];
        if (xb > u_c[j]) xb = u_c[j];
        x_bar[j] = xb;
        z_bar[j] = (1.0 / sigma) * (xb - rz);

        double xh = 2.0 * xb - x[j];
        x_hat[j] = xh;
        w_half[j] = (1.0 / denom_w) * (sigma * lam_q * w_q[j] + xh);
    }
}

__kernel void sgs_prep_temp_n_kernel(
    const uint n,
    const double sigma,
    __global const double* x_hat,
    __global const double* qw,
    __global const double* qw_half,
    __global double* temp_n)
{
    uint j = get_global_id(0);
    if (j < n) {
        temp_n[j] = x_hat[j] + sigma * (qw[j] - qw_half[j]);
    }
}

__kernel void sgs_y_step_kernel(
    const uint m,
    const double sig_lam_a,
    const double inv_sig_lam_a,
    __global const double* l_k,
    __global const double* u_k,
    __global const double* y,
    __global const double* r_y,
    __global double* y_bar,
    __global double* temp_m)
{
    uint i = get_global_id(0);
    if (i < m) {
        double ry = r_y[i] - sig_lam_a * y[i];
        double py = ry;
        if (py < l_k[i]) py = l_k[i];
        if (py > u_k[i]) py = u_k[i];
        double yb = inv_sig_lam_a * (py - ry);
        y_bar[i] = yb;
        temp_m[i] = yb - y[i];
    }
}

__kernel void sgs_w_step_kernel(
    const uint n,
    const double w_step,
    __global const double* w_half,
    __global const double* at_delta_y,
    __global double* w_bar)
{
    uint j = get_global_id(0);
    if (j < n) {
        w_bar[j] = w_half[j] + w_step * at_delta_y[j];
    }
}

__kernel void halpern_average_kernel(
    const uint n,
    const uint m,
    const double coef_0,
    const double coef_hat,
    __global const double* x_r0,
    __global const double* w_r0,
    __global const double* z_r0,
    __global const double* y_r0,
    __global const double* x_bar,
    __global const double* w_bar,
    __global const double* z_bar,
    __global const double* y_bar,
    __global double* x,
    __global double* w_q,
    __global double* z,
    __global double* y)
{
    uint j = get_global_id(0);
    if (j < n) {
        x[j] = coef_0 * x_r0[j] + coef_hat * (2.0 * x_bar[j] - x[j]);
        w_q[j] = coef_0 * w_r0[j] + coef_hat * (2.0 * w_bar[j] - w_q[j]);
        z[j] = coef_0 * z_r0[j] + coef_hat * (2.0 * z_bar[j] - z[j]);
    }
    if (j < m) {
        y[j] = coef_0 * y_r0[j] + coef_hat * (2.0 * y_bar[j] - y[j]);
    }
}
)";

// Structure for the CSR Problem in Host RAM
struct HostCSR {
    uint32_t rows{0}, cols{0};
    uint64_t nnz{0};
    std::vector<int32_t> row_ptr;
    std::vector<int32_t> col_idx;
    std::vector<double>  val;
};

struct HostProblem {
    uint32_t n{0}, m{0};
    HostCSR P, A, AT;
    std::vector<double> c;
    std::vector<double> l_k, u_k;
    std::vector<double> l_c, u_c;
};

static HostProblem load_binary_dataset(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Could not open binary dataset: " + path);

    HostProblem prob;
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
    prob.l_k.resize(prob.m); prob.u_k.resize(prob.m);
    prob.l_c.resize(prob.n); prob.u_c.resize(prob.n);
    f.read(reinterpret_cast<char*>(prob.c.data()), prob.n * sizeof(double));
    f.read(reinterpret_cast<char*>(prob.l_k.data()), prob.m * sizeof(double));
    f.read(reinterpret_cast<char*>(prob.u_k.data()), prob.m * sizeof(double));
    f.read(reinterpret_cast<char*>(prob.l_c.data()), prob.n * sizeof(double));
    f.read(reinterpret_cast<char*>(prob.u_c.data()), prob.n * sizeof(double));

    return prob;
}

int main() {
    std::cout << "=======================================================================\n";
    std::cout << "  GPU ACCELERATED HPR-QP IN MODERN C++20 (NVIDIA GeForce MX250 / OpenCL)\n";
    std::cout << "  Problem: CVXQP1_L (n = 10,000, m = 15,000, Nonzeros = 94,966)\n";
    std::cout << "=======================================================================\n";

    try {
        OpenCLLoader loader;
        OpenCLContext clCtx;
        std::cout << "Selected GPU Device   : " << clCtx.device_name << "\n";

        std::string dataset_path = "Data/cvxqp1_l.bin";
        HostProblem prob = load_binary_dataset(dataset_path);
        std::cout << "Dataset Loaded in RAM : n=" << prob.n << ", m=" << prob.m
                  << ", nnz=" << (prob.P.nnz + prob.A.nnz) << "\n";

        // JIT Compile OpenCL Kernels
        cl_int err = 0;
        cl_program prog = clCreateProgramWithSource(clCtx.context, 1, &GPU_OPENCL_SOURCE, nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("clCreateProgramWithSource failed: " + std::to_string(err));

        err = clBuildProgram(prog, 1, &clCtx.device, "-cl-fast-relaxed-math -cl-mad-enable", nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_size = 0;
            clGetProgramBuildInfo(prog, clCtx.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::vector<char> build_log(log_size + 1, 0);
            clGetProgramBuildInfo(prog, clCtx.device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), nullptr);
            std::cerr << "Build Log:\n" << build_log.data() << "\n";
            throw std::runtime_error("clBuildProgram failed.");
        }

        cl_kernel k_spmv     = clCreateKernel(prog, "spmv_csr_kernel", &err);
        cl_kernel k_step1_2  = clCreateKernel(prog, "fused_step1_2_kernel", &err);
        cl_kernel k_sgs_prep = clCreateKernel(prog, "sgs_prep_temp_n_kernel", &err);
        cl_kernel k_sgs_y    = clCreateKernel(prog, "sgs_y_step_kernel", &err);
        cl_kernel k_sgs_w    = clCreateKernel(prog, "sgs_w_step_kernel", &err);
        cl_kernel k_halpern  = clCreateKernel(prog, "halpern_average_kernel", &err);

        uint32_t n = prob.n;
        uint32_t m = prob.m;

        // Allocate Persistent GPU Buffers in VRAM (100% VRAM Resident)
        GPUBuffer d_P_row(clCtx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (n + 1) * sizeof(int32_t), prob.P.row_ptr.data());
        GPUBuffer d_P_col(clCtx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, prob.P.nnz * sizeof(int32_t), prob.P.col_idx.data());
        GPUBuffer d_P_val(clCtx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, prob.P.nnz * sizeof(double), prob.P.val.data());

        GPUBuffer d_A_row(clCtx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (m + 1) * sizeof(int32_t), prob.A.row_ptr.data());
        GPUBuffer d_A_col(clCtx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, prob.A.nnz * sizeof(int32_t), prob.A.col_idx.data());
        GPUBuffer d_A_val(clCtx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, prob.A.nnz * sizeof(double), prob.A.val.data());

        GPUBuffer d_AT_row(clCtx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (n + 1) * sizeof(int32_t), prob.AT.row_ptr.data());
        GPUBuffer d_AT_col(clCtx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, prob.AT.nnz * sizeof(int32_t), prob.AT.col_idx.data());
        GPUBuffer d_AT_val(clCtx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, prob.AT.nnz * sizeof(double), prob.AT.val.data());

        GPUBuffer d_c(clCtx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(double), prob.c.data());
        GPUBuffer d_l_k(clCtx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, m * sizeof(double), prob.l_k.data());
        GPUBuffer d_u_k(clCtx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, m * sizeof(double), prob.u_k.data());
        GPUBuffer d_l_c(clCtx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(double), prob.l_c.data());
        GPUBuffer d_u_c(clCtx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(double), prob.u_c.data());

        // Working Iterate Vectors
        GPUBuffer d_x(clCtx.context, CL_MEM_READ_WRITE, n * sizeof(double));
        GPUBuffer d_w_q(clCtx.context, CL_MEM_READ_WRITE, n * sizeof(double));
        GPUBuffer d_z(clCtx.context, CL_MEM_READ_WRITE, n * sizeof(double));
        GPUBuffer d_y(clCtx.context, CL_MEM_READ_WRITE, m * sizeof(double));

        GPUBuffer d_x_bar(clCtx.context, CL_MEM_READ_WRITE, n * sizeof(double));
        GPUBuffer d_w_bar(clCtx.context, CL_MEM_READ_WRITE, n * sizeof(double));
        GPUBuffer d_z_bar(clCtx.context, CL_MEM_READ_WRITE, n * sizeof(double));
        GPUBuffer d_y_bar(clCtx.context, CL_MEM_READ_WRITE, m * sizeof(double));

        GPUBuffer d_x_hat(clCtx.context, CL_MEM_READ_WRITE, n * sizeof(double));
        GPUBuffer d_w_half(clCtx.context, CL_MEM_READ_WRITE, n * sizeof(double));
        GPUBuffer d_r_y(clCtx.context, CL_MEM_READ_WRITE, m * sizeof(double));

        GPUBuffer d_qw(clCtx.context, CL_MEM_READ_WRITE, n * sizeof(double));
        GPUBuffer d_qw_half(clCtx.context, CL_MEM_READ_WRITE, n * sizeof(double));
        GPUBuffer d_aty(clCtx.context, CL_MEM_READ_WRITE, n * sizeof(double));
        GPUBuffer d_temp_n(clCtx.context, CL_MEM_READ_WRITE, n * sizeof(double));
        GPUBuffer d_temp_m(clCtx.context, CL_MEM_READ_WRITE, m * sizeof(double));

        GPUBuffer d_x_r0(clCtx.context, CL_MEM_READ_WRITE, n * sizeof(double));
        GPUBuffer d_w_r0(clCtx.context, CL_MEM_READ_WRITE, n * sizeof(double));
        GPUBuffer d_z_r0(clCtx.context, CL_MEM_READ_WRITE, n * sizeof(double));
        GPUBuffer d_y_r0(clCtx.context, CL_MEM_READ_WRITE, m * sizeof(double));

        std::cout << "Allocated 24 Working Buffers in GPU VRAM (Zero PCIe transfer overhead during iterations)\n\n";

        std::vector<double> zero_n(n, 0.0);
        std::vector<double> zero_m(m, 0.0);

        const double sigma = 1.0;
        const double lam_q = 9.17e4;
        const double lam_a = 2.17e5;
        const double denom_w = 1.0 + sigma * lam_q;
        const double sig_lam_a = sigma * lam_a;
        const double inv_sig_lam_a = 1.0 / sig_lam_a;
        const double w_step = sigma / denom_w;

        size_t global_n = ((n + 63) / 64) * 64;
        size_t global_m = ((m + 63) / 64) * 64;
        size_t local_sz = 64;

        int test_iters[] = {100, 300};
        for (int iters : test_iters) {
            std::cout << "--- Benchmarking " << iters << " Iterations on Modern C++20 GPU Solver ---\n";

            // Initialize GPU vectors to 0
            d_x.write(clCtx.queue, zero_n.data(), n * sizeof(double));
            d_w_q.write(clCtx.queue, zero_n.data(), n * sizeof(double));
            d_z.write(clCtx.queue, zero_n.data(), n * sizeof(double));
            d_y.write(clCtx.queue, zero_m.data(), m * sizeof(double));
            d_x_r0.write(clCtx.queue, zero_n.data(), n * sizeof(double));
            d_w_r0.write(clCtx.queue, zero_n.data(), n * sizeof(double));
            d_z_r0.write(clCtx.queue, zero_n.data(), n * sizeof(double));
            d_y_r0.write(clCtx.queue, zero_m.data(), m * sizeof(double));
            clFinish(clCtx.queue);

            auto t_start = std::chrono::high_resolution_clock::now();

            int inner_t = 0;
            for (int k = 0; k < iters; ++k) {
                inner_t++;

                // 1) SpMV: Q * w_q -> qw
                clSetKernelArg(k_spmv, 0, sizeof(uint32_t), &n);
                clSetKernelArg(k_spmv, 1, sizeof(cl_mem), &d_P_row.mem);
                clSetKernelArg(k_spmv, 2, sizeof(cl_mem), &d_P_col.mem);
                clSetKernelArg(k_spmv, 3, sizeof(cl_mem), &d_P_val.mem);
                clSetKernelArg(k_spmv, 4, sizeof(cl_mem), &d_w_q.mem);
                clSetKernelArg(k_spmv, 5, sizeof(cl_mem), &d_qw.mem);
                clEnqueueNDRangeKernel(clCtx.queue, k_spmv, 1, nullptr, &global_n, &local_sz, 0, nullptr, nullptr);

                // SpMV: AT * y -> aty
                clSetKernelArg(k_spmv, 0, sizeof(uint32_t), &n);
                clSetKernelArg(k_spmv, 1, sizeof(cl_mem), &d_AT_row.mem);
                clSetKernelArg(k_spmv, 2, sizeof(cl_mem), &d_AT_col.mem);
                clSetKernelArg(k_spmv, 3, sizeof(cl_mem), &d_AT_val.mem);
                clSetKernelArg(k_spmv, 4, sizeof(cl_mem), &d_y.mem);
                clSetKernelArg(k_spmv, 5, sizeof(cl_mem), &d_aty.mem);
                clEnqueueNDRangeKernel(clCtx.queue, k_spmv, 1, nullptr, &global_n, &local_sz, 0, nullptr, nullptr);

                // 2) Fused Step 1 & 2: Box Projection, Reflection, w_half
                clSetKernelArg(k_step1_2, 0, sizeof(uint32_t), &n);
                clSetKernelArg(k_step1_2, 1, sizeof(double), &sigma);
                clSetKernelArg(k_step1_2, 2, sizeof(double), &denom_w);
                clSetKernelArg(k_step1_2, 3, sizeof(double), &lam_q);
                clSetKernelArg(k_step1_2, 4, sizeof(cl_mem), &d_x.mem);
                clSetKernelArg(k_step1_2, 5, sizeof(cl_mem), &d_qw.mem);
                clSetKernelArg(k_step1_2, 6, sizeof(cl_mem), &d_aty.mem);
                clSetKernelArg(k_step1_2, 7, sizeof(cl_mem), &d_c.mem);
                clSetKernelArg(k_step1_2, 8, sizeof(cl_mem), &d_l_c.mem);
                clSetKernelArg(k_step1_2, 9, sizeof(cl_mem), &d_u_c.mem);
                clSetKernelArg(k_step1_2, 10, sizeof(cl_mem), &d_w_q.mem);
                clSetKernelArg(k_step1_2, 11, sizeof(cl_mem), &d_x_bar.mem);
                clSetKernelArg(k_step1_2, 12, sizeof(cl_mem), &d_z_bar.mem);
                clSetKernelArg(k_step1_2, 13, sizeof(cl_mem), &d_x_hat.mem);
                clSetKernelArg(k_step1_2, 14, sizeof(cl_mem), &d_w_half.mem);
                clEnqueueNDRangeKernel(clCtx.queue, k_step1_2, 1, nullptr, &global_n, &local_sz, 0, nullptr, nullptr);

                // 3) Step 3: sGS
                // Q * w_half -> qw_half
                clSetKernelArg(k_spmv, 0, sizeof(uint32_t), &n);
                clSetKernelArg(k_spmv, 1, sizeof(cl_mem), &d_P_row.mem);
                clSetKernelArg(k_spmv, 2, sizeof(cl_mem), &d_P_col.mem);
                clSetKernelArg(k_spmv, 3, sizeof(cl_mem), &d_P_val.mem);
                clSetKernelArg(k_spmv, 4, sizeof(cl_mem), &d_w_half.mem);
                clSetKernelArg(k_spmv, 5, sizeof(cl_mem), &d_qw_half.mem);
                clEnqueueNDRangeKernel(clCtx.queue, k_spmv, 1, nullptr, &global_n, &local_sz, 0, nullptr, nullptr);

                // Prep temp_n
                clSetKernelArg(k_sgs_prep, 0, sizeof(uint32_t), &n);
                clSetKernelArg(k_sgs_prep, 1, sizeof(double), &sigma);
                clSetKernelArg(k_sgs_prep, 2, sizeof(cl_mem), &d_x_hat.mem);
                clSetKernelArg(k_sgs_prep, 3, sizeof(cl_mem), &d_qw.mem);
                clSetKernelArg(k_sgs_prep, 4, sizeof(cl_mem), &d_qw_half.mem);
                clSetKernelArg(k_sgs_prep, 5, sizeof(cl_mem), &d_temp_n.mem);
                clEnqueueNDRangeKernel(clCtx.queue, k_sgs_prep, 1, nullptr, &global_n, &local_sz, 0, nullptr, nullptr);

                // A * temp_n -> r_y
                clSetKernelArg(k_spmv, 0, sizeof(uint32_t), &m);
                clSetKernelArg(k_spmv, 1, sizeof(cl_mem), &d_A_row.mem);
                clSetKernelArg(k_spmv, 2, sizeof(cl_mem), &d_A_col.mem);
                clSetKernelArg(k_spmv, 3, sizeof(cl_mem), &d_A_val.mem);
                clSetKernelArg(k_spmv, 4, sizeof(cl_mem), &d_temp_n.mem);
                clSetKernelArg(k_spmv, 5, sizeof(cl_mem), &d_r_y.mem);
                clEnqueueNDRangeKernel(clCtx.queue, k_spmv, 1, nullptr, &global_m, &local_sz, 0, nullptr, nullptr);

                // sGS y step
                clSetKernelArg(k_sgs_y, 0, sizeof(uint32_t), &m);
                clSetKernelArg(k_sgs_y, 1, sizeof(double), &sig_lam_a);
                clSetKernelArg(k_sgs_y, 2, sizeof(double), &inv_sig_lam_a);
                clSetKernelArg(k_sgs_y, 3, sizeof(cl_mem), &d_l_k.mem);
                clSetKernelArg(k_sgs_y, 4, sizeof(cl_mem), &d_u_k.mem);
                clSetKernelArg(k_sgs_y, 5, sizeof(cl_mem), &d_y.mem);
                clSetKernelArg(k_sgs_y, 6, sizeof(cl_mem), &d_r_y.mem);
                clSetKernelArg(k_sgs_y, 7, sizeof(cl_mem), &d_y_bar.mem);
                clSetKernelArg(k_sgs_y, 8, sizeof(cl_mem), &d_temp_m.mem);
                clEnqueueNDRangeKernel(clCtx.queue, k_sgs_y, 1, nullptr, &global_m, &local_sz, 0, nullptr, nullptr);

                // AT * temp_m -> temp_n
                clSetKernelArg(k_spmv, 0, sizeof(uint32_t), &n);
                clSetKernelArg(k_spmv, 1, sizeof(cl_mem), &d_AT_row.mem);
                clSetKernelArg(k_spmv, 2, sizeof(cl_mem), &d_AT_col.mem);
                clSetKernelArg(k_spmv, 3, sizeof(cl_mem), &d_AT_val.mem);
                clSetKernelArg(k_spmv, 4, sizeof(cl_mem), &d_temp_m.mem);
                clSetKernelArg(k_spmv, 5, sizeof(cl_mem), &d_temp_n.mem);
                clEnqueueNDRangeKernel(clCtx.queue, k_spmv, 1, nullptr, &global_n, &local_sz, 0, nullptr, nullptr);

                // sGS w step
                clSetKernelArg(k_sgs_w, 0, sizeof(uint32_t), &n);
                clSetKernelArg(k_sgs_w, 1, sizeof(double), &w_step);
                clSetKernelArg(k_sgs_w, 2, sizeof(cl_mem), &d_w_half.mem);
                clSetKernelArg(k_sgs_w, 3, sizeof(cl_mem), &d_temp_n.mem);
                clSetKernelArg(k_sgs_w, 4, sizeof(cl_mem), &d_w_bar.mem);
                clEnqueueNDRangeKernel(clCtx.queue, k_sgs_w, 1, nullptr, &global_n, &local_sz, 0, nullptr, nullptr);

                // 4) Halpern Averaging
                double coef_0 = 1.0 / (inner_t + 1.0);
                double coef_hat = static_cast<double>(inner_t) / (inner_t + 1.0);
                clSetKernelArg(k_halpern, 0, sizeof(uint32_t), &n);
                clSetKernelArg(k_halpern, 1, sizeof(uint32_t), &m);
                clSetKernelArg(k_halpern, 2, sizeof(double), &coef_0);
                clSetKernelArg(k_halpern, 3, sizeof(double), &coef_hat);
                clSetKernelArg(k_halpern, 4, sizeof(cl_mem), &d_x_r0.mem);
                clSetKernelArg(k_halpern, 5, sizeof(cl_mem), &d_w_r0.mem);
                clSetKernelArg(k_halpern, 6, sizeof(cl_mem), &d_z_r0.mem);
                clSetKernelArg(k_halpern, 7, sizeof(cl_mem), &d_y_r0.mem);
                clSetKernelArg(k_halpern, 8, sizeof(cl_mem), &d_x_bar.mem);
                clSetKernelArg(k_halpern, 9, sizeof(cl_mem), &d_w_bar.mem);
                clSetKernelArg(k_halpern, 10, sizeof(cl_mem), &d_z_bar.mem);
                clSetKernelArg(k_halpern, 11, sizeof(cl_mem), &d_y_bar.mem);
                clSetKernelArg(k_halpern, 12, sizeof(cl_mem), &d_x.mem);
                clSetKernelArg(k_halpern, 13, sizeof(cl_mem), &d_w_q.mem);
                clSetKernelArg(k_halpern, 14, sizeof(cl_mem), &d_z.mem);
                clSetKernelArg(k_halpern, 15, sizeof(cl_mem), &d_y.mem);
                clEnqueueNDRangeKernel(clCtx.queue, k_halpern, 1, nullptr, &global_n, &local_sz, 0, nullptr, nullptr);
            }

            clFinish(clCtx.queue);
            auto t_end = std::chrono::high_resolution_clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            double ms_per_iter = elapsed_ms / iters;

            // Read back optimal x
            std::vector<double> x_host(n);
            d_x_bar.read(clCtx.queue, x_host.data(), n * sizeof(double));

            std::cout << std::fixed << std::setprecision(2);
            std::cout << "  [New GPU C++20 on NVIDIA MX250] Total: " << std::setw(7) << elapsed_ms
                      << " ms | Latency: " << std::setprecision(3) << ms_per_iter << " ms/iter\n";
            std::cout << std::setprecision(4);
            std::cout << "  Final Optimal x[0..2] on GPU: [" << x_host[0] << ", "
                      << x_host[1] << ", " << x_host[2] << "]\n\n";
        }

        // Release Kernels and Program
        clReleaseKernel(k_spmv); clReleaseKernel(k_step1_2); clReleaseKernel(k_sgs_prep);
        clReleaseKernel(k_sgs_y); clReleaseKernel(k_sgs_w); clReleaseKernel(k_halpern);
        clReleaseProgram(prog);

    } catch (const std::exception& ex) {
        std::cerr << "Fatal Error in C++ GPU Solver: " << ex.what() << "\n";
        return 1;
    }

    std::cout << "=======================================================================\n";
    return 0;
}
