#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
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

// OpenCL Function Pointers
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

static pfn_clGetPlatformIDs clGetPlatformIDs;
static pfn_clGetPlatformInfo clGetPlatformInfo;
static pfn_clGetDeviceIDs clGetDeviceIDs;
static pfn_clGetDeviceInfo clGetDeviceInfo;
static pfn_clCreateContext clCreateContext;
static pfn_clCreateCommandQueue clCreateCommandQueue;
static pfn_clCreateBuffer clCreateBuffer;
static pfn_clCreateProgramWithSource clCreateProgramWithSource;
static pfn_clBuildProgram clBuildProgram;
static pfn_clGetProgramBuildInfo clGetProgramBuildInfo;
static pfn_clCreateKernel clCreateKernel;
static pfn_clSetKernelArg clSetKernelArg;
static pfn_clEnqueueNDRangeKernel clEnqueueNDRangeKernel;
static pfn_clEnqueueReadBuffer clEnqueueReadBuffer;
static pfn_clEnqueueWriteBuffer clEnqueueWriteBuffer;
static pfn_clFinish clFinish;
static pfn_clReleaseMemObject clReleaseMemObject;
static pfn_clReleaseKernel clReleaseKernel;
static pfn_clReleaseProgram clReleaseProgram;
static pfn_clReleaseCommandQueue clReleaseCommandQueue;
static pfn_clReleaseContext clReleaseContext;

static int init_opencl_symbols(HMODULE hCl) {
    clGetPlatformIDs = (pfn_clGetPlatformIDs)GetProcAddress(hCl, "clGetPlatformIDs");
    clGetPlatformInfo = (pfn_clGetPlatformInfo)GetProcAddress(hCl, "clGetPlatformInfo");
    clGetDeviceIDs = (pfn_clGetDeviceIDs)GetProcAddress(hCl, "clGetDeviceIDs");
    clGetDeviceInfo = (pfn_clGetDeviceInfo)GetProcAddress(hCl, "clGetDeviceInfo");
    clCreateContext = (pfn_clCreateContext)GetProcAddress(hCl, "clCreateContext");
    clCreateCommandQueue = (pfn_clCreateCommandQueue)GetProcAddress(hCl, "clCreateCommandQueue");
    clCreateBuffer = (pfn_clCreateBuffer)GetProcAddress(hCl, "clCreateBuffer");
    clCreateProgramWithSource = (pfn_clCreateProgramWithSource)GetProcAddress(hCl, "clCreateProgramWithSource");
    clBuildProgram = (pfn_clBuildProgram)GetProcAddress(hCl, "clBuildProgram");
    clGetProgramBuildInfo = (pfn_clGetProgramBuildInfo)GetProcAddress(hCl, "clGetProgramBuildInfo");
    clCreateKernel = (pfn_clCreateKernel)GetProcAddress(hCl, "clCreateKernel");
    clSetKernelArg = (pfn_clSetKernelArg)GetProcAddress(hCl, "clSetKernelArg");
    clEnqueueNDRangeKernel = (pfn_clEnqueueNDRangeKernel)GetProcAddress(hCl, "clEnqueueNDRangeKernel");
    clEnqueueReadBuffer = (pfn_clEnqueueReadBuffer)GetProcAddress(hCl, "clEnqueueReadBuffer");
    clEnqueueWriteBuffer = (pfn_clEnqueueWriteBuffer)GetProcAddress(hCl, "clEnqueueWriteBuffer");
    clFinish = (pfn_clFinish)GetProcAddress(hCl, "clFinish");
    clReleaseMemObject = (pfn_clReleaseMemObject)GetProcAddress(hCl, "clReleaseMemObject");
    clReleaseKernel = (pfn_clReleaseKernel)GetProcAddress(hCl, "clReleaseKernel");
    clReleaseProgram = (pfn_clReleaseProgram)GetProcAddress(hCl, "clReleaseProgram");
    clReleaseCommandQueue = (pfn_clReleaseCommandQueue)GetProcAddress(hCl, "clReleaseCommandQueue");
    clReleaseContext = (pfn_clReleaseContext)GetProcAddress(hCl, "clReleaseContext");
    return (clGetPlatformIDs && clCreateContext && clEnqueueNDRangeKernel);
}

// OpenCL GPU Kernel Source Code
static const char* GPU_KERNEL_SRC = 
"__kernel void spmv_csr_kernel(\n"
"    const uint rows,\n"
"    __global const int* row_ptr,\n"
"    __global const int* col_idx,\n"
"    __global const double* val,\n"
"    __global const double* x,\n"
"    __global double* y\n"
") {\n"
"    uint i = get_global_id(0);\n"
"    if (i < rows) {\n"
"        double sum = 0.0;\n"
"        int start = row_ptr[i];\n"
"        int end = row_ptr[i + 1];\n"
"        for (int p = start; p < end; ++p) {\n"
"            sum += val[p] * x[col_idx[p]];\n"
"        }\n"
"        y[i] = sum;\n"
"    }\n"
"}\n"
"\n"
"__kernel void fused_step1_2_kernel(\n"
"    const uint n,\n"
"    const double sigma,\n"
"    const double denom_w,\n"
"    const double lam_q,\n"
"    __global const double* x,\n"
"    __global const double* qw,\n"
"    __global const double* aty,\n"
"    __global const double* c,\n"
"    __global const double* l_c,\n"
"    __global const double* u_c,\n"
"    __global const double* w_q,\n"
"    __global double* x_bar,\n"
"    __global double* z_bar,\n"
"    __global double* x_hat,\n"
"    __global double* w_half\n"
") {\n"
"    uint j = get_global_id(0);\n"
"    if (j < n) {\n"
"        double rz = x[j] + sigma * (-qw[j] + aty[j] - c[j]);\n"
"        double xb = rz;\n"
"        if (xb < l_c[j]) xb = l_c[j];\n"
"        if (xb > u_c[j]) xb = u_c[j];\n"
"        x_bar[j] = xb;\n"
"        z_bar[j] = (1.0 / sigma) * (xb - rz);\n"
"\n"
"        double xh = 2.0 * xb - x[j];\n"
"        x_hat[j] = xh;\n"
"        w_half[j] = (1.0 / denom_w) * (sigma * lam_q * w_q[j] + xh);\n"
"    }\n"
"}\n"
"\n"
"__kernel void sgs_prep_temp_n_kernel(\n"
"    const uint n,\n"
"    const double sigma,\n"
"    __global const double* x_hat,\n"
"    __global const double* qw,\n"
"    __global const double* qw_half,\n"
"    __global double* temp_n\n"
") {\n"
"    uint j = get_global_id(0);\n"
"    if (j < n) {\n"
"        temp_n[j] = x_hat[j] + sigma * (qw[j] - qw_half[j]);\n"
"    }\n"
"}\n"
"\n"
"__kernel void sgs_y_step_kernel(\n"
"    const uint m,\n"
"    const double sig_lam_a,\n"
"    const double inv_sig_lam_a,\n"
"    __global const double* l_k,\n"
"    __global const double* u_k,\n"
"    __global const double* y,\n"
"    __global double* r_y,\n"
"    __global double* y_bar,\n"
"    __global double* temp_m\n"
") {\n"
"    uint i = get_global_id(0);\n"
"    if (i < m) {\n"
"        double ry = r_y[i] - sig_lam_a * y[i];\n"
"        double py = ry;\n"
"        if (py < l_k[i]) py = l_k[i];\n"
"        if (py > u_k[i]) py = u_k[i];\n"
"        double yb = inv_sig_lam_a * (py - ry);\n"
"        y_bar[i] = yb;\n"
"        temp_m[i] = yb - y[i];\n"
"    }\n"
"}\n"
"\n"
"__kernel void sgs_w_step_kernel(\n"
"    const uint n,\n"
"    const double w_step,\n"
"    __global const double* w_half,\n"
"    __global const double* at_delta_y,\n"
"    __global double* w_bar\n"
") {\n"
"    uint j = get_global_id(0);\n"
"    if (j < n) {\n"
"        w_bar[j] = w_half[j] + w_step * at_delta_y[j];\n"
"    }\n"
"}\n"
"\n"
"__kernel void halpern_average_kernel(\n"
"    const uint n,\n"
"    const uint m,\n"
"    const double coef_0,\n"
"    const double coef_hat,\n"
"    __global const double* x_r0,\n"
"    __global const double* w_r0,\n"
"    __global const double* z_r0,\n"
"    __global const double* y_r0,\n"
"    __global const double* x_bar,\n"
"    __global const double* w_bar,\n"
"    __global const double* z_bar,\n"
"    __global const double* y_bar,\n"
"    __global double* x,\n"
"    __global double* w_q,\n"
"    __global double* z,\n"
"    __global double* y\n"
") {\n"
"    uint j = get_global_id(0);\n"
"    if (j < n) {\n"
"        x[j] = coef_0 * x_r0[j] + coef_hat * (2.0 * x_bar[j] - x[j]);\n"
"        w_q[j] = coef_0 * w_r0[j] + coef_hat * (2.0 * w_bar[j] - w_q[j]);\n"
"        z[j] = coef_0 * z_r0[j] + coef_hat * (2.0 * z_bar[j] - z[j]);\n"
"    }\n"
"    if (j < m) {\n"
"        y[j] = coef_0 * y_r0[j] + coef_hat * (2.0 * y_bar[j] - y[j]);\n"
"    }\n"
"}\n";

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

int main(void) {
    printf("=======================================================================\n");
    printf("  GPU ACCELERATED HPR-QP IN C (NVIDIA GeForce MX250 / OpenCL)\n");
    printf("  Problem: CVXQP1_L (n = 10,000, m = 15,000, Nonzeros = 94,966)\n");
    printf("=======================================================================\n");

    HMODULE hCl = LoadLibraryA("OpenCL.dll");
    if (!hCl || !init_opencl_symbols(hCl)) {
        printf("Error: Could not load OpenCL.dll or resolve symbols.\n");
        return 1;
    }

    // Select NVIDIA GPU device
    cl_uint num_platforms = 0;
    clGetPlatformIDs(0, NULL, &num_platforms);
    cl_platform_id platforms[8];
    clGetPlatformIDs(num_platforms, platforms, NULL);

    cl_device_id gpu_dev = NULL;
    char dev_name[256] = "Unknown GPU";

    for (cl_uint p = 0; p < num_platforms; ++p) {
        cl_uint num_devs = 0;
        clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devs);
        if (num_devs > 0) {
            cl_device_id devs[4];
            clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, num_devs, devs, NULL);
            gpu_dev = devs[0];
            clGetDeviceInfo(gpu_dev, CL_DEVICE_NAME, sizeof(dev_name), dev_name, NULL);
            // Prefer NVIDIA GPU
            char plat_name[256];
            clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, sizeof(plat_name), plat_name, NULL);
            if (strstr(plat_name, "NVIDIA") != NULL) {
                break;
            }
        }
    }

    if (!gpu_dev) {
        printf("Error: No GPU device found.\n");
        return 1;
    }

    printf("Selected GPU Device   : %s\n", dev_name);

    cl_int err;
    cl_context ctx = clCreateContext(NULL, 1, &gpu_dev, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueue(ctx, gpu_dev, 0, &err);

    // Build OpenCL Program
    const char *src = GPU_KERNEL_SRC;
    cl_program prog = clCreateProgramWithSource(ctx, 1, &src, NULL, &err);
    err = clBuildProgram(prog, 1, &gpu_dev, "-cl-fast-relaxed-math", NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[4096];
        clGetProgramBuildInfo(prog, gpu_dev, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        printf("OpenCL Build Error:\n%s\n", log);
        return 1;
    }

    cl_kernel k_spmv = clCreateKernel(prog, "spmv_csr_kernel", &err);
    cl_kernel k_step1_2 = clCreateKernel(prog, "fused_step1_2_kernel", &err);
    cl_kernel k_sgs_prep = clCreateKernel(prog, "sgs_prep_temp_n_kernel", &err);
    cl_kernel k_sgs_y = clCreateKernel(prog, "sgs_y_step_kernel", &err);
    cl_kernel k_sgs_w = clCreateKernel(prog, "sgs_w_step_kernel", &err);
    cl_kernel k_halpern = clCreateKernel(prog, "halpern_average_kernel", &err);

    // Load Problem
    LargeDataset *ds = load_binary_dataset("Data/cvxqp1_l.bin");
    if (!ds) {
        printf("Error: Failed to load Data/cvxqp1_l.bin\n");
        return 1;
    }
    uint32_t n = ds->n;
    uint32_t m = ds->m;

    printf("Dataset Loaded in RAM : n=%u, m=%u, nnz=%llu\n", n, m, (unsigned long long)(ds->nnz_P + ds->nnz_A));

    // Allocate GPU Device Memory (VRAM)
    cl_mem d_P_row = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (n + 1) * sizeof(int32_t), ds->P_row_ptr, &err);
    cl_mem d_P_col = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ds->nnz_P * sizeof(int32_t), ds->P_col_idx, &err);
    cl_mem d_P_val = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ds->nnz_P * sizeof(double), ds->P_val, &err);

    cl_mem d_A_row = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (m + 1) * sizeof(int32_t), ds->A_row_ptr, &err);
    cl_mem d_A_col = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ds->nnz_A * sizeof(int32_t), ds->A_col_idx, &err);
    cl_mem d_A_val = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ds->nnz_A * sizeof(double), ds->A_val, &err);

    cl_mem d_AT_row = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (n + 1) * sizeof(int32_t), ds->AT_row_ptr, &err);
    cl_mem d_AT_col = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ds->nnz_AT * sizeof(int32_t), ds->AT_col_idx, &err);
    cl_mem d_AT_val = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ds->nnz_AT * sizeof(double), ds->AT_val, &err);

    cl_mem d_c = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(double), ds->c, &err);
    cl_mem d_l_k = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, m * sizeof(double), ds->l_k, &err);
    cl_mem d_u_k = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, m * sizeof(double), ds->u_k, &err);
    cl_mem d_l_c = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(double), ds->l_c, &err);
    cl_mem d_u_c = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(double), ds->u_c, &err);

    // Working Vector Buffers on GPU VRAM
    cl_mem d_x = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n * sizeof(double), NULL, &err);
    cl_mem d_w_q = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n * sizeof(double), NULL, &err);
    cl_mem d_z = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n * sizeof(double), NULL, &err);
    cl_mem d_y = clCreateBuffer(ctx, CL_MEM_READ_WRITE, m * sizeof(double), NULL, &err);

    cl_mem d_x_bar = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n * sizeof(double), NULL, &err);
    cl_mem d_w_bar = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n * sizeof(double), NULL, &err);
    cl_mem d_z_bar = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n * sizeof(double), NULL, &err);
    cl_mem d_y_bar = clCreateBuffer(ctx, CL_MEM_READ_WRITE, m * sizeof(double), NULL, &err);

    cl_mem d_x_hat = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n * sizeof(double), NULL, &err);
    cl_mem d_w_half = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n * sizeof(double), NULL, &err);
    cl_mem d_r_y = clCreateBuffer(ctx, CL_MEM_READ_WRITE, m * sizeof(double), NULL, &err);
    cl_mem d_qw = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n * sizeof(double), NULL, &err);
    cl_mem d_qw_half = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n * sizeof(double), NULL, &err);
    cl_mem d_aty = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n * sizeof(double), NULL, &err);
    cl_mem d_temp_n = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n * sizeof(double), NULL, &err);
    cl_mem d_temp_m = clCreateBuffer(ctx, CL_MEM_READ_WRITE, m * sizeof(double), NULL, &err);

    cl_mem d_x_r0 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n * sizeof(double), NULL, &err);
    cl_mem d_w_r0 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n * sizeof(double), NULL, &err);
    cl_mem d_z_r0 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n * sizeof(double), NULL, &err);
    cl_mem d_y_r0 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, m * sizeof(double), NULL, &err);

    printf("Allocated 24 Working Buffers in GPU VRAM (Zero PCIe transfer overhead during iterations)\n\n");

    double *zero_n = (double*)calloc(n, sizeof(double));
    double *zero_m = (double*)calloc(m, sizeof(double));

    double sigma = 1.0;
    double lam_q = 9.17e4;
    double lam_a = 2.17e5;
    double denom_w = 1.0 + sigma * lam_q;
    double sig_lam_a = sigma * lam_a;
    double inv_sig_lam_a = 1.0 / sig_lam_a;
    double w_step = sigma / denom_w;

    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);

    size_t global_n = ((n + 63) / 64) * 64;
    size_t global_m = ((m + 63) / 64) * 64;
    size_t local_sz = 64;

    int test_iters[2] = {100, 300};
    for (int it_idx = 0; it_idx < 2; ++it_idx) {
        int iters = test_iters[it_idx];
        printf("--- Benchmarking %d Iterations on GPU ---\n", iters);

        // Reset GPU working vectors to 0
        clEnqueueWriteBuffer(queue, d_x, CL_TRUE, 0, n * sizeof(double), zero_n, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, d_w_q, CL_TRUE, 0, n * sizeof(double), zero_n, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, d_z, CL_TRUE, 0, n * sizeof(double), zero_n, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, d_y, CL_TRUE, 0, m * sizeof(double), zero_m, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, d_x_r0, CL_TRUE, 0, n * sizeof(double), zero_n, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, d_w_r0, CL_TRUE, 0, n * sizeof(double), zero_n, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, d_z_r0, CL_TRUE, 0, n * sizeof(double), zero_n, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, d_y_r0, CL_TRUE, 0, m * sizeof(double), zero_m, 0, NULL, NULL);
        clFinish(queue);

        LARGE_INTEGER t_start, t_end;
        QueryPerformanceCounter(&t_start);

        int inner_t = 0;
        for (int k = 0; k < iters; ++k) {
            inner_t++;

            // 1) SpMV on GPU: Q * w_q -> qw
            clSetKernelArg(k_spmv, 0, sizeof(uint32_t), &n);
            clSetKernelArg(k_spmv, 1, sizeof(cl_mem), &d_P_row);
            clSetKernelArg(k_spmv, 2, sizeof(cl_mem), &d_P_col);
            clSetKernelArg(k_spmv, 3, sizeof(cl_mem), &d_P_val);
            clSetKernelArg(k_spmv, 4, sizeof(cl_mem), &d_w_q);
            clSetKernelArg(k_spmv, 5, sizeof(cl_mem), &d_qw);
            clEnqueueNDRangeKernel(queue, k_spmv, 1, NULL, &global_n, &local_sz, 0, NULL, NULL);

            // SpMV on GPU: AT * y -> aty
            clSetKernelArg(k_spmv, 0, sizeof(uint32_t), &n);
            clSetKernelArg(k_spmv, 1, sizeof(cl_mem), &d_AT_row);
            clSetKernelArg(k_spmv, 2, sizeof(cl_mem), &d_AT_col);
            clSetKernelArg(k_spmv, 3, sizeof(cl_mem), &d_AT_val);
            clSetKernelArg(k_spmv, 4, sizeof(cl_mem), &d_y);
            clSetKernelArg(k_spmv, 5, sizeof(cl_mem), &d_aty);
            clEnqueueNDRangeKernel(queue, k_spmv, 1, NULL, &global_n, &local_sz, 0, NULL, NULL);

            // 2) Fused Step 1 & 2 on GPU: Box projection, reflection, w_half
            clSetKernelArg(k_step1_2, 0, sizeof(uint32_t), &n);
            clSetKernelArg(k_step1_2, 1, sizeof(double), &sigma);
            clSetKernelArg(k_step1_2, 2, sizeof(double), &denom_w);
            clSetKernelArg(k_step1_2, 3, sizeof(double), &lam_q);
            clSetKernelArg(k_step1_2, 4, sizeof(cl_mem), &d_x);
            clSetKernelArg(k_step1_2, 5, sizeof(cl_mem), &d_qw);
            clSetKernelArg(k_step1_2, 6, sizeof(cl_mem), &d_aty);
            clSetKernelArg(k_step1_2, 7, sizeof(cl_mem), &d_c);
            clSetKernelArg(k_step1_2, 8, sizeof(cl_mem), &d_l_c);
            clSetKernelArg(k_step1_2, 9, sizeof(cl_mem), &d_u_c);
            clSetKernelArg(k_step1_2, 10, sizeof(cl_mem), &d_w_q);
            clSetKernelArg(k_step1_2, 11, sizeof(cl_mem), &d_x_bar);
            clSetKernelArg(k_step1_2, 12, sizeof(cl_mem), &d_z_bar);
            clSetKernelArg(k_step1_2, 13, sizeof(cl_mem), &d_x_hat);
            clSetKernelArg(k_step1_2, 14, sizeof(cl_mem), &d_w_half);
            clEnqueueNDRangeKernel(queue, k_step1_2, 1, NULL, &global_n, &local_sz, 0, NULL, NULL);

            // 3) Step 3: sGS
            // Q * w_half -> qw_half
            clSetKernelArg(k_spmv, 0, sizeof(uint32_t), &n);
            clSetKernelArg(k_spmv, 1, sizeof(cl_mem), &d_P_row);
            clSetKernelArg(k_spmv, 2, sizeof(cl_mem), &d_P_col);
            clSetKernelArg(k_spmv, 3, sizeof(cl_mem), &d_P_val);
            clSetKernelArg(k_spmv, 4, sizeof(cl_mem), &d_w_half);
            clSetKernelArg(k_spmv, 5, sizeof(cl_mem), &d_qw_half);
            clEnqueueNDRangeKernel(queue, k_spmv, 1, NULL, &global_n, &local_sz, 0, NULL, NULL);

            // Prep temp_n on GPU
            clSetKernelArg(k_sgs_prep, 0, sizeof(uint32_t), &n);
            clSetKernelArg(k_sgs_prep, 1, sizeof(double), &sigma);
            clSetKernelArg(k_sgs_prep, 2, sizeof(cl_mem), &d_x_hat);
            clSetKernelArg(k_sgs_prep, 3, sizeof(cl_mem), &d_qw);
            clSetKernelArg(k_sgs_prep, 4, sizeof(cl_mem), &d_qw_half);
            clSetKernelArg(k_sgs_prep, 5, sizeof(cl_mem), &d_temp_n);
            clEnqueueNDRangeKernel(queue, k_sgs_prep, 1, NULL, &global_n, &local_sz, 0, NULL, NULL);

            // A * temp_n -> r_y
            clSetKernelArg(k_spmv, 0, sizeof(uint32_t), &m);
            clSetKernelArg(k_spmv, 1, sizeof(cl_mem), &d_A_row);
            clSetKernelArg(k_spmv, 2, sizeof(cl_mem), &d_A_col);
            clSetKernelArg(k_spmv, 3, sizeof(cl_mem), &d_A_val);
            clSetKernelArg(k_spmv, 4, sizeof(cl_mem), &d_temp_n);
            clSetKernelArg(k_spmv, 5, sizeof(cl_mem), &d_r_y);
            clEnqueueNDRangeKernel(queue, k_spmv, 1, NULL, &global_m, &local_sz, 0, NULL, NULL);

            // sGS y step on GPU
            clSetKernelArg(k_sgs_y, 0, sizeof(uint32_t), &m);
            clSetKernelArg(k_sgs_y, 1, sizeof(double), &sig_lam_a);
            clSetKernelArg(k_sgs_y, 2, sizeof(double), &inv_sig_lam_a);
            clSetKernelArg(k_sgs_y, 3, sizeof(cl_mem), &d_l_k);
            clSetKernelArg(k_sgs_y, 4, sizeof(cl_mem), &d_u_k);
            clSetKernelArg(k_sgs_y, 5, sizeof(cl_mem), &d_y);
            clSetKernelArg(k_sgs_y, 6, sizeof(cl_mem), &d_r_y);
            clSetKernelArg(k_sgs_y, 7, sizeof(cl_mem), &d_y_bar);
            clSetKernelArg(k_sgs_y, 8, sizeof(cl_mem), &d_temp_m);
            clEnqueueNDRangeKernel(queue, k_sgs_y, 1, NULL, &global_m, &local_sz, 0, NULL, NULL);

            // AT * temp_m -> temp_n
            clSetKernelArg(k_spmv, 0, sizeof(uint32_t), &n);
            clSetKernelArg(k_spmv, 1, sizeof(cl_mem), &d_AT_row);
            clSetKernelArg(k_spmv, 2, sizeof(cl_mem), &d_AT_col);
            clSetKernelArg(k_spmv, 3, sizeof(cl_mem), &d_AT_val);
            clSetKernelArg(k_spmv, 4, sizeof(cl_mem), &d_temp_m);
            clSetKernelArg(k_spmv, 5, sizeof(cl_mem), &d_temp_n);
            clEnqueueNDRangeKernel(queue, k_spmv, 1, NULL, &global_n, &local_sz, 0, NULL, NULL);

            // sGS w step on GPU
            clSetKernelArg(k_sgs_w, 0, sizeof(uint32_t), &n);
            clSetKernelArg(k_sgs_w, 1, sizeof(double), &w_step);
            clSetKernelArg(k_sgs_w, 2, sizeof(cl_mem), &d_w_half);
            clSetKernelArg(k_sgs_w, 3, sizeof(cl_mem), &d_temp_n);
            clSetKernelArg(k_sgs_w, 4, sizeof(cl_mem), &d_w_bar);
            clEnqueueNDRangeKernel(queue, k_sgs_w, 1, NULL, &global_n, &local_sz, 0, NULL, NULL);

            // 4) Halpern Averaging on GPU
            double coef_0 = 1.0 / (inner_t + 1.0);
            double coef_hat = (double)inner_t / (inner_t + 1.0);
            clSetKernelArg(k_halpern, 0, sizeof(uint32_t), &n);
            clSetKernelArg(k_halpern, 1, sizeof(uint32_t), &m);
            clSetKernelArg(k_halpern, 2, sizeof(double), &coef_0);
            clSetKernelArg(k_halpern, 3, sizeof(double), &coef_hat);
            clSetKernelArg(k_halpern, 4, sizeof(cl_mem), &d_x_r0);
            clSetKernelArg(k_halpern, 5, sizeof(cl_mem), &d_w_r0);
            clSetKernelArg(k_halpern, 6, sizeof(cl_mem), &d_z_r0);
            clSetKernelArg(k_halpern, 7, sizeof(cl_mem), &d_y_r0);
            clSetKernelArg(k_halpern, 8, sizeof(cl_mem), &d_x_bar);
            clSetKernelArg(k_halpern, 9, sizeof(cl_mem), &d_w_bar);
            clSetKernelArg(k_halpern, 10, sizeof(cl_mem), &d_z_bar);
            clSetKernelArg(k_halpern, 11, sizeof(cl_mem), &d_y_bar);
            clSetKernelArg(k_halpern, 12, sizeof(cl_mem), &d_x);
            clSetKernelArg(k_halpern, 13, sizeof(cl_mem), &d_w_q);
            clSetKernelArg(k_halpern, 14, sizeof(cl_mem), &d_z);
            clSetKernelArg(k_halpern, 15, sizeof(cl_mem), &d_y);
            clEnqueueNDRangeKernel(queue, k_halpern, 1, NULL, &global_n, &local_sz, 0, NULL, NULL);
        }

        // Wait for GPU to finish all enqueued kernels
        clFinish(queue);
        QueryPerformanceCounter(&t_end);
        double elapsed_ms = (double)(t_end.QuadPart - t_start.QuadPart) * 1000.0 / freq.QuadPart;
        double ms_per_iter = elapsed_ms / iters;

        // Read back final optimal x from GPU
        double *x_host = (double*)malloc(n * sizeof(double));
        clEnqueueReadBuffer(queue, d_x_bar, CL_TRUE, 0, n * sizeof(double), x_host, 0, NULL, NULL);

        printf("  [New GPU C on NVIDIA MX250] Total: %7.2f ms | Latency: %.3f ms/iter\n",
               elapsed_ms, ms_per_iter);
        printf("  Final Optimal x[0..2] on GPU: [%.4f, %.4f, %.4f]\n\n",
               x_host[0], x_host[1], x_host[2]);

        free(x_host);
    }

    // Release GPU Resources
    free(zero_n); free(zero_m);
    clReleaseMemObject(d_P_row); clReleaseMemObject(d_P_col); clReleaseMemObject(d_P_val);
    clReleaseMemObject(d_A_row); clReleaseMemObject(d_A_col); clReleaseMemObject(d_A_val);
    clReleaseMemObject(d_AT_row); clReleaseMemObject(d_AT_col); clReleaseMemObject(d_AT_val);
    clReleaseMemObject(d_c); clReleaseMemObject(d_l_k); clReleaseMemObject(d_u_k);
    clReleaseMemObject(d_l_c); clReleaseMemObject(d_u_c);
    clReleaseMemObject(d_x); clReleaseMemObject(d_w_q); clReleaseMemObject(d_z); clReleaseMemObject(d_y);
    clReleaseMemObject(d_x_bar); clReleaseMemObject(d_w_bar); clReleaseMemObject(d_z_bar); clReleaseMemObject(d_y_bar);
    clReleaseMemObject(d_x_hat); clReleaseMemObject(d_w_half); clReleaseMemObject(d_r_y);
    clReleaseMemObject(d_qw); clReleaseMemObject(d_qw_half); clReleaseMemObject(d_aty);
    clReleaseMemObject(d_temp_n); clReleaseMemObject(d_temp_m);
    clReleaseMemObject(d_x_r0); clReleaseMemObject(d_w_r0); clReleaseMemObject(d_z_r0); clReleaseMemObject(d_y_r0);
    clReleaseKernel(k_spmv); clReleaseKernel(k_step1_2); clReleaseKernel(k_sgs_prep);
    clReleaseKernel(k_sgs_y); clReleaseKernel(k_sgs_w); clReleaseKernel(k_halpern);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    FreeLibrary(hCl);
    free_dataset(ds);

    printf("=======================================================================\n");
    return 0;
}
