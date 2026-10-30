/**
 * OpenCL Cross-Platform Dynamic Loader for C99
 * Supports dynamic runtime binding to OpenCL on Windows (OpenCL.dll)
 * and Linux / Unix (libOpenCL.so / libOpenCL.so.1).
 */
#ifndef OPENCL_LOADER_H
#define OPENCL_LOADER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
  #define CL_API_CALL __stdcall
  typedef HMODULE CL_LIB_HANDLE;
  #define LOAD_LIB(name) LoadLibraryA(name)
  #define GET_SYM(h, name) GetProcAddress(h, name)
  #define CLOSE_LIB(h) FreeLibrary(h)
  #define OPENCL_LIB_NAME "OpenCL.dll"
#else
  #include <dlfcn.h>
  #define CL_API_CALL
  typedef void* CL_LIB_HANDLE;
  #define LOAD_LIB(name) dlopen(name, RTLD_LAZY | RTLD_GLOBAL)
  #define GET_SYM(h, name) dlsym(h, name)
  #define CLOSE_LIB(h) dlclose(h)
  #define OPENCL_LIB_NAME "libOpenCL.so.1"
#endif

/* OpenCL Base Types */
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

/* OpenCL Standard Constants */
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

/* OpenCL Function Pointer Types */
typedef cl_int (CL_API_CALL *pfn_clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int (CL_API_CALL *pfn_clGetPlatformInfo)(cl_platform_id, cl_uint, size_t, void*, size_t*);
typedef cl_int (CL_API_CALL *pfn_clGetDeviceIDs)(cl_platform_id, cl_ulong, cl_uint, cl_device_id*, cl_uint*);
typedef cl_int (CL_API_CALL *pfn_clGetDeviceInfo)(cl_device_id, cl_uint, size_t, void*, size_t*);
typedef cl_context (CL_API_CALL *pfn_clCreateContext)(const void*, cl_uint, const cl_device_id*, void*, void*, cl_int*);
typedef cl_command_queue (CL_API_CALL *pfn_clCreateCommandQueue)(cl_context, cl_device_id, cl_ulong, cl_int*);
typedef cl_mem (CL_API_CALL *pfn_clCreateBuffer)(cl_context, cl_ulong, size_t, void*, cl_int*);
typedef cl_program (CL_API_CALL *pfn_clCreateProgramWithSource)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
typedef cl_int (CL_API_CALL *pfn_clBuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*, void*, void*);
typedef cl_int (CL_API_CALL *pfn_clGetProgramBuildInfo)(cl_program, cl_device_id, cl_uint, size_t, void*, size_t*);
typedef cl_kernel (CL_API_CALL *pfn_clCreateKernel)(cl_program, const char*, cl_int*);
typedef cl_int (CL_API_CALL *pfn_clSetKernelArg)(cl_kernel, cl_uint, size_t, const void*);
typedef cl_int (CL_API_CALL *pfn_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const void*, void*);
typedef cl_int (CL_API_CALL *pfn_clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_uint, size_t, size_t, void*, cl_uint, const void*, void*);
typedef cl_int (CL_API_CALL *pfn_clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_uint, size_t, size_t, const void*, cl_uint, const void*, void*);
typedef cl_int (CL_API_CALL *pfn_clFinish)(cl_command_queue);
typedef cl_int (CL_API_CALL *pfn_clReleaseMemObject)(cl_mem);
typedef cl_int (CL_API_CALL *pfn_clReleaseKernel)(cl_kernel);
typedef cl_int (CL_API_CALL *pfn_clReleaseProgram)(cl_program);
typedef cl_int (CL_API_CALL *pfn_clReleaseCommandQueue)(cl_command_queue);
typedef cl_int (CL_API_CALL *pfn_clReleaseContext)(cl_context);

typedef struct {
    CL_LIB_HANDLE handle;
    pfn_clGetPlatformIDs clGetPlatformIDs;
    pfn_clGetPlatformInfo clGetPlatformInfo;
    pfn_clGetDeviceIDs clGetDeviceIDs;
    pfn_clGetDeviceInfo clGetDeviceInfo;
    pfn_clCreateContext clCreateContext;
    pfn_clCreateCommandQueue clCreateCommandQueue;
    pfn_clCreateBuffer clCreateBuffer;
    pfn_clCreateProgramWithSource clCreateProgramWithSource;
    pfn_clBuildProgram clBuildProgram;
    pfn_clGetProgramBuildInfo clGetProgramBuildInfo;
    pfn_clCreateKernel clCreateKernel;
    pfn_clSetKernelArg clSetKernelArg;
    pfn_clEnqueueNDRangeKernel clEnqueueNDRangeKernel;
    pfn_clEnqueueReadBuffer clEnqueueReadBuffer;
    pfn_clEnqueueWriteBuffer clEnqueueWriteBuffer;
    pfn_clFinish clFinish;
    pfn_clReleaseMemObject clReleaseMemObject;
    pfn_clReleaseKernel clReleaseKernel;
    pfn_clReleaseProgram clReleaseProgram;
    pfn_clReleaseCommandQueue clReleaseCommandQueue;
    pfn_clReleaseContext clReleaseContext;
} OpenCLBindings;

static inline bool opencl_load_bindings(OpenCLBindings *b) {
    if (!b) return false;
    memset(b, 0, sizeof(OpenCLBindings));

    b->handle = LOAD_LIB(OPENCL_LIB_NAME);
#ifndef _WIN32
    if (!b->handle) {
        b->handle = LOAD_LIB("libOpenCL.so");
    }
#endif
    if (!b->handle) {
        return false;
    }

    #define BIND_FN(name) \
        b->name = (pfn_##name)GET_SYM(b->handle, #name); \
        if (!b->name) { CLOSE_LIB(b->handle); b->handle = NULL; return false; }

    BIND_FN(clGetPlatformIDs)
    BIND_FN(clGetPlatformInfo)
    BIND_FN(clGetDeviceIDs)
    BIND_FN(clGetDeviceInfo)
    BIND_FN(clCreateContext)
    BIND_FN(clCreateCommandQueue)
    BIND_FN(clCreateBuffer)
    BIND_FN(clCreateProgramWithSource)
    BIND_FN(clBuildProgram)
    BIND_FN(clGetProgramBuildInfo)
    BIND_FN(clCreateKernel)
    BIND_FN(clSetKernelArg)
    BIND_FN(clEnqueueNDRangeKernel)
    BIND_FN(clEnqueueReadBuffer)
    BIND_FN(clEnqueueWriteBuffer)
    BIND_FN(clFinish)
    BIND_FN(clReleaseMemObject)
    BIND_FN(clReleaseKernel)
    BIND_FN(clReleaseProgram)
    BIND_FN(clReleaseCommandQueue)
    BIND_FN(clReleaseContext)

    #undef BIND_FN
    return true;
}

static inline void opencl_unload_bindings(OpenCLBindings *b) {
    if (b && b->handle) {
        CLOSE_LIB(b->handle);
        memset(b, 0, sizeof(OpenCLBindings));
    }
}

#endif /* OPENCL_LOADER_H */
