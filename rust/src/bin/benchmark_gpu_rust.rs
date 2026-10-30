//! GPU-Accelerated HPR-QP in Rust targeting NVIDIA GeForce MX250 via OpenCL.
//! Zero PCIe memory transfer overhead during iterations.

use std::fs::File;
use std::io::{Read, BufReader};
use std::time::Instant;
use std::ffi::{c_void, CString};

// Windows Dynamic Library Loading FFI
extern "system" {
    fn LoadLibraryA(lpLibFileName: *const u8) -> *mut c_void;
    fn GetProcAddress(hModule: *mut c_void, lpProcName: *const u8) -> *mut c_void;
    fn FreeLibrary(hLibModule: *mut c_void) -> i32;
}

// OpenCL API Types and Constants
#[allow(non_camel_case_types)]
type cl_int = i32;
#[allow(non_camel_case_types)]
type cl_uint = u32;
#[allow(non_camel_case_types)]
type cl_ulong = u64;
#[allow(non_camel_case_types)]
type cl_platform_id = *mut c_void;
#[allow(non_camel_case_types)]
type cl_device_id = *mut c_void;
#[allow(non_camel_case_types)]
type cl_context = *mut c_void;
#[allow(non_camel_case_types)]
type cl_command_queue = *mut c_void;
#[allow(non_camel_case_types)]
type cl_mem = *mut c_void;
#[allow(non_camel_case_types)]
type cl_program = *mut c_void;
#[allow(non_camel_case_types)]
type cl_kernel = *mut c_void;

const CL_SUCCESS: cl_int = 0;
const CL_TRUE: cl_uint = 1;
const CL_DEVICE_NAME: cl_uint = 0x102B;
const CL_DEVICE_TYPE_GPU: cl_ulong = 1 << 2;
const CL_MEM_READ_ONLY: cl_ulong = 1 << 2;
const CL_MEM_READ_WRITE: cl_ulong = 1 << 0;
const CL_MEM_COPY_HOST_PTR: cl_ulong = 1 << 5;
const CL_PROGRAM_BUILD_LOG: cl_uint = 0x1183;

type PfnClGetPlatformIDs = unsafe extern "system" fn(cl_uint, *mut cl_platform_id, *mut cl_uint) -> cl_int;
type PfnClGetDeviceIDs = unsafe extern "system" fn(cl_platform_id, cl_ulong, cl_uint, *mut cl_device_id, *mut cl_uint) -> cl_int;
type PfnClGetDeviceInfo = unsafe extern "system" fn(cl_device_id, cl_uint, usize, *mut c_void, *mut usize) -> cl_int;
type PfnClCreateContext = unsafe extern "system" fn(*const c_void, cl_uint, *const cl_device_id, *mut c_void, *mut c_void, *mut cl_int) -> cl_context;
type PfnClCreateCommandQueue = unsafe extern "system" fn(cl_context, cl_device_id, cl_ulong, *mut cl_int) -> cl_command_queue;
type PfnClCreateBuffer = unsafe extern "system" fn(cl_context, cl_ulong, usize, *mut c_void, *mut cl_int) -> cl_mem;
type PfnClCreateProgramWithSource = unsafe extern "system" fn(cl_context, cl_uint, *const *const u8, *const usize, *mut cl_int) -> cl_program;
type PfnClBuildProgram = unsafe extern "system" fn(cl_program, cl_uint, *const cl_device_id, *const u8, *mut c_void, *mut c_void) -> cl_int;
type PfnClGetProgramBuildInfo = unsafe extern "system" fn(cl_program, cl_device_id, cl_uint, usize, *mut c_void, *mut usize) -> cl_int;
type PfnClCreateKernel = unsafe extern "system" fn(cl_program, *const u8, *mut cl_int) -> cl_kernel;
type PfnClSetKernelArg = unsafe extern "system" fn(cl_kernel, cl_uint, usize, *const c_void) -> cl_int;
type PfnClEnqueueNDRangeKernel = unsafe extern "system" fn(cl_command_queue, cl_kernel, cl_uint, *const usize, *const usize, *const usize, cl_uint, *const c_void, *mut c_void) -> cl_int;
type PfnClEnqueueReadBuffer = unsafe extern "system" fn(cl_command_queue, cl_mem, cl_uint, usize, usize, *mut c_void, cl_uint, *const c_void, *mut c_void) -> cl_int;
type PfnClEnqueueWriteBuffer = unsafe extern "system" fn(cl_command_queue, cl_mem, cl_uint, usize, usize, *const c_void, cl_uint, *const c_void, *mut c_void) -> cl_int;
type PfnClFinish = unsafe extern "system" fn(cl_command_queue) -> cl_int;
type PfnClReleaseMemObject = unsafe extern "system" fn(cl_mem) -> cl_int;
type PfnClReleaseKernel = unsafe extern "system" fn(cl_kernel) -> cl_int;
type PfnClReleaseProgram = unsafe extern "system" fn(cl_program) -> cl_int;
type PfnClReleaseCommandQueue = unsafe extern "system" fn(cl_command_queue) -> cl_int;
type PfnClReleaseContext = unsafe extern "system" fn(cl_context) -> cl_int;

#[derive(Clone)]
struct OpenCLApi {
    cl_get_platform_ids: PfnClGetPlatformIDs,
    cl_get_device_ids: PfnClGetDeviceIDs,
    cl_get_device_info: PfnClGetDeviceInfo,
    cl_create_context: PfnClCreateContext,
    cl_create_command_queue: PfnClCreateCommandQueue,
    cl_create_buffer: PfnClCreateBuffer,
    cl_create_program_with_source: PfnClCreateProgramWithSource,
    cl_build_program: PfnClBuildProgram,
    cl_get_program_build_info: PfnClGetProgramBuildInfo,
    cl_create_kernel: PfnClCreateKernel,
    cl_set_kernel_arg: PfnClSetKernelArg,
    cl_enqueue_nd_range_kernel: PfnClEnqueueNDRangeKernel,
    cl_enqueue_read_buffer: PfnClEnqueueReadBuffer,
    cl_enqueue_write_buffer: PfnClEnqueueWriteBuffer,
    cl_finish: PfnClFinish,
    cl_release_mem_object: PfnClReleaseMemObject,
    cl_release_kernel: PfnClReleaseKernel,
    cl_release_program: PfnClReleaseProgram,
    cl_release_command_queue: PfnClReleaseCommandQueue,
    cl_release_context: PfnClReleaseContext,
}

impl OpenCLApi {
    fn load() -> (Self, *mut c_void) {
        unsafe {
            let lib_name = b"OpenCL.dll\0";
            let handle = LoadLibraryA(lib_name.as_ptr());
            if handle.is_null() {
                panic!("Failed to load OpenCL.dll from system.");
            }

            macro_rules! get_fn {
                ($name:ident, $sym:expr, $t:ident) => {
                    let ptr = GetProcAddress(handle, $sym.as_ptr());
                    if ptr.is_null() {
                        panic!("Failed to load OpenCL symbol: {:?}", std::str::from_utf8($sym));
                    }
                    let $name: $t = std::mem::transmute(ptr);
                };
            }

            get_fn!(cl_get_platform_ids, b"clGetPlatformIDs\0", PfnClGetPlatformIDs);
            get_fn!(cl_get_device_ids, b"clGetDeviceIDs\0", PfnClGetDeviceIDs);
            get_fn!(cl_get_device_info, b"clGetDeviceInfo\0", PfnClGetDeviceInfo);
            get_fn!(cl_create_context, b"clCreateContext\0", PfnClCreateContext);
            get_fn!(cl_create_command_queue, b"clCreateCommandQueue\0", PfnClCreateCommandQueue);
            get_fn!(cl_create_buffer, b"clCreateBuffer\0", PfnClCreateBuffer);
            get_fn!(cl_create_program_with_source, b"clCreateProgramWithSource\0", PfnClCreateProgramWithSource);
            get_fn!(cl_build_program, b"clBuildProgram\0", PfnClBuildProgram);
            get_fn!(cl_get_program_build_info, b"clGetProgramBuildInfo\0", PfnClGetProgramBuildInfo);
            get_fn!(cl_create_kernel, b"clCreateKernel\0", PfnClCreateKernel);
            get_fn!(cl_set_kernel_arg, b"clSetKernelArg\0", PfnClSetKernelArg);
            get_fn!(cl_enqueue_nd_range_kernel, b"clEnqueueNDRangeKernel\0", PfnClEnqueueNDRangeKernel);
            get_fn!(cl_enqueue_read_buffer, b"clEnqueueReadBuffer\0", PfnClEnqueueReadBuffer);
            get_fn!(cl_enqueue_write_buffer, b"clEnqueueWriteBuffer\0", PfnClEnqueueWriteBuffer);
            get_fn!(cl_finish, b"clFinish\0", PfnClFinish);
            get_fn!(cl_release_mem_object, b"clReleaseMemObject\0", PfnClReleaseMemObject);
            get_fn!(cl_release_kernel, b"clReleaseKernel\0", PfnClReleaseKernel);
            get_fn!(cl_release_program, b"clReleaseProgram\0", PfnClReleaseProgram);
            get_fn!(cl_release_command_queue, b"clReleaseCommandQueue\0", PfnClReleaseCommandQueue);
            get_fn!(cl_release_context, b"clReleaseContext\0", PfnClReleaseContext);

            (Self {
                cl_get_platform_ids,
                cl_get_device_ids,
                cl_get_device_info,
                cl_create_context,
                cl_create_command_queue,
                cl_create_buffer,
                cl_create_program_with_source,
                cl_build_program,
                cl_get_program_build_info,
                cl_create_kernel,
                cl_set_kernel_arg,
                cl_enqueue_nd_range_kernel,
                cl_enqueue_read_buffer,
                cl_enqueue_write_buffer,
                cl_finish,
                cl_release_mem_object,
                cl_release_kernel,
                cl_release_program,
                cl_release_command_queue,
                cl_release_context,
            }, handle)
        }
    }
}

// OpenCL Kernels Source
static GPU_OPENCL_SOURCE: &str = r#"
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
"#;

#[allow(dead_code)]
struct CSRMatrix {
    rows: usize,
    cols: usize,
    nnz: usize,
    row_ptr: Vec<i32>,
    col_idx: Vec<i32>,
    val: Vec<f64>,
}

struct LargeQPProblem {
    n: usize,
    m: usize,
    p: CSRMatrix,
    a: CSRMatrix,
    at: CSRMatrix,
    c: Vec<f64>,
    l_k: Vec<f64>,
    u_k: Vec<f64>,
    l_c: Vec<f64>,
    u_c: Vec<f64>,
}

fn read_i32_vec<R: Read>(reader: &mut R, count: usize) -> Vec<i32> {
    let mut vec = vec![0i32; count];
    let byte_slice = unsafe {
        std::slice::from_raw_parts_mut(vec.as_mut_ptr() as *mut u8, count * 4)
    };
    reader.read_exact(byte_slice).expect("Failed to read i32 array");
    vec
}

fn read_f64_vec<R: Read>(reader: &mut R, count: usize) -> Vec<f64> {
    let mut vec = vec![0.0f64; count];
    let byte_slice = unsafe {
        std::slice::from_raw_parts_mut(vec.as_mut_ptr() as *mut u8, count * 8)
    };
    reader.read_exact(byte_slice).expect("Failed to read f64 array");
    vec
}

fn load_binary_dataset(path: &str) -> LargeQPProblem {
    let f = File::open(path)
        .or_else(|_| File::open(format!("../{}", path)))
        .expect("Could not open dataset file");
    let mut reader = BufReader::new(f);

    let mut u32_buf = [0u8; 4];
    let mut u64_buf = [0u8; 8];

    reader.read_exact(&mut u32_buf).unwrap();
    let n = u32::from_le_bytes(u32_buf) as usize;

    reader.read_exact(&mut u32_buf).unwrap();
    let m = u32::from_le_bytes(u32_buf) as usize;

    reader.read_exact(&mut u64_buf).unwrap();
    let nnz_p = u64::from_le_bytes(u64_buf) as usize;

    reader.read_exact(&mut u64_buf).unwrap();
    let nnz_a = u64::from_le_bytes(u64_buf) as usize;

    reader.read_exact(&mut u64_buf).unwrap();
    let nnz_at = u64::from_le_bytes(u64_buf) as usize;

    let p = CSRMatrix {
        rows: n,
        cols: n,
        nnz: nnz_p,
        row_ptr: read_i32_vec(&mut reader, n + 1),
        col_idx: read_i32_vec(&mut reader, nnz_p),
        val: read_f64_vec(&mut reader, nnz_p),
    };

    let a = CSRMatrix {
        rows: m,
        cols: n,
        nnz: nnz_a,
        row_ptr: read_i32_vec(&mut reader, m + 1),
        col_idx: read_i32_vec(&mut reader, nnz_a),
        val: read_f64_vec(&mut reader, nnz_a),
    };

    let at = CSRMatrix {
        rows: n,
        cols: m,
        nnz: nnz_at,
        row_ptr: read_i32_vec(&mut reader, n + 1),
        col_idx: read_i32_vec(&mut reader, nnz_at),
        val: read_f64_vec(&mut reader, nnz_at),
    };

    let c = read_f64_vec(&mut reader, n);
    let l_k = read_f64_vec(&mut reader, m);
    let u_k = read_f64_vec(&mut reader, m);
    let l_c = read_f64_vec(&mut reader, n);
    let u_c = read_f64_vec(&mut reader, n);

    LargeQPProblem { n, m, p, a, at, c, l_k, u_k, l_c, u_c }
}

fn main() {
    println!("=======================================================================");
    println!("  GPU ACCELERATED HPR-QP IN RUST (NVIDIA GeForce MX250 / OpenCL)");
    println!("  Problem: CVXQP1_L (n = 10,000, m = 15,000, Nonzeros = 94,966)");
    println!("=======================================================================");

    let (api, h_dll) = OpenCLApi::load();

    unsafe {
        // Enumerate platforms
        let mut num_platforms: cl_uint = 0;
        let mut err = (api.cl_get_platform_ids)(0, std::ptr::null_mut(), &mut num_platforms);
        assert_eq!(err, CL_SUCCESS, "clGetPlatformIDs failed");

        let mut platforms = vec![std::ptr::null_mut(); num_platforms as usize];
        (api.cl_get_platform_ids)(num_platforms, platforms.as_mut_ptr(), std::ptr::null_mut());

        let mut _sel_platform: cl_platform_id = std::ptr::null_mut();
        let mut sel_device: cl_device_id = std::ptr::null_mut();
        let mut device_name = String::new();

        // Search for NVIDIA GPU
        for &plat in &platforms {
            let mut num_devs: cl_uint = 0;
            if (api.cl_get_device_ids)(plat, CL_DEVICE_TYPE_GPU, 0, std::ptr::null_mut(), &mut num_devs) == CL_SUCCESS && num_devs > 0 {
                let mut devs = vec![std::ptr::null_mut(); num_devs as usize];
                (api.cl_get_device_ids)(plat, CL_DEVICE_TYPE_GPU, num_devs, devs.as_mut_ptr(), std::ptr::null_mut());
                for &dev in &devs {
                    let mut name_buf = [0u8; 256];
                    (api.cl_get_device_info)(dev, CL_DEVICE_NAME, name_buf.len(), name_buf.as_mut_ptr() as *mut c_void, std::ptr::null_mut());
                    let dname = String::from_utf8_lossy(&name_buf).trim_matches(char::from(0)).to_string();
                    if dname.contains("NVIDIA") || dname.contains("GeForce") {
                        _sel_platform = plat;
                        sel_device = dev;
                        device_name = dname;
                        break;
                    }
                }
            }
            if !sel_device.is_null() { break; }
        }

        if sel_device.is_null() {
            panic!("Could not find suitable NVIDIA OpenCL GPU device.");
        }

        println!("Selected GPU Device   : {}", device_name);

        let prob = load_binary_dataset("Data/cvxqp1_l.bin");
        println!("Dataset Loaded in RAM : n={}, m={}, nnz={}", prob.n, prob.m, prob.p.nnz + prob.a.nnz);

        let ctx = (api.cl_create_context)(std::ptr::null(), 1, &sel_device, std::ptr::null_mut(), std::ptr::null_mut(), &mut err);
        assert_eq!(err, CL_SUCCESS, "clCreateContext failed");

        let queue = (api.cl_create_command_queue)(ctx, sel_device, 0, &mut err);
        assert_eq!(err, CL_SUCCESS, "clCreateCommandQueue failed");

        // JIT Compile OpenCL Kernels
        let src_c = CString::new(GPU_OPENCL_SOURCE).unwrap();
        let src_ptr = src_c.as_ptr() as *const u8;
        let src_len = src_c.as_bytes().len();
        let prog = (api.cl_create_program_with_source)(ctx, 1, &src_ptr, &src_len, &mut err);
        assert_eq!(err, CL_SUCCESS, "clCreateProgramWithSource failed");

        let build_opts = CString::new("-cl-fast-relaxed-math -cl-mad-enable").unwrap();
        err = (api.cl_build_program)(prog, 1, &sel_device, build_opts.as_ptr() as *const u8, std::ptr::null_mut(), std::ptr::null_mut());
        if err != CL_SUCCESS {
            let mut log_size: usize = 0;
            (api.cl_get_program_build_info)(prog, sel_device, CL_PROGRAM_BUILD_LOG, 0, std::ptr::null_mut(), &mut log_size);
            let mut log_buf = vec![0u8; log_size];
            (api.cl_get_program_build_info)(prog, sel_device, CL_PROGRAM_BUILD_LOG, log_size, log_buf.as_mut_ptr() as *mut c_void, std::ptr::null_mut());
            eprintln!("Build Log:\n{}", String::from_utf8_lossy(&log_buf));
            panic!("clBuildProgram failed.");
        }

        let k_spmv     = (api.cl_create_kernel)(prog, b"spmv_csr_kernel\0".as_ptr(), &mut err);
        let k_step1_2  = (api.cl_create_kernel)(prog, b"fused_step1_2_kernel\0".as_ptr(), &mut err);
        let k_sgs_prep = (api.cl_create_kernel)(prog, b"sgs_prep_temp_n_kernel\0".as_ptr(), &mut err);
        let k_sgs_y    = (api.cl_create_kernel)(prog, b"sgs_y_step_kernel\0".as_ptr(), &mut err);
        let k_sgs_w    = (api.cl_create_kernel)(prog, b"sgs_w_step_kernel\0".as_ptr(), &mut err);
        let k_halpern  = (api.cl_create_kernel)(prog, b"halpern_average_kernel\0".as_ptr(), &mut err);

        let n = prob.n as u32;
        let m = prob.m as u32;

        // Allocate VRAM Buffers
        let create_buf = |flags: cl_ulong, bytes: usize, host_ptr: *const c_void| -> cl_mem {
            let mut e = 0;
            let m = (api.cl_create_buffer)(ctx, flags, bytes, host_ptr as *mut c_void, &mut e);
            assert_eq!(e, CL_SUCCESS, "clCreateBuffer failed");
            m
        };

        let d_p_row = create_buf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (prob.n + 1) * 4, prob.p.row_ptr.as_ptr() as *const c_void);
        let d_p_col = create_buf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, prob.p.nnz * 4, prob.p.col_idx.as_ptr() as *const c_void);
        let d_p_val = create_buf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, prob.p.nnz * 8, prob.p.val.as_ptr() as *const c_void);

        let d_a_row = create_buf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (prob.m + 1) * 4, prob.a.row_ptr.as_ptr() as *const c_void);
        let d_a_col = create_buf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, prob.a.nnz * 4, prob.a.col_idx.as_ptr() as *const c_void);
        let d_a_val = create_buf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, prob.a.nnz * 8, prob.a.val.as_ptr() as *const c_void);

        let d_at_row = create_buf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (prob.n + 1) * 4, prob.at.row_ptr.as_ptr() as *const c_void);
        let d_at_col = create_buf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, prob.at.nnz * 4, prob.at.col_idx.as_ptr() as *const c_void);
        let d_at_val = create_buf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, prob.at.nnz * 8, prob.at.val.as_ptr() as *const c_void);

        let d_c = create_buf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, prob.n * 8, prob.c.as_ptr() as *const c_void);
        let d_l_k = create_buf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, prob.m * 8, prob.l_k.as_ptr() as *const c_void);
        let d_u_k = create_buf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, prob.m * 8, prob.u_k.as_ptr() as *const c_void);
        let d_l_c = create_buf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, prob.n * 8, prob.l_c.as_ptr() as *const c_void);
        let d_u_c = create_buf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, prob.n * 8, prob.u_c.as_ptr() as *const c_void);

        let d_x = create_buf(CL_MEM_READ_WRITE, prob.n * 8, std::ptr::null());
        let d_w_q = create_buf(CL_MEM_READ_WRITE, prob.n * 8, std::ptr::null());
        let d_z = create_buf(CL_MEM_READ_WRITE, prob.n * 8, std::ptr::null());
        let d_y = create_buf(CL_MEM_READ_WRITE, prob.m * 8, std::ptr::null());

        let d_x_bar = create_buf(CL_MEM_READ_WRITE, prob.n * 8, std::ptr::null());
        let d_w_bar = create_buf(CL_MEM_READ_WRITE, prob.n * 8, std::ptr::null());
        let d_z_bar = create_buf(CL_MEM_READ_WRITE, prob.n * 8, std::ptr::null());
        let d_y_bar = create_buf(CL_MEM_READ_WRITE, prob.m * 8, std::ptr::null());

        let d_x_hat = create_buf(CL_MEM_READ_WRITE, prob.n * 8, std::ptr::null());
        let d_w_half = create_buf(CL_MEM_READ_WRITE, prob.n * 8, std::ptr::null());
        let d_r_y = create_buf(CL_MEM_READ_WRITE, prob.m * 8, std::ptr::null());

        let d_qw = create_buf(CL_MEM_READ_WRITE, prob.n * 8, std::ptr::null());
        let d_qw_half = create_buf(CL_MEM_READ_WRITE, prob.n * 8, std::ptr::null());
        let d_aty = create_buf(CL_MEM_READ_WRITE, prob.n * 8, std::ptr::null());
        let d_temp_n = create_buf(CL_MEM_READ_WRITE, prob.n * 8, std::ptr::null());
        let d_temp_m = create_buf(CL_MEM_READ_WRITE, prob.m * 8, std::ptr::null());

        let d_x_r0 = create_buf(CL_MEM_READ_WRITE, prob.n * 8, std::ptr::null());
        let d_w_r0 = create_buf(CL_MEM_READ_WRITE, prob.n * 8, std::ptr::null());
        let d_z_r0 = create_buf(CL_MEM_READ_WRITE, prob.n * 8, std::ptr::null());
        let d_y_r0 = create_buf(CL_MEM_READ_WRITE, prob.m * 8, std::ptr::null());

        println!("Allocated 24 Working Buffers in GPU VRAM (Zero PCIe transfer overhead during iterations)\n");

        let zero_n = vec![0.0f64; prob.n];
        let zero_m = vec![0.0f64; prob.m];

        let sigma = 1.0f64;
        let lam_q = 9.17e4f64;
        let lam_a = 2.17e5f64;
        let denom_w = 1.0 + sigma * lam_q;
        let sig_lam_a = sigma * lam_a;
        let inv_sig_lam_a = 1.0 / sig_lam_a;
        let w_step = sigma / denom_w;

        let global_n = ((prob.n + 63) / 64) * 64;
        let global_m = ((prob.m + 63) / 64) * 64;
        let local_sz = 64usize;

        for &iters in &[100, 300] {
            println!("--- Benchmarking {} Iterations on GPU in Rust ---", iters);

            // Zero initial working vectors
            (api.cl_enqueue_write_buffer)(queue, d_x, CL_TRUE, 0, prob.n * 8, zero_n.as_ptr() as *const c_void, 0, std::ptr::null(), std::ptr::null_mut());
            (api.cl_enqueue_write_buffer)(queue, d_w_q, CL_TRUE, 0, prob.n * 8, zero_n.as_ptr() as *const c_void, 0, std::ptr::null(), std::ptr::null_mut());
            (api.cl_enqueue_write_buffer)(queue, d_z, CL_TRUE, 0, prob.n * 8, zero_n.as_ptr() as *const c_void, 0, std::ptr::null(), std::ptr::null_mut());
            (api.cl_enqueue_write_buffer)(queue, d_y, CL_TRUE, 0, prob.m * 8, zero_m.as_ptr() as *const c_void, 0, std::ptr::null(), std::ptr::null_mut());
            (api.cl_enqueue_write_buffer)(queue, d_x_r0, CL_TRUE, 0, prob.n * 8, zero_n.as_ptr() as *const c_void, 0, std::ptr::null(), std::ptr::null_mut());
            (api.cl_enqueue_write_buffer)(queue, d_w_r0, CL_TRUE, 0, prob.n * 8, zero_n.as_ptr() as *const c_void, 0, std::ptr::null(), std::ptr::null_mut());
            (api.cl_enqueue_write_buffer)(queue, d_z_r0, CL_TRUE, 0, prob.n * 8, zero_n.as_ptr() as *const c_void, 0, std::ptr::null(), std::ptr::null_mut());
            (api.cl_enqueue_write_buffer)(queue, d_y_r0, CL_TRUE, 0, prob.m * 8, zero_m.as_ptr() as *const c_void, 0, std::ptr::null(), std::ptr::null_mut());
            (api.cl_finish)(queue);

            let t_start = Instant::now();
            let mut inner_t = 0usize;

            for _k in 0..iters {
                inner_t += 1;

                // 1) SpMV: Q * w_q -> qw
                (api.cl_set_kernel_arg)(k_spmv, 0, 4, &n as *const u32 as *const c_void);
                (api.cl_set_kernel_arg)(k_spmv, 1, std::mem::size_of::<cl_mem>(), &d_p_row as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_spmv, 2, std::mem::size_of::<cl_mem>(), &d_p_col as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_spmv, 3, std::mem::size_of::<cl_mem>(), &d_p_val as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_spmv, 4, std::mem::size_of::<cl_mem>(), &d_w_q as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_spmv, 5, std::mem::size_of::<cl_mem>(), &d_qw as *const cl_mem as *const c_void);
                (api.cl_enqueue_nd_range_kernel)(queue, k_spmv, 1, std::ptr::null(), &global_n, &local_sz, 0, std::ptr::null(), std::ptr::null_mut());

                // SpMV: AT * y -> aty
                (api.cl_set_kernel_arg)(k_spmv, 0, 4, &n as *const u32 as *const c_void);
                (api.cl_set_kernel_arg)(k_spmv, 1, std::mem::size_of::<cl_mem>(), &d_at_row as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_spmv, 2, std::mem::size_of::<cl_mem>(), &d_at_col as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_spmv, 3, std::mem::size_of::<cl_mem>(), &d_at_val as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_spmv, 4, std::mem::size_of::<cl_mem>(), &d_y as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_spmv, 5, std::mem::size_of::<cl_mem>(), &d_aty as *const cl_mem as *const c_void);
                (api.cl_enqueue_nd_range_kernel)(queue, k_spmv, 1, std::ptr::null(), &global_n, &local_sz, 0, std::ptr::null(), std::ptr::null_mut());

                // 2) Fused Step 1 & 2
                (api.cl_set_kernel_arg)(k_step1_2, 0, 4, &n as *const u32 as *const c_void);
                (api.cl_set_kernel_arg)(k_step1_2, 1, 8, &sigma as *const f64 as *const c_void);
                (api.cl_set_kernel_arg)(k_step1_2, 2, 8, &denom_w as *const f64 as *const c_void);
                (api.cl_set_kernel_arg)(k_step1_2, 3, 8, &lam_q as *const f64 as *const c_void);
                (api.cl_set_kernel_arg)(k_step1_2, 4, std::mem::size_of::<cl_mem>(), &d_x as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_step1_2, 5, std::mem::size_of::<cl_mem>(), &d_qw as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_step1_2, 6, std::mem::size_of::<cl_mem>(), &d_aty as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_step1_2, 7, std::mem::size_of::<cl_mem>(), &d_c as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_step1_2, 8, std::mem::size_of::<cl_mem>(), &d_l_c as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_step1_2, 9, std::mem::size_of::<cl_mem>(), &d_u_c as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_step1_2, 10, std::mem::size_of::<cl_mem>(), &d_w_q as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_step1_2, 11, std::mem::size_of::<cl_mem>(), &d_x_bar as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_step1_2, 12, std::mem::size_of::<cl_mem>(), &d_z_bar as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_step1_2, 13, std::mem::size_of::<cl_mem>(), &d_x_hat as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_step1_2, 14, std::mem::size_of::<cl_mem>(), &d_w_half as *const cl_mem as *const c_void);
                (api.cl_enqueue_nd_range_kernel)(queue, k_step1_2, 1, std::ptr::null(), &global_n, &local_sz, 0, std::ptr::null(), std::ptr::null_mut());

                // 3) Step 3: sGS
                // Q * w_half -> qw_half
                (api.cl_set_kernel_arg)(k_spmv, 0, 4, &n as *const u32 as *const c_void);
                (api.cl_set_kernel_arg)(k_spmv, 1, std::mem::size_of::<cl_mem>(), &d_p_row as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_spmv, 2, std::mem::size_of::<cl_mem>(), &d_p_col as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_spmv, 3, std::mem::size_of::<cl_mem>(), &d_p_val as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_spmv, 4, std::mem::size_of::<cl_mem>(), &d_w_half as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_spmv, 5, std::mem::size_of::<cl_mem>(), &d_qw_half as *const cl_mem as *const c_void);
                (api.cl_enqueue_nd_range_kernel)(queue, k_spmv, 1, std::ptr::null(), &global_n, &local_sz, 0, std::ptr::null(), std::ptr::null_mut());

                // Prep temp_n
                (api.cl_set_kernel_arg)(k_sgs_prep, 0, 4, &n as *const u32 as *const c_void);
                (api.cl_set_kernel_arg)(k_sgs_prep, 1, 8, &sigma as *const f64 as *const c_void);
                (api.cl_set_kernel_arg)(k_sgs_prep, 2, std::mem::size_of::<cl_mem>(), &d_x_hat as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_sgs_prep, 3, std::mem::size_of::<cl_mem>(), &d_qw as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_sgs_prep, 4, std::mem::size_of::<cl_mem>(), &d_qw_half as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_sgs_prep, 5, std::mem::size_of::<cl_mem>(), &d_temp_n as *const cl_mem as *const c_void);
                (api.cl_enqueue_nd_range_kernel)(queue, k_sgs_prep, 1, std::ptr::null(), &global_n, &local_sz, 0, std::ptr::null(), std::ptr::null_mut());

                // A * temp_n -> r_y
                (api.cl_set_kernel_arg)(k_spmv, 0, 4, &m as *const u32 as *const c_void);
                (api.cl_set_kernel_arg)(k_spmv, 1, std::mem::size_of::<cl_mem>(), &d_a_row as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_spmv, 2, std::mem::size_of::<cl_mem>(), &d_a_col as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_spmv, 3, std::mem::size_of::<cl_mem>(), &d_a_val as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_spmv, 4, std::mem::size_of::<cl_mem>(), &d_temp_n as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_spmv, 5, std::mem::size_of::<cl_mem>(), &d_r_y as *const cl_mem as *const c_void);
                (api.cl_enqueue_nd_range_kernel)(queue, k_spmv, 1, std::ptr::null(), &global_m, &local_sz, 0, std::ptr::null(), std::ptr::null_mut());

                // sGS y step
                (api.cl_set_kernel_arg)(k_sgs_y, 0, 4, &m as *const u32 as *const c_void);
                (api.cl_set_kernel_arg)(k_sgs_y, 1, 8, &sig_lam_a as *const f64 as *const c_void);
                (api.cl_set_kernel_arg)(k_sgs_y, 2, 8, &inv_sig_lam_a as *const f64 as *const c_void);
                (api.cl_set_kernel_arg)(k_sgs_y, 3, std::mem::size_of::<cl_mem>(), &d_l_k as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_sgs_y, 4, std::mem::size_of::<cl_mem>(), &d_u_k as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_sgs_y, 5, std::mem::size_of::<cl_mem>(), &d_y as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_sgs_y, 6, std::mem::size_of::<cl_mem>(), &d_r_y as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_sgs_y, 7, std::mem::size_of::<cl_mem>(), &d_y_bar as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_sgs_y, 8, std::mem::size_of::<cl_mem>(), &d_temp_m as *const cl_mem as *const c_void);
                (api.cl_enqueue_nd_range_kernel)(queue, k_sgs_y, 1, std::ptr::null(), &global_m, &local_sz, 0, std::ptr::null(), std::ptr::null_mut());

                // AT * temp_m -> temp_n
                (api.cl_set_kernel_arg)(k_spmv, 0, 4, &n as *const u32 as *const c_void);
                (api.cl_set_kernel_arg)(k_spmv, 1, std::mem::size_of::<cl_mem>(), &d_at_row as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_spmv, 2, std::mem::size_of::<cl_mem>(), &d_at_col as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_spmv, 3, std::mem::size_of::<cl_mem>(), &d_at_val as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_spmv, 4, std::mem::size_of::<cl_mem>(), &d_temp_m as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_spmv, 5, std::mem::size_of::<cl_mem>(), &d_temp_n as *const cl_mem as *const c_void);
                (api.cl_enqueue_nd_range_kernel)(queue, k_spmv, 1, std::ptr::null(), &global_n, &local_sz, 0, std::ptr::null(), std::ptr::null_mut());

                // sGS w step
                (api.cl_set_kernel_arg)(k_sgs_w, 0, 4, &n as *const u32 as *const c_void);
                (api.cl_set_kernel_arg)(k_sgs_w, 1, 8, &w_step as *const f64 as *const c_void);
                (api.cl_set_kernel_arg)(k_sgs_w, 2, std::mem::size_of::<cl_mem>(), &d_w_half as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_sgs_w, 3, std::mem::size_of::<cl_mem>(), &d_temp_n as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_sgs_w, 4, std::mem::size_of::<cl_mem>(), &d_w_bar as *const cl_mem as *const c_void);
                (api.cl_enqueue_nd_range_kernel)(queue, k_sgs_w, 1, std::ptr::null(), &global_n, &local_sz, 0, std::ptr::null(), std::ptr::null_mut());

                // 4) Halpern Averaging
                let coef_0 = 1.0f64 / (inner_t as f64 + 1.0);
                let coef_hat = inner_t as f64 / (inner_t as f64 + 1.0);
                (api.cl_set_kernel_arg)(k_halpern, 0, 4, &n as *const u32 as *const c_void);
                (api.cl_set_kernel_arg)(k_halpern, 1, 4, &m as *const u32 as *const c_void);
                (api.cl_set_kernel_arg)(k_halpern, 2, 8, &coef_0 as *const f64 as *const c_void);
                (api.cl_set_kernel_arg)(k_halpern, 3, 8, &coef_hat as *const f64 as *const c_void);
                (api.cl_set_kernel_arg)(k_halpern, 4, std::mem::size_of::<cl_mem>(), &d_x_r0 as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_halpern, 5, std::mem::size_of::<cl_mem>(), &d_w_r0 as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_halpern, 6, std::mem::size_of::<cl_mem>(), &d_z_r0 as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_halpern, 7, std::mem::size_of::<cl_mem>(), &d_y_r0 as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_halpern, 8, std::mem::size_of::<cl_mem>(), &d_x_bar as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_halpern, 9, std::mem::size_of::<cl_mem>(), &d_w_bar as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_halpern, 10, std::mem::size_of::<cl_mem>(), &d_z_bar as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_halpern, 11, std::mem::size_of::<cl_mem>(), &d_y_bar as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_halpern, 12, std::mem::size_of::<cl_mem>(), &d_x as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_halpern, 13, std::mem::size_of::<cl_mem>(), &d_w_q as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_halpern, 14, std::mem::size_of::<cl_mem>(), &d_z as *const cl_mem as *const c_void);
                (api.cl_set_kernel_arg)(k_halpern, 15, std::mem::size_of::<cl_mem>(), &d_y as *const cl_mem as *const c_void);
                (api.cl_enqueue_nd_range_kernel)(queue, k_halpern, 1, std::ptr::null(), &global_n, &local_sz, 0, std::ptr::null(), std::ptr::null_mut());
            }

            (api.cl_finish)(queue);
            let elapsed = t_start.elapsed();
            let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
            let ms_per_iter = elapsed_ms / (iters as f64);

            let mut x_host = vec![0.0f64; prob.n];
            (api.cl_enqueue_read_buffer)(queue, d_x_bar, CL_TRUE, 0, prob.n * 8, x_host.as_mut_ptr() as *mut c_void, 0, std::ptr::null(), std::ptr::null_mut());

            println!("  [New GPU Rust on NVIDIA MX250] Total: {:7.2} ms | Latency: {:.3} ms/iter", elapsed_ms, ms_per_iter);
            println!("  Final Optimal x[0..2] on GPU: [{:.4}, {:.4}, {:.4}]\n", x_host[0], x_host[1], x_host[2]);
        }

        // Cleanup
        let mem_bufs = [
            d_p_row, d_p_col, d_p_val, d_a_row, d_a_col, d_a_val, d_at_row, d_at_col, d_at_val,
            d_c, d_l_k, d_u_k, d_l_c, d_u_c,
            d_x, d_w_q, d_z, d_y, d_x_bar, d_w_bar, d_z_bar, d_y_bar,
            d_x_hat, d_w_half, d_r_y, d_qw, d_qw_half, d_aty, d_temp_n, d_temp_m,
            d_x_r0, d_w_r0, d_z_r0, d_y_r0,
        ];
        for mem in mem_bufs {
            (api.cl_release_mem_object)(mem);
        }

        (api.cl_release_kernel)(k_spmv);
        (api.cl_release_kernel)(k_step1_2);
        (api.cl_release_kernel)(k_sgs_prep);
        (api.cl_release_kernel)(k_sgs_y);
        (api.cl_release_kernel)(k_sgs_w);
        (api.cl_release_kernel)(k_halpern);
        (api.cl_release_program)(prog);
        (api.cl_release_command_queue)(queue);
        (api.cl_release_context)(ctx);
        FreeLibrary(h_dll);
    }

    println!("=======================================================================");
}
