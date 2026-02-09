use cudarc::driver::{CudaDevice, CudaSlice};
use cudarc::cublas::{CudaBlas as CudarcBlas, Gemm, GemmConfig};
use std::sync::Arc;
use half::f16;

use crate::tensor::{TensorHandle, TensorData, Precision};

/// Status codes for BLAS operations
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlasStatus {
    Success = 0,
    InitError = 1,
    ComputeError = 2,
    InvalidParameter = 3,
    PrecisionMismatch = 4,  // Phase 1.2: 新增
}

/// cuBLAS context
pub struct BlasContext {
    pub device: Arc<CudaDevice>,
    pub cublas: CudarcBlas,
}

// ============================================================================
// Context Management
// ============================================================================

#[no_mangle]
pub extern "C" fn blas_init(device_handle: usize) -> *mut BlasContext {
    if device_handle == 0 {
        eprintln!("blas_init: null device handle");
        return std::ptr::null_mut();
    }

    let device = unsafe {
        let ptr = device_handle as *const Arc<CudaDevice>;
        if ptr.is_null() {
            eprintln!("blas_init: invalid device handle");
            return std::ptr::null_mut();
        }
        Arc::clone(&*ptr)
    };

    // 直接建構，失敗時返回 null
    match CudarcBlas::new(device.clone()) {
        Ok(cublas) => {
            let ctx = BlasContext { device, cublas };
            Box::into_raw(Box::new(ctx))
        }
        Err(e) => {
            eprintln!("blas_init error: {:?}", e);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn blas_destroy(ctx: *mut BlasContext) {
    if !ctx.is_null() {
        unsafe {
            let _ = Box::from_raw(ctx);
        }
    }
}

// ============================================================================
// Phase 1.2: 輔助函數
// ============================================================================

/// 從 TensorHandle 取得精度
fn get_tensor_precision(handle: usize) -> Option<Precision> {
    if handle == 0 {
        return None;
    }
    let tensor = unsafe { &*(handle as *const TensorHandle) };
    Some(tensor.dtype)
}

/// 檢查三個 tensor 的精度是否一致
fn check_precision_match(a_handle: usize, b_handle: usize, c_handle: usize) -> Result<Precision, BlasStatus> {
    let a_prec = get_tensor_precision(a_handle).ok_or(BlasStatus::InvalidParameter)?;
    let b_prec = get_tensor_precision(b_handle).ok_or(BlasStatus::InvalidParameter)?;
    let c_prec = get_tensor_precision(c_handle).ok_or(BlasStatus::InvalidParameter)?;
    
    if a_prec != b_prec || b_prec != c_prec {
        return Err(BlasStatus::PrecisionMismatch);
    }
    
    Ok(a_prec)
}

/// 從 TensorHandle 取得 FP32 的 CudaSlice 指標
fn get_tensor_slice_f32(handle: usize) -> Option<&'static CudaSlice<f32>> {
    if handle == 0 {
        return None;
    }
    let tensor = unsafe { &*(handle as *const TensorHandle) };
    match &tensor.data {
        TensorData::FP32(slice) => Some(unsafe { &*(slice as *const _) }),
        _ => None,
    }
}

fn get_tensor_slice_f32_mut(handle: usize) -> Option<&'static mut CudaSlice<f32>> {
    if handle == 0 {
        return None;
    }
    let tensor = unsafe { &mut *(handle as *mut TensorHandle) };
    match &mut tensor.data {
        TensorData::FP32(slice) => Some(unsafe { &mut *(slice as *mut _) }),
        _ => None,
    }
}

/// 從 TensorHandle 取得 FP16 的 CudaSlice 指標
fn get_tensor_slice_f16(handle: usize) -> Option<&'static CudaSlice<f16>> {
    if handle == 0 {
        return None;
    }
    let tensor = unsafe { &*(handle as *const TensorHandle) };
    match &tensor.data {
        TensorData::FP16(slice) => Some(unsafe { &*(slice as *const _) }),
        _ => None,
    }
}

fn get_tensor_slice_f16_mut(handle: usize) -> Option<&'static mut CudaSlice<f16>> {
    if handle == 0 {
        return None;
    }
    let tensor = unsafe { &mut *(handle as *mut TensorHandle) };
    match &mut tensor.data {
        TensorData::FP16(slice) => Some(unsafe { &mut *(slice as *mut _) }),
        _ => None,
    }
}

/// 從 TensorHandle 取得 FP64 的 CudaSlice 指標
fn get_tensor_slice_f64(handle: usize) -> Option<&'static CudaSlice<f64>> {
    if handle == 0 {
        return None;
    }
    let tensor = unsafe { &*(handle as *const TensorHandle) };
    match &tensor.data {
        TensorData::FP64(slice) => Some(unsafe { &*(slice as *const _) }),
        _ => None,
    }
}

fn get_tensor_slice_f64_mut(handle: usize) -> Option<&'static mut CudaSlice<f64>> {
    if handle == 0 {
        return None;
    }
    let tensor = unsafe { &mut *(handle as *mut TensorHandle) };
    match &mut tensor.data {
        TensorData::FP64(slice) => Some(unsafe { &mut *(slice as *mut _) }),
        _ => None,
    }
}

// ============================================================================
// Phase 1.2: 統一的 GEMM 介面
// ============================================================================

/// 統一的 GEMM 介面（根據 CudaTensor 精度自動分發）
///
/// # C ABI
/// ```c
/// int blas_gemm(void* ctx, int m, int n, int k, 
///               double alpha, size_t a_handle, size_t b_handle,
///               double beta, size_t c_handle);
/// ```
#[no_mangle]
pub extern "C" fn blas_gemm(
    ctx: *mut BlasContext,
    m: i32,
    n: i32,
    k: i32,
    alpha: f64,
    a_handle: usize,
    b_handle: usize,
    beta: f64,
    c_handle: usize,
) -> BlasStatus {
    // 參數驗證
    if ctx.is_null() || a_handle == 0 || b_handle == 0 || c_handle == 0 {
        return BlasStatus::InvalidParameter;
    }
    
    if m <= 0 || n <= 0 || k <= 0 {
        return BlasStatus::InvalidParameter;
    }

    // 檢查精度是否一致
    let precision = match check_precision_match(a_handle, b_handle, c_handle) {
        Ok(p) => p,
        Err(status) => return status,
    };

    let ctx = unsafe { &mut *ctx };

    // 根據精度分發到對應的實作
    match precision {
        Precision::FP32 => blas_sgemm_internal(ctx, m, n, k, alpha as f32, a_handle, b_handle, beta as f32, c_handle),
        Precision::FP16 => blas_hgemm_internal(ctx, m, n, k, alpha as f32, a_handle, b_handle, beta as f32, c_handle),
        Precision::FP64 => blas_dgemm_internal(ctx, m, n, k, alpha, a_handle, b_handle, beta, c_handle),
        Precision::BF16 => {
            eprintln!("blas_gemm: BF16 not yet implemented");
            BlasStatus::ComputeError
        }
        _ => {
            eprintln!("blas_gemm: Unsupported precision {:?}", precision);
            BlasStatus::InvalidParameter
        }
    }
}

// ============================================================================
// BLAS Level 3: GEMM 內部實作
// ============================================================================

/// FP32 GEMM 內部實作（使用 TensorHandle）
fn blas_sgemm_internal(
    ctx: &mut BlasContext,
    m: i32,
    n: i32,
    k: i32,
    alpha: f32,
    a_handle: usize,
    b_handle: usize,
    beta: f32,
    c_handle: usize,
) -> BlasStatus {
    let a_dev = match get_tensor_slice_f32(a_handle) {
        Some(s) => s,
        None => return BlasStatus::InvalidParameter,
    };
    let b_dev = match get_tensor_slice_f32(b_handle) {
        Some(s) => s,
        None => return BlasStatus::InvalidParameter,
    };
    let c_dev = match get_tensor_slice_f32_mut(c_handle) {
        Some(s) => s,
        None => return BlasStatus::InvalidParameter,
    };

    let cfg = GemmConfig {
        transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        m: n,
        n: m,
        k: k,
        alpha,
        lda: n,
        ldb: k,
        beta,
        ldc: n,
    };

    match unsafe { ctx.cublas.gemm(cfg, b_dev, a_dev, c_dev) } {
        Ok(_) => {
            let _ = ctx.device.synchronize();
            BlasStatus::Success
        }
        Err(e) => {
            eprintln!("blas_sgemm_internal error: {:?}", e);
            BlasStatus::ComputeError
        }
    }
}

/// FP16 GEMM 內部實作（Phase 1.2: 透過 FP32 轉換）
/// 
/// 注意：這是暫時的 CPU 轉換方案，Phase 1.3 會優化為純 GPU 實作（使用 cublasGemmEx）
fn blas_hgemm_internal(
    ctx: &mut BlasContext,
    m: i32,
    n: i32,
    k: i32,
    alpha: f32,
    a_handle: usize,
    b_handle: usize,
    beta: f32,
    c_handle: usize,
) -> BlasStatus {
    match try_hgemm_via_f32(ctx, m, n, k, alpha, a_handle, b_handle, beta, c_handle) {
        Ok(_) => BlasStatus::Success,
        Err(e) => {
            eprintln!("blas_hgemm_internal error: {:?}", e);
            BlasStatus::ComputeError
        }
    }
}

/// FP16 GEMM 實作（透過 FP32 轉換）
fn try_hgemm_via_f32(
    ctx: &mut BlasContext,
    m: i32,
    n: i32,
    k: i32,
    alpha: f32,
    a_handle: usize,
    b_handle: usize,
    beta: f32,
    c_handle: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let m_usize = m as usize;
    let n_usize = n as usize;
    let k_usize = k as usize;

    // 取得 FP16 slices
    let a_f16 = get_tensor_slice_f16(a_handle).ok_or("Invalid A tensor")?;
    let b_f16 = get_tensor_slice_f16(b_handle).ok_or("Invalid B tensor")?;
    let c_f16 = get_tensor_slice_f16_mut(c_handle).ok_or("Invalid C tensor")?;

    // Step 1: D2H - 將 FP16 資料從 GPU 複製到 CPU
    let mut a_host_f16 = vec![f16::from_f32(0.0); m_usize * k_usize];
    let mut b_host_f16 = vec![f16::from_f32(0.0); k_usize * n_usize];
    let mut c_host_f16 = vec![f16::from_f32(0.0); m_usize * n_usize];

    ctx.device.dtoh_sync_copy_into(a_f16, &mut a_host_f16)?;
    ctx.device.dtoh_sync_copy_into(b_f16, &mut b_host_f16)?;
    ctx.device.dtoh_sync_copy_into(c_f16, &mut c_host_f16)?;

    // Step 2: 轉換 FP16 → FP32
    let a_f32: Vec<f32> = a_host_f16.iter().map(|x| x.to_f32()).collect();
    let b_f32: Vec<f32> = b_host_f16.iter().map(|x| x.to_f32()).collect();
    let mut c_f32: Vec<f32> = c_host_f16.iter().map(|x| x.to_f32()).collect();

    // Step 3: H2D - 將 FP32 資料複製到 GPU
    let a_dev_f32 = ctx.device.htod_sync_copy(&a_f32)?;
    let b_dev_f32 = ctx.device.htod_sync_copy(&b_f32)?;
    let mut c_dev_f32 = ctx.device.htod_sync_copy(&c_f32)?;

    // Step 4: 執行 FP32 GEMM
    let cfg = GemmConfig {
        transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        alpha,
        lda: n as i32,
        ldb: k as i32,
        beta,
        ldc: n as i32,
    };

    unsafe {
        ctx.cublas.gemm(cfg, &b_dev_f32, &a_dev_f32, &mut c_dev_f32)?;
    }
    ctx.device.synchronize()?;

    // Step 5: D2H - 將 FP32 結果複製回 CPU
    ctx.device.dtoh_sync_copy_into(&c_dev_f32, &mut c_f32)?;

    // Step 6: 轉換 FP32 → FP16
    let c_result_f16: Vec<f16> = c_f32.iter().map(|&x| f16::from_f32(x)).collect();

    // Step 7: H2D - 將 FP16 結果寫回 GPU
    ctx.device.htod_copy_into(c_result_f16, c_f16)?;
    ctx.device.synchronize()?;

    Ok(())
}

/// FP64 GEMM 內部實作（使用 TensorHandle）
fn blas_dgemm_internal(
    ctx: &mut BlasContext,
    m: i32,
    n: i32,
    k: i32,
    alpha: f64,
    a_handle: usize,
    b_handle: usize,
    beta: f64,
    c_handle: usize,
) -> BlasStatus {
    let a_dev = match get_tensor_slice_f64(a_handle) {
        Some(s) => s,
        None => return BlasStatus::InvalidParameter,
    };
    let b_dev = match get_tensor_slice_f64(b_handle) {
        Some(s) => s,
        None => return BlasStatus::InvalidParameter,
    };
    let c_dev = match get_tensor_slice_f64_mut(c_handle) {
        Some(s) => s,
        None => return BlasStatus::InvalidParameter,
    };

    let cfg = GemmConfig {
        transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        m: n,
        n: m,
        k: k,
        alpha,
        lda: n,
        ldb: k,
        beta,
        ldc: n,
    };

    match unsafe { ctx.cublas.gemm(cfg, b_dev, a_dev, c_dev) } {
        Ok(_) => {
            let _ = ctx.device.synchronize();
            BlasStatus::Success
        }
        Err(e) => {
            eprintln!("blas_dgemm_internal error: {:?}", e);
            BlasStatus::ComputeError
        }
    }
}

// ============================================================================
// 舊版 API（保持向後相容）
// ============================================================================

#[no_mangle]
pub extern "C" fn blas_sgemm(
    ctx: *mut BlasContext,
    m: i32,
    n: i32,
    k: i32,
    alpha: f32,
    a_ptr: usize,
    b_ptr: usize,
    beta: f32,
    c_ptr: usize,
) -> BlasStatus {
    if ctx.is_null() || a_ptr == 0 || b_ptr == 0 || c_ptr == 0 {
        return BlasStatus::InvalidParameter;
    }

    if m <= 0 || n <= 0 || k <= 0 {
        return BlasStatus::InvalidParameter;
    }

    let ctx = unsafe { &mut *ctx };

    match try_sgemm(ctx, m, n, k, alpha, a_ptr, b_ptr, beta, c_ptr) {
        Ok(_) => BlasStatus::Success,
        Err(e) => {
            eprintln!("blas_sgemm error: {:?}", e);
            BlasStatus::ComputeError
        }
    }
}

fn try_sgemm(
    ctx: &mut BlasContext,
    m: i32,
    n: i32,
    k: i32,
    alpha: f32,
    a_ptr: usize,
    b_ptr: usize,
    beta: f32,
    c_ptr: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let a_dev = unsafe { &*(a_ptr as *const CudaSlice<f32>) };
    let b_dev = unsafe { &*(b_ptr as *const CudaSlice<f32>) };
    let c_dev = unsafe { &mut *(c_ptr as *mut CudaSlice<f32>) };

    let cfg = GemmConfig {
        transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        m: n,
        n: m,
        k: k,
        alpha,
        lda: n,
        ldb: k,
        beta,
        ldc: n,
    };

    unsafe {
        ctx.cublas.gemm(cfg, b_dev, a_dev, c_dev)?;
    }

    ctx.device.synchronize()?;

    Ok(())
}

#[no_mangle]
pub extern "C" fn blas_hgemm(
    _ctx: *mut BlasContext,
    _m: i32,
    _n: i32,
    _k: i32,
    _alpha: f32,
    _a_ptr: usize,
    _b_ptr: usize,
    _beta: f32,
    _c_ptr: usize,
) -> BlasStatus {
    eprintln!("blas_hgemm: Use blas_gemm with CudaTensor instead");
    BlasStatus::ComputeError
}

#[no_mangle]
pub extern "C" fn blas_dgemm(
    ctx: *mut BlasContext,
    m: i32,
    n: i32,
    k: i32,
    alpha: f64,
    a_ptr: usize,
    b_ptr: usize,
    beta: f64,
    c_ptr: usize,
) -> BlasStatus {
    if ctx.is_null() || a_ptr == 0 || b_ptr == 0 || c_ptr == 0 {
        return BlasStatus::InvalidParameter;
    }

    if m <= 0 || n <= 0 || k <= 0 {
        return BlasStatus::InvalidParameter;
    }

    let ctx = unsafe { &mut *ctx };

    match try_dgemm(ctx, m, n, k, alpha, a_ptr, b_ptr, beta, c_ptr) {
        Ok(_) => BlasStatus::Success,
        Err(e) => {
            eprintln!("blas_dgemm error: {:?}", e);
            BlasStatus::ComputeError
        }
    }
}

fn try_dgemm(
    ctx: &mut BlasContext,
    m: i32,
    n: i32,
    k: i32,
    alpha: f64,
    a_ptr: usize,
    b_ptr: usize,
    beta: f64,
    c_ptr: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let a_dev = unsafe { &*(a_ptr as *const CudaSlice<f64>) };
    let b_dev = unsafe { &*(b_ptr as *const CudaSlice<f64>) };
    let c_dev = unsafe { &mut *(c_ptr as *mut CudaSlice<f64>) };

    let cfg = GemmConfig {
        transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        m: n,
        n: m,
        k: k,
        alpha,
        lda: n,
        ldb: k,
        beta,
        ldc: n,
    };

    unsafe {
        ctx.cublas.gemm(cfg, b_dev, a_dev, c_dev)?;
    }

    ctx.device.synchronize()?;

    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{tensor_allocate, tensor_free, tensor_copy_from_f32, tensor_copy_to_f32};
    use crate::device::DeviceContext;
    use serial_test::serial;

    // 輔助函數：安全地建立測試環境
    fn try_create_test_env() -> Option<(Arc<CudaDevice>, *mut BlasContext)> {
        let device = match CudaDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("⚠️  CUDA device not available: {:?}", e);
                return None;
            }
        };

        let device_handle = &device as *const Arc<CudaDevice> as usize;
        let ctx = blas_init(device_handle);

        if ctx.is_null() {
            eprintln!("⚠️  cuBLAS initialization failed");
            return None;
        }

        Some((device, ctx))
    }

    #[test]
    #[serial]
    fn test_blas_init() {
        match try_create_test_env() {
            Some((_device, ctx)) => {
                println!("✅ cuBLAS initialized successfully");
                blas_destroy(ctx);
            }
            None => {
                println!("⚠️  Test skipped: CUDA/cuBLAS not available");
            }
        }
    }

    #[test]
    #[serial]
    fn test_sgemm_2x2() {
        let (device, ctx) = match try_create_test_env() {
            Some(env) => env,
            None => {
                println!("⚠️  Test skipped: CUDA/cuBLAS not available");
                return;
            }
        };

        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        let mut c = vec![0.0f32; 4];

        let a_dev = device.htod_sync_copy(&a).unwrap();
        let b_dev = device.htod_sync_copy(&b).unwrap();
        let mut c_dev = device.htod_sync_copy(&c).unwrap();

        let a_ptr = &a_dev as *const _ as usize;
        let b_ptr = &b_dev as *const _ as usize;
        let c_ptr = &mut c_dev as *mut _ as usize;

        let status = blas_sgemm(ctx, 2, 2, 2, 1.0, a_ptr, b_ptr, 0.0, c_ptr);
        assert!(matches!(status, BlasStatus::Success));

        device.dtoh_sync_copy_into(&c_dev, &mut c).unwrap();

        let expected = vec![19.0, 22.0, 43.0, 50.0];
        for (i, (&result, &expect)) in c.iter().zip(expected.iter()).enumerate() {
            assert!(
                (result - expect).abs() < 1e-5,
                "Mismatch at {}: {} != {}",
                i, result, expect
            );
        }

        println!("✅ SGEMM test passed");
        blas_destroy(ctx);
    }

    #[test]
    #[serial]
    fn test_dgemm_2x2() {
        let (device, ctx) = match try_create_test_env() {
            Some(env) => env,
            None => {
                println!("⚠️  Test skipped: CUDA/cuBLAS not available");
                return;
            }
        };

        let a = vec![1.0f64, 2.0, 3.0, 4.0];
        let b = vec![5.0f64, 6.0, 7.0, 8.0];
        let mut c = vec![0.0f64; 4];

        let a_dev = device.htod_sync_copy(&a).unwrap();
        let b_dev = device.htod_sync_copy(&b).unwrap();
        let mut c_dev = device.htod_sync_copy(&c).unwrap();

        let a_ptr = &a_dev as *const _ as usize;
        let b_ptr = &b_dev as *const _ as usize;
        let c_ptr = &mut c_dev as *mut _ as usize;

        let status = blas_dgemm(ctx, 2, 2, 2, 1.0, a_ptr, b_ptr, 0.0, c_ptr);
        assert!(matches!(status, BlasStatus::Success));

        device.dtoh_sync_copy_into(&c_dev, &mut c).unwrap();

        let expected = vec![19.0, 22.0, 43.0, 50.0];
        for (i, (&result, &expect)) in c.iter().zip(expected.iter()).enumerate() {
            assert!(
                (result - expect).abs() < 1e-10,
                "Mismatch at {}: {} != {}",
                i, result, expect
            );
        }

        println!("✅ DGEMM test passed");
        blas_destroy(ctx);
    }

    #[test]
    #[serial]
    fn test_unified_gemm_fp32() {
        let (device, blas_ctx) = match try_create_test_env() {
            Some(env) => env,
            None => {
                println!("⚠️  Test skipped: CUDA/cuBLAS not available");
                return;
            }
        };

        // 建立 DeviceContext
        let device_ctx = DeviceContext { device: device.clone() };
        let device_ctx_ptr = &device_ctx as *const DeviceContext;

        // 建立 FP32 tensors
        let a_handle = tensor_allocate(device_ctx_ptr, 4, Precision::FP32 as i32);
        let b_handle = tensor_allocate(device_ctx_ptr, 4, Precision::FP32 as i32);
        let c_handle = tensor_allocate(device_ctx_ptr, 4, Precision::FP32 as i32);

        assert_ne!(a_handle, 0);
        assert_ne!(b_handle, 0);
        assert_ne!(c_handle, 0);

        // 複製資料
        let a_data = [1.0f32, 2.0, 3.0, 4.0];
        let b_data = [5.0f32, 6.0, 7.0, 8.0];

        let status_a = tensor_copy_from_f32(device_ctx_ptr, a_handle, a_data.as_ptr(), 4);
        let status_b = tensor_copy_from_f32(device_ctx_ptr, b_handle, b_data.as_ptr(), 4);
        assert_eq!(status_a, 0);
        assert_eq!(status_b, 0);

        // 執行 unified GEMM
        let status = blas_gemm(blas_ctx, 2, 2, 2, 1.0, a_handle, b_handle, 0.0, c_handle);
        assert!(matches!(status, BlasStatus::Success));

        // 取回結果
        let mut result = [0.0f32; 4];
        let status_c = tensor_copy_to_f32(device_ctx_ptr, c_handle, result.as_mut_ptr(), 4);
        assert_eq!(status_c, 0);

        let expected = [19.0f32, 22.0, 43.0, 50.0];
        for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-5,
                "Mismatch at {}: {} != {}",
                i, r, e
            );
        }

        println!("✅ Unified GEMM (FP32) test passed");

        // 清理
        tensor_free(device_ctx_ptr, a_handle);
        tensor_free(device_ctx_ptr, b_handle);
        tensor_free(device_ctx_ptr, c_handle);
        blas_destroy(blas_ctx);
    }

    #[test]
    #[serial]
    fn test_hgemm_2x2() {
        let (device, blas_ctx) = match try_create_test_env() {
            Some(env) => env,
            None => {
                println!("⚠️  Test skipped: CUDA/cuBLAS not available");
                return;
            }
        };

        // 建立 DeviceContext
        let device_ctx = DeviceContext { device: device.clone() };
        let device_ctx_ptr = &device_ctx as *const DeviceContext;

        // 建立 FP16 tensors
        let a_handle = tensor_allocate(device_ctx_ptr, 4, Precision::FP16 as i32);
        let b_handle = tensor_allocate(device_ctx_ptr, 4, Precision::FP16 as i32);
        let c_handle = tensor_allocate(device_ctx_ptr, 4, Precision::FP16 as i32);

        assert_ne!(a_handle, 0);
        assert_ne!(b_handle, 0);
        assert_ne!(c_handle, 0);

        // 複製資料（FP32 會自動轉換為 FP16）
        let a_data = [1.0f32, 2.0, 3.0, 4.0];
        let b_data = [5.0f32, 6.0, 7.0, 8.0];

        let status_a = tensor_copy_from_f32(device_ctx_ptr, a_handle, a_data.as_ptr(), 4);
        let status_b = tensor_copy_from_f32(device_ctx_ptr, b_handle, b_data.as_ptr(), 4);
        assert_eq!(status_a, 0);
        assert_eq!(status_b, 0);

        // 執行 unified GEMM (會自動使用 FP16 路徑)
        let status = blas_gemm(blas_ctx, 2, 2, 2, 1.0, a_handle, b_handle, 0.0, c_handle);
        assert!(matches!(status, BlasStatus::Success), "HGEMM failed: {:?}", status);

        // 取回結果（FP16 會自動轉換為 FP32）
        let mut result = [0.0f32; 4];
        let status_c = tensor_copy_to_f32(device_ctx_ptr, c_handle, result.as_mut_ptr(), 4);
        assert_eq!(status_c, 0);

        let expected = [19.0f32, 22.0, 43.0, 50.0];
        for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 0.1,  // FP16 精度較低
                "Mismatch at {}: {} != {}",
                i, r, e
            );
        }

        println!("✅ HGEMM (FP16 via unified interface) test passed");

        // 清理
        tensor_free(device_ctx_ptr, a_handle);
        tensor_free(device_ctx_ptr, b_handle);
        tensor_free(device_ctx_ptr, c_handle);
        blas_destroy(blas_ctx);
    }

    #[test]
    #[serial]
    fn test_precision_mismatch() {
        let (device, blas_ctx) = match try_create_test_env() {
            Some(env) => env,
            None => {
                println!("⚠️  Test skipped: CUDA/cuBLAS not available");
                return;
            }
        };

        let device_ctx = DeviceContext { device: device.clone() };
        let device_ctx_ptr = &device_ctx as *const DeviceContext;

        // 建立不同精度的 tensors
        let a_handle = tensor_allocate(device_ctx_ptr, 4, Precision::FP32 as i32);
        let b_handle = tensor_allocate(device_ctx_ptr, 4, Precision::FP16 as i32);  // 不匹配！
        let c_handle = tensor_allocate(device_ctx_ptr, 4, Precision::FP32 as i32);

        // 應該返回 PrecisionMismatch
        let status = blas_gemm(blas_ctx, 2, 2, 2, 1.0, a_handle, b_handle, 0.0, c_handle);
        assert!(matches!(status, BlasStatus::PrecisionMismatch));

        println!("✅ Precision mismatch detection test passed");

        tensor_free(device_ctx_ptr, a_handle);
        tensor_free(device_ctx_ptr, b_handle);
        tensor_free(device_ctx_ptr, c_handle);
        blas_destroy(blas_ctx);
    }
}