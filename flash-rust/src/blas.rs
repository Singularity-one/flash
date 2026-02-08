use cudarc::driver::{CudaDevice, CudaSlice};
use cudarc::cublas::{CudaBlas as CudarcBlas, Gemm, GemmConfig};
use std::sync::Arc;

/// Status codes for BLAS operations
#[repr(C)]
pub enum BlasStatus {
    Success = 0,
    InitError = 1,
    ComputeError = 2,
    InvalidParameter = 3,
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

    match try_blas_init(device) {
        Ok(ctx) => Box::into_raw(Box::new(ctx)),
        Err(e) => {
            eprintln!("blas_init error: {:?}", e);
            std::ptr::null_mut()
        }
    }
}

fn try_blas_init(device: Arc<CudaDevice>) -> Result<BlasContext, Box<dyn std::error::Error>> {
    let cublas = CudarcBlas::new(device.clone())?;
    Ok(BlasContext { device, cublas })
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
// BLAS Level 3: GEMM
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    fn create_test_context() -> (*mut BlasContext, Arc<CudaDevice>) {
        // CudaDevice::new 已經返回 Arc<CudaDevice>
        let device = CudaDevice::new(0).expect("Failed to create device");

        // 重點修正：取得指向 Arc<CudaDevice> 的指標，而非 Arc 內部資料的指標
        // blas_init 期望 device_handle 是 *const Arc<CudaDevice>
        let device_handle = &device as *const Arc<CudaDevice> as usize;

        let ctx = blas_init(device_handle);
        assert!(!ctx.is_null(), "Failed to initialize BLAS context");

        (ctx, device)
    }

    #[test]
    #[serial]
    fn test_blas_init() {
        let (ctx, _device) = create_test_context();
        blas_destroy(ctx);
    }

    #[test]
    #[serial]
    fn test_sgemm_2x2() {
        let (ctx, device) = create_test_context();

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

        blas_destroy(ctx);
    }
}