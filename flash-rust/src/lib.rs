use cudarc::driver::CudaDevice;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use std::sync::Arc;

#[repr(C)]
pub enum GemmStatus {
    Success = 0,
    DeviceInitError = 1,
    MemoryError = 2,
    ComputeError = 3,
    InvalidDimension = 4,
}

pub struct GemmContext {
    device: Arc<CudaDevice>,
    cublas: CudaBlas,
}

#[no_mangle]
pub extern "C" fn gemm_init() -> *mut GemmContext {
    match try_init() {
        Ok(ctx) => Box::into_raw(Box::new(ctx)),
        Err(_) => std::ptr::null_mut(),
    }
}

fn try_init() -> Result<GemmContext, Box<dyn std::error::Error>> {
    let device = CudaDevice::new(0)?;
    let cublas = CudaBlas::new(device.clone())?;
    Ok(GemmContext { device, cublas })
}

/// C ABI: 執行單精度矩陣乘法 (SGEMM) - ROW-MAJOR 版本
///
/// C = alpha * A * B + beta * C
///
/// # 參數
/// - m: A 的行數, C 的行數
/// - n: B 的列數, C 的列數
/// - k: A 的列數, B 的行數
/// - A: m × k 矩陣 (row-major)
/// - B: k × n 矩陣 (row-major)
/// - C: m × n 矩陣 (row-major)
#[no_mangle]
pub extern "C" fn gemm_sgemm(
    ctx: *mut GemmContext,
    m: i32,
    n: i32,
    k: i32,
    alpha: f32,
    a: *const f32,
    b: *const f32,
    beta: f32,
    c: *mut f32,
) -> GemmStatus {
    if ctx.is_null() || a.is_null() || b.is_null() || c.is_null() {
        return GemmStatus::InvalidDimension;
    }

    if m <= 0 || n <= 0 || k <= 0 {
        return GemmStatus::InvalidDimension;
    }

    let ctx = unsafe { &mut *ctx };

    match try_gemm(ctx, m, n, k, alpha, a, b, beta, c) {
        Ok(_) => GemmStatus::Success,
        Err(e) => {
            eprintln!("GEMM error: {:?}", e);
            GemmStatus::ComputeError
        }
    }
}

fn try_gemm(
    ctx: &mut GemmContext,
    m: i32,
    n: i32,
    k: i32,
    alpha: f32,
    a_host: *const f32,
    b_host: *const f32,
    beta: f32,
    c_host: *mut f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let m_usize = m as usize;
    let n_usize = n as usize;
    let k_usize = k as usize;

    let a_slice = unsafe { std::slice::from_raw_parts(a_host, m_usize * k_usize) };
    let b_slice = unsafe { std::slice::from_raw_parts(b_host, k_usize * n_usize) };
    let c_slice = unsafe { std::slice::from_raw_parts_mut(c_host, m_usize * n_usize) };

    let a_dev = ctx.device.htod_sync_copy(a_slice)?;
    let b_dev = ctx.device.htod_sync_copy(b_slice)?;
    let mut c_dev = ctx.device.htod_sync_copy(c_slice)?;

    // ================================================================
    // Row-major to Column-major 轉換
    // ================================================================
    //
    // 問題：cuBLAS 使用 column-major，Java/C 使用 row-major
    //
    // 關鍵觀察：
    // - Row-major 的 A(m×k) 在記憶體中 = Column-major 的 A^T(k×m)
    //
    // 我們要計算: C = A × B (row-major)
    //
    // 技巧：C^T = (A × B)^T = B^T × A^T
    //
    // 由於記憶體佈局的關係：
    // - 把 row-major 的 B(k×n) 當作 column-major 讀，得到 B^T(n×k)
    // - 把 row-major 的 A(m×k) 當作 column-major 讀，得到 A^T(k×m)
    // - 計算 B^T × A^T = C^T(n×m) in column-major
    // - 把 column-major 的 C^T(n×m) 當作 row-major 讀，得到 C(m×n)
    //
    // cuBLAS gemm: C_out = α * A_cublas * B_cublas + β * C_out
    //
    // 對應關係：
    // - A_cublas = B (我們的 B，被讀成 B^T)，維度 n × k
    // - B_cublas = A (我們的 A，被讀成 A^T)，維度 k × m
    // - C_out = C (被讀成 C^T)，維度 n × m
    //
    // ================================================================

    let cfg = GemmConfig {
        // 不需要轉置，因為記憶體佈局差異已經隱含了轉置
        transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,

        // cuBLAS 的 m, n, k（注意：這是 column-major 視角）
        // A_cublas(n×k) × B_cublas(k×m) = C_out(n×m)
        m: n,  // C_out 的行數
        n: m,  // C_out 的列數
        k: k,  // 共同維度

        alpha,

        // lda = A_cublas 的 leading dimension = n（因為 A_cublas 是 n×k）
        lda: n,

        // ldb = B_cublas 的 leading dimension = k（因為 B_cublas 是 k×m）
        ldb: k,

        beta,

        // ldc = C_out 的 leading dimension = n（因為 C_out 是 n×m）
        ldc: n,
    };

    unsafe {
        // 注意參數順序：B 是 cuBLAS 的 A，A 是 cuBLAS 的 B
        ctx.cublas.gemm(cfg, &b_dev, &a_dev, &mut c_dev)?;
    }

    ctx.device.synchronize()?;
    ctx.device.dtoh_sync_copy_into(&c_dev, c_slice)?;

    Ok(())
}

#[no_mangle]
pub extern "C" fn gemm_destroy(ctx: *mut GemmContext) {
    if !ctx.is_null() {
        unsafe {
            let _ = Box::from_raw(ctx);
        }
    }
}

#[no_mangle]
pub extern "C" fn gemm_get_device_name(
    ctx: *const GemmContext,
    buffer: *mut i8,
    buffer_size: usize,
) -> bool {
    if ctx.is_null() || buffer.is_null() || buffer_size == 0 {
        return false;
    }

    let ctx = unsafe { &*ctx };

    let name = match ctx.device.name() {
        Ok(n) => n,
        Err(_) => return false,
    };

    let name_bytes = name.as_bytes();
    let copy_len = name_bytes.len().min(buffer_size - 1);

    unsafe {
        std::ptr::copy_nonoverlapping(
            name_bytes.as_ptr() as *const i8,
            buffer,
            copy_len,
        );
        *buffer.add(copy_len) = 0;
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemm_2x2() {
        let ctx = gemm_init();
        assert!(!ctx.is_null());

        // A = [1 2]    B = [5 6]
        //     [3 4]        [7 8]
        //
        // C = A × B = [1*5+2*7  1*6+2*8] = [19 22]
        //             [3*5+4*7  3*6+4*8]   [43 50]
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        let mut c = vec![0.0f32; 4];

        let status = gemm_sgemm(ctx, 2, 2, 2, 1.0,
                                a.as_ptr(), b.as_ptr(),
                                0.0, c.as_mut_ptr());

        assert!(matches!(status, GemmStatus::Success));

        let expected = vec![19.0, 22.0, 43.0, 50.0];
        for (i, (&result, &expect)) in c.iter().zip(expected.iter()).enumerate() {
            assert!(
                (result - expect).abs() < 1e-5,
                "Mismatch at index {}: got {}, expected {}",
                i, result, expect
            );
        }

        gemm_destroy(ctx);
    }

    #[test]
    fn test_gemm_identity() {
        let ctx = gemm_init();
        assert!(!ctx.is_null());

        // I × B = B
        let identity = vec![1.0f32, 0.0, 0.0, 1.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        let mut c = vec![0.0f32; 4];

        let status = gemm_sgemm(ctx, 2, 2, 2, 1.0,
                                identity.as_ptr(), b.as_ptr(),
                                0.0, c.as_mut_ptr());

        assert!(matches!(status, GemmStatus::Success));

        for (i, (&result, &expect)) in c.iter().zip(b.iter()).enumerate() {
            assert!(
                (result - expect).abs() < 1e-5,
                "Mismatch at index {}: got {}, expected {}",
                i, result, expect
            );
        }

        gemm_destroy(ctx);
    }

    #[test]
    fn test_gemm_non_square() {
        let ctx = gemm_init();
        assert!(!ctx.is_null());

        // A (3×2) × B (2×4) = C (3×4)
        // A = [1 2]
        //     [3 4]
        //     [5 6]
        // B = [1 2 3 4]
        //     [5 6 7 8]
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0f32; 12];

        let status = gemm_sgemm(ctx, 3, 4, 2, 1.0,
                                a.as_ptr(), b.as_ptr(),
                                0.0, c.as_mut_ptr());

        assert!(matches!(status, GemmStatus::Success));

        // C = [1*1+2*5  1*2+2*6  1*3+2*7  1*4+2*8]   [11 14 17 20]
        //     [3*1+4*5  3*2+4*6  3*3+4*7  3*4+4*8] = [23 30 37 44]
        //     [5*1+6*5  5*2+6*6  5*3+6*7  5*4+6*8]   [35 46 57 68]
        let expected = vec![11.0, 14.0, 17.0, 20.0,
                           23.0, 30.0, 37.0, 44.0,
                           35.0, 46.0, 57.0, 68.0];

        for (i, (&result, &expect)) in c.iter().zip(expected.iter()).enumerate() {
            assert!(
                (result - expect).abs() < 1e-5,
                "Mismatch at index {}: got {}, expected {}",
                i, result, expect
            );
        }

        gemm_destroy(ctx);
    }
}