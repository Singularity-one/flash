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
/// 輸入/輸出矩陣都是 row-major 佈局
/// C = alpha * A * B + beta * C
///
/// # 參數
/// - m, n, k: 矩陣維度 (A: m×k, B: k×n, C: m×n)
/// - a, b, c: row-major 矩陣數據
///
/// # 實現細節
/// cuBLAS 使用 column-major，所以我們用 "row-major trick":
/// (A × B)_row = (B^T × A^T)_col
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

    // Row-major trick 正確實作：
    //
    // 我們要算: C(m×n) = A(m×k) × B(k×n)  [row-major]
    //
    // Row-major 矩陣在記憶體中的排列，如果用 column-major 來讀：
    // - A(m×k) row-major = A^T(k×m) column-major
    // - B(k×n) row-major = B^T(n×k) column-major
    // - C(m×n) row-major = C^T(n×m) column-major
    //
    // 我們需要: C^T = (A×B)^T = B^T × A^T
    //
    // 所以 cuBLAS 調用: C^T(n×m) = B^T(n×k) × A^T(k×m)
    //                   ↑         ↑          ↑
    //                gemm output  gemm A     gemm B
    //
    // cuBLAS gemm: C = α*op(A)*op(B) + β*C
    // 這裡 op(A)=B^T, op(B)=A^T, 但因為記憶體已經是轉置的排列，
    // 所以 transa=N, transb=N (不需要額外轉置！)

    let cfg = GemmConfig {
        transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,

        // C^T 是 n×m
        m: n,
        n: m,
        k: k,

        alpha,

        // B 作為 cuBLAS 的 "A": B^T(n×k) column-major, lda = n
        lda: n,

        // A 作為 cuBLAS 的 "B": A^T(k×m) column-major, ldb = k
        ldb: k,

        beta,

        // C^T(n×m) column-major, ldc = n
        ldc: n,
    };

    unsafe {
        // 注意順序：B 是 cuBLAS 的第一個矩陣，A 是第二個
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
    fn test_gemm_row_major() {
        let ctx = gemm_init();
        assert!(!ctx.is_null());

        // Row-major 2×2 矩陣
        // A = [1 2]    B = [5 6]
        //     [3 4]        [7 8]
        let a = vec![1.0f32, 2.0, 3.0, 4.0];  // row-major
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        let mut c = vec![0.0f32; 4];

        let status = gemm_sgemm(ctx, 2, 2, 2, 1.0,
                                a.as_ptr(), b.as_ptr(),
                                0.0, c.as_mut_ptr());

        assert!(matches!(status, GemmStatus::Success));

        // 期望結果 (row-major):
        // C = A×B = [19 22]  →  {19, 22, 43, 50}
        //           [43 50]
        let expected = vec![19.0, 22.0, 43.0, 50.0];

        for (i, (&result, &expect)) in c.iter().zip(expected.iter()).enumerate() {
            assert!(
                (result - expect).abs() < 1e-5,
                "Mismatch at index {}: {} != {}",
                i, result, expect
            );
        }

        gemm_destroy(ctx);
    }
}