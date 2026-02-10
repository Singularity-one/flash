//! Phase 4: Element-wise operations (CPU fallback implementation)
//!
//! Provides common element-wise operations:
//! - Binary: add, sub, mul, div
//! - Unary: exp, log, sqrt, pow, tanh, sigmoid, relu, gelu, silu
//! - Scalar: scale, fill
//! - Reduction: sum, max, mean
//! - Cast: precision conversion
//!
//! Note: Using CPU fallback for simplicity. GPU kernels can be added later.

use cudarc::driver::CudaDevice;
use std::sync::Arc;
use half::{f16, bf16};

use crate::tensor::{TensorHandle, TensorData, Precision};
use crate::device::DeviceContext;

/// Status codes for ops operations
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpsStatus {
    Success = 0,
    InvalidParameter = 1,
    SizeMismatch = 2,
    PrecisionMismatch = 3,
    ComputeError = 4,
    NotSupported = 5,
}

// ============================================================================
// Binary Operations: add, sub, mul, div
// ============================================================================

/// Element-wise addition: c = a + b
#[no_mangle]
pub extern "C" fn ops_add(
    ctx: *const DeviceContext,
    a_handle: usize,
    b_handle: usize,
    c_handle: usize,
) -> OpsStatus {
    binary_op(ctx, a_handle, b_handle, c_handle, |a, b| a + b)
}

/// Element-wise subtraction: c = a - b
#[no_mangle]
pub extern "C" fn ops_sub(
    ctx: *const DeviceContext,
    a_handle: usize,
    b_handle: usize,
    c_handle: usize,
) -> OpsStatus {
    binary_op(ctx, a_handle, b_handle, c_handle, |a, b| a - b)
}

/// Element-wise multiplication: c = a * b
#[no_mangle]
pub extern "C" fn ops_mul(
    ctx: *const DeviceContext,
    a_handle: usize,
    b_handle: usize,
    c_handle: usize,
) -> OpsStatus {
    binary_op(ctx, a_handle, b_handle, c_handle, |a, b| a * b)
}

/// Element-wise division: c = a / b
#[no_mangle]
pub extern "C" fn ops_div(
    ctx: *const DeviceContext,
    a_handle: usize,
    b_handle: usize,
    c_handle: usize,
) -> OpsStatus {
    binary_op(ctx, a_handle, b_handle, c_handle, |a, b| a / b)
}

fn binary_op<F>(
    ctx: *const DeviceContext,
    a_handle: usize,
    b_handle: usize,
    c_handle: usize,
    op: F,
) -> OpsStatus
where
    F: Fn(f64, f64) -> f64,
{
    if ctx.is_null() || a_handle == 0 || b_handle == 0 || c_handle == 0 {
        return OpsStatus::InvalidParameter;
    }

    let ctx = unsafe { &*ctx };
    let a = unsafe { &*(a_handle as *const TensorHandle) };
    let b = unsafe { &*(b_handle as *const TensorHandle) };
    let c = unsafe { &mut *(c_handle as *mut TensorHandle) };

    if a.element_count != b.element_count || a.element_count != c.element_count {
        return OpsStatus::SizeMismatch;
    }

    if a.dtype != b.dtype || a.dtype != c.dtype {
        return OpsStatus::PrecisionMismatch;
    }

    match binary_op_impl(&ctx.device, a, b, c, &op) {
        Ok(_) => OpsStatus::Success,
        Err(e) => {
            eprintln!("binary_op error: {:?}", e);
            OpsStatus::ComputeError
        }
    }
}

fn binary_op_impl<F>(
    device: &Arc<CudaDevice>,
    a: &TensorHandle,
    b: &TensorHandle,
    c: &mut TensorHandle,
    op: &F,
) -> Result<(), Box<dyn std::error::Error>>
where
    F: Fn(f64, f64) -> f64,
{
    let count = a.element_count;

    match (&a.data, &b.data, &mut c.data) {
        (TensorData::FP32(a_slice), TensorData::FP32(b_slice), TensorData::FP32(c_slice)) => {
            let mut a_host = vec![0.0f32; count];
            let mut b_host = vec![0.0f32; count];
            device.dtoh_sync_copy_into(a_slice, &mut a_host)?;
            device.dtoh_sync_copy_into(b_slice, &mut b_host)?;

            let c_host: Vec<f32> = a_host.iter().zip(b_host.iter())
                .map(|(&a, &b)| op(a as f64, b as f64) as f32)
                .collect();

            device.htod_copy_into(c_host, c_slice)?;
        }
        (TensorData::FP64(a_slice), TensorData::FP64(b_slice), TensorData::FP64(c_slice)) => {
            let mut a_host = vec![0.0f64; count];
            let mut b_host = vec![0.0f64; count];
            device.dtoh_sync_copy_into(a_slice, &mut a_host)?;
            device.dtoh_sync_copy_into(b_slice, &mut b_host)?;

            let c_host: Vec<f64> = a_host.iter().zip(b_host.iter())
                .map(|(&a, &b)| op(a, b))
                .collect();

            device.htod_copy_into(c_host, c_slice)?;
        }
        (TensorData::FP16(a_slice), TensorData::FP16(b_slice), TensorData::FP16(c_slice)) => {
            let mut a_host = vec![f16::from_f32(0.0); count];
            let mut b_host = vec![f16::from_f32(0.0); count];
            device.dtoh_sync_copy_into(a_slice, &mut a_host)?;
            device.dtoh_sync_copy_into(b_slice, &mut b_host)?;

            let c_host: Vec<f16> = a_host.iter().zip(b_host.iter())
                .map(|(a, b)| f16::from_f64(op(a.to_f64(), b.to_f64())))
                .collect();

            device.htod_copy_into(c_host, c_slice)?;
        }
        (TensorData::BF16(a_slice), TensorData::BF16(b_slice), TensorData::BF16(c_slice)) => {
            let mut a_host = vec![bf16::from_f32(0.0); count];
            let mut b_host = vec![bf16::from_f32(0.0); count];
            device.dtoh_sync_copy_into(a_slice, &mut a_host)?;
            device.dtoh_sync_copy_into(b_slice, &mut b_host)?;

            let c_host: Vec<bf16> = a_host.iter().zip(b_host.iter())
                .map(|(a, b)| bf16::from_f64(op(a.to_f64(), b.to_f64())))
                .collect();

            device.htod_copy_into(c_host, c_slice)?;
        }
        _ => return Err("Unsupported precision combination".into()),
    }

    device.synchronize()?;
    Ok(())
}

// ============================================================================
// Unary Operations: exp, log, sqrt, tanh, sigmoid, relu, gelu, silu
// ============================================================================

/// Element-wise exponential: y = exp(x)
#[no_mangle]
pub extern "C" fn ops_exp(
    ctx: *const DeviceContext,
    x_handle: usize,
    y_handle: usize,
) -> OpsStatus {
    unary_op(ctx, x_handle, y_handle, |x| x.exp())
}

/// Element-wise natural log: y = ln(x)
#[no_mangle]
pub extern "C" fn ops_log(
    ctx: *const DeviceContext,
    x_handle: usize,
    y_handle: usize,
) -> OpsStatus {
    unary_op(ctx, x_handle, y_handle, |x| x.ln())
}

/// Element-wise square root: y = sqrt(x)
#[no_mangle]
pub extern "C" fn ops_sqrt(
    ctx: *const DeviceContext,
    x_handle: usize,
    y_handle: usize,
) -> OpsStatus {
    unary_op(ctx, x_handle, y_handle, |x| x.sqrt())
}

/// Element-wise tanh: y = tanh(x)
#[no_mangle]
pub extern "C" fn ops_tanh(
    ctx: *const DeviceContext,
    x_handle: usize,
    y_handle: usize,
) -> OpsStatus {
    unary_op(ctx, x_handle, y_handle, |x| x.tanh())
}

/// Element-wise sigmoid: y = 1 / (1 + exp(-x))
#[no_mangle]
pub extern "C" fn ops_sigmoid(
    ctx: *const DeviceContext,
    x_handle: usize,
    y_handle: usize,
) -> OpsStatus {
    unary_op(ctx, x_handle, y_handle, |x| 1.0 / (1.0 + (-x).exp()))
}

/// Element-wise ReLU: y = max(0, x)
#[no_mangle]
pub extern "C" fn ops_relu(
    ctx: *const DeviceContext,
    x_handle: usize,
    y_handle: usize,
) -> OpsStatus {
    unary_op(ctx, x_handle, y_handle, |x| x.max(0.0))
}

/// Element-wise GELU: y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
#[no_mangle]
pub extern "C" fn ops_gelu(
    ctx: *const DeviceContext,
    x_handle: usize,
    y_handle: usize,
) -> OpsStatus {
    let sqrt_2_over_pi = (2.0_f64 / std::f64::consts::PI).sqrt();
    unary_op(ctx, x_handle, y_handle, move |x| {
        0.5 * x * (1.0 + (sqrt_2_over_pi * (x + 0.044715 * x.powi(3))).tanh())
    })
}

/// Element-wise SiLU (Swish): y = x * sigmoid(x)
#[no_mangle]
pub extern "C" fn ops_silu(
    ctx: *const DeviceContext,
    x_handle: usize,
    y_handle: usize,
) -> OpsStatus {
    unary_op(ctx, x_handle, y_handle, |x| x / (1.0 + (-x).exp()))
}

/// Element-wise power: y = x^n
#[no_mangle]
pub extern "C" fn ops_pow(
    ctx: *const DeviceContext,
    x_handle: usize,
    y_handle: usize,
    n: f64,
) -> OpsStatus {
    unary_op(ctx, x_handle, y_handle, move |x| x.powf(n))
}

/// Element-wise negation: y = -x
#[no_mangle]
pub extern "C" fn ops_neg(
    ctx: *const DeviceContext,
    x_handle: usize,
    y_handle: usize,
) -> OpsStatus {
    unary_op(ctx, x_handle, y_handle, |x| -x)
}

/// Element-wise absolute value: y = |x|
#[no_mangle]
pub extern "C" fn ops_abs(
    ctx: *const DeviceContext,
    x_handle: usize,
    y_handle: usize,
) -> OpsStatus {
    unary_op(ctx, x_handle, y_handle, |x| x.abs())
}

fn unary_op<F>(
    ctx: *const DeviceContext,
    x_handle: usize,
    y_handle: usize,
    op: F,
) -> OpsStatus
where
    F: Fn(f64) -> f64,
{
    if ctx.is_null() || x_handle == 0 || y_handle == 0 {
        return OpsStatus::InvalidParameter;
    }

    let ctx = unsafe { &*ctx };
    let x = unsafe { &*(x_handle as *const TensorHandle) };
    let y = unsafe { &mut *(y_handle as *mut TensorHandle) };

    if x.element_count != y.element_count {
        return OpsStatus::SizeMismatch;
    }

    if x.dtype != y.dtype {
        return OpsStatus::PrecisionMismatch;
    }

    match unary_op_impl(&ctx.device, x, y, &op) {
        Ok(_) => OpsStatus::Success,
        Err(e) => {
            eprintln!("unary_op error: {:?}", e);
            OpsStatus::ComputeError
        }
    }
}

fn unary_op_impl<F>(
    device: &Arc<CudaDevice>,
    x: &TensorHandle,
    y: &mut TensorHandle,
    op: &F,
) -> Result<(), Box<dyn std::error::Error>>
where
    F: Fn(f64) -> f64,
{
    let count = x.element_count;

    match (&x.data, &mut y.data) {
        (TensorData::FP32(x_slice), TensorData::FP32(y_slice)) => {
            let mut x_host = vec![0.0f32; count];
            device.dtoh_sync_copy_into(x_slice, &mut x_host)?;

            let y_host: Vec<f32> = x_host.iter()
                .map(|&x| op(x as f64) as f32)
                .collect();

            device.htod_copy_into(y_host, y_slice)?;
        }
        (TensorData::FP64(x_slice), TensorData::FP64(y_slice)) => {
            let mut x_host = vec![0.0f64; count];
            device.dtoh_sync_copy_into(x_slice, &mut x_host)?;

            let y_host: Vec<f64> = x_host.iter()
                .map(|&x| op(x))
                .collect();

            device.htod_copy_into(y_host, y_slice)?;
        }
        (TensorData::FP16(x_slice), TensorData::FP16(y_slice)) => {
            let mut x_host = vec![f16::from_f32(0.0); count];
            device.dtoh_sync_copy_into(x_slice, &mut x_host)?;

            let y_host: Vec<f16> = x_host.iter()
                .map(|x| f16::from_f64(op(x.to_f64())))
                .collect();

            device.htod_copy_into(y_host, y_slice)?;
        }
        (TensorData::BF16(x_slice), TensorData::BF16(y_slice)) => {
            let mut x_host = vec![bf16::from_f32(0.0); count];
            device.dtoh_sync_copy_into(x_slice, &mut x_host)?;

            let y_host: Vec<bf16> = x_host.iter()
                .map(|x| bf16::from_f64(op(x.to_f64())))
                .collect();

            device.htod_copy_into(y_host, y_slice)?;
        }
        _ => return Err("Unsupported precision".into()),
    }

    device.synchronize()?;
    Ok(())
}

// ============================================================================
// Scalar Operations: scale, fill
// ============================================================================

/// Scale tensor: y = alpha * x
#[no_mangle]
pub extern "C" fn ops_scale(
    ctx: *const DeviceContext,
    x_handle: usize,
    y_handle: usize,
    alpha: f64,
) -> OpsStatus {
    unary_op(ctx, x_handle, y_handle, move |x| alpha * x)
}

/// Fill tensor with constant value
#[no_mangle]
pub extern "C" fn ops_fill(
    ctx: *const DeviceContext,
    x_handle: usize,
    value: f64,
) -> OpsStatus {
    if ctx.is_null() || x_handle == 0 {
        return OpsStatus::InvalidParameter;
    }

    let ctx = unsafe { &*ctx };
    let x = unsafe { &mut *(x_handle as *mut TensorHandle) };

    match fill_impl(&ctx.device, x, value) {
        Ok(_) => OpsStatus::Success,
        Err(e) => {
            eprintln!("fill error: {:?}", e);
            OpsStatus::ComputeError
        }
    }
}

fn fill_impl(
    device: &Arc<CudaDevice>,
    x: &mut TensorHandle,
    value: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let count = x.element_count;

    match &mut x.data {
        TensorData::FP32(slice) => {
            let host = vec![value as f32; count];
            device.htod_copy_into(host, slice)?;
        }
        TensorData::FP64(slice) => {
            let host = vec![value; count];
            device.htod_copy_into(host, slice)?;
        }
        TensorData::FP16(slice) => {
            let host = vec![f16::from_f64(value); count];
            device.htod_copy_into(host, slice)?;
        }
        TensorData::BF16(slice) => {
            let host = vec![bf16::from_f64(value); count];
            device.htod_copy_into(host, slice)?;
        }
        _ => return Err("Unsupported precision".into()),
    }

    device.synchronize()?;
    Ok(())
}

// ============================================================================
// Reduction Operations: sum, max, mean
// ============================================================================

/// Reduce sum: returns sum of all elements
#[no_mangle]
pub extern "C" fn ops_sum(
    ctx: *const DeviceContext,
    x_handle: usize,
    result: *mut f64,
) -> OpsStatus {
    reduction_op(ctx, x_handle, result, |v| v.iter().sum())
}

/// Reduce max: returns maximum element
#[no_mangle]
pub extern "C" fn ops_max(
    ctx: *const DeviceContext,
    x_handle: usize,
    result: *mut f64,
) -> OpsStatus {
    reduction_op(ctx, x_handle, result, |v| {
        v.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    })
}

/// Reduce min: returns minimum element
#[no_mangle]
pub extern "C" fn ops_min(
    ctx: *const DeviceContext,
    x_handle: usize,
    result: *mut f64,
) -> OpsStatus {
    reduction_op(ctx, x_handle, result, |v| {
        v.iter().cloned().fold(f64::INFINITY, f64::min)
    })
}

/// Reduce mean: returns mean of all elements
#[no_mangle]
pub extern "C" fn ops_mean(
    ctx: *const DeviceContext,
    x_handle: usize,
    result: *mut f64,
) -> OpsStatus {
    if ctx.is_null() || x_handle == 0 || result.is_null() {
        return OpsStatus::InvalidParameter;
    }

    let x = unsafe { &*(x_handle as *const TensorHandle) };
    let count = x.element_count as f64;

    reduction_op(ctx, x_handle, result, move |v| v.iter().sum::<f64>() / count)
}

fn reduction_op<F>(
    ctx: *const DeviceContext,
    x_handle: usize,
    result: *mut f64,
    op: F,
) -> OpsStatus
where
    F: Fn(&[f64]) -> f64,
{
    if ctx.is_null() || x_handle == 0 || result.is_null() {
        return OpsStatus::InvalidParameter;
    }

    let ctx = unsafe { &*ctx };
    let x = unsafe { &*(x_handle as *const TensorHandle) };

    match reduction_impl(&ctx.device, x, &op) {
        Ok(value) => {
            unsafe { *result = value; }
            OpsStatus::Success
        }
        Err(e) => {
            eprintln!("reduction error: {:?}", e);
            OpsStatus::ComputeError
        }
    }
}

fn reduction_impl<F>(
    device: &Arc<CudaDevice>,
    x: &TensorHandle,
    op: &F,
) -> Result<f64, Box<dyn std::error::Error>>
where
    F: Fn(&[f64]) -> f64,
{
    let count = x.element_count;
    let values: Vec<f64>;

    match &x.data {
        TensorData::FP32(slice) => {
            let mut host = vec![0.0f32; count];
            device.dtoh_sync_copy_into(slice, &mut host)?;
            values = host.iter().map(|&x| x as f64).collect();
        }
        TensorData::FP64(slice) => {
            let mut host = vec![0.0f64; count];
            device.dtoh_sync_copy_into(slice, &mut host)?;
            values = host;
        }
        TensorData::FP16(slice) => {
            let mut host = vec![f16::from_f32(0.0); count];
            device.dtoh_sync_copy_into(slice, &mut host)?;
            values = host.iter().map(|x| x.to_f64()).collect();
        }
        TensorData::BF16(slice) => {
            let mut host = vec![bf16::from_f32(0.0); count];
            device.dtoh_sync_copy_into(slice, &mut host)?;
            values = host.iter().map(|x| x.to_f64()).collect();
        }
        _ => return Err("Unsupported precision".into()),
    }

    Ok(op(&values))
}

// ============================================================================
// Cast Operation: precision conversion
// ============================================================================

/// Cast tensor to different precision
/// Copies data from src to dst with precision conversion
#[no_mangle]
pub extern "C" fn ops_cast(
    ctx: *const DeviceContext,
    src_handle: usize,
    dst_handle: usize,
) -> OpsStatus {
    if ctx.is_null() || src_handle == 0 || dst_handle == 0 {
        return OpsStatus::InvalidParameter;
    }

    let ctx = unsafe { &*ctx };
    let src = unsafe { &*(src_handle as *const TensorHandle) };
    let dst = unsafe { &mut *(dst_handle as *mut TensorHandle) };

    if src.element_count != dst.element_count {
        return OpsStatus::SizeMismatch;
    }

    match cast_impl(&ctx.device, src, dst) {
        Ok(_) => OpsStatus::Success,
        Err(e) => {
            eprintln!("cast error: {:?}", e);
            OpsStatus::ComputeError
        }
    }
}

fn cast_impl(
    device: &Arc<CudaDevice>,
    src: &TensorHandle,
    dst: &mut TensorHandle,
) -> Result<(), Box<dyn std::error::Error>> {
    let count = src.element_count;

    // First, get source data as f64
    let values: Vec<f64> = match &src.data {
        TensorData::FP32(slice) => {
            let mut host = vec![0.0f32; count];
            device.dtoh_sync_copy_into(slice, &mut host)?;
            host.iter().map(|&x| x as f64).collect()
        }
        TensorData::FP64(slice) => {
            let mut host = vec![0.0f64; count];
            device.dtoh_sync_copy_into(slice, &mut host)?;
            host
        }
        TensorData::FP16(slice) => {
            let mut host = vec![f16::from_f32(0.0); count];
            device.dtoh_sync_copy_into(slice, &mut host)?;
            host.iter().map(|x| x.to_f64()).collect()
        }
        TensorData::BF16(slice) => {
            let mut host = vec![bf16::from_f32(0.0); count];
            device.dtoh_sync_copy_into(slice, &mut host)?;
            host.iter().map(|x| x.to_f64()).collect()
        }
        _ => return Err("Unsupported source precision".into()),
    };

    // Then, write to destination in target precision
    match &mut dst.data {
        TensorData::FP32(slice) => {
            let host: Vec<f32> = values.iter().map(|&x| x as f32).collect();
            device.htod_copy_into(host, slice)?;
        }
        TensorData::FP64(slice) => {
            device.htod_copy_into(values, slice)?;
        }
        TensorData::FP16(slice) => {
            let host: Vec<f16> = values.iter().map(|&x| f16::from_f64(x)).collect();
            device.htod_copy_into(host, slice)?;
        }
        TensorData::BF16(slice) => {
            let host: Vec<bf16> = values.iter().map(|&x| bf16::from_f64(x)).collect();
            device.htod_copy_into(host, slice)?;
        }
        _ => return Err("Unsupported destination precision".into()),
    }

    device.synchronize()?;
    Ok(())
}

// ============================================================================
// In-place Operations (for gradient computation)
// ============================================================================

/// In-place add: x += alpha * y (for gradient accumulation)
#[no_mangle]
pub extern "C" fn ops_axpy(
    ctx: *const DeviceContext,
    alpha: f64,
    x_handle: usize,
    y_handle: usize,
) -> OpsStatus {
    if ctx.is_null() || x_handle == 0 || y_handle == 0 {
        return OpsStatus::InvalidParameter;
    }

    let ctx = unsafe { &*ctx };
    let x = unsafe { &*(x_handle as *const TensorHandle) };
    let y = unsafe { &mut *(y_handle as *mut TensorHandle) };

    if x.element_count != y.element_count {
        return OpsStatus::SizeMismatch;
    }

    if x.dtype != y.dtype {
        return OpsStatus::PrecisionMismatch;
    }

    match axpy_impl(&ctx.device, alpha, x, y) {
        Ok(_) => OpsStatus::Success,
        Err(e) => {
            eprintln!("axpy error: {:?}", e);
            OpsStatus::ComputeError
        }
    }
}

fn axpy_impl(
    device: &Arc<CudaDevice>,
    alpha: f64,
    x: &TensorHandle,
    y: &mut TensorHandle,
) -> Result<(), Box<dyn std::error::Error>> {
    let count = x.element_count;

    match (&x.data, &mut y.data) {
        (TensorData::FP32(x_slice), TensorData::FP32(y_slice)) => {
            let mut x_host = vec![0.0f32; count];
            let mut y_host = vec![0.0f32; count];
            device.dtoh_sync_copy_into(x_slice, &mut x_host)?;
            device.dtoh_sync_copy_into(y_slice, &mut y_host)?;

            for (y_val, &x_val) in y_host.iter_mut().zip(x_host.iter()) {
                *y_val += (alpha * x_val as f64) as f32;
            }

            device.htod_copy_into(y_host, y_slice)?;
        }
        (TensorData::FP64(x_slice), TensorData::FP64(y_slice)) => {
            let mut x_host = vec![0.0f64; count];
            let mut y_host = vec![0.0f64; count];
            device.dtoh_sync_copy_into(x_slice, &mut x_host)?;
            device.dtoh_sync_copy_into(y_slice, &mut y_host)?;

            for (y_val, &x_val) in y_host.iter_mut().zip(x_host.iter()) {
                *y_val += alpha * x_val;
            }

            device.htod_copy_into(y_host, y_slice)?;
        }
        (TensorData::FP16(x_slice), TensorData::FP16(y_slice)) => {
            let mut x_host = vec![f16::from_f32(0.0); count];
            let mut y_host = vec![f16::from_f32(0.0); count];
            device.dtoh_sync_copy_into(x_slice, &mut x_host)?;
            device.dtoh_sync_copy_into(y_slice, &mut y_host)?;

            let result: Vec<f16> = x_host.iter().zip(y_host.iter())
                .map(|(x, y)| f16::from_f64(y.to_f64() + alpha * x.to_f64()))
                .collect();

            device.htod_copy_into(result, y_slice)?;
        }
        (TensorData::BF16(x_slice), TensorData::BF16(y_slice)) => {
            let mut x_host = vec![bf16::from_f32(0.0); count];
            let mut y_host = vec![bf16::from_f32(0.0); count];
            device.dtoh_sync_copy_into(x_slice, &mut x_host)?;
            device.dtoh_sync_copy_into(y_slice, &mut y_host)?;

            let result: Vec<bf16> = x_host.iter().zip(y_host.iter())
                .map(|(x, y)| bf16::from_f64(y.to_f64() + alpha * x.to_f64()))
                .collect();

            device.htod_copy_into(result, y_slice)?;
        }
        _ => return Err("Unsupported precision".into()),
    }

    device.synchronize()?;
    Ok(())
}

// ============================================================================
// Clipping Operations (for gradient clipping)
// ============================================================================

/// Clip values to range [min_val, max_val]
#[no_mangle]
pub extern "C" fn ops_clip(
    ctx: *const DeviceContext,
    x_handle: usize,
    y_handle: usize,
    min_val: f64,
    max_val: f64,
) -> OpsStatus {
    unary_op(ctx, x_handle, y_handle, move |x| x.max(min_val).min(max_val))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{tensor_allocate, tensor_free, tensor_copy_from_f32, tensor_copy_to_f32};
    use serial_test::serial;

    fn try_create_device() -> Option<DeviceContext> {
        match CudaDevice::new(0) {
            Ok(device) => Some(DeviceContext { device }),
            Err(e) => {
                eprintln!("⚠️  CUDA device not available: {:?}", e);
                None
            }
        }
    }

    #[test]
    #[serial]
    fn test_add() {
        let ctx = match try_create_device() {
            Some(c) => c,
            None => {
                println!("⚠️  Test skipped");
                return;
            }
        };
        let ctx_ptr = &ctx as *const DeviceContext;

        let a_handle = tensor_allocate(ctx_ptr, 4, Precision::FP32 as i32);
        let b_handle = tensor_allocate(ctx_ptr, 4, Precision::FP32 as i32);
        let c_handle = tensor_allocate(ctx_ptr, 4, Precision::FP32 as i32);

        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        tensor_copy_from_f32(ctx_ptr, a_handle, a.as_ptr(), 4);
        tensor_copy_from_f32(ctx_ptr, b_handle, b.as_ptr(), 4);

        let status = ops_add(ctx_ptr, a_handle, b_handle, c_handle);
        assert!(matches!(status, OpsStatus::Success));

        let mut result = vec![0.0f32; 4];
        tensor_copy_to_f32(ctx_ptr, c_handle, result.as_mut_ptr(), 4);

        let expected = vec![6.0, 8.0, 10.0, 12.0];
        for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((r - e).abs() < 1e-6, "Mismatch at {}: {} != {}", i, r, e);
        }

        println!("✅ Add test passed");

        tensor_free(ctx_ptr, a_handle);
        tensor_free(ctx_ptr, b_handle);
        tensor_free(ctx_ptr, c_handle);
    }

    #[test]
    #[serial]
    fn test_mul() {
        let ctx = match try_create_device() {
            Some(c) => c,
            None => return,
        };
        let ctx_ptr = &ctx as *const DeviceContext;

        let a_handle = tensor_allocate(ctx_ptr, 4, Precision::FP32 as i32);
        let b_handle = tensor_allocate(ctx_ptr, 4, Precision::FP32 as i32);
        let c_handle = tensor_allocate(ctx_ptr, 4, Precision::FP32 as i32);

        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![2.0f32, 3.0, 4.0, 5.0];
        tensor_copy_from_f32(ctx_ptr, a_handle, a.as_ptr(), 4);
        tensor_copy_from_f32(ctx_ptr, b_handle, b.as_ptr(), 4);

        let status = ops_mul(ctx_ptr, a_handle, b_handle, c_handle);
        assert!(matches!(status, OpsStatus::Success));

        let mut result = vec![0.0f32; 4];
        tensor_copy_to_f32(ctx_ptr, c_handle, result.as_mut_ptr(), 4);

        let expected = vec![2.0, 6.0, 12.0, 20.0];
        for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((r - e).abs() < 1e-6, "Mismatch at {}: {} != {}", i, r, e);
        }

        println!("✅ Mul test passed");

        tensor_free(ctx_ptr, a_handle);
        tensor_free(ctx_ptr, b_handle);
        tensor_free(ctx_ptr, c_handle);
    }

    #[test]
    #[serial]
    fn test_exp() {
        let ctx = match try_create_device() {
            Some(c) => c,
            None => return,
        };
        let ctx_ptr = &ctx as *const DeviceContext;

        let x_handle = tensor_allocate(ctx_ptr, 4, Precision::FP32 as i32);
        let y_handle = tensor_allocate(ctx_ptr, 4, Precision::FP32 as i32);

        let x = vec![0.0f32, 1.0, 2.0, -1.0];
        tensor_copy_from_f32(ctx_ptr, x_handle, x.as_ptr(), 4);

        let status = ops_exp(ctx_ptr, x_handle, y_handle);
        assert!(matches!(status, OpsStatus::Success));

        let mut result = vec![0.0f32; 4];
        tensor_copy_to_f32(ctx_ptr, y_handle, result.as_mut_ptr(), 4);

        let expected: Vec<f32> = x.iter().map(|&v| v.exp()).collect();
        for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((r - e).abs() < 1e-5, "Mismatch at {}: {} != {}", i, r, e);
        }

        println!("✅ Exp test passed");

        tensor_free(ctx_ptr, x_handle);
        tensor_free(ctx_ptr, y_handle);
    }

    #[test]
    #[serial]
    fn test_gelu() {
        let ctx = match try_create_device() {
            Some(c) => c,
            None => return,
        };
        let ctx_ptr = &ctx as *const DeviceContext;

        let x_handle = tensor_allocate(ctx_ptr, 4, Precision::FP32 as i32);
        let y_handle = tensor_allocate(ctx_ptr, 4, Precision::FP32 as i32);

        let x = vec![-1.0f32, 0.0, 1.0, 2.0];
        tensor_copy_from_f32(ctx_ptr, x_handle, x.as_ptr(), 4);

        let status = ops_gelu(ctx_ptr, x_handle, y_handle);
        assert!(matches!(status, OpsStatus::Success));

        let mut result = vec![0.0f32; 4];
        tensor_copy_to_f32(ctx_ptr, y_handle, result.as_mut_ptr(), 4);

        // GELU(-1) ≈ -0.159, GELU(0) = 0, GELU(1) ≈ 0.841, GELU(2) ≈ 1.955
        assert!((result[0] - (-0.159)).abs() < 0.01, "GELU(-1) = {}", result[0]);
        assert!(result[1].abs() < 0.001, "GELU(0) = {}", result[1]);
        assert!((result[2] - 0.841).abs() < 0.01, "GELU(1) = {}", result[2]);
        assert!((result[3] - 1.955).abs() < 0.01, "GELU(2) = {}", result[3]);

        println!("✅ GELU test passed: {:?}", result);

        tensor_free(ctx_ptr, x_handle);
        tensor_free(ctx_ptr, y_handle);
    }

    #[test]
    #[serial]
    fn test_reduction_sum() {
        let ctx = match try_create_device() {
            Some(c) => c,
            None => return,
        };
        let ctx_ptr = &ctx as *const DeviceContext;

        let x_handle = tensor_allocate(ctx_ptr, 4, Precision::FP32 as i32);

        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        tensor_copy_from_f32(ctx_ptr, x_handle, x.as_ptr(), 4);

        let mut result: f64 = 0.0;
        let status = ops_sum(ctx_ptr, x_handle, &mut result);
        assert!(matches!(status, OpsStatus::Success));

        assert!((result - 10.0).abs() < 1e-6, "Sum = {}", result);

        println!("✅ Sum test passed: {}", result);

        tensor_free(ctx_ptr, x_handle);
    }

    #[test]
    #[serial]
    fn test_fill() {
        let ctx = match try_create_device() {
            Some(c) => c,
            None => return,
        };
        let ctx_ptr = &ctx as *const DeviceContext;

        let x_handle = tensor_allocate(ctx_ptr, 4, Precision::FP32 as i32);

        let status = ops_fill(ctx_ptr, x_handle, 3.14);
        assert!(matches!(status, OpsStatus::Success));

        let mut result = vec![0.0f32; 4];
        tensor_copy_to_f32(ctx_ptr, x_handle, result.as_mut_ptr(), 4);

        for &v in &result {
            assert!((v - 3.14).abs() < 1e-5, "Fill value = {}", v);
        }

        println!("✅ Fill test passed");

        tensor_free(ctx_ptr, x_handle);
    }

    #[test]
    #[serial]
    fn test_cast_fp32_to_fp16() {
        let ctx = match try_create_device() {
            Some(c) => c,
            None => return,
        };
        let ctx_ptr = &ctx as *const DeviceContext;

        let src_handle = tensor_allocate(ctx_ptr, 4, Precision::FP32 as i32);
        let dst_handle = tensor_allocate(ctx_ptr, 4, Precision::FP16 as i32);

        let src = vec![1.5f32, 2.5, 3.5, 4.5];
        tensor_copy_from_f32(ctx_ptr, src_handle, src.as_ptr(), 4);

        let status = ops_cast(ctx_ptr, src_handle, dst_handle);
        assert!(matches!(status, OpsStatus::Success));

        let mut result = vec![0.0f32; 4];
        tensor_copy_to_f32(ctx_ptr, dst_handle, result.as_mut_ptr(), 4);

        for (i, (&r, &e)) in result.iter().zip(src.iter()).enumerate() {
            assert!((r - e).abs() < 0.01, "Cast mismatch at {}: {} != {}", i, r, e);
        }

        println!("✅ Cast FP32->FP16 test passed");

        tensor_free(ctx_ptr, src_handle);
        tensor_free(ctx_ptr, dst_handle);
    }

    #[test]
    #[serial]
    fn test_axpy() {
        let ctx = match try_create_device() {
            Some(c) => c,
            None => return,
        };
        let ctx_ptr = &ctx as *const DeviceContext;

        let x_handle = tensor_allocate(ctx_ptr, 4, Precision::FP32 as i32);
        let y_handle = tensor_allocate(ctx_ptr, 4, Precision::FP32 as i32);

        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let y = vec![10.0f32, 20.0, 30.0, 40.0];
        tensor_copy_from_f32(ctx_ptr, x_handle, x.as_ptr(), 4);
        tensor_copy_from_f32(ctx_ptr, y_handle, y.as_ptr(), 4);

        // y = y + 2.0 * x
        let status = ops_axpy(ctx_ptr, 2.0, x_handle, y_handle);
        assert!(matches!(status, OpsStatus::Success));

        let mut result = vec![0.0f32; 4];
        tensor_copy_to_f32(ctx_ptr, y_handle, result.as_mut_ptr(), 4);

        let expected = vec![12.0, 24.0, 36.0, 48.0];
        for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((r - e).abs() < 1e-6, "Mismatch at {}: {} != {}", i, r, e);
        }

        println!("✅ AXPY test passed");

        tensor_free(ctx_ptr, x_handle);
        tensor_free(ctx_ptr, y_handle);
    }
}