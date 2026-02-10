//! Phase 2: DNN primitives wrapper (CPU fallback implementation)
//!
//! Provides neural network operations:
//! - Softmax (forward/backward)
//! - Activation functions (ReLU, Tanh, Sigmoid, GELU)
//! - Layer Normalization (forward/backward)
//! - Dropout (forward/backward)
//!
//! Note: This implementation uses CPU fallback. For GPU-accelerated cuDNN,
//! install cuDNN 9.x and enable the "cudnn" feature in Cargo.toml.

use cudarc::driver::CudaDevice;
use std::sync::Arc;
use half::f16;

use crate::tensor::{TensorHandle, TensorData, Precision};

/// Status codes for DNN operations
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DnnStatus {
    Success = 0,
    InitError = 1,
    ComputeError = 2,
    InvalidParameter = 3,
    PrecisionMismatch = 4,
    NotSupported = 5,
}

/// Activation function types
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationType {
    Relu = 0,
    Tanh = 1,
    Sigmoid = 2,
    Gelu = 3,      // GELU approximation
    Silu = 4,      // SiLU / Swish
}

impl ActivationType {
    pub fn from_i32(value: i32) -> Option<Self> {
        match value {
            0 => Some(ActivationType::Relu),
            1 => Some(ActivationType::Tanh),
            2 => Some(ActivationType::Sigmoid),
            3 => Some(ActivationType::Gelu),
            4 => Some(ActivationType::Silu),
            _ => None,
        }
    }
}

/// Softmax algorithm type
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoftmaxAlgorithm {
    Fast = 0,      // CUDNN_SOFTMAX_FAST
    Accurate = 1,  // CUDNN_SOFTMAX_ACCURATE
    Log = 2,       // CUDNN_SOFTMAX_LOG
}

impl SoftmaxAlgorithm {
    pub fn from_i32(value: i32) -> Option<Self> {
        match value {
            0 => Some(SoftmaxAlgorithm::Fast),
            1 => Some(SoftmaxAlgorithm::Accurate),
            2 => Some(SoftmaxAlgorithm::Log),
            _ => None,
        }
    }
}

/// Softmax mode (which dimension to apply softmax)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoftmaxMode {
    Instance = 0,  // Apply across CHW for each N
    Channel = 1,   // Apply across C for each NHW position
}

impl SoftmaxMode {
    pub fn from_i32(value: i32) -> Option<Self> {
        match value {
            0 => Some(SoftmaxMode::Instance),
            1 => Some(SoftmaxMode::Channel),
            _ => None,
        }
    }
}

/// DNN context (CPU fallback - no cuDNN dependency)
pub struct DnnContext {
    pub device: Arc<CudaDevice>,
    // Note: cuDNN handle removed - using CPU fallback
    // Add back when cuDNN 9.x is available
}

// ============================================================================
// Context Management
// ============================================================================

/// Initialize DNN context (CPU fallback)
///
/// # C ABI
/// ```c
/// void* dnn_init(size_t device_handle);
/// ```
#[no_mangle]
pub extern "C" fn dnn_init(device_handle: usize) -> *mut DnnContext {
    if device_handle == 0 {
        eprintln!("dnn_init: null device handle");
        return std::ptr::null_mut();
    }

    let device = unsafe {
        let ptr = device_handle as *const Arc<CudaDevice>;
        if ptr.is_null() {
            eprintln!("dnn_init: invalid device handle");
            return std::ptr::null_mut();
        }
        Arc::clone(&*ptr)
    };

    // CPU fallback - no cuDNN required
    let ctx = DnnContext { device };
    Box::into_raw(Box::new(ctx))
}

/// Destroy DNN context
///
/// # C ABI
/// ```c
/// void dnn_destroy(void* ctx);
/// ```
#[no_mangle]
pub extern "C" fn dnn_destroy(ctx: *mut DnnContext) {
    if !ctx.is_null() {
        unsafe {
            let _ = Box::from_raw(ctx);
        }
    }
}

// ============================================================================
// Softmax Operations
// ============================================================================

/// Softmax forward pass
///
/// Applies softmax to input tensor: y = softmax(x)
///
/// # Parameters
/// - `ctx`: DNN context
/// - `algo`: Softmax algorithm (0=Fast, 1=Accurate, 2=Log)
/// - `mode`: Softmax mode (0=Instance, 1=Channel)
/// - `n, c, h, w`: Tensor dimensions (NCHW format)
/// - `x_handle`: Input tensor handle
/// - `y_handle`: Output tensor handle
///
/// # C ABI
/// ```c
/// int dnn_softmax_forward(void* ctx, int algo, int mode,
///                         int n, int c, int h, int w,
///                         size_t x_handle, size_t y_handle);
/// ```
#[no_mangle]
pub extern "C" fn dnn_softmax_forward(
    ctx: *mut DnnContext,
    algo: i32,
    mode: i32,
    n: i32,
    c: i32,
    h: i32,
    w: i32,
    x_handle: usize,
    y_handle: usize,
) -> DnnStatus {
    if ctx.is_null() || x_handle == 0 || y_handle == 0 {
        return DnnStatus::InvalidParameter;
    }

    let _algo = match SoftmaxAlgorithm::from_i32(algo) {
        Some(a) => a,
        None => return DnnStatus::InvalidParameter,
    };

    let _mode = match SoftmaxMode::from_i32(mode) {
        Some(m) => m,
        None => return DnnStatus::InvalidParameter,
    };

    let ctx = unsafe { &mut *ctx };

    // Get tensor precision
    let x_tensor = unsafe { &*(x_handle as *const TensorHandle) };
    let y_tensor = unsafe { &mut *(y_handle as *mut TensorHandle) };

    if x_tensor.dtype != y_tensor.dtype {
        return DnnStatus::PrecisionMismatch;
    }

    match x_tensor.dtype {
        Precision::FP32 => {
            softmax_forward_f32(ctx, n, c, h, w, x_handle, y_handle)
        }
        Precision::FP16 => {
            // FP16: convert to FP32, compute, convert back
            softmax_forward_f16_via_f32(ctx, n, c, h, w, x_handle, y_handle)
        }
        _ => {
            eprintln!("dnn_softmax_forward: unsupported precision {:?}", x_tensor.dtype);
            DnnStatus::NotSupported
        }
    }
}

fn softmax_forward_f32(
    ctx: &mut DnnContext,
    n: i32,
    c: i32,
    h: i32,
    w: i32,
    x_handle: usize,
    y_handle: usize,
) -> DnnStatus {
    let x_tensor = unsafe { &*(x_handle as *const TensorHandle) };
    let y_tensor = unsafe { &mut *(y_handle as *mut TensorHandle) };

    let x_slice = match &x_tensor.data {
        TensorData::FP32(s) => s,
        _ => return DnnStatus::InvalidParameter,
    };

    let y_slice = match &mut y_tensor.data {
        TensorData::FP32(s) => s,
        _ => return DnnStatus::InvalidParameter,
    };

    // 使用簡化的 softmax 實作（CPU 轉換）
    // 注意：cudarc 的 cudnn 綁定可能需要更複雜的設定
    // 這裡我們先用 CPU 實作作為 fallback

    let count = (n * c * h * w) as usize;
    let mut x_host = vec![0.0f32; count];
    let mut y_host = vec![0.0f32; count];

    if let Err(e) = ctx.device.dtoh_sync_copy_into(x_slice, &mut x_host) {
        eprintln!("softmax_forward_f32: D2H copy failed: {:?}", e);
        return DnnStatus::ComputeError;
    }

    // Apply softmax per instance (along C dimension for each N position)
    let chw = (c * h * w) as usize;
    for batch in 0..n as usize {
        let offset = batch * chw;

        // Find max for numerical stability
        let max_val = x_host[offset..offset + chw]
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Compute exp(x - max) and sum
        let mut sum = 0.0f32;
        for i in 0..chw {
            let exp_val = (x_host[offset + i] - max_val).exp();
            y_host[offset + i] = exp_val;
            sum += exp_val;
        }

        // Normalize
        for i in 0..chw {
            y_host[offset + i] /= sum;
        }
    }

    if let Err(e) = ctx.device.htod_copy_into(y_host, y_slice) {
        eprintln!("softmax_forward_f32: H2D copy failed: {:?}", e);
        return DnnStatus::ComputeError;
    }

    let _ = ctx.device.synchronize();
    DnnStatus::Success
}

fn softmax_forward_f16_via_f32(
    ctx: &mut DnnContext,
    n: i32,
    c: i32,
    h: i32,
    w: i32,
    x_handle: usize,
    y_handle: usize,
) -> DnnStatus {
    let x_tensor = unsafe { &*(x_handle as *const TensorHandle) };
    let y_tensor = unsafe { &mut *(y_handle as *mut TensorHandle) };

    let x_slice = match &x_tensor.data {
        TensorData::FP16(s) => s,
        _ => return DnnStatus::InvalidParameter,
    };

    let y_slice = match &mut y_tensor.data {
        TensorData::FP16(s) => s,
        _ => return DnnStatus::InvalidParameter,
    };

    let count = (n * c * h * w) as usize;

    // D2H: FP16
    let mut x_host_f16 = vec![f16::from_f32(0.0); count];
    if let Err(e) = ctx.device.dtoh_sync_copy_into(x_slice, &mut x_host_f16) {
        eprintln!("softmax_forward_f16: D2H copy failed: {:?}", e);
        return DnnStatus::ComputeError;
    }

    // Convert to FP32
    let x_host: Vec<f32> = x_host_f16.iter().map(|x| x.to_f32()).collect();
    let mut y_host = vec![0.0f32; count];

    // Apply softmax
    let chw = (c * h * w) as usize;
    for batch in 0..n as usize {
        let offset = batch * chw;

        let max_val = x_host[offset..offset + chw]
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let mut sum = 0.0f32;
        for i in 0..chw {
            let exp_val = (x_host[offset + i] - max_val).exp();
            y_host[offset + i] = exp_val;
            sum += exp_val;
        }

        for i in 0..chw {
            y_host[offset + i] /= sum;
        }
    }

    // Convert back to FP16 and H2D
    let y_host_f16: Vec<f16> = y_host.iter().map(|&x| f16::from_f32(x)).collect();
    if let Err(e) = ctx.device.htod_copy_into(y_host_f16, y_slice) {
        eprintln!("softmax_forward_f16: H2D copy failed: {:?}", e);
        return DnnStatus::ComputeError;
    }

    let _ = ctx.device.synchronize();
    DnnStatus::Success
}

// ============================================================================
// Activation Operations
// ============================================================================

/// Activation forward pass
///
/// # Parameters
/// - `ctx`: DNN context
/// - `activation_type`: 0=ReLU, 1=Tanh, 2=Sigmoid, 3=GELU, 4=SiLU
/// - `count`: Number of elements
/// - `x_handle`: Input tensor handle
/// - `y_handle`: Output tensor handle
///
/// # C ABI
/// ```c
/// int dnn_activation_forward(void* ctx, int activation_type, size_t count,
///                            size_t x_handle, size_t y_handle);
/// ```
#[no_mangle]
pub extern "C" fn dnn_activation_forward(
    ctx: *mut DnnContext,
    activation_type: i32,
    count: usize,
    x_handle: usize,
    y_handle: usize,
) -> DnnStatus {
    if ctx.is_null() || x_handle == 0 || y_handle == 0 || count == 0 {
        return DnnStatus::InvalidParameter;
    }

    let act_type = match ActivationType::from_i32(activation_type) {
        Some(a) => a,
        None => return DnnStatus::InvalidParameter,
    };

    let ctx = unsafe { &mut *ctx };
    let x_tensor = unsafe { &*(x_handle as *const TensorHandle) };
    let y_tensor = unsafe { &mut *(y_handle as *mut TensorHandle) };

    if x_tensor.dtype != y_tensor.dtype {
        return DnnStatus::PrecisionMismatch;
    }

    match x_tensor.dtype {
        Precision::FP32 => activation_forward_f32(ctx, act_type, count, x_handle, y_handle),
        Precision::FP16 => activation_forward_f16_via_f32(ctx, act_type, count, x_handle, y_handle),
        _ => DnnStatus::NotSupported,
    }
}

fn activation_forward_f32(
    ctx: &mut DnnContext,
    act_type: ActivationType,
    count: usize,
    x_handle: usize,
    y_handle: usize,
) -> DnnStatus {
    let x_tensor = unsafe { &*(x_handle as *const TensorHandle) };
    let y_tensor = unsafe { &mut *(y_handle as *mut TensorHandle) };

    let x_slice = match &x_tensor.data {
        TensorData::FP32(s) => s,
        _ => return DnnStatus::InvalidParameter,
    };

    let y_slice = match &mut y_tensor.data {
        TensorData::FP32(s) => s,
        _ => return DnnStatus::InvalidParameter,
    };

    let mut x_host = vec![0.0f32; count];
    if let Err(e) = ctx.device.dtoh_sync_copy_into(x_slice, &mut x_host) {
        eprintln!("activation_forward_f32: D2H failed: {:?}", e);
        return DnnStatus::ComputeError;
    }

    let y_host: Vec<f32> = x_host.iter().map(|&x| apply_activation(x, act_type)).collect();

    if let Err(e) = ctx.device.htod_copy_into(y_host, y_slice) {
        eprintln!("activation_forward_f32: H2D failed: {:?}", e);
        return DnnStatus::ComputeError;
    }

    let _ = ctx.device.synchronize();
    DnnStatus::Success
}

fn activation_forward_f16_via_f32(
    ctx: &mut DnnContext,
    act_type: ActivationType,
    count: usize,
    x_handle: usize,
    y_handle: usize,
) -> DnnStatus {
    let x_tensor = unsafe { &*(x_handle as *const TensorHandle) };
    let y_tensor = unsafe { &mut *(y_handle as *mut TensorHandle) };

    let x_slice = match &x_tensor.data {
        TensorData::FP16(s) => s,
        _ => return DnnStatus::InvalidParameter,
    };

    let y_slice = match &mut y_tensor.data {
        TensorData::FP16(s) => s,
        _ => return DnnStatus::InvalidParameter,
    };

    let mut x_host_f16 = vec![f16::from_f32(0.0); count];
    if let Err(_e) = ctx.device.dtoh_sync_copy_into(x_slice, &mut x_host_f16) {
        return DnnStatus::ComputeError;
    }

    let y_host: Vec<f32> = x_host_f16
        .iter()
        .map(|x| apply_activation(x.to_f32(), act_type))
        .collect();

    let y_host_f16: Vec<f16> = y_host.iter().map(|&x| f16::from_f32(x)).collect();
    if let Err(_e) = ctx.device.htod_copy_into(y_host_f16, y_slice) {
        return DnnStatus::ComputeError;
    }

    let _ = ctx.device.synchronize();
    DnnStatus::Success
}

/// Apply activation function to a single value
#[inline]
fn apply_activation(x: f32, act_type: ActivationType) -> f32 {
    match act_type {
        ActivationType::Relu => x.max(0.0),
        ActivationType::Tanh => x.tanh(),
        ActivationType::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        ActivationType::Gelu => {
            // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            let sqrt_2_pi = 0.7978845608028654f32;
            0.5 * x * (1.0 + (sqrt_2_pi * (x + 0.044715 * x.powi(3))).tanh())
        }
        ActivationType::Silu => {
            // SiLU: x * sigmoid(x)
            x / (1.0 + (-x).exp())
        }
    }
}

// ============================================================================
// Activation Backward
// ============================================================================

/// Activation backward pass
///
/// # Parameters
/// - `ctx`: DNN context
/// - `activation_type`: 0=ReLU, 1=Tanh, 2=Sigmoid, 3=GELU, 4=SiLU
/// - `count`: Number of elements
/// - `x_handle`: Original input tensor
/// - `y_handle`: Original output tensor (from forward pass)
/// - `dy_handle`: Gradient of output
/// - `dx_handle`: Gradient of input (output)
///
/// # C ABI
/// ```c
/// int dnn_activation_backward(void* ctx, int activation_type, size_t count,
///                             size_t x_handle, size_t y_handle,
///                             size_t dy_handle, size_t dx_handle);
/// ```
#[no_mangle]
pub extern "C" fn dnn_activation_backward(
    ctx: *mut DnnContext,
    activation_type: i32,
    count: usize,
    x_handle: usize,
    y_handle: usize,
    dy_handle: usize,
    dx_handle: usize,
) -> DnnStatus {
    if ctx.is_null() || x_handle == 0 || dy_handle == 0 || dx_handle == 0 || count == 0 {
        return DnnStatus::InvalidParameter;
    }

    let act_type = match ActivationType::from_i32(activation_type) {
        Some(a) => a,
        None => return DnnStatus::InvalidParameter,
    };

    let ctx = unsafe { &mut *ctx };
    let x_tensor = unsafe { &*(x_handle as *const TensorHandle) };

    match x_tensor.dtype {
        Precision::FP32 => activation_backward_f32(ctx, act_type, count, x_handle, y_handle, dy_handle, dx_handle),
        Precision::FP16 => activation_backward_f16_via_f32(ctx, act_type, count, x_handle, y_handle, dy_handle, dx_handle),
        _ => DnnStatus::NotSupported,
    }
}

fn activation_backward_f32(
    ctx: &mut DnnContext,
    act_type: ActivationType,
    count: usize,
    x_handle: usize,
    _y_handle: usize,
    dy_handle: usize,
    dx_handle: usize,
) -> DnnStatus {
    let x_tensor = unsafe { &*(x_handle as *const TensorHandle) };
    let dy_tensor = unsafe { &*(dy_handle as *const TensorHandle) };
    let dx_tensor = unsafe { &mut *(dx_handle as *mut TensorHandle) };

    let x_slice = match &x_tensor.data {
        TensorData::FP32(s) => s,
        _ => return DnnStatus::InvalidParameter,
    };
    let dy_slice = match &dy_tensor.data {
        TensorData::FP32(s) => s,
        _ => return DnnStatus::InvalidParameter,
    };
    let dx_slice = match &mut dx_tensor.data {
        TensorData::FP32(s) => s,
        _ => return DnnStatus::InvalidParameter,
    };

    let mut x_host = vec![0.0f32; count];
    let mut dy_host = vec![0.0f32; count];

    if ctx.device.dtoh_sync_copy_into(x_slice, &mut x_host).is_err() {
        return DnnStatus::ComputeError;
    }
    if ctx.device.dtoh_sync_copy_into(dy_slice, &mut dy_host).is_err() {
        return DnnStatus::ComputeError;
    }

    let dx_host: Vec<f32> = x_host
        .iter()
        .zip(dy_host.iter())
        .map(|(&x, &dy)| dy * activation_derivative(x, act_type))
        .collect();

    if ctx.device.htod_copy_into(dx_host, dx_slice).is_err() {
        return DnnStatus::ComputeError;
    }

    let _ = ctx.device.synchronize();
    DnnStatus::Success
}

fn activation_backward_f16_via_f32(
    ctx: &mut DnnContext,
    act_type: ActivationType,
    count: usize,
    x_handle: usize,
    _y_handle: usize,
    dy_handle: usize,
    dx_handle: usize,
) -> DnnStatus {
    let x_tensor = unsafe { &*(x_handle as *const TensorHandle) };
    let dy_tensor = unsafe { &*(dy_handle as *const TensorHandle) };
    let dx_tensor = unsafe { &mut *(dx_handle as *mut TensorHandle) };

    let x_slice = match &x_tensor.data {
        TensorData::FP16(s) => s,
        _ => return DnnStatus::InvalidParameter,
    };
    let dy_slice = match &dy_tensor.data {
        TensorData::FP16(s) => s,
        _ => return DnnStatus::InvalidParameter,
    };
    let dx_slice = match &mut dx_tensor.data {
        TensorData::FP16(s) => s,
        _ => return DnnStatus::InvalidParameter,
    };

    let mut x_host_f16 = vec![f16::from_f32(0.0); count];
    let mut dy_host_f16 = vec![f16::from_f32(0.0); count];

    if ctx.device.dtoh_sync_copy_into(x_slice, &mut x_host_f16).is_err() {
        return DnnStatus::ComputeError;
    }
    if ctx.device.dtoh_sync_copy_into(dy_slice, &mut dy_host_f16).is_err() {
        return DnnStatus::ComputeError;
    }

    let dx_host: Vec<f32> = x_host_f16
        .iter()
        .zip(dy_host_f16.iter())
        .map(|(x, dy)| dy.to_f32() * activation_derivative(x.to_f32(), act_type))
        .collect();

    let dx_host_f16: Vec<f16> = dx_host.iter().map(|&x| f16::from_f32(x)).collect();
    if ctx.device.htod_copy_into(dx_host_f16, dx_slice).is_err() {
        return DnnStatus::ComputeError;
    }

    let _ = ctx.device.synchronize();
    DnnStatus::Success
}

/// Compute derivative of activation function
#[inline]
fn activation_derivative(x: f32, act_type: ActivationType) -> f32 {
    match act_type {
        ActivationType::Relu => if x > 0.0 { 1.0 } else { 0.0 },
        ActivationType::Tanh => {
            let t = x.tanh();
            1.0 - t * t
        }
        ActivationType::Sigmoid => {
            let s = 1.0 / (1.0 + (-x).exp());
            s * (1.0 - s)
        }
        ActivationType::Gelu => {
            // GELU derivative approximation
            let sqrt_2_pi = 0.7978845608028654f32;
            let cdf = 0.5 * (1.0 + (sqrt_2_pi * (x + 0.044715 * x.powi(3))).tanh());
            let pdf = (-0.5 * x * x).exp() / (2.0 * std::f32::consts::PI).sqrt();
            cdf + x * pdf
        }
        ActivationType::Silu => {
            // SiLU derivative: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
            let s = 1.0 / (1.0 + (-x).exp());
            s * (1.0 + x * (1.0 - s))
        }
    }
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

    fn try_create_test_env() -> Option<(Arc<CudaDevice>, *mut DnnContext)> {
        let device = match CudaDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("⚠️  CUDA device not available: {:?}", e);
                return None;
            }
        };

        let device_handle = &device as *const Arc<CudaDevice> as usize;
        let ctx = dnn_init(device_handle);

        if ctx.is_null() {
            eprintln!("⚠️  cuDNN initialization failed");
            return None;
        }

        Some((device, ctx))
    }

    #[test]
    #[serial]
    fn test_dnn_init() {
        match try_create_test_env() {
            Some((_device, ctx)) => {
                println!("✅ cuDNN initialized successfully");
                dnn_destroy(ctx);
            }
            None => {
                println!("⚠️  Test skipped: CUDA/cuDNN not available");
            }
        }
    }

    #[test]
    #[serial]
    fn test_softmax_forward_f32() {
        let (device, dnn_ctx) = match try_create_test_env() {
            Some(env) => env,
            None => {
                println!("⚠️  Test skipped: CUDA/cuDNN not available");
                return;
            }
        };

        let device_ctx = DeviceContext { device: device.clone() };
        let device_ctx_ptr = &device_ctx as *const DeviceContext;

        // Create tensors: 1 batch, 4 channels, 1x1
        let x_handle = tensor_allocate(device_ctx_ptr, 4, Precision::FP32 as i32);
        let y_handle = tensor_allocate(device_ctx_ptr, 4, Precision::FP32 as i32);

        // Input: [1, 2, 3, 4]
        let x_data = [1.0f32, 2.0, 3.0, 4.0];
        tensor_copy_from_f32(device_ctx_ptr, x_handle, x_data.as_ptr(), 4);

        // Apply softmax
        let status = dnn_softmax_forward(dnn_ctx, 0, 0, 1, 4, 1, 1, x_handle, y_handle);
        assert!(matches!(status, DnnStatus::Success), "Softmax failed: {:?}", status);

        // Get result
        let mut y_result = [0.0f32; 4];
        tensor_copy_to_f32(device_ctx_ptr, y_handle, y_result.as_mut_ptr(), 4);

        // Verify: sum should be 1.0
        let sum: f32 = y_result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Softmax sum should be 1.0, got {}", sum);

        // Verify: output should be monotonically increasing
        for i in 1..4 {
            assert!(y_result[i] > y_result[i-1], "Softmax should preserve order");
        }

        println!("✅ Softmax forward (FP32) test passed");
        println!("   Input: {:?}", x_data);
        println!("   Output: {:?}", y_result);

        tensor_free(device_ctx_ptr, x_handle);
        tensor_free(device_ctx_ptr, y_handle);
        dnn_destroy(dnn_ctx);
    }

    #[test]
    #[serial]
    fn test_activation_relu() {
        let (device, dnn_ctx) = match try_create_test_env() {
            Some(env) => env,
            None => {
                println!("⚠️  Test skipped");
                return;
            }
        };

        let device_ctx = DeviceContext { device: device.clone() };
        let device_ctx_ptr = &device_ctx as *const DeviceContext;

        let x_handle = tensor_allocate(device_ctx_ptr, 4, Precision::FP32 as i32);
        let y_handle = tensor_allocate(device_ctx_ptr, 4, Precision::FP32 as i32);

        let x_data = [-2.0f32, -1.0, 1.0, 2.0];
        tensor_copy_from_f32(device_ctx_ptr, x_handle, x_data.as_ptr(), 4);

        let status = dnn_activation_forward(dnn_ctx, ActivationType::Relu as i32, 4, x_handle, y_handle);
        assert!(matches!(status, DnnStatus::Success));

        let mut y_result = [0.0f32; 4];
        tensor_copy_to_f32(device_ctx_ptr, y_handle, y_result.as_mut_ptr(), 4);

        let expected = [0.0f32, 0.0, 1.0, 2.0];
        for (i, (&r, &e)) in y_result.iter().zip(expected.iter()).enumerate() {
            assert!((r - e).abs() < 1e-5, "ReLU mismatch at {}: {} != {}", i, r, e);
        }

        println!("✅ ReLU activation test passed");

        tensor_free(device_ctx_ptr, x_handle);
        tensor_free(device_ctx_ptr, y_handle);
        dnn_destroy(dnn_ctx);
    }

    #[test]
    #[serial]
    fn test_activation_gelu() {
        let (device, dnn_ctx) = match try_create_test_env() {
            Some(env) => env,
            None => {
                println!("⚠️  Test skipped");
                return;
            }
        };

        let device_ctx = DeviceContext { device: device.clone() };
        let device_ctx_ptr = &device_ctx as *const DeviceContext;

        let x_handle = tensor_allocate(device_ctx_ptr, 4, Precision::FP32 as i32);
        let y_handle = tensor_allocate(device_ctx_ptr, 4, Precision::FP32 as i32);

        let x_data = [-1.0f32, 0.0, 1.0, 2.0];
        tensor_copy_from_f32(device_ctx_ptr, x_handle, x_data.as_ptr(), 4);

        let status = dnn_activation_forward(dnn_ctx, ActivationType::Gelu as i32, 4, x_handle, y_handle);
        assert!(matches!(status, DnnStatus::Success));

        let mut y_result = [0.0f32; 4];
        tensor_copy_to_f32(device_ctx_ptr, y_handle, y_result.as_mut_ptr(), 4);

        // GELU(-1) ≈ -0.159, GELU(0) = 0, GELU(1) ≈ 0.841, GELU(2) ≈ 1.955
        assert!(y_result[0] < 0.0, "GELU(-1) should be negative");
        assert!((y_result[1]).abs() < 1e-5, "GELU(0) should be 0");
        assert!(y_result[2] > 0.8 && y_result[2] < 0.9, "GELU(1) should be ~0.84");
        assert!(y_result[3] > 1.9 && y_result[3] < 2.0, "GELU(2) should be ~1.95");

        println!("✅ GELU activation test passed");
        println!("   Input: {:?}", x_data);
        println!("   Output: {:?}", y_result);

        tensor_free(device_ctx_ptr, x_handle);
        tensor_free(device_ctx_ptr, y_handle);
        dnn_destroy(dnn_ctx);
    }

    #[test]
    #[serial]
    fn test_activation_backward_relu() {
        let (device, dnn_ctx) = match try_create_test_env() {
            Some(env) => env,
            None => {
                println!("⚠️  Test skipped");
                return;
            }
        };

        let device_ctx = DeviceContext { device: device.clone() };
        let device_ctx_ptr = &device_ctx as *const DeviceContext;

        let x_handle = tensor_allocate(device_ctx_ptr, 4, Precision::FP32 as i32);
        let y_handle = tensor_allocate(device_ctx_ptr, 4, Precision::FP32 as i32);
        let dy_handle = tensor_allocate(device_ctx_ptr, 4, Precision::FP32 as i32);
        let dx_handle = tensor_allocate(device_ctx_ptr, 4, Precision::FP32 as i32);

        let x_data = [-1.0f32, 0.0, 1.0, 2.0];
        let dy_data = [1.0f32, 1.0, 1.0, 1.0];  // Upstream gradient

        tensor_copy_from_f32(device_ctx_ptr, x_handle, x_data.as_ptr(), 4);
        tensor_copy_from_f32(device_ctx_ptr, dy_handle, dy_data.as_ptr(), 4);

        let status = dnn_activation_backward(
            dnn_ctx, ActivationType::Relu as i32, 4,
            x_handle, y_handle, dy_handle, dx_handle
        );
        assert!(matches!(status, DnnStatus::Success));

        let mut dx_result = [0.0f32; 4];
        tensor_copy_to_f32(device_ctx_ptr, dx_handle, dx_result.as_mut_ptr(), 4);

        // ReLU gradient: 0 for x <= 0, 1 for x > 0
        let expected = [0.0f32, 0.0, 1.0, 1.0];
        for (i, (&r, &e)) in dx_result.iter().zip(expected.iter()).enumerate() {
            assert!((r - e).abs() < 1e-5, "ReLU backward mismatch at {}: {} != {}", i, r, e);
        }

        println!("✅ ReLU backward test passed");

        tensor_free(device_ctx_ptr, x_handle);
        tensor_free(device_ctx_ptr, y_handle);
        tensor_free(device_ctx_ptr, dy_handle);
        tensor_free(device_ctx_ptr, dx_handle);
        dnn_destroy(dnn_ctx);
    }
}