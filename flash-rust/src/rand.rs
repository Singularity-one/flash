//! Phase 3: cuRAND wrapper for random number generation
//!
//! Provides GPU-accelerated random number generation:
//! - Uniform distribution [0, 1)
//! - Normal (Gaussian) distribution
//! - Seed management
//!
//! Note: cuRAND generates FP32 natively. For FP16/BF16, use tensor conversion.

use cudarc::driver::CudaDevice;
use cudarc::curand::{CudaRng, result::CurandError};
use std::sync::Arc;
use half::f16;

use crate::tensor::{TensorHandle, TensorData, Precision};
use crate::device::DeviceContext;

/// Status codes for RAND operations
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RandStatus {
    Success = 0,
    InitError = 1,
    GenerateError = 2,
    InvalidParameter = 3,
    PrecisionNotSupported = 4,
}

/// cuRAND context
pub struct RandContext {
    pub device: Arc<CudaDevice>,
    pub rng: CudaRng,
}

// ============================================================================
// Context Management
// ============================================================================

/// Initialize cuRAND context with default seed
///
/// # C ABI
/// ```c
/// void* rand_init(size_t device_handle);
/// ```
#[no_mangle]
pub extern "C" fn rand_init(device_handle: usize) -> *mut RandContext {
    rand_init_with_seed(device_handle, 0)
}

/// Initialize cuRAND context with specified seed
///
/// # C ABI
/// ```c
/// void* rand_init_with_seed(size_t device_handle, uint64_t seed);
/// ```
#[no_mangle]
pub extern "C" fn rand_init_with_seed(device_handle: usize, seed: u64) -> *mut RandContext {
    if device_handle == 0 {
        eprintln!("rand_init: null device handle");
        return std::ptr::null_mut();
    }

    let device = unsafe {
        let ptr = device_handle as *const Arc<CudaDevice>;
        if ptr.is_null() {
            eprintln!("rand_init: invalid device handle");
            return std::ptr::null_mut();
        }
        Arc::clone(&*ptr)
    };

    match CudaRng::new(seed, device.clone()) {
        Ok(rng) => {
            let ctx = RandContext { device, rng };
            Box::into_raw(Box::new(ctx))
        }
        Err(e) => {
            eprintln!("rand_init error: {:?}", e);
            std::ptr::null_mut()
        }
    }
}

/// Destroy cuRAND context
///
/// # C ABI
/// ```c
/// void rand_destroy(void* ctx);
/// ```
#[no_mangle]
pub extern "C" fn rand_destroy(ctx: *mut RandContext) {
    if !ctx.is_null() {
        unsafe {
            let _ = Box::from_raw(ctx);
        }
    }
}

/// Set random seed
///
/// # C ABI
/// ```c
/// int rand_set_seed(void* ctx, uint64_t seed);
/// ```
#[no_mangle]
pub extern "C" fn rand_set_seed(ctx: *mut RandContext, seed: u64) -> RandStatus {
    if ctx.is_null() {
        return RandStatus::InvalidParameter;
    }

    let ctx = unsafe { &mut *ctx };

    // Create a new RNG with the new seed
    match CudaRng::new(seed, ctx.device.clone()) {
        Ok(rng) => {
            ctx.rng = rng;
            RandStatus::Success
        }
        Err(e) => {
            eprintln!("rand_set_seed error: {:?}", e);
            RandStatus::InitError
        }
    }
}

// ============================================================================
// Uniform Distribution
// ============================================================================

/// Generate uniform random numbers in [0, 1)
///
/// Fills the tensor with uniformly distributed random values.
/// For FP16/BF16 tensors, generates FP32 internally then converts.
///
/// # C ABI
/// ```c
/// int rand_uniform(void* ctx, size_t tensor_handle);
/// ```
#[no_mangle]
pub extern "C" fn rand_uniform(ctx: *mut RandContext, tensor_handle: usize) -> RandStatus {
    if ctx.is_null() || tensor_handle == 0 {
        return RandStatus::InvalidParameter;
    }

    let ctx = unsafe { &mut *ctx };
    let tensor = unsafe { &mut *(tensor_handle as *mut TensorHandle) };

    match tensor.dtype {
        Precision::FP32 => uniform_f32(ctx, tensor),
        Precision::FP16 => uniform_f16_via_f32(ctx, tensor),
        Precision::FP64 => uniform_f64(ctx, tensor),
        Precision::BF16 => uniform_bf16_via_f32(ctx, tensor),
        _ => {
            eprintln!("rand_uniform: unsupported precision {:?}", tensor.dtype);
            RandStatus::PrecisionNotSupported
        }
    }
}

fn uniform_f32(ctx: &mut RandContext, tensor: &mut TensorHandle) -> RandStatus {
    let slice = match &mut tensor.data {
        TensorData::FP32(s) => s,
        _ => return RandStatus::InvalidParameter,
    };

    match ctx.rng.fill_with_uniform(slice) {
        Ok(_) => {
            let _ = ctx.device.synchronize();
            RandStatus::Success
        }
        Err(e) => {
            eprintln!("rand_uniform_f32 error: {:?}", e);
            RandStatus::GenerateError
        }
    }
}

fn uniform_f64(ctx: &mut RandContext, tensor: &mut TensorHandle) -> RandStatus {
    let slice = match &mut tensor.data {
        TensorData::FP64(s) => s,
        _ => return RandStatus::InvalidParameter,
    };

    match ctx.rng.fill_with_uniform(slice) {
        Ok(_) => {
            let _ = ctx.device.synchronize();
            RandStatus::Success
        }
        Err(e) => {
            eprintln!("rand_uniform_f64 error: {:?}", e);
            RandStatus::GenerateError
        }
    }
}

fn uniform_f16_via_f32(ctx: &mut RandContext, tensor: &mut TensorHandle) -> RandStatus {
    let count = tensor.element_count;

    // Generate FP32 on device
    let mut temp_f32 = match ctx.device.alloc_zeros::<f32>(count) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("rand_uniform_f16: alloc failed: {:?}", e);
            return RandStatus::GenerateError;
        }
    };

    if let Err(e) = ctx.rng.fill_with_uniform(&mut temp_f32) {
        eprintln!("rand_uniform_f16: generate failed: {:?}", e);
        return RandStatus::GenerateError;
    }

    // Copy to host, convert, copy back
    let mut host_f32 = vec![0.0f32; count];
    if let Err(e) = ctx.device.dtoh_sync_copy_into(&temp_f32, &mut host_f32) {
        return RandStatus::GenerateError;
    }

    let host_f16: Vec<f16> = host_f32.iter().map(|&x| f16::from_f32(x)).collect();

    let slice = match &mut tensor.data {
        TensorData::FP16(s) => s,
        _ => return RandStatus::InvalidParameter,
    };

    if let Err(e) = ctx.device.htod_copy_into(host_f16, slice) {
        return RandStatus::GenerateError;
    }

    let _ = ctx.device.synchronize();
    RandStatus::Success
}

fn uniform_bf16_via_f32(ctx: &mut RandContext, tensor: &mut TensorHandle) -> RandStatus {
    let count = tensor.element_count;

    let mut temp_f32 = match ctx.device.alloc_zeros::<f32>(count) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("rand_uniform_bf16: alloc failed: {:?}", e);
            return RandStatus::GenerateError;
        }
    };

    if let Err(e) = ctx.rng.fill_with_uniform(&mut temp_f32) {
        return RandStatus::GenerateError;
    }

    let mut host_f32 = vec![0.0f32; count];
    if let Err(e) = ctx.device.dtoh_sync_copy_into(&temp_f32, &mut host_f32) {
        return RandStatus::GenerateError;
    }

    let host_bf16: Vec<half::bf16> = host_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();

    let slice = match &mut tensor.data {
        TensorData::BF16(s) => s,
        _ => return RandStatus::InvalidParameter,
    };

    if let Err(e) = ctx.device.htod_copy_into(host_bf16, slice) {
        return RandStatus::GenerateError;
    }

    let _ = ctx.device.synchronize();
    RandStatus::Success
}

// ============================================================================
// Normal Distribution
// ============================================================================

/// Generate normal (Gaussian) random numbers
///
/// Fills the tensor with normally distributed random values.
///
/// # Parameters
/// - `ctx`: RAND context
/// - `tensor_handle`: Output tensor
/// - `mean`: Mean of the distribution
/// - `stddev`: Standard deviation
///
/// # C ABI
/// ```c
/// int rand_normal(void* ctx, size_t tensor_handle, double mean, double stddev);
/// ```
#[no_mangle]
pub extern "C" fn rand_normal(
    ctx: *mut RandContext,
    tensor_handle: usize,
    mean: f64,
    stddev: f64,
) -> RandStatus {
    if ctx.is_null() || tensor_handle == 0 {
        return RandStatus::InvalidParameter;
    }

    if stddev <= 0.0 {
        eprintln!("rand_normal: stddev must be positive");
        return RandStatus::InvalidParameter;
    }

    let ctx = unsafe { &mut *ctx };
    let tensor = unsafe { &mut *(tensor_handle as *mut TensorHandle) };

    match tensor.dtype {
        Precision::FP32 => normal_f32(ctx, tensor, mean as f32, stddev as f32),
        Precision::FP16 => normal_f16_via_f32(ctx, tensor, mean as f32, stddev as f32),
        Precision::FP64 => normal_f64(ctx, tensor, mean, stddev),
        Precision::BF16 => normal_bf16_via_f32(ctx, tensor, mean as f32, stddev as f32),
        _ => {
            eprintln!("rand_normal: unsupported precision {:?}", tensor.dtype);
            RandStatus::PrecisionNotSupported
        }
    }
}

fn normal_f32(ctx: &mut RandContext, tensor: &mut TensorHandle, mean: f32, stddev: f32) -> RandStatus {
    let slice = match &mut tensor.data {
        TensorData::FP32(s) => s,
        _ => return RandStatus::InvalidParameter,
    };

    match ctx.rng.fill_with_normal(slice, mean, stddev) {
        Ok(_) => {
            let _ = ctx.device.synchronize();
            RandStatus::Success
        }
        Err(e) => {
            eprintln!("rand_normal_f32 error: {:?}", e);
            RandStatus::GenerateError
        }
    }
}

fn normal_f64(ctx: &mut RandContext, tensor: &mut TensorHandle, mean: f64, stddev: f64) -> RandStatus {
    let slice = match &mut tensor.data {
        TensorData::FP64(s) => s,
        _ => return RandStatus::InvalidParameter,
    };

    match ctx.rng.fill_with_normal(slice, mean, stddev) {
        Ok(_) => {
            let _ = ctx.device.synchronize();
            RandStatus::Success
        }
        Err(e) => {
            eprintln!("rand_normal_f64 error: {:?}", e);
            RandStatus::GenerateError
        }
    }
}

fn normal_f16_via_f32(ctx: &mut RandContext, tensor: &mut TensorHandle, mean: f32, stddev: f32) -> RandStatus {
    let count = tensor.element_count;

    let mut temp_f32 = match ctx.device.alloc_zeros::<f32>(count) {
        Ok(s) => s,
        Err(_) => return RandStatus::GenerateError,
    };

    if let Err(_) = ctx.rng.fill_with_normal(&mut temp_f32, mean, stddev) {
        return RandStatus::GenerateError;
    }

    let mut host_f32 = vec![0.0f32; count];
    if let Err(_) = ctx.device.dtoh_sync_copy_into(&temp_f32, &mut host_f32) {
        return RandStatus::GenerateError;
    }

    let host_f16: Vec<f16> = host_f32.iter().map(|&x| f16::from_f32(x)).collect();

    let slice = match &mut tensor.data {
        TensorData::FP16(s) => s,
        _ => return RandStatus::InvalidParameter,
    };

    if let Err(_) = ctx.device.htod_copy_into(host_f16, slice) {
        return RandStatus::GenerateError;
    }

    let _ = ctx.device.synchronize();
    RandStatus::Success
}

fn normal_bf16_via_f32(ctx: &mut RandContext, tensor: &mut TensorHandle, mean: f32, stddev: f32) -> RandStatus {
    let count = tensor.element_count;

    let mut temp_f32 = match ctx.device.alloc_zeros::<f32>(count) {
        Ok(s) => s,
        Err(_) => return RandStatus::GenerateError,
    };

    if let Err(_) = ctx.rng.fill_with_normal(&mut temp_f32, mean, stddev) {
        return RandStatus::GenerateError;
    }

    let mut host_f32 = vec![0.0f32; count];
    if let Err(_) = ctx.device.dtoh_sync_copy_into(&temp_f32, &mut host_f32) {
        return RandStatus::GenerateError;
    }

    let host_bf16: Vec<half::bf16> = host_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();

    let slice = match &mut tensor.data {
        TensorData::BF16(s) => s,
        _ => return RandStatus::InvalidParameter,
    };

    if let Err(_) = ctx.device.htod_copy_into(host_bf16, slice) {
        return RandStatus::GenerateError;
    }

    let _ = ctx.device.synchronize();
    RandStatus::Success
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Generate uniform random numbers in a custom range [low, high)
///
/// # C ABI
/// ```c
/// int rand_uniform_range(void* ctx, size_t tensor_handle, double low, double high);
/// ```
#[no_mangle]
pub extern "C" fn rand_uniform_range(
    ctx: *mut RandContext,
    tensor_handle: usize,
    low: f64,
    high: f64,
) -> RandStatus {
    if ctx.is_null() || tensor_handle == 0 {
        return RandStatus::InvalidParameter;
    }

    if low >= high {
        eprintln!("rand_uniform_range: low must be less than high");
        return RandStatus::InvalidParameter;
    }

    // First generate [0, 1)
    let status = rand_uniform(ctx, tensor_handle);
    if status != RandStatus::Success {
        return status;
    }

    // Then scale to [low, high): x = low + x * (high - low)
    let ctx = unsafe { &*ctx };
    let tensor = unsafe { &mut *(tensor_handle as *mut TensorHandle) };
    let scale = high - low;

    match &mut tensor.data {
        TensorData::FP32(slice) => {
            let mut host = vec![0.0f32; tensor.element_count];
            if let Err(_) = ctx.device.dtoh_sync_copy_into(slice, &mut host) {
                return RandStatus::GenerateError;
            }
            for x in &mut host {
                *x = (low as f32) + *x * (scale as f32);
            }
            if let Err(_) = ctx.device.htod_copy_into(host, slice) {
                return RandStatus::GenerateError;
            }
        }
        TensorData::FP64(slice) => {
            let mut host = vec![0.0f64; tensor.element_count];
            if let Err(_) = ctx.device.dtoh_sync_copy_into(slice, &mut host) {
                return RandStatus::GenerateError;
            }
            for x in &mut host {
                *x = low + *x * scale;
            }
            if let Err(_) = ctx.device.htod_copy_into(host, slice) {
                return RandStatus::GenerateError;
            }
        }
        TensorData::FP16(slice) => {
            let mut host = vec![f16::from_f32(0.0); tensor.element_count];
            if let Err(_) = ctx.device.dtoh_sync_copy_into(slice, &mut host) {
                return RandStatus::GenerateError;
            }
            let scaled: Vec<f16> = host.iter()
                .map(|x| f16::from_f32((low as f32) + x.to_f32() * (scale as f32)))
                .collect();
            if let Err(_) = ctx.device.htod_copy_into(scaled, slice) {
                return RandStatus::GenerateError;
            }
        }
        TensorData::BF16(slice) => {
            let mut host = vec![half::bf16::from_f32(0.0); tensor.element_count];
            if let Err(_) = ctx.device.dtoh_sync_copy_into(slice, &mut host) {
                return RandStatus::GenerateError;
            }
            let scaled: Vec<half::bf16> = host.iter()
                .map(|x| half::bf16::from_f32((low as f32) + x.to_f32() * (scale as f32)))
                .collect();
            if let Err(_) = ctx.device.htod_copy_into(scaled, slice) {
                return RandStatus::GenerateError;
            }
        }
        _ => return RandStatus::PrecisionNotSupported,
    }

    let _ = ctx.device.synchronize();
    RandStatus::Success
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{tensor_allocate, tensor_free, tensor_copy_to_f32};
    use serial_test::serial;

    fn try_create_test_env() -> Option<(Arc<CudaDevice>, *mut RandContext)> {
        let device = match CudaDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("⚠️  CUDA device not available: {:?}", e);
                return None;
            }
        };

        let device_handle = &device as *const Arc<CudaDevice> as usize;
        let ctx = rand_init_with_seed(device_handle, 12345);

        if ctx.is_null() {
            eprintln!("⚠️  cuRAND initialization failed");
            return None;
        }

        Some((device, ctx))
    }

    #[test]
    #[serial]
    fn test_rand_init() {
        match try_create_test_env() {
            Some((_device, ctx)) => {
                println!("✅ cuRAND initialized successfully");
                rand_destroy(ctx);
            }
            None => {
                println!("⚠️  Test skipped: CUDA/cuRAND not available");
            }
        }
    }

    #[test]
    #[serial]
    fn test_uniform_f32() {
        let (device, rand_ctx) = match try_create_test_env() {
            Some(env) => env,
            None => {
                println!("⚠️  Test skipped");
                return;
            }
        };

        let device_ctx = DeviceContext { device: device.clone() };
        let device_ctx_ptr = &device_ctx as *const DeviceContext;

        let tensor_handle = tensor_allocate(device_ctx_ptr, 1000, Precision::FP32 as i32);
        assert_ne!(tensor_handle, 0);

        let status = rand_uniform(rand_ctx, tensor_handle);
        assert!(matches!(status, RandStatus::Success), "Uniform generation failed");

        // Verify values are in [0, 1)
        let mut result = vec![0.0f32; 1000];
        tensor_copy_to_f32(device_ctx_ptr, tensor_handle, result.as_mut_ptr(), 1000);

        let mut in_range = true;
        let mut sum = 0.0f32;
        for &x in &result {
            if x < 0.0 || x >= 1.0 {
                in_range = false;
                break;
            }
            sum += x;
        }
        assert!(in_range, "Some values out of range [0, 1)");

        // Mean should be approximately 0.5
        let mean = sum / 1000.0;
        assert!(mean > 0.4 && mean < 0.6, "Mean {} is not close to 0.5", mean);

        println!("✅ Uniform FP32 test passed (mean: {:.3})", mean);

        tensor_free(device_ctx_ptr, tensor_handle);
        rand_destroy(rand_ctx);
    }

    #[test]
    #[serial]
    fn test_uniform_f16() {
        let (device, rand_ctx) = match try_create_test_env() {
            Some(env) => env,
            None => {
                println!("⚠️  Test skipped");
                return;
            }
        };

        let device_ctx = DeviceContext { device: device.clone() };
        let device_ctx_ptr = &device_ctx as *const DeviceContext;

        let tensor_handle = tensor_allocate(device_ctx_ptr, 1000, Precision::FP16 as i32);

        let status = rand_uniform(rand_ctx, tensor_handle);
        assert!(matches!(status, RandStatus::Success));

        let mut result = vec![0.0f32; 1000];
        tensor_copy_to_f32(device_ctx_ptr, tensor_handle, result.as_mut_ptr(), 1000);

        let sum: f32 = result.iter().sum();
        let mean = sum / 1000.0;
        assert!(mean > 0.4 && mean < 0.6, "Mean {} not close to 0.5", mean);

        println!("✅ Uniform FP16 test passed (mean: {:.3})", mean);

        tensor_free(device_ctx_ptr, tensor_handle);
        rand_destroy(rand_ctx);
    }

    #[test]
    #[serial]
    fn test_normal_f32() {
        let (device, rand_ctx) = match try_create_test_env() {
            Some(env) => env,
            None => {
                println!("⚠️  Test skipped");
                return;
            }
        };

        let device_ctx = DeviceContext { device: device.clone() };
        let device_ctx_ptr = &device_ctx as *const DeviceContext;

        let tensor_handle = tensor_allocate(device_ctx_ptr, 10000, Precision::FP32 as i32);

        let mean_target = 5.0;
        let stddev_target = 2.0;
        let status = rand_normal(rand_ctx, tensor_handle, mean_target, stddev_target);
        assert!(matches!(status, RandStatus::Success));

        let mut result = vec![0.0f32; 10000];
        tensor_copy_to_f32(device_ctx_ptr, tensor_handle, result.as_mut_ptr(), 10000);

        // Compute sample mean
        let sum: f32 = result.iter().sum();
        let mean = sum / 10000.0;
        assert!((mean - mean_target as f32).abs() < 0.1,
                "Mean {} not close to target {}", mean, mean_target);

        // Compute sample stddev
        let variance: f32 = result.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / 10000.0;
        let stddev = variance.sqrt();
        assert!((stddev - stddev_target as f32).abs() < 0.2,
                "Stddev {} not close to target {}", stddev, stddev_target);

        println!("✅ Normal FP32 test passed (mean: {:.3}, stddev: {:.3})", mean, stddev);

        tensor_free(device_ctx_ptr, tensor_handle);
        rand_destroy(rand_ctx);
    }

    #[test]
    #[serial]
    fn test_uniform_range() {
        let (device, rand_ctx) = match try_create_test_env() {
            Some(env) => env,
            None => {
                println!("⚠️  Test skipped");
                return;
            }
        };

        let device_ctx = DeviceContext { device: device.clone() };
        let device_ctx_ptr = &device_ctx as *const DeviceContext;

        let tensor_handle = tensor_allocate(device_ctx_ptr, 1000, Precision::FP32 as i32);

        let low = -5.0;
        let high = 5.0;
        let status = rand_uniform_range(rand_ctx, tensor_handle, low, high);
        assert!(matches!(status, RandStatus::Success));

        let mut result = vec![0.0f32; 1000];
        tensor_copy_to_f32(device_ctx_ptr, tensor_handle, result.as_mut_ptr(), 1000);

        let mut in_range = true;
        for &x in &result {
            if x < low as f32 || x >= high as f32 {
                in_range = false;
                break;
            }
        }
        assert!(in_range, "Some values out of range [{}, {})", low, high);

        let sum: f32 = result.iter().sum();
        let mean = sum / 1000.0;
        assert!(mean > -1.0 && mean < 1.0, "Mean {} not close to 0", mean);

        println!("✅ Uniform range [{}, {}) test passed (mean: {:.3})", low, high, mean);

        tensor_free(device_ctx_ptr, tensor_handle);
        rand_destroy(rand_ctx);
    }

    #[test]
    #[serial]
    fn test_set_seed_reproducibility() {
        let (device, rand_ctx) = match try_create_test_env() {
            Some(env) => env,
            None => {
                println!("⚠️  Test skipped");
                return;
            }
        };

        let device_ctx = DeviceContext { device: device.clone() };
        let device_ctx_ptr = &device_ctx as *const DeviceContext;

        let tensor_handle = tensor_allocate(device_ctx_ptr, 100, Precision::FP32 as i32);

        // Generate with seed 42
        rand_set_seed(rand_ctx, 42);
        rand_uniform(rand_ctx, tensor_handle);
        let mut result1 = vec![0.0f32; 100];
        tensor_copy_to_f32(device_ctx_ptr, tensor_handle, result1.as_mut_ptr(), 100);

        // Generate again with same seed
        rand_set_seed(rand_ctx, 42);
        rand_uniform(rand_ctx, tensor_handle);
        let mut result2 = vec![0.0f32; 100];
        tensor_copy_to_f32(device_ctx_ptr, tensor_handle, result2.as_mut_ptr(), 100);

        // Should be identical
        for (i, (&a, &b)) in result1.iter().zip(result2.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "Mismatch at {}: {} != {}", i, a, b);
        }

        println!("✅ Seed reproducibility test passed");

        tensor_free(device_ctx_ptr, tensor_handle);
        rand_destroy(rand_ctx);
    }
}