use cudarc::driver::{CudaDevice, CudaSlice};
use std::sync::Arc;
use half::{f16, bf16};

/// Precision types for tensors
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    FP32 = 0,
    FP16 = 1,
    FP64 = 2,
    BF16 = 3,
    INT8 = 4,
    INT4 = 5,
}

impl Precision {
    pub fn from_i32(value: i32) -> Option<Self> {
        match value {
            0 => Some(Precision::FP32),
            1 => Some(Precision::FP16),
            2 => Some(Precision::FP64),
            3 => Some(Precision::BF16),
            4 => Some(Precision::INT8),
            5 => Some(Precision::INT4),
            _ => None,
        }
    }

    pub fn bytes_per_element(&self) -> usize {
        match self {
            Precision::FP32 => 4,
            Precision::FP16 => 2,
            Precision::FP64 => 8,
            Precision::BF16 => 2,
            Precision::INT8 => 1,
            Precision::INT4 => 1,
        }
    }
}

/// Tensor handle wrapping type-erased GPU memory
///
/// Note: FP16/BF16 now use half::f16/bf16 directly (requires cudarc f16 feature)
pub enum TensorData {
    FP32(CudaSlice<f32>),
    FP16(CudaSlice<f16>),  // Uses half::f16 directly
    FP64(CudaSlice<f64>),
    BF16(CudaSlice<bf16>), // Uses half::bf16 directly
    INT8(CudaSlice<i8>),
}

pub struct TensorHandle {
    pub data: TensorData,
    pub element_count: usize,
    pub dtype: Precision,
}

// ============================================================================
// Tensor Allocation
// ============================================================================

/// Allocate a tensor on GPU
///
/// # C ABI
/// ```c
/// size_t tensor_allocate(void* ctx, size_t element_count, int dtype);
/// ```
#[no_mangle]
pub extern "C" fn tensor_allocate(
    ctx: *const super::device::DeviceContext,
    element_count: usize,
    dtype: i32,
) -> usize {
    if ctx.is_null() || element_count == 0 {
        return 0;
    }

    let precision = match Precision::from_i32(dtype) {
        Some(p) => p,
        None => {
            eprintln!("tensor_allocate: invalid precision {}", dtype);
            return 0;
        }
    };

    let ctx = unsafe { &*ctx };

    match try_allocate(&ctx.device, element_count, precision) {
        Ok(handle) => Box::into_raw(Box::new(handle)) as usize,
        Err(e) => {
            eprintln!("tensor_allocate error: {:?}", e);
            0
        }
    }
}

fn try_allocate(
    device: &Arc<CudaDevice>,
    element_count: usize,
    dtype: Precision,
) -> Result<TensorHandle, Box<dyn std::error::Error>> {
    let data = match dtype {
        Precision::FP32 => TensorData::FP32(device.alloc_zeros::<f32>(element_count)?),
        Precision::FP16 => TensorData::FP16(device.alloc_zeros::<f16>(element_count)?),
        Precision::FP64 => TensorData::FP64(device.alloc_zeros::<f64>(element_count)?),
        Precision::BF16 => TensorData::BF16(device.alloc_zeros::<bf16>(element_count)?),
        Precision::INT8 => TensorData::INT8(device.alloc_zeros::<i8>(element_count)?),
        Precision::INT4 => {
            return Err("INT4 not yet implemented".into());
        }
    };

    Ok(TensorHandle {
        data,
        element_count,
        dtype,
    })
}

/// Free a tensor
///
/// # C ABI
/// ```c
/// void tensor_free(void* ctx, size_t handle);
/// ```
#[no_mangle]
pub extern "C" fn tensor_free(_ctx: *const super::device::DeviceContext, handle: usize) {
    if handle != 0 {
        unsafe {
            let _ = Box::from_raw(handle as *mut TensorHandle);
        }
    }
}

// ============================================================================
// Data Copy: FP32 Interface
// ============================================================================

/// Copy from host FP32 array to tensor (with conversion if needed)
///
/// # C ABI
/// ```c
/// int tensor_copy_from_f32(void* ctx, size_t handle, const float* host_ptr, size_t count);
/// ```
#[no_mangle]
pub extern "C" fn tensor_copy_from_f32(
    ctx: *const super::device::DeviceContext,
    handle: usize,
    host_ptr: *const f32,
    count: usize,
) -> i32 {
    if ctx.is_null() || handle == 0 || host_ptr.is_null() || count == 0 {
        return 1;
    }

    let ctx = unsafe { &*ctx };
    let tensor = unsafe { &mut *(handle as *mut TensorHandle) };

    if tensor.element_count != count {
        eprintln!(
            "tensor_copy_from_f32: count mismatch: {} != {}",
            count, tensor.element_count
        );
        return 2;
    }

    let host_data = unsafe { std::slice::from_raw_parts(host_ptr, count) };

    match try_copy_from_f32(&ctx.device, tensor, host_data) {
        Ok(_) => 0,
        Err(e) => {
            eprintln!("tensor_copy_from_f32 error: {:?}", e);
            3
        }
    }
}

fn try_copy_from_f32(
    device: &Arc<CudaDevice>,
    tensor: &mut TensorHandle,
    host_data: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    match &mut tensor.data {
        TensorData::FP32(slice) => {
            device.htod_copy_into(host_data.to_vec(), slice)?;
        }
        TensorData::FP16(slice) => {
            // Convert f32 -> f16
            let f16_data: Vec<f16> = host_data
                .iter()
                .map(|&x| f16::from_f32(x))
                .collect();
            device.htod_copy_into(f16_data, slice)?;
        }
        TensorData::FP64(slice) => {
            // Convert f32 -> f64
            let f64_data: Vec<f64> = host_data.iter().map(|&x| x as f64).collect();
            device.htod_copy_into(f64_data, slice)?;
        }
        TensorData::BF16(slice) => {
            // Convert f32 -> bf16
            let bf16_data: Vec<bf16> = host_data
                .iter()
                .map(|&x| bf16::from_f32(x))
                .collect();
            device.htod_copy_into(bf16_data, slice)?;
        }
        TensorData::INT8(_) => {
            return Err("Cannot convert f32 to int8 automatically".into());
        }
    }

    device.synchronize()?;
    Ok(())
}

/// Copy from tensor to host FP32 array (with conversion if needed)
///
/// # C ABI
/// ```c
/// int tensor_copy_to_f32(void* ctx, size_t handle, float* host_ptr, size_t count);
/// ```
#[no_mangle]
pub extern "C" fn tensor_copy_to_f32(
    ctx: *const super::device::DeviceContext,
    handle: usize,
    host_ptr: *mut f32,
    count: usize,
) -> i32 {
    if ctx.is_null() || handle == 0 || host_ptr.is_null() || count == 0 {
        return 1;
    }

    let ctx = unsafe { &*ctx };
    let tensor = unsafe { &*(handle as *const TensorHandle) };

    if tensor.element_count != count {
        eprintln!(
            "tensor_copy_to_f32: count mismatch: {} != {}",
            count, tensor.element_count
        );
        return 2;
    }

    let host_data = unsafe { std::slice::from_raw_parts_mut(host_ptr, count) };

    match try_copy_to_f32(&ctx.device, tensor, host_data) {
        Ok(_) => 0,
        Err(e) => {
            eprintln!("tensor_copy_to_f32 error: {:?}", e);
            3
        }
    }
}

fn try_copy_to_f32(
    device: &Arc<CudaDevice>,
    tensor: &TensorHandle,
    host_data: &mut [f32],
) -> Result<(), Box<dyn std::error::Error>> {
    match &tensor.data {
        TensorData::FP32(slice) => {
            device.dtoh_sync_copy_into(slice, host_data)?;
        }
        TensorData::FP16(slice) => {
            // Convert f16 -> f32
            let mut f16_data = vec![f16::from_f32(0.0); tensor.element_count];
            device.dtoh_sync_copy_into(slice, &mut f16_data)?;
            for (i, val) in f16_data.iter().enumerate() {
                host_data[i] = val.to_f32();
            }
        }
        TensorData::FP64(slice) => {
            // Convert f64 -> f32
            let mut f64_data = vec![0.0f64; tensor.element_count];
            device.dtoh_sync_copy_into(slice, &mut f64_data)?;
            for (i, &val) in f64_data.iter().enumerate() {
                host_data[i] = val as f32;
            }
        }
        TensorData::BF16(slice) => {
            // Convert bf16 -> f32
            let mut bf16_data = vec![bf16::from_f32(0.0); tensor.element_count];
            device.dtoh_sync_copy_into(slice, &mut bf16_data)?;
            for (i, val) in bf16_data.iter().enumerate() {
                host_data[i] = val.to_f32();
            }
        }
        TensorData::INT8(_) => {
            return Err("Cannot convert int8 to f32 automatically".into());
        }
    }

    Ok(())
}

// ============================================================================
// Data Copy: FP64 Interface
// ============================================================================

/// Copy from host FP64 array to tensor (with conversion if needed)
///
/// # C ABI
/// ```c
/// int tensor_copy_from_f64(void* ctx, size_t handle, const double* host_ptr, size_t count);
/// ```
#[no_mangle]
pub extern "C" fn tensor_copy_from_f64(
    ctx: *const super::device::DeviceContext,
    handle: usize,
    host_ptr: *const f64,
    count: usize,
) -> i32 {
    if ctx.is_null() || handle == 0 || host_ptr.is_null() || count == 0 {
        return 1;
    }

    let ctx = unsafe { &*ctx };
    let tensor = unsafe { &mut *(handle as *mut TensorHandle) };

    if tensor.element_count != count {
        return 2;
    }

    let host_data = unsafe { std::slice::from_raw_parts(host_ptr, count) };

    match try_copy_from_f64(&ctx.device, tensor, host_data) {
        Ok(_) => 0,
        Err(e) => {
            eprintln!("tensor_copy_from_f64 error: {:?}", e);
            3
        }
    }
}

fn try_copy_from_f64(
    device: &Arc<CudaDevice>,
    tensor: &mut TensorHandle,
    host_data: &[f64],
) -> Result<(), Box<dyn std::error::Error>> {
    match &mut tensor.data {
        TensorData::FP32(slice) => {
            let f32_data: Vec<f32> = host_data.iter().map(|&x| x as f32).collect();
            device.htod_copy_into(f32_data, slice)?;
        }
        TensorData::FP16(slice) => {
            let f16_data: Vec<f16> = host_data
                .iter()
                .map(|&x| f16::from_f64(x))
                .collect();
            device.htod_copy_into(f16_data, slice)?;
        }
        TensorData::FP64(slice) => {
            device.htod_copy_into(host_data.to_vec(), slice)?;
        }
        TensorData::BF16(slice) => {
            let bf16_data: Vec<bf16> = host_data
                .iter()
                .map(|&x| bf16::from_f64(x))
                .collect();
            device.htod_copy_into(bf16_data, slice)?;
        }
        TensorData::INT8(_) => {
            return Err("Cannot convert f64 to int8 automatically".into());
        }
    }

    device.synchronize()?;
    Ok(())
}

/// Copy from tensor to host FP64 array (with conversion if needed)
///
/// # C ABI
/// ```c
/// int tensor_copy_to_f64(void* ctx, size_t handle, double* host_ptr, size_t count);
/// ```
#[no_mangle]
pub extern "C" fn tensor_copy_to_f64(
    ctx: *const super::device::DeviceContext,
    handle: usize,
    host_ptr: *mut f64,
    count: usize,
) -> i32 {
    if ctx.is_null() || handle == 0 || host_ptr.is_null() || count == 0 {
        return 1;
    }

    let ctx = unsafe { &*ctx };
    let tensor = unsafe { &*(handle as *const TensorHandle) };

    if tensor.element_count != count {
        return 2;
    }

    let host_data = unsafe { std::slice::from_raw_parts_mut(host_ptr, count) };

    match try_copy_to_f64(&ctx.device, tensor, host_data) {
        Ok(_) => 0,
        Err(e) => {
            eprintln!("tensor_copy_to_f64 error: {:?}", e);
            3
        }
    }
}

fn try_copy_to_f64(
    device: &Arc<CudaDevice>,
    tensor: &TensorHandle,
    host_data: &mut [f64],
) -> Result<(), Box<dyn std::error::Error>> {
    match &tensor.data {
        TensorData::FP32(slice) => {
            let mut f32_data = vec![0.0f32; tensor.element_count];
            device.dtoh_sync_copy_into(slice, &mut f32_data)?;
            for (i, &val) in f32_data.iter().enumerate() {
                host_data[i] = val as f64;
            }
        }
        TensorData::FP16(slice) => {
            let mut f16_data = vec![f16::from_f32(0.0); tensor.element_count];
            device.dtoh_sync_copy_into(slice, &mut f16_data)?;
            for (i, val) in f16_data.iter().enumerate() {
                host_data[i] = val.to_f64();
            }
        }
        TensorData::FP64(slice) => {
            device.dtoh_sync_copy_into(slice, host_data)?;
        }
        TensorData::BF16(slice) => {
            let mut bf16_data = vec![bf16::from_f32(0.0); tensor.element_count];
            device.dtoh_sync_copy_into(slice, &mut bf16_data)?;
            for (i, val) in bf16_data.iter().enumerate() {
                host_data[i] = val.to_f64();
            }
        }
        TensorData::INT8(_) => {
            return Err("Cannot convert int8 to f64 automatically".into());
        }
    }

    Ok(())
}

// ============================================================================
// Helper: Get raw pointer from TensorHandle (for BLAS operations)
// ============================================================================

impl TensorHandle {
    /// Get raw device pointer for use in cuBLAS/cuDNN calls
    pub fn as_device_ptr(&self) -> usize {
        match &self.data {
            TensorData::FP32(slice) => slice as *const _ as usize,
            TensorData::FP16(slice) => slice as *const _ as usize,
            TensorData::FP64(slice) => slice as *const _ as usize,
            TensorData::BF16(slice) => slice as *const _ as usize,
            TensorData::INT8(slice) => slice as *const _ as usize,
        }
    }

    pub fn as_device_ptr_mut(&mut self) -> usize {
        match &mut self.data {
            TensorData::FP32(slice) => slice as *mut _ as usize,
            TensorData::FP16(slice) => slice as *mut _ as usize,
            TensorData::FP64(slice) => slice as *mut _ as usize,
            TensorData::BF16(slice) => slice as *mut _ as usize,
            TensorData::INT8(slice) => slice as *mut _ as usize,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    fn create_test_device() -> Option<super::super::device::DeviceContext> {
        match CudaDevice::new(0) {
            Ok(device) => Some(super::super::device::DeviceContext { device }),
            Err(e) => {
                eprintln!("⚠️  CUDA device not available: {:?}", e);
                None
            }
        }
    }

    #[test]
    #[serial]
    fn test_allocate_fp32() {
        let ctx = match create_test_device() {
            Some(c) => c,
            None => {
                println!("⚠️  Test skipped: CUDA not available");
                return;
            }
        };
        let ctx_ptr = &ctx as *const _;

        let handle = tensor_allocate(ctx_ptr, 100, 0); // FP32
        assert_ne!(handle, 0);

        tensor_free(ctx_ptr, handle);
        println!("✅ FP32 tensor allocation test passed");
    }

    #[test]
    #[serial]
    fn test_fp32_roundtrip() {
        let ctx = match create_test_device() {
            Some(c) => c,
            None => {
                println!("⚠️  Test skipped: CUDA not available");
                return;
            }
        };
        let ctx_ptr = &ctx as *const _;

        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let handle = tensor_allocate(ctx_ptr, data.len(), 0); // FP32

        let status = tensor_copy_from_f32(ctx_ptr, handle, data.as_ptr(), data.len());
        assert_eq!(status, 0);

        let mut result = vec![0.0f32; data.len()];
        let status = tensor_copy_to_f32(ctx_ptr, handle, result.as_mut_ptr(), result.len());
        assert_eq!(status, 0);

        for (i, (&expected, &actual)) in data.iter().zip(result.iter()).enumerate() {
            assert!(
                (expected - actual).abs() < 1e-6,
                "Mismatch at {}: {} != {}",
                i,
                expected,
                actual
            );
        }

        tensor_free(ctx_ptr, handle);
        println!("✅ FP32 roundtrip test passed");
    }

    #[test]
    #[serial]
    fn test_fp16_conversion() {
        let ctx = match create_test_device() {
            Some(c) => c,
            None => {
                println!("⚠️  Test skipped: CUDA not available");
                return;
            }
        };
        let ctx_ptr = &ctx as *const _;

        let data = vec![1.0f32, 2.5, 3.7, 4.2];
        let handle = tensor_allocate(ctx_ptr, data.len(), 1); // FP16

        let status = tensor_copy_from_f32(ctx_ptr, handle, data.as_ptr(), data.len());
        assert_eq!(status, 0);

        let mut result = vec![0.0f32; data.len()];
        let status = tensor_copy_to_f32(ctx_ptr, handle, result.as_mut_ptr(), result.len());
        assert_eq!(status, 0);

        for (i, (&expected, &actual)) in data.iter().zip(result.iter()).enumerate() {
            assert!(
                (expected - actual).abs() < 0.01, // FP16 has lower precision
                "Mismatch at {}: {} != {}",
                i,
                expected,
                actual
            );
        }

        tensor_free(ctx_ptr, handle);
        println!("✅ FP16 conversion test passed");
    }
}