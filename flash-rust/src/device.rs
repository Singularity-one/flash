use cudarc::driver::{CudaDevice as CudarcDevice, CudaSlice};
use std::sync::Arc;

/// Status codes for device operations
#[repr(C)]
pub enum DeviceStatus {
    Success = 0,
    DeviceNotFound = 1,
    MemoryError = 2,
    CopyError = 3,
    InvalidParameter = 4,
}

/// CUDA device context
pub struct DeviceContext {
    pub device: Arc<CudarcDevice>,
}

// ============================================================================
// Device Initialization
// ============================================================================

/// Initialize CUDA device by ID
///
/// # C ABI
/// ```c
/// void* device_init(int device_id);
/// ```
#[no_mangle]
pub extern "C" fn device_init(device_id: i32) -> *mut DeviceContext {
    match try_device_init(device_id) {
        Ok(ctx) => Box::into_raw(Box::new(ctx)),
        Err(e) => {
            eprintln!("device_init error: {:?}", e);
            std::ptr::null_mut()
        }
    }
}

fn try_device_init(device_id: i32) -> Result<DeviceContext, Box<dyn std::error::Error>> {
    let device = CudarcDevice::new(device_id as usize)?;
    Ok(DeviceContext { device })
}

/// Get device handle for creating BLAS/DNN contexts
///
/// Returns an opaque handle to the internal Arc<CudaDevice>
///
/// # C ABI
/// ```c
/// size_t device_get_handle(void* ctx);
/// ```
#[no_mangle]
pub extern "C" fn device_get_handle(ctx: *const DeviceContext) -> usize {
    if ctx.is_null() {
        return 0;
    }

    let ctx = unsafe { &*ctx };

    // Return pointer to the Arc<CudaDevice>
    &ctx.device as *const _ as usize
}

/// Destroy device context
///
/// # C ABI
/// ```c
/// void device_destroy(void* ctx);
/// ```
#[no_mangle]
pub extern "C" fn device_destroy(ctx: *mut DeviceContext) {
    if !ctx.is_null() {
        unsafe {
            let _ = Box::from_raw(ctx);
        }
    }
}

// ============================================================================
// Device Information
// ============================================================================

/// Get device name
///
/// # C ABI
/// ```c
/// bool device_get_name(void* ctx, char* buffer, size_t buffer_size);
/// ```
#[no_mangle]
pub extern "C" fn device_get_name(
    ctx: *const DeviceContext,
    buffer: *mut i8,
    buffer_size: usize,
) -> bool {
    if ctx.is_null() || buffer.is_null() || buffer_size == 0 {
        return false;
    }

    let ctx = unsafe { &*ctx };

    let name: String = match ctx.device.name() {
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

/// Get total device memory in bytes
///
/// # C ABI
/// ```c
/// size_t device_get_total_memory(void* ctx);
/// ```
#[no_mangle]
pub extern "C" fn device_get_total_memory(ctx: *const DeviceContext) -> usize {
    if ctx.is_null() {
        return 0;
    }

    let _ctx = unsafe { &*ctx };

    // cudarc doesn't expose total memory directly, return 0 for now
    // TODO: Add via raw CUDA API if needed
    0
}

// ============================================================================
// Memory Management
// ============================================================================

/// Allocate device memory
///
/// Returns a device pointer (as usize for cross-language compatibility)
///
/// # C ABI
/// ```c
/// size_t device_allocate(void* ctx, size_t bytes);
/// ```
#[no_mangle]
pub extern "C" fn device_allocate(ctx: *mut DeviceContext, bytes: usize) -> usize {
    if ctx.is_null() || bytes == 0 {
        return 0;
    }

    let ctx = unsafe { &mut *ctx };

    let slice = match ctx.device.alloc_zeros::<u8>(bytes) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("device_allocate error: {:?}", e);
            return 0;
        }
    };

    // Store CudaSlice<u8> and return its address
    let raw_ptr = Box::into_raw(Box::new(slice));
    raw_ptr as usize
}

/// Free device memory
///
/// # C ABI
/// ```c
/// void device_free(void* ctx, size_t dev_ptr);
/// ```
#[no_mangle]
pub extern "C" fn device_free(_ctx: *mut DeviceContext, dev_ptr: usize) {
    if dev_ptr == 0 {
        return;
    }

    unsafe {
        // Convert back to Box<CudaSlice<u8>> and drop
        let ptr = dev_ptr as *mut CudaSlice<u8>;
        if !ptr.is_null() {
            let _ = Box::from_raw(ptr);
        }
    }
}

// ============================================================================
// Memory Copy Operations
// ============================================================================

/// Copy data from host to device
///
/// # C ABI
/// ```c
/// int device_copy_htod(void* ctx, const void* host_ptr, size_t dev_ptr, size_t bytes);
/// ```
#[no_mangle]
pub extern "C" fn device_copy_htod(
    ctx: *mut DeviceContext,
    host_ptr: *const u8,
    dev_ptr: usize,
    bytes: usize,
) -> DeviceStatus {
    if ctx.is_null() || host_ptr.is_null() || dev_ptr == 0 || bytes == 0 {
        return DeviceStatus::InvalidParameter;
    }

    let ctx = unsafe { &mut *ctx };
    let host_slice = unsafe { std::slice::from_raw_parts(host_ptr, bytes) };

    match try_copy_htod(ctx, host_slice, dev_ptr) {
        Ok(_) => DeviceStatus::Success,
        Err(e) => {
            eprintln!("device_copy_htod error: {:?}", e);
            DeviceStatus::CopyError
        }
    }
}

fn try_copy_htod(
    ctx: &mut DeviceContext,
    host_slice: &[u8],
    dev_ptr: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let slice = unsafe { &mut *(dev_ptr as *mut CudaSlice<u8>) };

    // htod_copy_into requires Vec, so we convert
    ctx.device.htod_copy_into(host_slice.to_vec(), slice)?;
    Ok(())
}

/// Copy data from device to host
///
/// # C ABI
/// ```c
/// int device_copy_dtoh(void* ctx, size_t dev_ptr, void* host_ptr, size_t bytes);
/// ```
#[no_mangle]
pub extern "C" fn device_copy_dtoh(
    ctx: *mut DeviceContext,
    dev_ptr: usize,
    host_ptr: *mut u8,
    bytes: usize,
) -> DeviceStatus {
    if ctx.is_null() || dev_ptr == 0 || host_ptr.is_null() || bytes == 0 {
        return DeviceStatus::InvalidParameter;
    }

    let ctx = unsafe { &mut *ctx };
    let host_slice = unsafe { std::slice::from_raw_parts_mut(host_ptr, bytes) };

    match try_copy_dtoh(ctx, dev_ptr, host_slice) {
        Ok(_) => DeviceStatus::Success,
        Err(e) => {
            eprintln!("device_copy_dtoh error: {:?}", e);
            DeviceStatus::CopyError
        }
    }
}

fn try_copy_dtoh(
    ctx: &mut DeviceContext,
    dev_ptr: usize,
    host_slice: &mut [u8],
) -> Result<(), Box<dyn std::error::Error>> {
    let slice = unsafe { &*(dev_ptr as *const CudaSlice<u8>) };
    ctx.device.dtoh_sync_copy_into(slice, host_slice)?;
    Ok(())
}

/// Synchronize device (wait for all operations to complete)
///
/// # C ABI
/// ```c
/// int device_synchronize(void* ctx);
/// ```
#[no_mangle]
pub extern "C" fn device_synchronize(ctx: *mut DeviceContext) -> DeviceStatus {
    if ctx.is_null() {
        return DeviceStatus::InvalidParameter;
    }

    let ctx = unsafe { &mut *ctx };

    match ctx.device.synchronize() {
        Ok(_) => DeviceStatus::Success,
        Err(e) => {
            eprintln!("device_synchronize error: {:?}", e);
            DeviceStatus::CopyError
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_init() {
        let ctx = device_init(0);
        assert!(!ctx.is_null());
        device_destroy(ctx);
    }

    #[test]
    fn test_device_name() {
        let ctx = device_init(0);
        assert!(!ctx.is_null());

        let mut buffer = [0i8; 256];
        let success = device_get_name(ctx, buffer.as_mut_ptr(), buffer.len());
        assert!(success);

        device_destroy(ctx);
    }

    #[test]
    fn test_memory_allocation() {
        let ctx = device_init(0);
        assert!(!ctx.is_null());

        let dev_ptr = device_allocate(ctx, 1024);
        assert_ne!(dev_ptr, 0);

        device_free(ctx, dev_ptr);
        device_destroy(ctx);
    }

    #[test]
    fn test_memory_copy() {
        let ctx = device_init(0);
        assert!(!ctx.is_null());

        let host_data = vec![1u8, 2, 3, 4, 5];
        let dev_ptr = device_allocate(ctx, host_data.len());
        assert_ne!(dev_ptr, 0);

        // H2D
        let status = device_copy_htod(
            ctx,
            host_data.as_ptr(),
            dev_ptr,
            host_data.len(),
        );
        assert!(matches!(status, DeviceStatus::Success));

        // D2H
        let mut result = vec![0u8; host_data.len()];
        let status = device_copy_dtoh(
            ctx,
            dev_ptr,
            result.as_mut_ptr(),
            result.len(),
        );
        assert!(matches!(status, DeviceStatus::Success));

        assert_eq!(host_data, result);

        device_free(ctx, dev_ptr);
        device_destroy(ctx);
    }
}