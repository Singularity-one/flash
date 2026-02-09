// Phase 1: Modular architecture with multi-precision support
//
// This file serves as the main entry point, re-exporting all C ABI functions
// from individual modules.

mod device;
mod tensor;  // Phase 1: Multi-precision tensor support
mod blas;
mod gemm;

// Re-export device functions
pub use device::{
    device_init,
    device_destroy,
    device_get_handle,
    device_get_name,
    device_get_total_memory,
    device_allocate,
    device_free,
    device_copy_htod,
    device_copy_dtoh,
    device_synchronize,
};

// Re-export tensor functions (Phase 1)
pub use tensor::{
    tensor_allocate,
    tensor_free,
    tensor_copy_from_f32,
    tensor_copy_to_f32,
    tensor_copy_from_f64,
    tensor_copy_to_f64,
};

// Re-export BLAS functions (Phase 1)
pub use blas::{
    blas_init,
    blas_destroy,
    blas_sgemm,
    blas_hgemm,
    blas_dgemm,
};

// Re-export GEMM functions (Phase 0 legacy - for backward compatibility)
pub use gemm::{
    gemm_init,
    gemm_sgemm,
    gemm_destroy,
    gemm_get_device_name,
};