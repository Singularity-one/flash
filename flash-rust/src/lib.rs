// Phase 4: Modular architecture with multi-precision support
//
// This file serves as the main entry point, re-exporting all C ABI functions
// from individual modules.

mod device;
mod tensor;  // Phase 1: Multi-precision tensor support
mod blas;
mod gemm;
mod dnn;     // Phase 2: DNN primitives (CPU fallback)
mod rand;    // Phase 3: cuRAND wrapper
mod ops;     // Phase 4: Element-wise operations

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

// Re-export BLAS functions (Phase 1.2)
pub use blas::{
    blas_init,
    blas_destroy,
    blas_gemm,     // 新增：統一的 GEMM 介面
    blas_sgemm,    // 保留：向後相容
    blas_hgemm,    // 保留：向後相容（但會提示使用 blas_gemm）
    blas_dgemm,    // 保留：向後相容
};

// Re-export GEMM functions (Phase 0 legacy - for backward compatibility)
pub use gemm::{
    gemm_init,
    gemm_sgemm,
    gemm_destroy,
    gemm_get_device_name,
};

// Re-export DNN functions (Phase 2)
pub use dnn::{
    dnn_init,
    dnn_destroy,
    dnn_softmax_forward,
    dnn_activation_forward,
    dnn_activation_backward,
};

// Re-export RAND functions (Phase 3)
pub use rand::{
    rand_init,
    rand_init_with_seed,
    rand_destroy,
    rand_set_seed,
    rand_uniform,
    rand_normal,
    rand_uniform_range,
};

// Re-export OPS functions (Phase 4)
pub use ops::{
    // Binary operations
    ops_add,
    ops_sub,
    ops_mul,
    ops_div,
    // Unary operations
    ops_exp,
    ops_log,
    ops_sqrt,
    ops_tanh,
    ops_sigmoid,
    ops_relu,
    ops_gelu,
    ops_silu,
    ops_pow,
    ops_neg,
    ops_abs,
    // Scalar operations
    ops_scale,
    ops_fill,
    // Reduction operations
    ops_sum,
    ops_max,
    ops_min,
    ops_mean,
    // Cast operation
    ops_cast,
    // In-place operations
    ops_axpy,
    ops_clip,
};