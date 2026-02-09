package com.flash;

import com.flash.exception.PrecisionMismatchException;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;

/**
 * CUDA BLAS (Basic Linear Algebra Subprograms) operations.
 *
 * <p>Provides GPU-accelerated matrix operations with multi-precision support.</p>
 *
 * <h2>Recommended Usage (Phase 1.2+):</h2>
 * <pre>{@code
 * try (CudaDevice device = new CudaDevice(0);
 *      CudaBlas blas = new CudaBlas(device)) {
 *
 *     float[] a = {1, 2, 3, 4};
 *     float[] b = {5, 6, 7, 8};
 *
 *     CudaTensor a_tensor = CudaTensor.fromFloat(device, a, Precision.FP16);
 *     CudaTensor b_tensor = CudaTensor.fromFloat(device, b, Precision.FP16);
 *     CudaTensor c_tensor = CudaTensor.allocate(device, 4, Precision.FP16);
 *
 *     blas.gemm(2, 2, 2, 1.0, a_tensor, b_tensor, 0.0, c_tensor);
 *
 *     float[] result = c_tensor.toFloatArray();
 * }
 * }</pre>
 *
 * @since 0.2.0
 */
public class CudaBlas implements AutoCloseable {

    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup LIBRARY;

    private static final MethodHandle BLAS_INIT;
    private static final MethodHandle BLAS_DESTROY;
    private static final MethodHandle BLAS_GEMM;
    private static final MethodHandle BLAS_SGEMM;
    private static final MethodHandle BLAS_HGEMM;
    private static final MethodHandle BLAS_DGEMM;
    private static final MethodHandle DEVICE_GET_HANDLE;

    private final CudaDevice device;
    private final MemorySegment context;
    private boolean closed = false;

    static {
        try {
            Path libPath = NativeLibraryLoader.loadLibrary();
            LIBRARY = SymbolLookup.libraryLookup(libPath, Arena.global());

            BLAS_INIT = LINKER.downcallHandle(
                    LIBRARY.find("blas_init").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_LONG)
            );

            BLAS_DESTROY = LINKER.downcallHandle(
                    LIBRARY.find("blas_destroy").orElseThrow(),
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS)
            );

            BLAS_GEMM = LINKER.downcallHandle(
                    LIBRARY.find("blas_gemm").orElseThrow(),
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_INT,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_DOUBLE,
                            ValueLayout.JAVA_LONG,
                            ValueLayout.JAVA_LONG,
                            ValueLayout.JAVA_DOUBLE,
                            ValueLayout.JAVA_LONG
                    )
            );

            BLAS_SGEMM = LINKER.downcallHandle(
                    LIBRARY.find("blas_sgemm").orElseThrow(),
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_INT,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_FLOAT,
                            ValueLayout.JAVA_LONG,
                            ValueLayout.JAVA_LONG,
                            ValueLayout.JAVA_FLOAT,
                            ValueLayout.JAVA_LONG
                    )
            );

            BLAS_HGEMM = LINKER.downcallHandle(
                    LIBRARY.find("blas_hgemm").orElseThrow(),
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_INT,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_FLOAT,
                            ValueLayout.JAVA_LONG,
                            ValueLayout.JAVA_LONG,
                            ValueLayout.JAVA_FLOAT,
                            ValueLayout.JAVA_LONG
                    )
            );

            BLAS_DGEMM = LINKER.downcallHandle(
                    LIBRARY.find("blas_dgemm").orElseThrow(),
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_INT,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_DOUBLE,
                            ValueLayout.JAVA_LONG,
                            ValueLayout.JAVA_LONG,
                            ValueLayout.JAVA_DOUBLE,
                            ValueLayout.JAVA_LONG
                    )
            );

            DEVICE_GET_HANDLE = LINKER.downcallHandle(
                    LIBRARY.find("device_get_handle").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.ADDRESS)
            );

        } catch (Exception e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    /**
     * BLAS operation status codes.
     */
    public enum Status {
        SUCCESS(0),
        INIT_ERROR(1),
        COMPUTE_ERROR(2),
        INVALID_PARAMETER(3),
        PRECISION_MISMATCH(4);

        private final int code;

        Status(int code) {
            this.code = code;
        }

        public int getCode() {
            return code;
        }

        public static Status fromCode(int code) {
            for (Status s : values()) {
                if (s.code == code) return s;
            }
            throw new IllegalArgumentException("Unknown status code: " + code);
        }
    }

    public CudaBlas(CudaDevice device) {
        if (device == null || device.isClosed()) {
            throw new IllegalArgumentException("Device cannot be null or closed");
        }

        this.device = device;

        try {
            long deviceHandle = getDeviceHandle(device);
            context = (MemorySegment) BLAS_INIT.invoke(deviceHandle);
            if (context.address() == 0) {
                throw new RuntimeException("Failed to initialize cuBLAS.");
            }
        } catch (Throwable e) {
            throw new RuntimeException("Failed to initialize cuBLAS context", e);
        }
    }

    private long getDeviceHandle(CudaDevice device) throws Throwable {
        return (long) DEVICE_GET_HANDLE.invoke(device.getContextSegment());
    }

    // ========================================================================
    // Phase 1.2: 統一的 GEMM 介面（推薦）
    // ========================================================================

    /**
     * Unified GEMM with automatic precision handling.
     * C = alpha * A * B + beta * C
     */
    public Status gemm(int m, int n, int k,
                       double alpha, CudaTensor a, CudaTensor b,
                       double beta, CudaTensor c) {
        ensureNotClosed();
        CudaTensor.ensureSamePrecision(a, b, c);

        try {
            int statusCode = (int) BLAS_GEMM.invoke(
                    context, m, n, k, alpha,
                    a.getHandle(), b.getHandle(),
                    beta, c.getHandle()
            );

            Status status = Status.fromCode(statusCode);
            if (status == Status.PRECISION_MISMATCH) {
                throw new PrecisionMismatchException("Precision mismatch in GEMM");
            }
            return status;
        } catch (PrecisionMismatchException e) {
            throw e;
        } catch (Throwable e) {
            throw new RuntimeException("GEMM operation failed", e);
        }
    }

    /**
     * Simple matrix multiply: C = A * B
     */
    public Status multiply(int m, int n, int k, CudaTensor a, CudaTensor b, CudaTensor c) {
        return gemm(m, n, k, 1.0, a, b, 0.0, c);
    }

    // ========================================================================
    // Legacy API (raw device pointers)
    // ========================================================================

    public Status sgemm(int m, int n, int k,
                        float alpha, long a_ptr, long b_ptr,
                        float beta, long c_ptr) {
        ensureNotClosed();
        try {
            int statusCode = (int) BLAS_SGEMM.invoke(
                    context, m, n, k, alpha, a_ptr, b_ptr, beta, c_ptr
            );
            return Status.fromCode(statusCode);
        } catch (Throwable e) {
            throw new RuntimeException("SGEMM operation failed", e);
        }
    }

    public Status multiply(int m, int n, int k, long a_ptr, long b_ptr, long c_ptr) {
        return sgemm(m, n, k, 1.0f, a_ptr, b_ptr, 0.0f, c_ptr);
    }

    public Status hgemm(int m, int n, int k,
                        float alpha, long a_ptr, long b_ptr,
                        float beta, long c_ptr) {
        ensureNotClosed();
        try {
            int statusCode = (int) BLAS_HGEMM.invoke(
                    context, m, n, k, alpha, a_ptr, b_ptr, beta, c_ptr
            );
            return Status.fromCode(statusCode);
        } catch (Throwable e) {
            throw new RuntimeException("HGEMM operation failed", e);
        }
    }

    public Status dgemm(int m, int n, int k,
                        double alpha, long a_ptr, long b_ptr,
                        double beta, long c_ptr) {
        ensureNotClosed();
        try {
            int statusCode = (int) BLAS_DGEMM.invoke(
                    context, m, n, k, alpha, a_ptr, b_ptr, beta, c_ptr
            );
            return Status.fromCode(statusCode);
        } catch (Throwable e) {
            throw new RuntimeException("DGEMM operation failed", e);
        }
    }

    // ========================================================================
    // Resource Management
    // ========================================================================

    @Override
    public void close() {
        if (!closed) {
            try {
                BLAS_DESTROY.invoke(context);
                closed = true;
            } catch (Throwable e) {
                throw new RuntimeException("Failed to destroy cuBLAS context", e);
            }
        }
    }

    public boolean isClosed() {
        return closed;
    }

    public CudaDevice getDevice() {
        return device;
    }

    private void ensureNotClosed() {
        if (closed) {
            throw new IllegalStateException("CudaBlas context is closed");
        }
    }
}