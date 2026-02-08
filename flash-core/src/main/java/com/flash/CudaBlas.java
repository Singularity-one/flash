package com.flash;
import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;

/**
 * CUDA BLAS (Basic Linear Algebra Subprograms) operations.
 *
 * <p>Provides GPU-accelerated matrix operations.</p>
 *
 * <h2>Currently Implemented:</h2>
 * <ul>
 *   <li>Level 3: SGEMM (matrix-matrix multiplication)</li>
 * </ul>
 *
 * <h2>Example usage:</h2>
 * <pre>{@code
 * try (CudaDevice device = new CudaDevice(0);
 *      CudaBlas blas = new CudaBlas(device)) {
 *
 *     // Allocate matrices on GPU
 *     float[] a = {1, 2, 3, 4};
 *     float[] b = {5, 6, 7, 8};
 *     float[] c = new float[4];
 *
 *     long a_ptr = device.allocate(a.length * Float.BYTES);
 *     long b_ptr = device.allocate(b.length * Float.BYTES);
 *     long c_ptr = device.allocate(c.length * Float.BYTES);
 *
 *     // Copy to GPU
 *     device.copyHostToDevice(a, a_ptr);
 *     device.copyHostToDevice(b, b_ptr);
 *
 *     // Matrix multiply: C = A × B
 *     blas.sgemm(2, 2, 2, 1.0f, a_ptr, b_ptr, 0.0f, c_ptr);
 *
 *     // Copy result back
 *     device.copyDeviceToHost(c_ptr, c);
 *
 *     // Clean up
 *     device.free(a_ptr);
 *     device.free(b_ptr);
 *     device.free(c_ptr);
 * }
 * }</pre>
 *
 * @author Singularity-one
 * @since 0.2.0
 */
public class CudaBlas implements AutoCloseable {

    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup LIBRARY;

    private static final MethodHandle BLAS_INIT;
    private static final MethodHandle BLAS_DESTROY;
    private static final MethodHandle BLAS_SGEMM;
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
        /** Operation completed successfully */
        SUCCESS(0),
        /** Failed to initialize cuBLAS */
        INIT_ERROR(1),
        /** Computation error during operation */
        COMPUTE_ERROR(2),
        /** Invalid parameters provided */
        INVALID_PARAMETER(3);

        private final int code;

        Status(int code) {
            this.code = code;
        }

        public static Status fromCode(int code) {
            for (Status s : values()) {
                if (s.code == code) return s;
            }
            throw new IllegalArgumentException("Unknown status code: " + code);
        }
    }

    /**
     * Creates a new CudaBlas instance using the specified device.
     *
     * @param device the CUDA device to use for operations
     * @throws RuntimeException if cuBLAS initialization fails
     */
    public CudaBlas(CudaDevice device) {
        if (device == null || device.isClosed()) {
            throw new IllegalArgumentException("Device cannot be null or closed");
        }

        this.device = device;

        try {
            // Get device handle from CudaDevice
            long deviceHandle = getDeviceHandle(device);

            context = (MemorySegment) BLAS_INIT.invoke(deviceHandle);
            if (context.address() == 0) {
                throw new RuntimeException(
                        "Failed to initialize cuBLAS. Check CUDA installation and GPU availability."
                );
            }
        } catch (Throwable e) {
            throw new RuntimeException("Failed to initialize cuBLAS context", e);
        }
    }

    private long getDeviceHandle(CudaDevice device) throws Throwable {
        return (long) DEVICE_GET_HANDLE.invoke(device.getContextSegment());
    }

    // ========================================================================
    // BLAS Level 3: Matrix-Matrix Operations
    // ========================================================================

    /**
     * Single-precision general matrix multiplication (SGEMM).
     *
     * <p>Computes: <b>C = α × A × B + β × C</b></p>
     *
     * <p>Matrix dimensions:</p>
     * <ul>
     *   <li>A: m × k (row-major)</li>
     *   <li>B: k × n (row-major)</li>
     *   <li>C: m × n (row-major)</li>
     * </ul>
     *
     * @param m     number of rows in A and C
     * @param n     number of columns in B and C
     * @param k     number of columns in A / rows in B
     * @param alpha scalar multiplier for A×B
     * @param a_ptr device pointer to matrix A [m × k]
     * @param b_ptr device pointer to matrix B [k × n]
     * @param beta  scalar multiplier for C
     * @param c_ptr device pointer to matrix C [m × n] (input/output)
     * @return operation status
     * @throws IllegalStateException if context is closed
     */
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

    /**
     * Convenience method for simple matrix multiplication: C = A × B.
     *
     * @param m     number of rows in A and C
     * @param n     number of columns in B and C
     * @param k     number of columns in A / rows in B
     * @param a_ptr device pointer to matrix A
     * @param b_ptr device pointer to matrix B
     * @param c_ptr device pointer to matrix C (output)
     * @return operation status
     */
    public Status multiply(int m, int n, int k, long a_ptr, long b_ptr, long c_ptr) {
        return sgemm(m, n, k, 1.0f, a_ptr, b_ptr, 0.0f, c_ptr);
    }

    // ========================================================================
    // Resource Management
    // ========================================================================

    /**
     * Releases the cuBLAS context.
     *
     * <p>After calling this method, the instance cannot be used.</p>
     */
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

    /**
     * Checks if this cuBLAS context has been closed.
     *
     * @return true if closed, false otherwise
     */
    public boolean isClosed() {
        return closed;
    }

    /**
     * Gets the associated CUDA device.
     *
     * @return the CUDA device
     */
    public CudaDevice getDevice() {
        return device;
    }

    private void ensureNotClosed() {
        if (closed) {
            throw new IllegalStateException("CudaBlas context is closed");
        }
    }
}