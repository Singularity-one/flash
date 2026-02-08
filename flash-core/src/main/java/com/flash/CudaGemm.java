package com.flash;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;

/**
 * CUDA GPU accelerated matrix operations via cuBLAS.
 *
 * <p>This class provides high-performance matrix multiplication using NVIDIA GPUs.
 * It wraps the cuBLAS library through Rust FFI.</p>
 *
 * <h2>Example usage:</h2>
 * <pre>{@code
 * try (CudaGemm gemm = new CudaGemm()) {
 *     System.out.println("GPU: " + gemm.getDeviceName());
 *
 *     // 2x2 matrix multiplication (row-major)
 *     // A = [1 2]    B = [5 6]
 *     //     [3 4]        [7 8]
 *     float[] a = {1, 2, 3, 4};
 *     float[] b = {5, 6, 7, 8};
 *     float[] c = new float[4];
 *
 *     gemm.multiply(2, 2, 2, a, b, c);
 *     // c = {19, 22, 43, 50}
 *     // C = [19 22]
 *     //     [43 50]
 * }
 * }</pre>
 *
 * <h2>Matrix Layout:</h2>
 * <p>All matrices use <b>row-major</b> layout. For a 2×3 matrix:</p>
 * <pre>
 * Mathematical:     Memory layout:
 * [a b c]           {a, b, c, d, e, f}
 * [d e f]
 * </pre>
 *
 * @author Singularity-one
 * @since 0.1.0
 */
public class CudaGemm implements AutoCloseable {

    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup LIBRARY;

    private static final MethodHandle GEMM_INIT;
    private static final MethodHandle GEMM_SGEMM;
    private static final MethodHandle GEMM_DESTROY;
    private static final MethodHandle GEMM_GET_DEVICE_NAME;

    private final MemorySegment context;
    private boolean closed = false;

    static {
        try {
            Path libPath = NativeLibraryLoader.loadLibrary();
            LIBRARY = SymbolLookup.libraryLookup(libPath, Arena.global());

            GEMM_INIT = LINKER.downcallHandle(
                    LIBRARY.find("gemm_init").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.ADDRESS)
            );

            GEMM_SGEMM = LINKER.downcallHandle(
                    LIBRARY.find("gemm_sgemm").orElseThrow(),
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_INT,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_FLOAT,
                            ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_FLOAT,
                            ValueLayout.ADDRESS
                    )
            );

            GEMM_DESTROY = LINKER.downcallHandle(
                    LIBRARY.find("gemm_destroy").orElseThrow(),
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS)
            );

            GEMM_GET_DEVICE_NAME = LINKER.downcallHandle(
                    LIBRARY.find("gemm_get_device_name").orElseThrow(),
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_BOOLEAN,
                            ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_LONG
                    )
            );

        } catch (Exception e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    /**
     * Operation status codes returned by CUDA operations.
     */
    public enum Status {
        /** Operation completed successfully */
        SUCCESS(0),
        /** Failed to initialize CUDA device */
        DEVICE_INIT_ERROR(1),
        /** GPU memory allocation failed */
        MEMORY_ERROR(2),
        /** Computation error during execution */
        COMPUTE_ERROR(3),
        /** Invalid matrix dimensions provided */
        INVALID_DIMENSION(4);

        private final int code;

        Status(int code) {
            this.code = code;
        }

        /**
         * Converts an integer status code to Status enum.
         *
         * @param code the integer status code
         * @return corresponding Status enum value
         * @throws IllegalArgumentException if code is unknown
         */
        public static Status fromCode(int code) {
            for (Status s : values()) {
                if (s.code == code) return s;
            }
            throw new IllegalArgumentException("Unknown status code: " + code);
        }
    }

    /**
     * Creates a new CudaGemm instance and initializes the CUDA context.
     *
     * <p>This will:</p>
     * <ul>
     *   <li>Load the native library</li>
     *   <li>Initialize CUDA device 0</li>
     *   <li>Create a cuBLAS handle</li>
     * </ul>
     *
     * @throws RuntimeException if CUDA initialization fails
     */
    public CudaGemm() {
        try {
            context = (MemorySegment) GEMM_INIT.invoke();
            if (context.address() == 0) {
                throw new RuntimeException("Failed to initialize CUDA device. " +
                        "Check: 1) NVIDIA GPU exists, 2) CUDA driver installed, 3) GPU not in use");
            }
        } catch (Throwable e) {
            throw new RuntimeException("Failed to initialize GEMM context", e);
        }
    }

    /**
     * Performs single-precision general matrix multiplication (SGEMM).
     *
     * <p>Computes: <b>C = α × A × B + β × C</b></p>
     *
     * <p>Matrix dimensions:</p>
     * <ul>
     *   <li>A: m × k</li>
     *   <li>B: k × n</li>
     *   <li>C: m × n</li>
     * </ul>
     *
     * <p>All matrices must be in <b>row-major</b> layout.</p>
     *
     * @param m     number of rows in matrices A and C
     * @param n     number of columns in matrices B and C
     * @param k     number of columns in A / number of rows in B
     * @param alpha scalar multiplier for A×B product
     * @param a     input matrix A, size [m × k], row-major
     * @param b     input matrix B, size [k × n], row-major
     * @param beta  scalar multiplier for matrix C (use 0 to ignore existing C values)
     * @param c     input/output matrix C, size [m × n], row-major
     * @return operation status
     * @throws IllegalStateException if this context has been closed
     * @throws IllegalArgumentException if array sizes don't match dimensions
     */
    public Status sgemm(int m, int n, int k,
                        float alpha, float[] a, float[] b,
                        float beta, float[] c) {
        if (closed) {
            throw new IllegalStateException("CudaGemm context is closed");
        }
        if (a.length != m * k) {
            throw new IllegalArgumentException(
                    "Matrix A size mismatch: expected " + (m * k) + ", got " + a.length);
        }
        if (b.length != k * n) {
            throw new IllegalArgumentException(
                    "Matrix B size mismatch: expected " + (k * n) + ", got " + b.length);
        }
        if (c.length != m * n) {
            throw new IllegalArgumentException(
                    "Matrix C size mismatch: expected " + (m * n) + ", got " + c.length);
        }

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment aSeg = arena.allocateFrom(ValueLayout.JAVA_FLOAT, a);
            MemorySegment bSeg = arena.allocateFrom(ValueLayout.JAVA_FLOAT, b);
            MemorySegment cSeg = arena.allocateFrom(ValueLayout.JAVA_FLOAT, c);

            int statusCode = (int) GEMM_SGEMM.invoke(
                    context, m, n, k, alpha, aSeg, bSeg, beta, cSeg
            );

            MemorySegment.copy(cSeg, ValueLayout.JAVA_FLOAT, 0, c, 0, c.length);

            return Status.fromCode(statusCode);
        } catch (Throwable e) {
            throw new RuntimeException("GEMM computation failed", e);
        }
    }

    /**
     * Convenience method for simple matrix multiplication: C = A × B.
     *
     * <p>Equivalent to {@code sgemm(m, n, k, 1.0f, a, b, 0.0f, c)}</p>
     *
     * @param m number of rows in A and C
     * @param n number of columns in B and C
     * @param k number of columns in A / rows in B
     * @param a input matrix A [m × k]
     * @param b input matrix B [k × n]
     * @param c output matrix C [m × n]
     * @return operation status
     * @see #sgemm(int, int, int, float, float[], float[], float, float[])
     */
    public Status multiply(int m, int n, int k, float[] a, float[] b, float[] c) {
        return sgemm(m, n, k, 1.0f, a, b, 0.0f, c);
    }

    /**
     * Returns the name of the CUDA GPU device.
     *
     * @return GPU device name, e.g., "NVIDIA GeForce RTX 3060 Ti"
     */
    public String getDeviceName() {
        if (closed) {
            throw new IllegalStateException("CudaGemm context is closed");
        }

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment buffer = arena.allocate(256);
            boolean success = (boolean) GEMM_GET_DEVICE_NAME.invoke(context, buffer, 256L);

            if (!success) {
                return "Unknown Device";
            }
            return buffer.getString(0);
        } catch (Throwable e) {
            return "Error: " + e.getMessage();
        }
    }

    /**
     * Releases the CUDA context and associated resources.
     *
     * <p>After calling this method, the instance cannot be used.</p>
     */
    @Override
    public void close() {
        if (!closed) {
            try {
                GEMM_DESTROY.invoke(context);
                closed = true;
            } catch (Throwable e) {
                throw new RuntimeException("Failed to destroy GEMM context", e);
            }
        }
    }
}