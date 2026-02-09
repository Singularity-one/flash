package com.flash.exception;

/**
 * Exception thrown when tensor precisions don't match in an operation.
 *
 * <p>flash enforces strict precision matching - operations between tensors
 * of different precisions will fail rather than perform implicit conversions.</p>
 *
 * <h2>Example:</h2>
 * <pre>{@code
 * CudaTensor a = CudaTensor.allocate(device, 100, Precision.FP16);
 * CudaTensor b = CudaTensor.allocate(device, 100, Precision.FP32);
 *
 * // This will throw PrecisionMismatchException
 * blas.gemm(10, 10, 10, 1.0, a, b, 0.0, c);
 * }</pre>
 *
 * @author Singularity-one
 * @since 0.2.0
 */
public class PrecisionMismatchException extends RuntimeException {

    /**
     * Constructs a new precision mismatch exception with the specified detail message.
     *
     * @param message the detail message
     */
    public PrecisionMismatchException(String message) {
        super(message);
    }

    /**
     * Constructs a new precision mismatch exception with the specified detail message and cause.
     *
     * @param message the detail message
     * @param cause the cause
     */
    public PrecisionMismatchException(String message, Throwable cause) {
        super(message, cause);
    }
}