package com.flash;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;

/**
 * Element-wise GPU operations.
 *
 * <p>Provides common element-wise operations on tensors:</p>
 * <ul>
 *   <li>Binary: add, sub, mul, div</li>
 *   <li>Unary: exp, log, sqrt, tanh, sigmoid, relu, gelu, silu</li>
 *   <li>Scalar: scale, fill</li>
 *   <li>Reduction: sum, max, min, mean</li>
 *   <li>Cast: precision conversion</li>
 *   <li>In-place: axpy (y += alpha*x), clip</li>
 * </ul>
 *
 * <h2>Example usage:</h2>
 * <pre>{@code
 * try (CudaDevice device = new CudaDevice(0)) {
 *     CudaTensor a = CudaTensor.fromFloat(device, new float[]{1, 2, 3, 4}, Precision.FP32);
 *     CudaTensor b = CudaTensor.fromFloat(device, new float[]{5, 6, 7, 8}, Precision.FP32);
 *     CudaTensor c = CudaTensor.allocate(device, 4, Precision.FP32);
 *
 *     // c = a + b
 *     CudaOps.add(device, a, b, c);
 *
 *     // c = gelu(a)
 *     CudaOps.gelu(device, a, c);
 *
 *     // sum = sum(a)
 *     double sum = CudaOps.sum(device, a);
 * }
 * }</pre>
 *
 * @since 0.5.0
 */
public final class CudaOps {

    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup LIBRARY;

    // Binary operations
    private static final MethodHandle OPS_ADD;
    private static final MethodHandle OPS_SUB;
    private static final MethodHandle OPS_MUL;
    private static final MethodHandle OPS_DIV;

    // Unary operations
    private static final MethodHandle OPS_EXP;
    private static final MethodHandle OPS_LOG;
    private static final MethodHandle OPS_SQRT;
    private static final MethodHandle OPS_TANH;
    private static final MethodHandle OPS_SIGMOID;
    private static final MethodHandle OPS_RELU;
    private static final MethodHandle OPS_GELU;
    private static final MethodHandle OPS_SILU;
    private static final MethodHandle OPS_POW;
    private static final MethodHandle OPS_NEG;
    private static final MethodHandle OPS_ABS;

    // Scalar operations
    private static final MethodHandle OPS_SCALE;
    private static final MethodHandle OPS_FILL;

    // Reduction operations
    private static final MethodHandle OPS_SUM;
    private static final MethodHandle OPS_MAX;
    private static final MethodHandle OPS_MIN;
    private static final MethodHandle OPS_MEAN;

    // Cast operation
    private static final MethodHandle OPS_CAST;

    // In-place operations
    private static final MethodHandle OPS_AXPY;
    private static final MethodHandle OPS_CLIP;

    static {
        try {
            Path libPath = NativeLibraryLoader.loadLibrary();
            LIBRARY = SymbolLookup.libraryLookup(libPath, Arena.global());

            // Binary: (ctx, a, b, c) -> int
            FunctionDescriptor binaryDesc = FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,
                    ValueLayout.ADDRESS,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG
            );

            OPS_ADD = LINKER.downcallHandle(LIBRARY.find("ops_add").orElseThrow(), binaryDesc);
            OPS_SUB = LINKER.downcallHandle(LIBRARY.find("ops_sub").orElseThrow(), binaryDesc);
            OPS_MUL = LINKER.downcallHandle(LIBRARY.find("ops_mul").orElseThrow(), binaryDesc);
            OPS_DIV = LINKER.downcallHandle(LIBRARY.find("ops_div").orElseThrow(), binaryDesc);

            // Unary: (ctx, x, y) -> int
            FunctionDescriptor unaryDesc = FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,
                    ValueLayout.ADDRESS,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG
            );

            OPS_EXP = LINKER.downcallHandle(LIBRARY.find("ops_exp").orElseThrow(), unaryDesc);
            OPS_LOG = LINKER.downcallHandle(LIBRARY.find("ops_log").orElseThrow(), unaryDesc);
            OPS_SQRT = LINKER.downcallHandle(LIBRARY.find("ops_sqrt").orElseThrow(), unaryDesc);
            OPS_TANH = LINKER.downcallHandle(LIBRARY.find("ops_tanh").orElseThrow(), unaryDesc);
            OPS_SIGMOID = LINKER.downcallHandle(LIBRARY.find("ops_sigmoid").orElseThrow(), unaryDesc);
            OPS_RELU = LINKER.downcallHandle(LIBRARY.find("ops_relu").orElseThrow(), unaryDesc);
            OPS_GELU = LINKER.downcallHandle(LIBRARY.find("ops_gelu").orElseThrow(), unaryDesc);
            OPS_SILU = LINKER.downcallHandle(LIBRARY.find("ops_silu").orElseThrow(), unaryDesc);
            OPS_NEG = LINKER.downcallHandle(LIBRARY.find("ops_neg").orElseThrow(), unaryDesc);
            OPS_ABS = LINKER.downcallHandle(LIBRARY.find("ops_abs").orElseThrow(), unaryDesc);

            // Pow: (ctx, x, y, n) -> int
            OPS_POW = LINKER.downcallHandle(
                    LIBRARY.find("ops_pow").orElseThrow(),
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_INT,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_LONG,
                            ValueLayout.JAVA_LONG,
                            ValueLayout.JAVA_DOUBLE
                    )
            );

            // Scale: (ctx, x, y, alpha) -> int
            OPS_SCALE = LINKER.downcallHandle(
                    LIBRARY.find("ops_scale").orElseThrow(),
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_INT,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_LONG,
                            ValueLayout.JAVA_LONG,
                            ValueLayout.JAVA_DOUBLE
                    )
            );

            // Fill: (ctx, x, value) -> int
            OPS_FILL = LINKER.downcallHandle(
                    LIBRARY.find("ops_fill").orElseThrow(),
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_INT,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_LONG,
                            ValueLayout.JAVA_DOUBLE
                    )
            );

            // Reduction: (ctx, x, result_ptr) -> int
            FunctionDescriptor reductionDesc = FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,
                    ValueLayout.ADDRESS,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.ADDRESS
            );

            OPS_SUM = LINKER.downcallHandle(LIBRARY.find("ops_sum").orElseThrow(), reductionDesc);
            OPS_MAX = LINKER.downcallHandle(LIBRARY.find("ops_max").orElseThrow(), reductionDesc);
            OPS_MIN = LINKER.downcallHandle(LIBRARY.find("ops_min").orElseThrow(), reductionDesc);
            OPS_MEAN = LINKER.downcallHandle(LIBRARY.find("ops_mean").orElseThrow(), reductionDesc);

            // Cast: (ctx, src, dst) -> int
            OPS_CAST = LINKER.downcallHandle(LIBRARY.find("ops_cast").orElseThrow(), unaryDesc);

            // AXPY: (ctx, alpha, x, y) -> int
            OPS_AXPY = LINKER.downcallHandle(
                    LIBRARY.find("ops_axpy").orElseThrow(),
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_INT,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_DOUBLE,
                            ValueLayout.JAVA_LONG,
                            ValueLayout.JAVA_LONG
                    )
            );

            // Clip: (ctx, x, y, min, max) -> int
            OPS_CLIP = LINKER.downcallHandle(
                    LIBRARY.find("ops_clip").orElseThrow(),
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_INT,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_LONG,
                            ValueLayout.JAVA_LONG,
                            ValueLayout.JAVA_DOUBLE,
                            ValueLayout.JAVA_DOUBLE
                    )
            );

        } catch (Exception e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    /**
     * Operation status codes.
     */
    public enum Status {
        SUCCESS(0),
        INVALID_PARAMETER(1),
        SIZE_MISMATCH(2),
        PRECISION_MISMATCH(3),
        COMPUTE_ERROR(4),
        NOT_SUPPORTED(5);

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

    private CudaOps() {
        // Utility class, no instances
    }

    // ========================================================================
    // Binary Operations
    // ========================================================================

    /**
     * Element-wise addition: c = a + b
     */
    public static Status add(CudaDevice device, CudaTensor a, CudaTensor b, CudaTensor c) {
        return binaryOp(OPS_ADD, device, a, b, c);
    }

    /**
     * Element-wise subtraction: c = a - b
     */
    public static Status sub(CudaDevice device, CudaTensor a, CudaTensor b, CudaTensor c) {
        return binaryOp(OPS_SUB, device, a, b, c);
    }

    /**
     * Element-wise multiplication: c = a * b
     */
    public static Status mul(CudaDevice device, CudaTensor a, CudaTensor b, CudaTensor c) {
        return binaryOp(OPS_MUL, device, a, b, c);
    }

    /**
     * Element-wise division: c = a / b
     */
    public static Status div(CudaDevice device, CudaTensor a, CudaTensor b, CudaTensor c) {
        return binaryOp(OPS_DIV, device, a, b, c);
    }

    private static Status binaryOp(MethodHandle handle, CudaDevice device, CudaTensor a, CudaTensor b, CudaTensor c) {
        try {
            int code = (int) handle.invoke(device.getContextSegment(), a.getHandle(), b.getHandle(), c.getHandle());
            return Status.fromCode(code);
        } catch (Throwable e) {
            throw new RuntimeException("Binary operation failed", e);
        }
    }

    // ========================================================================
    // Unary Operations
    // ========================================================================

    /**
     * Element-wise exponential: y = exp(x)
     */
    public static Status exp(CudaDevice device, CudaTensor x, CudaTensor y) {
        return unaryOp(OPS_EXP, device, x, y);
    }

    /**
     * Element-wise natural logarithm: y = ln(x)
     */
    public static Status log(CudaDevice device, CudaTensor x, CudaTensor y) {
        return unaryOp(OPS_LOG, device, x, y);
    }

    /**
     * Element-wise square root: y = sqrt(x)
     */
    public static Status sqrt(CudaDevice device, CudaTensor x, CudaTensor y) {
        return unaryOp(OPS_SQRT, device, x, y);
    }

    /**
     * Element-wise hyperbolic tangent: y = tanh(x)
     */
    public static Status tanh(CudaDevice device, CudaTensor x, CudaTensor y) {
        return unaryOp(OPS_TANH, device, x, y);
    }

    /**
     * Element-wise sigmoid: y = 1 / (1 + exp(-x))
     */
    public static Status sigmoid(CudaDevice device, CudaTensor x, CudaTensor y) {
        return unaryOp(OPS_SIGMOID, device, x, y);
    }

    /**
     * Element-wise ReLU: y = max(0, x)
     */
    public static Status relu(CudaDevice device, CudaTensor x, CudaTensor y) {
        return unaryOp(OPS_RELU, device, x, y);
    }

    /**
     * Element-wise GELU: y = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
     */
    public static Status gelu(CudaDevice device, CudaTensor x, CudaTensor y) {
        return unaryOp(OPS_GELU, device, x, y);
    }

    /**
     * Element-wise SiLU (Swish): y = x * sigmoid(x)
     */
    public static Status silu(CudaDevice device, CudaTensor x, CudaTensor y) {
        return unaryOp(OPS_SILU, device, x, y);
    }

    /**
     * Element-wise negation: y = -x
     */
    public static Status neg(CudaDevice device, CudaTensor x, CudaTensor y) {
        return unaryOp(OPS_NEG, device, x, y);
    }

    /**
     * Element-wise absolute value: y = |x|
     */
    public static Status abs(CudaDevice device, CudaTensor x, CudaTensor y) {
        return unaryOp(OPS_ABS, device, x, y);
    }

    /**
     * Element-wise power: y = x^n
     */
    public static Status pow(CudaDevice device, CudaTensor x, CudaTensor y, double n) {
        try {
            int code = (int) OPS_POW.invoke(device.getContextSegment(), x.getHandle(), y.getHandle(), n);
            return Status.fromCode(code);
        } catch (Throwable e) {
            throw new RuntimeException("Pow operation failed", e);
        }
    }

    private static Status unaryOp(MethodHandle handle, CudaDevice device, CudaTensor x, CudaTensor y) {
        try {
            int code = (int) handle.invoke(device.getContextSegment(), x.getHandle(), y.getHandle());
            return Status.fromCode(code);
        } catch (Throwable e) {
            throw new RuntimeException("Unary operation failed", e);
        }
    }

    // ========================================================================
    // Scalar Operations
    // ========================================================================

    /**
     * Scale tensor: y = alpha * x
     */
    public static Status scale(CudaDevice device, CudaTensor x, CudaTensor y, double alpha) {
        try {
            int code = (int) OPS_SCALE.invoke(device.getContextSegment(), x.getHandle(), y.getHandle(), alpha);
            return Status.fromCode(code);
        } catch (Throwable e) {
            throw new RuntimeException("Scale operation failed", e);
        }
    }

    /**
     * Fill tensor with constant value
     */
    public static Status fill(CudaDevice device, CudaTensor x, double value) {
        try {
            int code = (int) OPS_FILL.invoke(device.getContextSegment(), x.getHandle(), value);
            return Status.fromCode(code);
        } catch (Throwable e) {
            throw new RuntimeException("Fill operation failed", e);
        }
    }

    // ========================================================================
    // Reduction Operations
    // ========================================================================

    /**
     * Compute sum of all elements
     */
    public static double sum(CudaDevice device, CudaTensor x) {
        return reduction(OPS_SUM, device, x);
    }

    /**
     * Compute maximum element
     */
    public static double max(CudaDevice device, CudaTensor x) {
        return reduction(OPS_MAX, device, x);
    }

    /**
     * Compute minimum element
     */
    public static double min(CudaDevice device, CudaTensor x) {
        return reduction(OPS_MIN, device, x);
    }

    /**
     * Compute mean of all elements
     */
    public static double mean(CudaDevice device, CudaTensor x) {
        return reduction(OPS_MEAN, device, x);
    }

    private static double reduction(MethodHandle handle, CudaDevice device, CudaTensor x) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment resultSeg = arena.allocate(ValueLayout.JAVA_DOUBLE);
            int code = (int) handle.invoke(device.getContextSegment(), x.getHandle(), resultSeg);
            if (code != 0) {
                throw new RuntimeException("Reduction failed with status: " + Status.fromCode(code));
            }
            return resultSeg.get(ValueLayout.JAVA_DOUBLE, 0);
        } catch (Throwable e) {
            throw new RuntimeException("Reduction operation failed", e);
        }
    }

    // ========================================================================
    // Cast Operation
    // ========================================================================

    /**
     * Cast tensor to different precision
     *
     * @param device the CUDA device
     * @param src source tensor
     * @param dst destination tensor (must have same element count, different precision allowed)
     * @return operation status
     */
    public static Status cast(CudaDevice device, CudaTensor src, CudaTensor dst) {
        try {
            int code = (int) OPS_CAST.invoke(device.getContextSegment(), src.getHandle(), dst.getHandle());
            return Status.fromCode(code);
        } catch (Throwable e) {
            throw new RuntimeException("Cast operation failed", e);
        }
    }

    // ========================================================================
    // In-place Operations
    // ========================================================================

    /**
     * In-place add: y = y + alpha * x (useful for gradient accumulation)
     */
    public static Status axpy(CudaDevice device, double alpha, CudaTensor x, CudaTensor y) {
        try {
            int code = (int) OPS_AXPY.invoke(device.getContextSegment(), alpha, x.getHandle(), y.getHandle());
            return Status.fromCode(code);
        } catch (Throwable e) {
            throw new RuntimeException("AXPY operation failed", e);
        }
    }

    /**
     * Clip values to range [minVal, maxVal]
     */
    public static Status clip(CudaDevice device, CudaTensor x, CudaTensor y, double minVal, double maxVal) {
        try {
            int code = (int) OPS_CLIP.invoke(device.getContextSegment(), x.getHandle(), y.getHandle(), minVal, maxVal);
            return Status.fromCode(code);
        } catch (Throwable e) {
            throw new RuntimeException("Clip operation failed", e);
        }
    }
}