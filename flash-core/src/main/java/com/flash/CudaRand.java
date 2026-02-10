package com.flash;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;

/**
 * CUDA Random Number Generation (cuRAND) operations.
 *
 * <p>Provides GPU-accelerated random number generation:</p>
 * <ul>
 *   <li>Uniform distribution [0, 1) or custom range</li>
 *   <li>Normal (Gaussian) distribution</li>
 *   <li>Seed management for reproducibility</li>
 * </ul>
 *
 * <h2>Example usage:</h2>
 * <pre>{@code
 * try (CudaDevice device = new CudaDevice(0);
 *      CudaRand rand = new CudaRand(device, 12345)) {
 *
 *     // Generate uniform random numbers in [0, 1)
 *     CudaTensor uniform = CudaTensor.allocate(device, 1000, Precision.FP32);
 *     rand.uniform(uniform);
 *
 *     // Generate normal distribution (mean=0, stddev=1)
 *     CudaTensor normal = CudaTensor.allocate(device, 1000, Precision.FP32);
 *     rand.normal(normal, 0.0, 1.0);
 *
 *     // Weight initialization: Xavier/He initialization
 *     CudaTensor weights = CudaTensor.allocate(device, 768 * 768, Precision.FP32);
 *     double stddev = Math.sqrt(2.0 / 768);  // He initialization
 *     rand.normal(weights, 0.0, stddev);
 * }
 * }</pre>
 *
 * @since 0.4.0
 */
public class CudaRand implements AutoCloseable {

    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup LIBRARY;

    private static final MethodHandle RAND_INIT;
    private static final MethodHandle RAND_INIT_WITH_SEED;
    private static final MethodHandle RAND_DESTROY;
    private static final MethodHandle RAND_SET_SEED;
    private static final MethodHandle RAND_UNIFORM;
    private static final MethodHandle RAND_NORMAL;
    private static final MethodHandle RAND_UNIFORM_RANGE;
    private static final MethodHandle DEVICE_GET_HANDLE;

    private final CudaDevice device;
    private final MemorySegment context;
    private boolean closed = false;

    static {
        try {
            Path libPath = NativeLibraryLoader.loadLibrary();
            LIBRARY = SymbolLookup.libraryLookup(libPath, Arena.global());

            RAND_INIT = LINKER.downcallHandle(
                    LIBRARY.find("rand_init").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_LONG)
            );

            RAND_INIT_WITH_SEED = LINKER.downcallHandle(
                    LIBRARY.find("rand_init_with_seed").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG)
            );

            RAND_DESTROY = LINKER.downcallHandle(
                    LIBRARY.find("rand_destroy").orElseThrow(),
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS)
            );

            RAND_SET_SEED = LINKER.downcallHandle(
                    LIBRARY.find("rand_set_seed").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG)
            );

            RAND_UNIFORM = LINKER.downcallHandle(
                    LIBRARY.find("rand_uniform").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG)
            );

            RAND_NORMAL = LINKER.downcallHandle(
                    LIBRARY.find("rand_normal").orElseThrow(),
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_INT,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_LONG,
                            ValueLayout.JAVA_DOUBLE,
                            ValueLayout.JAVA_DOUBLE
                    )
            );

            RAND_UNIFORM_RANGE = LINKER.downcallHandle(
                    LIBRARY.find("rand_uniform_range").orElseThrow(),
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_INT,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_LONG,
                            ValueLayout.JAVA_DOUBLE,
                            ValueLayout.JAVA_DOUBLE
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
     * RAND operation status codes.
     */
    public enum Status {
        SUCCESS(0),
        INIT_ERROR(1),
        GENERATE_ERROR(2),
        INVALID_PARAMETER(3),
        PRECISION_NOT_SUPPORTED(4);

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

    /**
     * Creates a new CudaRand instance with default seed.
     *
     * @param device the CUDA device to use
     * @throws RuntimeException if cuRAND initialization fails
     */
    public CudaRand(CudaDevice device) {
        this(device, 0);
    }

    /**
     * Creates a new CudaRand instance with specified seed.
     *
     * @param device the CUDA device to use
     * @param seed random seed for reproducibility
     * @throws RuntimeException if cuRAND initialization fails
     */
    public CudaRand(CudaDevice device, long seed) {
        if (device == null || device.isClosed()) {
            throw new IllegalArgumentException("Device cannot be null or closed");
        }

        this.device = device;

        try {
            long deviceHandle = getDeviceHandle(device);
            context = (MemorySegment) RAND_INIT_WITH_SEED.invoke(deviceHandle, seed);
            if (context.address() == 0) {
                throw new RuntimeException("Failed to initialize cuRAND.");
            }
        } catch (Throwable e) {
            throw new RuntimeException("Failed to initialize cuRAND context", e);
        }
    }

    private long getDeviceHandle(CudaDevice device) throws Throwable {
        return (long) DEVICE_GET_HANDLE.invoke(device.getContextSegment());
    }

    // ========================================================================
    // Seed Management
    // ========================================================================

    /**
     * Sets the random seed for reproducibility.
     *
     * <p>Using the same seed will produce the same sequence of random numbers.</p>
     *
     * @param seed the random seed
     * @return operation status
     */
    public Status setSeed(long seed) {
        ensureNotClosed();

        try {
            int statusCode = (int) RAND_SET_SEED.invoke(context, seed);
            return Status.fromCode(statusCode);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to set seed", e);
        }
    }

    // ========================================================================
    // Uniform Distribution
    // ========================================================================

    /**
     * Fills tensor with uniform random values in [0, 1).
     *
     * @param tensor output tensor to fill
     * @return operation status
     */
    public Status uniform(CudaTensor tensor) {
        ensureNotClosed();

        try {
            int statusCode = (int) RAND_UNIFORM.invoke(context, tensor.getHandle());
            return Status.fromCode(statusCode);
        } catch (Throwable e) {
            throw new RuntimeException("Uniform generation failed", e);
        }
    }

    /**
     * Fills tensor with uniform random values in [low, high).
     *
     * @param tensor output tensor to fill
     * @param low lower bound (inclusive)
     * @param high upper bound (exclusive)
     * @return operation status
     */
    public Status uniform(CudaTensor tensor, double low, double high) {
        ensureNotClosed();

        if (low >= high) {
            throw new IllegalArgumentException("low must be less than high");
        }

        try {
            int statusCode = (int) RAND_UNIFORM_RANGE.invoke(context, tensor.getHandle(), low, high);
            return Status.fromCode(statusCode);
        } catch (Throwable e) {
            throw new RuntimeException("Uniform range generation failed", e);
        }
    }

    // ========================================================================
    // Normal Distribution
    // ========================================================================

    /**
     * Fills tensor with standard normal distribution (mean=0, stddev=1).
     *
     * @param tensor output tensor to fill
     * @return operation status
     */
    public Status normal(CudaTensor tensor) {
        return normal(tensor, 0.0, 1.0);
    }

    /**
     * Fills tensor with normal (Gaussian) distribution.
     *
     * @param tensor output tensor to fill
     * @param mean mean of the distribution
     * @param stddev standard deviation (must be positive)
     * @return operation status
     */
    public Status normal(CudaTensor tensor, double mean, double stddev) {
        ensureNotClosed();

        if (stddev <= 0) {
            throw new IllegalArgumentException("stddev must be positive");
        }

        try {
            int statusCode = (int) RAND_NORMAL.invoke(context, tensor.getHandle(), mean, stddev);
            return Status.fromCode(statusCode);
        } catch (Throwable e) {
            throw new RuntimeException("Normal generation failed", e);
        }
    }

    // ========================================================================
    // Convenience Methods for Weight Initialization
    // ========================================================================

    /**
     * Xavier/Glorot uniform initialization.
     *
     * <p>Fills tensor with uniform values in [-limit, limit] where
     * limit = sqrt(6 / (fan_in + fan_out))</p>
     *
     * @param tensor output tensor
     * @param fanIn number of input units
     * @param fanOut number of output units
     * @return operation status
     */
    public Status xavierUniform(CudaTensor tensor, int fanIn, int fanOut) {
        double limit = Math.sqrt(6.0 / (fanIn + fanOut));
        return uniform(tensor, -limit, limit);
    }

    /**
     * Xavier/Glorot normal initialization.
     *
     * <p>Fills tensor with normal values where
     * stddev = sqrt(2 / (fan_in + fan_out))</p>
     *
     * @param tensor output tensor
     * @param fanIn number of input units
     * @param fanOut number of output units
     * @return operation status
     */
    public Status xavierNormal(CudaTensor tensor, int fanIn, int fanOut) {
        double stddev = Math.sqrt(2.0 / (fanIn + fanOut));
        return normal(tensor, 0.0, stddev);
    }

    /**
     * Kaiming/He uniform initialization (for ReLU networks).
     *
     * <p>Fills tensor with uniform values in [-limit, limit] where
     * limit = sqrt(6 / fan_in)</p>
     *
     * @param tensor output tensor
     * @param fanIn number of input units
     * @return operation status
     */
    public Status kaimingUniform(CudaTensor tensor, int fanIn) {
        double limit = Math.sqrt(6.0 / fanIn);
        return uniform(tensor, -limit, limit);
    }

    /**
     * Kaiming/He normal initialization (for ReLU networks).
     *
     * <p>Fills tensor with normal values where stddev = sqrt(2 / fan_in)</p>
     *
     * @param tensor output tensor
     * @param fanIn number of input units
     * @return operation status
     */
    public Status kaimingNormal(CudaTensor tensor, int fanIn) {
        double stddev = Math.sqrt(2.0 / fanIn);
        return normal(tensor, 0.0, stddev);
    }

    // ========================================================================
    // Resource Management
    // ========================================================================

    @Override
    public void close() {
        if (!closed) {
            try {
                RAND_DESTROY.invoke(context);
                closed = true;
            } catch (Throwable e) {
                throw new RuntimeException("Failed to destroy cuRAND context", e);
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
            throw new IllegalStateException("CudaRand context is closed");
        }
    }
}