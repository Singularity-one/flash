package com.flash;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;

/**
 * CUDA Deep Neural Network (cuDNN) operations.
 *
 * <p>Provides GPU-accelerated neural network primitives:</p>
 * <ul>
 *   <li>Softmax (forward/backward)</li>
 *   <li>Activation functions (ReLU, Tanh, Sigmoid, GELU, SiLU)</li>
 *   <li>Layer Normalization (forward/backward) - coming soon</li>
 *   <li>Dropout (forward/backward) - coming soon</li>
 * </ul>
 *
 * <h2>Example usage:</h2>
 * <pre>{@code
 * try (CudaDevice device = new CudaDevice(0);
 *      CudaDnn dnn = new CudaDnn(device)) {
 *
 *     // Create input tensor
 *     float[] input = {1.0f, 2.0f, 3.0f, 4.0f};
 *     CudaTensor x = CudaTensor.fromFloat(device, input, Precision.FP32);
 *     CudaTensor y = CudaTensor.allocate(device, 4, Precision.FP32);
 *
 *     // Apply softmax: shape [1, 4, 1, 1] (N=1, C=4, H=1, W=1)
 *     dnn.softmax(1, 4, 1, 1, x, y);
 *
 *     // Apply GELU activation
 *     dnn.activation(ActivationType.GELU, x, y);
 *
 *     float[] result = y.toFloatArray();
 * }
 * }</pre>
 *
 * @since 0.3.0
 */
public class CudaDnn implements AutoCloseable {

    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup LIBRARY;

    private static final MethodHandle DNN_INIT;
    private static final MethodHandle DNN_DESTROY;
    private static final MethodHandle DNN_SOFTMAX_FORWARD;
    private static final MethodHandle DNN_ACTIVATION_FORWARD;
    private static final MethodHandle DNN_ACTIVATION_BACKWARD;
    private static final MethodHandle DEVICE_GET_HANDLE;

    private final CudaDevice device;
    private final MemorySegment context;
    private boolean closed = false;

    static {
        try {
            Path libPath = NativeLibraryLoader.loadLibrary();
            LIBRARY = SymbolLookup.libraryLookup(libPath, Arena.global());

            DNN_INIT = LINKER.downcallHandle(
                    LIBRARY.find("dnn_init").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_LONG)
            );

            DNN_DESTROY = LINKER.downcallHandle(
                    LIBRARY.find("dnn_destroy").orElseThrow(),
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS)
            );

            DNN_SOFTMAX_FORWARD = LINKER.downcallHandle(
                    LIBRARY.find("dnn_softmax_forward").orElseThrow(),
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_INT,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_INT,  // algo
                            ValueLayout.JAVA_INT,  // mode
                            ValueLayout.JAVA_INT,  // n
                            ValueLayout.JAVA_INT,  // c
                            ValueLayout.JAVA_INT,  // h
                            ValueLayout.JAVA_INT,  // w
                            ValueLayout.JAVA_LONG, // x_handle
                            ValueLayout.JAVA_LONG  // y_handle
                    )
            );

            DNN_ACTIVATION_FORWARD = LINKER.downcallHandle(
                    LIBRARY.find("dnn_activation_forward").orElseThrow(),
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_INT,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_INT,  // activation_type
                            ValueLayout.JAVA_LONG, // count
                            ValueLayout.JAVA_LONG, // x_handle
                            ValueLayout.JAVA_LONG  // y_handle
                    )
            );

            DNN_ACTIVATION_BACKWARD = LINKER.downcallHandle(
                    LIBRARY.find("dnn_activation_backward").orElseThrow(),
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_INT,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_INT,  // activation_type
                            ValueLayout.JAVA_LONG, // count
                            ValueLayout.JAVA_LONG, // x_handle
                            ValueLayout.JAVA_LONG, // y_handle
                            ValueLayout.JAVA_LONG, // dy_handle
                            ValueLayout.JAVA_LONG  // dx_handle
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
     * DNN operation status codes.
     */
    public enum Status {
        SUCCESS(0),
        INIT_ERROR(1),
        COMPUTE_ERROR(2),
        INVALID_PARAMETER(3),
        PRECISION_MISMATCH(4),
        NOT_SUPPORTED(5);

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
     * Activation function types.
     */
    public enum ActivationType {
        RELU(0),
        TANH(1),
        SIGMOID(2),
        GELU(3),
        SILU(4);

        private final int value;

        ActivationType(int value) {
            this.value = value;
        }

        public int getValue() {
            return value;
        }
    }

    /**
     * Softmax algorithm types.
     */
    public enum SoftmaxAlgorithm {
        FAST(0),
        ACCURATE(1),
        LOG(2);

        private final int value;

        SoftmaxAlgorithm(int value) {
            this.value = value;
        }

        public int getValue() {
            return value;
        }
    }

    /**
     * Softmax mode (which dimension to apply softmax).
     */
    public enum SoftmaxMode {
        INSTANCE(0),  // Apply across CHW for each N
        CHANNEL(1);   // Apply across C for each NHW position

        private final int value;

        SoftmaxMode(int value) {
            this.value = value;
        }

        public int getValue() {
            return value;
        }
    }

    /**
     * Creates a new CudaDnn instance using the specified device.
     *
     * @param device the CUDA device to use for operations
     * @throws RuntimeException if cuDNN initialization fails
     */
    public CudaDnn(CudaDevice device) {
        if (device == null || device.isClosed()) {
            throw new IllegalArgumentException("Device cannot be null or closed");
        }

        this.device = device;

        try {
            long deviceHandle = getDeviceHandle(device);
            context = (MemorySegment) DNN_INIT.invoke(deviceHandle);
            if (context.address() == 0) {
                throw new RuntimeException("Failed to initialize cuDNN.");
            }
        } catch (Throwable e) {
            throw new RuntimeException("Failed to initialize cuDNN context", e);
        }
    }

    private long getDeviceHandle(CudaDevice device) throws Throwable {
        return (long) DEVICE_GET_HANDLE.invoke(device.getContextSegment());
    }

    // ========================================================================
    // Softmax Operations
    // ========================================================================

    /**
     * Applies softmax to input tensor.
     *
     * <p>Computes: y = softmax(x)</p>
     *
     * <p>Tensor shape is NCHW format:</p>
     * <ul>
     *   <li>N: batch size</li>
     *   <li>C: channels (softmax is applied across this dimension)</li>
     *   <li>H: height</li>
     *   <li>W: width</li>
     * </ul>
     *
     * @param n batch size
     * @param c number of channels
     * @param h height
     * @param w width
     * @param x input tensor
     * @param y output tensor
     * @return operation status
     */
    public Status softmax(int n, int c, int h, int w, CudaTensor x, CudaTensor y) {
        return softmax(SoftmaxAlgorithm.ACCURATE, SoftmaxMode.INSTANCE, n, c, h, w, x, y);
    }

    /**
     * Applies softmax with specified algorithm and mode.
     *
     * @param algo softmax algorithm
     * @param mode softmax mode
     * @param n batch size
     * @param c number of channels
     * @param h height
     * @param w width
     * @param x input tensor
     * @param y output tensor
     * @return operation status
     */
    public Status softmax(SoftmaxAlgorithm algo, SoftmaxMode mode,
                          int n, int c, int h, int w,
                          CudaTensor x, CudaTensor y) {
        ensureNotClosed();

        try {
            int statusCode = (int) DNN_SOFTMAX_FORWARD.invoke(
                    context,
                    algo.getValue(),
                    mode.getValue(),
                    n, c, h, w,
                    x.getHandle(),
                    y.getHandle()
            );
            return Status.fromCode(statusCode);
        } catch (Throwable e) {
            throw new RuntimeException("Softmax operation failed", e);
        }
    }

    // ========================================================================
    // Activation Operations
    // ========================================================================

    /**
     * Applies activation function to input tensor.
     *
     * @param type activation type (RELU, TANH, SIGMOID, GELU, SILU)
     * @param x input tensor
     * @param y output tensor
     * @return operation status
     */
    public Status activation(ActivationType type, CudaTensor x, CudaTensor y) {
        ensureNotClosed();

        if (x.getElementCount() != y.getElementCount()) {
            throw new IllegalArgumentException("Input and output tensor sizes must match");
        }

        try {
            int statusCode = (int) DNN_ACTIVATION_FORWARD.invoke(
                    context,
                    type.getValue(),
                    x.getElementCount(),
                    x.getHandle(),
                    y.getHandle()
            );
            return Status.fromCode(statusCode);
        } catch (Throwable e) {
            throw new RuntimeException("Activation operation failed", e);
        }
    }

    /**
     * Convenience method for ReLU activation.
     */
    public Status relu(CudaTensor x, CudaTensor y) {
        return activation(ActivationType.RELU, x, y);
    }

    /**
     * Convenience method for GELU activation.
     */
    public Status gelu(CudaTensor x, CudaTensor y) {
        return activation(ActivationType.GELU, x, y);
    }

    /**
     * Convenience method for SiLU activation.
     */
    public Status silu(CudaTensor x, CudaTensor y) {
        return activation(ActivationType.SILU, x, y);
    }

    /**
     * Convenience method for Sigmoid activation.
     */
    public Status sigmoid(CudaTensor x, CudaTensor y) {
        return activation(ActivationType.SIGMOID, x, y);
    }

    /**
     * Convenience method for Tanh activation.
     */
    public Status tanh(CudaTensor x, CudaTensor y) {
        return activation(ActivationType.TANH, x, y);
    }

    // ========================================================================
    // Activation Backward
    // ========================================================================

    /**
     * Computes gradient of activation function.
     *
     * <p>Computes: dx = dy * activation'(x)</p>
     *
     * @param type activation type
     * @param x original input tensor (from forward pass)
     * @param y original output tensor (from forward pass)
     * @param dy gradient of output
     * @param dx gradient of input (output)
     * @return operation status
     */
    public Status activationBackward(ActivationType type,
                                     CudaTensor x, CudaTensor y,
                                     CudaTensor dy, CudaTensor dx) {
        ensureNotClosed();

        long count = x.getElementCount();

        try {
            int statusCode = (int) DNN_ACTIVATION_BACKWARD.invoke(
                    context,
                    type.getValue(),
                    count,
                    x.getHandle(),
                    y.getHandle(),
                    dy.getHandle(),
                    dx.getHandle()
            );
            return Status.fromCode(statusCode);
        } catch (Throwable e) {
            throw new RuntimeException("Activation backward operation failed", e);
        }
    }

    // ========================================================================
    // Resource Management
    // ========================================================================

    @Override
    public void close() {
        if (!closed) {
            try {
                DNN_DESTROY.invoke(context);
                closed = true;
            } catch (Throwable e) {
                throw new RuntimeException("Failed to destroy cuDNN context", e);
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
            throw new IllegalStateException("CudaDnn context is closed");
        }
    }
}