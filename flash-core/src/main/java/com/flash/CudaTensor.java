package com.flash;

import com.flash.exception.PrecisionMismatchException;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;

/**
 * GPU memory buffer with type information.
 *
 * <p>CudaTensor encapsulates a device memory pointer along with its
 * element count and data type. This allows flash to support FP16/BF16
 * which Java cannot represent natively.</p>
 *
 * <h2>Example usage:</h2>
 * <pre>{@code
 * try (CudaDevice device = new CudaDevice(0)) {
 *     // Create FP16 tensor from float array (conversion on GPU)
 *     float[] data = {1.0f, 2.0f, 3.0f, 4.0f};
 *     CudaTensor tensor = CudaTensor.fromFloat(device, data, Precision.FP16);
 *
 *     // Use in BLAS operations
 *     blas.gemm(2, 2, 2, 1.0, a, b, 0.0, c);
 *
 *     // Get result back as float array (conversion on GPU)
 *     float[] result = tensor.toFloatArray();
 *
 *     tensor.close();
 * }
 * }</pre>
 *
 * @author Singularity-one
 * @since 0.2.0
 */
public class CudaTensor implements AutoCloseable {

    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup LIBRARY;

    private static final MethodHandle TENSOR_ALLOCATE;
    private static final MethodHandle TENSOR_FREE;
    private static final MethodHandle TENSOR_COPY_FROM_F32;
    private static final MethodHandle TENSOR_COPY_TO_F32;
    private static final MethodHandle TENSOR_COPY_FROM_F64;
    private static final MethodHandle TENSOR_COPY_TO_F64;

    static {
        try {
            LIBRARY = NativeLibraryLoader.getLibrary();

            TENSOR_ALLOCATE = LINKER.downcallHandle(
                    LIBRARY.find("tensor_allocate").orElseThrow(),
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_LONG,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_LONG,
                            ValueLayout.JAVA_INT
                    )
            );

            TENSOR_FREE = LINKER.downcallHandle(
                    LIBRARY.find("tensor_free").orElseThrow(),
                    FunctionDescriptor.ofVoid(
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_LONG
                    )
            );

            TENSOR_COPY_FROM_F32 = LINKER.downcallHandle(
                    LIBRARY.find("tensor_copy_from_f32").orElseThrow(),
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_INT,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_LONG,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_LONG
                    )
            );

            TENSOR_COPY_TO_F32 = LINKER.downcallHandle(
                    LIBRARY.find("tensor_copy_to_f32").orElseThrow(),
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_INT,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_LONG,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_LONG
                    )
            );

            TENSOR_COPY_FROM_F64 = LINKER.downcallHandle(
                    LIBRARY.find("tensor_copy_from_f64").orElseThrow(),
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_INT,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_LONG,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_LONG
                    )
            );

            TENSOR_COPY_TO_F64 = LINKER.downcallHandle(
                    LIBRARY.find("tensor_copy_to_f64").orElseThrow(),
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_INT,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_LONG,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_LONG
                    )
            );

        } catch (Exception e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    private long handle;
    private final long elementCount;
    private final Precision dtype;
    private final CudaDevice device;
    private boolean closed = false;

    /**
     * Private constructor - use factory methods instead.
     */
    private CudaTensor(long handle, long elementCount, Precision dtype, CudaDevice device) {
        this.handle = handle;
        this.elementCount = elementCount;
        this.dtype = dtype;
        this.device = device;
    }

    // ========================================================================
    // Factory Methods
    // ========================================================================

    /**
     * Allocates an empty tensor on the GPU.
     *
     * @param device the CUDA device
     * @param elementCount number of elements
     * @param dtype data precision
     * @return new CudaTensor with uninitialized data
     * @throws RuntimeException if allocation fails
     */
    public static CudaTensor allocate(CudaDevice device, long elementCount, Precision dtype) {
        if (device == null || device.isClosed()) {
            throw new IllegalArgumentException("Device cannot be null or closed");
        }
        if (elementCount <= 0) {
            throw new IllegalArgumentException("Element count must be positive: " + elementCount);
        }

        try {
            long handle = (long) TENSOR_ALLOCATE.invoke(
                    device.getContextSegment(),
                    elementCount,
                    dtype.getValue()
            );

            if (handle == 0) {
                throw new RuntimeException(
                        "Failed to allocate tensor: " + elementCount + " elements of " + dtype
                );
            }

            return new CudaTensor(handle, elementCount, dtype, device);
        } catch (Throwable e) {
            throw new RuntimeException("Tensor allocation failed", e);
        }
    }

    /**
     * Creates a tensor from a float array with GPU precision conversion.
     *
     * @param device the CUDA device
     * @param data source data (FP32)
     * @param dtype target precision on GPU
     * @return new CudaTensor with converted data
     * @throws RuntimeException if creation fails
     */
    public static CudaTensor fromFloat(CudaDevice device, float[] data, Precision dtype) {
        if (data == null || data.length == 0) {
            throw new IllegalArgumentException("Data cannot be null or empty");
        }

        CudaTensor tensor = allocate(device, data.length, dtype);

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment hostSeg = arena.allocateFrom(ValueLayout.JAVA_FLOAT, data);

            int status = (int) TENSOR_COPY_FROM_F32.invoke(
                    device.getContextSegment(),
                    tensor.handle,
                    hostSeg,
                    (long) data.length
            );

            if (status != 0) {
                tensor.close();
                throw new RuntimeException("Failed to copy data to tensor (status: " + status + ")");
            }

            return tensor;
        } catch (Throwable e) {
            tensor.close();
            throw new RuntimeException("Failed to create tensor from float array", e);
        }
    }

    /**
     * Creates a tensor from a double array with GPU precision conversion.
     *
     * @param device the CUDA device
     * @param data source data (FP64)
     * @param dtype target precision on GPU
     * @return new CudaTensor with converted data
     * @throws RuntimeException if creation fails
     */
    public static CudaTensor fromDouble(CudaDevice device, double[] data, Precision dtype) {
        if (data == null || data.length == 0) {
            throw new IllegalArgumentException("Data cannot be null or empty");
        }

        CudaTensor tensor = allocate(device, data.length, dtype);

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment hostSeg = arena.allocateFrom(ValueLayout.JAVA_DOUBLE, data);

            int status = (int) TENSOR_COPY_FROM_F64.invoke(
                    device.getContextSegment(),
                    tensor.handle,
                    hostSeg,
                    (long) data.length
            );

            if (status != 0) {
                tensor.close();
                throw new RuntimeException("Failed to copy data to tensor (status: " + status + ")");
            }

            return tensor;
        } catch (Throwable e) {
            tensor.close();
            throw new RuntimeException("Failed to create tensor from double array", e);
        }
    }

    // ========================================================================
    // Data Retrieval
    // ========================================================================

    /**
     * Retrieves tensor data as a float array (GPU converts to FP32).
     *
     * @return float array containing tensor data
     * @throws IllegalStateException if tensor is closed
     * @throws RuntimeException if copy fails
     */
    public float[] toFloatArray() {
        ensureNotClosed();

        try (Arena arena = Arena.ofConfined()) {
            float[] result = new float[(int) elementCount];
            MemorySegment hostSeg = arena.allocateFrom(ValueLayout.JAVA_FLOAT, result);

            int status = (int) TENSOR_COPY_TO_F32.invoke(
                    device.getContextSegment(),
                    handle,
                    hostSeg,
                    elementCount
            );

            if (status != 0) {
                throw new RuntimeException("Failed to copy tensor to float array (status: " + status + ")");
            }

            MemorySegment.copy(hostSeg, ValueLayout.JAVA_FLOAT, 0, result, 0, result.length);
            return result;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to retrieve tensor as float array", e);
        }
    }

    /**
     * Retrieves tensor data as a double array (GPU converts to FP64).
     *
     * @return double array containing tensor data
     * @throws IllegalStateException if tensor is closed
     * @throws RuntimeException if copy fails
     */
    public double[] toDoubleArray() {
        ensureNotClosed();

        try (Arena arena = Arena.ofConfined()) {
            double[] result = new double[(int) elementCount];
            MemorySegment hostSeg = arena.allocateFrom(ValueLayout.JAVA_DOUBLE, result);

            int status = (int) TENSOR_COPY_TO_F64.invoke(
                    device.getContextSegment(),
                    handle,
                    hostSeg,
                    elementCount
            );

            if (status != 0) {
                throw new RuntimeException("Failed to copy tensor to double array (status: " + status + ")");
            }

            MemorySegment.copy(hostSeg, ValueLayout.JAVA_DOUBLE, 0, result, 0, result.length);
            return result;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to retrieve tensor as double array", e);
        }
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    /**
     * Gets the device pointer handle for FFI calls.
     *
     * @return device pointer (opaque handle)
     */
    public long getHandle() {
        ensureNotClosed();
        return handle;
    }

    /**
     * Gets the number of elements in this tensor.
     *
     * @return element count
     */
    public long getElementCount() {
        return elementCount;
    }

    /**
     * Gets the data precision of this tensor.
     *
     * @return precision type
     */
    public Precision getPrecision() {
        return dtype;
    }

    /**
     * Gets the associated CUDA device.
     *
     * @return the CUDA device
     */
    public CudaDevice getDevice() {
        return device;
    }

    /**
     * Checks if this tensor has been closed.
     *
     * @return true if closed, false otherwise
     */
    public boolean isClosed() {
        return closed;
    }

    /**
     * Gets the total size in bytes.
     *
     * @return size in bytes
     */
    public long sizeInBytes() {
        return elementCount * dtype.getBytesPerElement();
    }

    // ========================================================================
    // Precision Validation
    // ========================================================================

    /**
     * Ensures all tensors have the same precision.
     *
     * @param tensors tensors to check
     * @throws PrecisionMismatchException if precisions don't match
     */
    public static void ensureSamePrecision(CudaTensor... tensors) {
        if (tensors.length < 2) {
            return;
        }

        Precision expected = tensors[0].dtype;
        for (int i = 1; i < tensors.length; i++) {
            if (tensors[i].dtype != expected) {
                throw new PrecisionMismatchException(
                        "Precision mismatch: expected " + expected +
                                " but tensor[" + i + "] has " + tensors[i].dtype
                );
            }
        }
    }

    // ========================================================================
    // Resource Management
    // ========================================================================

    /**
     * Releases the GPU memory held by this tensor.
     *
     * <p>After calling this method, the tensor cannot be used.</p>
     */
    @Override
    public void close() {
        if (!closed) {
            try {
                TENSOR_FREE.invoke(device.getContextSegment(), handle);
                closed = true;
            } catch (Throwable e) {
                throw new RuntimeException("Failed to free tensor", e);
            }
        }
    }

    private void ensureNotClosed() {
        if (closed) {
            throw new IllegalStateException("CudaTensor is closed");
        }
    }

    @Override
    public String toString() {
        if (closed) {
            return "CudaTensor[CLOSED]";
        }
        return String.format("CudaTensor[elements=%d, dtype=%s, bytes=%d]",
                elementCount, dtype, sizeInBytes());
    }
}