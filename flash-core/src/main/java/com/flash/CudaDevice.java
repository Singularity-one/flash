package com.flash;
import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.ByteBuffer;
import java.nio.file.Path;

/**
 * CUDA device management and memory operations.
 *
 * <p>This class provides low-level GPU device control including:</p>
 * <ul>
 *   <li>Device initialization and selection</li>
 *   <li>GPU memory allocation and deallocation</li>
 *   <li>Host-to-Device (H2D) and Device-to-Host (D2H) memory transfers</li>
 *   <li>Device synchronization</li>
 * </ul>
 *
 * <h2>Example usage:</h2>
 * <pre>{@code
 * try (CudaDevice device = new CudaDevice(0)) {
 *     System.out.println("GPU: " + device.getName());
 *
 *     // Allocate 1KB on GPU
 *     long devicePtr = device.allocate(1024);
 *
 *     // Copy data to GPU
 *     float[] data = {1.0f, 2.0f, 3.0f, 4.0f};
 *     device.copyHostToDevice(data, devicePtr);
 *
 *     // Copy back from GPU
 *     float[] result = new float[4];
 *     device.copyDeviceToHost(devicePtr, result);
 *
 *     // Free GPU memory
 *     device.free(devicePtr);
 * }
 * }</pre>
 *
 * <h2>Memory Layout:</h2>
 * <p>All memory operations use <b>row-major</b> layout for consistency with Java.</p>
 *
 * @author Singularity-one
 * @since 0.2.0
 */
public class CudaDevice implements AutoCloseable {

    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup LIBRARY;

    private static final MethodHandle DEVICE_INIT;
    private static final MethodHandle DEVICE_DESTROY;
    private static final MethodHandle DEVICE_GET_NAME;
    private static final MethodHandle DEVICE_GET_TOTAL_MEMORY;
    private static final MethodHandle DEVICE_ALLOCATE;
    private static final MethodHandle DEVICE_FREE;
    private static final MethodHandle DEVICE_COPY_HTOD;
    private static final MethodHandle DEVICE_COPY_DTOH;
    private static final MethodHandle DEVICE_SYNCHRONIZE;

    private final MemorySegment context;
    private boolean closed = false;

    static {
        try {
            Path libPath = NativeLibraryLoader.loadLibrary();
            LIBRARY = SymbolLookup.libraryLookup(libPath, Arena.global());

            DEVICE_INIT = LINKER.downcallHandle(
                    LIBRARY.find("device_init").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_INT)
            );

            DEVICE_DESTROY = LINKER.downcallHandle(
                    LIBRARY.find("device_destroy").orElseThrow(),
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS)
            );

            DEVICE_GET_NAME = LINKER.downcallHandle(
                    LIBRARY.find("device_get_name").orElseThrow(),
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_BOOLEAN,
                            ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_LONG
                    )
            );

            DEVICE_GET_TOTAL_MEMORY = LINKER.downcallHandle(
                    LIBRARY.find("device_get_total_memory").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.ADDRESS)
            );

            DEVICE_ALLOCATE = LINKER.downcallHandle(
                    LIBRARY.find("device_allocate").orElseThrow(),
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_LONG,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_LONG
                    )
            );

            DEVICE_FREE = LINKER.downcallHandle(
                    LIBRARY.find("device_free").orElseThrow(),
                    FunctionDescriptor.ofVoid(
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_LONG
                    )
            );

            DEVICE_COPY_HTOD = LINKER.downcallHandle(
                    LIBRARY.find("device_copy_htod").orElseThrow(),
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_INT,
                            ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_LONG,
                            ValueLayout.JAVA_LONG
                    )
            );

            DEVICE_COPY_DTOH = LINKER.downcallHandle(
                    LIBRARY.find("device_copy_dtoh").orElseThrow(),
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_INT,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_LONG,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_LONG
                    )
            );

            DEVICE_SYNCHRONIZE = LINKER.downcallHandle(
                    LIBRARY.find("device_synchronize").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS)
            );

        } catch (Exception e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    /**
     * Device operation status codes.
     */
    public enum Status {
        /** Operation completed successfully */
        SUCCESS(0),
        /** CUDA device not found */
        DEVICE_NOT_FOUND(1),
        /** GPU memory operation failed */
        MEMORY_ERROR(2),
        /** Memory copy operation failed */
        COPY_ERROR(3),
        /** Invalid parameter provided */
        INVALID_PARAMETER(4);

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
     * Creates a new CudaDevice instance for the specified device ID.
     *
     * @param deviceId CUDA device ID (typically 0 for single-GPU systems)
     * @throws RuntimeException if device initialization fails
     */
    public CudaDevice(int deviceId) {
        try {
            context = (MemorySegment) DEVICE_INIT.invoke(deviceId);
            if (context.address() == 0) {
                throw new RuntimeException(
                        "Failed to initialize CUDA device " + deviceId + ". " +
                                "Check: 1) GPU exists, 2) CUDA driver installed, 3) Device not in use"
                );
            }
        } catch (Throwable e) {
            throw new RuntimeException("Failed to initialize CUDA device " + deviceId, e);
        }
    }

    /**
     * Creates a CudaDevice for device 0 (default GPU).
     *
     * @throws RuntimeException if device initialization fails
     */
    public CudaDevice() {
        this(0);
    }

    /**
     * Returns the name of the GPU device.
     *
     * @return GPU device name (e.g., "NVIDIA GeForce RTX 3060 Ti")
     * @throws IllegalStateException if device is closed
     */
    public String getName() {
        ensureNotClosed();

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment buffer = arena.allocate(256);
            boolean success = (boolean) DEVICE_GET_NAME.invoke(context, buffer, 256L);

            if (!success) {
                return "Unknown Device";
            }
            return buffer.getString(0);
        } catch (Throwable e) {
            return "Error: " + e.getMessage();
        }
    }

    /**
     * Returns the total memory available on the GPU in bytes.
     *
     * @return total GPU memory in bytes
     * @throws IllegalStateException if device is closed
     */
    public long getTotalMemory() {
        ensureNotClosed();

        try {
            return (long) DEVICE_GET_TOTAL_MEMORY.invoke(context);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get total memory", e);
        }
    }

    /**
     * Allocates memory on the GPU.
     *
     * @param bytes number of bytes to allocate
     * @return device pointer (use with copy operations and BLAS functions)
     * @throws IllegalStateException if device is closed
     * @throws IllegalArgumentException if bytes <= 0
     * @throws RuntimeException if allocation fails
     */
    public long allocate(long bytes) {
        ensureNotClosed();

        if (bytes <= 0) {
            throw new IllegalArgumentException("Allocation size must be positive: " + bytes);
        }

        try {
            long ptr = (long) DEVICE_ALLOCATE.invoke(context, bytes);
            if (ptr == 0) {
                throw new RuntimeException(
                        "Failed to allocate " + bytes + " bytes on GPU. " +
                                "Check available GPU memory."
                );
            }
            return ptr;
        } catch (Throwable e) {
            throw new RuntimeException("GPU memory allocation failed", e);
        }
    }

    /**
     * Frees previously allocated GPU memory.
     *
     * @param devicePtr device pointer returned from {@link #allocate(long)}
     * @throws IllegalStateException if device is closed
     */
    public void free(long devicePtr) {
        ensureNotClosed();

        if (devicePtr == 0) {
            return;
        }

        try {
            DEVICE_FREE.invoke(context, devicePtr);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to free GPU memory", e);
        }
    }

    /**
     * Copies data from host (CPU) memory to device (GPU) memory.
     *
     * @param hostData source array on CPU
     * @param devicePtr destination pointer on GPU
     * @return operation status
     * @throws IllegalStateException if device is closed
     * @throws IllegalArgumentException if parameters are invalid
     */
    public Status copyHostToDevice(float[] hostData, long devicePtr) {
        ensureNotClosed();

        if (hostData == null || hostData.length == 0) {
            throw new IllegalArgumentException("Host data cannot be null or empty");
        }
        if (devicePtr == 0) {
            throw new IllegalArgumentException("Device pointer cannot be null");
        }

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment hostSeg = arena.allocateFrom(ValueLayout.JAVA_FLOAT, hostData);
            long bytes = (long) hostData.length * Float.BYTES;

            int statusCode = (int) DEVICE_COPY_HTOD.invoke(
                    context, hostSeg, devicePtr, bytes
            );

            return Status.fromCode(statusCode);
        } catch (Throwable e) {
            throw new RuntimeException("Host to device copy failed", e);
        }
    }

    /**
     * Copies data from device (GPU) memory to host (CPU) memory.
     *
     * @param devicePtr source pointer on GPU
     * @param hostData destination array on CPU
     * @return operation status
     * @throws IllegalStateException if device is closed
     * @throws IllegalArgumentException if parameters are invalid
     */
    public Status copyDeviceToHost(long devicePtr, float[] hostData) {
        ensureNotClosed();

        if (devicePtr == 0) {
            throw new IllegalArgumentException("Device pointer cannot be null");
        }
        if (hostData == null || hostData.length == 0) {
            throw new IllegalArgumentException("Host data cannot be null or empty");
        }

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment hostSeg = arena.allocateFrom(ValueLayout.JAVA_FLOAT, hostData);
            long bytes = (long) hostData.length * Float.BYTES;

            int statusCode = (int) DEVICE_COPY_DTOH.invoke(
                    context, devicePtr, hostSeg, bytes
            );

            MemorySegment.copy(hostSeg, ValueLayout.JAVA_FLOAT, 0, hostData, 0, hostData.length);

            return Status.fromCode(statusCode);
        } catch (Throwable e) {
            throw new RuntimeException("Device to host copy failed", e);
        }
    }

    /**
     * Synchronizes the device, blocking until all GPU operations complete.
     *
     * @return operation status
     * @throws IllegalStateException if device is closed
     */
    public Status synchronize() {
        ensureNotClosed();

        try {
            int statusCode = (int) DEVICE_SYNCHRONIZE.invoke(context);
            return Status.fromCode(statusCode);
        } catch (Throwable e) {
            throw new RuntimeException("Device synchronization failed", e);
        }
    }

    /**
     * Releases the device context and all associated resources.
     *
     * <p>After calling this method, the instance cannot be used.</p>
     */
    @Override
    public void close() {
        if (!closed) {
            try {
                DEVICE_DESTROY.invoke(context);
                closed = true;
            } catch (Throwable e) {
                throw new RuntimeException("Failed to destroy device context", e);
            }
        }
    }

    /**
     * Checks if this device has been closed.
     *
     * @return true if closed, false otherwise
     */
    public boolean isClosed() {
        return closed;
    }

    /**
     * Gets the internal context segment (package-private for CudaBlas).
     *
     * @return the context memory segment
     */
    MemorySegment getContextSegment() {
        return context;
    }

    private void ensureNotClosed() {
        if (closed) {
            throw new IllegalStateException("CudaDevice is closed");
        }
    }
}
