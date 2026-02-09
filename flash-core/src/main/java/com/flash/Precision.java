package com.flash;

/**
 * Supported numerical precisions for GPU operations.
 *
 * <p>Note: FP16 and BF16 are not natively supported by Java.
 * Use {@link CudaTensor} to handle these types.</p>
 *
 * <h2>Precision Support:</h2>
 * <ul>
 *   <li><b>FP32</b> - 32-bit single precision (native Java {@code float})</li>
 *   <li><b>FP16</b> - 16-bit half precision (requires GPU conversion)</li>
 *   <li><b>FP64</b> - 64-bit double precision (native Java {@code double})</li>
 *   <li><b>BF16</b> - 16-bit Brain Float (requires Ampere+ GPU, GPU conversion)</li>
 *   <li><b>INT8</b> - 8-bit integer (reserved for future use)</li>
 *   <li><b>INT4</b> - 4-bit integer (reserved for future use)</li>
 * </ul>
 *
 * @author Singularity-one
 * @since 0.2.0
 */
public enum Precision {
    /** 32-bit single precision floating point */
    FP32(0, 4),

    /** 16-bit half precision floating point */
    FP16(1, 2),

    /** 64-bit double precision floating point */
    FP64(2, 8),

    /** 16-bit Brain Float (requires Ampere+ GPU) */
    BF16(3, 2),

    /** 8-bit integer (reserved for future use) */
    INT8(4, 1),

    /** 4-bit integer (reserved for future use) */
    INT4(5, 1);

    private final int value;
    private final int bytesPerElement;

    Precision(int value, int bytesPerElement) {
        this.value = value;
        this.bytesPerElement = bytesPerElement;
    }

    /**
     * Gets the integer value used in native FFI calls.
     *
     * @return integer representation of this precision
     */
    public int getValue() {
        return value;
    }

    /**
     * Gets the number of bytes per element for this precision.
     *
     * @return bytes per element
     */
    public int getBytesPerElement() {
        return bytesPerElement;
    }

    /**
     * Converts an integer value to Precision enum.
     *
     * @param value the integer value from native code
     * @return corresponding Precision enum
     * @throws IllegalArgumentException if value is unknown
     */
    public static Precision fromValue(int value) {
        for (Precision p : values()) {
            if (p.value == value) {
                return p;
            }
        }
        throw new IllegalArgumentException("Unknown precision value: " + value);
    }

    /**
     * Checks if this precision is natively supported by Java.
     *
     * @return true if FP32 or FP64, false otherwise
     */
    public boolean isNativeJavaType() {
        return this == FP32 || this == FP64;
    }

    /**
     * Checks if this precision requires GPU conversion from Java types.
     *
     * @return true if FP16 or BF16, false otherwise
     */
    public boolean requiresGpuConversion() {
        return this == FP16 || this == BF16;
    }
}
