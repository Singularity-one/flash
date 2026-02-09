package com.flash.test;

import com.flash.CudaDevice;
import com.flash.CudaTensor;
import com.flash.Precision;
import com.flash.exception.PrecisionMismatchException;
import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for CudaTensor.
 */
@DisplayName("CudaTensor API")
class CudaTensorTest {

    private static CudaDevice device;

    @BeforeAll
    static void setUp() {
        device = new CudaDevice(0);
        System.out.println("  Device: " + device.getName());
    }

    @AfterAll
    static void tearDown() {
        if (device != null) {
            device.close();
        }
    }

    // ========================================================================
    // Allocation Tests
    // ========================================================================

    @Test
    @DisplayName("Allocate FP32 tensor")
    void testAllocateFP32() {
        try (CudaTensor tensor = CudaTensor.allocate(device, 100, Precision.FP32)) {
            assertNotNull(tensor);
            assertEquals(100, tensor.getElementCount());
            assertEquals(Precision.FP32, tensor.getPrecision());
            assertEquals(400, tensor.sizeInBytes()); // 100 * 4
            assertFalse(tensor.isClosed());
        }
    }

    @Test
    @DisplayName("Allocate FP16 tensor")
    void testAllocateFP16() {
        try (CudaTensor tensor = CudaTensor.allocate(device, 100, Precision.FP16)) {
            assertEquals(100, tensor.getElementCount());
            assertEquals(Precision.FP16, tensor.getPrecision());
            assertEquals(200, tensor.sizeInBytes()); // 100 * 2
        }
    }

    @Test
    @DisplayName("Allocate FP64 tensor")
    void testAllocateFP64() {
        try (CudaTensor tensor = CudaTensor.allocate(device, 100, Precision.FP64)) {
            assertEquals(100, tensor.getElementCount());
            assertEquals(Precision.FP64, tensor.getPrecision());
            assertEquals(800, tensor.sizeInBytes()); // 100 * 8
        }
    }

    @Test
    @DisplayName("Allocate BF16 tensor")
    void testAllocateBF16() {
        try (CudaTensor tensor = CudaTensor.allocate(device, 100, Precision.BF16)) {
            assertEquals(100, tensor.getElementCount());
            assertEquals(Precision.BF16, tensor.getPrecision());
            assertEquals(200, tensor.sizeInBytes()); // 100 * 2
        }
    }

    @Test
    @DisplayName("Invalid allocation throws exception")
    void testInvalidAllocation() {
        assertThrows(IllegalArgumentException.class, () -> {
            CudaTensor.allocate(device, 0, Precision.FP32);
        });

        assertThrows(IllegalArgumentException.class, () -> {
            CudaTensor.allocate(device, -100, Precision.FP32);
        });

        assertThrows(IllegalArgumentException.class, () -> {
            CudaTensor.allocate(null, 100, Precision.FP32);
        });
    }

    // ========================================================================
    // FP32 Round-trip Tests
    // ========================================================================

    @Test
    @DisplayName("FP32 tensor: float round-trip")
    void testFP32FloatRoundtrip() {
        float[] original = {1.0f, 2.5f, -3.7f, 4.2f, 0.0f};

        try (CudaTensor tensor = CudaTensor.fromFloat(device, original, Precision.FP32)) {
            float[] result = tensor.toFloatArray();

            assertArrayEquals(original, result, 1e-6f);
        }
    }

    @Test
    @DisplayName("FP32 tensor: double round-trip")
    void testFP32DoubleRoundtrip() {
        double[] original = {1.0, 2.5, -3.7, 4.2, 0.0};

        try (CudaTensor tensor = CudaTensor.fromDouble(device, original, Precision.FP32)) {
            double[] result = tensor.toDoubleArray();

            assertArrayEquals(original, result, 1e-6);
        }
    }

    // ========================================================================
    // FP16 Conversion Tests
    // ========================================================================

    @Test
    @DisplayName("FP16 tensor: float conversion")
    void testFP16FloatConversion() {
        float[] original = {1.0f, 2.5f, 3.7f, 4.2f};

        try (CudaTensor tensor = CudaTensor.fromFloat(device, original, Precision.FP16)) {
            assertEquals(Precision.FP16, tensor.getPrecision());

            float[] result = tensor.toFloatArray();

            // FP16 has lower precision, use larger tolerance
            for (int i = 0; i < original.length; i++) {
                assertEquals(original[i], result[i], 0.01f,
                        "Mismatch at index " + i);
            }
        }
    }

    @Test
    @DisplayName("FP16 tensor: precision loss acceptable")
    void testFP16PrecisionLoss() {
        float[] original = {1.2345f, 2.3456f, 3.4567f};

        try (CudaTensor tensor = CudaTensor.fromFloat(device, original, Precision.FP16)) {
            float[] result = tensor.toFloatArray();

            // FP16 loses precision but should be close
            for (int i = 0; i < original.length; i++) {
                assertTrue(Math.abs(original[i] - result[i]) < 0.01,
                        "FP16 precision loss too large at index " + i);
            }
        }
    }

    // ========================================================================
    // FP64 Conversion Tests
    // ========================================================================

    @Test
    @DisplayName("FP64 tensor: double round-trip")
    void testFP64DoubleRoundtrip() {
        double[] original = {1.234567890123, 2.345678901234, -3.456789012345};

        try (CudaTensor tensor = CudaTensor.fromDouble(device, original, Precision.FP64)) {
            assertEquals(Precision.FP64, tensor.getPrecision());

            double[] result = tensor.toDoubleArray();

            assertArrayEquals(original, result, 1e-10);
        }
    }

    @Test
    @DisplayName("FP64 tensor: float to double conversion")
    void testFP64FromFloat() {
        float[] original = {1.5f, 2.5f, 3.5f};

        try (CudaTensor tensor = CudaTensor.fromFloat(device, original, Precision.FP64)) {
            double[] result = tensor.toDoubleArray();

            for (int i = 0; i < original.length; i++) {
                assertEquals((double) original[i], result[i], 1e-6);
            }
        }
    }

    // ========================================================================
    // BF16 Conversion Tests
    // ========================================================================

    @Test
    @DisplayName("BF16 tensor: float conversion")
    void testBF16FloatConversion() {
        float[] original = {1.0f, 2.0f, 3.0f, 4.0f};

        try (CudaTensor tensor = CudaTensor.fromFloat(device, original, Precision.BF16)) {
            assertEquals(Precision.BF16, tensor.getPrecision());

            float[] result = tensor.toFloatArray();

            // BF16 has lower precision than FP16
            for (int i = 0; i < original.length; i++) {
                assertEquals(original[i], result[i], 0.05f,
                        "Mismatch at index " + i);
            }
        }
    }

    // ========================================================================
    // Precision Matching Tests
    // ========================================================================

    @Test
    @DisplayName("ensureSamePrecision: matching precisions")
    void testEnsureSamePrecisionMatching() {
        try (CudaTensor t1 = CudaTensor.allocate(device, 10, Precision.FP32);
             CudaTensor t2 = CudaTensor.allocate(device, 10, Precision.FP32);
             CudaTensor t3 = CudaTensor.allocate(device, 10, Precision.FP32)) {

            assertDoesNotThrow(() -> {
                CudaTensor.ensureSamePrecision(t1, t2, t3);
            });
        }
    }

    @Test
    @DisplayName("ensureSamePrecision: mismatched precisions")
    void testEnsureSamePrecisionMismatch() {
        try (CudaTensor t1 = CudaTensor.allocate(device, 10, Precision.FP32);
             CudaTensor t2 = CudaTensor.allocate(device, 10, Precision.FP16)) {

            assertThrows(PrecisionMismatchException.class, () -> {
                CudaTensor.ensureSamePrecision(t1, t2);
            });
        }
    }

    // ========================================================================
    // Resource Management Tests
    // ========================================================================

    @Test
    @DisplayName("Tensor can be closed")
    void testClose() {
        CudaTensor tensor = CudaTensor.allocate(device, 100, Precision.FP32);
        assertFalse(tensor.isClosed());

        tensor.close();
        assertTrue(tensor.isClosed());
    }

    @Test
    @DisplayName("Operations on closed tensor throw exception")
    void testClosedTensorThrows() {
        CudaTensor tensor = CudaTensor.allocate(device, 100, Precision.FP32);
        tensor.close();

        assertThrows(IllegalStateException.class, () -> {
            tensor.getHandle();
        });

        assertThrows(IllegalStateException.class, () -> {
            tensor.toFloatArray();
        });
    }

    @Test
    @DisplayName("try-with-resources auto-closes")
    void testAutoClose() {
        CudaTensor tensor;

        try (CudaTensor t = CudaTensor.allocate(device, 100, Precision.FP32)) {
            tensor = t;
            assertFalse(tensor.isClosed());
        }

        assertTrue(tensor.isClosed());
    }

    // ========================================================================
    // toString Tests
    // ========================================================================

    @Test
    @DisplayName("toString provides useful info")
    void testToString() {
        try (CudaTensor tensor = CudaTensor.allocate(device, 100, Precision.FP16)) {
            String str = tensor.toString();

            assertTrue(str.contains("100"));
            assertTrue(str.contains("FP16"));
            assertTrue(str.contains("200")); // bytes
        }
    }

    @Test
    @DisplayName("toString shows CLOSED state")
    void testToStringClosed() {
        CudaTensor tensor = CudaTensor.allocate(device, 100, Precision.FP32);
        tensor.close();

        String str = tensor.toString();
        assertTrue(str.contains("CLOSED"));
    }
}