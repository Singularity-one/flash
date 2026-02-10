package com.flash.test;

import com.flash.CudaDevice;
import com.flash.CudaOps;
import com.flash.CudaTensor;
import com.flash.Precision;
import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for CudaOps.
 */
@DisplayName("CudaOps API")
class CudaOpsTest {

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
    // Binary Operations Tests
    // ========================================================================

    @Test
    @DisplayName("Add: c = a + b")
    void testAdd() {
        try (CudaTensor a = CudaTensor.fromFloat(device, new float[]{1, 2, 3, 4}, Precision.FP32);
             CudaTensor b = CudaTensor.fromFloat(device, new float[]{5, 6, 7, 8}, Precision.FP32);
             CudaTensor c = CudaTensor.allocate(device, 4, Precision.FP32)) {

            var status = CudaOps.add(device, a, b, c);
            assertEquals(CudaOps.Status.SUCCESS, status);

            float[] result = c.toFloatArray();
            assertArrayEquals(new float[]{6, 8, 10, 12}, result, 1e-6f);
        }
    }

    @Test
    @DisplayName("Sub: c = a - b")
    void testSub() {
        try (CudaTensor a = CudaTensor.fromFloat(device, new float[]{10, 20, 30, 40}, Precision.FP32);
             CudaTensor b = CudaTensor.fromFloat(device, new float[]{1, 2, 3, 4}, Precision.FP32);
             CudaTensor c = CudaTensor.allocate(device, 4, Precision.FP32)) {

            var status = CudaOps.sub(device, a, b, c);
            assertEquals(CudaOps.Status.SUCCESS, status);

            float[] result = c.toFloatArray();
            assertArrayEquals(new float[]{9, 18, 27, 36}, result, 1e-6f);
        }
    }

    @Test
    @DisplayName("Mul: c = a * b")
    void testMul() {
        try (CudaTensor a = CudaTensor.fromFloat(device, new float[]{1, 2, 3, 4}, Precision.FP32);
             CudaTensor b = CudaTensor.fromFloat(device, new float[]{2, 3, 4, 5}, Precision.FP32);
             CudaTensor c = CudaTensor.allocate(device, 4, Precision.FP32)) {

            var status = CudaOps.mul(device, a, b, c);
            assertEquals(CudaOps.Status.SUCCESS, status);

            float[] result = c.toFloatArray();
            assertArrayEquals(new float[]{2, 6, 12, 20}, result, 1e-6f);
        }
    }

    @Test
    @DisplayName("Div: c = a / b")
    void testDiv() {
        try (CudaTensor a = CudaTensor.fromFloat(device, new float[]{10, 20, 30, 40}, Precision.FP32);
             CudaTensor b = CudaTensor.fromFloat(device, new float[]{2, 4, 5, 8}, Precision.FP32);
             CudaTensor c = CudaTensor.allocate(device, 4, Precision.FP32)) {

            var status = CudaOps.div(device, a, b, c);
            assertEquals(CudaOps.Status.SUCCESS, status);

            float[] result = c.toFloatArray();
            assertArrayEquals(new float[]{5, 5, 6, 5}, result, 1e-6f);
        }
    }

    // ========================================================================
    // Unary Operations Tests
    // ========================================================================

    @Test
    @DisplayName("Exp: y = exp(x)")
    void testExp() {
        try (CudaTensor x = CudaTensor.fromFloat(device, new float[]{0, 1, 2, -1}, Precision.FP32);
             CudaTensor y = CudaTensor.allocate(device, 4, Precision.FP32)) {

            var status = CudaOps.exp(device, x, y);
            assertEquals(CudaOps.Status.SUCCESS, status);

            float[] result = y.toFloatArray();
            assertEquals(1.0f, result[0], 1e-5f);
            assertEquals(2.71828f, result[1], 1e-4f);
            assertEquals(7.38906f, result[2], 1e-4f);
            assertEquals(0.36788f, result[3], 1e-4f);
        }
    }

    @Test
    @DisplayName("Sqrt: y = sqrt(x)")
    void testSqrt() {
        try (CudaTensor x = CudaTensor.fromFloat(device, new float[]{1, 4, 9, 16}, Precision.FP32);
             CudaTensor y = CudaTensor.allocate(device, 4, Precision.FP32)) {

            var status = CudaOps.sqrt(device, x, y);
            assertEquals(CudaOps.Status.SUCCESS, status);

            float[] result = y.toFloatArray();
            assertArrayEquals(new float[]{1, 2, 3, 4}, result, 1e-6f);
        }
    }

    @Test
    @DisplayName("ReLU: y = max(0, x)")
    void testRelu() {
        try (CudaTensor x = CudaTensor.fromFloat(device, new float[]{-2, -1, 0, 1, 2}, Precision.FP32);
             CudaTensor y = CudaTensor.allocate(device, 5, Precision.FP32)) {

            var status = CudaOps.relu(device, x, y);
            assertEquals(CudaOps.Status.SUCCESS, status);

            float[] result = y.toFloatArray();
            assertArrayEquals(new float[]{0, 0, 0, 1, 2}, result, 1e-6f);
        }
    }

    @Test
    @DisplayName("Sigmoid: y = 1 / (1 + exp(-x))")
    void testSigmoid() {
        try (CudaTensor x = CudaTensor.fromFloat(device, new float[]{0, 1, -1, 10}, Precision.FP32);
             CudaTensor y = CudaTensor.allocate(device, 4, Precision.FP32)) {

            var status = CudaOps.sigmoid(device, x, y);
            assertEquals(CudaOps.Status.SUCCESS, status);

            float[] result = y.toFloatArray();
            assertEquals(0.5f, result[0], 1e-5f);
            assertEquals(0.7311f, result[1], 1e-3f);
            assertEquals(0.2689f, result[2], 1e-3f);
            assertTrue(result[3] > 0.999f);
        }
    }

    @Test
    @DisplayName("GELU activation")
    void testGelu() {
        try (CudaTensor x = CudaTensor.fromFloat(device, new float[]{-1, 0, 1, 2}, Precision.FP32);
             CudaTensor y = CudaTensor.allocate(device, 4, Precision.FP32)) {

            var status = CudaOps.gelu(device, x, y);
            assertEquals(CudaOps.Status.SUCCESS, status);

            float[] result = y.toFloatArray();
            assertEquals(-0.159f, result[0], 0.01f);
            assertEquals(0.0f, result[1], 0.001f);
            assertEquals(0.841f, result[2], 0.01f);
            assertEquals(1.955f, result[3], 0.01f);
        }
    }

    @Test
    @DisplayName("SiLU (Swish) activation")
    void testSilu() {
        try (CudaTensor x = CudaTensor.fromFloat(device, new float[]{0, 1, -1, 2}, Precision.FP32);
             CudaTensor y = CudaTensor.allocate(device, 4, Precision.FP32)) {

            var status = CudaOps.silu(device, x, y);
            assertEquals(CudaOps.Status.SUCCESS, status);

            float[] result = y.toFloatArray();
            assertEquals(0.0f, result[0], 1e-5f);
            assertEquals(0.7311f, result[1], 1e-3f);
            assertEquals(-0.2689f, result[2], 1e-3f);
        }
    }

    @Test
    @DisplayName("Pow: y = x^n")
    void testPow() {
        try (CudaTensor x = CudaTensor.fromFloat(device, new float[]{1, 2, 3, 4}, Precision.FP32);
             CudaTensor y = CudaTensor.allocate(device, 4, Precision.FP32)) {

            var status = CudaOps.pow(device, x, y, 2.0);
            assertEquals(CudaOps.Status.SUCCESS, status);

            float[] result = y.toFloatArray();
            assertArrayEquals(new float[]{1, 4, 9, 16}, result, 1e-5f);
        }
    }
    // ========================================================================
    // Scalar Operations Tests (繼續加到 CudaOpsTest class 內)
    // ========================================================================

    @Test
    @DisplayName("Scale: y = alpha * x")
    void testScale() {
        try (CudaTensor x = CudaTensor.fromFloat(device, new float[]{1, 2, 3, 4}, Precision.FP32);
             CudaTensor y = CudaTensor.allocate(device, 4, Precision.FP32)) {

            var status = CudaOps.scale(device, x, y, 2.5);
            assertEquals(CudaOps.Status.SUCCESS, status);

            float[] result = y.toFloatArray();
            assertArrayEquals(new float[]{2.5f, 5f, 7.5f, 10f}, result, 1e-5f);
        }
    }

    @Test
    @DisplayName("Fill: x[:] = value")
    void testFill() {
        try (CudaTensor x = CudaTensor.allocate(device, 4, Precision.FP32)) {

            var status = CudaOps.fill(device, x, 3.14);
            assertEquals(CudaOps.Status.SUCCESS, status);

            float[] result = x.toFloatArray();
            for (float v : result) {
                assertEquals(3.14f, v, 1e-5f);
            }
        }
    }

    // ========================================================================
    // Reduction Operations Tests
    // ========================================================================

    @Test
    @DisplayName("Sum: reduce sum")
    void testSum() {
        try (CudaTensor x = CudaTensor.fromFloat(device, new float[]{1, 2, 3, 4, 5}, Precision.FP32)) {
            double sum = CudaOps.sum(device, x);
            assertEquals(15.0, sum, 1e-6);
        }
    }

    @Test
    @DisplayName("Max: reduce max")
    void testMax() {
        try (CudaTensor x = CudaTensor.fromFloat(device, new float[]{1, 5, 3, 2, 4}, Precision.FP32)) {
            double max = CudaOps.max(device, x);
            assertEquals(5.0, max, 1e-6);
        }
    }

    @Test
    @DisplayName("Min: reduce min")
    void testMin() {
        try (CudaTensor x = CudaTensor.fromFloat(device, new float[]{5, 1, 3, 2, 4}, Precision.FP32)) {
            double min = CudaOps.min(device, x);
            assertEquals(1.0, min, 1e-6);
        }
    }

    @Test
    @DisplayName("Mean: reduce mean")
    void testMean() {
        try (CudaTensor x = CudaTensor.fromFloat(device, new float[]{1, 2, 3, 4, 5}, Precision.FP32)) {
            double mean = CudaOps.mean(device, x);
            assertEquals(3.0, mean, 1e-6);
        }
    }

    // ========================================================================
    // Cast Operation Tests
    // ========================================================================

    @Test
    @DisplayName("Cast: FP32 to FP16")
    void testCastFP32toFP16() {
        try (CudaTensor src = CudaTensor.fromFloat(device, new float[]{1.5f, 2.5f, 3.5f, 4.5f}, Precision.FP32);
             CudaTensor dst = CudaTensor.allocate(device, 4, Precision.FP16)) {

            var status = CudaOps.cast(device, src, dst);
            assertEquals(CudaOps.Status.SUCCESS, status);

            float[] result = dst.toFloatArray();
            assertEquals(1.5f, result[0], 0.01f);
            assertEquals(2.5f, result[1], 0.01f);
        }
    }

    // ========================================================================
    // In-place Operations Tests
    // ========================================================================

    @Test
    @DisplayName("AXPY: y = y + alpha * x")
    void testAxpy() {
        try (CudaTensor x = CudaTensor.fromFloat(device, new float[]{1, 2, 3, 4}, Precision.FP32);
             CudaTensor y = CudaTensor.fromFloat(device, new float[]{10, 20, 30, 40}, Precision.FP32)) {

            var status = CudaOps.axpy(device, 2.0, x, y);
            assertEquals(CudaOps.Status.SUCCESS, status);

            float[] result = y.toFloatArray();
            assertArrayEquals(new float[]{12, 24, 36, 48}, result, 1e-5f);
        }
    }

    @Test
    @DisplayName("Clip: y = clip(x, min, max)")
    void testClip() {
        try (CudaTensor x = CudaTensor.fromFloat(device, new float[]{-5, -1, 0, 1, 5}, Precision.FP32);
             CudaTensor y = CudaTensor.allocate(device, 5, Precision.FP32)) {

            var status = CudaOps.clip(device, x, y, -2.0, 2.0);
            assertEquals(CudaOps.Status.SUCCESS, status);

            float[] result = y.toFloatArray();
            assertArrayEquals(new float[]{-2, -1, 0, 1, 2}, result, 1e-5f);
        }
    }

    // ========================================================================
    // Error Handling Tests
    // ========================================================================

    @Test
    @DisplayName("Size mismatch returns error")
    void testSizeMismatch() {
        try (CudaTensor a = CudaTensor.allocate(device, 4, Precision.FP32);
             CudaTensor b = CudaTensor.allocate(device, 5, Precision.FP32);
             CudaTensor c = CudaTensor.allocate(device, 4, Precision.FP32)) {

            var status = CudaOps.add(device, a, b, c);
            assertEquals(CudaOps.Status.SIZE_MISMATCH, status);
        }
    }

    @Test
    @DisplayName("Precision mismatch returns error")
    void testPrecisionMismatch() {
        try (CudaTensor a = CudaTensor.allocate(device, 4, Precision.FP32);
             CudaTensor b = CudaTensor.allocate(device, 4, Precision.FP16);
             CudaTensor c = CudaTensor.allocate(device, 4, Precision.FP32)) {

            var status = CudaOps.add(device, a, b, c);
            assertEquals(CudaOps.Status.PRECISION_MISMATCH, status);
        }
    }
}