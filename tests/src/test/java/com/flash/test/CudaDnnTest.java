package com.flash.test;

import com.flash.CudaDevice;
import com.flash.CudaDnn;
import com.flash.CudaTensor;
import com.flash.Precision;
import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for CudaDnn.
 */
@DisplayName("CudaDnn API")
class CudaDnnTest {

    private static CudaDevice device;
    private static CudaDnn dnn;

    @BeforeAll
    static void setUp() {
        device = new CudaDevice(0);
        dnn = new CudaDnn(device);
        System.out.println("  Device: " + device.getName());
    }

    @AfterAll
    static void tearDown() {
        if (dnn != null) {
            dnn.close();
        }
        if (device != null) {
            device.close();
        }
    }

    @Test
    @DisplayName("cuDNN initialization works")
    void testDnnInit() {
        assertNotNull(dnn);
        assertFalse(dnn.isClosed());
        assertEquals(device, dnn.getDevice());
    }

    // ========================================================================
    // Softmax Tests
    // ========================================================================

    @Test
    @DisplayName("Softmax: FP32 basic test")
    void testSoftmaxFP32() {
        float[] input = {1.0f, 2.0f, 3.0f, 4.0f};

        try (CudaTensor x = CudaTensor.fromFloat(device, input, Precision.FP32);
             CudaTensor y = CudaTensor.allocate(device, 4, Precision.FP32)) {

            // Apply softmax: shape [1, 4, 1, 1] (N=1, C=4, H=1, W=1)
            var status = dnn.softmax(1, 4, 1, 1, x, y);
            assertEquals(CudaDnn.Status.SUCCESS, status);

            float[] result = y.toFloatArray();

            // Verify sum = 1.0
            float sum = 0;
            for (float v : result) {
                sum += v;
            }
            assertEquals(1.0f, sum, 1e-5f, "Softmax sum should be 1.0");

            // Verify monotonically increasing (since input is increasing)
            for (int i = 1; i < result.length; i++) {
                assertTrue(result[i] > result[i-1], "Softmax should preserve order");
            }

            System.out.println("  Input:  " + java.util.Arrays.toString(input));
            System.out.println("  Output: " + java.util.Arrays.toString(result));
        }
    }

    @Test
    @DisplayName("Softmax: FP16 basic test")
    void testSoftmaxFP16() {
        float[] input = {1.0f, 2.0f, 3.0f, 4.0f};

        try (CudaTensor x = CudaTensor.fromFloat(device, input, Precision.FP16);
             CudaTensor y = CudaTensor.allocate(device, 4, Precision.FP16)) {

            var status = dnn.softmax(1, 4, 1, 1, x, y);
            assertEquals(CudaDnn.Status.SUCCESS, status);

            float[] result = y.toFloatArray();

            float sum = 0;
            for (float v : result) {
                sum += v;
            }
            assertEquals(1.0f, sum, 0.01f, "Softmax sum should be ~1.0 (FP16 precision)");
        }
    }

    @Test
    @DisplayName("Softmax: batch test")
    void testSoftmaxBatch() {
        // 2 batches, 3 channels each
        float[] input = {
                1.0f, 2.0f, 3.0f,  // batch 0
                4.0f, 5.0f, 6.0f   // batch 1
        };

        try (CudaTensor x = CudaTensor.fromFloat(device, input, Precision.FP32);
             CudaTensor y = CudaTensor.allocate(device, 6, Precision.FP32)) {

            var status = dnn.softmax(2, 3, 1, 1, x, y);
            assertEquals(CudaDnn.Status.SUCCESS, status);

            float[] result = y.toFloatArray();

            // Verify each batch sums to 1.0
            float sum0 = result[0] + result[1] + result[2];
            float sum1 = result[3] + result[4] + result[5];

            assertEquals(1.0f, sum0, 1e-5f, "Batch 0 sum should be 1.0");
            assertEquals(1.0f, sum1, 1e-5f, "Batch 1 sum should be 1.0");
        }
    }

    // ========================================================================
    // Activation Tests
    // ========================================================================

    @Test
    @DisplayName("Activation: ReLU")
    void testRelu() {
        float[] input = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};

        try (CudaTensor x = CudaTensor.fromFloat(device, input, Precision.FP32);
             CudaTensor y = CudaTensor.allocate(device, 5, Precision.FP32)) {

            var status = dnn.relu(x, y);
            assertEquals(CudaDnn.Status.SUCCESS, status);

            float[] result = y.toFloatArray();
            float[] expected = {0.0f, 0.0f, 0.0f, 1.0f, 2.0f};

            assertArrayEquals(expected, result, 1e-5f);
        }
    }

    @Test
    @DisplayName("Activation: GELU")
    void testGelu() {
        float[] input = {-1.0f, 0.0f, 1.0f, 2.0f};

        try (CudaTensor x = CudaTensor.fromFloat(device, input, Precision.FP32);
             CudaTensor y = CudaTensor.allocate(device, 4, Precision.FP32)) {

            var status = dnn.gelu(x, y);
            assertEquals(CudaDnn.Status.SUCCESS, status);

            float[] result = y.toFloatArray();

            // GELU(-1) ≈ -0.159, GELU(0) = 0, GELU(1) ≈ 0.841, GELU(2) ≈ 1.955
            assertTrue(result[0] < 0, "GELU(-1) should be negative");
            assertEquals(0.0f, result[1], 1e-5f, "GELU(0) should be 0");
            assertEquals(0.841f, result[2], 0.01f, "GELU(1) should be ~0.841");
            assertEquals(1.955f, result[3], 0.01f, "GELU(2) should be ~1.955");

            System.out.println("  GELU output: " + java.util.Arrays.toString(result));
        }
    }

    @Test
    @DisplayName("Activation: Sigmoid")
    void testSigmoid() {
        float[] input = {-2.0f, 0.0f, 2.0f};

        try (CudaTensor x = CudaTensor.fromFloat(device, input, Precision.FP32);
             CudaTensor y = CudaTensor.allocate(device, 3, Precision.FP32)) {

            var status = dnn.sigmoid(x, y);
            assertEquals(CudaDnn.Status.SUCCESS, status);

            float[] result = y.toFloatArray();

            // sigmoid(-2) ≈ 0.119, sigmoid(0) = 0.5, sigmoid(2) ≈ 0.881
            assertEquals(0.119f, result[0], 0.01f);
            assertEquals(0.5f, result[1], 1e-5f);
            assertEquals(0.881f, result[2], 0.01f);
        }
    }

    @Test
    @DisplayName("Activation: Tanh")
    void testTanh() {
        float[] input = {-1.0f, 0.0f, 1.0f};

        try (CudaTensor x = CudaTensor.fromFloat(device, input, Precision.FP32);
             CudaTensor y = CudaTensor.allocate(device, 3, Precision.FP32)) {

            var status = dnn.tanh(x, y);
            assertEquals(CudaDnn.Status.SUCCESS, status);

            float[] result = y.toFloatArray();

            // tanh(-1) ≈ -0.762, tanh(0) = 0, tanh(1) ≈ 0.762
            assertEquals(-0.762f, result[0], 0.01f);
            assertEquals(0.0f, result[1], 1e-5f);
            assertEquals(0.762f, result[2], 0.01f);
        }
    }

    @Test
    @DisplayName("Activation: SiLU")
    void testSilu() {
        float[] input = {-1.0f, 0.0f, 1.0f, 2.0f};

        try (CudaTensor x = CudaTensor.fromFloat(device, input, Precision.FP32);
             CudaTensor y = CudaTensor.allocate(device, 4, Precision.FP32)) {

            var status = dnn.silu(x, y);
            assertEquals(CudaDnn.Status.SUCCESS, status);

            float[] result = y.toFloatArray();

            // SiLU(-1) ≈ -0.269, SiLU(0) = 0, SiLU(1) ≈ 0.731, SiLU(2) ≈ 1.762
            assertEquals(-0.269f, result[0], 0.01f);
            assertEquals(0.0f, result[1], 1e-5f);
            assertEquals(0.731f, result[2], 0.01f);
            assertEquals(1.762f, result[3], 0.01f);

            System.out.println("  SiLU output: " + java.util.Arrays.toString(result));
        }
    }

    @Test
    @DisplayName("Activation: FP16 GELU")
    void testGeluFP16() {
        float[] input = {-1.0f, 0.0f, 1.0f, 2.0f};

        try (CudaTensor x = CudaTensor.fromFloat(device, input, Precision.FP16);
             CudaTensor y = CudaTensor.allocate(device, 4, Precision.FP16)) {

            var status = dnn.gelu(x, y);
            assertEquals(CudaDnn.Status.SUCCESS, status);

            float[] result = y.toFloatArray();

            // FP16 has lower precision
            assertTrue(result[0] < 0, "GELU(-1) should be negative");
            assertEquals(0.0f, result[1], 0.05f, "GELU(0) should be ~0");
            assertEquals(0.841f, result[2], 0.05f, "GELU(1) should be ~0.841");
        }
    }

    // ========================================================================
    // Activation Backward Tests
    // ========================================================================

    @Test
    @DisplayName("Activation Backward: ReLU")
    void testReluBackward() {
        float[] xData = {-1.0f, 0.0f, 1.0f, 2.0f};
        float[] dyData = {1.0f, 1.0f, 1.0f, 1.0f};  // Upstream gradient

        try (CudaTensor x = CudaTensor.fromFloat(device, xData, Precision.FP32);
             CudaTensor y = CudaTensor.allocate(device, 4, Precision.FP32);
             CudaTensor dy = CudaTensor.fromFloat(device, dyData, Precision.FP32);
             CudaTensor dx = CudaTensor.allocate(device, 4, Precision.FP32)) {

            var status = dnn.activationBackward(CudaDnn.ActivationType.RELU, x, y, dy, dx);
            assertEquals(CudaDnn.Status.SUCCESS, status);

            float[] result = dx.toFloatArray();

            // ReLU gradient: 0 for x <= 0, 1 for x > 0
            float[] expected = {0.0f, 0.0f, 1.0f, 1.0f};
            assertArrayEquals(expected, result, 1e-5f);
        }
    }

    @Test
    @DisplayName("Activation Backward: GELU")
    void testGeluBackward() {
        float[] xData = {0.0f, 1.0f};
        float[] dyData = {1.0f, 1.0f};

        try (CudaTensor x = CudaTensor.fromFloat(device, xData, Precision.FP32);
             CudaTensor y = CudaTensor.allocate(device, 2, Precision.FP32);
             CudaTensor dy = CudaTensor.fromFloat(device, dyData, Precision.FP32);
             CudaTensor dx = CudaTensor.allocate(device, 2, Precision.FP32)) {

            var status = dnn.activationBackward(CudaDnn.ActivationType.GELU, x, y, dy, dx);
            assertEquals(CudaDnn.Status.SUCCESS, status);

            float[] result = dx.toFloatArray();

            // GELU'(0) = 0.5, GELU'(1) ≈ 1.08
            assertEquals(0.5f, result[0], 0.05f);
            assertTrue(result[1] > 1.0f && result[1] < 1.2f, "GELU'(1) should be ~1.08");

            System.out.println("  GELU backward: " + java.util.Arrays.toString(result));
        }
    }

    // ========================================================================
    // Error Handling Tests
    // ========================================================================

    @Test
    @DisplayName("Operations on closed context throw exception")
    void testClosedContextThrows() {
        CudaDnn tempDnn = new CudaDnn(device);
        tempDnn.close();

        assertTrue(tempDnn.isClosed());

        try (CudaTensor x = CudaTensor.allocate(device, 4, Precision.FP32);
             CudaTensor y = CudaTensor.allocate(device, 4, Precision.FP32)) {

            assertThrows(IllegalStateException.class, () -> {
                tempDnn.relu(x, y);
            });
        }
    }

    @Test
    @DisplayName("Size mismatch throws exception")
    void testSizeMismatch() {
        try (CudaTensor x = CudaTensor.allocate(device, 4, Precision.FP32);
             CudaTensor y = CudaTensor.allocate(device, 8, Precision.FP32)) {

            assertThrows(IllegalArgumentException.class, () -> {
                dnn.relu(x, y);
            });
        }
    }
}