package com.flash.test;

import com.flash.CudaDevice;
import com.flash.CudaRand;
import com.flash.CudaTensor;
import com.flash.Precision;
import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for CudaRand.
 */
@DisplayName("CudaRand API")
class CudaRandTest {

    private static CudaDevice device;
    private static CudaRand rand;

    @BeforeAll
    static void setUp() {
        device = new CudaDevice(0);
        rand = new CudaRand(device, 12345);
        System.out.println("  Device: " + device.getName());
    }

    @AfterAll
    static void tearDown() {
        if (rand != null) {
            rand.close();
        }
        if (device != null) {
            device.close();
        }
    }

    @Test
    @DisplayName("cuRAND initialization works")
    void testRandInit() {
        assertNotNull(rand);
        assertFalse(rand.isClosed());
        assertEquals(device, rand.getDevice());
    }

    // ========================================================================
    // Uniform Distribution Tests
    // ========================================================================

    @Test
    @DisplayName("Uniform: FP32 values in [0, 1)")
    void testUniformFP32() {
        try (CudaTensor tensor = CudaTensor.allocate(device, 1000, Precision.FP32)) {
            var status = rand.uniform(tensor);
            assertEquals(CudaRand.Status.SUCCESS, status);

            float[] result = tensor.toFloatArray();

            // Verify all values in [0, 1)
            for (float v : result) {
                assertTrue(v >= 0.0f && v < 1.0f, "Value out of range: " + v);
            }

            // Verify mean is approximately 0.5
            float sum = 0;
            for (float v : result) {
                sum += v;
            }
            float mean = sum / result.length;
            assertTrue(mean > 0.4f && mean < 0.6f, "Mean " + mean + " not close to 0.5");

            System.out.println("  Uniform FP32 mean: " + mean);
        }
    }

    @Test
    @DisplayName("Uniform: FP16 values in [0, 1)")
    void testUniformFP16() {
        try (CudaTensor tensor = CudaTensor.allocate(device, 1000, Precision.FP16)) {
            var status = rand.uniform(tensor);
            assertEquals(CudaRand.Status.SUCCESS, status);

            float[] result = tensor.toFloatArray();

            float sum = 0;
            for (float v : result) {
                sum += v;
            }
            float mean = sum / result.length;
            assertTrue(mean > 0.4f && mean < 0.6f, "Mean " + mean + " not close to 0.5");

            System.out.println("  Uniform FP16 mean: " + mean);
        }
    }

    @Test
    @DisplayName("Uniform: custom range [-5, 5)")
    void testUniformRange() {
        try (CudaTensor tensor = CudaTensor.allocate(device, 1000, Precision.FP32)) {
            var status = rand.uniform(tensor, -5.0, 5.0);
            assertEquals(CudaRand.Status.SUCCESS, status);

            float[] result = tensor.toFloatArray();

            // Verify all values in [-5, 5)
            for (float v : result) {
                assertTrue(v >= -5.0f && v < 5.0f, "Value out of range: " + v);
            }

            // Verify mean is approximately 0
            float sum = 0;
            for (float v : result) {
                sum += v;
            }
            float mean = sum / result.length;
            assertTrue(mean > -1.0f && mean < 1.0f, "Mean " + mean + " not close to 0");

            System.out.println("  Uniform range mean: " + mean);
        }
    }

    // ========================================================================
    // Normal Distribution Tests
    // ========================================================================

    @Test
    @DisplayName("Normal: standard distribution (mean=0, stddev=1)")
    void testNormalStandard() {
        try (CudaTensor tensor = CudaTensor.allocate(device, 10000, Precision.FP32)) {
            var status = rand.normal(tensor);
            assertEquals(CudaRand.Status.SUCCESS, status);

            float[] result = tensor.toFloatArray();

            // Compute sample mean
            float sum = 0;
            for (float v : result) {
                sum += v;
            }
            float mean = sum / result.length;
            assertTrue(Math.abs(mean) < 0.1f, "Mean " + mean + " not close to 0");

            // Compute sample stddev
            float variance = 0;
            for (float v : result) {
                variance += (v - mean) * (v - mean);
            }
            float stddev = (float) Math.sqrt(variance / result.length);
            assertTrue(Math.abs(stddev - 1.0f) < 0.1f, "Stddev " + stddev + " not close to 1");

            System.out.println("  Normal mean: " + mean + ", stddev: " + stddev);
        }
    }

    @Test
    @DisplayName("Normal: custom distribution (mean=5, stddev=2)")
    void testNormalCustom() {
        try (CudaTensor tensor = CudaTensor.allocate(device, 10000, Precision.FP32)) {
            var status = rand.normal(tensor, 5.0, 2.0);
            assertEquals(CudaRand.Status.SUCCESS, status);

            float[] result = tensor.toFloatArray();

            // Compute sample mean
            float sum = 0;
            for (float v : result) {
                sum += v;
            }
            float mean = sum / result.length;
            assertTrue(Math.abs(mean - 5.0f) < 0.2f, "Mean " + mean + " not close to 5");

            // Compute sample stddev
            float variance = 0;
            for (float v : result) {
                variance += (v - mean) * (v - mean);
            }
            float stddev = (float) Math.sqrt(variance / result.length);
            assertTrue(Math.abs(stddev - 2.0f) < 0.2f, "Stddev " + stddev + " not close to 2");

            System.out.println("  Normal (5, 2) mean: " + mean + ", stddev: " + stddev);
        }
    }

    @Test
    @DisplayName("Normal: FP16 distribution")
    void testNormalFP16() {
        try (CudaTensor tensor = CudaTensor.allocate(device, 10000, Precision.FP16)) {
            var status = rand.normal(tensor, 0.0, 1.0);
            assertEquals(CudaRand.Status.SUCCESS, status);

            float[] result = tensor.toFloatArray();

            float sum = 0;
            for (float v : result) {
                sum += v;
            }
            float mean = sum / result.length;
            assertTrue(Math.abs(mean) < 0.2f, "Mean " + mean + " not close to 0");

            System.out.println("  Normal FP16 mean: " + mean);
        }
    }

    // ========================================================================
    // Weight Initialization Tests
    // ========================================================================

    @Test
    @DisplayName("Xavier uniform initialization")
    void testXavierUniform() {
        int fanIn = 768;
        int fanOut = 768;
        double expectedLimit = Math.sqrt(6.0 / (fanIn + fanOut));

        try (CudaTensor weights = CudaTensor.allocate(device, fanIn * fanOut, Precision.FP32)) {
            var status = rand.xavierUniform(weights, fanIn, fanOut);
            assertEquals(CudaRand.Status.SUCCESS, status);

            float[] result = weights.toFloatArray();

            // Check all values are within expected range
            for (float v : result) {
                assertTrue(Math.abs(v) <= expectedLimit * 1.1f, "Value " + v + " exceeds limit");
            }

            // Check mean is approximately 0
            float sum = 0;
            for (float v : result) {
                sum += v;
            }
            float mean = sum / result.length;
            assertTrue(Math.abs(mean) < 0.01f, "Mean " + mean + " not close to 0");

            System.out.println("  Xavier uniform: limit=" + expectedLimit + ", mean=" + mean);
        }
    }

    @Test
    @DisplayName("Kaiming/He normal initialization")
    void testKaimingNormal() {
        int fanIn = 768;
        double expectedStddev = Math.sqrt(2.0 / fanIn);

        try (CudaTensor weights = CudaTensor.allocate(device, fanIn * 768, Precision.FP32)) {
            var status = rand.kaimingNormal(weights, fanIn);
            assertEquals(CudaRand.Status.SUCCESS, status);

            float[] result = weights.toFloatArray();

            // Compute sample stddev
            float sum = 0;
            for (float v : result) {
                sum += v;
            }
            float mean = sum / result.length;

            float variance = 0;
            for (float v : result) {
                variance += (v - mean) * (v - mean);
            }
            float stddev = (float) Math.sqrt(variance / result.length);

            assertTrue(Math.abs(stddev - expectedStddev) < 0.01f,
                    "Stddev " + stddev + " not close to expected " + expectedStddev);

            System.out.println("  Kaiming normal: expected_stddev=" + expectedStddev + ", actual_stddev=" + stddev);
        }
    }

    // ========================================================================
    // Seed Tests
    // ========================================================================

    @Test
    @DisplayName("Seed reproducibility")
    void testSeedReproducibility() {
        try (CudaTensor tensor1 = CudaTensor.allocate(device, 100, Precision.FP32);
             CudaTensor tensor2 = CudaTensor.allocate(device, 100, Precision.FP32)) {

            // Generate with seed 42
            rand.setSeed(42);
            rand.uniform(tensor1);
            float[] result1 = tensor1.toFloatArray();

            // Generate again with same seed
            rand.setSeed(42);
            rand.uniform(tensor2);
            float[] result2 = tensor2.toFloatArray();

            // Should be identical
            assertArrayEquals(result1, result2, 1e-6f, "Results should be identical with same seed");

            System.out.println("  Seed reproducibility: verified");
        }
    }

    @Test
    @DisplayName("Different seeds produce different results")
    void testDifferentSeeds() {
        try (CudaTensor tensor1 = CudaTensor.allocate(device, 100, Precision.FP32);
             CudaTensor tensor2 = CudaTensor.allocate(device, 100, Precision.FP32)) {

            rand.setSeed(42);
            rand.uniform(tensor1);
            float[] result1 = tensor1.toFloatArray();

            rand.setSeed(123);
            rand.uniform(tensor2);
            float[] result2 = tensor2.toFloatArray();

            // Should be different
            boolean allSame = true;
            for (int i = 0; i < result1.length; i++) {
                if (Math.abs(result1[i] - result2[i]) > 1e-6f) {
                    allSame = false;
                    break;
                }
            }
            assertFalse(allSame, "Different seeds should produce different results");

            System.out.println("  Different seeds: verified");
        }
    }

    // ========================================================================
    // Error Handling Tests
    // ========================================================================

    @Test
    @DisplayName("Invalid stddev throws exception")
    void testInvalidStddev() {
        try (CudaTensor tensor = CudaTensor.allocate(device, 100, Precision.FP32)) {
            assertThrows(IllegalArgumentException.class, () -> {
                rand.normal(tensor, 0.0, -1.0);
            });

            assertThrows(IllegalArgumentException.class, () -> {
                rand.normal(tensor, 0.0, 0.0);
            });
        }
    }

    @Test
    @DisplayName("Invalid range throws exception")
    void testInvalidRange() {
        try (CudaTensor tensor = CudaTensor.allocate(device, 100, Precision.FP32)) {
            assertThrows(IllegalArgumentException.class, () -> {
                rand.uniform(tensor, 5.0, 3.0);  // low > high
            });

            assertThrows(IllegalArgumentException.class, () -> {
                rand.uniform(tensor, 5.0, 5.0);  // low == high
            });
        }
    }

    @Test
    @DisplayName("Operations on closed context throw exception")
    void testClosedContextThrows() {
        CudaRand tempRand = new CudaRand(device, 0);
        tempRand.close();

        assertTrue(tempRand.isClosed());

        try (CudaTensor tensor = CudaTensor.allocate(device, 100, Precision.FP32)) {
            assertThrows(IllegalStateException.class, () -> {
                tempRand.uniform(tensor);
            });
        }
    }
}