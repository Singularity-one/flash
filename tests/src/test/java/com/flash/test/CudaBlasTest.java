package com.flash.test;

import com.flash.CudaDevice;
import com.flash.CudaBlas;
import com.flash.CudaTensor;
import com.flash.Precision;
import com.flash.exception.PrecisionMismatchException;
import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for CudaBlas.
 */
@DisplayName("CudaBlas API")
class CudaBlasTest {

    private static CudaDevice device;
    private static CudaBlas blas;

    @BeforeAll
    static void setUp() {
        device = new CudaDevice(0);
        blas = new CudaBlas(device);
        System.out.println("  Device: " + device.getName());
    }

    @AfterAll
    static void tearDown() {
        if (blas != null) {
            blas.close();
        }
        if (device != null) {
            device.close();
        }
    }

    @Test
    @DisplayName("cuBLAS initialization works")
    void testBlasInit() {
        assertNotNull(blas);
        assertFalse(blas.isClosed());
        assertEquals(device, blas.getDevice());
    }

    // ========================================================================
    // Legacy API Tests (raw device pointers)
    // ========================================================================

    @Test
    @DisplayName("SGEMM: 2×2 matrix multiplication (legacy API)")
    void testSgemm2x2() {
        float[] a = {1, 2, 3, 4};
        float[] b = {5, 6, 7, 8};
        float[] c = new float[4];

        long a_ptr = device.allocate(a.length * Float.BYTES);
        long b_ptr = device.allocate(b.length * Float.BYTES);
        long c_ptr = device.allocate(c.length * Float.BYTES);

        device.copyHostToDevice(a, a_ptr);
        device.copyHostToDevice(b, b_ptr);
        device.copyHostToDevice(c, c_ptr);

        var status = blas.sgemm(2, 2, 2, 1.0f, a_ptr, b_ptr, 0.0f, c_ptr);
        assertEquals(CudaBlas.Status.SUCCESS, status);

        device.copyDeviceToHost(c_ptr, c);
        assertArrayEquals(new float[]{19, 22, 43, 50}, c, 1e-5f);

        device.free(a_ptr);
        device.free(b_ptr);
        device.free(c_ptr);
    }

    @Test
    @DisplayName("SGEMM: alpha and beta parameters")
    void testSgemmAlphaBeta() {
        float[] a = {1, 2, 3, 4};
        float[] b = {5, 6, 7, 8};
        float[] c = {1, 1, 1, 1};

        long a_ptr = device.allocate(a.length * Float.BYTES);
        long b_ptr = device.allocate(b.length * Float.BYTES);
        long c_ptr = device.allocate(c.length * Float.BYTES);

        device.copyHostToDevice(a, a_ptr);
        device.copyHostToDevice(b, b_ptr);
        device.copyHostToDevice(c, c_ptr);

        // C = 2*(A*B) + 3*C
        var status = blas.sgemm(2, 2, 2, 2.0f, a_ptr, b_ptr, 3.0f, c_ptr);
        assertEquals(CudaBlas.Status.SUCCESS, status);

        device.copyDeviceToHost(c_ptr, c);
        assertArrayEquals(new float[]{41, 47, 89, 103}, c, 1e-4f);

        device.free(a_ptr);
        device.free(b_ptr);
        device.free(c_ptr);
    }

    @Test
    @DisplayName("Non-square matrix (3×2 × 2×4 = 3×4)")
    void testNonSquare() {
        float[] a = {1, 2, 3, 4, 5, 6};
        float[] b = {1, 2, 3, 4, 5, 6, 7, 8};
        float[] c = new float[12];

        long a_ptr = device.allocate(a.length * Float.BYTES);
        long b_ptr = device.allocate(b.length * Float.BYTES);
        long c_ptr = device.allocate(c.length * Float.BYTES);

        device.copyHostToDevice(a, a_ptr);
        device.copyHostToDevice(b, b_ptr);
        device.copyHostToDevice(c, c_ptr);

        var status = blas.multiply(3, 4, 2, a_ptr, b_ptr, c_ptr);
        assertEquals(CudaBlas.Status.SUCCESS, status);

        device.copyDeviceToHost(c_ptr, c);
        assertArrayEquals(new float[]{11, 14, 17, 20, 23, 30, 37, 44, 35, 46, 57, 68}, c, 1e-5f);

        device.free(a_ptr);
        device.free(b_ptr);
        device.free(c_ptr);
    }

    @Test
    @DisplayName("Operations on closed context throw exception")
    void testClosedContextThrows() {
        CudaBlas tempBlas = new CudaBlas(device);
        tempBlas.close();

        assertTrue(tempBlas.isClosed());

        assertThrows(IllegalStateException.class, () -> {
            tempBlas.sgemm(2, 2, 2, 1.0f, 0, 0, 0.0f, 0);
        });
    }

    // ========================================================================
    // Phase 1.2: Unified GEMM Tests (CudaTensor)
    // ========================================================================

    @Test
    @DisplayName("Unified GEMM: FP32 with CudaTensor")
    void testUnifiedGemmFP32() {
        float[] a_data = {1, 2, 3, 4};
        float[] b_data = {5, 6, 7, 8};

        try (CudaTensor a = CudaTensor.fromFloat(device, a_data, Precision.FP32);
             CudaTensor b = CudaTensor.fromFloat(device, b_data, Precision.FP32);
             CudaTensor c = CudaTensor.allocate(device, 4, Precision.FP32)) {

            var status = blas.gemm(2, 2, 2, 1.0, a, b, 0.0, c);
            assertEquals(CudaBlas.Status.SUCCESS, status);

            float[] result = c.toFloatArray();
            assertArrayEquals(new float[]{19, 22, 43, 50}, result, 1e-5f);
        }
    }

    @Test
    @DisplayName("Unified GEMM: FP16 with CudaTensor")
    void testUnifiedGemmFP16() {
        float[] a_data = {1, 2, 3, 4};
        float[] b_data = {5, 6, 7, 8};

        try (CudaTensor a = CudaTensor.fromFloat(device, a_data, Precision.FP16);
             CudaTensor b = CudaTensor.fromFloat(device, b_data, Precision.FP16);
             CudaTensor c = CudaTensor.allocate(device, 4, Precision.FP16)) {

            var status = blas.gemm(2, 2, 2, 1.0, a, b, 0.0, c);
            assertEquals(CudaBlas.Status.SUCCESS, status);

            float[] result = c.toFloatArray();
            // FP16 精度較低，使用較大的容差
            assertArrayEquals(new float[]{19, 22, 43, 50}, result, 0.1f);
        }
    }

    @Test
    @DisplayName("Unified GEMM: FP64 with CudaTensor")
    void testUnifiedGemmFP64() {
        double[] a_data = {1, 2, 3, 4};
        double[] b_data = {5, 6, 7, 8};

        try (CudaTensor a = CudaTensor.fromDouble(device, a_data, Precision.FP64);
             CudaTensor b = CudaTensor.fromDouble(device, b_data, Precision.FP64);
             CudaTensor c = CudaTensor.allocate(device, 4, Precision.FP64)) {

            var status = blas.gemm(2, 2, 2, 1.0, a, b, 0.0, c);
            assertEquals(CudaBlas.Status.SUCCESS, status);

            double[] result = c.toDoubleArray();
            assertArrayEquals(new double[]{19, 22, 43, 50}, result, 1e-10);
        }
    }

    @Test
    @DisplayName("Unified GEMM: multiply convenience method")
    void testUnifiedMultiply() {
        float[] a_data = {1, 0, 0, 1};  // Identity
        float[] b_data = {5, 6, 7, 8};

        try (CudaTensor a = CudaTensor.fromFloat(device, a_data, Precision.FP32);
             CudaTensor b = CudaTensor.fromFloat(device, b_data, Precision.FP32);
             CudaTensor c = CudaTensor.allocate(device, 4, Precision.FP32)) {

            var status = blas.multiply(2, 2, 2, a, b, c);
            assertEquals(CudaBlas.Status.SUCCESS, status);

            float[] result = c.toFloatArray();
            assertArrayEquals(b_data, result, 1e-5f);
        }
    }

    @Test
    @DisplayName("Unified GEMM: Precision mismatch throws exception")
    void testUnifiedGemmPrecisionMismatch() {
        try (CudaTensor a = CudaTensor.allocate(device, 4, Precision.FP32);
             CudaTensor b = CudaTensor.allocate(device, 4, Precision.FP16);
             CudaTensor c = CudaTensor.allocate(device, 4, Precision.FP32)) {

            assertThrows(PrecisionMismatchException.class, () -> {
                blas.gemm(2, 2, 2, 1.0, a, b, 0.0, c);
            });
        }
    }

    @Test
    @DisplayName("Unified GEMM: alpha and beta with FP32")
    void testUnifiedGemmAlphaBeta() {
        float[] a_data = {1, 2, 3, 4};
        float[] b_data = {5, 6, 7, 8};
        float[] c_data = {1, 1, 1, 1};

        try (CudaTensor a = CudaTensor.fromFloat(device, a_data, Precision.FP32);
             CudaTensor b = CudaTensor.fromFloat(device, b_data, Precision.FP32);
             CudaTensor c = CudaTensor.fromFloat(device, c_data, Precision.FP32)) {

            // C = 2*(A*B) + 3*C
            var status = blas.gemm(2, 2, 2, 2.0, a, b, 3.0, c);
            assertEquals(CudaBlas.Status.SUCCESS, status);

            float[] result = c.toFloatArray();
            assertArrayEquals(new float[]{41, 47, 89, 103}, result, 1e-4f);
        }
    }

    @Test
    @DisplayName("Unified GEMM: Non-square with CudaTensor")
    void testUnifiedGemmNonSquare() {
        float[] a_data = {1, 2, 3, 4, 5, 6};  // 3×2
        float[] b_data = {1, 2, 3, 4, 5, 6, 7, 8};  // 2×4

        try (CudaTensor a = CudaTensor.fromFloat(device, a_data, Precision.FP32);
             CudaTensor b = CudaTensor.fromFloat(device, b_data, Precision.FP32);
             CudaTensor c = CudaTensor.allocate(device, 12, Precision.FP32)) {  // 3×4

            var status = blas.gemm(3, 4, 2, 1.0, a, b, 0.0, c);
            assertEquals(CudaBlas.Status.SUCCESS, status);

            float[] result = c.toFloatArray();
            assertArrayEquals(new float[]{11, 14, 17, 20, 23, 30, 37, 44, 35, 46, 57, 68}, result, 1e-5f);
        }
    }
}