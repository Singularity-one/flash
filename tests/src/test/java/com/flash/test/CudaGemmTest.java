package com.flash.test;

import com.flash.CudaGemm;
import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

/**
 * API tests for CudaGemm.
 */
@DisplayName("CudaGemm API")
class CudaGemmTest {

    private static CudaGemm gemm;

    @BeforeAll
    static void setUp() {
        gemm = new CudaGemm();
        System.out.println("  Device: " + gemm.getDeviceName());
    }

    @AfterAll
    static void tearDown() {
        if (gemm != null) {
            gemm.close();
        }
    }

    @Test
    @DisplayName("2×2 matrix multiplication")
    void test2x2Multiplication() {
        // A = [1 2]    B = [5 6]    C = [19 22]
        //     [3 4]        [7 8]        [43 50]
        float[] a = {1, 2, 3, 4};
        float[] b = {5, 6, 7, 8};
        float[] c = new float[4];

        var status = gemm.multiply(2, 2, 2, a, b, c);

        assertEquals(CudaGemm.Status.SUCCESS, status);
        assertArrayEquals(new float[]{19, 22, 43, 50}, c, 1e-5f);
    }

    @Test
    @DisplayName("Alpha and beta parameters")
    void testAlphaBeta() {
        float[] a = {1, 2, 3, 4};
        float[] b = {5, 6, 7, 8};
        float[] c = {1, 1, 1, 1};

        // C = 2*(A*B) + 3*C
        var status = gemm.sgemm(2, 2, 2, 2.0f, a, b, 3.0f, c);

        assertEquals(CudaGemm.Status.SUCCESS, status);
        assertArrayEquals(new float[]{41, 47, 89, 103}, c, 1e-4f);
    }

    @Test
    @DisplayName("Non-square matrices (3×2 × 2×4 = 3×4)")
    void testNonSquare() {
        // A: 3×2    B: 2×4    C: 3×4
        float[] a = {1, 2, 3, 4, 5, 6};
        float[] b = {1, 2, 3, 4, 5, 6, 7, 8};
        float[] c = new float[12];

        var status = gemm.multiply(3, 4, 2, a, b, c);

        assertEquals(CudaGemm.Status.SUCCESS, status);
        float[] expected = {11, 14, 17, 20, 23, 30, 37, 44, 35, 46, 57, 68};
        assertArrayEquals(expected, c, 1e-5f);
    }

    @Test
    @DisplayName("Large matrix (512×512)")
    void testLargeMatrix() {
        int n = 512;
        float[] a = new float[n * n];
        float[] b = new float[n * n];
        float[] c = new float[n * n];

        // Initialize with identity matrix
        for (int i = 0; i < n; i++) {
            a[i * n + i] = 1.0f;
            b[i * n + i] = 1.0f;
        }

        var status = gemm.multiply(n, n, n, a, b, c);

        assertEquals(CudaGemm.Status.SUCCESS, status);

        // C should also be identity
        for (int i = 0; i < n; i++) {
            assertEquals(1.0f, c[i * n + i], 1e-5f, "Diagonal element mismatch at " + i);
        }
    }

    @Test
    @DisplayName("Invalid dimensions throw exception")
    void testInvalidDimensions() {
        float[] a = {1, 2, 3, 4};
        float[] b = {5, 6, 7, 8};
        float[] c = new float[4];

        assertThrows(IllegalArgumentException.class, () -> {
            gemm.multiply(3, 2, 2, a, b, c);  // a.length should be 6
        });
    }
}