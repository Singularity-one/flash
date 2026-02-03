package com.cuda.poc;
import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

class CudaGemmTest {

    private static CudaGemm gemm;

    @BeforeAll
    static void setUp() {
        gemm = new CudaGemm();
        System.out.println("設備: " + gemm.getDeviceName());
    }

    @AfterAll
    static void tearDown() {
        if (gemm != null) {
            gemm.close();
        }
    }

    @Test
    @DisplayName("測試 2×2 矩陣乘法 - Row-Major")
    void testSmallGemm() {
        // Row-major 輸入
        // A = [1 2]    B = [5 6]
        //     [3 4]        [7 8]
        float[] a = {1, 2, 3, 4};  // row-major
        float[] b = {5, 6, 7, 8};  // row-major
        float[] c = new float[4];

        CudaGemm.Status status = gemm.multiply(2, 2, 2, a, b, c);

        assertEquals(CudaGemm.Status.SUCCESS, status);

        // C = A×B = [19 22]  →  {19, 22, 43, 50}
        //           [43 50]
        float[] expected = {19, 22, 43, 50};
        assertArrayEquals(expected, c, 1e-5f);
    }

    @Test
    @DisplayName("測試 alpha 和 beta 參數")
    void testAlphaBeta() {
        // A = [1 2]    B = [5 6]
        //     [3 4]        [7 8]
        float[] a = {1, 2, 3, 4};
        float[] b = {5, 6, 7, 8};
        float[] c = {1, 1, 1, 1};

        // C = 2*A*B + 3*C
        CudaGemm.Status status = gemm.sgemm(2, 2, 2, 2.0f, a, b, 3.0f, c);

        assertEquals(CudaGemm.Status.SUCCESS, status);

        // A×B = [19 22]
        //       [43 50]
        // 2*A×B = [38 44]
        //         [86 100]
        // 3*C = [3 3]
        //       [3 3]
        // 結果 = [41 47]  →  {41, 47, 89, 103}
        //        [89 103]
        float[] expected = {41, 47, 89, 103};
        assertArrayEquals(expected, c, 1e-4f);
    }

    @Test
    @DisplayName("測試非方陣 - 3×2 × 2×4 = 3×4")
    void testNonSquare() {
        // A: 3×2    B: 2×4    C: 3×4
        // A = [1 2]    B = [1 2 3 4]
        //     [3 4]        [5 6 7 8]
        //     [5 6]
        float[] a = {
                1, 2,  // row 0
                3, 4,  // row 1
                5, 6   // row 2
        };

        float[] b = {
                1, 2, 3, 4,  // row 0
                5, 6, 7, 8   // row 1
        };

        float[] c = new float[12];  // 3×4

        CudaGemm.Status status = gemm.multiply(3, 4, 2, a, b, c);

        assertEquals(CudaGemm.Status.SUCCESS, status);

        // 手動計算：
        // Row 0: [1,2] · [[1,2,3,4], [5,6,7,8]]^T
        //      = [1*1+2*5, 1*2+2*6, 1*3+2*7, 1*4+2*8]
        //      = [11, 14, 17, 20]
        // Row 1: [3,4] · ... = [23, 30, 37, 44]
        // Row 2: [5,6] · ... = [35, 46, 57, 68]
        float[] expected = {
                11, 14, 17, 20,  // row 0
                23, 30, 37, 44,  // row 1
                35, 46, 57, 68   // row 2
        };

        assertArrayEquals(expected, c, 1e-4f);
    }

    @Test
    @DisplayName("測試單位矩陣性質 I×A = A")
    void testIdentityMatrix() {
        // I: 3×3    A: 3×2
        float[] identity = {
                1, 0, 0,
                0, 1, 0,
                0, 0, 1
        };

        float[] a = {
                2, 3,
                4, 5,
                6, 7
        };

        float[] c = new float[6];  // 3×2

        CudaGemm.Status status = gemm.multiply(3, 2, 3, identity, a, c);

        assertEquals(CudaGemm.Status.SUCCESS, status);
        assertArrayEquals(a, c, 1e-5f);  // I×A 應該等於 A
    }
}
