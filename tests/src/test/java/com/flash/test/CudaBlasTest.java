package com.flash.test;
import com.flash.CudaDevice;
import com.flash.CudaBlas;
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

    @Test
    @DisplayName("SGEMM: 2×2 matrix multiplication")
    void testSgemm2x2() {
        // A = [1 2]    B = [5 6]    Expected C = [19 22]
        //     [3 4]        [7 8]                  [43 50]
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

        // Expected: 2*[19,22,43,50] + 3*[1,1,1,1] = [41,47,89,103]
        assertArrayEquals(new float[]{41, 47, 89, 103}, c, 1e-4f);

        device.free(a_ptr);
        device.free(b_ptr);
        device.free(c_ptr);
    }

    @Test
    @DisplayName("SGEMM: multiply convenience method")
    void testMultiply() {
        float[] a = {1, 0, 0, 1};  // Identity matrix
        float[] b = {5, 6, 7, 8};
        float[] c = new float[4];

        long a_ptr = device.allocate(a.length * Float.BYTES);
        long b_ptr = device.allocate(b.length * Float.BYTES);
        long c_ptr = device.allocate(c.length * Float.BYTES);

        device.copyHostToDevice(a, a_ptr);
        device.copyHostToDevice(b, b_ptr);
        device.copyHostToDevice(c, c_ptr);

        var status = blas.multiply(2, 2, 2, a_ptr, b_ptr, c_ptr);
        assertEquals(CudaBlas.Status.SUCCESS, status);

        device.copyDeviceToHost(c_ptr, c);

        assertArrayEquals(b, c, 1e-5f);

        device.free(a_ptr);
        device.free(b_ptr);
        device.free(c_ptr);
    }

    @Test
    @DisplayName("Non-square matrix (3×2 × 2×4 = 3×4)")
    void testNonSquare() {
        // A: 3×2    B: 2×4    C: 3×4
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

        float[] expected = {11, 14, 17, 20, 23, 30, 37, 44, 35, 46, 57, 68};
        assertArrayEquals(expected, c, 1e-5f);

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
}