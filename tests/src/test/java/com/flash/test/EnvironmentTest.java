package com.flash.test;

import com.flash.CudaGemm;
import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Validates the runtime environment for flash.
 *
 * <p>These tests verify:</p>
 * <ul>
 *   <li>Native library can be loaded</li>
 *   <li>CUDA device is accessible</li>
 *   <li>Basic GPU operations work</li>
 * </ul>
 */
@DisplayName("Environment Validation")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class EnvironmentTest {

    @Test
    @Order(1)
    @DisplayName("✓ Native library loads")
    void nativeLibraryLoads() {
        assertDoesNotThrow(() -> {
            try (var gemm = new CudaGemm()) {
                // Success
            }
        }, "Failed to load native library. Did you run build.ps1/build.sh?");
    }

    @Test
    @Order(2)
    @DisplayName("✓ CUDA device detected")
    void cudaDeviceDetected() {
        try (var gemm = new CudaGemm()) {
            String name = gemm.getDeviceName();

            assertNotNull(name, "Device name is null");
            assertFalse(name.startsWith("Error"), "Failed to get device name: " + name);
            assertFalse(name.equals("Unknown Device"), "No CUDA device found");

            System.out.println("  GPU: " + name);
        }
    }

    @Test
    @Order(3)
    @DisplayName("✓ Basic computation works")
    void basicComputationWorks() {
        try (var gemm = new CudaGemm()) {
            float[] a = {1, 0, 0, 1};  // Identity matrix
            float[] b = {5, 6, 7, 8};
            float[] c = new float[4];

            var status = gemm.multiply(2, 2, 2, a, b, c);

            assertEquals(CudaGemm.Status.SUCCESS, status, "GEMM failed");
            assertArrayEquals(b, c, 1e-5f, "Identity multiplication failed");
        }
    }
}