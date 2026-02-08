package com.flash.test;
import com.flash.CudaDevice;
import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for CudaDevice.
 */
@DisplayName("CudaDevice API")
class CudaDeviceTest {

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

    @Test
    @DisplayName("Device initialization works")
    void testDeviceInit() {
        assertNotNull(device);
        assertFalse(device.isClosed());
    }

    @Test
    @DisplayName("Get device name")
    void testGetDeviceName() {
        String name = device.getName();
        assertNotNull(name);
        assertFalse(name.isEmpty());
        assertFalse(name.startsWith("Error"));
        assertFalse(name.equals("Unknown Device"));
    }

    @Test
    @DisplayName("Memory allocation and deallocation")
    void testMemoryAllocation() {
        long ptr = device.allocate(1024);
        assertNotEquals(0, ptr, "Allocation should return non-null pointer");

        device.free(ptr);
    }

    @Test
    @DisplayName("Host to Device copy")
    void testHostToDeviceCopy() {
        float[] data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        long ptr = device.allocate(data.length * Float.BYTES);

        var status = device.copyHostToDevice(data, ptr);
        assertEquals(CudaDevice.Status.SUCCESS, status);

        device.free(ptr);
    }

    @Test
    @DisplayName("Device to Host copy")
    void testDeviceToHostCopy() {
        float[] data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        long ptr = device.allocate(data.length * Float.BYTES);

        device.copyHostToDevice(data, ptr);

        float[] result = new float[data.length];
        var status = device.copyDeviceToHost(ptr, result);

        assertEquals(CudaDevice.Status.SUCCESS, status);
        assertArrayEquals(data, result, 1e-6f);

        device.free(ptr);
    }

    @Test
    @DisplayName("Round-trip copy preserves data")
    void testRoundTripCopy() {
        float[] original = {-3.14f, 2.71f, 1.41f, 0.0f, 42.0f};
        long ptr = device.allocate(original.length * Float.BYTES);

        device.copyHostToDevice(original, ptr);

        float[] copied = new float[original.length];
        device.copyDeviceToHost(ptr, copied);

        assertArrayEquals(original, copied, 1e-6f);

        device.free(ptr);
    }

    @Test
    @DisplayName("Device synchronization works")
    void testSynchronize() {
        var status = device.synchronize();
        assertEquals(CudaDevice.Status.SUCCESS, status);
    }

    @Test
    @DisplayName("Large allocation (1MB)")
    void testLargeAllocation() {
        long bytes = 1024 * 1024; // 1MB
        long ptr = device.allocate(bytes);
        assertNotEquals(0, ptr);

        device.free(ptr);
    }

    @Test
    @DisplayName("Invalid allocation size throws exception")
    void testInvalidAllocationSize() {
        assertThrows(IllegalArgumentException.class, () -> {
            device.allocate(0);
        });

        assertThrows(IllegalArgumentException.class, () -> {
            device.allocate(-100);
        });
    }

    @Test
    @DisplayName("Operations on closed device throw exception")
    void testClosedDeviceThrows() {
        CudaDevice tempDevice = new CudaDevice(0);
        tempDevice.close();

        assertTrue(tempDevice.isClosed());

        assertThrows(IllegalStateException.class, () -> {
            tempDevice.getName();
        });

        assertThrows(IllegalStateException.class, () -> {
            tempDevice.allocate(1024);
        });
    }
}
