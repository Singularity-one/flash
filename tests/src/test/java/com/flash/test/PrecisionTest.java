package com.flash.test;

import com.flash.Precision;
import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for Precision enum.
 */
@DisplayName("Precision API")
class PrecisionTest {

    @Test
    @DisplayName("All precisions have correct values")
    void testPrecisionValues() {
        assertEquals(0, Precision.FP32.getValue());
        assertEquals(1, Precision.FP16.getValue());
        assertEquals(2, Precision.FP64.getValue());
        assertEquals(3, Precision.BF16.getValue());
        assertEquals(4, Precision.INT8.getValue());
        assertEquals(5, Precision.INT4.getValue());
    }

    @Test
    @DisplayName("All precisions have correct byte sizes")
    void testByteSizes() {
        assertEquals(4, Precision.FP32.getBytesPerElement());
        assertEquals(2, Precision.FP16.getBytesPerElement());
        assertEquals(8, Precision.FP64.getBytesPerElement());
        assertEquals(2, Precision.BF16.getBytesPerElement());
        assertEquals(1, Precision.INT8.getBytesPerElement());
        assertEquals(1, Precision.INT4.getBytesPerElement());
    }

    @Test
    @DisplayName("fromValue conversion works")
    void testFromValue() {
        assertEquals(Precision.FP32, Precision.fromValue(0));
        assertEquals(Precision.FP16, Precision.fromValue(1));
        assertEquals(Precision.FP64, Precision.fromValue(2));
        assertEquals(Precision.BF16, Precision.fromValue(3));
        assertEquals(Precision.INT8, Precision.fromValue(4));
        assertEquals(Precision.INT4, Precision.fromValue(5));
    }

    @Test
    @DisplayName("fromValue throws on invalid value")
    void testFromValueInvalid() {
        assertThrows(IllegalArgumentException.class, () -> {
            Precision.fromValue(99);
        });
    }

    @Test
    @DisplayName("isNativeJavaType detection")
    void testIsNativeJavaType() {
        assertTrue(Precision.FP32.isNativeJavaType());
        assertTrue(Precision.FP64.isNativeJavaType());

        assertFalse(Precision.FP16.isNativeJavaType());
        assertFalse(Precision.BF16.isNativeJavaType());
        assertFalse(Precision.INT8.isNativeJavaType());
        assertFalse(Precision.INT4.isNativeJavaType());
    }

    @Test
    @DisplayName("requiresGpuConversion detection")
    void testRequiresGpuConversion() {
        assertTrue(Precision.FP16.requiresGpuConversion());
        assertTrue(Precision.BF16.requiresGpuConversion());

        assertFalse(Precision.FP32.requiresGpuConversion());
        assertFalse(Precision.FP64.requiresGpuConversion());
        assertFalse(Precision.INT8.requiresGpuConversion());
        assertFalse(Precision.INT4.requiresGpuConversion());
    }
}