package com.cuda.poc;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;

public class CudaGemm implements AutoCloseable {

    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup LIBRARY;

    private static final MethodHandle GEMM_INIT;
    private static final MethodHandle GEMM_SGEMM;
    private static final MethodHandle GEMM_DESTROY;
    private static final MethodHandle GEMM_GET_DEVICE_NAME;

    private final MemorySegment context;
    private boolean closed = false;

    static {
        try {
            // 使用智能載入器
            Path libPath = NativeLibraryLoader.loadLibrary();
            LIBRARY = SymbolLookup.libraryLookup(libPath, Arena.global());
            System.out.println("✓ CUDA GEMM 庫載入成功");

        } catch (Exception e) {
            System.err.println("✗ 載入失敗: " + e.getMessage());
            throw new ExceptionInInitializerError(e);
        }

        try {
            GEMM_INIT = LINKER.downcallHandle(
                    LIBRARY.find("gemm_init").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.ADDRESS)
            );

            GEMM_SGEMM = LINKER.downcallHandle(
                    LIBRARY.find("gemm_sgemm").orElseThrow(),
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_INT,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_FLOAT,
                            ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_FLOAT,
                            ValueLayout.ADDRESS
                    )
            );

            GEMM_DESTROY = LINKER.downcallHandle(
                    LIBRARY.find("gemm_destroy").orElseThrow(),
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS)
            );

            GEMM_GET_DEVICE_NAME = LINKER.downcallHandle(
                    LIBRARY.find("gemm_get_device_name").orElseThrow(),
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_BOOLEAN,
                            ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_LONG
                    )
            );

        } catch (Throwable e) {
            throw new ExceptionInInitializerError(e);
        }
    }



    public enum Status {
        SUCCESS(0),
        DEVICE_INIT_ERROR(1),
        MEMORY_ERROR(2),
        COMPUTE_ERROR(3),
        INVALID_DIMENSION(4);

        private final int code;

        Status(int code) {
            this.code = code;
        }

        public static Status fromCode(int code) {
            for (Status s : values()) {
                if (s.code == code) return s;
            }
            throw new IllegalArgumentException("Unknown status code: " + code);
        }
    }

    public CudaGemm() {
        try {
            context = (MemorySegment) GEMM_INIT.invoke();
            if (context.address() == 0) {
                throw new RuntimeException("Failed to initialize CUDA device");
            }
        } catch (Throwable e) {
            throw new RuntimeException("Failed to initialize GEMM context", e);
        }
    }

    public Status sgemm(
            int m, int n, int k,
            float alpha,
            float[] a,
            float[] b,
            float beta,
            float[] c
    ) {
        if (closed) {
            throw new IllegalStateException("GEMM context is closed");
        }

        if (a.length != m * k || b.length != k * n || c.length != m * n) {
            throw new IllegalArgumentException("Invalid matrix dimensions");
        }

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment aSeg = arena.allocateFrom(ValueLayout.JAVA_FLOAT, a);
            MemorySegment bSeg = arena.allocateFrom(ValueLayout.JAVA_FLOAT, b);
            MemorySegment cSeg = arena.allocateFrom(ValueLayout.JAVA_FLOAT, c);

            int statusCode = (int) GEMM_SGEMM.invoke(
                    context, m, n, k, alpha, aSeg, bSeg, beta, cSeg
            );

            MemorySegment.copy(cSeg, ValueLayout.JAVA_FLOAT, 0, c, 0, c.length);

            return Status.fromCode(statusCode);

        } catch (Throwable e) {
            throw new RuntimeException("GEMM computation failed", e);
        }
    }

    public String getDeviceName() {
        if (closed) {
            throw new IllegalStateException("GEMM context is closed");
        }

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment buffer = arena.allocate(256);

            boolean success = (boolean) GEMM_GET_DEVICE_NAME.invoke(
                    context, buffer, 256L
            );

            if (!success) {
                return "Unknown Device";
            }

            return buffer.getString(0);

        } catch (Throwable e) {
            return "Error: " + e.getMessage();
        }
    }

    @Override
    public void close() {
        if (!closed) {
            try {
                GEMM_DESTROY.invoke(context);
                closed = true;
            } catch (Throwable e) {
                throw new RuntimeException("Failed to destroy GEMM context", e);
            }
        }
    }

    public Status multiply(int m, int n, int k, float[] a, float[] b, float[] c) {
        return sgemm(m, n, k, 1.0f, a, b, 0.0f, c);
    }
}
