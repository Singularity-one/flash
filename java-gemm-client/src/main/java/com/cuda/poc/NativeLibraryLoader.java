package com.cuda.poc;

import java.io.*;
import java.nio.file.*;

/**
 * 原生庫載入器 - 支援從 JAR 內部載入 DLL
 */
public class NativeLibraryLoader {

    private static final String LIB_NAME = "cuda_gemm_ffi";
    private static Path extractedLibPath = null;

    /**
     * 載入原生庫
     * 優先順序：
     * 1. 開發模式：從 Rust 編譯目錄載入
     * 2. JAR 模式：從 resources 解壓後載入
     * 3. 系統路徑：使用 java.library.path
     */
    public static Path loadLibrary() throws IOException {
        if (extractedLibPath != null) {
            return extractedLibPath;
        }

        String osName = System.getProperty("os.name").toLowerCase();
        String libFileName;

        if (osName.contains("win")) {
            libFileName = LIB_NAME + ".dll";
        } else if (osName.contains("mac")) {
            libFileName = "lib" + LIB_NAME + ".dylib";
        } else {
            libFileName = "lib" + LIB_NAME + ".so";
        }

        // 策略 1: 開發模式 - 從 Rust 編譯目錄載入
        Path devPath = Paths.get("..", "rust-cuda-gemm", "target", "release", libFileName);
        if (Files.exists(devPath)) {
            System.out.println("✓ [開發模式] 載入: " + devPath.toAbsolutePath());
            extractedLibPath = devPath.toAbsolutePath();
            return extractedLibPath;
        }

        // 策略 2: JAR 模式 - 從 resources 解壓
        String resourcePath = "/native/" + getOsArch() + "/" + libFileName;
        InputStream libStream = NativeLibraryLoader.class.getResourceAsStream(resourcePath);

        if (libStream != null) {
            try {
                extractedLibPath = extractLibraryFromJar(libStream, libFileName);
                System.out.println("✓ [JAR 模式] 解壓到: " + extractedLibPath);
                return extractedLibPath;
            } finally {
                libStream.close();
            }
        }

        // 策略 3: 系統路徑查找
        String systemLibPath = System.getProperty("java.library.path");
        if (systemLibPath != null) {
            for (String path : systemLibPath.split(File.pathSeparator)) {
                Path libPath = Paths.get(path, libFileName);
                if (Files.exists(libPath)) {
                    System.out.println("✓ [系統路徑] 載入: " + libPath);
                    extractedLibPath = libPath;
                    return extractedLibPath;
                }
            }
        }

        throw new IOException(
                "找不到原生庫: " + libFileName + "\n" +
                        "請確認:\n" +
                        "1. Rust 已編譯: cargo build --release\n" +
                        "2. DLL 已放入: src/main/resources/native/<os-arch>/\n" +
                        "3. 或設置 -Djava.library.path"
        );
    }

    /**
     * 從 JAR 解壓原生庫到臨時目錄
     */
    private static Path extractLibraryFromJar(InputStream input, String libFileName)
            throws IOException {

        // 創建臨時目錄（程式結束時自動刪除）
        Path tempDir = Files.createTempDirectory("cuda-gemm-native-");
        tempDir.toFile().deleteOnExit();

        Path tempLib = tempDir.resolve(libFileName);
        tempLib.toFile().deleteOnExit();

        // 複製 DLL 到臨時目錄
        Files.copy(input, tempLib, StandardCopyOption.REPLACE_EXISTING);

        return tempLib;
    }

    /**
     * 獲取操作系統和架構標識
     */
    private static String getOsArch() {
        String osName = System.getProperty("os.name").toLowerCase();
        String osArch = System.getProperty("os.arch").toLowerCase();

        String os;
        if (osName.contains("win")) {
            os = "windows";
        } else if (osName.contains("mac")) {
            os = "macos";
        } else if (osName.contains("linux")) {
            os = "linux";
        } else {
            os = "unknown";
        }

        String arch;
        if (osArch.contains("amd64") || osArch.contains("x86_64")) {
            arch = "x86_64";
        } else if (osArch.contains("aarch64") || osArch.contains("arm64")) {
            arch = "aarch64";
        } else {
            arch = "unknown";
        }

        return os + "-" + arch;
    }
}
