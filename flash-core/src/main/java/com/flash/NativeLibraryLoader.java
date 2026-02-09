package com.flash;

import java.io.*;
import java.lang.foreign.Arena;
import java.lang.foreign.SymbolLookup;
import java.nio.file.*;

/**
 * Loads the native Rust library for the current platform.
 *
 * <p>Search order:</p>
 * <ol>
 *   <li>JAR resources: {@code /native/{os}-{arch}/}</li>
 *   <li>System library path: {@code java.library.path}</li>
 * </ol>
 *
 * @author Singularity-one
 * @since 0.1.0
 */
public class NativeLibraryLoader {

    private static final String LIB_NAME = "flash";
    private static Path extractedLibPath = null;
    private static SymbolLookup library = null;

    /**
     * Loads and returns the path to the native library.
     *
     * @return path to the native library file
     * @throws IOException if library cannot be found or extracted
     */
    public static synchronized Path loadLibrary() throws IOException {
        if (extractedLibPath != null) {
            return extractedLibPath;
        }

        String libFileName = getLibraryFileName();
        String platform = getPlatform();

        // Strategy 1: From JAR resources
        String resourcePath = "/native/" + platform + "/" + libFileName;
        InputStream libStream = NativeLibraryLoader.class.getResourceAsStream(resourcePath);

        if (libStream != null) {
            try {
                extractedLibPath = extractFromJar(libStream, libFileName);
                return extractedLibPath;
            } finally {
                libStream.close();
            }
        }

        // Strategy 2: System library path
        String systemPath = System.getProperty("java.library.path");
        if (systemPath != null) {
            for (String dir : systemPath.split(File.pathSeparator)) {
                Path libPath = Paths.get(dir, libFileName);
                if (Files.exists(libPath)) {
                    extractedLibPath = libPath;
                    return extractedLibPath;
                }
            }
        }

        throw new IOException(
                "Native library not found: " + libFileName + "\n" +
                        "Platform: " + platform + "\n" +
                        "Searched:\n" +
                        "  1. JAR resource: " + resourcePath + "\n" +
                        "  2. java.library.path: " + systemPath + "\n\n" +
                        "Did you run the build script? (build.ps1 or build.sh)"
        );
    }

    /**
     * Gets the SymbolLookup for the native library.
     *
     * @return SymbolLookup instance
     * @throws RuntimeException if library cannot be loaded
     */
    public static synchronized SymbolLookup getLibrary() {
        if (library != null) {
            return library;
        }

        try {
            Path libPath = loadLibrary();
            library = SymbolLookup.libraryLookup(libPath, Arena.global());
            return library;
        } catch (IOException e) {
            throw new RuntimeException("Failed to load native library", e);
        }
    }

    private static String getLibraryFileName() {
        String os = System.getProperty("os.name").toLowerCase();
        if (os.contains("win")) {
            return LIB_NAME + ".dll";
        } else if (os.contains("mac")) {
            return "lib" + LIB_NAME + ".dylib";
        } else {
            return "lib" + LIB_NAME + ".so";
        }
    }

    private static String getPlatform() {
        String os = System.getProperty("os.name").toLowerCase();
        String arch = System.getProperty("os.arch").toLowerCase();

        String osName;
        if (os.contains("win")) {
            osName = "windows";
        } else if (os.contains("mac")) {
            osName = "macos";
        } else {
            osName = "linux";
        }

        String archName;
        if (arch.contains("amd64") || arch.contains("x86_64")) {
            archName = "x86_64";
        } else if (arch.contains("aarch64") || arch.contains("arm64")) {
            archName = "aarch64";
        } else {
            archName = arch;
        }

        return osName + "-" + archName;
    }

    private static Path extractFromJar(InputStream input, String fileName) throws IOException {
        Path tempDir = Files.createTempDirectory("flash-native-");
        tempDir.toFile().deleteOnExit();

        Path tempLib = tempDir.resolve(fileName);
        tempLib.toFile().deleteOnExit();

        Files.copy(input, tempLib, StandardCopyOption.REPLACE_EXISTING);
        return tempLib;
    }
}