package com.cuda.poc;

import java.util.Arrays;

public class Main {

    public static void main(String[] args) {
        System.out.println("=".repeat(60));
        System.out.println("cudarc + Java FFM POC - GEMM 驗證");
        System.out.println("=".repeat(60));
        System.out.println();

        try (CudaGemm gemm = new CudaGemm()) {
            System.out.println("CUDA Device: " + gemm.getDeviceName());
            System.out.println();

            testSmallGemm(gemm);
            testLargeGemm(gemm);
            benchmarkGemm(gemm);

        } catch (Exception e) {
            System.err.println("錯誤: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }

        System.out.println();
        System.out.println("✅ POC 驗證成功！");
    }

    private static void testSmallGemm(CudaGemm gemm) {
        System.out.println("測試 1: 小型矩陣 (2×2)");
        System.out.println("-".repeat(60));

        float[] a = {1, 3, 2, 4};
        float[] b = {5, 7, 6, 8};
        float[] c = new float[4];

        System.out.println("A = ");
        printMatrix(a, 2, 2);
        System.out.println("B = ");
        printMatrix(b, 2, 2);

        CudaGemm.Status status = gemm.multiply(2, 2, 2, a, b, c);

        System.out.println("狀態: " + status);
        System.out.println("C = A * B =");
        printMatrix(c, 2, 2);

        float[] expected = {19, 43, 22, 50};
        boolean correct = Arrays.equals(c, expected);
        System.out.println("結果驗證: " + (correct ? "✅ 正確" : "❌ 錯誤"));
        System.out.println();
    }

    private static void testLargeGemm(CudaGemm gemm) {
        System.out.println("測試 2: 較大矩陣 (100×100)");
        System.out.println("-".repeat(60));

        int n = 100;
        float[] a = new float[n * n];
        float[] b = new float[n * n];
        float[] c = new float[n * n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                a[i + j * n] = (i == j) ? 1.0f : 0.0f;
                b[i + j * n] = (float) (i + j);
            }
        }

        long start = System.nanoTime();
        CudaGemm.Status status = gemm.multiply(n, n, n, a, b, c);
        long elapsed = System.nanoTime() - start;

        System.out.printf("維度: %d × %d × %d%n", n, n, n);
        System.out.println("狀態: " + status);
        System.out.printf("計算時間: %.2f ms%n", elapsed / 1e6);
        System.out.println();
    }

    private static void benchmarkGemm(CudaGemm gemm) {
        System.out.println("測試 3: 性能基準");
        System.out.println("-".repeat(60));

        int[] sizes = {64, 128, 256, 512};

        System.out.printf("%-10s %-15s %-15s%n", "Size", "Time (ms)", "GFLOPS");
        System.out.println("-".repeat(60));

        for (int n : sizes) {
            float[] a = randomMatrix(n, n);
            float[] b = randomMatrix(n, n);
            float[] c = new float[n * n];

            gemm.multiply(n, n, n, a, b, c);

            int iterations = 5;
            long totalTime = 0;

            for (int i = 0; i < iterations; i++) {
                Arrays.fill(c, 0.0f);
                long start = System.nanoTime();
                gemm.multiply(n, n, n, a, b, c);
                totalTime += System.nanoTime() - start;
            }

            double avgTime = totalTime / (double) iterations / 1e6;
            double gflops = (2.0 * n * n * n) / (avgTime / 1000.0) / 1e9;

            System.out.printf("%-10d %-15.2f %-15.2f%n", n, avgTime, gflops);
        }
        System.out.println();
    }

    private static float[] randomMatrix(int rows, int cols) {
        float[] matrix = new float[rows * cols];
        for (int i = 0; i < matrix.length; i++) {
            matrix[i] = (float) (Math.random() * 2.0 - 1.0);
        }
        return matrix;
    }

    private static void printMatrix(float[] matrix, int rows, int cols) {
        for (int i = 0; i < rows; i++) {
            System.out.print("  [");
            for (int j = 0; j < cols; j++) {
                System.out.printf("%6.1f", matrix[i + j * rows]);
            }
            System.out.println(" ]");
        }
    }
}
