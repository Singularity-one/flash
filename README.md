## 🚀 快速開始

### 前置需求
- Rust 1.70+
- Java 22+
- CUDA Toolkit 12.6+ (可選，cudarc 會自動處理)

### 建置
```bash
# 一鍵建置（Windows）
.\build.ps1

# 或手動
cd rust-cuda-gemm && cargo build --release
cd ../java-gemm-client && mvn clean compile
```

### 執行測試
```bash
cd java-gemm-client
mvn test
```