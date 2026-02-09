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

### 開發步驟
```bash
cd D:\gitHub\flash\flash-rust
cargo clean
cargo build --release
cargo test -- --test-threads=1 --nocapture
cd ..
 .\build.ps1
mvn test
```

### Phase 1
```bash
未完成
BLAS Level 1
BLAS Level 2 
BLAS Level 3 
cudarc 0.12.1 確實沒有導出 Level-1 BLAS 的 FFI 
```