Write-Host "cudarc + Java FFM POC - 建置腳本" -ForegroundColor Cyan

Set-Location rust-cuda-gemm
cargo build --release
Set-Location ..

Set-Location java-gemm-client
mvn clean compile
mvn test
mvn exec:java -D"exec.mainClass"="com.cuda.poc.Main"
