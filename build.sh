#!/bin/bash
set -e

echo "========================================"
echo "cudarc + Java FFM POC - 建置腳本"
echo "========================================"

cd rust-cuda-gemm
cargo build --release
cd ..

cd java-gemm-client
mvn clean compile
mvn test
mvn exec:java -Dexec.mainClass="com.cuda.poc.Main"