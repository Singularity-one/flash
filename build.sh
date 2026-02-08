#!/bin/bash
# flash Build Script (Linux/macOS)
# 只負責 Rust build + 複製 native library
# Java 編譯/測試請在 IntelliJ 執行

set -e

echo "=== flash Build ==="
echo ""

# Detect platform
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)
[[ "$ARCH" == "x86_64" ]] && ARCH="x86_64"
[[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]] && ARCH="aarch64"

PLATFORM="${OS}-${ARCH}"

if [[ "$OS" == "darwin" ]]; then
    LIB_NAME="libflash.dylib"
else
    LIB_NAME="libflash.so"
fi

echo "Platform: $PLATFORM"
echo ""

# 1. Build Rust
echo "[1/2] Building Rust..."
cd flash-rust
cargo build --release
cd ..
echo "  OK"

# 2. Copy native library
echo "[2/2] Copying native library..."
TARGET_DIR="flash-core/src/main/resources/native/${PLATFORM}"
SOURCE_LIB="flash-rust/target/release/$LIB_NAME"

if [[ ! -f "$SOURCE_LIB" ]]; then
    echo "ERROR: Native library not found: $SOURCE_LIB"
    exit 1
fi

mkdir -p "$TARGET_DIR"
cp "$SOURCE_LIB" "$TARGET_DIR/"
echo "  -> $TARGET_DIR/$LIB_NAME"

echo ""
echo "=== Build Complete ==="
echo ""
echo "Next: Open IntelliJ and run tests"
echo ""