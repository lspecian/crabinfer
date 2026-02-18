#!/bin/bash
set -euo pipefail

# CrabInfer Build Script
# Builds the Rust core for iOS and generates Swift bindings via UniFFI

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
CORE_DIR="$PROJECT_DIR/crabinfer-core"
TARGET_DIR="$PROJECT_DIR/target"
OUT_DIR="$PROJECT_DIR/generated"
SWIFT_OUT="$PROJECT_DIR/crabinfer-swift/Sources/CrabInfer/Generated"

# Minimum iOS deployment target (Metal requires iOS 14+, we target 15+)
export IPHONEOS_DEPLOYMENT_TARGET=15.0

echo "🦀 CrabInfer Build Script"
echo "========================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
echo ""
echo "📋 Checking prerequisites..."

if ! command -v cargo &> /dev/null; then
    echo -e "${RED}Error: Rust/Cargo not found. Install from https://rustup.rs${NC}"
    exit 1
fi

if ! command -v xcrun &> /dev/null; then
    echo -e "${RED}Error: Xcode command line tools not found. Run: xcode-select --install${NC}"
    exit 1
fi

# Ensure iOS targets are installed
echo "📱 Checking Rust iOS targets..."
for target in aarch64-apple-ios aarch64-apple-ios-sim; do
    if ! rustup target list --installed | grep -q "$target"; then
        echo "  Installing $target..."
        rustup target add "$target"
    else
        echo -e "  ${GREEN}✓${NC} $target"
    fi
done

# Build from project root (workspace target dir is $PROJECT_DIR/target)
cd "$PROJECT_DIR"

# Build for iOS device (arm64)
# --lib: only build the staticlib, not bins (which can't link for iOS)
# -p: specify the crate explicitly
echo ""
echo "🔨 Building for iOS device (aarch64-apple-ios)..."
RUSTFLAGS="-C target-feature=+neon" cargo build \
    --lib -p crabinfer-core \
    --target aarch64-apple-ios \
    --release

# Build for iOS simulator (arm64 - Apple Silicon Mac)
echo ""
echo "🔨 Building for iOS simulator (aarch64-apple-ios-sim)..."
RUSTFLAGS="-C target-feature=+neon" cargo build \
    --lib -p crabinfer-core \
    --target aarch64-apple-ios-sim \
    --release

# Verify the static libraries were built
IOS_LIB="$TARGET_DIR/aarch64-apple-ios/release/libcrabinfer_core.a"
SIM_LIB="$TARGET_DIR/aarch64-apple-ios-sim/release/libcrabinfer_core.a"

if [[ ! -f "$IOS_LIB" ]]; then
    echo -e "${RED}Error: iOS device library not found at $IOS_LIB${NC}"
    exit 1
fi
if [[ ! -f "$SIM_LIB" ]]; then
    echo -e "${RED}Error: iOS simulator library not found at $SIM_LIB${NC}"
    exit 1
fi

echo -e "  ${GREEN}✓${NC} iOS device:    $(du -h "$IOS_LIB" | cut -f1)"
echo -e "  ${GREEN}✓${NC} iOS simulator: $(du -h "$SIM_LIB" | cut -f1)"

# Generate Swift bindings via UniFFI
echo ""
echo "🔗 Generating Swift bindings..."
mkdir -p "$OUT_DIR"
mkdir -p "$SWIFT_OUT"

# Build uniffi-bindgen for the host (macOS) then run it against the iOS library
cargo run -p crabinfer-core --bin uniffi-bindgen generate \
    --library "$IOS_LIB" \
    --language swift \
    --out-dir "$OUT_DIR"

# Copy generated Swift files
cp "$OUT_DIR"/*.swift "$SWIFT_OUT/" 2>/dev/null || true

# Create XCFramework
echo ""
echo "📦 Creating XCFramework..."

XCFRAMEWORK_DIR="$PROJECT_DIR/CrabInferCore.xcframework"
rm -rf "$XCFRAMEWORK_DIR"

# Prepare headers
HEADER_DIR="$OUT_DIR/include"
mkdir -p "$HEADER_DIR"
cp "$OUT_DIR"/*FFI.h "$HEADER_DIR/" 2>/dev/null || true

# Use the UniFFI-generated modulemap (must match the module name
# that the generated Swift code imports: crabinfer_coreFFI)
cp "$OUT_DIR/crabinfer_coreFFI.modulemap" "$HEADER_DIR/module.modulemap"

xcodebuild -create-xcframework \
    -library "$IOS_LIB" \
    -headers "$HEADER_DIR" \
    -library "$SIM_LIB" \
    -headers "$HEADER_DIR" \
    -output "$XCFRAMEWORK_DIR"

echo ""
echo -e "${GREEN}✅ Build complete!${NC}"
echo ""
echo "Generated files:"
echo "  📁 XCFramework: $XCFRAMEWORK_DIR"
echo "  📁 Swift bindings: $SWIFT_OUT"
echo "  📁 Headers: $HEADER_DIR"
echo ""
echo "Next steps:"
echo "  1. Add the XCFramework to your Xcode project"
echo "  2. Add the generated Swift files to your target"
echo "  3. Import CrabInfer in your Swift code"
echo ""
echo "Or use the Swift Package:"
echo "  open crabinfer-swift/Package.swift"
