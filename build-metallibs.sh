#!/bin/bash
set -euo pipefail

# Pre-compile Candle's Metal shaders into .metallib binaries.
#
# This eliminates runtime shader compilation on iOS, which avoids the
# XPC_ERROR_CONNECTION_INTERRUPTED panic when the Metal compiler service
# gets killed under memory pressure.
#
# Usage: ./build-metallibs.sh [candle-source-dir]
#
# The compiled .metallib files should be bundled into the iOS app bundle
# and passed to MetalDevice::set_metallib_dir() at startup.

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
CANDLE_DIR="${1:-$PROJECT_DIR/../candle}"
METAL_SRC="$CANDLE_DIR/candle-metal-kernels/src/metal_src"
OUT_DIR="$PROJECT_DIR/generated/metallibs"

# Minimum deployment target (must match build.sh)
export IPHONEOS_DEPLOYMENT_TARGET=15.0

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

if [[ ! -d "$METAL_SRC" ]]; then
    echo -e "${RED}Error: Metal source directory not found: $METAL_SRC${NC}"
    echo "Usage: $0 [path-to-candle-repo]"
    exit 1
fi

echo "Metal shader pre-compilation"
echo "============================"
echo "Source: $METAL_SRC"
echo "Output: $OUT_DIR"
echo ""

mkdir -p "$OUT_DIR"

# Metal shaders to compile (matches Source enum in candle-metal-kernels)
# Note: utils.metal is a header-only utility, not a standalone kernel library
SHADERS=(
    affine
    binary
    cast
    conv
    fill
    indexing
    mlx_gemm
    mlx_sort
    quantized
    random
    reduce
    scaled_dot_product_attention
    sort
    ternary
    unary
)

FAILED=0
COMPILED=0

# Shaders using bfloat16 require Metal 3.1 (iOS 17+).
# All others compile with Metal 3.0 (iOS 16+).
BFLOAT16_SHADERS="mlx_sort scaled_dot_product_attention"

for shader in "${SHADERS[@]}"; do
    src="$METAL_SRC/${shader}.metal"
    air="$OUT_DIR/${shader}.air"
    lib="$OUT_DIR/${shader}.metallib"

    if [[ ! -f "$src" ]]; then
        echo -e "  ${YELLOW}SKIP${NC} ${shader}.metal (not found)"
        continue
    fi

    printf "  %-45s" "${shader}.metal"

    # Select Metal standard version: 3.1 for bfloat16, 3.0 for everything else
    if echo "$BFLOAT16_SHADERS" | grep -qw "$shader"; then
        METAL_STD="metal3.1"
    else
        METAL_STD="metal3.0"
    fi

    # Step 1: Compile .metal -> .air (Metal Intermediate Representation)
    # -ffast-math: matches Candle's default CANDLE_METAL_ENABLE_FAST_MATH=true
    if ! xcrun -sdk iphoneos metal \
        -c "$src" \
        -o "$air" \
        -std=$METAL_STD \
        -ffast-math \
        -mios-version-min=$IPHONEOS_DEPLOYMENT_TARGET \
        2>/dev/null; then
        echo -e "${RED}FAIL (compile)${NC}"
        FAILED=$((FAILED + 1))
        continue
    fi

    # Step 2: Link .air -> .metallib (Metal Library binary)
    if ! xcrun -sdk iphoneos metallib \
        "$air" \
        -o "$lib" \
        2>/dev/null; then
        echo -e "${RED}FAIL (link)${NC}"
        rm -f "$air"
        FAILED=$((FAILED + 1))
        continue
    fi

    # Clean up intermediate .air file
    rm -f "$air"

    size=$(du -h "$lib" | cut -f1 | xargs)
    echo -e "${GREEN}OK${NC} ($size)"
    COMPILED=$((COMPILED + 1))
done

echo ""
echo "Results: $COMPILED compiled, $FAILED failed, ${#SHADERS[@]} total"
echo ""

if [[ $FAILED -gt 0 ]]; then
    echo -e "${YELLOW}Warning: $FAILED shaders failed to compile.${NC}"
    echo "These will fall back to runtime compilation."
    echo ""
fi

echo "Metallib files: $OUT_DIR/"
ls -lh "$OUT_DIR"/*.metallib 2>/dev/null || echo "(none)"
echo ""
echo "Bundle these into your iOS app and call:"
echo '  device.as_metal_device()?.set_metallib_dir("/path/to/metallibs")'
