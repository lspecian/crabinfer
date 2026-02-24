#!/usr/bin/env bash
# prepare-spm-release.sh — Update Package.swift with GitHub Releases URL and checksum
#
# Usage: ./scripts/prepare-spm-release.sh v0.2.0 /path/to/CrabInferCore.xcframework.zip
#
# This script:
# 1. Computes SHA256 of the XCFramework zip
# 2. Updates Package.swift with the release URL and checksum
set -euo pipefail

if [ $# -ne 2 ]; then
  echo "Usage: $0 <tag> <xcframework-zip-path>"
  echo "Example: $0 v0.2.0 ./CrabInferCore.xcframework.zip"
  exit 1
fi

TAG="$1"
ZIP_PATH="$2"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_SWIFT="$ROOT/crabinfer-swift/Package.swift"

if [ ! -f "$ZIP_PATH" ]; then
  echo "Error: XCFramework zip not found at $ZIP_PATH"
  exit 1
fi

if [ ! -f "$PACKAGE_SWIFT" ]; then
  echo "Error: Package.swift not found at $PACKAGE_SWIFT"
  exit 1
fi

# Compute SHA256 checksum
CHECKSUM=$(shasum -a 256 "$ZIP_PATH" | awk '{print $1}')
URL="https://github.com/lspecian/crabinfer/releases/download/$TAG/CrabInferCore.xcframework.zip"

echo "XCFramework checksum: $CHECKSUM"
echo "Download URL: $URL"

# Generate updated Package.swift
cat > "$PACKAGE_SWIFT" << EOF
// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "CrabInfer",
    platforms: [
        .iOS(.v16),
        .macOS(.v13),
    ],
    products: [
        .library(
            name: "CrabInfer",
            targets: ["CrabInfer"]
        ),
    ],
    targets: [
        // The Swift wrapper around the Rust core
        .target(
            name: "CrabInfer",
            dependencies: ["CrabInferCore"],
            path: "Sources/CrabInfer"
        ),
        // The XCFramework containing the compiled Rust library
        .binaryTarget(
            name: "CrabInferCore",
            url: "$URL",
            checksum: "$CHECKSUM"
        ),
        .testTarget(
            name: "CrabInferTests",
            dependencies: ["CrabInfer"]
        ),
    ]
)
EOF

echo ""
echo "Updated $PACKAGE_SWIFT"
echo "  URL: $URL"
echo "  Checksum: $CHECKSUM"
echo ""
echo "Next: commit and push the updated Package.swift"
