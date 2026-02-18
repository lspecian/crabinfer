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
            path: "../CrabInferCore.xcframework"
        ),
        .testTarget(
            name: "CrabInferTests",
            dependencies: ["CrabInfer"]
        ),
    ]
)
