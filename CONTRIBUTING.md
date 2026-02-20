# Contributing to CrabInfer

Thank you for your interest in contributing to CrabInfer! This project is in its early stages, and contributions of all kinds are welcome.

## Getting Started

### Prerequisites

- **Rust** (stable, latest): [rustup.rs](https://rustup.rs)
- **iOS targets** (for cross-compilation):
  ```bash
  rustup target add aarch64-apple-ios aarch64-apple-ios-sim
  ```
- **Xcode** with command line tools:
  ```bash
  xcode-select --install
  ```

### Building

```bash
# Build the Rust core
cargo build

# Build for iOS (generates XCFramework)
./build.sh

# Run tests
cargo test
```

## How to Contribute

### Reporting Bugs

- Open an issue on GitHub with a clear description of the problem.
- Include your device/OS info, Rust version, and Xcode version.
- If possible, include a minimal reproduction case.

### Suggesting Features

- Open an issue describing the feature and its use case.
- Explain why it would be useful to the broader community.

### Submitting Pull Requests

1. **Fork** the repository and create your branch from `main`.
2. **Write code** that follows the existing style and conventions.
3. **Add tests** for any new functionality.
4. **Run the test suite** to make sure nothing is broken:
   ```bash
   cargo test
   cargo clippy -- -D warnings
   cargo fmt --check
   ```
5. **Write a clear PR description** explaining what changed and why.

## Code Style

- **Rust**: Follow standard Rust conventions. Run `cargo fmt` before committing.
- **Swift**: Follow [Swift API Design Guidelines](https://www.swift.org/documentation/api-design-guidelines/).
- Use `cargo clippy` to catch common mistakes and improve code quality.

## Architecture Guidelines

CrabInfer has a layered architecture. When contributing, keep these principles in mind:

- **Memory safety first** — Every allocation should be accounted for. Use Rust's ownership model to prevent leaks.
- **iOS lifecycle awareness** — Code must handle memory pressure notifications and background transitions gracefully.
- **Graceful degradation** — Features should fall back to simpler alternatives rather than failing outright (e.g., Metal GPU to CPU NEON to smaller model).
- **UniFFI boundary** — Keep the FFI surface area small. Complex logic belongs in Rust, not Swift.

## Areas Where Help Is Needed

Check the [README](README.md) for the current development status. High-impact areas include:

- Candle integration with Metal backend
- GGUF model loading
- Token streaming
- Context window management
- Graceful degradation chain
- Swift Package distribution
- Testing on various iOS devices

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
