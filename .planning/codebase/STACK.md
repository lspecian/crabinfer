# Technology Stack

**Analysis Date:** 2026-03-12

## Languages

**Primary:**
- Rust (Edition 2021) — all four workspace crates: `crabinfer-core`, `crabinfer-server`, `crabinfer-cli`, `crabinfer-node`

**Secondary:**
- Swift — `crabinfer-swift/` bindings via UniFFI, iOS/macOS consumer layer
- CUDA — `crabinfer-core/src/serving/kernels/paged_attention.cu` custom kernel
- Metal Shading Language — compiled to `.metallib` for Apple Silicon GPU kernels (loaded via `candle-metal-kernels`)
- TypeScript/JavaScript — `crabinfer-node/` type declarations (`index.d.ts`, `index.js`)

## Runtime

**Environment:**
- Native binary (no VM). iOS/macOS target via Rust + UniFFI; Linux via Docker.
- Node.js >= 18 for `crabinfer-node` bindings (`napi9` ABI)

**Package Manager:**
- Cargo (Rust workspace) — `Cargo.lock` present
- npm (node binding) — `crabinfer-node/package-lock.json` present

## Frameworks

**Core ML:**
- `candle-core` / `candle-nn` / `candle-transformers` — local fork at `../candle/` (branch `ios-metal-fix`); fixes Llama RoPE + GQA for iOS Metal
- `candle-metal-kernels` — Metal GPU kernel dispatch (optional, `metal` feature)
- `candle-flash-attn` — FlashAttention-2 for CUDA sm_80+ (optional, `flash-attn` feature)

**HTTP Server:**
- `axum` 0.8 — async web framework for `crabinfer-server`
- `tokio` 1 (rt-multi-thread, sync, macros, signal) — async runtime

**Middleware:**
- `tower-http` 0.6 (cors) — CORS layer; permissive in dev, body limit 1 MB

**CLI:**
- `clap` 4 (derive) — argument parsing for `crabinfer-cli` and `crabinfer-server`
- `rustyline` 14 — interactive chat REPL
- `rpassword` 7 — hidden password/API key input

**FFI / Bindings:**
- `uniffi` 0.28 (cli feature) — Swift bridge code generation from `crabinfer.udl`
- `napi` 2 (async, napi9, tokio_rt) + `napi-derive` 2 — Node.js bindings
- `napi-build` 2 — build-time napi setup

**Serialization:**
- `serde` 1 (derive) + `serde_json` 1 — JSON throughout all crates

**Tokenization:**
- `tokenizers` 0.21 (`fancy-regex` feature, no `onig` C dependency) — HuggingFace tokenizer

**Testing:**
- Built-in Rust `#[test]` / `cargo test`. ~195 tests across workspace.

## Key Dependencies

**Critical:**
- `candle-core` (fork) — tensor operations and device abstraction (CPU/Metal/CUDA)
- `uniffi` 0.28 — Swift FFI; must be `panic = "unwind"` in release profile
- `reqwest` 0.12 (`blocking`, `json`, `rustls-tls`, no default features) — HTTP client for cloud providers; gated behind `providers` feature
- `tokenizers` 0.21 — model tokenization; `fancy-regex` avoids Oniguruma C library (iOS link issue)

**Infrastructure:**
- `tokio` 1 — async runtime across server, CLI serve command, and Node bindings
- `tracing` 0.1 + `tracing-subscriber` 0.3 — structured logging across all crates
- `thiserror` 2 — error type derivation in `crabinfer-core`
- `half` 2.7 — f16/bf16 types for CUDA kernel dispatch
- `memmap2` 0.9 — memory-mapped GGUF model file loading
- `libc` 0.2 — sysctl for device detection (macOS) and system memory queries
- `sha2` 0.10 — SHA-256 integrity verification of downloaded GGUF files (optional, `providers` feature)
- `tokio-stream` 0.1 — token streaming in `crabinfer-server`
- `objc2-metal` 0.3.1 — low-level Metal API bindings (optional, `metal` feature)

## Configuration

**Environment:**
- No `.env` file mechanism; API keys are stored in an in-process `CredentialManager` singleton (`crabinfer-core/src/credentials.rs`)
- Swift apps populate `CredentialManager` at launch (e.g., from iOS Keychain)
- For server/CLI usage, keys are passed via CLI flags or `rpassword` interactive prompt

**Feature Flags (Cargo features):**
- `metal` (default) — Metal GPU acceleration; disable for Linux builds
- `cuda` — NVIDIA GPU acceleration
- `flash-attn` — FlashAttention-2 (requires `cuda`, sm_80+)
- `cpu-only` — force CPU-only mode (CI, Linux without GPU)
- `providers` — enables cloud provider HTTP clients + model download manager

**Build:**
- `Cargo.toml` (workspace root) — workspace members, release profile, candle patch
- `build.sh` — iOS multi-arch build + UniFFI Swift binding generation
- `build-metallibs.sh` — pre-compiles Metal `.metallib` files
- `Dockerfile` — multi-stage Linux CPU-only server image (Rust 1.82 builder → debian:bookworm-slim)
- `deploy/crabinfer-server.service` — systemd unit for Linux server deployment

## Platform Requirements

**Development:**
- macOS with Xcode command-line tools (for Metal/iOS targets)
- Rust toolchain with iOS targets: `aarch64-apple-ios`, `aarch64-apple-ios-sim`, `x86_64-apple-ios`
- Local candle fork at `../candle/` relative to project root
- iOS deployment target: 15.0 (Metal requires iOS 14+)

**Production:**
- iOS/macOS: static library (`crate-type = ["lib", "staticlib"]`) consumed via Swift Package Manager (`crabinfer-swift/Package.swift`)
- Linux server: Docker image (`debian:bookworm-slim`) exposes port 8080
- Node.js: platform-specific `.node` native addons distributed as optional npm dependencies for darwin-arm64, darwin-x64, linux-x64-gnu, linux-arm64-gnu

---

*Stack analysis: 2026-03-12*
