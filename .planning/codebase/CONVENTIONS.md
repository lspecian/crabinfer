# Coding Conventions

**Analysis Date:** 2026-03-12

## Naming Patterns

**Files:**
- Snake case for all Rust source files: `block_pool.rs`, `engine_loop.rs`, `ngram_draft.rs`
- Module directories use snake case: `crabinfer-core/src/serving/`, `crabinfer-core/src/providers/`
- Each module has a `mod.rs` when it is a directory: `serving/mod.rs`, `providers/mod.rs`, `kernels/mod.rs`
- Binary entry points: `main.rs` per crate, `bin/uniffi-bindgen.rs` for special binaries

**Functions:**
- Public functions: `snake_case` — `fn complete()`, `fn is_available()`, `fn detect_device()`
- Constructor convention: `fn new(config: FooConfig) -> Result<Self, CrabInferError>` or `fn new(...) -> Self`
- Builder methods use `with_` prefix: `fn with_persist_path(mut self, path: &str) -> Self`, `fn with_max_messages(mut self, max: usize) -> Self`, `fn with_chunker(mut self, ...) -> Self`
- Predicate functions use `is_` prefix: `fn is_available() -> bool`, `fn is_empty() -> bool`, `fn is_model_loaded() -> bool`
- Private helpers are also snake case: `fn resolve_model()`, `fn build_messages()`, `fn hash_ngram()`
- Test helpers inside `mod tests` use descriptive snake case: `fn test_config()`, `fn mock_local()`, `fn simple_request()`

**Variables and Fields:**
- Snake case throughout: `provider_name`, `api_key`, `base_url`, `max_tokens`
- Boolean fields prefer verb forms: `is_local`, `privacy_mode`, `enable_prefix_cache`
- Counter fields use `_count` suffix: `prefix_cache_hits`, `prefix_cache_misses`, `tokens_generated`

**Types (structs, enums, traits):**
- `PascalCase` for all types: `BlockPool`, `RoutingDecision`, `CrabInferError`, `ProviderTier`
- Config structs end with `Config`: `BlockPoolConfig`, `SchedulerConfig`, `ServingEngineConfig`, `KVCacheConfig`
- Error type: singular `CrabInferError` (one top-level error enum in `crabinfer-core/src/lib.rs`)
- Trait names are noun-based roles: `Provider`, `Tool`, `McpClient`, `ModelRunner`
- Type aliases use PascalCase: `type SeqId = u64`, `type NgramKey = u64`
- UniFFI-exported types append derive macros: `#[derive(Debug, Clone, uniffi::Record)]`

## Code Style

**Formatting:**
- Standard `rustfmt` formatting assumed (no custom `.rustfmt.toml` present)
- Line continuations for long method chains place `.method()` on new lines
- Inline comments on the same line with two spaces before `//`

**Linting:**
- Minimal `#[allow(...)]` usage — only `#[allow(dead_code)]` for internal JSON-RPC fields in `mcp.rs` and `#[allow(unused_mut)]` for conditional compilation branches in `lib.rs`
- No `#![deny(...)]` crate-level lint configuration found

## Import Organization

**Order pattern:**
1. External crate imports (`use candle_core::...`, `use serde::...`, `use reqwest::...`)
2. Internal crate imports (`use crate::...`) or sibling module imports (`use super::...`)

**Import style:**
- Grouped braces for multiple items from the same path: `use crate::provider::{CompletionRequest, CompletionResponse, ModelDescriptor, Provider}`
- Test modules always import `use super::*;` at the top

**Feature-gated imports:**
- `#[cfg(feature = "cuda")]` wraps CUDA-specific use statements inline: `crabinfer-core/src/serving/engine_loop.rs:20`
- `#[cfg(feature = "providers")]` wraps cloud provider modules: `crabinfer-core/src/lib.rs:28-31`

## Error Handling

**Primary pattern — `map_err` to domain error:**
```rust
std::fs::write(path, json).map_err(|e| CrabInferError::StorageError {
    reason: format!("Failed to write file: {}", e),
})?;
```

**Pattern — `unwrap()` only on RwLock guards (infallible in practice):**
```rust
manager().keys.read().unwrap().get(provider).cloned()
```
Used in `credentials.rs` for `RwLockReadGuard` operations where poison is not expected.

**Pattern — `Result` propagation with `?`:**
All public functions that can fail return `Result<T, CrabInferError>`. The `?` operator is used throughout to propagate errors.

**Error type:**
- Single `CrabInferError` enum in `crabinfer-core/src/lib.rs` using `thiserror::Error`
- Each variant carries a descriptive, actionable error message with recovery hints
- Example: `ModelTooLarge { file_size_mb: u64, max_allowed_mb: u64 }`, `AuthenticationFailed { provider: String }`
- All variants are `#[uniffi(flat_error)]`-compatible for Swift FFI

**Panics:**
- `catch_unwind(AssertUnwindSafe(...))` used in `engine.rs` to catch candle panics at the Metal inference boundary
- `todo!()` and `unimplemented!()` are absent; the codebase does not use panic-on-unimplemented stubs
- `expect()` is rare and reserved for developer-facing invariants within test helpers

## Logging

**Framework:** `tracing` crate (`tracing = "0.1"`)

**Production logging (serving engine):**
- `tracing::info!()` — startup, engine milestones, CUDA graph captures
- `tracing::warn!()` — non-fatal failures (swap buffer allocation, draft sync errors)
- `tracing::debug!()` — per-request scheduling, CUDA graph lazy captures
- `tracing::error!()` — forward pass failures
- Used extensively in `crabinfer-core/src/serving/engine_loop.rs`

**Legacy logging (core engine / MCP):**
- Custom `log_debug!` macro in `engine.rs` uses `write!(std::io::stderr(), ...)` with silent failure — required because iOS stderr is unreliable
- `eprintln!()` used in `mcp.rs` for MCP server notification errors (lines 588, 596)

**No logging in:**
- Provider implementations, router, tools, vectorstore — they rely on `Result` propagation instead

## Comments

**Module-level docs:**
Every module begins with a `//!` doc comment describing purpose, design rationale, and sometimes algorithm references. Example:
```rust
//! Block pool: physical KV cache block allocation with prefix caching.
//!
//! Manages a fixed-size pool of physical blocks, providing O(1) allocation,
//! O(1) freeing, and content-hash-based prefix caching with LRU eviction.
//!
//! Design follows vLLM V1's BlockPool, simplified for unified memory
```

**Inline section comments:**
Long files use comment banners to demarcate sections:
```rust
// ---------------------------------------------------------------------------
// Types (UniFFI-exported)
// ---------------------------------------------------------------------------
```
or box-drawing style:
```rust
// ─── Configuration ───────────────────────────────────────────────────────
```

**Field-level doc comments:**
Public struct fields carry `///` doc comments explaining units, constraints, and behavior — especially for config structs. Example from `ServingEngineConfig` and `SamplingParams`.

**Algorithm references:**
Algorithm-heavy modules cite papers with author/year:
- `speculative.rs`: "Leviathan et al. 2023"
- `ngram_draft.rs`: "REST: Retrieval-Based Speculative Decoding (He et al. 2023)"

**TODO comments:**
Very sparse — only 3 found in the entire codebase:
- `engine_loop.rs:1726`: `// TODO: logprobs for speculative tokens`
- `engine_loop.rs:1761`: same
- `attention.rs:120`: prefix cache comment

## Function Design

**Size:** Functions are focused; large files (e.g. `engine_loop.rs` at 2613 lines) are broken into well-named private helpers. The public API surface per function is small.

**Parameters:** Config structs are preferred over long parameter lists. Constructors take a single `*Config` struct. Builder methods allow optional configuration.

**Return Values:**
- Fallible operations return `Result<T, CrabInferError>`
- Infallible operations return `T` or `Option<T>`
- `Option<T>` for lookups that may have no result: `fn get(&self, name: &str) -> Option<&Arc<dyn Tool>>`

## Module Design

**Exports:**
- `pub` for types and functions consumed by other crates or FFI
- `pub(crate)` is occasionally used for internal cross-module items
- `pub` fields in structs (no getters/setters pattern — direct field access)

**Trait implementations:**
- `Default` implemented on config structs with sensible values: `SpeculativeConfig::default()`, `NgramDraftConfig::default()`, `SamplingParams::default()`
- `Display` and `FromStr` implemented on enum types exposed at API boundary (`QuantizationMethod`)
- `Send + Sync` bounds on trait objects: `pub trait Provider: Send + Sync`, `pub trait Tool: Send + Sync`

**UniFFI pattern:**
Types crossing the Swift FFI boundary derive `uniffi::Record` (structs), `uniffi::Enum` (enums), or `uniffi::Object` (opaque handles). Public methods on `uniffi::Object` types are grouped under `#[uniffi::export] impl TypeName { ... }`.

---

*Convention analysis: 2026-03-12*
