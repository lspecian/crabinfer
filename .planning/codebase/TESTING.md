# Testing Patterns

**Analysis Date:** 2026-03-12

## Test Framework

**Runner:**
- Rust's built-in `cargo test` (no external test runner)
- No `jest.config.*` or `vitest.config.*` — all tests are Rust

**Assertion Library:**
- Standard `assert!`, `assert_eq!`, `assert_ne!` macros
- `matches!` macro for pattern matching on error variants: `assert!(matches!(err, CrabInferError::AuthenticationFailed { .. }))`

**Run Commands:**
```bash
cargo test --workspace                          # Run all unit tests across all crates
cargo test -p crabinfer-core                    # Run only core crate tests
cargo test -p crabinfer-core -- --nocapture     # Show println! output from tests
cargo test --ignored --test provider_integration -- --nocapture  # Run integration tests
cargo test --workspace --features providers     # Include provider-gated tests
```

## Test File Organization

**Location:**
- **Inline unit tests**: co-located with source in `#[cfg(test)] mod tests { ... }` blocks at the bottom of each `.rs` file
- **Integration tests**: separate file at `crabinfer-core/tests/provider_integration.rs`

**Naming:**
- Test functions: `fn test_<what_it_tests>()` — descriptive snake case
- Test helper functions: named `test_config()`, `mock_local()`, `simple_request()`, `make_metadata()` — no `test_` prefix
- Integration test file: `provider_integration.rs`

**Structure (unit tests at bottom of source file):**
```rust
// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    // optional: use crate::some_other::Type;

    fn helper_fixture() -> SomeType { ... }

    #[test]
    fn test_behavior_under_condition() {
        // arrange
        // act
        // assert
    }
}
```

## Test Structure

**Suite Organization:**

Unit test modules in `#[cfg(test)] mod tests` always start with `use super::*;` to import everything from the parent module.

Integration tests are grouped with section comments:
```rust
// =========================================================================
// 1. Provider availability checks
// =========================================================================

// =========================================================================
// 2. Cloud provider completions
// =========================================================================
```

**Fixture pattern — config builder helpers:**
```rust
fn test_config(num_blocks: usize) -> BlockPoolConfig {
    BlockPoolConfig {
        block_size: 16,
        num_blocks,
        enable_prefix_cache: true,
    }
}

fn mock_local(available: bool, should_fail: bool) -> Box<dyn Provider> {
    Box::new(MockProvider { provider_name: "local", available, should_fail })
}

fn test_request() -> CompletionRequest {
    CompletionRequest {
        model: "test".to_string(),
        messages: vec![ChatMessage { role: "user".to_string(), content: "hello".to_string() }],
        max_tokens: 100,
        temperature: 0.7,
        ...
    }
}
```

**Assertion style:**
```rust
assert_eq!(resp.provider_name, "local");
assert_eq!(d.reason, RoutingReason::LocalAvailable);
assert!(d.is_local);
assert!(matches!(router.complete(&test_request()).unwrap_err(), CrabInferError::ProviderNotAvailable { .. }));
```

## Mocking

**Framework:** Manual mock structs (no `mockall` or similar crate)

**Pattern — implementing trait on a struct:**
```rust
struct MockProvider {
    provider_name: &'static str,
    available: bool,
    should_fail: bool,
}

impl Provider for MockProvider {
    fn name(&self) -> &str { self.provider_name }
    fn complete(&self, _request: &CompletionRequest) -> Result<CompletionResponse, CrabInferError> {
        if self.should_fail {
            return Err(CrabInferError::NetworkError { reason: "mock failure".to_string() });
        }
        Ok(CompletionResponse {
            content: format!("from {}", self.provider_name),
            model: "mock-model".to_string(),
            provider_name: self.provider_name.to_string(),
            stop_reason: "end_turn".to_string(),
            input_tokens: 10,
            output_tokens: 5,
            routing_info: String::new(),
        })
    }
    fn is_available(&self) -> bool { self.available }
    // ...
}
```

This pattern is used in `router.rs` test module for `MockProvider`, and in `serving/` modules for `ModelRunner`.

**What to Mock:**
- External trait implementations (Provider, Tool, ModelRunner)
- Network I/O: pass `should_fail: bool` to mock errors
- Anything that requires real infrastructure (APIs, filesystem in some tests)

**What NOT to Mock:**
- Pure logic (parsers, schedulers, vector math) — tested with real implementations
- Config structs and value types — constructed directly in tests

## Fixtures and Factories

**Test Data:**
```rust
// Metadata fixture (vectorstore tests)
fn make_metadata(source: &str, idx: usize) -> ChunkMetadata {
    ChunkMetadata {
        source: source.to_string(),
        chunk_index: idx,
        start_offset: 0,
    }
}

// Integration test request builder
fn simple_request(model: &str, prompt: &str) -> CompletionRequest {
    CompletionRequest {
        model: model.to_string(),
        messages: vec![ChatMessage { role: "user".to_string(), content: prompt.to_string() }],
        max_tokens: 50,
        temperature: 0.3,
        ...
    }
}
```

**Location:**
- Fixtures are defined as private `fn` helpers inside `mod tests` within the same file
- No shared fixture directory or external fixture files
- Integration test helpers defined at the top of `crabinfer-core/tests/provider_integration.rs`

**Temporary Files in Tests:**
- Use `std::env::temp_dir()` for persistence tests: `crabinfer-core/src/facts.rs` and `vectorstore.rs`
- `tempfile` crate is NOT used — temp paths constructed manually

## Coverage

**Requirements:** None enforced — no coverage thresholds configured

**View Coverage:**
```bash
cargo llvm-cov --workspace  # if cargo-llvm-cov is installed
```

**Observed coverage by area:**
- Router logic: very high — all 5 routing policies and edge cases tested
- Scheduler / block pool / KV cache: high — 273 tests in `serving/` alone
- Vector store / facts / conversation: good — persistence round-trips tested
- Provider implementations (OpenAI, Anthropic, Google, Ollama): integration tests only (`#[ignore]`) — no unit test coverage without live keys
- Engine inference (GGUF loading, Metal forward pass): no tests — requires model files

## Test Types

**Unit Tests:**
- Scope: pure logic, data structures, parsers, config validation
- Location: inline `#[cfg(test)] mod tests` in each source file
- Run by default with `cargo test`
- Count: ~471 test functions across `crabinfer-core/src/`

**Integration Tests:**
- Scope: real HTTP calls to cloud provider APIs (OpenAI, Anthropic, Google, Ollama)
- Location: `crabinfer-core/tests/provider_integration.rs`
- All marked `#[ignore]` — require `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` env vars
- Run with: `cargo test --ignored --test provider_integration`

**E2E Tests:**
- Electron demo has Playwright tests at `examples/electron-demo/e2e/`
- Not part of the Rust test suite

## Common Patterns

**Async Testing:**
- Serving engine tests are synchronous (single-threaded Scheduler, BlockPool)
- No `#[tokio::test]` or async test functions found in unit tests
- Integration tests are blocking (use `reqwest::blocking`)

**Error Testing:**
```rust
// Check error type using matches!
assert!(matches!(
    router.complete(&test_request()).unwrap_err(),
    CrabInferError::ProviderNotAvailable { .. }
));

// Check that a call returns error
let result = router.complete(&request);
assert!(result.is_err(), "Should fail when data_sovereignty=true and only cloud providers");

// Unwrap with print for integration tests
let result = provider.complete(&request).unwrap();
assert!(!result.content.is_empty());
```

**Persistence Round-Trip Testing:**
Tests that write to disk use `std::env::temp_dir()` and clean up by relying on temp path uniqueness:
```rust
let path = std::env::temp_dir().join("test_store.json");
let path_str = path.to_str().unwrap();
store.save().unwrap();
let loaded = MemoryStore::load(path_str).unwrap();
assert_eq!(loaded.len(), 2);
```

**State Mutation Testing:**
Scheduler and block pool tests call methods step-by-step and assert intermediate state:
```rust
let mut pool = BlockPool::new(test_config(4));
assert_eq!(pool.num_free_blocks(), 4);
let blocks = pool.allocate(2).unwrap();
assert_eq!(pool.num_free_blocks(), 2);
pool.free(&blocks);
assert_eq!(pool.num_free_blocks(), 4);
```

**Integration Test Output:**
Integration tests use `println!` with `[TEST]`/`[PASS]`/`[INFO]` prefixes and require `-- --nocapture` to see output. This pattern is only in `provider_integration.rs`.

---

*Testing analysis: 2026-03-12*
