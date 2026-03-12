# Codebase Concerns

**Analysis Date:** 2026-03-12

## Tech Debt

**Serving Engine Supports Only Llama Architecture:**
- Issue: `ARCHITECTURE_REGISTRY` in `crabinfer-core/src/serving/models/mod.rs` contains a single entry (`"llama"`). Mistral, Phi3, Gemma, Qwen2, and GPT2 architectures are documented in comments and the `LlamaModel` code paths handle them implicitly via identical attention layouts, but none are registered by name.
- Files: `crabinfer-core/src/serving/models/mod.rs` lines 126-129
- Impact: Any GGUF with `general.architecture != "llama"` (e.g., `"mistral"`, `"qwen2"`, `"phi3"`) fails to load in the serving engine with an `UnsupportedArchitecture` error even though the actual model code would handle them correctly.
- Fix approach: Add architecture aliases to `ARCHITECTURE_REGISTRY`, mapping `"mistral"`, `"qwen2"`, `"phi3"`, `"gemma"` to `llama_factory`.

**N-gram Draft Module Not Wired to Engine Loop:**
- Issue: `crabinfer-core/src/serving/ngram_draft.rs` implements `NgramDrafter` with full tests, but it is never instantiated or called in `crabinfer-core/src/serving/engine_loop.rs`. There is no `NgramDraftConfig` field in `ServingEngineConfig` and no code path invoking `NgramDrafter::draft_tokens`.
- Files: `crabinfer-core/src/serving/ngram_draft.rs`, `crabinfer-core/src/serving/engine_loop.rs`
- Impact: Speculative decoding with n-gram drafting (faster than a draft model, zero memory overhead) is documented and tested but silently unavailable at runtime.
- Fix approach: Add `ngram_draft: Option<NgramDraftConfig>` to `ServingEngineConfig`; integrate `NgramDrafter` alongside `SpeculativeState` in the engine loop's speculative decoding path.

**Legacy Engine Is a Concurrency Bottleneck:**
- Issue: The legacy `CrabInferEngine` (GGUF/Metal path) serializes all requests behind `AppState::inference_lock` (a `tokio::sync::Mutex`). Each token generation step is dispatched with `tokio::task::spawn_blocking`. The full streaming path loops one `spawn_blocking` call per token.
- Files: `crabinfer-server/src/state.rs:13`, `crabinfer-server/src/routes/openai.rs:56`, `crabinfer-server/src/routes/anthropic.rs:50`
- Impact: One concurrent inference request at a time for the legacy engine, regardless of hardware parallelism. All other requests queue behind the lock.
- Fix approach: Migrate all production use to the PagedAttention serving engine. The legacy engine should be sunset or restricted to single-user embedded use (Swift/Node bindings).

**Logprobs Unavailable for Speculative Tokens:**
- Issue: Two `TODO` comments in `crabinfer-core/src/serving/engine_loop.rs` (lines 1726, 1761) mark `logprob: None` for speculative tokens. When speculative decoding is active, every accepted draft token and the final token report `null` logprob even when the caller requested `logprobs: true`.
- Files: `crabinfer-core/src/serving/engine_loop.rs:1726`, `crabinfer-core/src/serving/engine_loop.rs:1761`
- Impact: Clients relying on logprobs for confidence scoring, watermarking, or structured decoding receive incomplete data when speculative decoding is active.
- Fix approach: Collect and pass draft token logprobs from the acceptance step in `crabinfer-core/src/serving/speculative.rs` back to `engine_loop.rs`.

**Prefix Cache Miss for Prefill with Cached Prefixes:**
- Issue: A `TODO` in `crabinfer-core/src/serving/models/attention.rs` (lines 120-123) explains that when a sequence has prefix cache hits during prefill, the attention only covers new tokens and misses cross-attention with the cached K/V blocks in the paged cache.
- Files: `crabinfer-core/src/serving/models/attention.rs:120-124`
- Impact: Prefix caching memory savings are achieved but the quality benefit (attending to full context) is not fully realized during prefill phases. This causes incorrect attention for multi-turn conversations with long shared prefixes.
- Fix approach: Implement per-token paged attention for prefill, or read cached K/V from paged cache blocks during prefill attention computation.

**Local Candle Fork as Build Dependency:**
- Issue: `Cargo.toml` workspace patches `candle-core`, `candle-nn`, and `candle-transformers` to `../candle/` (a sibling directory). The declared dependency in `crabinfer-core/Cargo.toml` points to `github.com/lspecian/candle` on branch `ios-metal-fix`, not the upstream `huggingface/candle`.
- Files: `Cargo.toml:12-15`, `crabinfer-core/Cargo.toml:23-27`
- Impact: The repository does not build without a separately cloned `../candle/` directory. CI, contributors, and downstream users all require a manual out-of-band setup step. The fork diverges from upstream, making security patches and feature updates manual merges.
- Fix approach: Land the RoPE + GQA fixes upstream or vendor the patched files directly. Document the setup requirement prominently; add a build script or git submodule.

**Static Model Catalog May Become Stale:**
- Issue: `crabinfer-core/src/catalog.rs` embeds a static list of GGUF models compiled into the binary. New model releases, URL changes, or SHA256 updates require a code change and binary rebuild.
- Files: `crabinfer-core/src/catalog.rs:149`
- Impact: Users cannot discover or download models released after the last binary build. Stale download URLs silently fail at runtime.
- Fix approach: Supplement the static catalog with an optional remote JSON catalog fetched at runtime (with a TTL and offline fallback).

**Histogram Buckets Are Non-Cumulative as Stored:**
- Issue: `crabinfer-server/src/state.rs` stores per-bucket counts as individual non-cumulative values (`bucket_counts[i]`). The `to_prometheus` method correctly accumulates them when emitting, but any code reading raw `bucket_counts` directly would see incorrect non-cumulative values.
- Files: `crabinfer-server/src/state.rs:74-81`
- Impact: Low risk currently (only `to_prometheus` reads buckets), but the data model is misleading and could cause bugs if histogram values are inspected outside `to_prometheus`.
- Fix approach: Either store cumulative counts or add a comment clarifying that cumulation happens only at emit time.

## Security Considerations

**No Authentication on HTTP Server:**
- Risk: The server exposes `/v1/chat/completions`, `/v1/messages`, and `/metrics` with `CorsLayer::permissive()` and zero authentication. Any process or browser tab with network access can submit inference requests and read metrics.
- Files: `crabinfer-server/src/routes/mod.rs:27`
- Current mitigation: Body size is capped at 1MB (`MAX_REQUEST_BODY_BYTES`). The server is intended for local use, but there is no enforcement of that scope.
- Recommendations: Add optional API key authentication via an `Authorization: Bearer` header check at the router level, controlled by a `CRABINFER_API_KEY` env var. Also add a `--host 127.0.0.1` default to prevent accidental exposure on network interfaces.

**Shell Execution Tool Has Weak Blocklist:**
- Risk: `ShellExecTool` in `crabinfer-core/src/tools.rs` passes the user-supplied command string directly to `sh -c` after a substring blocklist check. The blocklist (`BLOCKED_COMMANDS`) uses 12 hardcoded strings and is trivially bypassed (e.g., `rm  -rf /`, base64-encoded commands, environment variable substitution, process substitution).
- Files: `crabinfer-core/src/tools.rs:321-368`
- Current mitigation: None beyond the blocklist.
- Recommendations: Replace the blocklist with an allowlist of permitted commands, or add a configurable sandbox mode (e.g., disable `shell_exec` by default, require explicit opt-in). At minimum, add `--no-profile`, `--restricted` flags and redirect stdin from `/dev/null`.

**File Read/Write Tools Have No Path Restriction:**
- Risk: `FileReadTool` and `FileWriteTool` in `crabinfer-core/src/tools.rs` accept any path string (absolute or relative) with no `canonicalize`, no chroot, and no allowed-path allowlist. An AI agent can read `/etc/passwd`, SSH private keys, or `.env` files; it can write to arbitrary locations the process user owns.
- Files: `crabinfer-core/src/tools.rs:170-238`
- Current mitigation: None.
- Recommendations: Add a configurable `working_dir` restriction; resolve the path with `std::fs::canonicalize` and reject paths that escape the allowed root.

**API Keys Stored in Plain Memory:**
- Risk: `CredentialManager` stores API keys as `String` values in a `HashMap` inside a global `RwLock`. Keys remain in process heap memory for the process lifetime with no zeroing on removal.
- Files: `crabinfer-core/src/credentials.rs:16-26`
- Current mitigation: `remove_api_key` drops the `String`, but Rust's default allocator does not zero memory on drop.
- Recommendations: Use `secrecy::Secret<String>` or a zeroing string type (`zeroize` crate) for sensitive key storage. This is especially relevant for the iOS/macOS deployment target where other processes could potentially read process memory.

**`CorsLayer::permissive()` Enables Cross-Origin Access:**
- Risk: Any webpage can make cross-origin requests to the inference server when it's running on a known port.
- Files: `crabinfer-server/src/routes/mod.rs:27`
- Current mitigation: The 1MB body limit prevents very large payloads.
- Recommendations: Restrict CORS to `localhost` origins only, or make it configurable. `CorsLayer::permissive()` is appropriate for development but not production or shared network deployments.

## Performance Bottlenecks

**KV Cache Block Estimation Ignores Actual GPU Memory:**
- Problem: `estimate_kv_cache_blocks` in `crabinfer-server/src/lib.rs` calculates a budget from total system RAM (25%, capped at 8GB) on non-CUDA builds. It does not subtract the model weight footprint already loaded into memory.
- Files: `crabinfer-server/src/lib.rs:639-681`
- Cause: GPU memory query (`gpu_memory.rs`) is only accurate on CUDA; on Metal/CPU the function falls back to total system RAM.
- Improvement path: Measure model weight size in bytes after loading (available from GGUF metadata) and subtract it from the budget before computing block count.

**Legacy Streaming: One `spawn_blocking` Per Token:**
- Problem: The legacy engine streaming path in `crabinfer-server/src/routes/openai.rs:356` calls `tokio::task::spawn_blocking` for every single token. Each call incurs thread pool overhead and a round-trip through the Tokio scheduler.
- Files: `crabinfer-server/src/routes/openai.rs:356-415`
- Cause: `CrabInferEngine::next_token` is synchronous (CPU/Metal model inference). The streaming loop cannot amortize thread pool scheduling across tokens.
- Improvement path: Move the entire generation loop inside a single `spawn_blocking` call and stream results via an `mpsc` channel, as the serving engine already does.

**`logits.clone()` in Hot Path:**
- Problem: In `crabinfer-core/src/serving/engine_loop.rs:815`, `logits.clone()` is called unconditionally during CUDA graph batch dispatch. Cloning a GPU tensor allocates and copies device memory.
- Files: `crabinfer-core/src/serving/engine_loop.rs:815`
- Cause: CUDA graph replay requires output buffers to be pre-allocated graph buffers; a clone is used to extract the result without aliasing the graph buffer.
- Improvement path: Use a dedicated output staging buffer allocated at graph capture time and copy only the needed slice.

## Fragile Areas

**Speculative Decoding + Prefix Cache Interaction:**
- Files: `crabinfer-core/src/serving/speculative.rs`, `crabinfer-core/src/serving/scheduler.rs`, `crabinfer-core/src/serving/engine_loop.rs`
- Why fragile: The speculative engine (`SpeculativeState`) maintains separate draft model KV caches (`draft_kv_manager`, `draft_blocks`) that must stay in sync with the target model's block assignments. Prefix cache hits in the scheduler affect only target model blocks; draft blocks are not prefix-cached. Any change to the scheduler's prefix cache path risks desync between draft and target block tables.
- Safe modification: Any changes to `Scheduler::lookup_prefix_cache` or block eviction must also account for `SpeculativeState::draft_blocks` cleanup.
- Test coverage: `crabinfer-core/src/serving/speculative.rs` has unit tests for draft/verification logic but no tests for the combined speculative + prefix cache code path.

**`unsafe impl Send/Sync` on CUDA Handles:**
- Files: `crabinfer-core/src/serving/cuda_graphs.rs:155-157`, `crabinfer-core/src/serving/swap.rs:67-68`
- Why fragile: `CudaGraphExecHandle` and `SwapBuffer` manually implement `Send + Sync`. Comments assert single-threaded access, but there is no compile-time or runtime enforcement of this invariant. A future refactor moving graph execution to a different thread would silently become unsound.
- Safe modification: Add a thread-ID assertion in debug builds (`std::thread::current().id()` check) or wrap the handle in a `std::cell::Cell`-like guard.
- Test coverage: None for thread-safety invariants.

**Poison Handling Inconsistency:**
- Files: `crabinfer-core/src/engine.rs:109-111`, `crabinfer-core/src/backends/candle.rs:56-58`, `crabinfer-core/src/credentials.rs:39-60`, `crabinfer-core/src/lib.rs:450-986`
- Why fragile: The engine and candle backend recover from poisoned `Mutex` guards via `into_inner()`, preserving availability. The `CredentialManager` and UniFFI-exposed types in `lib.rs` use bare `.unwrap()` on `lock()`, which panics if the lock is poisoned (e.g., by a panic inside a previous lock holder). A single panic in a token generation loop that holds a credential lock would permanently break the process.
- Safe modification: Replace all `lock().unwrap()` with `lock().unwrap_or_else(|p| p.into_inner())` or restructure to avoid holding locks across fallible operations.

**Metal NaN Workaround Is Model-Specific:**
- Files: `crabinfer-core/src/engine.rs:519-530`, `crabinfer-core/src/engine.rs:723-730`
- Why fragile: The workaround calls `has_nan` on logits and retries with sequential prefill if NaN is detected. This is gated on `needs_sequential_prefill` which checks model-specific metadata. If a new model triggers the same Metal NaN bug without matching the existing detection heuristic, inference silently produces garbage tokens.
- Safe modification: Make `has_nan` detection unconditional (minor performance cost) or expand the detection to any model on Metal with certain attention head configurations.

## Scaling Limits

**Engine Loop Runs on a Single Thread:**
- Current capacity: One CPU/GPU thread per serving engine instance.
- Limit: Throughput is bounded by a single forward pass thread. Tensor parallelism (splitting the model across GPUs) is not implemented.
- Scaling path: The `ModelRunner` trait would need a tensor-parallel variant; the engine loop's batch construction would need to partition and synchronize across devices.

**In-Memory KV Cache Only (When Swap Disabled):**
- Current capacity: KV cache is bounded by GPU/unified memory.
- Limit: With `swap_space: 0.0` (the default), preempted sequences have their KV blocks discarded (recompute-on-resume). With large models on memory-constrained devices, active sequence count drops under load.
- Scaling path: The swap infrastructure (`crabinfer-core/src/serving/swap.rs`) is implemented; the `--swap-space` CLI flag exposes it. Enabling swap by default (e.g., 4GB) would improve throughput under memory pressure.

## Dependencies at Risk

**Pinned to a Personal Candle Fork:**
- Risk: `github.com/lspecian/candle` on branch `ios-metal-fix` is a personal fork of `huggingface/candle`. If the fork becomes abandoned, diverges significantly from upstream, or if the author's account changes, builds break with no upstream fallback.
- Impact: Core inference (Metal GPU, RoPE, GQA) depends entirely on this fork.
- Migration plan: Upstream the RoPE + GQA fixes to `huggingface/candle` or vendor the modified files inside the repository to eliminate the external fork dependency.

## Missing Critical Features

**No `/v1/completions` (Text Completions) Endpoint:**
- Problem: The OpenAI-compatible server only implements `/v1/chat/completions`. The raw text completion endpoint (`POST /v1/completions`) is absent.
- Blocks: Integration with clients that use legacy completions API (some code generation tools, LangChain, and others).

**No `/v1/embeddings` Endpoint:**
- Problem: There is no embeddings endpoint. The RAG pipeline (`crabinfer-core/src/embedding.rs`) provides local embeddings but they are not exposed via the HTTP server.
- Blocks: Standard LLM application stacks that use the same server for both chat and embeddings (OpenAI-compatible clients, LlamaIndex, LangChain).

**No Rate Limiting:**
- Problem: No per-IP or per-API-key rate limiting exists anywhere in the server. The only protection is the 1MB body cap.
- Blocks: Production or shared deployments where a single client could exhaust GPU resources.

## Test Coverage Gaps

**All Provider Integration Tests Are Ignored by Default:**
- What's not tested: Every integration test in `crabinfer-core/tests/provider_integration.rs` is annotated `#[ignore]`. OpenAI, Anthropic, Google, Ollama, vLLM, router policy, and streaming provider tests never run in CI.
- Files: `crabinfer-core/tests/provider_integration.rs:94-448`
- Risk: Provider API changes, routing regressions, and streaming bugs go undetected until user-reported.
- Priority: Medium — add a CI stage gated on secrets availability to run a subset of provider tests.

**No Tests for Speculative Decoding + Prefix Cache Interaction:**
- What's not tested: The combined code path where prefix cache hits occur on sequences that are also under speculative decoding.
- Files: `crabinfer-core/src/serving/engine_loop.rs`, `crabinfer-core/src/serving/speculative.rs`, `crabinfer-core/src/serving/scheduler.rs`
- Risk: Silent block table corruption leading to incorrect tokens or panics.
- Priority: High.

**No Tests for File Tool Path Traversal:**
- What's not tested: `FileReadTool` and `FileWriteTool` have no tests verifying that `../` path traversal attempts are rejected.
- Files: `crabinfer-core/src/tools.rs:170-238`
- Risk: A malicious or misconfigured agent tool call reads sensitive files outside the working directory.
- Priority: High (security).

**No End-to-End Server Tests:**
- What's not tested: No tests start the Axum server and send actual HTTP requests to `/v1/chat/completions` or `/v1/messages`. Route handler tests are unit tests only.
- Files: `crabinfer-server/src/routes/`
- Risk: Route wiring, middleware stacking, and SSE streaming bugs are not caught.
- Priority: Medium.

---

*Concerns audit: 2026-03-12*
