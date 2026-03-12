# Architecture

**Analysis Date:** 2026-03-12

## Pattern Overview

**Overall:** Layered SDK + Server monorepo with FFI boundary abstraction

**Key Characteristics:**
- `crabinfer-core` is the single source of truth for all inference logic; all other crates are thin consumption layers over it
- FFI boundaries are cleanly isolated: UniFFI scaffolding for Swift, napi-rs for Node.js — both expose the same API surface
- Two inference engines coexist: a legacy single-request GGUF engine and a new PagedAttention continuous-batching engine selectable at runtime
- The `Provider` trait abstracts local and cloud backends behind a single interface, enabling transparent routing
- Feature flags (`metal`, `providers`) gate platform-specific and cloud-dependent code at compile time

## Layers

**Core Library (`crabinfer-core`):**
- Purpose: All inference logic, provider abstractions, routing, agent runtime, RAG pipeline
- Location: `crabinfer-core/src/`
- Contains: Engine, router, agent, provider trait + implementations, MCP client/server, knowledge/RAG, serving engine
- Depends on: candle (local fork at `../candle/`), tokenizers, serde, uniffi
- Used by: `crabinfer-server`, `crabinfer-cli`, `crabinfer-node`, `crabinfer-swift`

**HTTP Server (`crabinfer-server`):**
- Purpose: OpenAI-compatible and Anthropic-compatible REST API server
- Location: `crabinfer-server/src/`
- Contains: Axum routes, AppState, ServerMetrics (Prometheus histograms), chat template resolution
- Depends on: `crabinfer-core`, axum, tokio, tower-http, tokenizers
- Used by: external HTTP clients (OpenAI SDK, curl, python-client example)

**CLI (`crabinfer-cli`):**
- Purpose: Command-line interface for all operations: inference, chat REPL, auth, model downloads, MCP management, serving
- Location: `crabinfer-cli/src/`
- Contains: `main.rs` with Clap parser, individual `cmd_*.rs` modules per subcommand
- Depends on: `crabinfer-core`, `crabinfer-server`, clap, tokio, tracing-subscriber
- Used by: end users via terminal

**Node.js Binding (`crabinfer-node`):**
- Purpose: napi-rs bindings exposing the same API surface as Swift SDK to Node.js/Electron
- Location: `crabinfer-node/src/`
- Contains: Parallel module structure to core (engine, router, provider, agent, knowledge, memory, stream, vllm, download)
- Depends on: `crabinfer-core`, napi, napi-derive
- Used by: Node.js applications, Electron desktop apps

**Swift Binding (`crabinfer-swift`):**
- Purpose: UniFFI-generated Swift wrapper over `crabinfer-core`
- Location: `crabinfer-swift/Sources/CrabInfer/`
- Contains: `CrabInfer.swift` (UniFFI-generated bindings), `Discovery.swift` (Bonjour server discovery)
- Depends on: UniFFI scaffolding generated from `crabinfer-core/src/crabinfer.udl`
- Used by: iOS/macOS Swift apps

**Serving Subsystem (`crabinfer-core/src/serving/`):**
- Purpose: Production-grade PagedAttention continuous-batching engine
- Location: `crabinfer-core/src/serving/`
- Contains: Block pool, KV cache, scheduler, sequence management, model runners, speculative decoding, CUDA graphs, quantization, safetensors loader, CPU swap
- Depends on: candle-core, platform kernel backends
- Used by: `crabinfer-server` when `--serving` flag is active

## Data Flow

**Legacy Single-Request Inference:**

1. Client sends HTTP POST `/v1/chat/completions` to `crabinfer-server`
2. Axum handler in `crabinfer-server/src/routes/openai.rs` acquires `AppState.inference_lock` (serializes requests)
3. Chat messages formatted via `crabinfer-server/src/chat_template.rs`
4. `CrabInferEngine.complete()` called from `crabinfer-core/src/engine.rs`
5. Candle runs GGUF model forward pass on Metal GPU or CPU
6. Tokens decoded and streamed (SSE) or collected into full response
7. `CompletionResponse` serialized to OpenAI JSON format and returned

**PagedAttention Serving Flow:**

1. HTTP handler submits request to `EngineHandle` channel (`crabinfer-core/src/serving/engine_loop.rs`)
2. Engine loop thread drains submission channel, calls `Scheduler.schedule()` (`crabinfer-core/src/serving/scheduler.rs`)
3. Scheduler allocates KV cache blocks from `BlockPool` (`crabinfer-core/src/serving/block_pool.rs`)
4. Batched inputs constructed; `ModelRunner.forward()` called with `ForwardContext`
5. Optional speculative decoding via `SpeculativeState` (`crabinfer-core/src/serving/speculative.rs`) adds draft tokens
6. Sampled tokens sent to per-request `tokio::sync::mpsc` channels
7. HTTP handler reads tokens from channel and streams SSE to client

**Agent Tool-Calling Loop:**

1. User input arrives at `CrabInferAgent.run()` in `crabinfer-core/src/lib.rs`
2. `Agent.run()` in `crabinfer-core/src/agent.rs` builds prompt: system + facts + knowledge context + conversation + tools
3. `Provider.complete()` called; response parsed for JSON tool call blocks
4. If tool calls found: `ToolRegistry.execute()` dispatches to built-in tools or MCP client
5. Tool outputs appended to conversation as assistant messages; loop repeats
6. When LLM returns text-only response: `AgentResponse` returned with text + tool execution trace

**Provider Routing:**

1. `CrabInferRouter` implements the `Provider` trait (`crabinfer-core/src/router.rs`)
2. `Router.complete()` evaluates: routing policy, memory pressure (`quick_check_pressure()`), network availability
3. Decision recorded as `RoutingDecision` with `RoutingReason`
4. Selected provider's `complete()` called; on error, fallback provider tried
5. `CompletionResponse.routing_info` carries the decision metadata back to caller

**State Management:**
- `CrabInferEngine` uses interior mutability (`Mutex<InferenceState>`) for thread-safe token generation
- `CrabInferAgent` wraps `Agent` in a `Mutex` for single-owner access
- `AppState` (server) uses `Arc<AppState>` cloned into each Axum handler
- Streaming state in `CrabInferProvider`/`CrabInferRouter` stored as `Mutex<Option<Box<dyn Iterator>>>` — clients call `stream_next()` in a poll loop

## Key Abstractions

**`Provider` trait (`crabinfer-core/src/provider.rs`):**
- Purpose: Unified interface for local and cloud LLM inference
- Examples: `crabinfer-core/src/providers/local.rs`, `crabinfer-core/src/providers/openai.rs`, `crabinfer-core/src/providers/anthropic.rs`, `crabinfer-core/src/providers/google.rs`, `crabinfer-core/src/providers/ollama.rs`, `crabinfer-core/src/providers/vllm.rs`
- Pattern: `fn complete(&self, request: &CompletionRequest) -> Result<CompletionResponse>` + `fn stream(...)` + `fn available_models()`

**`ModelRunner` trait (`crabinfer-core/src/serving/models/mod.rs`):**
- Purpose: PagedAttention-native model forward pass interface
- Examples: `crabinfer-core/src/serving/models/llama.rs`, `crabinfer-core/src/serving/models/attention.rs`
- Pattern: `fn forward(&self, ctx: &ForwardContext) -> Result<Tensor>` with paged KV cache context

**`Tool` trait (`crabinfer-core/src/tools.rs`):**
- Purpose: Agent-callable tools with JSON schema for LLM function calling
- Pattern: Tools registered in `ToolRegistry`, dispatched by name with JSON arguments

**`EmbeddingProvider` trait (`crabinfer-core/src/embedding.rs`):**
- Purpose: Pluggable embedding backends for the RAG pipeline
- Pattern: `fn embed(&self, text: &str) -> Vec<f32>`; default implementation is TF-IDF

**`Backend` trait (`crabinfer-core/src/backend.rs`):**
- Purpose: ML runtime abstraction (currently only `CandleBackend` in `crabinfer-core/src/backends/candle.rs`)
- Pattern: Thin wrapper over Candle device/tensor operations

## Entry Points

**HTTP Server:**
- Location: `crabinfer-server/src/main.rs` → `crabinfer-server/src/lib.rs::run_server()`
- Triggers: `crabinfer serve --model <path>` CLI command, or direct binary execution
- Responsibilities: Load model (legacy or serving engine), bind Axum router, serve until SIGINT/SIGTERM with graceful drain

**CLI Binary:**
- Location: `crabinfer-cli/src/main.rs`
- Triggers: User runs `crabinfer <subcommand>`
- Responsibilities: Parse args with Clap, dispatch to `cmd_*.rs` handler, delegate to `crabinfer-core` or `crabinfer-server`

**UniFFI FFI Entry (Swift):**
- Location: `crabinfer-core/src/lib.rs` (all `#[uniffi::export]` items)
- Triggers: Swift app instantiates `CrabInferEngine`, `CrabInferProvider`, `CrabInferRouter`, or `CrabInferAgent`
- Responsibilities: Thin wrappers delegating to inner Rust types; UniFFI generates the Swift/Rust glue

**napi-rs Entry (Node.js):**
- Location: `crabinfer-node/src/lib.rs`
- Triggers: `require('crabinfer')` or `import from 'crabinfer'` in Node.js
- Responsibilities: Exposes same object surface as Swift — Engine, Provider, Router, Agent, KnowledgeBase, etc.

**Serving Engine Loop:**
- Location: `crabinfer-core/src/serving/engine_loop.rs::EngineHandle::start()`
- Triggers: `run_server()` when `config.serving == true`
- Responsibilities: Spawns dedicated thread for continuous batching; returns `EngineHandle` for async HTTP handler communication

## Error Handling

**Strategy:** `CrabInferError` enum (defined in `crabinfer-core/src/lib.rs`) propagates through `Result<T, CrabInferError>` at all public boundaries; mapped to HTTP status codes in server routes

**Patterns:**
- UniFFI requires `#[uniffi(flat_error)]` — all variants produce string messages via `thiserror`
- Candle panics wrapped in `std::panic::catch_unwind` inside `engine.rs` to prevent FFI boundary crossing
- Server routes return `(StatusCode, Json<ErrorResponse>)` on error; no unwraps in handlers
- `CrabInferError::FallbackAfterError` — router records primary failure and continues to next provider

## Cross-Cutting Concerns

**Logging:** `tracing` crate throughout; `tracing_subscriber::fmt()` initialized in `crabinfer-cli/src/main.rs` and `crabinfer-server/src/main.rs`; `log_debug!` macro in `engine.rs` for iOS-safe stderr writes

**Validation:** `InvalidConfig` error variant; EngineConfig validated at construction; sampling parameters (temperature 0.0–2.0, top_p 0.0–1.0) documented in error messages

**Authentication:** Provider API keys stored via `credentials.rs` (feature-gated `providers`); `set_credential`/`get_credential` exported to Swift/Node; `api_key_override` field in `CompletionRequest` for per-request overrides

**Memory Safety:** `MemoryPressureManager` (`crabinfer-core/src/memory_pressure.rs`) monitors iOS memory pressure levels (Normal/Warning/Critical/Terminal); `Router` checks `quick_check_pressure()` before dispatching to local engine

---

*Architecture analysis: 2026-03-12*
