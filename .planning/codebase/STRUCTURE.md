# Codebase Structure

**Analysis Date:** 2026-03-12

## Directory Layout

```
crabinfer/                          # Workspace root
├── Cargo.toml                      # Workspace manifest + candle patch
├── Cargo.lock
├── build.sh                        # Convenience build script
├── build-metallibs.sh              # Pre-compile Metal .metallib files for iOS
├── Dockerfile                      # Container image for server deployment
├── crabinfer-core/                 # Core library crate (all inference logic)
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs                  # UniFFI exports + top-level types (DeviceInfo, EngineConfig, etc.)
│   │   ├── engine.rs               # Legacy GGUF inference engine (Metal + CPU)
│   │   ├── device.rs               # Device detection + capability recommendations
│   │   ├── memory_pressure.rs      # iOS memory lifecycle (Normal/Warning/Critical/Terminal)
│   │   ├── router.rs               # Smart routing (5 policies, 3 tiers)
│   │   ├── provider.rs             # Provider trait + unified types (CompletionRequest/Response)
│   │   ├── agent.rs                # Agent runtime (tool-calling loop)
│   │   ├── tools.rs                # Tool trait + ToolRegistry + built-in tools
│   │   ├── mcp.rs                  # MCP client (stdio + HTTP) and server
│   │   ├── knowledge.rs            # RAG pipeline (KnowledgeBase)
│   │   ├── vectorstore.rs          # In-process vector store (cosine similarity)
│   │   ├── embedding.rs            # EmbeddingProvider trait + TF-IDF default
│   │   ├── chunker.rs              # Document text chunker
│   │   ├── conversation.rs         # ConversationMemory (multi-turn history)
│   │   ├── facts.rs                # MemoryStore (persistent key-value facts)
│   │   ├── prompt.rs               # SystemPrompt builder
│   │   ├── catalog.rs              # Curated model catalog with device compatibility
│   │   ├── chat_template.rs        # Chat template formatting (chatml, llama3, phi3, gemma)
│   │   ├── backend.rs              # Backend trait (ML runtime abstraction)
│   │   ├── credentials.rs          # API key storage (feature: providers)
│   │   ├── download.rs             # HuggingFace model download manager (feature: providers)
│   │   ├── stress.rs               # Load/unload stress test for memory leak detection
│   │   ├── crabinfer.udl           # UniFFI interface definition language file
│   │   ├── backends/
│   │   │   ├── mod.rs
│   │   │   └── candle.rs           # CandleBackend implementation
│   │   ├── providers/
│   │   │   ├── mod.rs
│   │   │   ├── local.rs            # LocalProvider (wraps CrabInferEngine)
│   │   │   ├── openai.rs           # OpenAI API provider
│   │   │   ├── anthropic.rs        # Anthropic API provider
│   │   │   ├── google.rs           # Google Gemini API provider
│   │   │   ├── ollama.rs           # Ollama self-hosted provider
│   │   │   ├── vllm.rs             # vLLM provider (health, metrics, guided decoding)
│   │   │   └── http_utils.rs       # Shared HTTP utilities for cloud providers
│   │   ├── serving/                # PagedAttention continuous-batching engine
│   │   │   ├── mod.rs
│   │   │   ├── engine_loop.rs      # EngineHandle, ServingEngineConfig, batching loop
│   │   │   ├── scheduler.rs        # Token-budget scheduler
│   │   │   ├── block_pool.rs       # KV cache block allocator with prefix caching
│   │   │   ├── block.rs            # Block types and metadata
│   │   │   ├── kv_cache.rs         # KVCacheConfig and management
│   │   │   ├── sequence.rs         # Sequence/SeqId lifecycle, SamplingParams
│   │   │   ├── speculative.rs      # Speculative decoding with draft model
│   │   │   ├── cuda_graphs.rs      # CUDA graph capture and replay
│   │   │   ├── gpu_memory.rs       # GPU memory profiling
│   │   │   ├── quantization.rs     # QuantizationMethod, KVCacheDType
│   │   │   ├── safetensors_loader.rs # HuggingFace safetensors model loading
│   │   │   ├── swap.rs             # CPU swap space for KV cache blocks
│   │   │   ├── ngram_draft.rs      # N-gram draft model for speculative decoding
│   │   │   ├── models/
│   │   │   │   ├── mod.rs          # ModelRunner trait, ModelConfig, ForwardContext
│   │   │   │   ├── llama.rs        # Llama-style model runner
│   │   │   │   └── attention.rs    # Paged attention implementation
│   │   │   └── kernels/
│   │   │       ├── mod.rs          # KernelBackend trait, BLOCK_SIZE constant
│   │   │       ├── backend.rs      # Backend dispatch
│   │   │       ├── cpu_backend.rs  # CPU kernel fallback
│   │   │       ├── cuda_backend.rs # CUDA kernel dispatch
│   │   │       ├── metal_backend.rs # Metal kernel dispatch
│   │   │       └── paged_attention.cu # CUDA paged attention kernel
│   │   └── bin/                    # Core library standalone binaries (if any)
│   ├── examples/                   # Rust usage examples for the core crate
│   └── tests/                      # Integration tests for core library
├── crabinfer-server/               # HTTP API server crate
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs                 # Server binary entry point
│       ├── lib.rs                  # run_server(), ServerConfig, engine loader functions
│       ├── state.rs                # AppState, ServerMetrics, Histogram
│       ├── chat_template.rs        # Chat template resolution and formatting
│       ├── error.rs                # Server error types
│       ├── routes/
│       │   ├── mod.rs              # Axum router setup (create_router)
│       │   ├── openai.rs           # POST /v1/chat/completions, GET /v1/models
│       │   ├── anthropic.rs        # POST /v1/messages
│       │   └── health.rs           # GET /health, /ready, /metrics
│       └── types/
│           ├── common.rs           # Shared request/response types
│           └── openai.rs           # OpenAI-specific wire types
├── crabinfer-cli/                  # CLI binary crate
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs                 # Clap parser + command dispatch
│       ├── cmd_run.rs              # `crabinfer run` — one-shot inference
│       ├── cmd_chat.rs             # `crabinfer chat` — interactive REPL
│       ├── cmd_serve.rs            # `crabinfer serve` — starts HTTP server
│       ├── cmd_bench.rs            # `crabinfer bench` — CPU vs Metal benchmarks
│       ├── cmd_info.rs             # `crabinfer info` — device capabilities
│       ├── cmd_auth.rs             # `crabinfer auth` — API key management
│       ├── cmd_models.rs           # `crabinfer models` — catalog + downloads
│       ├── cmd_assistant.rs        # `crabinfer assistant` — agent with tool calling
│       └── cmd_mcp.rs              # `crabinfer mcp` — MCP server management
├── crabinfer-node/                 # Node.js binding crate (napi-rs)
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs                  # napi-rs module root
│   │   ├── engine.rs               # CrabInferEngine binding
│   │   ├── provider.rs             # CrabInferProvider binding
│   │   ├── router.rs               # CrabInferRouter binding
│   │   ├── agent.rs                # CrabInferAgent binding
│   │   ├── vllm.rs                 # CrabInferVllm binding
│   │   ├── knowledge.rs            # KnowledgeBase binding
│   │   ├── memory.rs               # ConversationMemory + MemoryStore bindings
│   │   ├── download.rs             # Model download manager binding
│   │   ├── stream.rs               # TokenStream async iterator
│   │   ├── enums.rs                # Shared enum types
│   │   ├── error.rs                # Error type mapping
│   │   ├── functions.rs            # Top-level functions (detect_device, etc.)
│   │   └── types.rs                # Shared record types
│   └── npm/                        # Pre-built native binaries per platform
│       ├── darwin-arm64/
│       ├── darwin-x64/
│       ├── linux-arm64-gnu/
│       └── linux-x64-gnu/
├── crabinfer-swift/                # Swift package (UniFFI-generated bindings)
│   └── Sources/CrabInfer/
│       ├── CrabInfer.swift         # UniFFI-generated Swift bindings
│       └── Discovery.swift         # Bonjour/mDNS server discovery
├── examples/
│   ├── ios-demo/                   # SwiftUI iOS demo app
│   │   └── CrabInferDemo/          # Xcode project sources
│   ├── electron-demo/              # Electron + React menu bar app
│   │   └── src/
│   │       ├── main/               # Electron main process
│   │       ├── preload/            # Preload script (contextBridge)
│   │       └── renderer/           # React UI
│   ├── node-minimal/               # Minimal Node.js usage example
│   └── python-client/              # Python via OpenAI-compatible server
├── docs/
│   ├── guides/                     # Developer guides
│   ├── build/                      # Build artifacts and docs output
│   └── screenshots/                # UI screenshots
├── deploy/                         # Deployment configuration files
├── packaging/
│   └── homebrew/                   # Homebrew formula
├── scripts/                        # Build and utility scripts
└── .github/
    └── workflows/                  # CI/CD GitHub Actions workflows
```

## Directory Purposes

**`crabinfer-core/src/`:**
- Purpose: All inference logic — the only crate with actual ML computation
- Contains: 26 source modules covering engine, routing, providers, agent, RAG, MCP, serving
- Key files: `lib.rs` (FFI exports), `engine.rs` (GGUF inference), `router.rs` (smart routing), `agent.rs` (tool loop), `serving/engine_loop.rs` (continuous batching)

**`crabinfer-core/src/serving/`:**
- Purpose: Production-grade PagedAttention engine separate from the legacy engine
- Contains: 13+ modules for block management, scheduling, model runners, speculative decoding, hardware kernels
- Key files: `engine_loop.rs` (the batching loop), `scheduler.rs` (token budget scheduling), `block_pool.rs` (KV cache allocation)

**`crabinfer-core/src/providers/`:**
- Purpose: Cloud and local provider implementations behind the `Provider` trait
- Contains: One file per backend: `local.rs`, `openai.rs`, `anthropic.rs`, `google.rs`, `ollama.rs`, `vllm.rs`, `http_utils.rs`

**`crabinfer-server/src/routes/`:**
- Purpose: Axum HTTP route handlers — one file per API surface
- Contains: `openai.rs` (chat completions + models), `anthropic.rs` (messages endpoint), `health.rs` (health + metrics)

**`crabinfer-cli/src/`:**
- Purpose: CLI subcommand implementations — one `cmd_*.rs` per subcommand
- Contains: 9 command modules; `main.rs` is only dispatch + Clap definitions

**`crabinfer-node/npm/`:**
- Purpose: Pre-compiled native `.node` binaries for each supported platform
- Generated: Yes (by napi-rs cross-compilation)
- Committed: Yes (for npm distribution without requiring Rust toolchain)

## Key File Locations

**Entry Points:**
- `crabinfer-cli/src/main.rs`: CLI binary `main()`, Clap parser, subcommand dispatch
- `crabinfer-server/src/main.rs`: Server binary entry; delegates to `run_server()` in `lib.rs`
- `crabinfer-core/src/lib.rs`: All UniFFI-exported symbols (Swift FFI entry point)
- `crabinfer-node/src/lib.rs`: napi-rs module declaration root

**Configuration:**
- `Cargo.toml`: Workspace members, release profile (LTO fat, strip, panic=unwind), candle patch
- `crabinfer-core/src/crabinfer.udl`: UniFFI interface definition — drives Swift binding generation
- `crabinfer-server/src/lib.rs`: `ServerConfig` struct with all env var overrides documented

**Core Logic:**
- `crabinfer-core/src/engine.rs`: GGUF model loading + token generation with Metal/CPU support
- `crabinfer-core/src/router.rs`: `Router` struct, 5 `RoutingPolicy` variants, 3 `ProviderTier` types
- `crabinfer-core/src/agent.rs`: `Agent` struct, tool-calling loop, `AgentConfig`, `AgentResponse`
- `crabinfer-core/src/serving/engine_loop.rs`: `EngineHandle`, `ServingEngineConfig`, batching loop
- `crabinfer-core/src/serving/scheduler.rs`: Token-budget continuous batching scheduler

**Testing:**
- `crabinfer-core/tests/`: Integration tests for core library
- Individual `src/*.rs` files contain `#[cfg(test)]` unit test modules inline

## Naming Conventions

**Files:**
- Snake case: `engine_loop.rs`, `block_pool.rs`, `chat_template.rs`
- CLI commands: `cmd_{subcommand}.rs` (e.g., `cmd_serve.rs`, `cmd_chat.rs`)
- Provider implementations: `{provider_name}.rs` (e.g., `openai.rs`, `vllm.rs`)
- Route handlers: `{api_surface}.rs` (e.g., `openai.rs`, `anthropic.rs`, `health.rs`)

**Directories:**
- Snake case matching Rust module names: `serving/`, `providers/`, `backends/`, `kernels/`, `models/`

**Types:**
- Structs: PascalCase — `CrabInferEngine`, `EngineConfig`, `CompletionRequest`, `RoutingDecision`
- Traits: PascalCase — `Provider`, `ModelRunner`, `Tool`, `EmbeddingProvider`, `Backend`
- Enums: PascalCase variants — `RoutingPolicy::LocalFirst`, `MemoryPressure::Critical`
- Error variants: PascalCase with `{ field }` for context — `ModelTooLarge { file_size_mb, max_allowed_mb }`

## Where to Add New Code

**New cloud provider:**
- Implementation: `crabinfer-core/src/providers/{provider}.rs` implementing `Provider` trait
- Registration: Add match arm in `CrabInferProvider::new()` in `crabinfer-core/src/lib.rs`
- Tier assignment: Update `router::resolve_tier()` in `crabinfer-core/src/router.rs`

**New CLI subcommand:**
- Handler: `crabinfer-cli/src/cmd_{name}.rs`
- Registration: Add variant to `Commands` enum and match arm in `main()` in `crabinfer-cli/src/main.rs`

**New HTTP route:**
- Handler: `crabinfer-server/src/routes/{surface}.rs` or add to existing file
- Registration: Add `.route(...)` call in `crabinfer-server/src/routes/mod.rs::create_router()`

**New built-in tool for agent:**
- Implementation: `crabinfer-core/src/tools.rs` — implement `Tool` trait, register in default `ToolRegistry::new()`

**New serving engine component:**
- Implementation: `crabinfer-core/src/serving/{component}.rs`
- Registration: Add `pub mod {component};` to `crabinfer-core/src/serving/mod.rs`

**New UniFFI-exported type:**
- Add `#[derive(uniffi::Record)]` / `#[derive(uniffi::Object)]` / `#[derive(uniffi::Enum)]` to struct/enum
- Add `#[uniffi::export]` to impl block
- Declare in `crabinfer-core/src/crabinfer.udl` if not using proc-macros

**New Node.js binding:**
- Add `pub mod {name};` to `crabinfer-node/src/lib.rs`
- Implement in `crabinfer-node/src/{name}.rs` using napi-rs `#[napi]` macros

**Utilities:**
- Shared HTTP utilities for cloud providers: `crabinfer-core/src/providers/http_utils.rs`
- Shared wire types for server: `crabinfer-server/src/types/common.rs`

## Special Directories

**`target/`:**
- Purpose: Cargo build artifacts
- Generated: Yes
- Committed: No

**`crabinfer-node/npm/`:**
- Purpose: Pre-built native binaries for npm distribution
- Generated: Yes (cross-compilation via napi-rs)
- Committed: Yes

**`.planning/`:**
- Purpose: GSD planning documents for phased development
- Generated: Manually by GSD tools
- Committed: Yes

**`examples/electron-demo/out/`:**
- Purpose: Compiled Electron app output
- Generated: Yes
- Committed: Yes (contains pre-built renderer assets)

---

*Structure analysis: 2026-03-12*
