# External Integrations

**Analysis Date:** 2026-03-12

## APIs & External Services

**Cloud LLM Providers (all behind `providers` feature flag):**

- **OpenAI** — chat completions and streaming
  - Provider: `crabinfer-core/src/providers/openai.rs`
  - SDK/Client: `reqwest` 0.12 blocking HTTP + SSE
  - Default base URL: `https://api.openai.com`
  - Also supports Azure OpenAI and any OpenAI-compatible endpoint via `base_url` override
  - Auth: runtime credential `"openai"` key in `CredentialManager`, or explicit `api_key` in `ProviderConfig`

- **Anthropic** — Messages API with streaming
  - Provider: `crabinfer-core/src/providers/anthropic.rs`
  - SDK/Client: `reqwest` 0.12 blocking HTTP + SSE
  - Default base URL: `https://api.anthropic.com`
  - Auth: runtime credential `"anthropic"` key in `CredentialManager`

- **Google Gemini** — Generative Language API with streaming
  - Provider: `crabinfer-core/src/providers/google.rs`
  - SDK/Client: `reqwest` 0.12 blocking HTTP + SSE
  - Default base URL: `https://generativelanguage.googleapis.com`
  - Note: Gemini has no system role; system prompt is prepended to first user message
  - Auth: runtime credential `"google"` key in `CredentialManager`

- **Ollama** — local Ollama server (self-hosted tier)
  - Provider: `crabinfer-core/src/providers/ollama.rs`
  - SDK/Client: `reqwest` 0.12 blocking HTTP + JSON-line streaming
  - Default base URL: `http://localhost:11434`
  - Auth: none

- **vLLM** — self-hosted vLLM inference server
  - Provider: `crabinfer-core/src/providers/vllm.rs`
  - SDK/Client: `reqwest` 0.12 blocking HTTP + SSE (OpenAI-compatible `/v1/chat/completions`)
  - Extras: health check, dynamic model discovery, Prometheus `/metrics` scraping (`VllmServerMetrics`), guided decoding, repetition_penalty, min_p
  - Auth: optional bearer token

**Embeddings:**

- **OpenAI Embeddings** — `text-embedding-3-small` for high-quality dense vectors
  - Implementation: `crabinfer-core/src/embedding.rs` (`OpenAIEmbedder`)
  - SDK/Client: `reqwest` via shared HTTP utils
  - Auth: shares `"openai"` credential from `CredentialManager`

## Data Storage

**Databases:**
- None — no relational or document database dependency

**Model Files:**
- GGUF format: local filesystem, memory-mapped via `memmap2` (`crabinfer-core/src/engine.rs`)
- Safetensors format: local filesystem, loaded via `crabinfer-core/src/serving/safetensors_loader.rs`
- Default model directory: `/models` (Docker), configurable path in CLI/server flags

**File Storage:**
- Local filesystem only for model storage
- Model download manager (`crabinfer-core/src/download.rs`) writes to configurable `storage_dir` with resume support and SHA-256 integrity checks
- Source: HuggingFace Hub GGUF repositories (URLs embedded in `crabinfer-core/src/catalog.rs`)

**Vector Store:**
- In-process, no external dependency (`crabinfer-core/src/vectorstore.rs`)
- Cosine similarity search over `Vec<StoredVector>`
- Optional persistence to disk (JSON serialization via `serde_json`)

**Caching:**
- In-process KV cache (PagedAttention block pool) at `crabinfer-core/src/serving/kv_cache.rs` and `block_pool.rs`
- CPU swap buffer for KV cache offload: `crabinfer-core/src/serving/swap.rs`
- No external cache service (no Redis, Memcached, etc.)

## Authentication & Identity

**Auth Provider:**
- Custom in-process `CredentialManager` singleton (`crabinfer-core/src/credentials.rs`)
- Thread-safe `RwLock<HashMap<String, String>>` mapping provider names to API keys
- No OAuth, JWT, or session management — this is a library/SDK, not a multi-user service
- Swift apps are expected to populate keys from iOS Keychain at app launch
- CLI uses `rpassword` for interactive key entry (`crabinfer-cli/src/main.rs`)

**Server authentication:**
- `crabinfer-server` exposes unauthenticated endpoints (no auth middleware in `crabinfer-server/src/routes/mod.rs`)
- Intended for localhost or private network deployment

## Monitoring & Observability

**Metrics:**
- Built-in Prometheus-format metrics endpoint: `GET /metrics` (`crabinfer-server/src/routes/health.rs`)
- Tracked via `ServerMetrics` in `crabinfer-server/src/state.rs` using atomic counters and custom `Histogram` (no external deps)
- Latency buckets: 5ms–120s (request), 1ms–500ms (inter-token latency)
- vLLM provider can also scrape remote vLLM Prometheus metrics (`VllmServerMetrics`)

**Error Tracking:**
- Not integrated. No Sentry, Datadog, etc.

**Logs:**
- `tracing` 0.1 + `tracing-subscriber` 0.3 across all crates
- Structured logging to stderr; format configured at binary startup

## MCP (Model Context Protocol)

**MCP Client:**
- `crabinfer-core/src/mcp.rs`
- Two transport modes:
  - **Stdio**: spawns child process, communicates via stdin/stdout JSON-RPC 2.0 (`McpStdioClient`)
  - **HTTP**: connects to remote MCP server via HTTP + JSON-RPC 2.0 (`McpHttpClient`)
- Supports tool discovery (`list_tools`) and invocation (`call_tool`)
- `McpServerRegistry` for multi-server management

**MCP Server:**
- `McpServer` in `crabinfer-core/src/mcp.rs` exposes CrabInfer's own tool registry as an MCP server

## CI/CD & Deployment

**Hosting:**
- iOS/macOS: Swift Package Manager distribution (`crabinfer-swift/Package.swift`, SPM release script at `deploy/prepare-spm-release.sh`)
- Linux: Docker (`Dockerfile` — multi-stage, debian:bookworm-slim runtime, port 8080)
- Linux daemon: systemd unit (`deploy/crabinfer-server.service`)
- Node.js: npm package `@crabinfer/node` with platform-specific optional dependencies (darwin-arm64, darwin-x64, linux-x64-gnu, linux-arm64-gnu)

**CI Pipeline:**
- Not detected (no `.github/`, `.circleci/`, `.gitlab-ci.yml` found)

## Webhooks & Callbacks

**Incoming:**
- None (no webhook receiver endpoints)

**Outgoing:**
- None (no webhook dispatch)

**Progress Callbacks:**
- `DownloadProgressListener` callback interface (`crabinfer-core/src/download.rs`) — UniFFI callback trait implemented by Swift callers to receive download progress events
- `TokenCallback` pattern for streaming token output — implemented in Swift via UniFFI

## HuggingFace Hub

**Model Downloads:**
- `crabinfer-core/src/download.rs` — downloads GGUF + `tokenizer.json` from HuggingFace Hub URLs
- Curated model catalog in `crabinfer-core/src/catalog.rs` — static list of `CatalogEntry` records with `hf_repo`, `gguf_file`, `tokenizer_repo`, `sha256`, `requires_auth` fields
- Some catalog entries `requires_auth = true` (gated HuggingFace repos)
- Download uses `reqwest` with HTTP range requests for resume support

## Environment Configuration

**Required env vars:**
- None strictly required at compile time
- API keys are runtime values stored in `CredentialManager`, not environment variables
- Server/CLI accept model path, host, port, and other options via CLI flags (not env vars)

**Secrets location:**
- iOS: iOS Keychain (populated by app, passed to `CredentialManager`)
- CLI: interactive `rpassword` prompt or explicit `--api-key` flag
- Docker/Linux: caller responsibility (pass as CLI args to the binary; no env var injection built in)

---

*Integration audit: 2026-03-12*
