# CrabInfer — Local AI Inference SDK for Apple Silicon

**Safe, memory-aware LLM inference with smart routing, tool calling, and agent runtime.**

CrabInfer is a full-stack AI SDK: local GGUF inference on Apple Silicon (Metal GPU), cloud provider routing (OpenAI, Anthropic, Google, Ollama, vLLM), an autonomous agent with tool calling and MCP, and persistent memory/knowledge. Ships as a Swift SDK (iOS/macOS), Node.js binding (Electron/desktop), Rust crate, CLI, and OpenAI-compatible server.

## Quick Start

### Swift (iOS / macOS)

```swift
import CrabInfer

// Detect device and load a model
let device = CrabInfer.detectDevice()
let engine = try CrabInfer.Engine(modelPath: "/path/to/qwen3-1.7b-q4_k_m.gguf")
try await engine.loadModel()

// Stream tokens
for try await token in engine.stream("Explain quantum computing simply") {
    print(token.text, terminator: "")
}

// Or use the smart router (local-first, cloud fallback)
let router = try CrabInfer.Router(
    policy: .localFirst,
    localConfig: engineConfig,
    cloudConfigs: [CrabInfer.cloudConfig(provider: "openai", apiKey: key, model: "gpt-4o")]
)
let response = try await router.complete(request)
```

### Node.js (Electron / desktop)

```javascript
const { CrabInferEngine, detectDevice, modelCatalog } = require('@crabinfer/node')

const device = detectDevice()
console.log(`${device.chipName} · ${device.totalMemoryBytes / 1e9} GB`)

const engine = new CrabInferEngine({ modelPath: '/path/to/model.gguf', useMetal: true, ... })
await engine.loadModel('/path/to/model.gguf')
const result = await engine.complete('What is Rust?', 200)
```

### CLI

```bash
# One-shot inference
crabinfer run --model /path/to/model.gguf --prompt "Hello, world"

# Interactive chat
crabinfer chat --model /path/to/model.gguf

# AI assistant with tool calling
crabinfer assistant --provider openai --model gpt-4o

# OpenAI-compatible API server
crabinfer serve --model /path/to/model.gguf --port 8080
```

### Python (via OpenAI-compatible server)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "What is Rust?"}],
)
print(response.choices[0].message.content)
```

## Features

### Inference Engine
- **GGUF model loading** with mmap-based lazy loading
- **Metal GPU acceleration** on Apple Silicon (3-10x faster than CPU)
- **Memory pressure management** with iOS lifecycle awareness (Normal/Warning/Critical/Terminal)
- **Device capability detection** — auto-selects optimal quantization and context length
- **Model catalog** with curated entries, download manager, SHA256 verification
- **Streaming** via async iterators (Swift `AsyncSequence`, Node.js `AsyncIterator`)

### Smart Routing
- **5 routing policies**: LocalFirst, CloudFirst, LocalOnly, Auto, SelfHostedFirst
- **3 provider tiers**: Local, SelfHosted (Ollama/vLLM), Cloud (OpenAI/Anthropic/Google)
- **Automatic fallback** — local fails? Try self-hosted, then cloud
- **Privacy mode** — block all cloud providers
- **Data sovereignty** — restrict to local + self-hosted only

### Agent Runtime
- **Autonomous tool-calling loop**: user input -> LLM -> parse tool calls -> execute -> feed back -> repeat
- **Built-in tools**: file_read, file_write, file_list, shell_exec, web_fetch
- **MCP (Model Context Protocol)**: connect to external tool servers via stdio or HTTP
- **Persistent memory**: conversation history + key-value facts, auto-injected into prompts
- **Knowledge base (RAG)**: TF-IDF embedder, vector store, chunking, query for prompt
- **System prompt builder**: composable identity + instructions + context with token budgets

### Providers
- **Local**: GGUF models via Candle (Metal + CPU)
- **OpenAI**: GPT-4o, GPT-4o-mini, o1, o3
- **Anthropic**: Claude Sonnet, Claude Haiku
- **Google**: Gemini Pro, Gemini Flash
- **Ollama**: Any locally-served model
- **vLLM**: Self-hosted with metrics, health checks, guided decoding

### Platform Support
- **Swift SDK** (iOS 16+, macOS 13+) via UniFFI
- **Node.js binding** via napi-rs (macOS arm64/x64, Linux x64/arm64)
- **Rust crate** (`crabinfer-core`)
- **CLI** (`crabinfer`)
- **HTTP server** (OpenAI + Anthropic compatible)
- **Docker** container

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Applications                                                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐│
│  │ iOS Demo │ │ Electron │ │   CLI    │ │  HTTP Server     ││
│  │ (Swift)  │ │  (React) │ │ (Rust)   │ │ (OpenAI compat)  ││
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────────┬─────────┘│
├───────┼─────────────┼────────────┼────────────────┼──────────┤
│  SDKs │             │            │                │           │
│  ┌────┴─────┐ ┌─────┴────┐      │                │           │
│  │Swift SDK │ │ Node.js  │      │                │           │
│  │(UniFFI)  │ │(napi-rs) │      │                │           │
│  └────┬─────┘ └─────┬────┘      │                │           │
├───────┴──────────────┴───────────┴────────────────┴──────────┤
│  crabinfer-core (Rust)                                        │
│  ┌───────────────────────────────────────────────────────────┐│
│  │  Agent Runtime (tool calling, MCP, memory, knowledge)     ││
│  │  Smart Router (local/self-hosted/cloud, 5 policies)       ││
│  │  Providers (local, OpenAI, Anthropic, Google, Ollama, vLLM)│
│  │  Engine (GGUF, Metal GPU, memory pressure, device detect) ││
│  │  Candle (tensor ops, Metal backend, CPU NEON)             ││
│  └───────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────┘
```

## Supported Devices

| Device | RAM | Max Model | Expected tok/s |
|---|---|---|---|
| iPhone 12 Pro+ | 6GB | 3B Q4_K_M | ~15 tok/s |
| iPhone 14 Pro+ | 6GB | 3B Q4_K_M / 7B Q2_K | ~15-18 tok/s |
| iPhone 15 Pro+ | 8GB | 7B Q4_K_M | ~11 tok/s |
| iPhone 16 Pro+ | 8GB | 7B Q4_K_M | ~14 tok/s |
| iPad Pro M1+ | 8-16GB | 13B Q4_K_M | ~8-12 tok/s |
| Mac (Apple Silicon) | 16-192GB | 70B+ | ~20-40 tok/s |

## Project Structure

```
crabinfer/
├── crabinfer-core/             # Rust: inference, routing, agent, providers
│   └── src/
│       ├── engine.rs           # GGUF inference engine (Metal + CPU)
│       ├── device.rs           # Device detection + recommendations
│       ├── memory_pressure.rs  # iOS memory lifecycle management
│       ├── router.rs           # Smart routing (local/cloud/self-hosted)
│       ├── agent.rs            # Agent runtime (tool calling loop)
│       ├── tools.rs            # Tool framework + built-in tools
│       ├── mcp.rs              # MCP client/server (stdio + HTTP)
│       ├── conversation.rs     # Conversation memory + persistence
│       ├── facts.rs            # Persistent key-value facts
│       ├── knowledge.rs        # RAG knowledge base
│       ├── prompt.rs           # System prompt builder
│       ├── providers/          # OpenAI, Anthropic, Google, Ollama, vLLM, Local
│       ├── catalog.rs          # Curated model catalog
│       └── download.rs         # Model download manager
├── crabinfer-swift/            # Swift SDK (UniFFI bindings)
├── crabinfer-node/             # Node.js binding (napi-rs)
├── crabinfer-cli/              # CLI: run, chat, assistant, serve, mcp, auth, models
├── crabinfer-server/           # OpenAI/Anthropic-compatible HTTP server
├── examples/
│   ├── ios-demo/               # SwiftUI demo with chat, model loader, Metal
│   ├── electron-demo/          # Electron + React menu bar app with agent
│   ├── node-minimal/           # 20-line Node.js inference example
│   └── python-client/          # Python OpenAI SDK against crabinfer serve
├── .github/workflows/          # CI (build + test) and Release pipelines
└── packaging/                  # Homebrew formula, Docker, SPM release scripts
```

## Building

### Prerequisites

```bash
# Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# iOS targets (for Swift SDK)
rustup target add aarch64-apple-ios aarch64-apple-ios-sim

# Xcode command line tools (macOS)
xcode-select --install
```

### Build from source

```bash
# Build everything (macOS with Metal)
cargo build --workspace

# Build core with specific features
cargo build -p crabinfer-core --features "metal,providers"

# Build Node.js binding
cd crabinfer-node && npm run build

# Build iOS XCFramework
./build.sh

# Run tests (195 unit tests + integration)
cargo test --workspace
```

## Guides

- [Add LLM inference to your iOS app](docs/guides/ios-quickstart.md)
- [Build an Electron app with local AI](docs/guides/electron-quickstart.md)
- [Run a local OpenAI-compatible server](docs/guides/server-quickstart.md)
- [Smart routing: local + cloud in one API](docs/guides/smart-routing.md)

## License

Apache-2.0

## Credits

- [Candle](https://github.com/huggingface/candle) by Hugging Face — tensor operations and Metal backend
- [UniFFI](https://mozilla.github.io/uniffi-rs/) by Mozilla — Rust <-> Swift bridge
- [napi-rs](https://napi.rs/) — Rust <-> Node.js bridge
