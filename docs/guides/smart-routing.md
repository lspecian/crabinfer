# Smart Routing: Local + Cloud in One API

CrabInfer's Smart Router gives you a single API that automatically selects between local inference, self-hosted servers, and cloud providers. If the local model runs out of memory, it falls back to your Mac Studio on the LAN. If that's down, it hits the cloud. Your app code doesn't change.

## Architecture

```
┌─────────────────────────────────────────────┐
│  Your App                                   │
│  router.complete(request) / router.stream() │
└──────────────────┬──────────────────────────┘
                   │
          ┌────────┴────────┐
          │  Smart Router   │
          │  (select_provider) │
          └───┬────┬────┬───┘
              │    │    │
     ┌────────┘    │    └────────┐
     v             v             v
  ┌──────┐   ┌──────────┐   ┌────────┐
  │Local │   │Self-Hosted│   │ Cloud  │
  │Engine│   │Ollama/vLLM│   │OpenAI  │
  │(Metal)│  │CrabInfer  │   │Anthropic│
  └──────┘   └──────────┘   │Google  │
                             └────────┘
```

## Three Provider Tiers

| Tier | Examples | Latency | Privacy | Cost |
|---|---|---|---|---|
| **Local** | CrabInfer Engine (Metal) | Lowest | Full | Free |
| **Self-Hosted** | Ollama, vLLM, CrabInfer Server | Low (LAN) | High | Electricity |
| **Cloud** | OpenAI, Anthropic, Google | Higher | Varies | Per-token |

## Five Routing Policies

| Policy | Order | Use Case |
|---|---|---|
| `LocalFirst` | Local -> Self-Hosted -> Cloud | Privacy-focused, minimize cost |
| `CloudFirst` | Cloud -> Self-Hosted -> Local | Best quality, local as fallback |
| `LocalOnly` | Local only, error if unavailable | Offline / air-gapped |
| `Auto` | Local -> Self-Hosted (by latency) -> Cloud | Best of all worlds |
| `SelfHostedFirst` | Self-Hosted -> Local -> Cloud | Team server as primary |

## Quick Start

### Swift (iOS / macOS)

```swift
import CrabInfer

let router = try CrabInfer.Router(
    policy: .localFirst,
    localConfig: CrabInfer.Config(modelPath: "/path/to/model.gguf"),
    cloudConfigs: [
        CrabInfer.cloudConfig(provider: "openai", apiKey: openaiKey, model: "gpt-4o"),
        CrabInfer.cloudConfig(provider: "anthropic", apiKey: claudeKey, model: "claude-sonnet-4-5-20250929"),
    ]
)

let request = CompletionRequest(
    messages: [
        ChatMessage(role: .system, content: "You are a helpful assistant."),
        ChatMessage(role: .user, content: "Explain smart routing in 2 sentences.")
    ],
    maxTokens: 200,
    temperature: 0.7,
    topP: 0.9,
    stream: false
)

let response = try await router.complete(request)
print(response.text)

// See which provider was used
if let decision = router.lastRoutingDecision {
    print("Provider: \(decision.providerName)")
    print("Reason: \(decision.reason)")
    print("Tier: \(decision.providerTier)")
    print("Latency: \(decision.latencyMs) ms")
}
```

### Node.js

```javascript
const { CrabInferRouter } = require('@crabinfer/node')

const router = new CrabInferRouter({
  policy: 'LocalFirst',
  privacyMode: false,
  dataSovereignty: false,
  engineConfig: {
    modelPath: '/path/to/model.gguf',
    useMetal: true,
    contextLength: 4096,
    maxTokens: 512,
    temperature: 0.7,
    topP: 0.9,
    memoryLimitBytes: 0,
    metallibPath: ''
  },
  providerConfigs: [
    { provider: 'openai', apiKey: process.env.OPENAI_API_KEY, model: 'gpt-4o' }
  ]
})

const response = await router.complete({
  messages: [{ role: 'user', content: 'Hello!' }],
  maxTokens: 200
})

console.log(response.text)
console.log(`Routed to: ${response.routingDecision.providerName}`)
```

### Rust

```rust
use crabinfer_core::router::{Router, RouterConfig, RoutingPolicy};
use crabinfer_core::provider::CompletionRequest;

let config = RouterConfig {
    policy: RoutingPolicy::LocalFirst,
    privacy_mode: false,
    ollama_is_local: true,
    data_sovereignty: false,
};

let router = Router::new(
    config,
    Some(local_provider),     // On-device engine
    vec![ollama_provider],    // Self-hosted
    vec![openai_provider],    // Cloud
);

let response = router.complete(&request)?;
println!("Provider: {}", response.routing_decision.provider_name);
```

## Automatic Fallback

The Router cascades through tiers when a provider fails:

```
User sends request
  │
  ├─ Try Local Engine
  │   ├─ Memory pressure Critical/Terminal? → Skip
  │   ├─ Model not loaded? → Skip
  │   └─ Generation error? → Fall to next tier
  │
  ├─ Try Self-Hosted (sorted by latency)
  │   ├─ Network unavailable? → Skip all
  │   ├─ Health check fails? → Try next server
  │   └─ Generation error? → Fall to next tier
  │
  └─ Try Cloud
      ├─ Privacy mode? → Error (blocked)
      ├─ Data sovereignty? → Error (blocked)
      └─ API error? → Try next cloud provider
```

Each routing decision records **why** a specific provider was chosen:

| Reason | Meaning |
|---|---|
| `LocalAvailable` | Local model loaded and memory OK |
| `MemoryPressureFallback` | Local skipped due to high memory |
| `NoLocalModel` | No local model configured/loaded |
| `CloudPreferred` | CloudFirst policy selected cloud |
| `NetworkUnavailable` | Network down, forced local |
| `PrivacyMode` | Cloud blocked by privacy config |
| `FallbackAfterError` | Primary failed, using fallback |
| `SelfHostedAvailable` | Self-hosted server is healthy |
| `SelfHostedPreferred` | SelfHostedFirst policy |
| `DataSovereignty` | Cloud blocked, using local/self-hosted |

## Privacy and Data Sovereignty

### Privacy Mode

Block all external providers — data never leaves the device:

```swift
let router = try CrabInfer.Router(
    policy: .localFirst,
    privacyMode: true,  // No cloud, no self-hosted
    localConfig: config
)
```

### Data Sovereignty

Allow self-hosted servers on your infrastructure but block public cloud APIs:

```swift
let router = try CrabInfer.Router(
    policy: .auto,
    dataSovereignty: true,  // No public cloud
    localConfig: config,
    cloudConfigs: [
        // This self-hosted server WILL be used
        CrabInfer.selfHostedConfig(provider: "ollama", baseUrl: "http://internal-server:11434"),
        // This cloud provider will be BLOCKED
        CrabInfer.cloudConfig(provider: "openai", apiKey: key, model: "gpt-4o")
    ]
)
```

## Network Awareness

The Router tracks network availability. On iOS, connect it to `NWPathMonitor`:

```swift
import Network

let monitor = NWPathMonitor()
monitor.pathUpdateHandler = { path in
    router.setNetworkAvailable(path.status == .satisfied)
}
monitor.start(queue: .global())
```

When the network goes down:
- `LocalFirst` / `Auto`: Seamlessly continues with local inference
- `CloudFirst`: Falls back to local
- `LocalOnly`: No change (never uses network)
- Self-hosted providers become unavailable

## Self-Hosted Providers

### Ollama

```swift
let ollamaConfig = CrabInfer.selfHostedConfig(
    provider: "ollama",
    baseUrl: "http://mac-studio.local:11434"
)
```

### vLLM

```swift
let vllmConfig = CrabInfer.selfHostedConfig(
    provider: "vllm",
    baseUrl: "http://gpu-server:8000"
)
```

### CrabInfer Server

Run `crabinfer serve` on a more powerful machine and use it as a self-hosted provider:

```bash
# On your Mac Studio
crabinfer serve --model /models/llama-70b-q4.gguf --host 0.0.0.0 --advertise
```

```swift
// From your iPhone app
let serverConfig = CrabInfer.selfHostedConfig(
    provider: "ollama",  // Compatible API
    baseUrl: "http://mac-studio.local:8080"
)
```

### Latency-Aware Selection

With `Auto` policy, self-hosted providers are sorted by measured latency. The Router periodically health-checks each server and picks the fastest one:

```
Self-hosted providers (sorted by latency):
  1. mac-studio.local:11434  — 3ms
  2. gpu-server:8000         — 12ms
  3. cloud-vm:8080           — 45ms
```

Latency is refreshed every 60 seconds.

## Streaming

Streaming works through the router just like direct engine access:

```swift
let request = CompletionRequest(
    messages: [ChatMessage(role: .user, content: "Tell me a story")],
    maxTokens: 500,
    temperature: 0.7,
    topP: 0.9,
    stream: true
)

for try await token in router.stream(request) {
    print(token.text, terminator: "")
}

// Check which provider streamed the response
if let decision = router.lastRoutingDecision {
    print("\n[Streamed via \(decision.providerName)]")
}
```

## Available Models

Query all models across all configured providers:

```swift
let models = try router.availableModels()
for model in models {
    print("\(model.name) — \(model.provider) (\(model.tier))")
}
```

## Next Steps

- **iOS Integration**: See the [iOS Guide](ios-quickstart.md) for a complete app with routing
- **Server Mode**: Run your own self-hosted provider — see [Server Guide](server-quickstart.md)
- **Electron App**: Desktop app with routing — see [Electron Guide](electron-quickstart.md)
