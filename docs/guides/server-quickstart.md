# Run a Local OpenAI-Compatible Server

CrabInfer ships an HTTP server that speaks the OpenAI and Anthropic APIs. Any app that works with the OpenAI SDK can use your local model with zero code changes.

## Quick Start

```bash
# Install CrabInfer
cargo install crabinfer --features metal

# Start the server
crabinfer serve --model /path/to/model.gguf --port 8080
```

That's it. The server is now running at `http://localhost:8080` with these endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/v1/models` | List loaded models |
| POST | `/v1/chat/completions` | OpenAI Chat Completions |
| POST | `/v1/messages` | Anthropic Messages |
| GET | `/metrics` | Server metrics (Prometheus) |

## Use with the OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"  # Local server doesn't need auth
)

response = client.chat.completions.create(
    model="local",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Rust?"}
    ],
    max_tokens=200,
    temperature=0.7,
    stream=False
)

print(response.choices[0].message.content)
```

### Streaming

```python
stream = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "Write a haiku about programming"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## Use with the OpenAI Node.js SDK

```javascript
import OpenAI from 'openai'

const client = new OpenAI({
  baseURL: 'http://localhost:8080/v1',
  apiKey: 'not-needed'
})

const response = await client.chat.completions.create({
  model: 'local',
  messages: [{ role: 'user', content: 'What is Rust?' }],
  max_tokens: 200
})

console.log(response.choices[0].message.content)
```

## Use with curl

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### Streaming with curl

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [{"role": "user", "content": "Write a story"}],
    "stream": true
  }' --no-buffer
```

## Anthropic-Compatible Endpoint

The server also exposes an Anthropic Messages API:

```python
import anthropic

client = anthropic.Anthropic(
    base_url="http://localhost:8080",
    api_key="not-needed"
)

message = client.messages.create(
    model="local",
    max_tokens=200,
    messages=[{"role": "user", "content": "What is Rust?"}]
)

print(message.content[0].text)
```

## Server Options

```
crabinfer serve [OPTIONS]

Options:
  --model <PATH>         Path to a GGUF model file (required)
  --port <PORT>          Port to listen on [default: 8080]
  --host <HOST>          Host to bind to [default: 127.0.0.1]
  --context-length <N>   Max context length [default: 4096]
  --cpu                  Disable Metal GPU (CPU only)
  --advertise            Advertise via Bonjour/mDNS for LAN discovery (macOS)
```

### Bind to all interfaces

To make the server accessible from other machines on your network:

```bash
crabinfer serve --model model.gguf --host 0.0.0.0 --port 8080
```

### LAN Discovery with Bonjour

On macOS, `--advertise` broadcasts the server via mDNS so other devices on the LAN can discover it automatically:

```bash
crabinfer serve --model model.gguf --advertise
```

Other CrabInfer clients (including the Smart Router) can discover and use this server as a self-hosted provider.

## Docker

Run the server in a Docker container (CPU-only on Linux):

```bash
docker run -p 8080:8080 \
  -v /path/to/models:/models \
  crabinfer/server:latest \
  --model /models/model.gguf
```

Build from source:

```bash
docker build -t crabinfer-server .
docker run -p 8080:8080 -v /models:/models crabinfer-server --model /models/model.gguf
```

## Health Check and Metrics

### Health

```bash
curl http://localhost:8080/health
# {"status": "ok", "model": "qwen3-1.7b", "uptime_seconds": 42}
```

### Models

```bash
curl http://localhost:8080/v1/models
# Returns OpenAI-compatible model list with the loaded model
```

### Metrics

```bash
curl http://localhost:8080/metrics
# Prometheus-format metrics:
# crabinfer_requests_total 42
# crabinfer_tokens_generated_total 8400
# crabinfer_avg_tokens_per_second 18.5
# crabinfer_active_requests 0
```

## Using with Smart Routing

CrabInfer's Smart Router can use a `crabinfer serve` instance as a self-hosted provider:

```swift
let router = try CrabInfer.Router(
    policy: .auto,
    localConfig: CrabInfer.Config(modelPath: "small-model.gguf"),
    cloudConfigs: [
        // Self-hosted CrabInfer server on your Mac Studio
        CrabInfer.selfHostedConfig(
            provider: "ollama",
            baseUrl: "http://mac-studio.local:8080"
        ),
        // Cloud fallback
        CrabInfer.cloudConfig(provider: "openai", apiKey: key, model: "gpt-4o")
    ]
)
```

With `Auto` routing, requests go to: local device -> self-hosted server -> cloud, based on availability and latency.

## Next Steps

- **Smart Routing**: Combine local + server + cloud — see [Smart Routing Guide](smart-routing.md)
- **Electron Integration**: Build a desktop app — see [Electron Guide](electron-quickstart.md)
- **iOS Client**: Connect from an iOS app — see [iOS Guide](ios-quickstart.md)
