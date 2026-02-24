# CrabInfer Python Client

Use the standard OpenAI Python SDK to talk to a local `crabinfer serve` instance.

## Setup

```bash
# Terminal 1: Start the server
crabinfer serve --model /path/to/model.gguf --port 8080

# Terminal 2: Install and run
pip install -r requirements.txt
python client.py "What is Rust?"
```

## What it does

1. Connects to `http://localhost:8080/v1` (no API key needed)
2. Sends a chat completion request using the OpenAI SDK
3. Shows both non-streaming and streaming modes

## Why this works

CrabInfer's server implements the OpenAI Chat Completions API (`/v1/chat/completions`), so any tool or library that works with the OpenAI SDK works with CrabInfer — zero code changes beyond pointing `base_url` at your local server.
