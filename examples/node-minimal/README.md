# CrabInfer Node.js — Minimal Example

A 40-line example showing local LLM inference with `@crabinfer/node`.

## Run

```bash
npm install
node index.mjs /path/to/model.gguf "What is Rust?"
```

## What it does

1. Detects device (chip, RAM, Metal GPU)
2. Loads a GGUF model with Metal acceleration
3. Streams tokens to stdout
4. Prints generation statistics (tok/s, TTFT, backend)
