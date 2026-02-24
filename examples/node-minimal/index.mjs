#!/usr/bin/env node
/**
 * CrabInfer Node.js — Minimal Example
 *
 * Usage:
 *   npm install @crabinfer/node
 *   node index.mjs /path/to/model.gguf "What is Rust?"
 */

import { CrabInferEngine, detectDevice, version } from '@crabinfer/node'

const modelPath = process.argv[2]
const prompt = process.argv[3] || 'What is Rust?'

if (!modelPath) {
  console.error('Usage: node index.mjs <model.gguf> [prompt]')
  process.exit(1)
}

// 1. Print version and device info
console.log(`CrabInfer v${version()}`)
const device = detectDevice()
console.log(`${device.chipName} · ${(device.totalMemoryBytes / 1e9).toFixed(1)} GB RAM · Metal: ${device.hasMetalGpu}`)
console.log()

// 2. Load model
console.log(`Loading ${modelPath}...`)
const engine = new CrabInferEngine({
  modelPath,
  maxTokens: 512,
  temperature: 0.7,
  topP: 0.9,
  contextLength: 4096,
  useMetal: true,
  memoryLimitBytes: 0,
  metallibPath: ''
})
await engine.loadModel(modelPath)

const info = engine.modelInfo()
console.log(`Model: ${info.modelName} (${info.quantization}, ${info.parameterCount} params)`)
console.log()

// 3. Stream tokens
console.log(`> ${prompt}`)
console.log()

const stream = engine.stream({
  messages: [{ role: 'user', content: prompt }],
  maxTokens: 200,
  temperature: 0.7
})

for await (const token of stream) {
  process.stdout.write(token.text)
  if (token.isEndOfSequence) break
}
console.log()

// 4. Print stats
const stats = engine.lastStats()
if (stats) {
  console.log()
  console.log(`${stats.tokensGenerated} tokens · ${stats.tokensPerSecond.toFixed(1)} tok/s · TTFT ${stats.timeToFirstTokenMs.toFixed(0)}ms · ${stats.computeBackend}`)
}
