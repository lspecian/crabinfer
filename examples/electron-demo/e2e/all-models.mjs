/**
 * CrabInfer Multi-Model Compatibility Test
 *
 * Tests every downloaded model end-to-end:
 *  1. Load model (CPU or Metal depending on arch)
 *  2. Generate tokens with a chat prompt
 *  3. Verify output (no panics, valid text, EOS works)
 *  4. Unload model
 *
 * Uses @crabinfer/node directly (no Electron UI overhead).
 *
 * Run: node e2e/all-models.mjs
 */

import { createRequire } from 'module'
import { resolve, dirname } from 'path'
import { fileURLToPath } from 'url'
import { readdirSync, existsSync, statSync } from 'fs'

const require = createRequire(import.meta.url)
const __dirname = dirname(fileURLToPath(import.meta.url))

// Load the native addon
const crabinfer = require('@crabinfer/node')

// ── Config ───────────────────────────────────────────────────────────────────

const MODELS_DIR = resolve(
  process.env.HOME,
  'Library/Application Support/CrabInfer/models'
)
const MAX_TOKENS = 64 // enough to verify generation works
const TEST_PROMPT_MESSAGES = [
  { role: 'system', content: 'You are a helpful assistant. Be brief.' },
  { role: 'user', content: 'What is 2+2? Answer in one word.' },
]

// Metal-safe architectures (from inference-bridge.ts)
const METAL_SAFE_ARCHS = new Set(['qwen2', 'qwen3'])

// Architectures that require CUDA FusedMoE kernel (unsupported on Metal/CPU)
const CUDA_ONLY_ARCHS = new Set(['qwen3moe'])

let passed = 0
let failed = 0
let skipped = 0
const results = []

function log(msg) {
  console.log(`[test] ${msg}`)
}

// ── Chat template formatting (mirrors inference-bridge.ts) ───────────────────

function applyChatTemplate(arch, messages) {
  const a = arch.toLowerCase()
  if (a === 'qwen2' || a === 'qwen3' || a.includes('qwen')) return chatML(messages)
  if (a === 'phi3' || a.includes('phi')) return phi3(messages)
  if (a.includes('gemma')) return gemma(messages)
  if (a.includes('llama')) return llama3(messages)
  return chatML(messages) // default
}

function chatML(msgs) {
  let p = ''
  for (const m of msgs) p += `<|im_start|>${m.role}\n${m.content}<|im_end|>\n`
  return p + '<|im_start|>assistant\n'
}

function phi3(msgs) {
  let p = ''
  for (const m of msgs) {
    const tag = m.role === 'system' ? 'system' : m.role === 'user' ? 'user' : 'assistant'
    p += `<|${tag}|>\n${m.content}<|end|>\n`
  }
  return p + '<|assistant|>\n'
}

function llama3(msgs) {
  let p = '<|begin_of_text|>'
  for (const m of msgs) p += `<|start_header_id|>${m.role}<|end_header_id|>\n\n${m.content}<|eot_id|>`
  return p + '<|start_header_id|>assistant<|end_header_id|>\n\n'
}

function gemma(msgs) {
  let p = ''
  for (const m of msgs) {
    const role = m.role === 'system' ? 'user' : m.role === 'assistant' ? 'model' : m.role
    p += `<start_of_turn>${role}\n${m.content}<end_of_turn>\n`
  }
  return p + '<start_of_turn>model\n'
}

// ── Discover downloaded models ───────────────────────────────────────────────

function discoverModels() {
  if (!existsSync(MODELS_DIR)) {
    log(`Models directory not found: ${MODELS_DIR}`)
    return []
  }

  const models = []
  for (const dir of readdirSync(MODELS_DIR)) {
    const modelDir = resolve(MODELS_DIR, dir)
    if (!statSync(modelDir).isDirectory()) continue

    // Find the .gguf file
    const files = readdirSync(modelDir)
    const gguf = files.find((f) => f.endsWith('.gguf'))
    if (!gguf) continue

    models.push({
      id: dir,
      path: resolve(modelDir, gguf),
      filename: gguf,
    })
  }

  return models
}

// ── Test a single model ──────────────────────────────────────────────────────

async function testModel(model) {
  const result = {
    id: model.id,
    filename: model.filename,
    arch: '',
    backend: '',
    loadTimeMs: 0,
    tokensGenerated: 0,
    tokensPerSecond: 0,
    outputSnippet: '',
    status: 'unknown',
    error: '',
  }

  try {
    // 1. Peek at metadata
    const device = crabinfer.detectDevice()
    const ctxLen = device.recommendedContextLength || 2048
    let arch = ''
    try {
      const meta = crabinfer.peekModelMetadata(model.path, ctxLen)
      arch = (meta.architecture || '').toLowerCase()
      result.arch = arch
      log(`  Arch: ${arch}, Params: ${meta.parameterCount}, Quant: ${meta.quantization}`)
    } catch (e) {
      log(`  Warning: Could not peek metadata: ${e.message}`)
    }

    // Skip architectures that require CUDA FusedMoE kernel
    if (CUDA_ONLY_ARCHS.has(arch)) {
      result.backend = 'N/A'
      result.status = 'skip'
      skipped++
      log(`  SKIPPED: ${arch} requires CUDA FusedMoE kernel (not available on Metal/CPU)`)
      results.push(result)
      return result
    }

    // 2. Create engine and load
    const useMetal = device.hasMetalGpu && METAL_SAFE_ARCHS.has(arch)
    result.backend = useMetal ? 'Metal' : 'CPU'

    const engine = new crabinfer.CrabInferEngine({
      modelPath: model.path,
      maxTokens: MAX_TOKENS,
      temperature: 0.7,
      topP: 1.0,
      contextLength: ctxLen,
      useMetal,
      memoryLimitBytes: device.availableMemoryBytes,
      metallibPath: '',
    })

    const loadStart = Date.now()
    await engine.loadModel(model.path)
    result.loadTimeMs = Date.now() - loadStart
    log(`  Loaded in ${(result.loadTimeMs / 1000).toFixed(1)}s on ${result.backend}`)

    // 3. Generate tokens
    const prompt = applyChatTemplate(arch, TEST_PROMPT_MESSAGES)
    let output = ''
    let tokenCount = 0

    engine.reset()
    let firstToken = true

    for (let i = 0; i < MAX_TOKENS; i++) {
      const token = engine.nextToken(firstToken ? prompt : '')
      firstToken = false

      if (!token || token.isEndOfSequence) break

      output += token.text
      tokenCount++
    }

    result.tokensGenerated = tokenCount
    result.outputSnippet = output.slice(0, 150).replace(/\n/g, '\\n')

    // Get stats
    try {
      const stats = engine.lastStats()
      if (stats) {
        result.tokensPerSecond = Math.round(stats.tokensPerSecond * 10) / 10
      }
    } catch {}

    log(`  Generated ${tokenCount} tokens at ${result.tokensPerSecond} tok/s`)
    log(`  Output: "${result.outputSnippet}"`)

    // 4. Validate output
    if (tokenCount === 0) {
      throw new Error('No tokens generated')
    }
    if (output.includes('byte index') && output.includes('char boundary')) {
      throw new Error('UTF-8 boundary panic leaked into output')
    }

    // 5. Unload
    engine.unloadModel()
    log(`  Unloaded OK`)

    result.status = 'pass'
    passed++
  } catch (e) {
    result.status = 'fail'
    result.error = e.message || String(e)
    failed++
    log(`  ERROR: ${result.error}`)
  }

  results.push(result)
  return result
}

// ── Main ─────────────────────────────────────────────────────────────────────

async function main() {
  console.log('╔══════════════════════════════════════════════════════════╗')
  console.log('║       CrabInfer Multi-Model Compatibility Test          ║')
  console.log('╚══════════════════════════════════════════════════════════╝\n')

  const device = crabinfer.detectDevice()
  log(`Device: ${device.chipName} (${(device.totalMemoryBytes / 1e9).toFixed(0)} GB RAM, Metal: ${device.hasMetalGpu})`)
  log(`CrabInfer version: ${crabinfer.version()}\n`)

  const models = discoverModels()
  if (models.length === 0) {
    log('No downloaded models found. Download models via the Electron app first.')
    process.exitCode = 1
    return
  }

  log(`Found ${models.length} downloaded model(s):\n`)
  for (const m of models) {
    log(`  • ${m.id} (${m.filename})`)
  }
  console.log('')

  // Test each model sequentially (to avoid memory pressure)
  for (let i = 0; i < models.length; i++) {
    const m = models[i]
    console.log(`\n─── [${i + 1}/${models.length}] Testing: ${m.id} ───`)
    await testModel(m)
  }

  // ── Summary table ───────────────────────────────────────────────────────
  console.log('\n\n╔══════════════════════════════════════════════════════════════════════════╗')
  console.log('║                            RESULTS                                     ║')
  console.log('╠══════════════════════════════════════════════════════════════════════════╣')

  const colId = 30
  const colArch = 10
  const colBackend = 7
  const colTokens = 8
  const colSpeed = 10
  const colStatus = 6

  console.log(
    `║ ${'Model'.padEnd(colId)} ${'Arch'.padEnd(colArch)} ${'Backend'.padEnd(colBackend)} ${'Tokens'.padEnd(colTokens)} ${'Tok/s'.padEnd(colSpeed)} ${'Status'.padEnd(colStatus)} ║`
  )
  console.log('╟──────────────────────────────────────────────────────────────────────────╢')

  for (const r of results) {
    const status = r.status === 'pass' ? '✅' : r.status === 'fail' ? '❌' : '⏭️'
    console.log(
      `║ ${r.id.padEnd(colId)} ${r.arch.padEnd(colArch)} ${r.backend.padEnd(colBackend)} ${String(r.tokensGenerated).padEnd(colTokens)} ${String(r.tokensPerSecond).padEnd(colSpeed)} ${status.padEnd(colStatus)} ║`
    )
    if (r.error) {
      console.log(`║   Error: ${r.error.slice(0, 62).padEnd(65)} ║`)
    }
  }

  console.log('╚══════════════════════════════════════════════════════════════════════════╝')
  console.log(`\n${passed} passed, ${failed} failed, ${skipped} skipped`)

  if (failed > 0) {
    console.log('\n💥 Some models failed!')
    process.exitCode = 1
  } else {
    console.log('\n🦀 All models passed!')
  }
}

main().catch((e) => {
  console.error('Unhandled error:', e)
  process.exitCode = 1
})
