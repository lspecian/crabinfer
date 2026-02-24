/**
 * Wraps @crabinfer/node for the Electron main process.
 * All inference runs in main process via napi-rs (in-process, no HTTP).
 */

import type { BrowserWindow } from 'electron'

// Dynamic require to avoid bundling issues with native addons
const crabinfer = require('@crabinfer/node')

type CrabInfer = typeof import('@crabinfer/node')

let engine: InstanceType<CrabInfer['CrabInferEngine']> | null = null
let downloadManager: InstanceType<CrabInfer['ModelDownloadManager']> | null = null
let modelArchitecture = ''
let stopRequested = false

/** Get or create the download manager */
export function getDownloadManager() {
  if (!downloadManager) {
    downloadManager = new crabinfer.ModelDownloadManager()
  }
  return downloadManager
}

/** Detect device capabilities */
export function detectDevice() {
  return crabinfer.detectDevice()
}

/** Get CrabInfer version */
export function getVersion(): string {
  return crabinfer.version()
}

/** Get the full model catalog */
export function getModelCatalog() {
  return crabinfer.modelCatalog()
}

/** Get recommended models for this device */
export function getRecommendedModels() {
  const device = crabinfer.detectDevice()
  return crabinfer.recommendedModels(device)
}

// Architectures known to work correctly on Metal GPU
// qwen3moe requires FusedMoE CUDA kernel, not available on Metal/CPU yet
const METAL_SAFE_ARCHS = new Set(['qwen2', 'qwen3'])

/** Load a GGUF model */
export async function loadModel(modelPath: string): Promise<void> {
  const device = crabinfer.detectDevice()
  const ctxLen = device.recommendedContextLength || 2048

  // Peek at GGUF header to detect architecture without loading weights
  let arch = ''
  try {
    const meta = crabinfer.peekModelMetadata(modelPath, ctxLen)
    arch = (meta.architecture || '').toLowerCase()
    console.log(`Model metadata: arch=${arch}, params=${meta.parameterCount}, quant=${meta.quantization}`)
  } catch (err) {
    console.warn('Failed to peek model metadata, falling back to CPU:', err)
  }

  // Use Metal only for architectures known to work correctly on GPU
  const useMetal = device.hasMetalGpu && METAL_SAFE_ARCHS.has(arch)

  engine = new crabinfer.CrabInferEngine({
    modelPath,
    maxTokens: 2048,
    temperature: 0.7,
    topP: 1.0,
    contextLength: ctxLen,
    useMetal,
    memoryLimitBytes: device.availableMemoryBytes,
    metallibPath: '',
  })
  await engine.loadModel(modelPath)

  // Cache the architecture for chat template formatting
  try {
    const info = engine.modelInfo()
    modelArchitecture = info.architecture || ''
    console.log(`Model loaded: ${info.architecture}, ${info.parameterCount} params, backend: ${useMetal ? 'Metal' : 'CPU'}`)
  } catch {
    modelArchitecture = ''
  }
}

/** Unload the current model */
export function unloadModel(): void {
  if (engine) {
    engine.unloadModel()
    engine = null
    modelArchitecture = ''
  }
}

/** Check if a model is loaded */
export function isModelLoaded(): boolean {
  return engine?.isModelLoaded() ?? false
}

// ─── Chat template formatting ─────────────────────────────────────────────────
// Mirrors crabinfer-core/src/chat_template.rs

function applyChatTemplate(
  architecture: string,
  messages: { role: string; content: string }[]
): string {
  const arch = architecture.toLowerCase()
  if (arch === 'qwen2' || arch === 'qwen3' || arch.includes('qwen')) {
    return chatMLTemplate(messages)
  }
  if (arch === 'phi3' || arch.includes('phi')) {
    return phi3Template(messages)
  }
  if (arch.includes('gemma')) {
    return gemmaTemplate(messages)
  }
  // Many "llama" architecture models (SmolLM2, etc.) actually use ChatML, not
  // the Llama 3 format. Default to ChatML as it's the most widely supported.
  return chatMLTemplate(messages)
}

/** ChatML format (Qwen2, Qwen3) */
function chatMLTemplate(messages: { role: string; content: string }[]): string {
  let prompt = ''
  for (const msg of messages) {
    prompt += `<|im_start|>${msg.role}\n${msg.content}<|im_end|>\n`
  }
  prompt += '<|im_start|>assistant\n'
  return prompt
}

/** Phi-3/4 format */
function phi3Template(messages: { role: string; content: string }[]): string {
  let prompt = ''
  for (const msg of messages) {
    const tag = msg.role === 'system' ? 'system' : msg.role === 'user' ? 'user' : 'assistant'
    prompt += `<|${tag}|>\n${msg.content}<|end|>\n`
  }
  prompt += '<|assistant|>\n'
  return prompt
}

/** Llama 3 format */
function llama3Template(messages: { role: string; content: string }[]): string {
  let prompt = '<|begin_of_text|>'
  for (const msg of messages) {
    prompt += `<|start_header_id|>${msg.role}<|end_header_id|>\n\n${msg.content}<|eot_id|>`
  }
  prompt += '<|start_header_id|>assistant<|end_header_id|>\n\n'
  return prompt
}

/** Gemma format */
function gemmaTemplate(messages: { role: string; content: string }[]): string {
  let prompt = ''
  for (const msg of messages) {
    const role = msg.role === 'system' ? 'user' : msg.role === 'assistant' ? 'model' : msg.role
    prompt += `<start_of_turn>${role}\n${msg.content}<end_of_turn>\n`
  }
  prompt += '<start_of_turn>model\n'
  return prompt
}

/** Fallback: simple role: content format */
function fallbackTemplate(messages: { role: string; content: string }[]): string {
  let prompt = ''
  for (const msg of messages) {
    prompt += `${msg.role}: ${msg.content}\n\n`
  }
  prompt += 'assistant: '
  return prompt
}

// ─── Inference ────────────────────────────────────────────────────────────────

/** Complete a prompt (non-streaming) */
export async function complete(
  messages: { role: string; content: string }[],
  maxTokens: number
): Promise<string> {
  if (!engine) throw new Error('No model loaded')
  const prompt = applyChatTemplate(modelArchitecture, messages)
  return engine.complete(prompt, maxTokens)
}

/** Request stop of current inference */
export function stopInference(): void {
  stopRequested = true
  if (engine) {
    engine.reset()
  }
}

/** Stream tokens via IPC to the renderer */
export async function streamToRenderer(
  win: BrowserWindow,
  messages: { role: string; content: string }[],
  maxTokens: number
): Promise<void> {
  if (!engine) throw new Error('No model loaded')

  stopRequested = false
  const prompt = applyChatTemplate(modelArchitecture, messages)

  // Token-by-token streaming via nextToken
  engine.reset()
  let firstToken = true

  for (let i = 0; i < maxTokens; i++) {
    if (stopRequested) {
      // User requested stop
      if (!win.isDestroyed()) {
        win.webContents.send('inference:stream-token', '', true)
      }
      break
    }

    const token = engine.nextToken(firstToken ? prompt : '')
    firstToken = false

    if (!token || token.isEndOfSequence) {
      if (!win.isDestroyed()) {
        win.webContents.send('inference:stream-token', '', true)
      }
      break
    }

    if (!win.isDestroyed()) {
      win.webContents.send('inference:stream-token', token.text, false)
    }

    // Yield to event loop every token so stop IPC can be processed
    await new Promise((resolve) => setImmediate(resolve))
  }

  stopRequested = false
}

/** Get stats from the last inference */
export function getLastStats() {
  if (!engine) return null
  const stats = engine.lastStats()
  if (!stats) return null
  return {
    tokensPerSecond: stats.tokensPerSecond,
    timeToFirstTokenMs: stats.timeToFirstTokenMs,
    totalTimeMs: stats.totalTimeMs,
    peakMemoryBytes: stats.peakMemoryBytes,
    backend: stats.computeBackend,
    provider: 'local',
    model: modelArchitecture || 'loaded',
  }
}

/** Set a credential for a cloud provider */
export function setCredential(provider: string, apiKey: string): void {
  crabinfer.setCredential(provider, apiKey)
}

/** Get configured providers */
export function getConfiguredProviders(): string[] {
  return crabinfer.configuredProviders()
}
