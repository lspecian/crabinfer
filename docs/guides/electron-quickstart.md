# Build an Electron App with Local AI

Run LLM inference natively in your Electron app using `@crabinfer/node`. No Python, no separate server process — the model runs in-process via Metal GPU on Apple Silicon.

## Prerequisites

- Node.js 18+
- macOS with Apple Silicon (Metal GPU) or Linux x64/arm64 (CPU)
- A GGUF model file

## Step 1: Install the Package

```bash
npm install @crabinfer/node
```

The package includes prebuilt native binaries for:
- macOS arm64 (Apple Silicon + Metal)
- macOS x64 (Intel, CPU only)
- Linux x64 (CPU)
- Linux arm64 (CPU)

## Step 2: Detect Device Capabilities

```javascript
const { detectDevice } = require('@crabinfer/node')

const device = detectDevice()
console.log(`${device.chipName} (${device.chipVariant})`)
console.log(`RAM: ${(device.totalMemoryBytes / 1e9).toFixed(1)} GB`)
console.log(`Metal GPU: ${device.hasMetalGpu ? 'Yes' : 'No'}`)
console.log(`Recommended: ${device.recommendedQuant} (up to ${device.maxModelSizeB}B)`)
```

## Step 3: Load a Model and Generate

```javascript
const { CrabInferEngine } = require('@crabinfer/node')

const engine = new CrabInferEngine({
  modelPath: '/path/to/model.gguf',
  maxTokens: 512,
  temperature: 0.7,
  topP: 0.9,
  contextLength: 4096,
  useMetal: true,
  memoryLimitBytes: 0,
  metallibPath: ''
})

// Load the model (async, runs on libuv thread pool)
await engine.loadModel('/path/to/model.gguf')

// Check model info
const info = engine.modelInfo()
console.log(`${info.modelName} — ${info.quantization} — ${info.parameterCount} params`)

// Generate a complete response
const result = await engine.complete('What is Rust?', 200)
console.log(result)
```

### Streaming Tokens

```javascript
const stream = engine.stream({
  messages: [{ role: 'user', content: 'Explain quantum computing' }],
  maxTokens: 200,
  temperature: 0.7
})

for await (const token of stream) {
  process.stdout.write(token.text)
  if (token.isEndOfSequence) break
}
```

`TokenStream` implements the async iterator protocol — use `for await...of` in any modern Node.js.

## Step 4: Electron Architecture

Native addons run in the **main process** only. Stream tokens to the renderer via IPC:

```
Renderer (React/Vue/Svelte)
  │  ipcRenderer.invoke / ipcRenderer.on
  v
Main Process (Node.js)
  │  Direct function calls
  v
@crabinfer/node (native addon, Metal GPU)
```

### Main Process — IPC Handlers

```typescript
// main/ipc-handlers.ts
import { ipcMain } from 'electron'
import { CrabInferEngine, detectDevice } from '@crabinfer/node'

let engine: InstanceType<typeof CrabInferEngine> | null = null

export function registerHandlers(mainWindow: BrowserWindow) {
  // Device info
  ipcMain.handle('detect-device', () => detectDevice())

  // Load model
  ipcMain.handle('load-model', async (_, modelPath: string) => {
    engine = new CrabInferEngine({
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
    return engine.modelInfo()
  })

  // Stream completion
  ipcMain.handle('complete-stream', async (_, prompt: string) => {
    if (!engine) throw new Error('No model loaded')

    const stream = engine.stream({
      messages: [{ role: 'user', content: prompt }],
      maxTokens: 512,
      temperature: 0.7
    })

    for await (const token of stream) {
      // Send each token to the renderer
      mainWindow.webContents.send('token', {
        text: token.text,
        isEnd: token.isEndOfSequence
      })
      if (token.isEndOfSequence) break
    }

    return engine.lastStats()
  })
}
```

### Preload Script — Typed Bridge

```typescript
// preload/index.ts
import { contextBridge, ipcRenderer } from 'electron'

contextBridge.exposeInMainWorld('crabinfer', {
  detectDevice: () => ipcRenderer.invoke('detect-device'),
  loadModel: (path: string) => ipcRenderer.invoke('load-model', path),
  completeStream: (prompt: string) => ipcRenderer.invoke('complete-stream', prompt),
  onToken: (callback: (token: { text: string; isEnd: boolean }) => void) => {
    ipcRenderer.on('token', (_, token) => callback(token))
  }
})
```

### Renderer — React Hook

```tsx
// renderer/hooks/useChat.ts
import { useState, useCallback } from 'react'

export function useChat() {
  const [messages, setMessages] = useState<{ role: string; text: string }[]>([])
  const [isStreaming, setIsStreaming] = useState(false)

  const send = useCallback(async (text: string) => {
    setMessages(prev => [...prev, { role: 'user', text }])
    setMessages(prev => [...prev, { role: 'assistant', text: '' }])
    setIsStreaming(true)

    // Listen for streaming tokens
    window.crabinfer.onToken((token) => {
      setMessages(prev => {
        const copy = [...prev]
        copy[copy.length - 1].text += token.text
        return copy
      })
    })

    const stats = await window.crabinfer.completeStream(text)
    setIsStreaming(false)
    return stats
  }, [])

  return { messages, isStreaming, send }
}
```

## Step 5: Secure API Key Storage

For cloud provider fallback, store API keys securely using Electron's `safeStorage`:

```typescript
import { safeStorage } from 'electron'
import Store from 'electron-store'

const store = new Store()

// Encrypt and store
function saveApiKey(provider: string, key: string) {
  if (safeStorage.isEncryptionAvailable()) {
    const encrypted = safeStorage.encryptString(key)
    store.set(`apiKeys.${provider}`, encrypted.toString('base64'))
  }
}

// Decrypt and retrieve
function getApiKey(provider: string): string | null {
  const encrypted = store.get(`apiKeys.${provider}`) as string | undefined
  if (!encrypted) return null
  if (safeStorage.isEncryptionAvailable()) {
    return safeStorage.decryptString(Buffer.from(encrypted, 'base64'))
  }
  return null
}
```

## Step 6: Menu Bar App (macOS)

For a persistent AI assistant, run as a tray app:

```typescript
import { app, Tray, BrowserWindow, globalShortcut, nativeImage } from 'electron'

app.dock?.hide() // Hide from Dock

const tray = new Tray(nativeImage.createFromPath('tray-icon.png'))
tray.setToolTip('CrabInfer')

// Global shortcut to toggle window
globalShortcut.register('CommandOrControl+Shift+Space', () => {
  if (mainWindow.isVisible()) {
    mainWindow.hide()
  } else {
    // Position under tray icon
    const bounds = tray.getBounds()
    mainWindow.setPosition(bounds.x - 200, bounds.y + bounds.height)
    mainWindow.show()
  }
})

// Hide on blur instead of closing
mainWindow.on('close', (e) => {
  if (!app.isQuitting) {
    e.preventDefault()
    mainWindow.hide()
  }
})
```

## Smart Routing

Use the Router to automatically fall back to cloud when the local model can't handle a request:

```javascript
const { CrabInferRouter, CrabInferProvider } = require('@crabinfer/node')

const router = new CrabInferRouter({
  policy: 'LocalFirst',
  privacyMode: false,
  dataSovereignty: false,
  engineConfig: { modelPath: '/path/to/model.gguf', useMetal: true, ... },
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

## Next Steps

- **Agent Runtime**: Add tool calling with the agent bridge — see the full Electron demo
- **Model Catalog**: Browse and download models from the curated catalog
- **Smart Routing**: See [Smart Routing Guide](smart-routing.md) for all 5 routing policies
- **Server Mode**: Run CrabInfer as an HTTP server — see [Server Guide](server-quickstart.md)

## Full Example

See the complete Electron demo at `examples/electron-demo/` which includes:
- Menu bar app with global shortcut
- React + Tailwind dark theme UI
- Model catalog with download progress
- Chat with streaming and stats
- Agent with tool calling and MCP
- Settings with encrypted API key storage
