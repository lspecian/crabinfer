# Add LLM Inference to Your iOS App

Run language models locally on iPhone and iPad with Metal GPU acceleration. This guide takes you from zero to streaming tokens in a SwiftUI app.

## Prerequisites

- Xcode 15+ with iOS 16+ SDK
- Rust toolchain: `rustup target add aarch64-apple-ios aarch64-apple-ios-sim`
- A GGUF model file (we'll download one in Step 3)

## Step 1: Add CrabInfer to Your Project

### Swift Package Manager

Add CrabInfer as a dependency in your `Package.swift` or via Xcode's package manager:

```swift
dependencies: [
    .package(url: "https://github.com/anthropics/crabinfer.git", from: "0.1.0")
]
```

Or build the XCFramework from source:

```bash
git clone https://github.com/anthropics/crabinfer.git
cd crabinfer
./build.sh  # Produces CrabInfer.xcframework
```

Drag the resulting `CrabInfer.xcframework` into your Xcode project.

## Step 2: Detect Device Capabilities

Before loading a model, check what the device can handle:

```swift
import CrabInfer

let device = CrabInfer.getDevice()
print("""
  \(device.chipName) (\(device.chipVariant))
  RAM: \(device.totalMemoryGB) GB (\(device.availableMemoryGB) GB free)
  Metal GPU: \(device.hasMetalGPU ? "Yes" : "No")
  Recommended: \(device.recommendedQuant) (up to \(device.maxModelSizeB)B params)
  Context: \(device.recommendedContextLength) tokens
""")
```

CrabInfer detects the chip (A15, M1, etc.), available RAM, Metal support, and recommends a quantization level and model size for the device.

| Device | RAM | Recommended |
|---|---|---|
| iPhone 12 Pro | 6 GB | 3B Q4_K_M |
| iPhone 15 Pro | 8 GB | 7B Q4_K_M |
| iPad Pro M1 | 8-16 GB | 13B Q4_K_M |
| Mac (Apple Silicon) | 16-192 GB | 70B+ |

## Step 3: Download a Model

CrabInfer includes a curated model catalog. Pick a model that fits the device:

```swift
// Browse the catalog
let catalog = CrabInfer.modelCatalog()
for model in catalog {
    print("\(model.name) — \(model.sizeGB) GB, \(model.paramCountB)B params")
}

// Download a model
let downloads = CrabInfer.Downloads()
let entry = catalog.first { $0.name.contains("Qwen3-1.7B") }!

// Track progress
downloads.onProgress { progress in
    print("Download: \(Int(progress.percentage))%")
}

try await downloads.download(entry)
print("Model saved to: \(entry.localPath)")
```

Or use your own GGUF file — any Hugging Face GGUF model works.

## Step 4: Load and Run Inference

```swift
// Create the engine
let engine = try CrabInfer.Engine(modelPath: "/path/to/model.gguf")

// Check model info
let model = try engine.modelInfo()
print("\(model.name) — \(model.formattedParams) params, \(model.quantization)")

// Generate a complete response
let response = try await engine.complete("What is Swift?", maxTokens: 200)
print(response)
```

### Streaming Tokens

For a chat-like experience, stream tokens as they're generated:

```swift
for try await token in engine.stream("Explain quantum computing simply") {
    print(token.text, terminator: "")
}
print() // newline after stream
```

`TokenStream` conforms to `AsyncSequence`, so you can use it anywhere Swift async sequences work.

## Step 5: Build a Chat View

Here's a minimal SwiftUI chat view:

```swift
import SwiftUI
import CrabInfer

struct ChatView: View {
    @State private var messages: [(role: String, text: String)] = []
    @State private var input = ""
    @State private var isGenerating = false

    let engine: CrabInfer.Engine

    var body: some View {
        VStack {
            ScrollView {
                ForEach(Array(messages.enumerated()), id: \.offset) { _, msg in
                    HStack {
                        if msg.role == "user" { Spacer() }
                        Text(msg.text)
                            .padding(10)
                            .background(msg.role == "user" ? Color.blue : Color.gray.opacity(0.2))
                            .foregroundColor(msg.role == "user" ? .white : .primary)
                            .cornerRadius(12)
                        if msg.role == "assistant" { Spacer() }
                    }
                    .padding(.horizontal)
                }
            }

            HStack {
                TextField("Message", text: $input)
                    .textFieldStyle(.roundedBorder)
                Button("Send") {
                    Task { await send() }
                }
                .disabled(input.isEmpty || isGenerating)
            }
            .padding()
        }
    }

    func send() async {
        let text = input
        input = ""
        messages.append((role: "user", text: text))
        messages.append((role: "assistant", text: ""))

        isGenerating = true
        let idx = messages.count - 1

        for try await token in engine.stream(text) {
            messages[idx].text += token.text
        }

        isGenerating = false
    }
}
```

## Step 6: Handle Memory Pressure

On iOS, memory is limited. CrabInfer monitors memory pressure and exposes it so your app can respond:

```swift
// Check current pressure
switch engine.memoryState {
case .normal:   break // All good
case .warning:  engine.reduceMemory() // Free caches
case .critical: engine.unload() // Unload model to survive
case .terminal: engine.unload() // App is about to be killed
}

// In your AppDelegate or SceneDelegate:
NotificationCenter.default.addObserver(
    forName: UIApplication.didReceiveMemoryWarningNotification,
    object: nil, queue: .main
) { _ in
    engine.reduceMemory()
}
```

### Memory estimation

Check if a model will fit before loading it:

```swift
let estimatedBytes = try CrabInfer.estimateMemory(
    modelPath: "/path/to/model.gguf",
    contextLength: 2048
)
let estimatedGB = Double(estimatedBytes) / 1e9
let device = CrabInfer.getDevice()

if estimatedGB > device.availableMemoryGB * 0.8 {
    print("Model too large for this device")
}
```

## Step 7: Generation Statistics

After each generation, inspect performance metrics:

```swift
if let stats = engine.lastStats {
    print("""
      Tokens: \(stats.tokensGenerated)
      Speed: \(String(format: "%.1f", stats.tokensPerSecond)) tok/s
      TTFT: \(String(format: "%.0f", stats.timeToFirstTokenMs)) ms
      Backend: \(stats.computeBackend)
      Peak memory: \(String(format: "%.0f", stats.peakMemoryMB)) MB
    """)
}
```

## Next Steps

- **Smart Routing**: Fall back to cloud when the model is too large — see [Smart Routing Guide](smart-routing.md)
- **Agent Runtime**: Add tool calling and MCP to your app — see the `Assistant/` files in the iOS demo
- **Background Downloads**: Use `BGProcessingTask` to download models while the app is in the background
- **Siri Integration**: Expose your AI via App Intents — see `CrabInferIntents.swift` in the demo

## Full Example

See the complete iOS demo app at `examples/ios-demo/` which includes:
- Model catalog with download + progress
- Chat with streaming and stop sequences
- Device info bar with memory pressure
- AI assistant with tool calling
- Siri Shortcuts and Background Tasks
