import Foundation

// MARK: - Download helper with progress tracking

class ModelDownloader: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private var continuation: CheckedContinuation<URL, Error>?
    private let onProgress: @Sendable (Double) -> Void

    init(onProgress: @escaping @Sendable (Double) -> Void) {
        self.onProgress = onProgress
    }

    func download(from url: URL) async throws -> URL {
        try await withCheckedThrowingContinuation { cont in
            self.continuation = cont
            let config = URLSessionConfiguration.default
            config.timeoutIntervalForResource = 3600
            let session = URLSession(configuration: config, delegate: self, delegateQueue: nil)
            session.downloadTask(with: url).resume()
        }
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didFinishDownloadingTo location: URL) {
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try? FileManager.default.copyItem(at: location, to: tmp)
        continuation?.resume(returning: tmp)
        continuation = nil
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64,
                    totalBytesWritten: Int64,
                    totalBytesExpectedToWrite: Int64) {
        guard totalBytesExpectedToWrite > 0 else { return }
        let progress = Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)
        onProgress(progress)
    }

    func urlSession(_ session: URLSession, task: URLSessionTask,
                    didCompleteWithError error: Error?) {
        if let error {
            continuation?.resume(throwing: error)
            continuation = nil
        }
    }
}

// MARK: - Model catalog

struct ModelEntry: Identifiable, Hashable {
    let id: String           // unique key, used as folder name
    let name: String         // display name
    let sizeMB: Int          // approximate download size
    let architecture: String // for chat template selection
    let modelURL: URL
    let tokenizerURL: URL
}

/// Available models for download. Sorted smallest-first so they appear
/// in a natural order on memory-constrained devices.
///
/// NOTE: All tokenizer URLs must be ungated (no HuggingFace login required).
/// Google Gemma models are gated and their tokenizer.json returns 401 without auth,
/// so we only include models with fully public tokenizers.
let modelCatalog: [ModelEntry] = [
    ModelEntry(
        id: "smollm2-360m-q8",
        name: "SmolLM2 360M (Q8_0)",
        sizeMB: 386,
        architecture: "llama",
        modelURL: URL(string: "https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF/resolve/main/smollm2-360m-instruct-q8_0.gguf")!,
        tokenizerURL: URL(string: "https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/resolve/main/tokenizer.json")!
    ),
    ModelEntry(
        id: "qwen3-0.6b-q4",
        name: "Qwen3 0.6B (Q4_K_M)",
        sizeMB: 380,
        architecture: "qwen3",
        modelURL: URL(string: "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf")!,
        tokenizerURL: URL(string: "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer.json")!
    ),
    ModelEntry(
        id: "qwen3-1.7b-q3",
        name: "Qwen3 1.7B (Q3_K_S)",
        sizeMB: 750,
        architecture: "qwen3",
        modelURL: URL(string: "https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q3_K_S.gguf")!,
        tokenizerURL: URL(string: "https://huggingface.co/Qwen/Qwen3-1.7B/resolve/main/tokenizer.json")!
    ),
    ModelEntry(
        id: "phi3-3.8b-q4",
        name: "Phi-3 Mini 3.8B (Q4)",
        sizeMB: 2300,
        architecture: "phi3",
        modelURL: URL(string: "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf")!,
        tokenizerURL: URL(string: "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/resolve/main/tokenizer.json")!
    ),
]

// MARK: - Chat message model

struct ChatMessage: Identifiable {
    let id = UUID()
    let role: Role
    var content: String
    var stats: GenerationStats?
    var isError: Bool = false

    enum Role {
        case user
        case assistant
    }
}

// MARK: - ViewModel

@MainActor
class InferenceViewModel: ObservableObject {
    @Published var deviceInfo = "Detecting..."
    @Published var deviceModel = ""
    @Published var deviceRAMGB: UInt64 = 0
    @Published var deviceHasMetal = false
    @Published var modelStatus = "No model loaded"
    @Published var isGenerating = false
    @Published var isLoading = false
    @Published var isModelLoaded = false
    @Published var pressureLevel: MemoryPressure = .normal
    @Published var pressureLabel = "Normal"

    // Chat state
    @Published var messages: [ChatMessage] = []

    // Download state
    @Published var isDownloading = false
    @Published var downloadProgress: Double = 0
    @Published var downloadStatus = ""

    // Model selection
    @Published var selectedModelId: String = modelCatalog[0].id
    @Published var downloadedModelIds: [String] = []
    @Published var activeModelId: String? = nil
    @Published var activeModelName: String? = nil

    private var engine: CrabInferEngine?

    /// Persistent Metal-inference thread.
    /// Lazy so the semaphore wait in InferenceThread.init() doesn't block
    /// SwiftUI's first render cycle (which causes the black screen on launch).
    private lazy var inferenceThread = InferenceThread()

    private var docsDir: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    /// Directory for a given model's files
    private func modelDir(for id: String) -> URL {
        docsDir.appendingPathComponent("models/\(id)")
    }

    @Published var isReady = false

    init() {
        refreshDownloadedModels()
        // Detect device on a background thread so the UI renders immediately.
        // Task {} inherits @MainActor from the class, so we must use Task.detached
        // to actually leave the main thread for the blocking FFI call.
        Task.detached { [weak self] in
            let info = detectDevice()
            await MainActor.run {
                guard let self else { return }
                let ramGB = info.totalMemoryBytes / (1024 * 1024 * 1024)
                let freeGB = info.availableMemoryBytes / (1024 * 1024 * 1024)
                self.deviceModel = info.deviceModel
                self.deviceRAMGB = ramGB
                self.deviceHasMetal = info.hasMetalGpu
                self.deviceInfo = """
                \(info.deviceModel)
                RAM: \(ramGB) GB (\(freeGB) GB free)
                Metal: \(info.hasMetalGpu ? "Yes" : "No")
                Recommended: \(info.recommendedQuant) (up to \(info.maxModelSizeB)B)
                """
                self.isReady = true
            }
        }
    }

    /// Scan the models directory to find which models have been downloaded.
    func refreshDownloadedModels() {
        var found: [String] = []
        for entry in modelCatalog {
            let dir = modelDir(for: entry.id)
            let modelFile = dir.appendingPathComponent("model.gguf")
            let tokFile = dir.appendingPathComponent("tokenizer.json")
            if FileManager.default.fileExists(atPath: modelFile.path) &&
               FileManager.default.fileExists(atPath: tokFile.path) {
                found.append(entry.id)
            }
        }
        downloadedModelIds = found

        if !isModelLoaded {
            if found.isEmpty {
                modelStatus = "No model — pick one to download"
            } else {
                modelStatus = "Model ready to load"
            }
        }
    }

    /// Whether the currently selected model has been downloaded.
    var isSelectedModelDownloaded: Bool {
        downloadedModelIds.contains(selectedModelId)
    }

    /// The catalog entry for the currently selected model.
    var selectedModel: ModelEntry {
        modelCatalog.first(where: { $0.id == selectedModelId }) ?? modelCatalog[0]
    }

    /// Short summary for the device info bar
    var deviceBarSummary: String {
        var parts = [deviceModel, "\(deviceRAMGB) GB"]
        if deviceHasMetal { parts.append("Metal") }
        if let name = activeModelName {
            parts.append(name)
        }
        return parts.joined(separator: " · ")
    }

    // MARK: - Download

    func downloadSelectedModel() async {
        let entry = selectedModel
        isDownloading = true
        downloadProgress = 0
        downloadStatus = "Downloading tokenizer..."

        let dir = modelDir(for: entry.id)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        let tokenizerDest = dir.appendingPathComponent("tokenizer.json")
        let modelDest = dir.appendingPathComponent("model.gguf")

        // Download tokenizer
        do {
            let downloader = ModelDownloader { [weak self] progress in
                Task { @MainActor in
                    self?.downloadProgress = progress * 0.01
                }
            }
            let tmpFile = try await downloader.download(from: entry.tokenizerURL)

            // Validate: tokenizer.json must start with '{' (not HTML error page)
            if let data = try? Data(contentsOf: tmpFile, options: .mappedIfSafe),
               let firstByte = data.first, firstByte != UInt8(ascii: "{") {
                downloadStatus = "Tokenizer download failed:\nReceived error page instead of JSON (model may require HuggingFace auth)"
                try? FileManager.default.removeItem(at: tmpFile)
                isDownloading = false
                return
            }

            if FileManager.default.fileExists(atPath: tokenizerDest.path) {
                try FileManager.default.removeItem(at: tokenizerDest)
            }
            try FileManager.default.moveItem(at: tmpFile, to: tokenizerDest)
        } catch {
            downloadStatus = "Tokenizer download failed:\n\(error.localizedDescription)"
            isDownloading = false
            return
        }

        // Download model weights
        downloadStatus = "Downloading \(entry.name) (\(entry.sizeMB) MB)..."
        downloadProgress = 0.01

        do {
            let sizeMB = entry.sizeMB
            let name = entry.name
            let downloader = ModelDownloader { [weak self] progress in
                Task { @MainActor in
                    self?.downloadProgress = 0.01 + progress * 0.99
                    let pct = Int((0.01 + progress * 0.99) * 100)
                    let mbDone = Int(progress * Double(sizeMB))
                    self?.downloadStatus = "Downloading \(name)...\n\(mbDone) / \(sizeMB) MB (\(pct)%)"
                }
            }
            let tmpFile = try await downloader.download(from: entry.modelURL)

            // Validate: GGUF files start with magic bytes "GGUF"
            if let data = try? Data(contentsOf: tmpFile, options: .mappedIfSafe),
               data.count >= 4 {
                let magic = String(data: data[0..<4], encoding: .ascii)
                if magic != "GGUF" {
                    downloadStatus = "Model download failed:\nFile is not a valid GGUF (may require HuggingFace auth)"
                    try? FileManager.default.removeItem(at: tmpFile)
                    isDownloading = false
                    return
                }
            }

            if FileManager.default.fileExists(atPath: modelDest.path) {
                try FileManager.default.removeItem(at: modelDest)
            }
            try FileManager.default.moveItem(at: tmpFile, to: modelDest)
        } catch {
            downloadStatus = "Model download failed:\n\(error.localizedDescription)"
            isDownloading = false
            return
        }

        downloadProgress = 1.0
        downloadStatus = "Download complete!"
        isDownloading = false
        refreshDownloadedModels()
    }

    /// Delete a downloaded model's files.
    func deleteModel(id: String) {
        let dir = modelDir(for: id)
        try? FileManager.default.removeItem(at: dir)
        if activeModelId == id {
            unloadCurrentModel()
        }
        refreshDownloadedModels()
    }

    // MARK: - Load / Unload

    func loadModel(id: String) async {
        // Unload any previously loaded model first
        if isModelLoaded {
            unloadCurrentModel()
        }

        isLoading = true
        modelStatus = "Loading model (Metal)..."

        let dir = modelDir(for: id)
        let modelPath = dir.appendingPathComponent("model.gguf").path

        do {
            let (eng, info) = try await inferenceThread.perform {
                let bundlePath = Bundle.main.bundlePath
                let candidatePaths = [
                    bundlePath + "/metallibs",
                    Bundle.main.resourcePath.map { $0 + "/metallibs" } ?? "",
                    bundlePath,
                ]

                var metallibDir = ""
                for candidate in candidatePaths where !candidate.isEmpty {
                    let probe = candidate + "/quantized.metallib"
                    if FileManager.default.fileExists(atPath: probe) {
                        metallibDir = candidate
                        NSLog("[CrabInfer-Swift] Found metallibs at: %@", candidate)
                        break
                    }
                }

                if metallibDir.isEmpty {
                    NSLog("[CrabInfer-Swift] WARNING: No metallib files found in app bundle!")
                    NSLog("[CrabInfer-Swift] Bundle path: %@", bundlePath)
                    if let contents = try? FileManager.default.contentsOfDirectory(atPath: bundlePath) {
                        let metalFiles = contents.filter { $0.hasSuffix(".metallib") || $0 == "metallibs" }
                        NSLog("[CrabInfer-Swift] Metal-related files in bundle: %@", metalFiles.description)
                        if metalFiles.isEmpty {
                            NSLog("[CrabInfer-Swift] All bundle items: %@", contents.prefix(30).description)
                        }
                    }
                }

                let config = EngineConfig(
                    modelPath: "",
                    maxTokens: 512,
                    temperature: 0.7,
                    topP: 0.9,
                    contextLength: 4096,
                    useMetal: true,
                    memoryLimitBytes: 0,
                    metallibPath: metallibDir
                )
                NSLog("[CrabInfer-Swift] EngineConfig metallibPath='%@'", metallibDir)
                let eng = try CrabInferEngine(config: config)
                try eng.loadModel(modelPath: modelPath)
                let info = try eng.modelInfo()
                return (eng, info)
            }

            engine = eng
            isModelLoaded = true
            activeModelId = id
            activeModelName = modelCatalog.first(where: { $0.id == id })?.name
            let mb = info.fileSizeBytes / (1024 * 1024)
            let backend = info.architecture
            modelStatus = """
            \(info.modelName)
            \(backend) | \(info.quantization) | \(mb) MB
            \(info.parameterCount / 1_000_000)M params | ctx \(info.contextLength)
            """
            updatePressure()
        } catch {
            modelStatus = "Load failed:\n\(error.localizedDescription)"
        }

        isLoading = false
    }

    func unloadCurrentModel() {
        engine?.unloadModel()
        engine = nil
        isModelLoaded = false
        activeModelId = nil
        activeModelName = nil
        messages = []
        modelStatus = downloadedModelIds.isEmpty
            ? "No model — pick one to download"
            : "Model ready to load"
    }

    // MARK: - Chat templates

    private let systemPrompt = "You are a helpful, concise assistant running on-device. Answer the user's question directly and briefly."

    /// Wrap raw user text in the correct instruct/chat template for the loaded model.
    private func chatPrompt(_ userText: String) -> String {
        guard let id = activeModelId,
              let entry = modelCatalog.first(where: { $0.id == id }) else {
            return "<|im_start|>system\n\(systemPrompt)<|im_end|>\n<|im_start|>user\n\(userText)<|im_end|>\n<|im_start|>assistant\n"
        }

        switch entry.architecture {
        case "qwen3":
            // /no_think disables Qwen3's chain-of-thought <think> blocks,
            // which waste tokens and can trigger Rust UTF-8 boundary errors with emoji.
            return "<|im_start|>system\n\(systemPrompt)\n/no_think<|im_end|>\n<|im_start|>user\n\(userText)<|im_end|>\n<|im_start|>assistant\n"
        case "qwen2":
            return "<|im_start|>system\n\(systemPrompt)<|im_end|>\n<|im_start|>user\n\(userText)<|im_end|>\n<|im_start|>assistant\n"
        case "gemma3", "gemma2", "gemma":
            return "<start_of_turn>user\n\(systemPrompt)\n\n\(userText)<end_of_turn>\n<start_of_turn>model\n"
        case "phi3":
            return "<|system|>\n\(systemPrompt)<|end|>\n<|user|>\n\(userText)<|end|>\n<|assistant|>\n"
        default:
            return "<|im_start|>system\n\(systemPrompt)<|im_end|>\n<|im_start|>user\n\(userText)<|im_end|>\n<|im_start|>assistant\n"
        }
    }

    /// Stop sequences that indicate the model has finished its turn.
    /// Includes both special token markers (which may or may not appear as literal
    /// text depending on how the tokenizer decodes them) and plain-text fallbacks
    /// for when models emit raw turn boundaries.
    private var stopSequences: [String] {
        // Architecture-specific special tokens
        var sequences: [String] = []

        if let id = activeModelId,
           let entry = modelCatalog.first(where: { $0.id == id }) {
            switch entry.architecture {
            case "qwen3", "qwen2":
                sequences += ["<|im_end|>", "<|im_start|>"]
            case "gemma3", "gemma2", "gemma":
                sequences += ["<end_of_turn>", "<start_of_turn>"]
            case "phi3":
                sequences += ["<|end|>", "<|user|>"]
            default:
                sequences += ["<|im_end|>", "<|im_start|>"]
            }
        } else {
            sequences += ["<|im_end|>", "<|im_start|>"]
        }

        // Plain-text fallbacks — catch models that decode turn markers as raw text.
        // Use \n prefix to avoid false positives mid-sentence.
        sequences += ["\nuser\n", "\nUser\n", "\nassistant\n", "\nAssistant\n",
                      "\nsystem\n", "\nSystem\n",
                      "\nuser:", "\nUser:", "\nassistant:", "\nAssistant:",
                      "\nsystem:", "\nSystem:",
                      "\n<user>", "\n<assistant>", "\n<system>"]

        return sequences
    }

    /// Check if any stop sequence appears in the text.
    /// Returns the text truncated before the stop sequence, or nil if none found.
    private func truncateAtStopSequence(_ text: String) -> String? {
        for stop in stopSequences {
            if let range = text.range(of: stop) {
                return String(text[text.startIndex..<range.lowerBound])
            }
        }
        return nil
    }

    /// Strip `<think>...</think>` blocks from model output.
    /// Qwen3 is a "thinking" model that emits chain-of-thought reasoning inside
    /// these tags before producing the actual response. During streaming:
    /// - If `<think>` is found but `</think>` hasn't appeared yet, return empty
    ///   string (still thinking).
    /// - If the full block is found, return only the content after `</think>`.
    /// - If no `<think>` tag is present, return the text unchanged.
    private func stripThinkBlocks(_ text: String) -> String {
        guard let thinkStart = text.range(of: "<think>") else {
            return text
        }
        guard let thinkEnd = text.range(of: "</think>") else {
            // Still inside think block — show nothing yet
            return String(text[text.startIndex..<thinkStart.lowerBound])
        }
        // Full think block found — return content after it
        let before = String(text[text.startIndex..<thinkStart.lowerBound])
        let after = String(text[thinkEnd.upperBound...])
        return (before + after).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func findMetallibDir() -> String {
        let bundlePath = Bundle.main.bundlePath
        let candidates = [
            bundlePath + "/metallibs",
            Bundle.main.resourcePath.map { $0 + "/metallibs" } ?? "",
            bundlePath,
        ]
        for candidate in candidates where !candidate.isEmpty {
            let probe = candidate + "/quantized.metallib"
            if FileManager.default.fileExists(atPath: probe) {
                return candidate
            }
        }
        return ""
    }

    // MARK: - Chat generation

    func sendMessage(_ text: String) async {
        guard engine != nil else {
            messages.append(ChatMessage(role: .assistant, content: "Please load a model first.", isError: true))
            return
        }

        // Add user message
        messages.append(ChatMessage(role: .user, content: text))

        // Add empty assistant message that we'll stream into
        messages.append(ChatMessage(role: .assistant, content: ""))
        let assistantIndex = messages.count - 1

        isGenerating = true
        engine!.reset()

        let eng = engine!
        let promptText = chatPrompt(text)

        let stream = inferenceThread.stream { yieldToken in
            for _ in 0..<UInt32(512) {
                do {
                    guard let tok = try eng.nextToken(prompt: promptText) else { break }
                    if tok.isEndOfSequence { break }
                    if !yieldToken(tok.text) { break }
                } catch {
                    NSLog("[CrabInfer-Swift] Token generation error: %@", "\(error)")
                    _ = yieldToken("\n[Generation stopped due to an error]")
                    break
                }
            }
        }

        var tokenCount: UInt32 = 0
        var hitStop = false
        var rawContent = ""
        for await token in stream {
            rawContent += token
            tokenCount += 1

            // Check if a stop sequence appeared in the raw accumulated text
            if let truncated = truncateAtStopSequence(rawContent) {
                rawContent = truncated
                hitStop = true
                messages[assistantIndex].content = stripThinkBlocks(rawContent)
                break
            }

            // Throttle UI updates — updating @Published on every token at ~37 tok/s
            // causes excessive SwiftUI re-renders and makes the UI feel sluggish.
            // Update every 4 tokens (~9 updates/s) for smooth streaming without lag.
            if tokenCount % 4 == 0 {
                messages[assistantIndex].content = stripThinkBlocks(rawContent)
            }

            if tokenCount % 8 == 0 { updatePressure() }
        }

        // Final update with clean content — always fires after streaming ends
        messages[assistantIndex].content = stripThinkBlocks(rawContent)
            .trimmingCharacters(in: .whitespacesAndNewlines)

        engine?.reset()
        messages[assistantIndex].stats = engine?.lastStats()
        isGenerating = false
    }

    func stopGenerating() {
        // Engine reset will cause nextToken to return nil on next call
        engine?.reset()
    }

    private func updatePressure() {
        guard let engine else { return }
        let p = engine.memoryPressure()
        pressureLevel = p
        switch p {
        case .normal:   pressureLabel = "Normal"
        case .warning:  pressureLabel = "Warning"
        case .critical: pressureLabel = "Critical"
        case .terminal: pressureLabel = "TERMINAL"
        }
    }
}
