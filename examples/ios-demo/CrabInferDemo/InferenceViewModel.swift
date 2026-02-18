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
        id: "qwen3-0.6b-q8",
        name: "Qwen3 0.6B (Q8_0)",
        sizeMB: 596,
        architecture: "qwen3",
        modelURL: URL(string: "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf")!,
        tokenizerURL: URL(string: "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer.json")!
    ),
    ModelEntry(
        id: "qwen3-1.7b-q4",
        name: "Qwen3 1.7B (Q4_K_M)",
        sizeMB: 1056,
        architecture: "qwen3",
        modelURL: URL(string: "https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf")!,
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

// MARK: - ViewModel

@MainActor
class InferenceViewModel: ObservableObject {
    @Published var deviceInfo = "Detecting..."
    @Published var modelStatus = "No model loaded"
    @Published var output = ""
    @Published var isGenerating = false
    @Published var isLoading = false
    @Published var isModelLoaded = false
    @Published var pressureLevel: MemoryPressure = .normal
    @Published var pressureLabel = "Normal"
    @Published var lastGenerationStats: GenerationStats? = nil
    @Published var generatingStatus = ""

    // Download state
    @Published var isDownloading = false
    @Published var downloadProgress: Double = 0
    @Published var downloadStatus = ""

    // Stress test state
    @Published var isStressTesting = false
    @Published var stressTestLog: [String] = []

    // Model selection
    @Published var selectedModelId: String = modelCatalog[0].id
    @Published var downloadedModelIds: [String] = []
    @Published var activeModelId: String? = nil

    private var engine: CrabInferEngine?

    /// Persistent Metal-inference thread.
    ///
    /// All Candle/Metal work runs on this single OS thread for the lifetime of
    /// the app. A plain DispatchQueue is not enough because:
    ///   - GCD worker threads have no RunLoop -> Metal completion callbacks never fire
    ///   - GCD recycles threads across async blocks -> Metal's device affinity breaks
    ///   - Lazy shader compilation can deadlock a plain serial queue (mutex re-entry)
    /// See InferenceThread.swift for the full explanation.
    private let inferenceThread = InferenceThread()

    private var docsDir: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    /// Directory for a given model's files
    private func modelDir(for id: String) -> URL {
        docsDir.appendingPathComponent("models/\(id)")
    }

    init() {
        let info = detectDevice()
        let ramGB = info.totalMemoryBytes / (1024 * 1024 * 1024)
        let freeGB = info.availableMemoryBytes / (1024 * 1024 * 1024)
        deviceInfo = """
        \(info.deviceModel)
        RAM: \(ramGB) GB (\(freeGB) GB free)
        Metal: \(info.hasMetalGpu ? "Yes" : "No")
        Recommended: \(info.recommendedQuant) (up to \(info.maxModelSizeB)B)
        """
        refreshDownloadedModels()
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

    // MARK: - Load / Unload / Switch

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
                    maxTokens: 256,
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
        output = ""
        lastGenerationStats = nil
        modelStatus = downloadedModelIds.isEmpty
            ? "No model — pick one to download"
            : "Model ready to load"
    }

    // MARK: - Stress Test

    func runStressTest(cycles: Int = 10, tokensPerCycle: Int = 10) async {
        guard isSelectedModelDownloaded else { return }

        // Unload any current model first
        if isModelLoaded { unloadCurrentModel() }

        isStressTesting = true
        stressTestLog = ["Starting stress test: \(cycles) cycles, \(tokensPerCycle) tokens each..."]

        let dir = modelDir(for: selectedModelId)
        let modelPath = dir.appendingPathComponent("model.gguf").path

        do {
            let config = EngineConfig(
                modelPath: "",
                maxTokens: UInt32(tokensPerCycle),
                temperature: 0.7,
                topP: 0.9,
                contextLength: 4096,
                useMetal: true,
                memoryLimitBytes: 0,
                metallibPath: findMetallibDir()
            )
            let eng = try await inferenceThread.perform {
                try CrabInferEngine(config: config)
            }

            let log = try await inferenceThread.perform {
                try eng.stressTest(
                    modelPath: modelPath,
                    cycles: UInt32(cycles),
                    tokensPerCycle: UInt32(tokensPerCycle)
                )
            }

            stressTestLog = log
        } catch {
            stressTestLog.append("ERROR: \(error.localizedDescription)")
        }

        isStressTesting = false
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

    // MARK: - Chat templates

    /// Wrap raw user text in the correct instruct/chat template for the loaded model.
    private func chatPrompt(_ userText: String) -> String {
        guard let id = activeModelId,
              let entry = modelCatalog.first(where: { $0.id == id }) else {
            // Fallback: ChatML (works with most models)
            return "<|im_start|>user\n\(userText)<|im_end|>\n<|im_start|>assistant\n"
        }

        switch entry.architecture {
        case "qwen3", "qwen2":
            return "<|im_start|>user\n\(userText)<|im_end|>\n<|im_start|>assistant\n"
        case "gemma3", "gemma2", "gemma":
            return "<start_of_turn>user\n\(userText)<end_of_turn>\n<start_of_turn>model\n"
        case "phi3":
            return "<|user|>\n\(userText)<|end|>\n<|assistant|>\n"
        default:
            return "<|im_start|>user\n\(userText)<|im_end|>\n<|im_start|>assistant\n"
        }
    }

    // MARK: - Generate

    func generate(prompt: String, maxTokens: UInt32) async {
        guard let engine else { return }
        isGenerating = true
        output = ""
        lastGenerationStats = nil
        generatingStatus = "Starting inference..."

        engine.reset()

        let eng = engine
        let promptText = chatPrompt(prompt)

        let stream = inferenceThread.stream { yieldToken in
            for _ in 0..<maxTokens {
                do {
                    guard let tok = try eng.nextToken(prompt: promptText) else { break }
                    if tok.isEndOfSequence { break }
                    if !yieldToken(tok.text) { break }
                } catch {
                    _ = yieldToken("\n[Error: \(error.localizedDescription)]")
                    break
                }
            }
        }

        var tokenCount: UInt32 = 0
        for await text in stream {
            output += text
            tokenCount += 1
            if tokenCount == 1 {
                generatingStatus = "Streaming tokens..."
            }
            if tokenCount % 8 == 0 { updatePressure() }
        }

        generatingStatus = ""

        engine.reset()
        lastGenerationStats = engine.lastStats()
        isGenerating = false
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
