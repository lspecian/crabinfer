/// CrabInfer — Swift SDK for on-device LLM inference
///
/// Provides a Swifty, async/await interface on top of the
/// UniFFI-generated bindings from the Rust core.
///
/// Usage:
///   let engine = try CrabInfer.Engine(modelPath: "path/to/model.gguf")
///   let response = try await engine.complete("Hello!")
///
///   // Or stream tokens:
///   for try await token in engine.stream("Tell me a story") {
///       print(token.text, terminator: "")
///   }

import Foundation

/// CrabInfer namespace
public enum CrabInfer {

    /// Get the CrabInfer SDK version
    public static var sdkVersion: String {
        version()
    }

    /// Detect current device capabilities
    public static func getDevice() -> Device {
        // Call the top-level UniFFI function (detectDevice in crabinfer_core.swift)
        let info: DeviceInfo = detectDevice()
        return Device(from: info)
    }

    /// Estimate peak memory usage for a model before loading (bytes).
    /// Reads only the GGUF header — fast (~ms).
    public static func estimateMemory(modelPath: String, contextLength: UInt32 = 4096) throws -> UInt64 {
        try estimateModelMemory(modelPath: modelPath, contextLength: contextLength)
    }

    // MARK: - Device

    /// Device capabilities
    public struct Device: Sendable {
        public let model: String
        public let totalMemoryGB: Double
        public let availableMemoryGB: Double
        public let hasMetalGPU: Bool
        public let hasNeuralEngine: Bool
        public let recommendedQuant: String
        public let maxModelSizeB: UInt32
        public let maxModelFileSizeBytes: UInt64
        public let chipName: String
        public let chipVariant: String
        public let recommendedContextLength: UInt32

        init(from info: DeviceInfo) {
            self.model = info.deviceModel
            self.totalMemoryGB = Double(info.totalMemoryBytes) / (1024.0 * 1024.0 * 1024.0)
            self.availableMemoryGB = Double(info.availableMemoryBytes) / (1024.0 * 1024.0 * 1024.0)
            self.hasMetalGPU = info.hasMetalGpu
            self.hasNeuralEngine = info.hasNeuralEngine
            self.recommendedQuant = info.recommendedQuant
            self.maxModelSizeB = info.maxModelSizeB
            self.maxModelFileSizeBytes = info.maxModelFileSizeBytes
            self.chipName = info.chipName
            self.chipVariant = info.chipVariant
            self.recommendedContextLength = info.recommendedContextLength
        }
    }

    // MARK: - Config

    /// Configuration for the inference engine
    public struct Config: Sendable {
        public var modelPath: String
        public var maxTokens: UInt32
        public var temperature: Float
        public var topP: Float
        public var contextLength: UInt32
        public var useMetal: Bool
        public var memoryLimitBytes: UInt64
        public var metallibPath: String

        public init(
            modelPath: String,
            maxTokens: UInt32 = 512,
            temperature: Float = 0.7,
            topP: Float = 0.9,
            contextLength: UInt32 = 2048,
            useMetal: Bool = true,
            memoryLimitBytes: UInt64 = 0,
            metallibPath: String = ""
        ) {
            self.modelPath = modelPath
            self.maxTokens = maxTokens
            self.temperature = temperature
            self.topP = topP
            self.contextLength = contextLength
            self.useMetal = useMetal
            self.memoryLimitBytes = memoryLimitBytes
            self.metallibPath = metallibPath
        }

        /// Convert to the UniFFI EngineConfig type
        func toEngineConfig() -> EngineConfig {
            EngineConfig(
                modelPath: modelPath,
                maxTokens: maxTokens,
                temperature: temperature,
                topP: topP,
                contextLength: contextLength,
                useMetal: useMetal,
                memoryLimitBytes: memoryLimitBytes,
                metallibPath: metallibPath
            )
        }
    }

    // MARK: - MemoryState

    /// Memory pressure levels
    public enum MemoryState: String, Sendable {
        case normal = "Normal"
        case warning = "Warning"
        case critical = "Critical"
        case terminal = "Terminal"

        init(from pressure: MemoryPressure) {
            switch pressure {
            case .normal: self = .normal
            case .warning: self = .warning
            case .critical: self = .critical
            case .terminal: self = .terminal
            }
        }
    }

    // MARK: - Token

    /// A generated token
    public struct Token: Sendable {
        public let text: String
        public let id: UInt32
        public let probability: Float
        public let isEnd: Bool

        init(from output: TokenOutput) {
            self.text = output.text
            self.id = output.tokenId
            self.probability = output.probability
            self.isEnd = output.isEndOfSequence
        }
    }

    // MARK: - Stats

    /// Generation statistics
    public struct Stats: Sendable {
        public let tokensGenerated: UInt32
        public let tokensPerSecond: Double
        public let timeToFirstTokenMs: Double
        public let totalTimeMs: Double
        public let peakMemoryMB: Double
        public let computeBackend: String

        init(from stats: GenerationStats) {
            self.tokensGenerated = stats.tokensGenerated
            self.tokensPerSecond = stats.tokensPerSecond
            self.timeToFirstTokenMs = stats.timeToFirstTokenMs
            self.totalTimeMs = stats.totalTimeMs
            self.peakMemoryMB = Double(stats.peakMemoryBytes) / (1024.0 * 1024.0)
            self.computeBackend = stats.computeBackend
        }
    }

    // MARK: - Model

    /// Loaded model metadata
    public struct Model: Sendable {
        public let name: String
        public let architecture: String
        public let parameterCount: UInt64
        public let quantization: String
        public let fileSizeBytes: UInt64
        public let contextLength: UInt32
        public let vocabSize: UInt32
        public let isMoE: Bool
        public let expertCount: UInt32
        public let expertUsedCount: UInt32
        public let activeParameterCount: UInt64

        init(from info: ModelInfo) {
            self.name = info.modelName
            self.architecture = info.architecture
            self.parameterCount = info.parameterCount
            self.quantization = info.quantization
            self.fileSizeBytes = info.fileSizeBytes
            self.contextLength = info.contextLength
            self.vocabSize = info.vocabSize
            self.isMoE = info.isMoe
            self.expertCount = info.expertCount
            self.expertUsedCount = info.expertUsedCount
            self.activeParameterCount = info.activeParameterCount
        }

        /// Formatted parameter count (e.g. "3.8B", "600M")
        public var formattedParams: String {
            if parameterCount >= 1_000_000_000 {
                return String(format: "%.1fB", Double(parameterCount) / 1e9)
            } else if parameterCount >= 1_000_000 {
                return String(format: "%.0fM", Double(parameterCount) / 1e6)
            } else {
                return "\(parameterCount)"
            }
        }
    }

    // MARK: - Engine

    /// The main inference engine
    public class Engine {
        private let rustEngine: CrabInferEngine
        private let config: Config

        /// Create an engine with a model path (auto-configures everything else)
        public convenience init(modelPath: String) throws {
            try self.init(config: Config(modelPath: modelPath))
        }

        /// Create an engine with full configuration
        public init(config: Config) throws {
            self.config = config
            self.rustEngine = try CrabInferEngine(config: config.toEngineConfig())
        }

        /// Load a model (can swap models after construction)
        public func loadModel(path: String) throws {
            try rustEngine.loadModel(modelPath: path)
        }

        /// Get info about the currently loaded model
        public func modelInfo() throws -> Model {
            Model(from: try rustEngine.modelInfo())
        }

        /// Whether a model is currently loaded
        public var isModelLoaded: Bool {
            rustEngine.isModelLoaded()
        }

        /// Generate a complete response (blocking, runs on background queue)
        public func complete(_ prompt: String, maxTokens: UInt32? = nil, temperature: Float? = nil) async throws -> String {
            let tokens = maxTokens ?? config.maxTokens
            let temp = temperature ?? config.temperature
            return try await Task.detached(priority: .userInitiated) { [rustEngine] in
                try rustEngine.complete(prompt: prompt, maxTokens: tokens, temperature: temp)
            }.value
        }

        /// Stream tokens as an AsyncSequence
        public func stream(_ prompt: String) -> TokenStream {
            TokenStream(engine: rustEngine, prompt: prompt, maxTokens: config.maxTokens)
        }

        /// Get the next token synchronously (for manual streaming control)
        public func nextToken(prompt: String) throws -> Token? {
            guard let output = try rustEngine.nextToken(prompt: prompt) else {
                return nil
            }
            return Token(from: output)
        }

        /// Reset generation state (clear KV cache, start fresh)
        public func reset() {
            rustEngine.reset()
        }

        /// Current memory pressure level
        public var memoryState: MemoryState {
            MemoryState(from: rustEngine.memoryPressure())
        }

        /// Statistics from last generation
        public var lastStats: Stats? {
            rustEngine.lastStats().map(Stats.init)
        }

        /// Trigger pressure-aware memory cleanup
        public func reduceMemory() {
            rustEngine.reduceMemory()
        }

        /// Unload the model to free memory
        public func unload() {
            rustEngine.unloadModel()
        }

        /// Current Metal GPU memory allocation in bytes
        public var metalAllocatedBytes: UInt64 {
            rustEngine.metalAllocatedBytes()
        }

        /// Run a load/unload stress test for memory leak detection
        public func stressTest(modelPath: String, cycles: UInt32, tokensPerCycle: UInt32) throws -> [String] {
            try rustEngine.stressTest(modelPath: modelPath, cycles: cycles, tokensPerCycle: tokensPerCycle)
        }
    }

    // MARK: - TokenStream (AsyncSequence)

    /// AsyncSequence for streaming tokens
    public struct TokenStream: AsyncSequence {
        public typealias Element = Token

        let engine: CrabInferEngine
        let prompt: String
        let maxTokens: UInt32

        public func makeAsyncIterator() -> TokenIterator {
            TokenIterator(engine: engine, prompt: prompt, remaining: maxTokens)
        }

        public struct TokenIterator: AsyncIteratorProtocol {
            let engine: CrabInferEngine
            let prompt: String
            var remaining: UInt32
            var done = false

            public mutating func next() async throws -> Token? {
                if done || remaining == 0 { return nil }

                let result = try await Task.detached(priority: .userInitiated) { [engine, prompt] in
                    try engine.nextToken(prompt: prompt)
                }.value

                guard let output = result else {
                    done = true
                    return nil
                }

                let token = Token(from: output)
                if token.isEnd {
                    done = true
                }
                remaining -= 1
                return token
            }
        }
    }
}

// MARK: - Router (Smart Routing)

extension CrabInfer {

    /// Smart router that automatically selects between local, self-hosted, and cloud providers
    /// based on memory pressure, model availability, network status, provider tier, and policy.
    ///
    /// Three-tier routing: on-device → self-hosted (LAN) → public cloud.
    ///
    /// Usage:
    /// ```swift
    /// let router = try CrabInfer.Router(
    ///     localConfig: CrabInfer.Config(modelPath: "model.gguf"),
    ///     cloudConfigs: [
    ///         CrabInfer.cloudConfig(provider: "openai", apiKey: key, model: "gpt-4o"),
    ///         CrabInfer.selfHostedConfig(provider: "ollama", baseUrl: "http://mac-studio:11434"),
    ///     ]
    /// )
    /// let response = try await router.complete(request)
    /// print(response.routingInfo) // "ollama (reason: SelfHostedAvailable, ...)"
    /// ```
    public class Router {
        private let rustRouter: CrabInferRouter

        /// Create a router with optional local and cloud/self-hosted providers.
        ///
        /// - Parameters:
        ///   - policy: Routing policy (default: `.localFirst`)
        ///   - privacyMode: If true, never route to cloud or self-hosted (default: false)
        ///   - dataSovereignty: If true, never route to public cloud (self-hosted still allowed)
        ///   - localConfig: Local engine config, or nil for cloud-only
        ///   - cloudConfigs: Provider configs (auto-classified into self-hosted/cloud by tier)
        public init(
            policy: RoutingPolicy = .localFirst,
            privacyMode: Bool = false,
            dataSovereignty: Bool = false,
            localConfig: Config? = nil,
            cloudConfigs: [ProviderConfig] = []
        ) throws {
            let routerConfig = RouterConfig(
                policy: policy,
                privacyMode: privacyMode,
                ollamaIsLocal: true,
                dataSovereignty: dataSovereignty
            )
            self.rustRouter = try CrabInferRouter(
                config: routerConfig,
                localConfig: localConfig?.toEngineConfig(),
                cloudConfigs: cloudConfigs
            )
        }

        /// Generate a complete response, routing automatically.
        public func complete(_ request: CompletionRequest) async throws -> CompletionResponse {
            try await Task.detached(priority: .userInitiated) { [rustRouter] in
                try rustRouter.complete(request: request)
            }.value
        }

        /// Stream tokens, routing automatically.
        public func stream(_ request: CompletionRequest) -> RouterTokenStream {
            RouterTokenStream(router: rustRouter, request: request)
        }

        /// Set network availability (call from NWPathMonitor handler).
        public func setNetworkAvailable(_ available: Bool) {
            rustRouter.setNetworkAvailable(available: available)
        }

        /// Routing decision from the last `complete()` or `stream()` call.
        public var lastRoutingDecision: RoutingDecision? {
            rustRouter.lastRoutingDecision()
        }

        /// All models available across all configured providers.
        public func availableModels() throws -> [ModelDescriptor] {
            try rustRouter.availableModels()
        }

        /// Whether any provider is available.
        public var isAvailable: Bool {
            rustRouter.isAvailable()
        }
    }

    // MARK: - RouterTokenStream

    /// AsyncSequence for streaming tokens through the router.
    public struct RouterTokenStream: AsyncSequence {
        public typealias Element = Token

        let router: CrabInferRouter
        let request: CompletionRequest

        public func makeAsyncIterator() -> RouterTokenIterator {
            RouterTokenIterator(router: router, request: request, started: false, done: false)
        }

        public struct RouterTokenIterator: AsyncIteratorProtocol {
            let router: CrabInferRouter
            let request: CompletionRequest
            var started: Bool
            var done: Bool

            public mutating func next() async throws -> Token? {
                if done { return nil }

                if !started {
                    try await Task.detached(priority: .userInitiated) { [router, request] in
                        try router.streamStart(request: request)
                    }.value
                    started = true
                }

                let result = try await Task.detached(priority: .userInitiated) { [router] in
                    try router.streamNext()
                }.value

                guard let output = result else {
                    done = true
                    return nil
                }

                let token = Token(from: output)
                if token.isEnd {
                    done = true
                }
                return token
            }
        }
    }
}

// MARK: - Model Catalog

extension CrabInfer {

    /// A model from the curated catalog with convenience properties.
    public struct CatalogModel: Sendable, Identifiable {
        /// The underlying catalog entry.
        public let entry: CatalogEntry

        public var id: String { entry.id }
        public var name: String { entry.name }
        public var sizeBytes: UInt64 { entry.sizeBytes }
        public var paramCountB: Float { entry.paramCountB }
        public var quant: String { entry.quant }
        public var isMoE: Bool { entry.isMoe }
        public var requiresAuth: Bool { entry.requiresAuth }

        /// Human-readable file size (e.g. "4.9 GB", "430 MB").
        public var formattedSize: String {
            if entry.sizeBytes >= 1_000_000_000 {
                return String(format: "%.1f GB", Double(entry.sizeBytes) / 1e9)
            } else {
                return String(format: "%.0f MB", Double(entry.sizeBytes) / 1e6)
            }
        }

        /// Human-readable parameter count (e.g. "8B", "30B/3B" for MoE).
        public var formattedParams: String {
            if entry.isMoe {
                return "\(entry.paramCountB)B/\(entry.activeParamCountB)B"
            } else {
                return "\(entry.paramCountB)B"
            }
        }

        /// Category as a string.
        public var categoryName: String {
            switch entry.category {
            case .general: return "General"
            case .code: return "Code"
            case .reasoning: return "Reasoning"
            }
        }

        init(from entry: CatalogEntry) {
            self.entry = entry
        }
    }

    /// Get the full model catalog.
    public static func catalog() -> [CatalogModel] {
        modelCatalog().map(CatalogModel.init)
    }

    /// Get models compatible with the current device.
    public static func catalogForDevice() -> [CatalogModel] {
        let device = detectDevice()
        return modelsForDevice(device: device).map(CatalogModel.init)
    }

    /// Get recommended models (best per category) for the current device.
    public static func recommendedModels() -> [CatalogModel] {
        let device = detectDevice()
        return crabinfer_core.recommendedModels(device: device).map(CatalogModel.init)
    }

    /// Look up a catalog model by ID.
    public static func catalogModel(id: String) -> CatalogModel? {
        catalogEntry(id: id).map(CatalogModel.init)
    }
}

// MARK: - Downloads

extension CrabInfer {

    /// Manages model downloads with async support.
    public class Downloads {
        private let manager: ModelDownloadManager

        /// Create with default storage directory.
        public init() throws {
            self.manager = try ModelDownloadManager()
        }

        /// Create with custom storage directory.
        public init(directory: String) throws {
            self.manager = try ModelDownloadManager.withDirectory(directory: directory)
        }

        /// Download a model asynchronously with progress reporting.
        ///
        /// - Parameters:
        ///   - model: A catalog model to download.
        ///   - onProgress: Optional progress callback (bytes downloaded, total bytes, phase).
        /// - Returns: A `DownloadedModel` with local file paths.
        public func download(
            _ model: CatalogModel,
            onProgress: ((UInt64, UInt64, String) -> Void)? = nil
        ) async throws -> DownloadedModel {
            let entry = model.entry
            let listener: SwiftProgressListener? = onProgress.map { SwiftProgressListener(handler: $0) }
            return try await Task.detached(priority: .userInitiated) { [manager] in
                try manager.download(
                    entry: entry,
                    listener: listener.map { $0 as DownloadProgressListener }
                )
            }.value
        }

        /// Check if a model is downloaded.
        public func isDownloaded(_ id: String) -> Bool {
            manager.isDownloaded(id: id)
        }

        /// Get the GGUF path for a downloaded model.
        public func modelPath(_ id: String) -> String? {
            manager.modelPath(id: id)
        }

        /// Get the tokenizer path for a downloaded model.
        public func tokenizerPath(_ id: String) -> String? {
            manager.tokenizerPath(id: id)
        }

        /// List all downloaded models.
        public func listDownloaded() -> [DownloadedModel] {
            manager.listDownloaded()
        }

        /// Delete a downloaded model.
        public func delete(_ id: String) throws {
            try manager.delete(id: id)
        }

        /// Total disk usage of all downloaded models.
        public var totalDiskUsage: UInt64 {
            manager.totalDiskUsage()
        }

        /// The storage directory path.
        public var storageDirectory: String {
            manager.storageDirectory()
        }
    }
}

/// Internal progress listener bridging Swift closure to UniFFI callback.
private class SwiftProgressListener: DownloadProgressListener {
    let handler: (UInt64, UInt64, String) -> Void

    init(handler: @escaping (UInt64, UInt64, String) -> Void) {
        self.handler = handler
    }

    func onProgress(bytesDownloaded: UInt64, bytesTotal: UInt64, phase: String) {
        handler(bytesDownloaded, bytesTotal, phase)
    }
}

// MARK: - SwiftUI Integration

#if canImport(SwiftUI)
import SwiftUI

extension CrabInfer {
    /// Observable wrapper for SwiftUI — drive UI from model inference
    @available(iOS 17.0, macOS 14.0, *)
    @Observable
    public class ObservableEngine {
        public var isLoading = false
        public var isGenerating = false
        public var currentOutput = ""
        public var memoryState: MemoryState = .normal
        public var error: Error?
        public var stats: Stats?

        private let engine: Engine

        public init(config: Config) throws {
            self.engine = try Engine(config: config)
        }

        /// Whether a model is loaded
        public var isModelLoaded: Bool {
            engine.isModelLoaded
        }

        /// Load a model, updating `isLoading` state
        public func loadModel(path: String) async {
            isLoading = true
            error = nil
            do {
                try await Task.detached(priority: .userInitiated) { [engine] in
                    try engine.loadModel(path: path)
                }.value
            } catch {
                self.error = error
            }
            isLoading = false
        }

        /// Generate a complete response, updating state throughout
        public func generate(_ prompt: String) async {
            isGenerating = true
            currentOutput = ""
            error = nil
            stats = nil

            do {
                let result = try await engine.complete(prompt)
                currentOutput = result
                stats = engine.lastStats
            } catch {
                self.error = error
            }

            memoryState = engine.memoryState
            isGenerating = false
        }

        /// Stream a response token by token, updating `currentOutput` in real time
        public func stream(_ prompt: String) async {
            isGenerating = true
            currentOutput = ""
            error = nil
            stats = nil

            do {
                for try await token in engine.stream(prompt) {
                    currentOutput += token.text
                }
                stats = engine.lastStats
            } catch {
                self.error = error
            }

            memoryState = engine.memoryState
            isGenerating = false
        }

        /// Reset the engine state
        public func reset() {
            engine.reset()
            currentOutput = ""
            stats = nil
        }

        /// Unload the model
        public func unload() {
            engine.unload()
            currentOutput = ""
            stats = nil
        }
    }
}
#endif

// MARK: - Credentials

extension CrabInfer {

    /// Credential management for cloud providers.
    ///
    /// Store API keys in the in-memory Rust credential manager.
    /// For persistence, load from iOS Keychain on app launch and call `setApiKey()`.
    ///
    /// ```swift
    /// // On app launch, load from Keychain:
    /// if let key = KeychainHelper.get("openai-api-key") {
    ///     CrabInfer.Credentials.setApiKey(provider: "openai", key: key)
    /// }
    ///
    /// // Then create providers normally (empty api_key falls back to manager):
    /// let config = ProviderConfig(providerType: "openai", apiKey: "", ...)
    /// let provider = try CrabInferProvider(config: config)
    /// ```
    public enum Credentials {

        /// Store an API key for a cloud provider.
        /// Provider names: "openai", "anthropic", "google", "ollama".
        public static func setApiKey(provider: String, key: String) {
            setCredential(provider: provider, apiKey: key)
        }

        /// Retrieve the stored API key for a provider.
        public static func getApiKey(provider: String) -> String? {
            getCredential(provider: provider)
        }

        /// Remove a stored API key.
        public static func removeApiKey(provider: String) {
            removeCredential(provider: provider)
        }

        /// Check if a key is stored for the given provider.
        public static func hasApiKey(provider: String) -> Bool {
            hasCredential(provider: provider)
        }

        /// List provider names that have stored keys.
        public static func configuredProviders() -> [String] {
            crabinfer.configuredProviders()
        }

        /// Validate a stored API key by making a lightweight API call.
        /// Returns true if valid, false if authentication fails.
        public static func validate(provider: String) async throws -> Bool {
            try await Task.detached(priority: .userInitiated) {
                try validateCredential(provider: provider)
            }.value
        }

        /// Load all keys from a dictionary (e.g., from Keychain batch read).
        public static func loadAll(_ credentials: [String: String]) {
            for (provider, key) in credentials {
                setApiKey(provider: provider, key: key)
            }
        }
    }

    // MARK: - Provider Config Builders

    /// Create a cloud provider config (OpenAI, Anthropic, Google).
    ///
    /// - Parameters:
    ///   - provider: Provider type: "openai", "anthropic", "google"
    ///   - apiKey: API key
    ///   - model: Default model ID (e.g., "gpt-4o", "claude-sonnet-4-20250514")
    ///   - baseUrl: Base URL override (empty = provider default)
    ///   - timeout: Request timeout in seconds (0 = default 30s)
    public static func cloudConfig(
        provider: String,
        apiKey: String,
        model: String = "",
        baseUrl: String = "",
        timeout: UInt32 = 0
    ) -> ProviderConfig {
        ProviderConfig(
            providerType: provider,
            apiKey: apiKey,
            baseUrl: baseUrl,
            defaultModel: model,
            timeoutSeconds: timeout,
            tierOverride: "cloud"
        )
    }

    /// Create a self-hosted provider config (Ollama, vLLM, or OpenAI-compatible proxy).
    ///
    /// - Parameters:
    ///   - provider: Provider type: "ollama", "vllm", or "openai" (for compatible proxies)
    ///   - baseUrl: Server URL (e.g., "http://mac-studio:11434")
    ///   - apiKey: Optional API key (empty if no auth)
    ///   - model: Default model ID (empty = use server default)
    ///   - timeout: Request timeout in seconds (0 = default 30s)
    public static func selfHostedConfig(
        provider: String,
        baseUrl: String,
        apiKey: String = "",
        model: String = "",
        timeout: UInt32 = 0
    ) -> ProviderConfig {
        ProviderConfig(
            providerType: provider,
            apiKey: apiKey,
            baseUrl: baseUrl,
            defaultModel: model,
            timeoutSeconds: timeout,
            tierOverride: "self_hosted"
        )
    }

    // MARK: - vLLM Provider

    /// vLLM provider with standard Provider methods plus vLLM-specific
    /// features: health checks, Prometheus metrics, and guided decoding.
    ///
    /// Usage:
    /// ```swift
    /// let vllm = try CrabInfer.Vllm(baseUrl: "http://my-vllm-server:8000")
    /// let healthy = try await vllm.health()
    /// let metrics = try await vllm.serverMetrics()
    /// let response = try await vllm.complete(request)
    /// ```
    public class Vllm {
        private let inner: CrabInferVllm

        /// Create a vLLM provider.
        /// - Parameters:
        ///   - baseUrl: vLLM server URL (e.g., "http://localhost:8000")
        ///   - apiKey: Optional API key (empty if no auth)
        ///   - defaultModel: Default model ID (empty = use server default)
        ///   - timeout: Request timeout in seconds (0 = default 30s)
        public init(baseUrl: String, apiKey: String = "", defaultModel: String = "", timeout: UInt32 = 0) throws {
            let config = ProviderConfig(
                providerType: "vllm",
                apiKey: apiKey,
                baseUrl: baseUrl,
                defaultModel: defaultModel,
                timeoutSeconds: timeout,
                tierOverride: ""
            )
            self.inner = try CrabInferVllm(config: config)
        }

        /// Provider name: "vllm".
        public var name: String { inner.name() }

        /// Generate a complete response (async).
        public func complete(_ request: CompletionRequest) async throws -> CompletionResponse {
            try await Task.detached(priority: .userInitiated) { [inner] in
                try inner.complete(request: request)
            }.value
        }

        /// Generate a complete response with vLLM-specific options (async).
        public func complete(_ request: CompletionRequest, options: VllmCompletionOptions) async throws -> CompletionResponse {
            try await Task.detached(priority: .userInitiated) { [inner] in
                try inner.completeWithOptions(request: request, options: options)
            }.value
        }

        /// Check vLLM server health (async).
        public func health() async throws -> Bool {
            try await Task.detached(priority: .userInitiated) { [inner] in
                try inner.health()
            }.value
        }

        /// Scrape key Prometheus metrics from the vLLM server (async).
        public func serverMetrics() async throws -> VllmServerMetrics {
            try await Task.detached(priority: .userInitiated) { [inner] in
                try inner.serverMetrics()
            }.value
        }

        /// List models currently served by the vLLM instance (async, live API call).
        public func availableModels() async throws -> [ModelDescriptor] {
            try await Task.detached(priority: .userInitiated) { [inner] in
                try inner.availableModels()
            }.value
        }

        /// Check if the vLLM server is reachable.
        public var isAvailable: Bool { inner.isAvailable() }

        /// Stream tokens from the vLLM server.
        public func stream(_ request: CompletionRequest) -> VllmTokenStream {
            VllmTokenStream(inner: inner, request: request)
        }
    }

    /// AsyncSequence that yields tokens from a vLLM streaming request.
    public struct VllmTokenStream: AsyncSequence {
        public typealias Element = TokenOutput

        fileprivate let inner: CrabInferVllm
        fileprivate let request: CompletionRequest

        public func makeAsyncIterator() -> AsyncIterator {
            AsyncIterator(inner: inner, request: request)
        }

        public struct AsyncIterator: AsyncIteratorProtocol {
            let inner: CrabInferVllm
            let request: CompletionRequest
            var started = false

            public mutating func next() async throws -> TokenOutput? {
                if !started {
                    try await Task.detached(priority: .userInitiated) { [inner, request] in
                        try inner.streamStart(request: request)
                    }.value
                    started = true
                }

                let token = try await Task.detached(priority: .userInitiated) { [inner] in
                    try inner.streamNext()
                }.value

                guard let token = token else { return nil }
                if token.isEndOfSequence { return nil }
                return token
            }
        }
    }
}
