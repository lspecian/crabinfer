/// CrabInfer — Swift SDK for on-device LLM inference
///
/// This provides a Swifty, async/await interface on top of the
/// UniFFI-generated bindings from the Rust core.
///
/// Usage:
///   let engine = try CrabInfer.Engine(modelPath: "path/to/model.gguf")
///   let response = try await engine.complete("Hello!")
///   
///   // Or stream tokens:
///   for try await token in engine.stream("Tell me a story") {
///       print(token, terminator: "")
///   }

import Foundation
// import CrabInferCore  // Will be available after build.sh generates bindings

/// CrabInfer namespace
public enum CrabInfer {
    
    /// Get the CrabInfer SDK version
    public static var sdkVersion: String {
        // Will call: crabinfer.version()
        "0.1.0"
    }
    
    /// Detect current device capabilities
    public static func detectDevice() -> Device {
        // Will call: crabinfer.detect_device()
        // For now, return stub
        Device(
            model: "stub",
            totalMemoryGB: 6.0,
            availableMemoryGB: 3.5,
            hasMetalGPU: true,
            hasNeuralEngine: true,
            recommendedQuant: "Q4_K_M",
            maxModelSizeB: 3
        )
    }
    
    /// Device capabilities
    public struct Device {
        public let model: String
        public let totalMemoryGB: Double
        public let availableMemoryGB: Double
        public let hasMetalGPU: Bool
        public let hasNeuralEngine: Bool
        public let recommendedQuant: String
        public let maxModelSizeB: Int
    }
    
    /// Configuration for the inference engine
    public struct Config {
        public var modelPath: String
        public var maxTokens: Int
        public var temperature: Float
        public var topP: Float
        public var contextLength: Int
        public var useMetal: Bool
        public var memoryLimitMB: Int?
        
        public init(
            modelPath: String,
            maxTokens: Int = 512,
            temperature: Float = 0.7,
            topP: Float = 0.9,
            contextLength: Int = 2048,
            useMetal: Bool = true,
            memoryLimitMB: Int? = nil
        ) {
            self.modelPath = modelPath
            self.maxTokens = maxTokens
            self.temperature = temperature
            self.topP = topP
            self.contextLength = contextLength
            self.useMetal = useMetal
            self.memoryLimitMB = memoryLimitMB
        }
    }
    
    /// Memory pressure levels
    public enum MemoryState: String {
        case normal = "Normal"
        case warning = "Warning — consider reducing context"
        case critical = "Critical — falling back to CPU"
        case terminal = "Terminal — unloading model"
    }
    
    /// A generated token
    public struct Token {
        public let text: String
        public let id: Int
        public let probability: Float
        public let isEnd: Bool
    }
    
    /// Generation statistics
    public struct Stats {
        public let tokensGenerated: Int
        public let tokensPerSecond: Double
        public let timeToFirstTokenMs: Double
        public let totalTimeMs: Double
        public let peakMemoryMB: Double
        public let computeBackend: String
    }
    
    /// The main inference engine
    public class Engine {
        private let config: Config
        // private var rustEngine: CrabInferEngine  // UniFFI-generated type
        
        /// Create an engine with a model path (auto-configures everything else)
        public convenience init(modelPath: String) throws {
            try self.init(config: Config(modelPath: modelPath))
        }
        
        /// Create an engine with full configuration
        public init(config: Config) throws {
            self.config = config
            
            // Will create the Rust engine via UniFFI:
            // let rustConfig = EngineConfig(
            //     modelPath: config.modelPath,
            //     maxTokens: UInt32(config.maxTokens),
            //     temperature: config.temperature,
            //     topP: config.topP,
            //     contextLength: UInt32(config.contextLength),
            //     useMetal: config.useMetal,
            //     memoryLimitBytes: UInt64((config.memoryLimitMB ?? 0) * 1024 * 1024)
            // )
            // self.rustEngine = try CrabInferEngine(config: rustConfig)
        }
        
        /// Generate a complete response
        public func complete(_ prompt: String, maxTokens: Int? = nil, temperature: Float? = nil) async throws -> String {
            return try await withCheckedThrowingContinuation { continuation in
                DispatchQueue.global(qos: .userInitiated).async { [self] in
                    do {
                        // Will call: rustEngine.complete(prompt, maxTokens, temperature)
                        let result = "[CrabInfer] Response to: \(prompt.prefix(50))"
                        continuation.resume(returning: result)
                    } catch {
                        continuation.resume(throwing: error)
                    }
                }
            }
        }
        
        /// Stream tokens as an AsyncSequence
        public func stream(_ prompt: String) -> TokenStream {
            TokenStream(engine: self, prompt: prompt)
        }
        
        /// Current memory pressure
        public var memoryState: MemoryState {
            // Will call: rustEngine.memoryPressure()
            .normal
        }
        
        /// Statistics from last generation
        public var lastStats: Stats? {
            // Will call: rustEngine.lastStats()
            nil
        }
        
        /// Unload the model to free memory
        public func unload() {
            // Will call: rustEngine.unloadModel()
        }
    }
    
    /// AsyncSequence for streaming tokens
    public struct TokenStream: AsyncSequence {
        public typealias Element = String
        
        let engine: Engine
        let prompt: String
        
        public func makeAsyncIterator() -> TokenIterator {
            TokenIterator(engine: engine, prompt: prompt)
        }
        
        public struct TokenIterator: AsyncIteratorProtocol {
            let engine: Engine
            let prompt: String
            var done = false
            
            public mutating func next() async throws -> String? {
                if done { return nil }
                
                // Will call: rustEngine.nextToken(prompt)
                // Returns nil when generation is complete
                // For now, stub:
                done = true
                return nil
            }
        }
    }
}

// MARK: - SwiftUI Integration

#if canImport(SwiftUI)
import SwiftUI

extension CrabInfer {
    /// Observable wrapper for SwiftUI
    @available(iOS 17.0, macOS 14.0, *)
    @Observable
    public class ObservableEngine {
        public var isLoading = false
        public var isGenerating = false
        public var currentOutput = ""
        public var memoryState: MemoryState = .normal
        public var error: Error?
        
        private let engine: Engine
        
        public init(config: Config) throws {
            self.engine = try Engine(config: config)
        }
        
        public func generate(_ prompt: String) async {
            isGenerating = true
            currentOutput = ""
            error = nil
            
            do {
                let result = try await engine.complete(prompt)
                currentOutput = result
            } catch {
                self.error = error
            }
            
            isGenerating = false
        }
    }
}
#endif
