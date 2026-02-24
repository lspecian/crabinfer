import Foundation
import CrabInfer

// MARK: - Chat Message

struct AssistantMessage: Identifiable {
    let id = UUID()
    let role: MessageRole
    var content: String
    let timestamp = Date()
    var toolExecutions: [CrabInfer.ToolExecution] = []
    var rounds: UInt32 = 0
    var isStreaming: Bool = false

    enum MessageRole: String {
        case user
        case assistant
        case system
    }
}

// MARK: - Assistant View Model

@MainActor
final class AssistantViewModel: ObservableObject {
    // Agent state
    @Published var messages: [AssistantMessage] = []
    @Published var isProcessing = false
    @Published var agentStatus: AgentStatus = .init()

    // Model state (shared with inference view model)
    @Published var isModelLoaded = false
    @Published var modelName: String = ""
    @Published var isLoading = false
    @Published var loadError: String?

    // Provider state
    @Published var useCloudProvider = false
    @Published var cloudProvider = "openai"
    @Published var cloudApiKey = ""
    @Published var cloudModel = "gpt-4o"

    private(set) var agent: CrabInfer.Agent?
    private let inferenceThread = InferenceThread()

    /// Whether the agent is ready to accept queries.
    var isReady: Bool { agent != nil }

    struct AgentStatus {
        var toolCount: Int = 0
        var factCount: Int = 0
        var toolNames: [String] = []
        var mcpServerCount: Int = 0
    }

    // MARK: - Agent Setup

    /// Create or recreate the agent with current settings.
    func setupAgent() {
        do {
            let config = CrabInfer.AgentConfig(
                maxToolRounds: 10,
                maxTokens: 2048,
                temperature: 0.7,
                topP: 0.9,
                ragTopK: 3
            )

            if useCloudProvider && !cloudApiKey.isEmpty {
                let providerConfig = CrabInfer.cloudConfig(
                    provider: cloudProvider,
                    apiKey: cloudApiKey,
                    model: cloudModel
                )
                agent = try CrabInfer.Agent(provider: providerConfig, config: config)
            } else {
                // Without a cloud provider or local model, we can't create an agent
                // The agent will be created when a model is loaded
                agent = nil
                return
            }

            // Set persistence paths
            let docsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
            let assistantDir = docsDir.appendingPathComponent("assistant")
            try? FileManager.default.createDirectory(at: assistantDir, withIntermediateDirectories: true)

            agent?.setConversationPath(assistantDir.appendingPathComponent("conversation.json").path)
            agent?.setFactsPath(assistantDir.appendingPathComponent("facts.json").path)
            agent?.setIdentity(
                "You are CrabInfer Assistant, a helpful AI running locally on the user's iPhone/iPad. " +
                "You can read/write files, run shell commands, and fetch web pages via tools. " +
                "Think step by step. Use tools when needed, then provide a clear summary."
            )

            refreshStatus()
        } catch {
            loadError = "Failed to create agent: \(error.localizedDescription)"
        }
    }

    // MARK: - Send Message

    func sendMessage(_ text: String) async {
        guard let agent = agent, !isProcessing else { return }

        let userMessage = AssistantMessage(role: .user, content: text)
        messages.append(userMessage)

        // Add placeholder assistant message
        var assistantMessage = AssistantMessage(role: .assistant, content: "", isStreaming: true)
        messages.append(assistantMessage)
        isProcessing = true

        do {
            let result = try await agent.run(text)

            // Update the assistant message with the result
            if let idx = messages.lastIndex(where: { $0.role == .assistant }) {
                messages[idx].content = result.text
                messages[idx].toolExecutions = result.toolExecutions
                messages[idx].rounds = result.rounds
                messages[idx].isStreaming = false
            }

            refreshStatus()
        } catch {
            if let idx = messages.lastIndex(where: { $0.role == .assistant }) {
                messages[idx].content = "Error: \(error.localizedDescription)"
                messages[idx].isStreaming = false
            }
        }

        isProcessing = false
    }

    // MARK: - Fact Management

    func addFact(key: String, value: String) {
        agent?.addFact(key: key, value: value)
        refreshStatus()
    }

    func removeFact(key: String) {
        agent?.removeFact(key: key)
        refreshStatus()
    }

    var facts: [CrabInfer.Fact] {
        agent?.facts ?? []
    }

    // MARK: - Conversation Management

    func clearConversation() {
        agent?.clearConversation()
        messages.removeAll()
        refreshStatus()
    }

    func save() {
        try? agent?.save()
    }

    // MARK: - Status

    func refreshStatus() {
        guard let agent = agent else {
            agentStatus = .init()
            return
        }
        agentStatus = AgentStatus(
            toolCount: agent.toolCount,
            factCount: agent.factCount,
            toolNames: agent.toolNames,
            mcpServerCount: 0
        )
    }
}
