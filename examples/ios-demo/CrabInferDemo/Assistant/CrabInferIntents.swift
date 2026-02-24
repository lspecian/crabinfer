import AppIntents
import CrabInfer

// MARK: - Ask CrabInfer Intent

/// Siri Shortcut: "Ask CrabInfer to..."
///
/// This App Intent allows users to interact with the CrabInfer assistant
/// via Siri, Shortcuts app, and Spotlight.
@available(iOS 16.0, *)
struct AskCrabInferIntent: AppIntent {
    static var title: LocalizedStringResource = "Ask CrabInfer"
    static var description = IntentDescription(
        "Ask the CrabInfer AI assistant a question or give it a task.",
        categoryName: "AI Assistant"
    )

    @Parameter(title: "Question or Task")
    var prompt: String

    @Parameter(title: "Provider", default: "openai")
    var provider: String

    @Parameter(title: "Model", default: "gpt-4o")
    var model: String

    static var parameterSummary: some ParameterSummary {
        Summary("Ask CrabInfer \(\.$prompt)")
    }

    func perform() async throws -> some IntentResult & ReturnsValue<String> & ProvidesDialog {
        // Create a temporary agent for this intent
        let apiKey = CrabInfer.Credentials.getApiKey(provider: provider) ?? ""
        guard !apiKey.isEmpty else {
            return .result(
                value: "No API key configured for \(provider). Please set it in the CrabInfer app.",
                dialog: "I need an API key for \(provider) to answer your question. Please configure it in the CrabInfer app."
            )
        }

        let providerConfig = CrabInfer.cloudConfig(
            provider: provider,
            apiKey: apiKey,
            model: model
        )

        let config = CrabInfer.AgentConfig(
            maxToolRounds: 5, // Limit for Siri responses
            maxTokens: 1024,  // Shorter for voice
            temperature: 0.7
        )

        let agent = try CrabInfer.Agent(provider: providerConfig, config: config)
        let result = try await agent.run(prompt)

        // Truncate for voice output
        let response = result.text.count > 500
            ? String(result.text.prefix(500)) + "..."
            : result.text

        return .result(value: response, dialog: IntentDialog(stringLiteral: response))
    }
}

// MARK: - Remember Fact Intent

/// Siri Shortcut: "Remember that..." — stores a fact in the assistant's memory.
@available(iOS 16.0, *)
struct RememberFactIntent: AppIntent {
    static var title: LocalizedStringResource = "Remember a Fact"
    static var description = IntentDescription(
        "Store a fact that CrabInfer will remember across conversations.",
        categoryName: "AI Assistant"
    )

    @Parameter(title: "Key (e.g., 'my name')")
    var key: String

    @Parameter(title: "Value (e.g., 'Luis')")
    var value: String

    static var parameterSummary: some ParameterSummary {
        Summary("Remember \(\.$key) is \(\.$value)")
    }

    func perform() async throws -> some IntentResult & ProvidesDialog {
        // Create a temporary agent just to store the fact
        // In a full app, this would use a shared agent instance
        let apiKey = CrabInfer.Credentials.getApiKey(provider: "openai") ?? ""
        let providerConfig = CrabInfer.cloudConfig(
            provider: "openai",
            apiKey: apiKey.isEmpty ? "placeholder" : apiKey,
            model: "gpt-4o"
        )

        let agent = try CrabInfer.Agent(provider: providerConfig)

        // Set persistence path
        let docsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let factsPath = docsDir.appendingPathComponent("assistant/facts.json").path
        agent.setFactsPath(factsPath)

        agent.addFact(key: key, value: value)
        try agent.save()

        return .result(dialog: "I'll remember that \(key) is \(value).")
    }
}

// MARK: - Shortcuts Provider

/// Registers CrabInfer intents with the Shortcuts app.
@available(iOS 16.0, *)
struct CrabInferShortcutsProvider: AppShortcutsProvider {
    static var appShortcuts: [AppShortcut] {
        AppShortcut(
            intent: AskCrabInferIntent(),
            phrases: [
                "Ask \(.applicationName) \(\.$prompt)",
                "Ask \(.applicationName) to \(\.$prompt)",
                "Hey \(.applicationName), \(\.$prompt)",
            ],
            shortTitle: "Ask CrabInfer",
            systemImageName: "bubble.left.and.text.bubble.right"
        )
        AppShortcut(
            intent: RememberFactIntent(),
            phrases: [
                "Tell \(.applicationName) to remember \(\.$key) is \(\.$value)",
                "Remember in \(.applicationName) that \(\.$key) is \(\.$value)",
            ],
            shortTitle: "Remember a Fact",
            systemImageName: "brain"
        )
    }
}
