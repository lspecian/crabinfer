import SwiftUI

struct AssistantSettingsView: View {
    @ObservedObject var viewModel: AssistantViewModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                // Provider section
                Section("Provider") {
                    Toggle("Use Cloud Provider", isOn: $viewModel.useCloudProvider)

                    if viewModel.useCloudProvider {
                        Picker("Provider", selection: $viewModel.cloudProvider) {
                            Text("OpenAI").tag("openai")
                            Text("Anthropic").tag("anthropic")
                            Text("Google").tag("google")
                        }

                        SecureField("API Key", text: $viewModel.cloudApiKey)
                            .textContentType(.password)

                        TextField("Model", text: $viewModel.cloudModel)
                            .autocorrectionDisabled()
                    }
                }

                // Agent status
                Section("Agent Status") {
                    HStack {
                        Text("Tools")
                        Spacer()
                        Text("\(viewModel.agentStatus.toolCount)")
                            .foregroundColor(.secondary)
                    }

                    if !viewModel.agentStatus.toolNames.isEmpty {
                        ForEach(viewModel.agentStatus.toolNames, id: \.self) { name in
                            HStack {
                                Image(systemName: "wrench")
                                    .foregroundColor(.orange)
                                    .font(.caption)
                                Text(name)
                                    .font(.callout.monospaced())
                            }
                        }
                    }

                    HStack {
                        Text("Facts remembered")
                        Spacer()
                        Text("\(viewModel.agentStatus.factCount)")
                            .foregroundColor(.secondary)
                    }
                }

                // Facts
                if !viewModel.facts.isEmpty {
                    Section("Remembered Facts") {
                        ForEach(viewModel.facts) { fact in
                            HStack {
                                VStack(alignment: .leading) {
                                    Text(fact.key)
                                        .font(.callout.bold())
                                    Text(fact.value)
                                        .font(.callout)
                                        .foregroundColor(.secondary)
                                }
                                Spacer()
                            }
                        }
                        .onDelete { indexSet in
                            for index in indexSet {
                                let fact = viewModel.facts[index]
                                viewModel.removeFact(key: fact.key)
                            }
                        }
                    }
                }

                // Actions
                Section {
                    Button("Apply Settings") {
                        viewModel.setupAgent()
                        dismiss()
                    }
                    .foregroundColor(.orange)

                    Button("Clear Conversation") {
                        viewModel.clearConversation()
                    }
                    .foregroundColor(.red)
                }
            }
            .navigationTitle("Assistant Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}
