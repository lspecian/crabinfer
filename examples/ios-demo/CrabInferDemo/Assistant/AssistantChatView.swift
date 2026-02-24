import SwiftUI

struct AssistantChatView: View {
    @ObservedObject var viewModel: AssistantViewModel
    @State private var inputText = ""
    @FocusState private var isInputFocused: Bool

    var body: some View {
        VStack(spacing: 0) {
            // Messages
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(spacing: 12) {
                        if viewModel.messages.isEmpty {
                            emptyState
                        }

                        ForEach(viewModel.messages) { message in
                            AssistantMessageBubble(message: message)
                                .id(message.id)
                        }
                    }
                    .padding()
                }
                .onChange(of: viewModel.messages.count) { _ in
                    if let last = viewModel.messages.last {
                        withAnimation(.easeOut(duration: 0.2)) {
                            proxy.scrollTo(last.id, anchor: .bottom)
                        }
                    }
                }
            }

            // Input bar
            inputBar
        }
        .background(Color(.systemBackground))
    }

    private var emptyState: some View {
        VStack(spacing: 12) {
            Spacer(minLength: 80)
            Text("🦀")
                .font(.system(size: 48))
            Text("CrabInfer Assistant")
                .font(.headline)
                .foregroundColor(.primary)
            Text(viewModel.agent != nil
                 ? "Ask me anything. I can read files, run commands, and fetch web pages."
                 : "Set up a provider in Settings to get started.")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)

            if viewModel.agentStatus.toolCount > 0 {
                HStack(spacing: 4) {
                    Image(systemName: "wrench.fill")
                        .font(.caption2)
                    Text("\(viewModel.agentStatus.toolCount) tools available")
                        .font(.caption)
                }
                .foregroundColor(.orange)
                .padding(.top, 4)
            }
            Spacer(minLength: 80)
        }
    }

    private var inputBar: some View {
        HStack(spacing: 8) {
            TextField("Ask the assistant...", text: $inputText, axis: .vertical)
                .textFieldStyle(.plain)
                .padding(10)
                .background(Color(.secondarySystemBackground))
                .cornerRadius(20)
                .lineLimit(1...5)
                .focused($isInputFocused)
                .disabled(viewModel.isProcessing || viewModel.agent == nil)
                .onSubmit {
                    send()
                }

            Button(action: send) {
                Image(systemName: viewModel.isProcessing ? "hourglass" : "arrow.up.circle.fill")
                    .font(.title2)
                    .foregroundColor(canSend ? .orange : .gray)
            }
            .disabled(!canSend)
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
        .background(Color(.systemBackground))
        .overlay(
            Divider(), alignment: .top
        )
    }

    private var canSend: Bool {
        !inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !viewModel.isProcessing
            && viewModel.agent != nil
    }

    private func send() {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        inputText = ""
        Task {
            await viewModel.sendMessage(text)
        }
    }
}
