import SwiftUI

struct ChatView: View {
    @ObservedObject var vm: InferenceViewModel
    @State private var inputText = ""
    @FocusState private var inputFocused: Bool

    var body: some View {
        VStack(spacing: 0) {
            messageList
            if vm.isLoading {
                loadingBar
            }
            inputBar
        }
    }

    // MARK: - Message list

    private var messageList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 12) {
                    if vm.messages.isEmpty {
                        centerState
                    }

                    ForEach(vm.messages) { message in
                        MessageBubble(message: message)
                            .id(message.id)
                    }
                }
                .padding(.horizontal, 16)
                .padding(.top, 12)
                .padding(.bottom, 8)
            }
            .onChange(of: vm.messages.last?.content) { _ in
                scrollToBottom(proxy: proxy)
            }
            .onChange(of: vm.messages.count) { _ in
                scrollToBottom(proxy: proxy)
            }
        }
    }

    private func scrollToBottom(proxy: ScrollViewProxy) {
        guard let lastId = vm.messages.last?.id else { return }
        withAnimation(.easeOut(duration: 0.15)) {
            proxy.scrollTo(lastId, anchor: .bottom)
        }
    }

    @ViewBuilder
    private var centerState: some View {
        if !vm.isReady {
            // App just launched, detecting device
            VStack(spacing: 12) {
                ProgressView()
                    .scaleEffect(1.2)
                Text("Detecting device...")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }
            .frame(maxWidth: .infinity)
            .padding(.top, 100)
        } else if vm.isLoading {
            // Model is loading
            VStack(spacing: 12) {
                ProgressView()
                    .scaleEffect(1.2)
                Text("Loading model...")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                Text("Compiling Metal shaders")
                    .font(.caption)
                    .foregroundColor(.secondary.opacity(0.7))
            }
            .frame(maxWidth: .infinity)
            .padding(.top, 100)
        } else if vm.isModelLoaded {
            VStack(spacing: 8) {
                Image(systemName: "bubble.left.and.bubble.right")
                    .font(.system(size: 36))
                    .foregroundColor(.secondary.opacity(0.5))
                Text("Start a conversation")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }
            .frame(maxWidth: .infinity)
            .padding(.top, 100)
        } else {
            VStack(spacing: 8) {
                Image(systemName: "cpu")
                    .font(.system(size: 36))
                    .foregroundColor(.secondary.opacity(0.5))
                Text("Load a model to start chatting")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                Text("Tap the device info bar above")
                    .font(.caption)
                    .foregroundColor(.secondary.opacity(0.7))
            }
            .frame(maxWidth: .infinity)
            .padding(.top, 100)
        }
    }

    // MARK: - Loading bar

    private var loadingBar: some View {
        HStack(spacing: 8) {
            ProgressView()
                .scaleEffect(0.8)
            Text(vm.modelStatus)
                .font(.caption)
                .foregroundColor(.secondary)
                .lineLimit(1)
            Spacer()
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(Color(.systemGray6))
    }

    // MARK: - Input bar

    private var inputBar: some View {
        HStack(spacing: 8) {
            TextField(inputPlaceholder, text: $inputText, axis: .vertical)
                .lineLimit(1...5)
                .textFieldStyle(.plain)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color(.systemGray5))
                .cornerRadius(20)
                .focused($inputFocused)

            if vm.isGenerating {
                Button(action: { vm.stopGenerating() }) {
                    Image(systemName: "stop.circle.fill")
                        .font(.title2)
                        .foregroundColor(.red)
                }
            } else {
                Button(action: send) {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.title2)
                        .foregroundColor(canSend ? .blue : .gray.opacity(0.5))
                }
                .disabled(!canSend)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(Color(.systemBackground))
    }

    private var inputPlaceholder: String {
        if !vm.isReady { return "Detecting device..." }
        if vm.isLoading { return "Loading model..." }
        if !vm.isModelLoaded { return "Load a model to chat" }
        return "Message..."
    }

    private var canSend: Bool {
        !inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !vm.isGenerating
            && vm.isModelLoaded
    }

    private func send() {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        inputText = ""
        Task { await vm.sendMessage(text) }
    }
}
