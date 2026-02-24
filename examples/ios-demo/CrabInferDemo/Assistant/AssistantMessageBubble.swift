import SwiftUI
import CrabInfer

struct AssistantMessageBubble: View {
    let message: AssistantMessage

    var body: some View {
        HStack {
            if message.role == .user { Spacer(minLength: 48) }

            VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 4) {
                // Message content
                Text(message.content.isEmpty && message.isStreaming ? "Thinking..." : message.content)
                    .font(.body)
                    .foregroundColor(message.role == .user ? .white : .primary)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(
                        message.role == .user
                            ? Color.blue
                            : Color(.secondarySystemBackground)
                    )
                    .cornerRadius(18)
                    .opacity(message.isStreaming && message.content.isEmpty ? 0.6 : 1.0)

                // Tool executions
                if !message.toolExecutions.isEmpty {
                    toolExecutionSection
                }

                // Rounds indicator
                if message.rounds > 1 {
                    Text("\(message.rounds) rounds")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }

            if message.role == .assistant { Spacer(minLength: 48) }
        }
    }

    private var toolExecutionSection: some View {
        VStack(alignment: .leading, spacing: 2) {
            ForEach(Array(message.toolExecutions.enumerated()), id: \.offset) { _, exec in
                ToolExecutionRow(execution: exec)
            }
        }
        .padding(.horizontal, 4)
    }
}

struct ToolExecutionRow: View {
    let execution: CrabInfer.ToolExecution
    @State private var isExpanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Button(action: { withAnimation { isExpanded.toggle() } }) {
                HStack(spacing: 4) {
                    Image(systemName: execution.isError ? "xmark.circle.fill" : "checkmark.circle.fill")
                        .foregroundColor(execution.isError ? .red : .green)
                        .font(.caption2)
                    Text(execution.toolName)
                        .font(.caption.monospaced())
                        .foregroundColor(.orange)
                    Spacer()
                    Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }
            .buttonStyle(.plain)

            if isExpanded {
                VStack(alignment: .leading, spacing: 4) {
                    if !execution.argumentsJson.isEmpty {
                        Text("args:")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                        Text(execution.argumentsJson.prefix(300) + (execution.argumentsJson.count > 300 ? "..." : ""))
                            .font(.caption2.monospaced())
                            .foregroundColor(.secondary)
                    }
                    Text("output:")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                    Text(execution.output.prefix(300) + (execution.output.count > 300 ? "..." : ""))
                        .font(.caption2.monospaced())
                        .foregroundColor(execution.isError ? .red : .secondary)
                }
                .padding(8)
                .background(Color(.tertiarySystemBackground))
                .cornerRadius(8)
            }
        }
    }
}
