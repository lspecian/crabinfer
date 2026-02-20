import SwiftUI

struct MessageBubble: View {
    let message: ChatMessage

    var body: some View {
        HStack {
            if message.role == .user { Spacer(minLength: 60) }

            VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 4) {
                Text(message.content)
                    .foregroundColor(.white)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(bubbleColor)
                    .cornerRadius(18)

                if let stats = message.stats {
                    statsLine(stats)
                }
            }

            if message.role == .assistant { Spacer(minLength: 60) }
        }
    }

    private var bubbleColor: Color {
        if message.isError {
            return Color(.systemRed).opacity(0.7)
        }
        switch message.role {
        case .user:
            return .blue
        case .assistant:
            return Color(.systemGray3)
        }
    }

    private func statsLine(_ stats: GenerationStats) -> some View {
        let toks = String(format: "%.1f", stats.tokensPerSecond)
        let ttft = String(format: "%.0f", stats.timeToFirstTokenMs)
        let backend = stats.computeBackend.contains("Metal") ? "Metal" : "CPU"
        let memMB = stats.peakMemoryBytes / (1024 * 1024)

        return Text("\(toks) tok/s · \(ttft)ms TTFT · \(backend) · \(memMB) MB")
            .font(.caption2)
            .foregroundColor(.secondary)
            .padding(.horizontal, 4)
    }
}
