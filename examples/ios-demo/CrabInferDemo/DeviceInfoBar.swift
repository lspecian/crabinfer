import SwiftUI

struct DeviceInfoBar: View {
    @ObservedObject var vm: InferenceViewModel
    @Binding var showModelSheet: Bool
    @State private var isExpanded = false

    var body: some View {
        VStack(spacing: 0) {
            // Compact bar — always visible
            Button(action: { withAnimation(.easeInOut(duration: 0.2)) { isExpanded.toggle() } }) {
                HStack(spacing: 8) {
                    Circle()
                        .fill(pressureColor)
                        .frame(width: 8, height: 8)

                    if vm.isReady {
                        Text(vm.deviceBarSummary)
                            .font(.caption)
                            .foregroundColor(.primary)
                            .lineLimit(1)
                    } else {
                        Text("Detecting device...")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }

                    Spacer()

                    if vm.isModelLoaded {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.caption)
                            .foregroundColor(.green)
                    }

                    Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color(.systemGray6))
            }
            .buttonStyle(.plain)

            // Expanded detail section
            if isExpanded {
                expandedContent
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
    }

    private var expandedContent: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Device details
            Text(vm.deviceInfo)
                .font(.caption.monospaced())
                .frame(maxWidth: .infinity, alignment: .leading)

            // Memory pressure
            HStack {
                Circle()
                    .fill(pressureColor)
                    .frame(width: 10, height: 10)
                Text("Memory: \(vm.pressureLabel)")
                    .font(.caption)
                Spacer()
            }

            Divider()

            // Model status
            Text(vm.modelStatus)
                .font(.caption.monospaced())
                .frame(maxWidth: .infinity, alignment: .leading)

            // Action buttons
            HStack(spacing: 8) {
                if vm.isModelLoaded {
                    Button("Unload Model") {
                        vm.unloadCurrentModel()
                    }
                    .font(.caption)
                    .buttonStyle(.bordered)
                    .tint(.red)
                } else {
                    Button("Manage Models") {
                        showModelSheet = true
                    }
                    .font(.caption)
                    .buttonStyle(.borderedProminent)
                }
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(Color(.systemGray6))
    }

    private var pressureColor: Color {
        switch vm.pressureLevel {
        case .normal: return .green
        case .warning: return .yellow
        case .critical, .terminal: return .red
        }
    }
}
