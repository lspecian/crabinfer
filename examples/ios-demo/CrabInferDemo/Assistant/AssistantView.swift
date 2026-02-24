import SwiftUI

struct AssistantView: View {
    @StateObject private var viewModel = AssistantViewModel()
    @State private var showSettings = false

    var body: some View {
        NavigationStack {
            AssistantChatView(viewModel: viewModel)
                .navigationTitle("Assistant")
                .navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .navigationBarLeading) {
                        statusBadge
                    }
                    ToolbarItem(placement: .navigationBarTrailing) {
                        Button(action: { showSettings = true }) {
                            Image(systemName: "gearshape")
                        }
                    }
                }
                .sheet(isPresented: $showSettings) {
                    AssistantSettingsView(viewModel: viewModel)
                }
                .onAppear {
                    if viewModel.agent == nil {
                        viewModel.setupAgent()
                    }
                }
        }
    }

    private var statusBadge: some View {
        HStack(spacing: 4) {
            Circle()
                .fill(viewModel.agent != nil ? Color.green : Color.gray)
                .frame(width: 8, height: 8)
            if viewModel.agentStatus.toolCount > 0 {
                Text("\(viewModel.agentStatus.toolCount) tools")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
    }
}
