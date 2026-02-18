import SwiftUI

struct ContentView: View {
    @StateObject private var vm = InferenceViewModel()
    @State private var prompt = "Write a haiku about Rust programming:"
    @State private var maxTokens: Double = 64

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    // Memory pressure indicator
                    HStack {
                        Circle()
                            .fill(pressureColor)
                            .frame(width: 12, height: 12)
                        Text("Memory: \(vm.pressureLabel)")
                            .font(.caption)
                        Spacer()
                    }

                    // Device info
                    GroupBox("Device") {
                        Text(vm.deviceInfo)
                            .font(.caption.monospaced())
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }

                    // Model selection & management
                    GroupBox("Model") {
                        Text(vm.modelStatus)
                            .font(.caption.monospaced())
                            .frame(maxWidth: .infinity, alignment: .leading)

                        // Model picker
                        if !vm.isModelLoaded {
                            Picker("Select model", selection: $vm.selectedModelId) {
                                ForEach(modelCatalog) { entry in
                                    HStack {
                                        Text(entry.name)
                                        if vm.downloadedModelIds.contains(entry.id) {
                                            Image(systemName: "checkmark.circle.fill")
                                                .foregroundColor(.green)
                                        }
                                    }
                                    .tag(entry.id)
                                }
                            }
                            .pickerStyle(.menu)
                            .padding(.top, 2)

                            Text("\(vm.selectedModel.sizeMB) MB download")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                        }

                        // Download progress
                        if vm.isDownloading {
                            VStack(alignment: .leading, spacing: 4) {
                                ProgressView(value: vm.downloadProgress)
                                    .progressViewStyle(.linear)
                                Text(vm.downloadStatus)
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                            }
                            .padding(.top, 4)
                        }

                        // Action buttons
                        if !vm.isDownloading && !vm.isModelLoaded {
                            if vm.isSelectedModelDownloaded {
                                // Model already downloaded — load or delete
                                HStack(spacing: 8) {
                                    Button(action: {
                                        Task { await vm.loadModel(id: vm.selectedModelId) }
                                    }) {
                                        Label(
                                            vm.isLoading ? "Loading..." : "Load Model",
                                            systemImage: "arrow.down.circle"
                                        )
                                        .frame(maxWidth: .infinity)
                                    }
                                    .buttonStyle(.borderedProminent)
                                    .disabled(vm.isLoading)

                                    Button(role: .destructive, action: {
                                        vm.deleteModel(id: vm.selectedModelId)
                                    }) {
                                        Image(systemName: "trash")
                                    }
                                    .buttonStyle(.bordered)
                                    .disabled(vm.isLoading)
                                }
                                .padding(.top, 4)
                            } else {
                                // Not downloaded yet — download button
                                Button(action: {
                                    Task { await vm.downloadSelectedModel() }
                                }) {
                                    Label(
                                        "Download \(vm.selectedModel.name)",
                                        systemImage: "icloud.and.arrow.down"
                                    )
                                    .frame(maxWidth: .infinity)
                                }
                                .buttonStyle(.borderedProminent)
                                .tint(.green)
                                .padding(.top, 4)

                                Text("Or copy model.gguf + tokenizer.json\nvia Files app")
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                            }
                        }

                        // Model loaded — show unload button
                        if vm.isModelLoaded {
                            HStack {
                                Label("Model Loaded", systemImage: "checkmark.circle.fill")
                                    .foregroundColor(.green)
                                Spacer()
                                Button("Unload") {
                                    vm.unloadCurrentModel()
                                }
                                .buttonStyle(.bordered)
                                .tint(.red)
                            }
                            .padding(.top, 4)
                        }
                    }

                    // Prompt
                    GroupBox("Generate") {
                        TextField("Enter prompt", text: $prompt, axis: .vertical)
                            .lineLimit(2...5)
                            .textFieldStyle(.roundedBorder)

                        HStack {
                            Text("Tokens: \(Int(maxTokens))")
                                .font(.caption)
                            Slider(value: $maxTokens, in: 10...256, step: 10)
                        }

                        Button(action: {
                            Task { await vm.generate(prompt: prompt, maxTokens: UInt32(maxTokens)) }
                        }) {
                            Label(
                                vm.isGenerating ? "Generating..." : "Generate",
                                systemImage: "sparkles"
                            )
                            .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(vm.isGenerating || !vm.isModelLoaded)
                    }

                    // Generation status
                    if vm.isGenerating && !vm.generatingStatus.isEmpty {
                        HStack {
                            ProgressView()
                                .scaleEffect(0.8)
                            Text(vm.generatingStatus)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }

                    // Output
                    if !vm.output.isEmpty {
                        GroupBox("Output") {
                            Text(vm.output)
                                .font(.body.monospaced())
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }

                    // Stats
                    if let stats = vm.lastGenerationStats {
                        GroupBox("Stats") {
                            VStack(spacing: 8) {
                                // Backend badge
                                HStack {
                                    Label(stats.computeBackend,
                                          systemImage: stats.computeBackend.contains("Metal")
                                              ? "bolt.fill" : "cpu")
                                        .font(.caption.bold())
                                        .padding(.horizontal, 8)
                                        .padding(.vertical, 4)
                                        .background(
                                            stats.computeBackend.contains("Metal")
                                                ? Color.purple.opacity(0.15)
                                                : Color.gray.opacity(0.15)
                                        )
                                        .cornerRadius(6)
                                    Spacer()
                                    Text("\(stats.tokensGenerated) tokens")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }

                                // Metrics grid
                                LazyVGrid(columns: [
                                    GridItem(.flexible()),
                                    GridItem(.flexible()),
                                ], spacing: 8) {
                                    StatCell(
                                        label: "Throughput",
                                        value: String(format: "%.1f", stats.tokensPerSecond),
                                        unit: "tok/s"
                                    )
                                    StatCell(
                                        label: "Time to First Token",
                                        value: String(format: "%.0f", stats.timeToFirstTokenMs),
                                        unit: "ms"
                                    )
                                    StatCell(
                                        label: "Total Time",
                                        value: stats.totalTimeMs >= 1000
                                            ? String(format: "%.2f", stats.totalTimeMs / 1000)
                                            : String(format: "%.0f", stats.totalTimeMs),
                                        unit: stats.totalTimeMs >= 1000 ? "s" : "ms"
                                    )
                                    StatCell(
                                        label: "Peak Memory",
                                        value: "\(stats.peakMemoryBytes / (1024 * 1024))",
                                        unit: "MB"
                                    )
                                }
                            }
                        }
                    }
                    // Stress Test
                    GroupBox("Stress Test") {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Load/unload cycles to detect memory leaks")
                                .font(.caption2)
                                .foregroundColor(.secondary)

                            Button(action: {
                                Task { await vm.runStressTest() }
                            }) {
                                Label(
                                    vm.isStressTesting ? "Running..." : "Run 10-Cycle Stress Test",
                                    systemImage: "arrow.triangle.2.circlepath"
                                )
                                .frame(maxWidth: .infinity)
                            }
                            .buttonStyle(.bordered)
                            .tint(.orange)
                            .disabled(vm.isStressTesting || !vm.isSelectedModelDownloaded || vm.isGenerating)

                            if vm.isStressTesting {
                                ProgressView()
                                    .frame(maxWidth: .infinity)
                            }

                            if !vm.stressTestLog.isEmpty {
                                ScrollView {
                                    VStack(alignment: .leading, spacing: 2) {
                                        ForEach(vm.stressTestLog, id: \.self) { line in
                                            Text(line)
                                                .font(.caption2.monospaced())
                                        }
                                    }
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                }
                                .frame(maxHeight: 200)
                            }
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("CrabInfer")
        }
    }

    var pressureColor: Color {
        switch vm.pressureLevel {
        case .normal: return .green
        case .warning: return .yellow
        case .critical, .terminal: return .red
        }
    }
}

/// A single stat metric cell with large value and caption label.
struct StatCell: View {
    let label: String
    let value: String
    let unit: String

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(alignment: .firstTextBaseline, spacing: 2) {
                Text(value)
                    .font(.title3.monospacedDigit().bold())
                Text(unit)
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
            Text(label)
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(8)
        .background(Color(.systemGray6))
        .cornerRadius(8)
    }
}
