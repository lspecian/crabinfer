import SwiftUI

struct ModelLoaderSheet: View {
    @ObservedObject var vm: InferenceViewModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                ForEach(modelCatalog) { entry in
                    modelRow(entry)
                }

                if vm.isDownloading {
                    Section("Download Progress") {
                        VStack(alignment: .leading, spacing: 6) {
                            ProgressView(value: vm.downloadProgress)
                                .progressViewStyle(.linear)
                            Text(vm.downloadStatus)
                                .font(.caption2)
                                .foregroundColor(.secondary)
                        }
                        .padding(.vertical, 4)
                    }
                }
            }
            .navigationTitle("Models")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    private func modelRow(_ entry: ModelEntry) -> some View {
        let isDownloaded = vm.downloadedModelIds.contains(entry.id)
        let isActive = vm.activeModelId == entry.id

        return Section {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text(entry.name)
                        .font(.headline)
                    Spacer()
                    if isActive {
                        Label("Loaded", systemImage: "checkmark.circle.fill")
                            .font(.caption)
                            .foregroundColor(.green)
                    } else if isDownloaded {
                        Image(systemName: "checkmark.circle")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }

                Text("\(entry.sizeMB) MB · \(entry.architecture)")
                    .font(.caption)
                    .foregroundColor(.secondary)

                // Actions
                if isActive {
                    Button("Unload") {
                        vm.unloadCurrentModel()
                    }
                    .font(.caption)
                    .buttonStyle(.bordered)
                    .tint(.red)
                } else if isDownloaded {
                    HStack(spacing: 8) {
                        Button(action: {
                            Task {
                                await vm.loadModel(id: entry.id)
                                dismiss()
                            }
                        }) {
                            Label(
                                vm.isLoading ? "Loading..." : "Load",
                                systemImage: "arrow.down.circle"
                            )
                            .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(vm.isLoading)

                        Button(role: .destructive, action: {
                            vm.deleteModel(id: entry.id)
                        }) {
                            Image(systemName: "trash")
                        }
                        .buttonStyle(.bordered)
                        .disabled(vm.isLoading)
                    }
                } else {
                    Button(action: {
                        vm.selectedModelId = entry.id
                        Task { await vm.downloadSelectedModel() }
                    }) {
                        Label("Download", systemImage: "icloud.and.arrow.down")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.green)
                    .disabled(vm.isDownloading)
                }
            }
            .padding(.vertical, 4)
        }
    }
}
