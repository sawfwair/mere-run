import SwiftUI

/// The one place on-device models live: what runs on this iPhone, what each
/// model does, and their download state. Reached from Create's and Chat's
/// model menus and from Settings; pickers elsewhere only ever *select* from
/// what is installed here.
struct OnDeviceModelsView: View {
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            OnDeviceModelsList()
                .toolbar {
                    ToolbarItem(placement: .confirmationAction) {
                        Button("Done") { dismiss() }
                    }
                }
        }
    }
}

/// The list itself, pushable from Settings or wrapped in a sheet.
struct OnDeviceModelsList: View {
    @ObservedObject private var local = LocalEngine.shared
    @AppStorage(LocalEngine.allowsCellularDownloadsKey) private var allowsCellularDownloads = false
    @State private var confirmingModel: LocalEngine.Model?

    var body: some View {
        List {
                Section {
                    ForEach(LocalEngine.imageModels) { model in
                        row(model)
                    }
                } header: {
                    Text("Images")
                } footer: {
                    Text("Pick which one creates your images from the model menu in Create. Swipe a downloaded model to remove it.")
                }

                Section {
                    ForEach(LocalEngine.chatModels) { model in
                        row(model)
                    }
                } header: {
                    Text("Chat")
                } footer: {
                    Text("Choose one from the model menu in Chat to talk entirely on-device.")
                }

                if local.reclaimableBytes > 0 {
                    Section {
                        Button {
                            local.reclaimSpace()
                        } label: {
                            HStack {
                                Label("Reclaim unused space", systemImage: "trash")
                                Spacer()
                                Text(ByteCountFormatter.string(
                                    fromByteCount: local.reclaimableBytes,
                                    countStyle: .file
                                ))
                                .foregroundStyle(MereTheme.textMuted)
                            }
                        }
                    } header: {
                        Text("Storage")
                    } footer: {
                        Text("Frees partial downloads and payloads no installed model uses.")
                    }
                }

                if local.isDownloadingModel {
                    Section {
                        Label("Safe to leave the app", systemImage: "arrow.down.circle")
                            .font(.footnote)
                    } footer: {
                        Text("iOS continues model transfers in the background and finishes verification when mere.run resumes.")
                    }
                }

                Section {
                    Toggle("Allow cellular downloads", isOn: $allowsCellularDownloads)
                } header: {
                    Text("Downloads")
                } footer: {
                    Text(allowsCellularDownloads
                        ? "New model downloads may use cellular, expensive, or constrained networks."
                        : "New model downloads wait for an unrestricted Wi-Fi connection.")
                }

                if let message = local.lastError {
                    Section {
                        Text(message)
                            .font(.footnote)
                            .foregroundStyle(MereTheme.failure)
                    }
                }
            }
        .scrollContentBackground(.hidden)
        .background(MereTheme.background.ignoresSafeArea())
        .navigationTitle("On this iPhone")
        .navigationBarTitleDisplayMode(.inline)
        .onAppear {
            local.refresh()
            local.refreshReclaimable()
        }
        .confirmationDialog(
            confirmingModel?.licenseNote ?? "",
            isPresented: Binding(
                get: { confirmingModel != nil },
                set: { if !$0 { confirmingModel = nil } }
            ),
            titleVisibility: .visible
        ) {
            Button("Accept and download") {
                if let model = confirmingModel {
                    Task { await LocalEngine.shared.download(model.id, acceptedTerms: true) }
                }
                confirmingModel = nil
            }
            Button("Cancel", role: .cancel) { confirmingModel = nil }
        }
    }

    @ViewBuilder
    private func row(_ model: LocalEngine.Model) -> some View {
        HStack(spacing: MereTheme.Spacing.m) {
            VStack(alignment: .leading, spacing: 2) {
                Text(model.title)
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(MereTheme.textPrimary)
                Text(model.detail)
                    .font(.caption)
                    .foregroundStyle(MereTheme.textMuted)
            }
            Spacer()
            if !model.isCompatible {
                HStack(spacing: MereTheme.Spacing.s) {
                    Text("Needs \(model.minimumMemoryGB ?? 0) GB\nof memory")
                        .font(.caption2)
                        .multilineTextAlignment(.trailing)
                        .foregroundStyle(MereTheme.textMuted)
                    if local.state(of: model.id) == .ready {
                        Image(systemName: "checkmark.circle")
                            .foregroundStyle(MereTheme.textMuted)
                    }
                }
            } else {
            switch local.state(of: model.id) {
            case .notInstalled:
                Button {
                    if model.licenseNote != nil {
                        confirmingModel = model
                    } else {
                        Task { await local.download(model.id) }
                    }
                } label: {
                    Text("Get")
                        .font(.subheadline.weight(.semibold))
                }
                .buttonStyle(.bordered)
                .overlay(alignment: .bottom) {
                    Text(model.sizeLabel)
                        .font(.system(size: 9))
                        .foregroundStyle(MereTheme.textMuted)
                        .offset(y: 12)
                }
            case .downloading(let progress):
                VStack(alignment: .trailing, spacing: 2) {
                    ProgressView().controlSize(.small)
                    Text(progress)
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(MereTheme.textSecondary)
                }
            case .ready:
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(MereTheme.success)
            }
            }
        }
        .padding(.vertical, 4)
        .swipeActions(edge: .trailing, allowsFullSwipe: false) {
            if local.state(of: model.id) == .ready {
                Button(role: .destructive) {
                    Task { await local.delete(model.id) }
                } label: {
                    Label("Remove", systemImage: "trash")
                }
            }
        }
    }
}
