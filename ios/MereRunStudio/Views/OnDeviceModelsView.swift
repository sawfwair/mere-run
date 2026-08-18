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

    var body: some View {
        List {
                Section {
                    ForEach(LocalEngine.imageModels) { model in
                        row(model)
                    }
                } header: {
                    Text("Images")
                } footer: {
                    Text("Pick which one creates your images from the model menu in Create.")
                }

                Section {
                    row(LocalEngine.chatModel)
                } header: {
                    Text("Chat")
                } footer: {
                    Text("Choose it from the model menu in Chat to talk entirely on-device.")
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
        .onAppear { local.refresh() }
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
            switch local.state(of: model.id) {
            case .notInstalled:
                Button {
                    Task { await local.download(model.id) }
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
        .padding(.vertical, 4)
    }
}
