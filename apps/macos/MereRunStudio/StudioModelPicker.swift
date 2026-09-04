import SwiftUI

/// The one model menu, shared by every surface that picks a model: the composer's chip, the
/// Converse thread header, and the inspector's model row. "Auto" (the mode's default) comes
/// first, then `model list` rows filtered to the mode's categories, installed before
/// downloadable, then a jump to Models. Every surface binds the same draft field, so a model
/// picked in one is the model the next run uses.
struct StudioModelPicker<Label: View>: View {
    let mode: StudioMode
    @Binding var model: String
    let modelInventory: [StudioModelInventoryRow]
    let onShowModels: () -> Void
    @ViewBuilder let label: () -> Label

    var body: some View {
        Menu {
            let defaultID = StudioModelNaming.defaultModelID(for: mode)
            Toggle(isOn: Binding(get: { model.isBlank }, set: { _ in model = "" })) {
                Text(defaultID.isEmpty ? "Auto" : "Auto · \(StudioModelNaming.displayName(defaultID))")
            }
            let choices = mode.modelChoices(from: modelInventory)
            let installed = choices.filter(\.isInstalled)
            let downloadable = choices.filter { !$0.isInstalled }
            if !installed.isEmpty {
                Section("Installed") {
                    ForEach(installed) { row in modelRow(row) }
                }
            }
            if !downloadable.isEmpty {
                Section("Needs download") {
                    ForEach(downloadable) { row in modelRow(row) }
                }
            }
            if choices.isEmpty {
                Text("No \(mode.title.lowercased()) models listed yet")
            }
            Divider()
            Button("Browse Models…", action: onShowModels)
        } label: {
            label()
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
    }

    private func modelRow(_ row: StudioModelInventoryRow) -> some View {
        Toggle(isOn: Binding(get: { row.id == model }, set: { _ in model = row.id })) {
            SwiftUI.Label(
                StudioModelNaming.displayName(row.id),
                systemImage: row.isInstalled ? "internaldrive" : "arrow.down.circle"
            )
        }
    }
}

/// The model picker drawn as a composer chip: the resolved model's name, a glyph when the model
/// is not ready to run, and the exact id in the tooltip. The composer's chip strip and the
/// Converse thread header share it, so a model reads the same way wherever it is picked.
struct StudioModelChip: View {
    let mode: StudioMode
    @Binding var model: String
    /// Every row of `model list`, installed or not; the menu filters it to the mode.
    let modelInventory: [StudioModelInventoryRow]
    let readiness: ModelReadinessState
    let onShowModels: () -> Void

    var body: some View {
        StudioModelPicker(mode: mode, model: $model, modelInventory: modelInventory, onShowModels: onShowModels) {
            StudioComposerChipLabel(title: label, leadingSystemImage: statusGlyph)
        }
        .fixedSize()
        .help(help)
        .accessibilityLabel("Model")
        .accessibilityValue(accessibilityValue)
    }

    private var resolvedModelID: String {
        StudioModelNaming.resolvedModelID(for: mode, model: model)
    }

    private var label: String {
        StudioModelNaming.displayLabel(for: mode, model: model)
    }

    /// A glyph before the model name when the model is not ready: missing locally, or unsupported.
    private var statusGlyph: String? {
        switch readiness {
        case .missingModel: return "arrow.down.circle"
        case .unsupported: return "exclamationmark.triangle"
        case .checking, .ready, .unknown: return nil
        }
    }

    private var help: String {
        let identity = resolvedModelID.isEmpty ? "Auto — the mode's default model" : "Model: \(resolvedModelID)"
        switch readiness {
        case .ready, .unknown: return identity
        default: return "\(identity) · \(readiness.message)"
        }
    }

    private var accessibilityValue: String {
        let identity = resolvedModelID.isEmpty ? "Automatic" : resolvedModelID
        return "\(identity), \(readiness.title)"
    }
}
