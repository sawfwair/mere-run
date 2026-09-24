import StudioKit
import SwiftUI

/// The one model menu, shared by every surface that picks a model: the composer's chip, the
/// Converse thread header, the inspector's model row, and the task workspace's chip. "Auto" (the
/// scope's default) comes first, then `model list` rows filtered to the scope's categories,
/// installed before downloadable, then a jump to Models. Every surface binds the same draft
/// field, so a model picked in one is the model the next run uses.
struct StudioModelPicker<Label: View>: View {
    let scope: StudioModelScope
    @Binding var model: String
    let modelInventory: [StudioModelInventoryRow]
    let onShowModels: () -> Void
    @ViewBuilder let label: () -> Label
    @Environment(\.studioModelTitles) private var titles

    init(
        scope: StudioModelScope,
        model: Binding<String>,
        modelInventory: [StudioModelInventoryRow],
        onShowModels: @escaping () -> Void,
        @ViewBuilder label: @escaping () -> Label
    ) {
        self.scope = scope
        _model = model
        self.modelInventory = modelInventory
        self.onShowModels = onShowModels
        self.label = label
    }

    init(
        mode: StudioMode,
        model: Binding<String>,
        modelInventory: [StudioModelInventoryRow],
        onShowModels: @escaping () -> Void,
        @ViewBuilder label: @escaping () -> Label
    ) {
        self.init(scope: StudioModelScope(mode: mode), model: model, modelInventory: modelInventory,
                  onShowModels: onShowModels, label: label)
    }

    var body: some View {
        Menu {
            let defaultID = scope.defaultModelID
            Toggle(isOn: Binding(get: { model.isBlank }, set: { _ in model = "" })) {
                Text(defaultID.isEmpty ? "Auto" : "Auto · \(StudioModelNaming.displayName(defaultID, titles: titles))")
            }
            let choices = scope.choices(from: modelInventory)
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
                Text("No \(scope.noun) models listed yet")
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
                StudioModelNaming.displayName(row),
                systemImage: row.isInstalled ? "internaldrive" : "arrow.down.circle"
            )
        }
    }
}

/// The model picker drawn as a composer chip: the resolved model's name, a glyph when the model
/// is not ready to run, and the exact id in the tooltip. The composer's chip strip, the Converse
/// thread header, and the task composer share it, so a model reads the same way wherever it is
/// picked.
struct StudioModelChip: View {
    let scope: StudioModelScope
    @Binding var model: String
    /// Every row of `model list`, installed or not; the menu filters it to the scope.
    let modelInventory: [StudioModelInventoryRow]
    let readiness: ModelReadinessState
    let onShowModels: () -> Void
    @Environment(\.studioModelTitles) private var titles

    init(
        scope: StudioModelScope,
        model: Binding<String>,
        modelInventory: [StudioModelInventoryRow],
        readiness: ModelReadinessState,
        onShowModels: @escaping () -> Void
    ) {
        self.scope = scope
        _model = model
        self.modelInventory = modelInventory
        self.readiness = readiness
        self.onShowModels = onShowModels
    }

    init(
        mode: StudioMode,
        model: Binding<String>,
        modelInventory: [StudioModelInventoryRow],
        readiness: ModelReadinessState,
        onShowModels: @escaping () -> Void
    ) {
        self.init(scope: StudioModelScope(mode: mode), model: model, modelInventory: modelInventory,
                  readiness: readiness, onShowModels: onShowModels)
    }

    var body: some View {
        StudioModelPicker(scope: scope, model: $model, modelInventory: modelInventory, onShowModels: onShowModels) {
            StudioComposerChipLabel(title: label, leadingSystemImage: statusGlyph)
        }
        .fixedSize()
        .help(help)
        .accessibilityLabel("Model")
        .accessibilityValue(accessibilityValue)
    }

    private var resolvedModelID: String {
        scope.resolvedModelID(model: model)
    }

    private var label: String {
        scope.displayLabel(model: model, titles: titles)
    }

    /// A glyph before the model name when the model is not ready: missing locally, or unsupported.
    private var statusGlyph: String? {
        switch readiness {
        case .missingModel: return "arrow.down.circle"
        case .unsupported: return "exclamationmark.triangle"
        case .notChecked, .checking, .ready, .unknown: return nil
        }
    }

    private var help: String {
        let identity = resolvedModelID.isEmpty ? "Auto — the task's default model" : "Model: \(resolvedModelID)"
        switch readiness {
        case .ready, .unknown, .notChecked: return identity
        default: return "\(identity) · \(readiness.message(titles: titles))"
        }
    }

    private var accessibilityValue: String {
        let identity = resolvedModelID.isEmpty ? "Automatic" : resolvedModelID
        return "\(identity), \(readiness.title)"
    }
}
