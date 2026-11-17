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
    let checkpointFolder: Binding<String?>?
    @ViewBuilder let label: () -> Label

    init(
        scope: StudioModelScope,
        model: Binding<String>,
        modelInventory: [StudioModelInventoryRow],
        onShowModels: @escaping () -> Void,
        checkpointFolder: Binding<String?>? = nil,
        @ViewBuilder label: @escaping () -> Label
    ) {
        self.scope = scope
        _model = model
        self.modelInventory = modelInventory
        self.onShowModels = onShowModels
        self.checkpointFolder = checkpointFolder
        self.label = label
    }

    var body: some View {
        Menu { items } label: {
            label()
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .accessibilityLabel("Model")
        .help("Choose a model for \(scope.noun)")
    }

    var items: some View {
        StudioModelMenuItems(
            scope: scope, model: $model, modelInventory: modelInventory,
            onShowModels: onShowModels, checkpointFolder: checkpointFolder
        )
    }
}

/// A view of its own so the recovery menu reads the same live inventory titles as the picker.
struct StudioModelMenuItems: View {
    let scope: StudioModelScope
    @Binding var model: String
    let modelInventory: [StudioModelInventoryRow]
    let onShowModels: () -> Void
    var checkpointFolder: Binding<String?>?
    @Environment(\.studioModelTitles) private var titles

    var body: some View {
        Text("Models for \(scope.noun)")
        let defaultID = scope.resolvedModelID(model: scope.selectingInstalledDefault(model: "", from: modelInventory))
        // A filled draft selects its named model; only an empty model selects Auto.
        Toggle(isOn: Binding(get: { model.isBlank }, set: { _ in model = "" })) {
            Text(defaultID.isEmpty ? "Auto" : "Auto · \(StudioModelNaming.displayName(defaultID, titles: titles))")
        }
        if let checkpointFolder {
            Section("Local checkpoint") {
                Button("Choose checkpoint folder…") {
                    if let folder = StudioFilePanels.chooseDirectory(title: "Choose checkpoint folder") {
                        checkpointFolder.wrappedValue = folder.path
                    }
                }
                if let path = checkpointFolder.wrappedValue, !path.isBlank {
                    Button("Clear checkpoint folder") { checkpointFolder.wrappedValue = nil }
                }
            }
        }
        let choices = scope.choices(from: modelInventory)
        if hasCheckpoint, !choices.isEmpty {
            Section("Runtime profile") {
                ForEach(choices) { row in modelRow(row) }
            }
        } else {
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
        }
        if choices.isEmpty {
            if scope.usesLocalCheckpoints {
                Text("This operation uses local checkpoints")
                Button("Choose checkpoint folder…") {
                    if let folder = StudioFilePanels.chooseDirectory(title: "Choose model checkpoints") {
                        model = folder.path
                    }
                }
            } else {
                Text("No compatible models listed yet")
            }
        }
        Divider()
        Button("Browse Models…", action: onShowModels)
    }

    private var hasCheckpoint: Bool {
        checkpointFolder?.wrappedValue?.isBlank == false
    }

    private func modelRow(_ row: StudioModelInventoryRow) -> some View {
        Toggle(isOn: Binding(get: { row.id == model }, set: { _ in model = row.id })) {
            SwiftUI.Label(
                StudioModelNaming.displayName(row),
                systemImage: hasCheckpoint ? "cpu" : (row.isInstalled ? "internaldrive" : "arrow.down.circle")
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
    let checkpointFolder: Binding<String?>?
    @Environment(\.studioModelTitles) private var titles

    init(
        scope: StudioModelScope,
        model: Binding<String>,
        modelInventory: [StudioModelInventoryRow],
        readiness: ModelReadinessState,
        onShowModels: @escaping () -> Void,
        checkpointFolder: Binding<String?>? = nil
    ) {
        self.scope = scope
        _model = model
        self.modelInventory = modelInventory
        self.readiness = readiness
        self.onShowModels = onShowModels
        self.checkpointFolder = checkpointFolder
    }

    var body: some View {
        StudioModelPicker(
            scope: scope, model: $model, modelInventory: modelInventory,
            onShowModels: onShowModels, checkpointFolder: checkpointFolder
        ) {
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

    private var checkpointPath: String? {
        guard let path = checkpointFolder?.wrappedValue, !path.isBlank else { return nil }
        return path
    }

    private var label: String {
        if let path = checkpointPath {
            return "Local · \(URL(fileURLWithPath: path).lastPathComponent)"
        }
        let name = scope.displayLabel(model: model, titles: titles)
        return "Model · \(name)"
    }

    /// A glyph before the model name when the model is not ready: missing locally, or unsupported.
    private var statusGlyph: String? {
        switch readiness {
        case .missingModel: return "arrow.down.circle"
        case .unsupported: return "exclamationmark.triangle"
        case .notChecked, .checking, .ready, .unknown: return checkpointPath == nil ? nil : "folder"
        }
    }

    private var help: String {
        let runtime = resolvedModelID.isEmpty ? "Auto — the mode's default model" : "Model: \(resolvedModelID)"
        let identity = checkpointPath.map { "\($0) · \(runtime)" } ?? runtime
        switch readiness {
        case .ready, .unknown, .notChecked: return identity
        default: return "\(identity) · \(readiness.message(titles: titles))"
        }
    }

    private var accessibilityValue: String {
        let identity = resolvedModelID.isEmpty ? "Automatic" : resolvedModelID
        if let path = checkpointPath {
            return "Local checkpoint \(URL(fileURLWithPath: path).lastPathComponent), \(identity), \(readiness.title)"
        }
        return "\(identity), \(readiness.title)"
    }
}
