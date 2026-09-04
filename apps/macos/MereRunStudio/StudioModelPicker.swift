import SwiftUI

/// The one model control, shared by the composer's model chip and the inspector's model row:
/// "Auto" (the mode's default), then `model list` rows filtered to the mode's category,
/// installed first, then a jump to Models. Both surfaces bind the same draft field.
struct StudioModelPicker<Label: View>: View {
    let mode: StudioMode
    @Binding var model: String
    let modelInventory: [StudioModelInventoryRow]
    let onShowModels: () -> Void
    @ViewBuilder let label: () -> Label

    /// The mode's template default, shown as "Auto".
    static func defaultModelID(for mode: StudioMode) -> String {
        CommandCatalog.template(id: mode.defaultTemplateID)?.defaultModel ?? ""
    }

    /// The resolved model id for a run: the explicit draft model, else the mode's default.
    static func resolvedModelID(for mode: StudioMode, model: String) -> String {
        let current = model.trimmingCharacters(in: .whitespacesAndNewlines)
        return current.isEmpty ? defaultModelID(for: mode) : current
    }

    static func displayLabel(for mode: StudioMode, model: String) -> String {
        let resolved = resolvedModelID(for: mode, model: model)
        return resolved.isEmpty ? "Auto" : StudioComposer.displayModelName(resolved)
    }

    var body: some View {
        Menu {
            let defaultID = Self.defaultModelID(for: mode)
            Toggle(isOn: Binding(get: { model.isBlank }, set: { _ in model = "" })) {
                Text(defaultID.isEmpty ? "Auto" : "Auto · \(StudioComposer.displayModelName(defaultID))")
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
                StudioComposer.displayModelName(row.id),
                systemImage: row.isInstalled ? "internaldrive" : "arrow.down.circle"
            )
        }
    }
}
