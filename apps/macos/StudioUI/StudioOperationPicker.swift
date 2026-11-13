import MereRunContract
import StudioKit
import SwiftUI

/// One operation selector, bound to the same field as the inspector and Command view.
struct StudioOperationSelection: Identifiable {
    struct Choice: Identifiable {
        let id: String
        let title: String
        var unavailableReason: String? = nil
    }

    let id: String
    let label: String
    let choices: [Choice]
    let value: Binding<String>

    var selectedTitle: String { choices.first { $0.id == value.wrappedValue }?.title ?? value.wrappedValue }
    var chipTitle: String { "\(label) · \(selectedTitle)" }

    init<Draft>(
        field: StudioContractField<Draft>, draft: Binding<Draft>,
        title: (String) -> String, unavailable: (String) -> String? = { _ in nil }
    ) {
        id = field.flag
        label = field.overrideID == .readImageAction ? "Operation" : field.label
        choices = field.option.choices.map { Choice(id: $0, title: title($0), unavailableReason: unavailable($0)) }
        value = Binding(get: { field.value(in: draft.wrappedValue).text ?? field.option.defaultValue ?? "" }, set: { value in
            var next = draft.wrappedValue
            field.write(.text(value), to: &next)
            draft.wrappedValue = next
        })
    }

    static func task(_ task: StudioTask, draft: Binding<StudioTaskDraft>, source: StudioScopeSource) -> [Self] {
        StudioTaskSchema.operationFields(for: task, draft: draft.wrappedValue, source: source).map { field in
            Self(field: field, draft: draft) { choice in
                field.overrideID == .variant ? StudioTaskSchema.variantTitle(choice)
                    : StudioContractChoiceTitles.title(for: choice, flag: field.flag)
            }
        }
    }

    static func prompt(
        _ mode: StudioMode, draft: Binding<StudioDraft>, source: StudioScopeSource,
        capabilities: [String: StudioModelCapability] = [:]
    ) -> [Self] {
        let routing = source.scope(mode: mode, draft: draft.wrappedValue)?.capability.routing
        let modelFlags = Set((routing?.modelFlags ?? []) + (routing?.families.compactMap(\.modelFlag) ?? []))
        return StudioContractSchema.fields(for: mode, readImageAction: draft.wrappedValue.readImageAction).filter { field in
            field.overrideID == .readImageAction || (field.option.kind == .choice
                && routing?.routingFlags.contains(field.flag) == true && !modelFlags.contains(field.flag))
        }.map { field in
            Self(field: field, draft: draft, title: { choice in
                field.overrideID == .readImageAction ? (StudioReadImageAction(rawValue: choice)?.title ?? choice)
                    : StudioContractChoiceTitles.title(for: choice, flag: field.flag)
            }, unavailable: { choice in
                guard field.overrideID == .readImageAction else { return nil }
                var next = draft.wrappedValue
                field.write(.text(choice), to: &next)
                switch StudioCommandAdapter.capabilityRequirement(for: mode, draft: next, source: source) {
                case .unavailable(let reason): return reason
                case .managedModel(let id): return capabilities[id]?.unavailableMessage(titles: .none)
                case nil: return nil
                }
            })
        }
    }
}

struct StudioOperationMenuItems: View {
    let selection: StudioOperationSelection

    var body: some View {
        ForEach(selection.choices) { choice in
            Toggle(isOn: Binding(get: { selection.value.wrappedValue == choice.id }, set: { _ in
                selection.value.wrappedValue = choice.id
            })) { Text(choice.title) }
            .disabled(choice.unavailableReason != nil)
            .help(choice.unavailableReason ?? choice.title)
        }
    }
}

struct StudioOperationPicker: View {
    let selection: StudioOperationSelection

    var body: some View {
        Menu { StudioOperationMenuItems(selection: selection) } label: {
            StudioComposerChipLabel(title: selection.chipTitle)
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .fixedSize()
        .accessibilityLabel(selection.label)
        .accessibilityValue(selection.selectedTitle)
        .help("Choose \(selection.label.lowercased()) before choosing its model")
    }
}

/// The inspector keeps the same unrestricted operation choices above model-specific settings.
struct StudioOperationSection: View {
    let selections: [StudioOperationSelection]
    let defaults: [StudioOperationSelection]
    @Environment(\.studioTaskSessions) private var sessions

    private var changed: [StudioOperationSelection] {
        selections.filter { selection in
            defaults.contains { $0.id == selection.id && $0.value.wrappedValue != selection.value.wrappedValue }
        }
    }

    var body: some View {
        if !selections.isEmpty {
            StudioInspectorSectionView(title: "Operation", canReset: !changed.isEmpty, onReset: {
                StudioUndoNaming.reset("Operation", in: sessions) {
                    for selection in changed {
                        if let baseline = defaults.first(where: { $0.id == selection.id }) {
                            selection.value.wrappedValue = baseline.value.wrappedValue
                        }
                    }
                }
            }) {
                ForEach(selections) { selection in StudioOperationPicker(selection: selection) }
            }
        }
    }
}
