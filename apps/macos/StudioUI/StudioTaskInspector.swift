import AppKit
import MereRunContract
import StudioKit
import SwiftUI

/// The inspector column of a task on the shared task workspace, rendered from the task's
/// contract the way `StudioInspector` renders a mode's: sections by the contract's `group`, a
/// disclosure for the `expert` tier, one control per option. Every control binds the task draft
/// the composer's chips edit, so a value changed in either shows in both, and the header counts
/// what differs from the template's defaults.
///
/// The composite editors a page drew by hand (cameras, ordered views, the manifest, face picks,
/// instruments, renoise, target ranks) come through the override builder. Until a page PR moves
/// its editor here, the builder draws each of the override's flags with the plain contract
/// control, so nothing the command takes is out of reach.
struct StudioTaskInspector: View {
    let task: StudioTask
    @Binding var draft: StudioTaskDraft
    let modelInventory: [StudioModelInventoryRow]
    let readiness: ModelReadinessState
    let onShowModels: () -> Void
    let onClose: () -> Void
    @Environment(\.studioModelTitles) private var titles

    @State private var showAdvanced = false

    static let width = StudioLayoutPolicy.inspectorWidth

    private var baseline: StudioTaskDraft { StudioTaskDraft(templateID: draft.templateID) }
    private var sections: [StudioTaskSection] { StudioTaskSchema.sections(for: task, draft: draft) }
    private var advancedFields: [StudioContractField<StudioTaskDraft>] { StudioTaskSchema.advanced(for: task, draft: draft) }
    private var changedCount: Int { StudioTaskSchema.changedCount(for: task, draft: draft) }
    private var dependencies: [String: (carries: Bool, dependsOn: String?)] { StudioTaskSchema.dependencies(for: draft) }

    var body: some View {
        VStack(spacing: 0) {
            header
            ScrollView {
                VStack(spacing: 0) {
                    ForEach(sections) { section in
                        sectionView(section)
                    }
                    outputSection
                    if !advancedFields.isEmpty {
                        advancedSection
                    }
                }
            }
        }
        .frame(width: Self.width)
        .background(MereRunTheme.background)
        .overlay(alignment: .leading) {
            Rectangle()
                .fill(MereRunTheme.border.opacity(0.53))
                .frame(width: 1)
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Inspector")
    }

    // MARK: Header

    private var header: some View {
        HStack(spacing: 8) {
            Text("\(task.domain.title) · \(task.title)")
                .font(.system(size: 12.5, weight: .semibold))
                .foregroundStyle(MereRunTheme.textPrimary)
                .lineLimit(1)
            if changedCount > 0 {
                Text("\(changedCount) changed")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(MereRunTheme.accent)
                    .padding(.horizontal, 6)
                    .frame(height: 18)
                    .background { Capsule().fill(MereRunTheme.accentSoft) }
                    .accessibilityLabel("\(changedCount) settings changed from the defaults")
            }
            Spacer(minLength: 0)
            Button(action: onClose) {
                Image(systemName: "xmark")
                    .font(.caption.weight(.semibold))
                    .frame(width: 22, height: 22)
            }
            .buttonStyle(.mereIcon(tint: MereRunTheme.textMuted))
            .help("Hide Inspector (⌥⌘I)")
            .accessibilityLabel("Hide Inspector")
        }
        .padding(.horizontal, 16)
        .frame(height: 44)
        .overlay(alignment: .bottom) {
            Rectangle().fill(MereRunTheme.border.opacity(0.4)).frame(height: 1)
        }
    }

    // MARK: Sections

    private func sectionView(_ section: StudioTaskSection) -> some View {
        StudioInspectorSectionView(
            title: section.title,
            canReset: section.changedCount(draft: draft, baseline: baseline) > 0,
            onReset: { section.reset(&draft, to: baseline) }
        ) {
            form(section.fields)
        }
    }

    /// Where the run saves, without a path field: routing names the file after the input when
    /// the run starts (`StudioOutputLocation.destination(for:)`), inside the domain's folder.
    @ViewBuilder
    private var outputSection: some View {
        if let flag = draft.capability?.output.flag, draft.capability?.options.contains(where: { $0.flag == flag }) == true {
            let folder = URL(fileURLWithPath: draft.text(flag)).deletingLastPathComponent()
            StudioInspectorSectionView(title: "Output", canReset: false, onReset: {}) {
                HStack(spacing: 8) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Saves to \(StudioOutputLocation.abbreviate(folder))")
                            .font(.system(size: 12.5, weight: .medium))
                            .foregroundStyle(MereRunTheme.textPrimary)
                            .lineLimit(2)
                            .truncationMode(.middle)
                        Text("Named after the input when the run starts.")
                            .font(MereRunTheme.captionFont)
                            .foregroundStyle(MereRunTheme.textMuted)
                    }
                    Spacer(minLength: 4)
                    Button("Reveal") { NSWorkspace.shared.activateFileViewerSelecting([folder]) }
                        .buttonStyle(.mereSecondary)
                        .help("Show the folder in Finder")
                }
                SettingsLink {
                    Text("Change in Settings…")
                        .font(.caption.weight(.medium))
                        .foregroundStyle(MereRunTheme.accent)
                }
                .buttonStyle(.plain)
            }
        }
    }

    private var advancedSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 6) {
                Button {
                    withAnimation(MereRunTheme.Motion.quick) { showAdvanced.toggle() }
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: showAdvanced ? "chevron.down" : "chevron.right")
                            .font(.system(size: 9, weight: .semibold))
                            .foregroundStyle(MereRunTheme.textMuted)
                        Text("Advanced · \(advancedFields.count) more")
                            .font(.callout.weight(.medium))
                            .foregroundStyle(MereRunTheme.textSecondary)
                    }
                }
                .buttonStyle(.plain)
                .accessibilityLabel(showAdvanced ? "Hide advanced settings" : "Show \(advancedFields.count) advanced settings")
                Spacer(minLength: 0)
                if showAdvanced, advancedFields.contains(where: { $0.changedCount(draft: draft, baseline: baseline) > 0 }) {
                    Button("Reset") {
                        for field in advancedFields { field.reset(&draft, to: baseline) }
                    }
                    .buttonStyle(.plain)
                    .font(.caption.weight(.medium))
                    .foregroundStyle(MereRunTheme.textMuted)
                    .help("Back to the defaults")
                }
            }
            if showAdvanced {
                form(advancedFields)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 14)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func form(_ fields: [StudioContractField<StudioTaskDraft>]) -> some View {
        ContractForm(fields: fields, dependencies: dependencies, draft: $draft) { override in
            overrideControl(override, in: fields)
        }
    }

    // MARK: Overrides

    @ViewBuilder
    private func overrideControl(_ override: StudioContractOverrideID, in fields: [StudioContractField<StudioTaskDraft>]) -> some View {
        switch override {
        case .model:
            modelPicker
        case .variant:
            variantControl
        case .renoise:
            StudioRenoiseOverride(draft: $draft)
        case .cameras, .orderedViews, .musicManifest, .faceIndex, .instruments, .targetRanks:
            // The page PR that owns the editor replaces this with it; until then every flag the
            // editor would write stays reachable as its plain control.
            plainControls(for: override, in: fields)
        case .dimensions, .seed, .steps, .guidance, .duration, .voiceProfile, .lora, .imageCanvas, .musicAdapters,
             .musicLMMode, .thinking, .orderedReferences, .readImageAction, .attachment, .regionPrompts:
            // Prompt-mode editors; a task template never declares them.
            EmptyView()
        }
    }

    @ViewBuilder
    private func plainControls(for override: StudioContractOverrideID, in fields: [StudioContractField<StudioTaskDraft>]) -> some View {
        if let field = fields.first(where: { $0.overrideID == override }), let capability = draft.capability {
            ForEach(capability.options.filter { option in
                field.bindings.contains { $0.fieldID == option.flag }
            }, id: \.flag) { option in
                ContractFormControl(field: StudioContractField(option: option, bindings: [.flag(option.flag)]), draft: $draft)
            }
        }
    }

    private var variantControl: some View {
        let templates = task.variantTemplates
        let selection = Binding(
            get: { draft.templateID },
            set: { draft.switchTemplate(to: $0) }
        )
        return StudioInspectorLabeledRow(task == .threeDFromImage ? "Engine" : "Operation") {
            if templates.count <= 4 {
                MereSegmentedControl(templates.map(\.id), selection: selection, accessibilityLabel: "Variant") {
                    StudioTaskSchema.variantTitle($0.rawValue)
                }
            } else {
                Picker("Variant", selection: selection) {
                    ForEach(templates) { template in
                        Text(template.title).tag(template.id)
                    }
                }
                .labelsHidden()
                .frame(maxWidth: 170)
            }
        }
    }

    private var modelPicker: some View {
        let scope = StudioTaskSchema.modelScope(for: draft)
        return StudioModelPicker(scope: scope, model: $draft.model, modelInventory: modelInventory, onShowModels: onShowModels) {
            HStack(spacing: 8) {
                if let glyph = modelStatusGlyph {
                    Image(systemName: glyph)
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(MereRunTheme.accent)
                }
                Text(scope.displayLabel(model: draft.model, titles: titles))
                    .font(.callout)
                    .foregroundStyle(MereRunTheme.textPrimary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Spacer(minLength: 0)
                Image(systemName: "chevron.down")
                    .font(.system(size: 9, weight: .semibold))
                    .foregroundStyle(MereRunTheme.textMuted)
            }
            .padding(.horizontal, 10)
            .frame(maxWidth: .infinity, minHeight: 32)
            .background { StudioInspectorFieldChrome() }
            .contentShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.base))
        }
        .help(readiness.blocksRun ? readiness.message(titles: titles) : "Model")
        .accessibilityLabel("Model")
        .accessibilityValue(scope.resolvedModelID(model: draft.model))
    }

    private var modelStatusGlyph: String? {
        switch readiness {
        case .missingModel: return "arrow.down.circle"
        case .unsupported: return "exclamationmark.triangle"
        case .notChecked, .checking, .ready, .unknown: return nil
        }
    }
}
