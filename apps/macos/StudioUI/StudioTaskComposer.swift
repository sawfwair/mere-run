import AppKit
import MereRunContract
import StudioKit
import SwiftUI

/// The composer of a task on the shared task workspace: the same panel as `StudioComposer` —
/// attachment well, prompt, chip strip, Run — drawn from the task's contract instead of a mode.
/// The well's slots are the capability's file inputs, the chips its `essential` options, the
/// model chip its template's scope, and every control binds the one `StudioTaskDraft` the
/// inspector and the Command view edit.
struct StudioTaskComposer: View {
    let task: StudioTask
    @Binding var draft: StudioTaskDraft
    let isRunning: Bool
    let queuedCount: Int
    let readiness: ModelReadinessState
    let modelInventory: [StudioModelInventoryRow]
    /// Whether the prompt field is drawn here. The Analyze canvas's `.text` input takes the
    /// positional itself, so the composer leaves it out for those tasks.
    var showsPrompt = true
    var promptFocus: FocusState<Bool>.Binding
    let onRun: () -> Void
    let onStop: () -> Void
    let onShowModels: () -> Void
    @Environment(\.studioModelTitles) private var titles

    @State private var editingChip: String?

    private enum Metrics {
        static let outerInsets = EdgeInsets(top: 12, leading: 24, bottom: 20, trailing: 24)
        static let innerInsets = EdgeInsets(top: 14, leading: 16, bottom: 12, trailing: 16)
        static let cornerRadius: CGFloat = 18
        static let rowSpacing: CGFloat = 12
        static let sendDiameter: CGFloat = 32
    }

    private var slots: [StudioAttachmentSlot] { draft.slots }
    private var presentation: StudioTaskPresentation { task.presentation }
    private var promptField: StudioTaskPromptField? { draft.capability.flatMap(StudioTaskSchema.promptField) }
    private var essentials: [StudioContractField<StudioTaskDraft>] { StudioTaskSchema.essentials(for: task, draft: draft) }

    var body: some View {
        VStack(alignment: .leading, spacing: Metrics.rowSpacing) {
            if !slots.isEmpty {
                attachmentWell
            }
            if showsPrompt, promptField != nil {
                promptEntry
            }
            chipStrip
        }
        .padding(Metrics.innerInsets)
        .background {
            RoundedRectangle(cornerRadius: Metrics.cornerRadius)
                .fill(MereRunTheme.surface)
                .overlay {
                    RoundedRectangle(cornerRadius: Metrics.cornerRadius)
                        .strokeBorder(MereRunTheme.border.opacity(promptFocus.wrappedValue ? 0 : 1), lineWidth: 1)
                }
                .mereShadow(radius: 12, y: 8)
        }
        .mereFocusRing(promptFocus.wrappedValue, cornerRadius: Metrics.cornerRadius)
        .padding(Metrics.outerInsets)
    }

    // MARK: - Attachment well

    private var attachmentWell: some View {
        HStack(alignment: .center, spacing: 8) {
            ForEach(slots) { slot in
                StudioAttachmentSlotView(slot: slot, draft: $draft, onPick: { StudioAttachmentPicker.pick(for: slot, into: &draft) })
            }
            VStack(alignment: .leading, spacing: 1) {
                ForEach(slots) { slot in
                    Text(slot.caption(in: draft))
                        .font(.caption.weight(.medium))
                        .foregroundStyle(MereRunTheme.textMuted)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
            }
            .padding(.leading, 4)
            Spacer(minLength: 0)
        }
    }

    // MARK: - Prompt

    private var promptEntry: some View {
        TextField(
            "",
            text: $draft.prompt,
            prompt: Text(presentation.promptPlaceholder.isEmpty ? "Prompt" : presentation.promptPlaceholder)
                .foregroundStyle(MereRunTheme.textMuted),
            axis: .vertical
        )
        .textFieldStyle(.plain)
        .font(.system(size: 15))
        .lineSpacing(3)
        .foregroundStyle(MereRunTheme.textPrimary)
        .lineLimit(1...5)
        .frame(minHeight: 22, alignment: .leading)
        .focused(promptFocus)
        .onSubmit(onRun)
        .accessibilityLabel(presentation.promptPlaceholder.isEmpty ? "Prompt" : presentation.promptPlaceholder)
    }

    // MARK: - Chip strip

    private var chipStrip: some View {
        HStack(alignment: .center, spacing: 6) {
            ForEach(essentials) { field in
                chip(for: field)
            }
            modelChip
            Spacer(minLength: 8)
            if isRunning { stopButton } else { sendButton }
        }
    }

    /// An essential option as a chip: its current value in the chip, its contract control in a
    /// popover. A choice with few values is a menu straight from the chip, like the mode chips.
    @ViewBuilder
    private func chip(for field: StudioContractField<StudioTaskDraft>) -> some View {
        if field.overrideID == .variant {
            variantChip(field)
        } else if field.control == .segmented || field.control == .picker {
            choiceChip(field)
        } else {
            Button {
                editingChip = field.id
            } label: {
                StudioComposerChipLabel(title: chipTitle(field), menu: false)
            }
            .buttonStyle(.plain)
            .accessibilityLabel(field.label)
            .accessibilityValue(chipTitle(field))
            .popover(
                isPresented: Binding(get: { editingChip == field.id }, set: { if !$0 { editingChip = nil } }),
                arrowEdge: .top
            ) {
                ContractFormControl(field: field, draft: $draft)
                    .frame(width: 260)
                    .padding(MereRunTheme.Spacing.md)
                    .background(MereRunTheme.background)
                    .foregroundStyle(MereRunTheme.textPrimary)
            }
        }
    }

    private func choiceChip(_ field: StudioContractField<StudioTaskDraft>) -> some View {
        Menu {
            ForEach(field.option.choices, id: \.self) { choice in
                Toggle(isOn: Binding(
                    get: { (field.value(in: draft).text ?? field.option.defaultValue) == choice },
                    set: { _ in field.write(.text(choice), to: &draft) }
                )) {
                    Text(StudioContractChoiceTitles.title(for: choice, flag: field.flag))
                }
            }
        } label: {
            StudioComposerChipLabel(title: chipTitle(field))
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .fixedSize()
        .accessibilityLabel(field.label)
        .accessibilityValue(chipTitle(field))
    }

    private func variantChip(_ field: StudioContractField<StudioTaskDraft>) -> some View {
        Menu {
            ForEach(task.variantTemplates) { template in
                Toggle(isOn: Binding(
                    get: { draft.templateID == template.id },
                    set: { _ in draft.switchTemplate(to: template.id) }
                )) {
                    Text(template.title)
                }
            }
        } label: {
            StudioComposerChipLabel(title: draft.template?.title ?? field.label)
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .fixedSize()
        .accessibilityLabel(field.label)
        .accessibilityValue(draft.template?.title ?? "")
    }

    /// "Threshold 0.3", "Native", "Latency offline": the label with the value, or the label alone
    /// for a switch that is on, or "Label auto" when the CLI picks.
    private func chipTitle(_ field: StudioContractField<StudioTaskDraft>) -> String {
        let value = field.value(in: draft)
        switch value {
        case .flag(let on):
            return on ? field.label : "\(field.label) off"
        case .unset:
            if let fallback = field.option.defaultValue, !fallback.isEmpty {
                return "\(field.label) \(StudioContractChoiceTitles.title(for: fallback, flag: field.flag).lowercased())"
            }
            return "\(field.label) auto"
        case .text(let text):
            if text.isEmpty { return "\(field.label) auto" }
            let shown = field.option.choices.isEmpty ? text : StudioContractChoiceTitles.title(for: text, flag: field.flag)
            return "\(field.label) \(shown)"
        case .integer(let integer):
            return "\(field.label) \(integer)"
        case .number(let number):
            return "\(field.label) \(StudioComposerPresets.decimalText(number))"
        }
    }

    // MARK: - Model chip

    private var modelChip: some View {
        StudioModelChip(
            scope: StudioTaskSchema.modelScope(for: draft),
            model: $draft.model,
            modelInventory: modelInventory,
            readiness: readiness,
            onShowModels: onShowModels
        )
    }

    // MARK: - Right cluster

    private var stopButton: some View {
        Button(action: onStop) {
            ZStack {
                Circle().fill(MereRunTheme.surfaceRaised)
                Image(systemName: "stop.fill")
                    .font(.caption.weight(.bold))
                    .foregroundStyle(MereRunTheme.textPrimary)
            }
            .frame(width: Metrics.sendDiameter, height: Metrics.sendDiameter)
        }
        .buttonStyle(.plain)
        .help(queuedCount == 0 ? "Stop (⌘.) · ⌘↩ queues another run" : "Stop (⌘.) · \(queuedCount) queued")
        .accessibilityLabel("Stop current run")
    }

    private var sendButton: some View {
        Button(action: onRun) {
            ZStack {
                Circle().fill(sendEnabled ? MereRunTheme.accent : MereRunTheme.surfaceRaised)
                Image(systemName: "arrow.up")
                    .font(.body.weight(.semibold))
                    .foregroundStyle(sendEnabled ? MereRunTheme.onAccent : MereRunTheme.textMuted)
            }
            .frame(width: Metrics.sendDiameter, height: Metrics.sendDiameter)
        }
        .buttonStyle(.plain)
        .disabled(!sendEnabled)
        .help(readiness.blocksRun ? readiness.message(titles: titles) : "Run (⌘↩)")
        .accessibilityLabel("Run")
        .accessibilityHint(readiness.blocksRun ? readiness.message(titles: titles) : "")
        .keyboardShortcut(.return, modifiers: .command)
    }

    private var sendEnabled: Bool {
        !readiness.blocksRun
    }
}
