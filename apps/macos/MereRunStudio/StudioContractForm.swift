import AppKit
import MereRunContract
import SwiftUI
import UniformTypeIdentifiers

/// Renders a capability's options straight from the contract: one control per `kind`, sliders and
/// steppers sized by the declared `range`, pickers filled from `choices`, rows hidden until their
/// `depends_on` option carries a value, and nothing but the draft field each option is bound to
/// underneath. Where a capability's option needs an editor the contract cannot describe — the
/// aspect pair, the seed row, the mask canvas, the ordered MiniMax references — the caller's
/// `override` builder draws it instead, keyed by `StudioContractOverrideID`.
///
/// The form never builds a command. It writes whatever draft the surface hands it: the prompt
/// tasks pass the typed `StudioDraft` the composer and inspector share, so `StudioCommandAdapter`
/// and `CommandTemplate.arguments(from:)` decide the argv exactly as before, and the Command
/// Console passes a `StudioConsoleDraft`, whose entries the contract itself turns into argv.
/// How a contract row names the option it edits.
enum ContractFormLabelStyle {
    /// The option's user-language label, at the head of an inspector row.
    case label
    /// The raw flag in a fixed monospaced column, the way the Command board draws it.
    case flag
}

struct ContractForm<Draft, Override: View>: View {
    let fields: [StudioContractField<Draft>]
    /// Per flag, whether the draft gives it a value and what it in turn depends on, so a row
    /// gated on another option stays hidden until that option carries something. The caller
    /// supplies it because the dependency walk spans every flag, including the ones a composite
    /// editor folds away and the ones another column owns.
    let dependencies: [String: (carries: Bool, dependsOn: String?)]
    @Binding var draft: Draft
    var labelStyle: ContractFormLabelStyle = .label
    @ViewBuilder let override: (StudioContractOverrideID) -> Override

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(visibleFields) { field in
                row(field)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// The fields whose dependencies are satisfied, in contract order.
    private var visibleFields: [StudioContractField<Draft>] {
        fields.filter { StudioContractSchema.isVisible($0, in: draft, dependencies: dependencies) }
    }

    @ViewBuilder
    private func row(_ field: StudioContractField<Draft>) -> some View {
        if let id = field.overrideID {
            override(id)
        } else {
            ContractFormControl(field: field, draft: $draft, labelStyle: labelStyle)
        }
    }
}

/// One contract option as the control its `kind` calls for.
struct ContractFormControl<Draft>: View {
    let field: StudioContractField<Draft>
    @Binding var draft: Draft
    var labelStyle: ContractFormLabelStyle = .label

    /// The flag column's width, so every row's control starts at the same x the way the board
    /// draws the raw form.
    static var flagColumnWidth: CGFloat { 168 }

    var body: some View {
        switch labelStyle {
        case .label: labelled
        case .flag: flagged
        }
    }

    /// The inspector's shape: the option's label, then its control.
    @ViewBuilder
    private var labelled: some View {
        switch field.control {
        case .toggle:
            Toggle(field.label, isOn: flagBinding)
                .toggleStyle(.checkbox)
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(MereRunTheme.textSecondary)
                .help(field.flag)
        case .segmented:
            StudioInspectorLabeledRow(field.label) {
                MereSegmentedControl(field.option.choices, selection: choiceBinding, accessibilityLabel: field.label) {
                    StudioContractChoiceTitles.title(for: $0, flag: field.flag)
                }
            }
            .help(field.flag)
        case .picker:
            StudioInspectorLabeledRow(field.label) {
                Picker(field.label, selection: choiceBinding) {
                    ForEach(field.option.choices, id: \.self) { choice in
                        Text(StudioContractChoiceTitles.title(for: choice, flag: field.flag)).tag(choice)
                    }
                }
                .labelsHidden()
                .frame(maxWidth: 150)
            }
            .help(field.flag)
        case .slider:
            slider
        case .stepper:
            stepper
        case .path:
            ContractFormPathRow(
                label: field.label,
                path: textBinding,
                isDirectory: field.kind == .directory,
                allowedTypes: Self.allowedTypes(for: field)
            )
        case .field:
            // Full width with the option's label as the placeholder, the way the boards draw a
            // free-text field: the label would only repeat what the empty field already says.
            StudioInspectorTextField(placeholder: field.label, text: textBinding, lines: 1...4)
                .help(field.flag)
        case .override:
            // The caller's builder draws this row; `ContractForm` never reaches here.
            EmptyView()
        }
    }

    /// The Command board's shape: the flag in a monospaced column, then its control. Numbers
    /// are typed rather than dragged here — this is the surface that shows what the argv says.
    @ViewBuilder
    private var flagged: some View {
        switch field.control {
        case .toggle:
            flagRow {
                Toggle("", isOn: flagBinding)
                    .toggleStyle(.switch)
                    .controlSize(.mini)
                    .labelsHidden()
                    .accessibilityLabel(field.label)
                Spacer(minLength: 0)
            }
        case .segmented:
            flagRow {
                MereSegmentedControl(field.option.choices, selection: choiceBinding, accessibilityLabel: field.label) {
                    StudioContractChoiceTitles.title(for: $0, flag: field.flag)
                }
            }
        case .picker:
            flagRow {
                Picker(field.label, selection: choiceBinding) {
                    ForEach(field.option.choices, id: \.self) { choice in
                        Text(StudioContractChoiceTitles.title(for: choice, flag: field.flag)).tag(choice)
                    }
                }
                .labelsHidden()
                .frame(maxWidth: 180)
                Spacer(minLength: 0)
            }
        case .slider, .stepper:
            // Typed, not dragged, and stored as the text the argv carries: the raw surface must
            // not round `0.0001` to what a slider would show, and an empty field means no flag.
            flagRow {
                StudioInspectorTextField(
                    placeholder: field.option.defaultValue ?? "",
                    text: textBinding,
                    isMonospaced: true
                )
            }
        case .path:
            flagRow {
                ContractFormPathRow(
                    label: field.label,
                    path: textBinding,
                    isDirectory: field.kind == .directory,
                    allowedTypes: Self.allowedTypes(for: field)
                )
            }
        case .field:
            flagRow {
                StudioInspectorTextField(placeholder: field.label, text: textBinding, lines: 1...4)
            }
        case .override:
            EmptyView()
        }
    }

    private func flagRow<Content: View>(@ViewBuilder content: () -> Content) -> some View {
        HStack(alignment: .center, spacing: 10) {
            Text(field.flag)
                .font(.system(size: 12, weight: .medium, design: .monospaced))
                .foregroundStyle(MereRunTheme.textSecondary)
                .lineLimit(1)
                .truncationMode(.middle)
                .frame(width: Self.flagColumnWidth, alignment: .leading)
                .help(field.option.label)
            content()
        }
        .frame(minHeight: 28)
    }

    // MARK: Numeric controls

    @ViewBuilder
    private var slider: some View {
        let range = field.option.range
        StudioInspectorSlider(
            label: field.label,
            value: numberBinding,
            range: (range?.min ?? 0)...(range?.max ?? 1),
            step: range?.step ?? (field.kind == .integer ? 1 : 0.01),
            format: format
        )
        .help(field.flag)
    }

    /// A range the contract leaves open at one or both ends: nudge it rather than pretend to
    /// know its span.
    private var stepper: some View {
        let lower: Double = field.option.range?.min ?? -.greatestFiniteMagnitude
        let upper: Double = field.option.range?.max ?? .greatestFiniteMagnitude
        let step: Double = field.option.range?.step ?? (field.kind == .integer ? 1 : 0.1)
        let value = numberBinding
        return StudioInspectorLabeledRow(field.label) {
            Stepper(value: value, in: lower...upper, step: step) {
                Text(format(value.wrappedValue))
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundStyle(MereRunTheme.textPrimary)
            }
            .accessibilityLabel(field.label)
            .accessibilityValue(format(value.wrappedValue))
        }
        .help(field.flag)
    }

    private func format(_ value: Double) -> String {
        field.kind == .integer ? String(Int(value.rounded())) : StudioComposerPresets.decimalText(value)
    }

    // MARK: Bindings

    private var flagBinding: Binding<Bool> {
        Binding(
            get: { field.value(in: draft).flag ?? false },
            set: { field.write(.flag($0), to: &draft) }
        )
    }

    private var textBinding: Binding<String> {
        Binding(
            get: { field.value(in: draft).text ?? "" },
            set: { field.write(.text($0), to: &draft) }
        )
    }

    private var choiceBinding: Binding<String> {
        Binding(
            get: {
                let value = field.value(in: draft).text ?? ""
                if field.option.choices.contains(value) { return value }
                return field.option.defaultValue ?? field.option.choices.first ?? value
            },
            set: { field.write(.text($0), to: &draft) }
        )
    }

    private var numberBinding: Binding<Double> {
        Binding(
            get: { field.value(in: draft).numericValue ?? field.defaultValue.numericValue ?? 0 },
            set: { field.write(field.kind == .integer ? .integer(Int($0.rounded())) : .number($0), to: &draft) }
        )
    }

    private static func allowedTypes(for field: StudioContractField<Draft>) -> [UTType] {
        switch field.flag {
        case "--audio", "--ref-audio", "--source-audio", "--reference-audio": return [.audio]
        case "--image", "--input", "--end-image", "--ref-image", "--mask": return [.image]
        case "--lrc-file", "--lyrics-file": return [.plainText]
        default: return [.data]
        }
    }
}

/// A file or directory option: the chosen name, a Choose button, and a clear button once set.
struct ContractFormPathRow: View {
    let label: String
    @Binding var path: String
    var isDirectory = false
    var allowedTypes: [UTType] = [.data]

    var body: some View {
        HStack(spacing: 8) {
            Text(path.isBlank ? "No \(label.lowercased())" : URL(fileURLWithPath: path).lastPathComponent)
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(path.isBlank ? MereRunTheme.textMuted : MereRunTheme.textPrimary)
                .lineLimit(1)
                .truncationMode(.middle)
                .help(path.isBlank ? label : path)
            Spacer(minLength: 4)
            Button(path.isBlank ? "Choose…" : "Change…") { choose() }
                .buttonStyle(.mereSecondary)
                .accessibilityLabel("Choose \(label.lowercased())")
            if !path.isBlank {
                Button {
                    path = ""
                } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 10, weight: .semibold))
                        .frame(width: 20, height: 20)
                }
                .buttonStyle(.mereIcon(tint: MereRunTheme.textMuted))
                .accessibilityLabel("Remove \(label.lowercased())")
            }
        }
        .frame(minHeight: 24)
        .accessibilityElement(children: .contain)
        .accessibilityLabel(label)
    }

    private func choose() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = isDirectory
        panel.canChooseFiles = !isDirectory
        if !isDirectory { panel.allowedContentTypes = allowedTypes }
        if panel.runModal() == .OK, let url = panel.url {
            path = url.path
        }
    }
}
