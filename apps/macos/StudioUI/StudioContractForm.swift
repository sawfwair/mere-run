import AppKit
import MereRunContract
import StudioKit
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
    /// The Command panel's shape: the label in the column with the flag under it in small
    /// monospace, over the raw controls of `.flag` — typed values, exactly what the argv carries —
    /// so the panel reads in plain words without hiding what it will run.
    case labelledFlag
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
    /// What the choice that leaves an option unset is called, for a choice option the contract
    /// gives no default: "Default" unless the caller says otherwise ("Custom" for a recipe, "From
    /// recipe" while one is chosen). A choice with a contract default has no such item: the
    /// default is what runs.
    var noneTitle: String?

    /// The flag column's width, so every row's control starts at the same x the way the board
    /// draws the raw form.
    static var flagColumnWidth: CGFloat { 168 }

    var body: some View {
        switch labelStyle {
        case .label: labelled
        case .flag, .labelledFlag: flagged
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
            StudioFittingChoice { shape in
                StudioInspectorLabeledRow(field.label) { choice(shape, menuWidth: 150) }
            }
            .help(field.flag)
        case .picker:
            StudioInspectorLabeledRow(field.label) { choice(.menu, menuWidth: 150) }
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
            StudioFittingChoice { shape in
                flagRow {
                    choice(shape, menuWidth: 180)
                    Spacer(minLength: 0)
                }
            }
        case .picker:
            flagRow {
                choice(.menu, menuWidth: 180)
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
            flagColumn
                .frame(width: Self.flagColumnWidth, alignment: .leading)
            content()
        }
        .frame(minHeight: 28)
    }

    /// The row's head: the flag alone for the console, or the label over the flag for the panel.
    @ViewBuilder
    private var flagColumn: some View {
        switch labelStyle {
        case .labelledFlag:
            VStack(alignment: .leading, spacing: 1) {
                Text(field.label)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(MereRunTheme.textSecondary)
                    .lineLimit(1)
                    .truncationMode(.tail)
                Text(field.flag)
                    .font(.system(size: 10.5, weight: .medium, design: .monospaced))
                    .foregroundStyle(MereRunTheme.textMuted)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            .help(field.flag)
            .accessibilityElement(children: .combine)
        case .label, .flag:
            Text(field.flag)
                .font(.system(size: 12, weight: .medium, design: .monospaced))
                .foregroundStyle(MereRunTheme.textSecondary)
                .lineLimit(1)
                .truncationMode(.middle)
                .help(field.option.label)
        }
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

    // MARK: Choices

    /// The item that leaves the option unset, when the contract declares no default: an empty
    /// choice ahead of the declared ones, so a fresh form never shows a selection the argv does
    /// not carry.
    private var unsetChoice: String? {
        guard field.option.defaultValue == nil, !field.option.choices.isEmpty else { return nil }
        return noneTitle ?? Self.noneTitle(for: field.flag)
    }

    private var choiceItems: [String] {
        unsetChoice == nil ? field.option.choices : [""] + field.option.choices
    }

    /// The option's choices, unset item first when it has one, as segments or a menu.
    private func choice(_ shape: StudioChoiceShape, menuWidth: CGFloat) -> some View {
        StudioChoiceControl(
            shape: shape,
            items: choiceItems,
            selection: choiceBinding,
            accessibilityLabel: field.label,
            menuWidth: menuWidth,
            title: choiceTitle
        )
    }

    private func choiceTitle(_ choice: String) -> String {
        choice.isEmpty ? (unsetChoice ?? "") : StudioContractChoiceTitles.title(for: choice, flag: field.flag)
    }

    /// "Custom" where the choices are named presets the user can do without; "Default" elsewhere.
    static func noneTitle(for flag: String) -> String {
        flag == "--recipe" ? "Custom" : "Default"
    }

    private var choiceBinding: Binding<String> {
        Binding(
            get: {
                let value = field.value(in: draft).text ?? ""
                if field.option.choices.contains(value) { return value }
                if unsetChoice != nil { return "" }
                return field.option.defaultValue ?? field.option.choices.first ?? value
            },
            set: { choice in
                field.write(choice.isEmpty && unsetChoice != nil ? .unset : .text(choice), to: &draft)
            }
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

/// The two ways a choice is drawn.
enum StudioChoiceShape {
    /// Every choice as a segment, read at a glance.
    case segments
    /// A pop-up menu, for choices that would not read whole as segments.
    case menu
}

/// Draws a choice's row with segments when the whole row — its label or flag column and every
/// segment at full width — fits the width the row is given, and with a menu when it does not.
/// The rule is the row's real width rather than a count of characters, so "Float16 · Float32"
/// stays segments in the 280 pt inspector while "Default · Small · Medium · Large" beside
/// "Variant" becomes a menu instead of truncating. The inspector's choice rows, its variant
/// row, and the Command view's flag rows all decide this way.
struct StudioFittingChoice<Row: View>: View {
    @ViewBuilder let row: (StudioChoiceShape) -> Row

    var body: some View {
        ViewThatFits(in: .horizontal) {
            row(.segments)
            row(.menu)
        }
    }
}

/// A choice as `MereSegmentedControl` or as a menu no wider than `menuWidth`.
struct StudioChoiceControl<Item: Hashable>: View {
    let shape: StudioChoiceShape
    let items: [Item]
    @Binding var selection: Item
    let accessibilityLabel: String
    let menuWidth: CGFloat
    let title: (Item) -> String

    var body: some View {
        switch shape {
        case .segments:
            MereSegmentedControl(items, selection: $selection, accessibilityLabel: accessibilityLabel, title: title)
        case .menu:
            Picker(accessibilityLabel, selection: $selection) {
                ForEach(items, id: \.self) { item in
                    Text(title(item)).tag(item)
                }
            }
            .labelsHidden()
            .frame(maxWidth: menuWidth)
        }
    }
}

/// A file or directory option: the chosen name, a Choose button, and a clear button once set.
struct ContractFormPathRow: View {
    let label: String
    @Binding var path: String
    var isDirectory = false
    var allowedTypes: [UTType] = [.data]
    var allowsMultipleSelection = false
    /// Project and Manage pages also let a user paste a path directly.
    var placeholder: String? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            if let placeholder {
                Text(label)
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                pathControls(placeholder: placeholder)
            } else {
                pathControls(placeholder: nil)
            }
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel(label)
    }

    private func pathControls(placeholder: String?) -> some View {
        HStack(spacing: 8) {
            if let placeholder {
                TextField(placeholder, text: $path)
                    .mereField()
            } else {
                Text(path.isBlank ? "No \(label.lowercased())" : URL(fileURLWithPath: path).lastPathComponent)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(path.isBlank ? MereRunTheme.textMuted : MereRunTheme.textPrimary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .help(path.isBlank ? label : path)
                Spacer(minLength: 4)
            }
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
    }

    private func choose() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = allowsMultipleSelection
        panel.canChooseDirectories = isDirectory
        panel.canChooseFiles = !isDirectory
        if !isDirectory { panel.allowedContentTypes = allowedTypes }
        if panel.runModal() == .OK {
            path = panel.urls.map(\.path).joined(separator: "\n")
        }
    }
}
