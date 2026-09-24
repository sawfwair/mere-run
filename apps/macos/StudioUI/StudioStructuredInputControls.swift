import StudioKit
import SwiftUI

// Controls over the small CLI value syntaxes `StudioStructuredInputs` owns. Each binds to the draft's
// string so the Command view and the argv fixtures see exactly what the page sends.

// MARK: - Instruments

/// Music ▸ Transcribe's expected instruments, the inspector's `.instruments` editor: chips
/// picked from the list the CLI prints with `--list-instruments`, searchable; a plain field when
/// the list cannot be read. It binds the task draft's `--instruments` value directly, so the
/// Command view and the argv carry exactly what the chips say.
struct StudioInstrumentPicker: View {
    @EnvironmentObject private var controller: MereRunController
    /// The `--instruments` value: comma-separated names, blank for automatic.
    @Binding var value: String
    @State private var names: [String]?
    @State private var loadFailed = false
    @State private var isPicking = false
    @State private var search = ""

    static let label = "Expected instruments"

    init(value: Binding<String>) {
        _value = value
    }

    init(draft: Binding<StudioTaskDraft>) {
        _value = Binding(
            get: { draft.wrappedValue.text("--instruments") },
            set: { draft.wrappedValue.form["--instruments"] = $0.isEmpty ? .unset : .text($0) }
        )
    }

    private var selected: [String] { StudioInstrumentList.decode(value) }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text(Self.label)
                    .font(.callout.weight(.medium))
                    .foregroundStyle(MereRunTheme.textSecondary)
                Spacer(minLength: 0)
                if let names {
                    Button {
                        search = ""
                        isPicking = true
                    } label: {
                        Label("Add instrument", systemImage: "plus")
                            .font(.caption.weight(.semibold))
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(MereRunTheme.accent)
                    .popover(isPresented: $isPicking, arrowEdge: .bottom) {
                        picker(names)
                    }
                }
            }
            if loadFailed {
                StudioInspectorTextField(placeholder: "voice, drums, electric_bass", text: $value)
                    .accessibilityLabel(Self.label)
                Text("The instrument list could not be read from the CLI; type group names separated by commas, or leave it blank for automatic.")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                    .fixedSize(horizontal: false, vertical: true)
            } else if selected.isEmpty {
                Text(names == nil ? "Reading the instrument list…" : "Automatic — the model decides which instruments it hears.")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                FlowLayout(spacing: 6) {
                    ForEach(selected, id: \.self) { name in
                        chip(name)
                    }
                }
            }
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel(Self.label)
        .accessibilityValue(selected.isEmpty ? "Automatic" : selected.map(StudioInstrumentList.displayName).joined(separator: ", "))
        .task { await load() }
    }

    private func chip(_ name: String) -> some View {
        HStack(spacing: 4) {
            Text(StudioInstrumentList.displayName(name))
                .font(MereRunTheme.captionFont)
            Button {
                remove(name)
            } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 8, weight: .bold))
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Remove \(StudioInstrumentList.displayName(name))")
        }
        .padding(.horizontal, 9)
        .padding(.vertical, 5)
        .background(MereRunTheme.accentSoft)
        .clipShape(Capsule())
    }

    private func picker(_ names: [String]) -> some View {
        let matches = names.filter {
            search.isBlank || StudioInstrumentList.displayName($0).localizedCaseInsensitiveContains(search)
        }
        return VStack(alignment: .leading, spacing: 8) {
            TextField("Search instruments", text: $search)
                .textFieldStyle(.roundedBorder)
            ScrollView {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(matches, id: \.self) { name in
                        let isSelected = selected.contains(name)
                        Button {
                            if isSelected { remove(name) } else { value = StudioInstrumentList.encode(selected + [name]) }
                        } label: {
                            HStack {
                                Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                                    .foregroundStyle(isSelected ? MereRunTheme.accent : MereRunTheme.textMuted)
                                Text(StudioInstrumentList.displayName(name))
                                Spacer()
                            }
                            .padding(.vertical, 3)
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                    }
                    if matches.isEmpty {
                        Text("No instrument matches \"\(search)\".")
                            .font(MereRunTheme.captionFont)
                            .foregroundStyle(MereRunTheme.textMuted)
                    }
                }
            }
            .frame(minHeight: 120, maxHeight: 260)
        }
        .padding(12)
        .frame(width: 260)
    }

    private func remove(_ name: String) {
        value = StudioInstrumentList.encode(selected.filter { $0 != name })
    }

    /// Reads the CLI's list once per launch, kept on the controller so rebuilding the inspector
    /// does not spawn the CLI again; a failure leaves the plain field.
    private func load() async {
        guard names == nil else { return }
        if let cached = controller.cachedInstrumentNames {
            names = cached
            return
        }
        let result = await controller.utilityCommandResult(args: StudioInstrumentList.listArguments)
        let parsed = StudioInstrumentList.parse(result.stdout)
        if result.exitCode == 0, !parsed.isEmpty {
            controller.cachedInstrumentNames = parsed
            names = parsed
        } else {
            loadFailed = true
        }
    }
}

// MARK: - Per-target LoRA ranks

/// Image ▸ Train's Klein per-target ranks: one row per module suffix, written as
/// `--lora-target-ranks suffix=rank,…`.
struct StudioTargetRankEditor: View {
    /// The `--lora-target-ranks` value; blank when there are no rows.
    @Binding var value: String
    @State private var ranks: [StudioTargetRank] = []

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                Text("Per-target ranks")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                Spacer()
                Button {
                    ranks.append(StudioTargetRank(suffix: "", rank: 64))
                } label: {
                    Label("Add target", systemImage: "plus")
                        .font(.caption.weight(.semibold))
                }
                .buttonStyle(.plain)
                .foregroundStyle(MereRunTheme.accent)
            }
            if ranks.isEmpty {
                Text("Every target uses the rank above. Add a module suffix to give it its own rank.")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                    .fixedSize(horizontal: false, vertical: true)
            }
            ForEach($ranks) { $entry in
                let index = ranks.firstIndex { $0.id == entry.id } ?? 0
                HStack(spacing: 6) {
                    TextField(".attn.to_q", text: $entry.suffix)
                        .textFieldStyle(.plain)
                        .font(.system(size: 11.5, design: .monospaced))
                        .padding(6)
                        .merePanel()
                        .accessibilityLabel("Target \(index + 1) module suffix")
                    Stepper(value: $entry.rank, in: 1...1_024) {
                        Text("rank \(entry.rank)")
                            .font(MereRunTheme.captionFont)
                            .monospacedDigit()
                            .frame(minWidth: 62, alignment: .leading)
                    }
                    .accessibilityLabel("Target \(index + 1) rank")
                    Button {
                        ranks.removeAll { $0.id == entry.id }
                    } label: {
                        Image(systemName: "minus.circle")
                            .frame(width: 22, height: 22)
                    }
                    .buttonStyle(.mereIcon)
                    .help("Remove")
                    .accessibilityLabel("Remove target \(index + 1)")
                }
            }
            ForEach(StudioTargetRank.problems(ranks), id: \.self) { problem in
                Label(problem, systemImage: "exclamationmark.circle")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textSecondary)
            }
        }
        .onAppear { ranks = StudioTargetRank.decode(value) }
        .onChange(of: ranks) { _, ranks in
            let encoded = StudioTargetRank.encode(ranks)
            if encoded != value { value = encoded }
        }
    }
}

// MARK: - Renoise

/// Woosh renoise on the task inspector (Sound ▸ Video Foley), the `.renoise` override: the
/// model's default, one amount, or one amount per step, written to `--renoise` exactly as the
/// CLI reads it. The mode is kept beside the task draft (`"<task>.renoiseMode"`) so "Per step"
/// survives an empty field and a rebuilt inspector; the problems the CLI would raise are shown
/// under the control, and the runner refuses the run while one stands.
struct StudioRenoiseOverride: View {
    @Binding var draft: StudioTaskDraft
    @StudioStoredValue("renoiseMode") private var mode = StudioRenoise.Mode.automatic

    private var value: Binding<String> {
        Binding(
            get: { draft.text("--renoise") },
            set: { draft.form["--renoise"] = $0.isEmpty ? .unset : .text($0) }
        )
    }

    var body: some View {
        StudioRenoiseControl(
            value: value, mode: $mode,
            steps: StudioRenoise.stepCount(in: draft.form, templateID: draft.templateID)
        )
    }
}

/// The renoise control itself: a mode segment, then the amount slider or the schedule field,
/// then whatever the CLI would object to. Drawn in the inspector's own controls.
struct StudioRenoiseControl: View {
    /// The `--renoise` value; blank for the model's default.
    @Binding var value: String
    @Binding var mode: StudioRenoise.Mode
    let steps: Int

    private var renoise: StudioRenoise { StudioRenoise(mode: mode, argument: value) }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Renoise")
                .font(.callout.weight(.medium))
                .foregroundStyle(MereRunTheme.textSecondary)
            MereSegmentedControl(StudioRenoise.Mode.allCases, selection: $mode, accessibilityLabel: "Renoise") {
                Self.segmentTitle($0)
            }
            switch mode {
            case .automatic:
                Text("The model's own renoise setting.")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            case .amount:
                let amount = Binding<Double>(
                    get: { renoise.amountValue ?? StudioRenoise.defaultAmount },
                    set: { value = StudioRenoise.amount(($0 * 100).rounded() / 100).argument }
                )
                StudioInspectorSlider(label: "Amount", value: amount, range: 0...1, step: 0.01) { _ in
                    // Text that is not a number stays visible until it is fixed, never replaced.
                    renoise.amountValue.map(StudioComposerPresets.decimalText) ?? value
                }
            case .schedule:
                StudioInspectorTextField(
                    placeholder: "One amount per step, 0 to 1: 0.3, 0.3, 0.2, …",
                    text: $value,
                    isMonospaced: true
                )
                .accessibilityLabel("Renoise schedule")
            }
            ForEach(renoise.problems(steps: steps), id: \.self) { problem in
                Label(problem, systemImage: "exclamationmark.circle")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .onAppear {
            // The draft's argument may have come from elsewhere (a Library rerun, a page draft
            // imported once): show the mode it reads as, keeping its value.
            let resolved = StudioRenoise.resolvedMode(stored: mode, argument: value)
            if resolved != mode { mode = resolved }
        }
        .onChange(of: mode) { _, mode in
            let kept = StudioRenoise.argument(switching: value, to: mode)
            if kept != value { value = kept }
        }
    }

    /// The segment's word, short enough for three to share the inspector's width.
    static func segmentTitle(_ mode: StudioRenoise.Mode) -> String {
        switch mode {
        case .automatic: return "Auto"
        case .amount: return "Amount"
        case .schedule: return "Per step"
        }
    }
}
