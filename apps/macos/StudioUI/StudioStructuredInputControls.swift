import StudioKit
import SwiftUI

// Controls over the small CLI value syntaxes `StudioStructuredInputs` owns. Each binds to the draft's
// string so the Command view and the argv fixtures see exactly what the page sends.

// MARK: - Instruments

/// Music ▸ Transcribe's expected instruments: chips picked from the list the CLI prints with
/// `--list-instruments`, searchable; a plain field when the list cannot be read.
struct StudioInstrumentPicker: View {
    @EnvironmentObject private var controller: MereRunController
    /// The `--instruments` value: comma-separated names, blank for automatic.
    @Binding var value: String
    @State private var names: [String]?
    @State private var loadFailed = false
    @State private var isPicking = false
    @State private var search = ""

    private var selected: [String] { StudioInstrumentList.decode(value) }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                Text("Expected instruments")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                Spacer()
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
                TextField("voice, drums, electric_bass — blank means automatic", text: $value)
                    .mereField()
                Text("The instrument list could not be read from the CLI; type group names separated by commas.")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                    .fixedSize(horizontal: false, vertical: true)
            } else if selected.isEmpty {
                Text(names == nil ? "Reading the instrument list…" : "Automatic — the model decides which instruments it hears.")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            } else {
                FlowLayout(spacing: 6) {
                    ForEach(selected, id: \.self) { name in
                        chip(name)
                    }
                }
            }
        }
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

    /// Reads the CLI's list once per launch, kept on the controller so rebuilding the page does not
    /// spawn the CLI again; a failure leaves the plain field.
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

/// Sound ▸ Generate's Woosh renoise: the model's default, one amount, or one amount per step. The
/// page keeps `mode` beside the draft, so "Per step" survives an empty field and a rebuilt view.
struct StudioRenoiseControl: View {
    /// The `--renoise` value; blank for the model's default.
    @Binding var value: String
    @Binding var mode: StudioRenoise.Mode
    let steps: Int

    private var renoise: StudioRenoise { StudioRenoise(mode: mode, argument: value) }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Renoise")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            Picker("Renoise", selection: $mode) {
                ForEach(StudioRenoise.Mode.allCases) { Text($0.title).tag($0) }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
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
                HStack {
                    Slider(value: amount, in: 0...1, step: 0.01)
                    Text(renoise.amountValue.map { $0.formatted(.number.precision(.fractionLength(2))) } ?? value)
                        .font(MereRunTheme.captionFont)
                        .monospacedDigit()
                        .lineLimit(1)
                        .frame(minWidth: 34, alignment: .trailing)
                }
                .accessibilityLabel("Renoise amount")
            case .schedule:
                TextField("One amount per step, 0 to 1: 0.3, 0.3, 0.2, …", text: $value)
                    .mereField()
                    .accessibilityLabel("Renoise schedule")
            }
            ForEach(renoise.problems(steps: steps), id: \.self) { problem in
                Label(problem, systemImage: "exclamationmark.circle")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textSecondary)
            }
        }
        .onAppear {
            // The draft's argument may have come from elsewhere (a Library rerun, a draft written
            // before the page kept a mode): show the mode it reads as, keeping its value.
            let resolved = StudioRenoise.resolvedMode(stored: mode, argument: value)
            if resolved != mode { mode = resolved }
        }
        .onChange(of: mode) { _, mode in
            let kept = StudioRenoise.argument(switching: value, to: mode)
            if kept != value { value = kept }
        }
    }
}
