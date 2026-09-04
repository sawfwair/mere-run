import AppKit
import MereRunContract
import SwiftUI
import UniformTypeIdentifiers

/// The inspector column: the current prompt task's standard parameters in sections (Prompt,
/// Output, Model & adapters, Sampling), each with Reset, and everything else the command takes
/// under a collapsed Advanced row. It binds the same `StudioDraft` the composer's chips edit,
/// so a change in either shows in both, and the header counts what differs from the mode's
/// defaults.
struct StudioInspector: View {
    let mode: StudioMode
    @Binding var draft: StudioDraft
    /// The draft the mode starts with; Reset and the modified count read against it.
    let baseline: StudioDraft
    let modelInventory: [StudioModelInventoryRow]
    let readiness: ModelReadinessState
    let lastSeed: String?
    let onShowModels: () -> Void
    let onShowAdapters: () -> Void
    let onShowRealtimeMusic: () -> Void
    let onClose: () -> Void

    @State private var showAdvanced = false
    @State private var editingSeed = false

    static let width = StudioLayoutPolicy.inspectorWidth

    private var sections: [StudioInspectorSection] {
        StudioInspectorSchema.sections(for: mode)
    }

    private var advancedCount: Int {
        StudioInspectorSchema.advancedFields(for: mode).count
    }

    private var changedCount: Int {
        StudioInspectorSchema.changedCount(mode: mode, draft: draft, baseline: baseline)
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            ScrollView {
                VStack(spacing: 0) {
                    ForEach(sections) { section in
                        sectionView(section)
                    }
                    if advancedCount > 0 {
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
            Text("\(mode.destination.domain.title) · \(mode.task.title)")
                .font(.system(size: 12.5, weight: .semibold))
                .foregroundStyle(MereRunTheme.textPrimary)
                .lineLimit(1)
            if changedCount > 0 {
                Text("\(changedCount) changed")
                    .font(.system(size: 10.5, weight: .semibold))
                    .foregroundStyle(MereRunTheme.accent)
                    .padding(.horizontal, 6)
                    .frame(height: 18)
                    .background { Capsule().fill(MereRunTheme.accentSoft) }
                    .accessibilityLabel("\(changedCount) settings changed from the defaults")
            }
            Spacer(minLength: 0)
            Button(action: onClose) {
                Image(systemName: "xmark")
                    .font(.system(size: 11, weight: .semibold))
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

    private func sectionView(_ section: StudioInspectorSection) -> some View {
        StudioInspectorSectionView(
            title: section.kind.title,
            canReset: section.changedCount(draft: draft, baseline: baseline) > 0,
            onReset: { section.reset(&draft, to: baseline) }
        ) {
            switch section.kind {
            case .prompt: promptSection
            case .output: outputSection
            case .model: modelSection
            case .sampling: samplingSection
            case .transcript: transcriptSection
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
                        Text("Advanced · \(advancedCount) more")
                            .font(.system(size: 12, weight: .medium))
                            .foregroundStyle(MereRunTheme.textSecondary)
                    }
                }
                .buttonStyle(.plain)
                .accessibilityLabel(showAdvanced ? "Hide advanced settings" : "Show \(advancedCount) advanced settings")
                Spacer(minLength: 0)
                if showAdvanced, advancedChanged {
                    resetButton { StudioInspectorSchema.resetAdvanced(for: mode, &draft, to: baseline) }
                }
            }
            if showAdvanced {
                StudioAdvancedOptions(
                    mode: mode,
                    draft: $draft,
                    onShowAdapters: onShowAdapters,
                    onShowRealtimeMusic: onShowRealtimeMusic
                )
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 14)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var advancedChanged: Bool {
        StudioInspectorSchema.advancedFields(for: mode).contains { $0.isChanged(draft, baseline) }
    }

    private func resetButton(_ action: @escaping () -> Void) -> some View {
        Button("Reset", action: action)
            .buttonStyle(.plain)
            .font(.system(size: 11, weight: .medium))
            .foregroundStyle(MereRunTheme.textMuted)
            .help("Back to the defaults")
    }

    // MARK: Prompt

    @ViewBuilder
    private var promptSection: some View {
        switch mode {
        case .readImage:
            MereSegmentedControl(
                StudioReadImageAction.allCases,
                selection: $draft.readImageAction,
                accessibilityLabel: "Read task"
            ) { $0.title }
        default:
            if let label = secondaryLabel {
                StudioInspectorTextField(placeholder: label, text: $draft.secondaryText, lines: 1...4)
            }
        }
    }

    private var secondaryLabel: String? {
        switch mode {
        case .createImage, .video: return "Negative prompt"
        case .chat, .code: return "System prompt"
        case .speak: return "Voice style"
        case .music: return "Lyrics"
        default: return nil
        }
    }

    // MARK: Output

    @ViewBuilder
    private var outputSection: some View {
        switch mode {
        case .createImage, .video:
            aspectControls
            if mode == .video { videoLengthControls }
        case .music:
            Toggle("Set exact length", isOn: $draft.useDuration)
                .toggleStyle(.checkbox)
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(MereRunTheme.textSecondary)
            if draft.useDuration {
                StudioInspectorSlider(label: "Length", value: $draft.durationSeconds, range: 5...300, step: 1, format: { "\(StudioComposerPresets.secondsText($0)) s" })
            }
            StudioInspectorLabeledRow("Quality") {
                MereSegmentedControl(["draft", "song", "final", "edit"], selection: $draft.musicQuality, accessibilityLabel: "Quality") {
                    $0.capitalized
                }
            }
        case .sfx:
            StudioInspectorSlider(label: "Length", value: $draft.durationSeconds, range: 0.5...30, step: 0.5, format: { "\(StudioComposerPresets.secondsText($0)) s" })
        default:
            EmptyView()
        }
    }

    private var aspectControls: some View {
        VStack(alignment: .leading, spacing: 10) {
            StudioAspectSegmentedControl(mode: mode, draft: $draft)
            HStack(spacing: 8) {
                StudioInspectorNumberField(label: "Width", value: $draft.width, range: 64...4096)
                Text("×")
                    .font(.system(size: 12))
                    .foregroundStyle(MereRunTheme.textMuted)
                StudioInspectorNumberField(label: "Height", value: $draft.height, range: 64...4096)
                Button("Swap") { StudioAspectPreset.swap(&draft) }
                    .buttonStyle(.mereSecondary)
                    .help("Swap width and height")
            }
        }
    }

    private var videoLengthControls: some View {
        VStack(alignment: .leading, spacing: 8) {
            StudioInspectorLabeledRow("Length") {
                MereSegmentedControl([false, true], selection: $draft.useDuration, accessibilityLabel: "Length unit") {
                    $0 ? "Seconds" : "Frames"
                }
            }
            if draft.useDuration {
                StudioInspectorSlider(label: "Seconds", value: $draft.durationSeconds, range: 1...30, step: 0.5, format: { StudioComposerPresets.secondsText($0) })
            } else {
                StudioInspectorSlider(
                    label: "Frames",
                    value: Binding(get: { Double(draft.numFrames) }, set: { draft.numFrames = Int($0) }),
                    range: 9...321, step: 8, format: { String(Int($0)) }
                )
            }
        }
    }

    // MARK: Model & adapters

    @ViewBuilder
    private var modelSection: some View {
        StudioModelPicker(mode: mode, model: $draft.model, modelInventory: modelInventory, onShowModels: onShowModels) {
            HStack(spacing: 8) {
                if let glyph = modelStatusGlyph {
                    Image(systemName: glyph)
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(MereRunTheme.accent)
                }
                Text(StudioModelNaming.displayLabel(for: mode, model: draft.model))
                    .font(.system(size: 13))
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
        .help(readiness.blocksRun ? readiness.message : "Model")
        .accessibilityLabel("Model")
        .accessibilityValue(StudioModelNaming.resolvedModelID(for: mode, model: draft.model))

        switch mode {
        case .createImage, .chat:
            loraRow
        case .music:
            musicAdaptersRow
        case .speak:
            MereSegmentedControl(["style", "clone"], selection: $draft.voiceMode, accessibilityLabel: "Voice") {
                $0 == "clone" ? "Cloned voice" : "Preset voice"
            }
        default:
            EmptyView()
        }
    }

    private var modelStatusGlyph: String? {
        switch readiness {
        case .missingModel: return "arrow.down.circle"
        case .unsupported: return "exclamationmark.triangle"
        case .checking, .ready, .unknown: return nil
        }
    }

    @ViewBuilder
    private var loraRow: some View {
        if draft.loraPath.isBlank {
            Menu {
                Button("Choose a file…") { chooseLoRAFile() }
                Button("From the catalog…", action: onShowAdapters)
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: "plus")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(MereRunTheme.textMuted)
                    Text("Add LoRA adapter")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(MereRunTheme.textSecondary)
                }
            }
            .menuStyle(.button)
            .buttonStyle(.plain)
            .menuIndicator(.hidden)
            .accessibilityLabel("Add LoRA adapter")
        } else {
            HStack(spacing: 8) {
                Image(systemName: "square.stack.3d.up")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(MereRunTheme.accent)
                Text(URL(fileURLWithPath: draft.loraPath).lastPathComponent)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(MereRunTheme.textPrimary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .help(draft.loraPath)
                Spacer(minLength: 4)
                TextField("Scale", value: $draft.loraScale, format: .number)
                    .textFieldStyle(.plain)
                    .font(.system(size: 12, design: .monospaced))
                    .multilineTextAlignment(.trailing)
                    .frame(width: 44)
                    .accessibilityLabel("LoRA scale")
                Button {
                    draft.loraPath = ""
                } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 10, weight: .semibold))
                        .frame(width: 20, height: 20)
                }
                .buttonStyle(.mereIcon(tint: MereRunTheme.textMuted))
                .accessibilityLabel("Remove LoRA adapter")
            }
        }
    }

    @ViewBuilder
    private var musicAdaptersRow: some View {
        let adapters = StudioAttachmentSlot.separatedPaths(draft.musicAdapterPaths)
        ForEach(Array(adapters.enumerated()), id: \.offset) { _, path in
            HStack(spacing: 8) {
                Image(systemName: "square.stack.3d.up")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(MereRunTheme.accent)
                Text(URL(fileURLWithPath: path).lastPathComponent)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(MereRunTheme.textPrimary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .help(path)
                Spacer(minLength: 4)
                Button {
                    draft.musicAdapterPaths = adapters.filter { $0 != path }.joined(separator: "\n")
                } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 10, weight: .semibold))
                        .frame(width: 20, height: 20)
                }
                .buttonStyle(.mereIcon(tint: MereRunTheme.textMuted))
                .accessibilityLabel("Remove adapter")
            }
        }
        Menu {
            Button("Choose files…") { chooseMusicAdapters() }
            Button("From the catalog…", action: onShowAdapters)
        } label: {
            HStack(spacing: 8) {
                Image(systemName: "plus")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(MereRunTheme.textMuted)
                Text("Add ACE-Step adapter")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(MereRunTheme.textSecondary)
            }
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .accessibilityLabel("Add ACE-Step adapter")
    }

    // MARK: Sampling

    @ViewBuilder
    private var samplingSection: some View {
        switch mode {
        case .createImage:
            stepsSlider(range: 1...30)
            StudioInspectorSlider(label: "Guidance", value: $draft.cfgScale, range: 0...8, step: 0.1, format: { StudioComposerPresets.decimalText($0) })
            seedRow
        case .video:
            stepsSlider(range: 1...60)
            StudioInspectorSlider(label: "Guidance", value: $draft.cfgScale, range: 0...8, step: 0.1, format: { StudioComposerPresets.decimalText($0) })
            seedRow
        case .music:
            Toggle("Override preset steps", isOn: $draft.musicOverrideSteps)
                .toggleStyle(.checkbox)
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(MereRunTheme.textSecondary)
            if draft.musicOverrideSteps { stepsSlider(range: 1...120) }
            seedRow
        case .sfx:
            stepsSlider(range: 1...64)
            seedRow
        case .chat, .code:
            StudioInspectorSlider(label: "Temperature", value: $draft.temperature, range: 0...2, step: 0.05, format: { StudioComposerPresets.decimalText($0) })
            StudioInspectorSlider(label: "Top-p", value: $draft.topP, range: 0...1, step: 0.01, format: { StudioComposerPresets.decimalText($0) })
            StudioInspectorLabeledRow("Max tokens") {
                StudioInspectorNumberField(label: "Max tokens", value: $draft.maxTokens, range: 1...32_768)
                    .frame(width: 88)
            }
            if mode == .chat {
                StudioInspectorLabeledRow("Thinking") {
                    MereSegmentedControl(TextThinkingMode.allCases, selection: $draft.thinkingMode, accessibilityLabel: "Thinking") {
                        switch $0 {
                        case .automatic: return "Auto"
                        case .show: return "Show"
                        case .hide: return "Off"
                        }
                    }
                }
                StudioInspectorLabeledRow("Response") {
                    MereSegmentedControl([TextResponseFormat.text, .jsonObject], selection: $draft.responseFormat, accessibilityLabel: "Response format") {
                        $0 == .text ? "Text" : "JSON"
                    }
                }
            }
        case .findObjects, .segment, .track:
            StudioInspectorSlider(
                label: "Threshold",
                value: Binding(get: { draft.visionThreshold }, set: { draft.visionThreshold = min(max($0, 0), 1) }),
                range: 0...1, step: 0.01, format: { StudioComposerPresets.decimalText($0) }
            )
        default:
            EmptyView()
        }
    }

    private func stepsSlider(range: ClosedRange<Double>) -> some View {
        StudioInspectorSlider(
            label: "Steps",
            value: Binding(get: { Double(draft.steps) }, set: { draft.steps = Int($0.rounded()) }),
            range: range, step: 1, format: { String(Int($0)) }
        )
    }

    private var seedRow: some View {
        let seedMode = StudioSeedMode(draft: draft)
        return HStack(spacing: 8) {
            Text("Seed")
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(MereRunTheme.textSecondary)
            Spacer(minLength: 0)
            if editingSeed {
                TextField("Seed", text: $draft.seed)
                    .textFieldStyle(.plain)
                    .font(.system(size: 12, design: .monospaced))
                    .padding(.horizontal, 8)
                    .frame(width: 96, height: 24)
                    .background { StudioInspectorFieldChrome(cornerRadius: MereRunTheme.Radius.sm) }
                    .onSubmit { editingSeed = false }
                    .accessibilityLabel("Seed")
            } else {
                Menu {
                    Toggle(isOn: Binding(get: { seedMode == .random }, set: { _ in StudioSeedMode.random.apply(to: &draft) })) {
                        Text("Random")
                    }
                    Button("Enter a seed…") { editingSeed = true }
                } label: {
                    StudioComposerChipLabel(title: seedMode == .random ? "Random" : draft.seed.trimmingCharacters(in: .whitespaces))
                }
                .menuStyle(.button)
                .buttonStyle(.plain)
                .menuIndicator(.hidden)
                .fixedSize()
                .accessibilityLabel("Seed")
                .accessibilityValue(seedMode.chipTitle)
            }
            Button("Reuse last") {
                if let lastSeed { draft.seed = lastSeed }
                editingSeed = false
            }
            .buttonStyle(.mereSecondary)
            .disabled(lastSeed == nil)
            .help(lastSeed.map { "Use the last run's seed (\($0))" } ?? "No earlier run to take a seed from")
        }
    }

    // MARK: Transcript

    private var transcriptSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            StudioInspectorLabeledRow("Language") {
                TextField("auto", text: $draft.language)
                    .textFieldStyle(.plain)
                    .font(.system(size: 12, design: .monospaced))
                    .padding(.horizontal, 8)
                    .frame(width: 96, height: 24)
                    .background { StudioInspectorFieldChrome(cornerRadius: MereRunTheme.Radius.sm) }
                    .accessibilityLabel("Language")
            }
            Toggle("Timestamps", isOn: $draft.timestamps)
                .toggleStyle(.checkbox)
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(MereRunTheme.textSecondary)
        }
    }

    // MARK: Pickers

    private func chooseLoRAFile() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.data]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        if panel.runModal() == .OK, let url = panel.url {
            draft.loraPath = url.path
        }
    }

    private func chooseMusicAdapters() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.data]
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        if panel.runModal() == .OK {
            let existing = StudioAttachmentSlot.separatedPaths(draft.musicAdapterPaths)
            let added = panel.urls.map(\.path).filter { !existing.contains($0) }
            draft.musicAdapterPaths = (existing + added).joined(separator: "\n")
        }
    }
}

// MARK: - Section chrome

/// Title (12 semibold) with Reset (11 muted) on the right, 14/16 padding, hairline below.
struct StudioInspectorSectionView<Content: View>: View {
    let title: String
    let canReset: Bool
    let onReset: () -> Void
    @ViewBuilder let content: () -> Content

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(title)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(MereRunTheme.textPrimary)
                    .accessibilityAddTraits(.isHeader)
                Spacer(minLength: 0)
                Button("Reset", action: onReset)
                    .buttonStyle(.plain)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(MereRunTheme.textMuted)
                    .opacity(canReset ? 1 : 0.55)
                    .disabled(!canReset)
                    .help(canReset ? "Back to the defaults" : "Already at the defaults")
                    .accessibilityLabel("Reset \(title)")
            }
            content()
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .overlay(alignment: .bottom) {
            Rectangle().fill(MereRunTheme.border.opacity(0.4)).frame(height: 1)
        }
    }
}

/// The board's `field`: `surface` on a 1pt `border` at 80%, radius 8, 32pt tall.
struct StudioInspectorFieldChrome: View {
    var cornerRadius: CGFloat = MereRunTheme.Radius.base

    var body: some View {
        RoundedRectangle(cornerRadius: cornerRadius)
            .fill(MereRunTheme.surface)
            .overlay {
                RoundedRectangle(cornerRadius: cornerRadius)
                    .strokeBorder(MereRunTheme.border.opacity(0.8), lineWidth: 1)
            }
    }
}

struct StudioInspectorTextField: View {
    let placeholder: String
    @Binding var text: String
    var lines: ClosedRange<Int> = 1...1

    var body: some View {
        TextField(
            "",
            text: $text,
            prompt: Text(placeholder).foregroundStyle(MereRunTheme.textMuted),
            axis: lines.upperBound > 1 ? .vertical : .horizontal
        )
        .textFieldStyle(.plain)
        .font(.system(size: 13))
        .foregroundStyle(MereRunTheme.textPrimary)
        .lineLimit(lines)
        .padding(.horizontal, 10)
        .padding(.vertical, 7)
        .frame(maxWidth: .infinity, minHeight: 32, alignment: .leading)
        .background { StudioInspectorFieldChrome() }
        .accessibilityLabel(placeholder)
    }
}

/// A mono number field in field chrome; the value clamps to `range`.
struct StudioInspectorNumberField: View {
    let label: String
    @Binding var value: Int
    let range: ClosedRange<Int>

    var body: some View {
        TextField(
            label,
            value: Binding(
                get: { value },
                set: { value = min(max($0, range.lowerBound), range.upperBound) }
            ),
            format: .number.grouping(.never)
        )
        .textFieldStyle(.plain)
        .font(.system(size: 13, design: .monospaced))
        .foregroundStyle(MereRunTheme.textPrimary)
        .padding(.horizontal, 10)
        .frame(maxWidth: .infinity, minHeight: 32, alignment: .leading)
        .background { StudioInspectorFieldChrome() }
        .accessibilityLabel(label)
    }
}

/// A 12pt secondary label with a control on the right.
struct StudioInspectorLabeledRow<Content: View>: View {
    let label: String
    @ViewBuilder let content: () -> Content

    init(_ label: String, @ViewBuilder content: @escaping () -> Content) {
        self.label = label
        self.content = content
    }

    var body: some View {
        HStack(spacing: 8) {
            Text(label)
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(MereRunTheme.textSecondary)
            Spacer(minLength: 0)
            content()
        }
        .frame(minHeight: 24)
    }
}

/// The board's `slider`: label and mono value above a 4pt track with an accent fill and a
/// 14pt thumb. Drag or use the arrow keys; VoiceOver adjusts it in `step`s.
struct StudioInspectorSlider: View {
    let label: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let step: Double
    let format: (Double) -> String

    private var fraction: Double {
        guard range.upperBound > range.lowerBound else { return 0 }
        return min(max((value - range.lowerBound) / (range.upperBound - range.lowerBound), 0), 1)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(label)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(MereRunTheme.textSecondary)
                Spacer(minLength: 0)
                Text(format(value))
                    .font(.system(size: 12, weight: .medium, design: .monospaced))
                    .foregroundStyle(MereRunTheme.textPrimary)
            }
            GeometryReader { geometry in
                let width = geometry.size.width
                ZStack(alignment: .leading) {
                    Capsule()
                        .fill(MereRunTheme.surfaceRaised)
                        .frame(height: 4)
                    Capsule()
                        .fill(MereRunTheme.accent)
                        .frame(width: max(0, width * fraction), height: 4)
                    Circle()
                        .fill(MereRunTheme.surface)
                        .overlay { Circle().strokeBorder(MereRunTheme.border, lineWidth: 1) }
                        .mereShadow(radius: 1, y: 1)
                        .frame(width: 14, height: 14)
                        .offset(x: max(0, min(width - 14, width * fraction - 7)))
                }
                .frame(height: 16)
                .contentShape(Rectangle())
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { drag in
                            set(fraction: drag.location.x / max(width, 1))
                        }
                )
            }
            .frame(height: 16)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(label)
        .accessibilityValue(format(value))
        .accessibilityAdjustableAction { direction in
            switch direction {
            case .increment: nudge(by: step)
            case .decrement: nudge(by: -step)
            @unknown default: break
            }
        }
        .focusable()
        .onKeyPress(.leftArrow) {
            nudge(by: -step)
            return .handled
        }
        .onKeyPress(.rightArrow) {
            nudge(by: step)
            return .handled
        }
    }

    private func set(fraction: CGFloat) {
        let raw = range.lowerBound + Double(min(max(fraction, 0), 1)) * (range.upperBound - range.lowerBound)
        let snapped = (raw / step).rounded() * step
        value = min(max(snapped, range.lowerBound), range.upperBound)
    }

    private func nudge(by delta: Double) {
        value = min(max(value + delta, range.lowerBound), range.upperBound)
    }
}

/// The aspect presets as a segmented control; a size outside them leaves every segment plain.
struct StudioAspectSegmentedControl: View {
    let mode: StudioMode
    @Binding var draft: StudioDraft

    var body: some View {
        let presets = StudioAspectPreset.inspectorPresets(for: mode)
        let selected = StudioAspectPreset.inspectorSelection(for: draft, mode: mode)
        HStack(spacing: 2) {
            ForEach(presets) { preset in
                MereSegment(title: preset.label, isSelected: preset == selected) {
                    preset.apply(to: &draft)
                }
                .help("\(preset.width) × \(preset.height)")
                .accessibilityLabel("\(preset.label), \(preset.width) by \(preset.height)")
            }
        }
        .padding(2)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background {
            RoundedRectangle(cornerRadius: 7)
                .fill(MereRunTheme.surfaceRaised)
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Aspect ratio")
    }
}
