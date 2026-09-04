import AppKit
import MereRunContract
import StudioKit
import SwiftUI
import UniformTypeIdentifiers

/// The inspector column, rendered from the contract. `StudioContractSchema` turns the current
/// task's capability into sections (the contract's `group`), a disclosure (its `tier`), and one
/// control per option (its `kind`, `range`, `choices`, and `depends_on`); `ContractForm` draws
/// them. The handful of controls the contract cannot describe — the aspect pair, the seed row, the
/// mask canvas, the ordered MiniMax references — come from this file through the override builder.
///
/// Every control binds the same `StudioDraft` the composer's chips edit, so a change in either
/// shows in both, and the header counts what differs from the mode's defaults.
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
    let onClose: () -> Void

    @EnvironmentObject private var controller: MereRunController
    @State private var showAdvanced = false
    @State private var editingSeed = false
    @State private var showImageEditor = false
    @State private var voiceProfiles: [StudioVoiceProfile] = []

    static let width = StudioLayoutPolicy.inspectorWidth

    private var sections: [StudioContractSection] {
        StudioInspectorSchema.sections(for: mode, draft: draft)
    }

    private var advancedFields: [StudioContractField<StudioDraft>] {
        StudioInspectorSchema.advancedFields(for: mode, draft: draft)
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
        .task {
            if mode == .speak { voiceProfiles = await controller.loadVoiceProfiles() }
        }
        .onAppear(perform: normalizeMiniMaxH3Draft)
        .onChange(of: draft.model) { _, _ in normalizeMiniMaxH3Draft() }
        .sheet(isPresented: $showImageEditor) {
            if !draft.inputPath.isBlank {
                StudioImageEditor(
                    inputURL: URL(fileURLWithPath: draft.inputPath),
                    outputWidth: draft.width,
                    outputHeight: draft.height,
                    maskPath: $draft.imageMaskPath,
                    outpaintTop: $draft.imageOutpaintTop,
                    outpaintRight: $draft.imageOutpaintRight,
                    outpaintBottom: $draft.imageOutpaintBottom,
                    outpaintLeft: $draft.imageOutpaintLeft,
                    feather: $draft.imageMaskFeather
                )
            }
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

    private func sectionView(_ section: StudioContractSection) -> some View {
        StudioInspectorSectionView(
            title: section.title,
            canReset: section.changedCount(draft: draft, baseline: baseline) > 0,
            onReset: { section.reset(&draft, to: baseline) }
        ) {
            form(section.fields)
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
                if showAdvanced, StudioInspectorSchema.advancedChanged(mode: mode, draft: draft, baseline: baseline) {
                    resetButton { StudioInspectorSchema.resetAdvanced(for: mode, &draft, to: baseline) }
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

    private func form(_ fields: [StudioContractField<StudioDraft>]) -> some View {
        ContractForm(
            fields: fields,
            dependencies: StudioContractSchema.dependencies(for: mode, draft: draft),
            draft: $draft
        ) { override in
            overrideControl(override)
        }
    }

    private func resetButton(_ action: @escaping () -> Void) -> some View {
        Button("Reset", action: action)
            .buttonStyle(.plain)
            .font(.caption.weight(.medium))
            .foregroundStyle(MereRunTheme.textMuted)
            .help("Back to the defaults")
    }

    // MARK: Overrides

    /// The controls the contract cannot describe: one editor spanning several flags, or one whose
    /// choices come from the machine rather than the contract.
    @ViewBuilder
    private func overrideControl(_ override: StudioContractOverrideID) -> some View {
        switch override {
        case .model: modelPicker
        case .dimensions: aspectControls
        case .seed: seedRow
        case .steps: stepsControl
        case .guidance:
            StudioInspectorSlider(
                label: "Guidance", value: $draft.cfgScale, range: 0...8, step: 0.1,
                format: { StudioComposerPresets.decimalText($0) }
            )
        case .duration: durationControls
        case .voiceProfile: voiceProfilePicker
        case .lora: loraRow
        case .imageCanvas: imageCanvasRow
        case .musicAdapters: musicAdaptersRow
        case .musicLMMode:
            StudioInspectorLabeledRow("LM planning") {
                MereSegmentedControl(["auto", "use", "disable"], selection: $draft.musicLMMode, accessibilityLabel: "LM planning") {
                    switch $0 {
                    case "use": return "On"
                    case "disable": return "Off"
                    default: return "Preset"
                    }
                }
            }
        case .thinking:
            StudioInspectorLabeledRow("Thinking") {
                MereSegmentedControl(TextThinkingMode.allCases, selection: $draft.thinkingMode, accessibilityLabel: "Thinking") {
                    switch $0 {
                    case .automatic: return "Auto"
                    case .show: return "Show"
                    case .hide: return "Off"
                    }
                }
            }
        case .orderedReferences: orderedReferences
        case .readImageAction:
            MereSegmentedControl(
                StudioReadImageAction.allCases,
                selection: $draft.readImageAction,
                accessibilityLabel: "Read task"
            ) { $0.title }
        case .attachment:
            EmptyView()
        }
    }

    // MARK: Model & adapters

    private var modelPicker: some View {
        StudioModelPicker(mode: mode, model: $draft.model, modelInventory: modelInventory, onShowModels: onShowModels) {
            HStack(spacing: 8) {
                if let glyph = modelStatusGlyph {
                    Image(systemName: glyph)
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(MereRunTheme.accent)
                }
                Text(StudioModelNaming.displayLabel(for: mode, model: draft.model))
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
        .help(readiness.blocksRun ? readiness.message : "Model")
        .accessibilityLabel("Model")
        .accessibilityValue(StudioModelNaming.resolvedModelID(for: mode, model: draft.model))
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
                        .font(.caption.weight(.medium))
                        .foregroundStyle(MereRunTheme.textMuted)
                    Text("Add LoRA adapter")
                        .font(.callout.weight(.medium))
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
                    .font(.caption.weight(.medium))
                    .foregroundStyle(MereRunTheme.accent)
                Text(URL(fileURLWithPath: draft.loraPath).lastPathComponent)
                    .font(.callout.weight(.medium))
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
                    .font(.caption.weight(.medium))
                    .foregroundStyle(MereRunTheme.accent)
                Text(URL(fileURLWithPath: path).lastPathComponent)
                    .font(.callout.weight(.medium))
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
                    .font(.caption.weight(.medium))
                    .foregroundStyle(MereRunTheme.textMuted)
                Text("Add ACE-Step adapter")
                    .font(.callout.weight(.medium))
                    .foregroundStyle(MereRunTheme.textSecondary)
            }
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .accessibilityLabel("Add ACE-Step adapter")
    }

    private var voiceProfilePicker: some View {
        StudioInspectorLabeledRow("Profile") {
            Picker("Profile", selection: $draft.voiceProfile) {
                Text("None").tag("")
                ForEach(voiceProfiles) { profile in
                    Text(profile.name).tag(profile.id)
                }
            }
            .labelsHidden()
            .frame(maxWidth: 150)
        }
    }

    // MARK: Output

    private var aspectControls: some View {
        VStack(alignment: .leading, spacing: 10) {
            StudioAspectSegmentedControl(mode: mode, draft: $draft)
            HStack(spacing: 8) {
                StudioInspectorNumberField(label: "Width", value: $draft.width, range: 64...4096)
                Text("×")
                    .font(.callout)
                    .foregroundStyle(MereRunTheme.textMuted)
                StudioInspectorNumberField(label: "Height", value: $draft.height, range: 64...4096)
                Button("Swap") { StudioAspectPreset.swap(&draft) }
                    .buttonStyle(.mereSecondary)
                    .help("Swap width and height")
            }
        }
    }

    @ViewBuilder
    private var durationControls: some View {
        switch mode {
        case .video:
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
        default:
            VStack(alignment: .leading, spacing: 8) {
                Toggle("Set exact length", isOn: $draft.useDuration)
                    .toggleStyle(.checkbox)
                    .font(.callout.weight(.medium))
                    .foregroundStyle(MereRunTheme.textSecondary)
                if draft.useDuration {
                    StudioInspectorSlider(label: "Length", value: $draft.durationSeconds, range: 5...300, step: 1, format: { "\(StudioComposerPresets.secondsText($0)) s" })
                }
            }
        }
    }

    // MARK: Sampling

    @ViewBuilder
    private var stepsControl: some View {
        switch mode {
        case .createImage:
            stepsSlider(range: 1...30)
        case .video where StudioVideoModelFamily(model: draft.model).isMiniMaxH3:
            VStack(alignment: .leading, spacing: 8) {
                Toggle(
                    "Override adaptive schedule",
                    isOn: Binding(get: { draft.h3Steps != nil }, set: { draft.h3Steps = $0 ? 21 : nil })
                )
                .toggleStyle(.checkbox)
                .font(.callout.weight(.medium))
                .foregroundStyle(MereRunTheme.textSecondary)
                if draft.h3Steps != nil {
                    StudioInspectorSlider(
                        label: "Schedule points",
                        value: Binding(get: { Double(draft.h3Steps ?? 21) }, set: { draft.h3Steps = Int($0.rounded()) }),
                        range: 1...64, step: 1, format: { String(Int($0)) }
                    )
                }
            }
        case .video:
            stepsSlider(range: 1...60)
        case .music:
            VStack(alignment: .leading, spacing: 8) {
                Toggle("Override preset steps", isOn: $draft.musicOverrideSteps)
                    .toggleStyle(.checkbox)
                    .font(.callout.weight(.medium))
                    .foregroundStyle(MereRunTheme.textSecondary)
                if draft.musicOverrideSteps { stepsSlider(range: 1...120) }
            }
        default:
            stepsSlider(range: 1...64)
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
                .font(.callout.weight(.medium))
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

    // MARK: Inputs

    @ViewBuilder
    private var imageCanvasRow: some View {
        if !draft.inputPath.isBlank {
            HStack(spacing: MereRunTheme.Spacing.sm) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(draft.imageMaskPath.isBlank && !hasImageOutpaint ? "Whole-image transform" : "Masked image edit")
                        .font(.system(size: 12.5, weight: .medium))
                    Text(draft.imageMaskPath.isBlank
                        ? (hasImageOutpaint ? "Outpaint edges are editable" : "Paint a mask or expand the canvas")
                        : URL(fileURLWithPath: draft.imageMaskPath).lastPathComponent)
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                }
                Spacer()
                Button("Edit canvas…") { showImageEditor = true }
                    .buttonStyle(.mereSecondary)
            }
            .padding(MereRunTheme.Spacing.sm)
            .background {
                RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                    .fill(MereRunTheme.accentSoft.opacity(0.55))
            }
        }
    }

    private var hasImageOutpaint: Bool {
        [
            draft.imageOutpaintTop,
            draft.imageOutpaintRight,
            draft.imageOutpaintBottom,
            draft.imageOutpaintLeft,
        ].contains { $0 > 0 }
    }

    @ViewBuilder
    private var orderedReferences: some View {
        let references = draft.h3ReferenceInputs ?? []
        VStack(alignment: .leading, spacing: 6) {
            Text("Ordered Ref2VA references")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            ForEach(Array(references.enumerated()), id: \.offset) { index, reference in
                HStack(spacing: 5) {
                    Text(reference)
                        .font(MereRunTheme.captionFont)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Spacer()
                    Button { moveH3Reference(index, by: -1) } label: { Image(systemName: "arrow.up") }
                        .disabled(index == 0)
                        .accessibilityLabel("Move up")
                    Button { moveH3Reference(index, by: 1) } label: { Image(systemName: "arrow.down") }
                        .disabled(index == references.count - 1)
                        .accessibilityLabel("Move down")
                    Button(role: .destructive) { removeH3Reference(index) } label: { Image(systemName: "trash") }
                        .accessibilityLabel("Remove reference")
                }
            }
            HStack {
                Button("Image") { chooseH3Reference(kind: "image", type: .image) }
                Button("Video") { chooseH3Reference(kind: "video", type: .movie) }
                Button("Audio") { chooseH3Reference(kind: "audio", type: .audio) }
            }
            .controlSize(.small)
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

    private func chooseH3Reference(kind: String, type: UTType) {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [type]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        draft.h3ReferenceInputs = (draft.h3ReferenceInputs ?? []) + ["\(kind):\(url.path)"]
    }

    private func moveH3Reference(_ index: Int, by offset: Int) {
        var references = draft.h3ReferenceInputs ?? []
        references.swapAt(index, index + offset)
        draft.h3ReferenceInputs = references
    }

    private func removeH3Reference(_ index: Int) {
        var references = draft.h3ReferenceInputs ?? []
        references.remove(at: index)
        draft.h3ReferenceInputs = references
    }

    /// MiniMax-H3 fixes the frame rate, aligns the size to 32 pixels and the frame count to 17n+5,
    /// and takes either an ordered reference list or a start/end keyframe pair, never both.
    private func normalizeMiniMaxH3Draft() {
        guard mode == .video, StudioVideoModelFamily(model: draft.model).isMiniMaxH3 else { return }
        draft.fps = 24
        draft.width = max(32, (draft.width / 32) * 32)
        draft.height = max(32, (draft.height / 32) * 32)
        draft.numFrames = StudioVideoModelFamily.alignedMiniMaxH3FrameCount(draft.numFrames)
        draft.audioPath = ""
        draft.timings = false
        draft.timingsOutputPath = ""
        if StudioVideoModelFamily(model: draft.model) == .miniMaxH3Ref2VA {
            draft.inputPath = ""
            draft.endImagePath = ""
        } else {
            draft.h3ReferenceInputs = []
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
                    .font(.callout.weight(.semibold))
                    .foregroundStyle(MereRunTheme.textPrimary)
                    .accessibilityAddTraits(.isHeader)
                Spacer(minLength: 0)
                Button("Reset", action: onReset)
                    .buttonStyle(.plain)
                    .font(.caption.weight(.medium))
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
    /// Monospaced where the value is a number or a path the reader compares character by
    /// character; proportional everywhere else.
    var isMonospaced = false

    var body: some View {
        TextField(
            "",
            text: $text,
            prompt: Text(placeholder).foregroundStyle(MereRunTheme.textMuted),
            axis: lines.upperBound > 1 ? .vertical : .horizontal
        )
        .textFieldStyle(.plain)
        .font(.system(size: 13, design: isMonospaced ? .monospaced : .default))
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
                .font(.callout.weight(.medium))
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
                    .font(.callout.weight(.medium))
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
