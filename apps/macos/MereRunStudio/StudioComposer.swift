import AppKit
import MereRunContract
import SwiftUI
import UniformTypeIdentifiers

/// The composer pinned under the canvas: the attachment well, the prompt, and a strip of the
/// mode's essential parameters as chips, with Run on the right. Slots and chips come from
/// `StudioComposerSchema.swift`; the remaining depth stays behind the options popover.
struct StudioComposer: View {
    let mode: StudioMode
    @Binding var draft: StudioDraft
    @Binding var showOptions: Bool
    let isRunning: Bool
    let queuedCount: Int
    let readiness: ModelReadinessState
    var sendBlocked: Bool = false
    /// Every row of `model list`, installed or not; the model chip filters it to the mode.
    let modelInventory: [StudioModelInventoryRow]
    /// The seed of the mode's most recent run, for "Reuse last".
    var lastSeed: String?
    var promptFocus: FocusState<Bool>.Binding
    let onRun: () -> Void
    let onStop: () -> Void
    let onShowModels: () -> Void
    let onShowAdapters: () -> Void
    let onShowRealtimeMusic: () -> Void

    @EnvironmentObject private var controller: MereRunController
    @State private var editingChip: StudioComposerChipKind?

    private enum Metrics {
        static let outerInsets = EdgeInsets(top: 12, leading: 24, bottom: 20, trailing: 24)
        static let innerInsets = EdgeInsets(top: 14, leading: 16, bottom: 12, trailing: 16)
        static let cornerRadius: CGFloat = 18
        static let rowSpacing: CGFloat = 12
        static let sendDiameter: CGFloat = 32
    }

    var body: some View {
        VStack(alignment: .leading, spacing: Metrics.rowSpacing) {
            if mode.showsAttachmentWell(for: draft) {
                attachmentWell
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }
            promptEntry
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
        .animation(MereRunTheme.Motion.standard, value: mode.showsAttachmentWell(for: draft))
    }

    // MARK: - Attachment well

    private var visibleSlots: [StudioAttachmentSlot] {
        mode.attachmentSlots.filter { !$0.isTransient || $0.isFilled(in: draft) }
    }

    private var attachmentWell: some View {
        HStack(alignment: .center, spacing: 8) {
            ForEach(visibleSlots) { slot in
                StudioAttachmentSlotView(slot: slot, draft: $draft, onPick: { pickFiles(for: slot) })
            }
            VStack(alignment: .leading, spacing: 1) {
                ForEach(visibleSlots) { slot in
                    Text(slot.caption(in: draft))
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(MereRunTheme.textMuted)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
            }
            .padding(.leading, 4)
            Spacer(minLength: 0)
        }
    }

    private func pickFiles(for slot: StudioAttachmentSlot) {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = slot.allowsMultiple
        panel.allowedContentTypes = slot.acceptedTypes
        guard panel.runModal() == .OK else { return }
        slot.attach(panel.urls, to: &draft)
    }

    // MARK: - Prompt

    @ViewBuilder
    private var promptEntry: some View {
        if mode == .listen {
            Text(mode.promptPlaceholder)
                .font(.system(size: 15))
                .foregroundStyle(MereRunTheme.textMuted)
                .frame(maxWidth: .infinity, minHeight: 22, alignment: .leading)
        } else {
            TextField(
                "",
                text: $draft.prompt,
                prompt: Text(mode.promptPlaceholder).foregroundStyle(MereRunTheme.textMuted),
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
            .accessibilityLabel(mode.promptPlaceholder)
        }
    }

    // MARK: - Chip strip

    private var chipStrip: some View {
        HStack(alignment: .center, spacing: 6) {
            ForEach(mode.composerChips) { kind in
                chip(for: kind)
            }
            Spacer(minLength: 8)
            HStack(spacing: 8) {
                optionsButton
                if showsPaperclip { paperclipButton }
                if isRunning { stopButton } else { sendButton }
            }
        }
    }

    @ViewBuilder
    private func chip(for kind: StudioComposerChipKind) -> some View {
        switch kind {
        case .dimensions: dimensionsChip
        case .duration: durationChip
        case .steps: stepsChip
        case .seed: seedChip
        case .threshold: thresholdChip
        case .readImageAction: readImageActionChip
        case .voiceMode: voiceModeChip
        case .thinking: thinkingChip
        case .model: modelChip
        }
    }

    private var dimensionsChip: some View {
        chipMenu(
            title: StudioComposerPresets.dimensionsTitle(draft),
            accessibilityLabel: "Size",
            kind: .dimensions
        ) {
            ForEach(StudioAspectPreset.presets(for: mode)) { preset in
                Toggle(isOn: Binding(get: { preset.matches(draft) }, set: { _ in preset.apply(to: &draft) })) {
                    Text("\(preset.label) · \(preset.width) × \(preset.height)")
                }
            }
            Divider()
            Button("Swap width and height") {
                let width = draft.width
                draft.width = draft.height
                draft.height = width
            }
            Button("Custom size…") { editingChip = .dimensions }
        } editor: {
            HStack(spacing: MereRunTheme.Spacing.xs) {
                numberField("Width", value: $draft.width, range: 64...4096)
                Text("×").foregroundStyle(MereRunTheme.textMuted)
                numberField("Height", value: $draft.height, range: 64...4096)
            }
        }
    }

    private var durationChip: some View {
        chipMenu(
            title: StudioComposerPresets.durationTitle(draft, mode: mode),
            accessibilityLabel: "Length",
            kind: .duration
        ) {
            if mode == .music {
                Toggle(isOn: Binding(get: { !draft.useDuration }, set: { _ in draft.useDuration = false })) {
                    Text("Preset length")
                }
                Divider()
            }
            ForEach(StudioComposerPresets.durations(for: mode), id: \.self) { seconds in
                Toggle(isOn: Binding(
                    get: { usesSeconds && draft.durationSeconds == seconds },
                    set: { _ in setDuration(seconds: seconds) }
                )) {
                    Text("\(StudioComposerPresets.secondsText(seconds)) s")
                }
            }
            if mode == .video {
                Divider()
                ForEach(StudioComposerPresets.videoFrameCounts, id: \.self) { frames in
                    Toggle(isOn: Binding(
                        get: { !draft.useDuration && draft.numFrames == frames },
                        set: { _ in draft.useDuration = false; draft.numFrames = frames }
                    )) {
                        Text("\(frames) frames")
                    }
                }
            }
            Divider()
            Button("Custom length…") { editingChip = .duration }
        } editor: {
            HStack(spacing: MereRunTheme.Spacing.xs) {
                if mode == .video {
                    Picker("Count", selection: $draft.useDuration) {
                        Text("Frames").tag(false)
                        Text("Seconds").tag(true)
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                }
                if mode == .video, !draft.useDuration {
                    numberField("Frames", value: $draft.numFrames, range: 1...600)
                } else {
                    decimalField("Seconds", value: Binding(
                        get: { draft.durationSeconds },
                        set: { setDuration(seconds: max(0.1, $0)) }
                    ))
                }
            }
        }
    }

    private var usesSeconds: Bool {
        mode == .sfx || draft.useDuration
    }

    private func setDuration(seconds: Double) {
        draft.durationSeconds = seconds
        if mode != .sfx { draft.useDuration = true }
    }

    private var stepsChip: some View {
        chipMenu(
            title: StudioComposerPresets.stepsTitle(draft, mode: mode),
            accessibilityLabel: "Steps",
            kind: .steps
        ) {
            if mode == .music {
                Toggle(isOn: Binding(get: { !draft.musicOverrideSteps }, set: { _ in draft.musicOverrideSteps = false })) {
                    Text("Preset steps")
                }
                Divider()
            }
            ForEach(StudioComposerPresets.steps(for: mode), id: \.self) { steps in
                Toggle(isOn: Binding(
                    get: { stepsAreExplicit && draft.steps == steps },
                    set: { _ in setSteps(steps) }
                )) {
                    Text(steps == 1 ? "1 step" : "\(steps) steps")
                }
            }
            Divider()
            Button("Custom steps…") { editingChip = .steps }
        } editor: {
            numberField("Steps", value: Binding(get: { draft.steps }, set: { setSteps($0) }), range: 1...200)
        }
    }

    private var stepsAreExplicit: Bool {
        mode != .music || draft.musicOverrideSteps
    }

    private func setSteps(_ steps: Int) {
        draft.steps = steps
        if mode == .music { draft.musicOverrideSteps = true }
    }

    private var seedChip: some View {
        let seedMode = StudioSeedMode(draft: draft)
        return chipMenu(title: seedMode.chipTitle, accessibilityLabel: "Seed", kind: .seed) {
            Toggle(isOn: Binding(get: { seedMode == .random }, set: { _ in StudioSeedMode.random.apply(to: &draft) })) {
                Text("Random")
            }
            Button("Enter a seed…") { editingChip = .seed }
            Divider()
            Button("Reuse last\(lastSeed.map { " (\($0))" } ?? "")") {
                if let lastSeed { draft.seed = lastSeed }
            }
            .disabled(lastSeed == nil)
        } editor: {
            HStack(spacing: MereRunTheme.Spacing.xs) {
                TextField("Seed", text: $draft.seed)
                    .frame(width: 120)
                    .mereField(cornerRadius: MereRunTheme.Radius.sm)
                Button("Random") { StudioSeedMode.random.apply(to: &draft) }
                    .buttonStyle(.mereSecondary)
            }
        }
    }

    private var thresholdChip: some View {
        chipMenu(
            title: StudioComposerPresets.thresholdTitle(draft),
            accessibilityLabel: "Threshold",
            kind: .threshold
        ) {
            ForEach(StudioComposerPresets.thresholds, id: \.self) { threshold in
                Toggle(isOn: Binding(
                    get: { draft.visionThreshold == threshold },
                    set: { _ in draft.visionThreshold = threshold }
                )) {
                    Text(StudioComposerPresets.decimalText(threshold))
                }
            }
            Divider()
            Button("Custom threshold…") { editingChip = .threshold }
        } editor: {
            decimalField("Threshold", value: Binding(
                get: { draft.visionThreshold },
                set: { draft.visionThreshold = min(max($0, 0), 1) }
            ))
        }
    }

    private var readImageActionChip: some View {
        chipMenu(title: draft.readImageAction.title, accessibilityLabel: "Task", kind: .readImageAction) {
            ForEach(StudioReadImageAction.allCases) { action in
                Toggle(isOn: Binding(
                    get: { draft.readImageAction == action },
                    set: { _ in draft.readImageAction = action }
                )) {
                    Text(action.title)
                }
                .disabled(readImageActionUnavailableMessage(action) != nil)
                .help(readImageActionUnavailableMessage(action) ?? action.title)
            }
        }
    }

    /// Why a Read Image variant cannot run on this machine, from the capability report.
    private func readImageActionUnavailableMessage(_ action: StudioReadImageAction) -> String? {
        guard mode == .readImage else { return nil }
        var candidateDraft = draft
        candidateDraft.readImageAction = action
        switch StudioCommandAdapter.capabilityRequirement(for: .readImage, draft: candidateDraft) {
        case .unavailable(let message):
            return message
        case .managedModel(let modelID):
            return controller.modelCapabilitiesByID[modelID]?.unavailableMessage
        case nil:
            return nil
        }
    }

    private var voiceModeChip: some View {
        chipMenu(title: StudioComposerPresets.voiceModeTitle(draft), accessibilityLabel: "Voice", kind: .voiceMode) {
            Toggle(isOn: Binding(get: { draft.voiceMode != "clone" }, set: { _ in draft.voiceMode = "style" })) {
                Text("Preset voice")
            }
            Toggle(isOn: Binding(get: { draft.voiceMode == "clone" }, set: { _ in draft.voiceMode = "clone" })) {
                Text("Cloned voice")
            }
        }
    }

    private var thinkingChip: some View {
        chipMenu(
            title: StudioComposerPresets.thinkingTitle(draft.thinkingMode),
            accessibilityLabel: "Thinking",
            kind: .thinking
        ) {
            ForEach(TextThinkingMode.allCases, id: \.self) { thinking in
                Toggle(isOn: Binding(
                    get: { draft.thinkingMode == thinking },
                    set: { _ in draft.thinkingMode = thinking }
                )) {
                    Text(StudioComposerPresets.thinkingTitle(thinking))
                }
            }
        }
    }

    /// A chip that opens a menu, with an optional popover for values the menu cannot type.
    private func chipMenu<Items: View, Editor: View>(
        title: String,
        accessibilityLabel: String,
        kind: StudioComposerChipKind,
        @ViewBuilder items: () -> Items,
        @ViewBuilder editor: () -> Editor
    ) -> some View {
        let editorView = editor()
        return Menu(content: items) {
            StudioComposerChipLabel(title: title)
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .fixedSize()
        .accessibilityLabel(accessibilityLabel)
        .accessibilityValue(title)
        .popover(
            isPresented: Binding(get: { editingChip == kind }, set: { if !$0 { editingChip = nil } }),
            arrowEdge: .top
        ) {
            editorView
                .padding(MereRunTheme.Spacing.md)
                .background(MereRunTheme.background)
                .foregroundStyle(MereRunTheme.textPrimary)
        }
    }

    private func chipMenu<Items: View>(
        title: String,
        accessibilityLabel: String,
        kind: StudioComposerChipKind,
        @ViewBuilder items: () -> Items
    ) -> some View {
        chipMenu(title: title, accessibilityLabel: accessibilityLabel, kind: kind, items: items) { EmptyView() }
    }

    private func numberField(_ label: String, value: Binding<Int>, range: ClosedRange<Int>) -> some View {
        TextField(
            label,
            value: Binding(get: { value.wrappedValue }, set: { value.wrappedValue = min(max($0, range.lowerBound), range.upperBound) }),
            format: .number
        )
        .frame(width: 72)
        .mereField(cornerRadius: MereRunTheme.Radius.sm)
        .accessibilityLabel(label)
    }

    private func decimalField(_ label: String, value: Binding<Double>) -> some View {
        TextField(label, value: value, format: .number)
            .frame(width: 84)
            .mereField(cornerRadius: MereRunTheme.Radius.sm)
            .accessibilityLabel(label)
    }

    // MARK: - Model chip

    private var modelChip: some View {
        Menu {
            Toggle(isOn: Binding(get: { draft.model.isBlank }, set: { _ in draft.model = "" })) {
                Text(defaultModelID.isEmpty ? "Auto" : "Auto · \(Self.displayModelName(defaultModelID))")
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
            StudioComposerChipLabel(title: modelLabel, leadingSystemImage: modelStatusGlyph)
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .fixedSize()
        .help(modelHelp)
        .accessibilityLabel("Model")
        .accessibilityValue(modelAccessibilityValue)
    }

    private func modelRow(_ row: StudioModelInventoryRow) -> some View {
        Toggle(isOn: Binding(get: { row.id == draft.model }, set: { _ in draft.model = row.id })) {
            Label(Self.displayModelName(row.id), systemImage: row.isInstalled ? "internaldrive" : "arrow.down.circle")
        }
    }

    /// The mode's template default, shown as "Auto".
    private var defaultModelID: String {
        CommandCatalog.template(id: mode.defaultTemplateID)?.defaultModel ?? ""
    }

    /// The resolved model id for this run: the explicit draft model, else the mode's default.
    private var rawModelID: String {
        let current = draft.model.trimmingCharacters(in: .whitespacesAndNewlines)
        return current.isEmpty ? defaultModelID : current
    }

    private var modelLabel: String {
        rawModelID.isEmpty ? "Auto" : Self.displayModelName(rawModelID)
    }

    /// A glyph before the model name when the model is not ready: missing locally, or unsupported.
    private var modelStatusGlyph: String? {
        switch readiness {
        case .missingModel: return "arrow.down.circle"
        case .unsupported: return "exclamationmark.triangle"
        case .checking, .ready, .unknown: return nil
        }
    }

    private var modelHelp: String {
        let identity = rawModelID.isEmpty ? "Auto — the mode's default model" : "Model: \(rawModelID)"
        switch readiness {
        case .ready, .unknown: return identity
        default: return "\(identity) · \(readiness.message)"
        }
    }

    private var modelAccessibilityValue: String {
        let identity = rawModelID.isEmpty ? "Automatic" : rawModelID
        return "\(identity), \(readiness.title)"
    }

    /// A human-facing label for a model id: drop the modality/category prefix and
    /// title-case the distinctive remainder ("text-agent-deepseek-v4-flash" →
    /// "Deepseek V4 Flash"). The exact id stays in the chip's tooltip, so the
    /// friendly name never hides what the CLI actually expects.
    static func displayModelName(_ id: String) -> String {
        let leaf = id.components(separatedBy: "/").last ?? id
        let prefixes = [
            "text-chat-", "text-agent-", "text-embed-", "text-code-",
            "image-", "video-", "music-", "sfx-", "speech-tts-", "speech-asr-", "speech-",
            "embed-", "vision-ground-", "vision-segment-", "vision-chat-", "vision-ocr-", "vision-", "text-"
        ]
        var core = leaf
        for prefix in prefixes where core.hasPrefix(prefix) {
            core = String(core.dropFirst(prefix.count))
            break
        }
        let words = core.split(separator: "-").map { token -> String in
            guard let first = token.first, first.isLetter else { return String(token) }
            return first.uppercased() + token.dropFirst()
        }
        let label = words.joined(separator: " ")
        return label.isEmpty ? leaf : label
    }

    // MARK: - Right cluster

    private var optionsButton: some View {
        Button {
            showOptions.toggle()
        } label: {
            Image(systemName: "slider.horizontal.3")
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(showOptions ? MereRunTheme.accent : MereRunTheme.textMuted)
                .frame(width: 28, height: 28)
        }
        .buttonStyle(.mereIcon)
        .help("More options")
        .accessibilityLabel("More options")
        .popover(isPresented: $showOptions, arrowEdge: .top) {
            StudioOptionsPanel(
                mode: mode,
                draft: $draft,
                onShowAdapters: onShowAdapters,
                onShowRealtimeMusic: onShowRealtimeMusic
            )
        }
    }

    /// Only a collapsed well (Chat's per-turn image) needs the paperclip; declared slots pick
    /// from the well itself.
    private var showsPaperclip: Bool {
        !mode.attachmentSlots.isEmpty && !mode.showsAttachmentWell(for: draft)
    }

    private var paperclipButton: some View {
        Button {
            if let slot = mode.attachmentSlots.first { pickFiles(for: slot) }
        } label: {
            Image(systemName: "paperclip")
                .font(.system(size: 15, weight: .medium))
                .foregroundStyle(MereRunTheme.textMuted)
                .frame(width: 28, height: 28)
        }
        .buttonStyle(.mereIcon)
        .help("Attach an image to this turn")
        .accessibilityLabel("Attach image")
    }

    private var stopButton: some View {
        Button(action: onStop) {
            ZStack {
                Circle().fill(MereRunTheme.surfaceRaised)
                Image(systemName: "stop.fill")
                    .font(.system(size: 11, weight: .bold))
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
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(sendEnabled ? MereRunTheme.onAccent : MereRunTheme.textMuted)
            }
            .frame(width: Metrics.sendDiameter, height: Metrics.sendDiameter)
        }
        .buttonStyle(.plain)
        .disabled(!sendEnabled)
        .help(sendHelp)
        .accessibilityLabel("Run")
        .accessibilityHint(accessibilityRunHint)
        .keyboardShortcut(.return, modifiers: .command)
    }

    private var sendEnabled: Bool {
        !readiness.blocksRun && !sendBlocked
    }

    private var sendHelp: String {
        if sendBlocked { return "Waiting for the current reply…" }
        if readiness.blocksRun { return readiness.message }
        return "Run (⌘↩)"
    }

    /// Spoken explanation of why Run is disabled, so the blocked state is not color-only.
    private var accessibilityRunHint: String {
        if sendBlocked { return "Waiting for the current reply to finish" }
        if readiness.blocksRun { return readiness.message }
        return ""
    }
}

/// The chip body: 24pt tall on `surfaceRaised`, 11.5pt medium, a muted chevron when it opens a menu.
struct StudioComposerChipLabel: View {
    let title: String
    var leadingSystemImage: String?
    var menu = true

    var body: some View {
        HStack(spacing: 4) {
            if let leadingSystemImage {
                Image(systemName: leadingSystemImage)
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(MereRunTheme.accent)
            }
            Text(title)
                .font(.system(size: 11.5, weight: .medium))
                .foregroundStyle(MereRunTheme.textPrimary)
                .lineLimit(1)
                .truncationMode(.middle)
                .frame(maxWidth: 180)
                .fixedSize(horizontal: true, vertical: false)
            if menu {
                Image(systemName: "chevron.down")
                    .font(.system(size: 8, weight: .semibold))
                    .foregroundStyle(MereRunTheme.textMuted)
            }
        }
        .padding(.horizontal, 9)
        .frame(height: 24)
        .background {
            RoundedRectangle(cornerRadius: MereRunTheme.Radius.sm)
                .fill(MereRunTheme.surfaceRaised)
        }
        .contentShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.sm))
    }
}

/// One 48×48 slot of the attachment well. Empty: a dashed outline with a plus. Filled: the
/// file's thumbnail (or a kind glyph for audio and video) with a hover-revealed remove button.
/// Accepts a drop, a paste (⌘V while focused), and a click to pick.
struct StudioAttachmentSlotView: View {
    let slot: StudioAttachmentSlot
    @Binding var draft: StudioDraft
    let onPick: () -> Void

    @State private var isDropTargeted = false
    @State private var hovering = false

    private static let side: CGFloat = 48
    private static let cornerRadius: CGFloat = MereRunTheme.Radius.base

    private var paths: [String] { slot.paths(in: draft) }
    private var isFilled: Bool { !paths.isEmpty }

    var body: some View {
        Button(action: onPick) {
            ZStack {
                if let first = paths.first {
                    thumbnail(for: URL(fileURLWithPath: first))
                } else {
                    Image(systemName: "plus")
                        .font(.system(size: 15, weight: .medium))
                        .foregroundStyle(MereRunTheme.textMuted)
                }
            }
            .frame(width: Self.side, height: Self.side)
            .clipShape(RoundedRectangle(cornerRadius: Self.cornerRadius))
            .overlay { outline }
            .overlay(alignment: .topTrailing) {
                if isFilled, hovering {
                    Button {
                        slot.clear(in: &draft)
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 13))
                            .symbolRenderingMode(.palette)
                            .foregroundStyle(MereRunTheme.surface, MereRunTheme.textPrimary)
                    }
                    .buttonStyle(.plain)
                    .padding(3)
                    .help("Remove")
                    .accessibilityLabel("Remove \(slot.label.lowercased())")
                }
            }
            .contentShape(RoundedRectangle(cornerRadius: Self.cornerRadius))
        }
        .buttonStyle(.plain)
        .focusable()
        .onHover { hovering = $0 }
        .dropDestination(for: URL.self) { urls, _ in
            let accepted = urls.filter(slot.accepts)
            guard !accepted.isEmpty else { return false }
            slot.attach(accepted, to: &draft)
            return true
        } isTargeted: { targeted in
            withAnimation(MereRunTheme.Motion.quick) { isDropTargeted = targeted }
        }
        .onPasteCommand(of: [.fileURL, .image]) { _ in paste() }
        .contextMenu {
            Button("Choose…", action: onPick)
            if isFilled {
                Button("Reveal in Finder") {
                    NSWorkspace.shared.activateFileViewerSelecting(paths.map { URL(fileURLWithPath: $0) })
                }
                Button("Remove") { slot.clear(in: &draft) }
            }
        }
        .help(isFilled ? paths.joined(separator: "\n") : "Drop, paste, or click to add \(slot.label.lowercased())")
        .accessibilityLabel(slot.label)
        .accessibilityValue(isFilled ? slot.caption(in: draft) : "Empty")
        .accessibilityHint("Drop a file, paste with Command-V, or click to choose")
    }

    @ViewBuilder
    private var outline: some View {
        let shape = RoundedRectangle(cornerRadius: Self.cornerRadius)
        if isDropTargeted {
            shape.strokeBorder(MereRunTheme.accent, style: StrokeStyle(lineWidth: 2, dash: [5, 4]))
        } else if isFilled {
            shape.strokeBorder(MereRunTheme.border, lineWidth: 1)
        } else {
            shape.strokeBorder(MereRunTheme.border, style: StrokeStyle(lineWidth: 1, dash: [4, 3]))
        }
    }

    @ViewBuilder
    private func thumbnail(for url: URL) -> some View {
        switch StudioOutputFileKind.classify(url) {
        case .image:
            StudioAsyncImagePreview(url: url, maxPixelSize: 160, contentMode: .fill, fallbackSystemImage: "photo")
        case .audio:
            kindGlyph("waveform")
        case .video:
            kindGlyph("film")
        default:
            kindGlyph("doc")
        }
    }

    private func kindGlyph(_ systemImage: String) -> some View {
        ZStack {
            MereRunTheme.surfaceRaised
            Image(systemName: systemImage)
                .font(.system(size: 16, weight: .medium))
                .foregroundStyle(MereRunTheme.accent)
        }
    }

    private func paste() {
        let pasteboard = NSPasteboard.general
        let urls = StudioAttachmentPasteboard.fileURLs(from: pasteboard, for: slot)
        if !urls.isEmpty {
            slot.attach(urls, to: &draft)
            return
        }
        guard slot.acceptedTypes.contains(where: { UTType.image.conforms(to: $0) }),
              let url = try? StudioAttachmentPasteboard.writePastedImage(from: pasteboard) else { return }
        slot.attach([url], to: &draft)
    }
}

/// The per-mode depth controls, presented as a popover from the composer's Options button
/// (inline depth, not a modal interruption). Fields mirror the Advanced surface via
/// `StudioOptionSchema`, so the two surfaces never drift.
struct StudioOptionsPanel: View {
    let mode: StudioMode
    @Binding var draft: StudioDraft
    let onShowAdapters: () -> Void
    let onShowRealtimeMusic: () -> Void
    @EnvironmentObject private var controller: MereRunController
    @State private var voiceProfiles: [StudioVoiceProfile] = []
    @State private var showImageEditor = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: MereRunTheme.Spacing.md) {
                Text("\(mode.title) options")
                    .font(MereRunTheme.sectionFont)
                    .foregroundStyle(MereRunTheme.textMuted)

                if let secondaryLabel {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(secondaryLabel)
                            .font(MereRunTheme.captionFont)
                            .foregroundStyle(MereRunTheme.textMuted)
                        TextField(secondaryLabel, text: $draft.secondaryText, axis: .vertical)
                            .lineLimit(1...4)
                            .mereField()
                    }
                }

                if mode == .speak {
                    speakOptions
                }

                if mode == .video {
                    videoOptions
                }

                if mode == .chat {
                    chatOptions
                }

                if mode == .createImage {
                    imageOptions
                }

                if mode == .music {
                    musicOptions
                }


                if !visibleOptionFields.isEmpty {
                    Divider().overlay(MereRunTheme.border.opacity(0.4))
                    ForEach(visibleOptionFields) { field in
                        optionRow(field)
                    }
                }
            }
            .padding(MereRunTheme.Spacing.lg)
        }
        .frame(width: 360)
        .frame(maxHeight: 480)
        .background(MereRunTheme.background)
        .foregroundStyle(MereRunTheme.textPrimary)
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
    }

    private var visibleOptionFields: [StudioOptionField] {
        let fields = StudioOptionSchema.fields(for: mode)
        guard mode == .video, videoFamily.isMiniMaxH3 else { return fields }
        return fields.filter { !["fps", "numFrames"].contains($0.id) }
    }

    private var videoFamily: StudioVideoModelFamily {
        StudioVideoModelFamily(model: draft.model)
    }

    @ViewBuilder
    private var speakOptions: some View {
        Picker("Voice", selection: $draft.voiceMode) {
            Text("Style").tag("style")
            Text("Clone").tag("clone")
        }
        .pickerStyle(.segmented)
        .labelsHidden()

        if draft.voiceMode == "clone" {
            Picker("Profile", selection: $draft.voiceProfile) {
                Text("None").tag("")
                ForEach(voiceProfiles) { profile in
                    Text(profile.name).tag(profile.id)
                }
            }
            HStack(spacing: MereRunTheme.Spacing.sm) {
                Text(draft.refAudioPath.isEmpty
                    ? "No reference audio"
                    : URL(fileURLWithPath: draft.refAudioPath).lastPathComponent)
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                    .lineLimit(1)
                Spacer()
                Button("Reference audio…") { chooseReferenceAudio() }
                    .controlSize(.small)
            }
            TextField("Save as profile (optional)", text: $draft.saveProfileName)
                .mereField()
        }
    }

    /// Renders one schema field as the appropriate control, bound through the draft key path.
    @ViewBuilder
    private func optionRow(_ field: StudioOptionField) -> some View {
        switch field.control {
        case let .int(keyPath, range, step):
            Stepper("\(field.label): \(draft[keyPath: keyPath])", value: binding(keyPath), in: range, step: step)
                .font(MereRunTheme.captionFont)
        case let .double(keyPath):
            HStack {
                Text(field.label)
                    .font(MereRunTheme.captionFont)
                Spacer()
                TextField(field.label, value: binding(keyPath), format: .number)
                    .frame(width: 80)
                    .mereField(cornerRadius: MereRunTheme.Radius.sm)
            }
        case let .bool(keyPath):
            Toggle(field.label, isOn: binding(keyPath))
                .font(MereRunTheme.captionFont)
        case let .text(keyPath, placeholder):
            HStack {
                Text(field.label)
                    .font(MereRunTheme.captionFont)
                Spacer()
                TextField(placeholder, text: binding(keyPath))
                    .frame(width: 150)
                    .mereField(cornerRadius: MereRunTheme.Radius.sm)
            }
        }
    }

    private func binding<Value>(_ keyPath: WritableKeyPath<StudioDraft, Value>) -> Binding<Value> {
        Binding(get: { draft[keyPath: keyPath] }, set: { draft[keyPath: keyPath] = $0 })
    }

    private func chooseReferenceAudio() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.audio]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        if panel.runModal() == .OK, let url = panel.url {
            draft.refAudioPath = url.path
        }
    }

    @ViewBuilder
    private var chatOptions: some View {
        Picker("Response", selection: $draft.responseFormat) {
            Text("Text").tag(TextResponseFormat.text)
            Text("JSON").tag(TextResponseFormat.jsonObject)
        }
        .pickerStyle(.segmented)

        Picker("Reasoning", selection: $draft.thinkingMode) {
            Text("Default").tag(TextThinkingMode.automatic)
            Text("Show").tag(TextThinkingMode.show)
            Text("Disable").tag(TextThinkingMode.hide)
        }
        .pickerStyle(.segmented)

        if draft.model.localizedCaseInsensitiveContains("inkling") {
            HStack {
                Text("Reasoning effort")
                    .font(MereRunTheme.captionFont)
                Spacer()
                TextField(
                    "0.9",
                    value: Binding(
                        get: { draft.reasoningEffort ?? 0.9 },
                        set: { draft.reasoningEffort = min(max($0, 0), 0.99) }
                    ),
                    format: .number
                )
                .frame(width: 80)
                .mereField(cornerRadius: MereRunTheme.Radius.sm)
            }
        }

        Stepper(
            "Context: \(draft.contextSize == 0 ? "model default" : String(draft.contextSize))",
            value: $draft.contextSize,
            in: 0...262_144,
            step: 1_024
        )
        .font(MereRunTheme.captionFont)
        Stepper(
            "Top-k: \(draft.topK == 0 ? "model default" : String(draft.topK))",
            value: $draft.topK,
            in: 0...512
        )
        .font(MereRunTheme.captionFont)
        TextField("Min-p (0 = model default)", value: $draft.minP, format: .number)
            .textFieldStyle(.roundedBorder)

        Picker("KV bits", selection: $draft.kvBits) {
            Text("Auto").tag(0)
            Text("4-bit").tag(4)
            Text("8-bit").tag(8)
        }
        .pickerStyle(.segmented)
        Picker("KV scheme", selection: $draft.kvQuantScheme) {
            Text("Auto").tag("")
            Text("Uniform").tag("uniform")
            Text("Polar").tag("polar")
            Text("Turbo").tag("turboquant")
        }
        .pickerStyle(.segmented)
        Stepper("KV group: \(draft.kvGroupSize)", value: $draft.kvGroupSize, in: 0...1_024, step: 8)
            .font(MereRunTheme.captionFont)
        Stepper(
            "Quantize after: \(draft.quantizedKVStart)",
            value: $draft.quantizedKVStart,
            in: 0...262_144,
            step: 128
        )
        .font(MereRunTheme.captionFont)

        HStack(spacing: MereRunTheme.Spacing.sm) {
            TextField("Catalog LoRA id or file", text: $draft.loraPath)
                .mereField()
            Button("Catalog…", action: onShowAdapters)
                .controlSize(.small)
            Button("Browse…") {
                let panel = NSOpenPanel()
                panel.allowedContentTypes = [.data]
                panel.allowsMultipleSelection = false
                panel.canChooseDirectories = false
                if panel.runModal() == .OK, let url = panel.url {
                    draft.loraPath = url.path
                }
            }
            .controlSize(.small)
        }
        if !draft.loraPath.isBlank {
            numberField("LoRA scale", value: $draft.loraScale)
        }

        TextField("Tools: write_file, shell_exec", text: $draft.tools)
            .mereField()
        TextField("Tool sandbox directory", text: $draft.sandboxDir)
            .mereField()
        Toggle("Tool loop", isOn: $draft.toolLoop)
            .font(MereRunTheme.captionFont)
        Toggle("Allow shell execution", isOn: $draft.allowShellExec)
            .font(MereRunTheme.captionFont)
        Toggle("Allow absolute tool paths", isOn: $draft.allowAbsoluteToolPaths)
            .font(MereRunTheme.captionFont)
        Toggle("Auto-approve tool calls", isOn: $draft.autoApproveTools)
            .font(MereRunTheme.captionFont)
        Toggle("Generation stats", isOn: $draft.stats)
            .font(MereRunTheme.captionFont)
        Toggle("Preflight only", isOn: $draft.preflight)
            .font(MereRunTheme.captionFont)
        if draft.preflight {
            Toggle("JSON preflight report", isOn: $draft.preflightJSON)
                .font(MereRunTheme.captionFont)
        }
        Toggle("Require installed model", isOn: $draft.requireInstalled)
            .font(MereRunTheme.captionFont)
    }

    @ViewBuilder
    private var videoOptions: some View {
        if videoFamily.isMiniMaxH3 {
            Text("Synchronized 24 fps video + 32 kHz stereo audio")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            Picker(
                "Weights",
                selection: Binding(
                    get: { draft.h3WeightMode ?? "auto" },
                    set: { draft.h3WeightMode = $0 }
                )
            ) {
                Text("Auto").tag("auto")
                Text("Quantized").tag("quantized")
                Text("BF16").tag("resident-bf16")
            }
            .pickerStyle(.segmented)
            Picker(
                "Denoise",
                selection: Binding(
                    get: { draft.h3AccelerationMode ?? "quality" },
                    set: { draft.h3AccelerationMode = $0 }
                )
            ) {
                Text("Exact").tag("quality")
                Text("Balanced").tag("balanced")
                Text("Maximum").tag("maximum")
            }
            .pickerStyle(.segmented)
            Text("Balanced and Maximum trade exact-seed trajectory fidelity for faster denoising.")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            Toggle(
                "Override adaptive schedule",
                isOn: Binding(
                    get: { draft.h3Steps != nil },
                    set: { draft.h3Steps = $0 ? 21 : nil }
                )
            )
            .font(MereRunTheme.captionFont)
            if draft.h3Steps != nil {
                Stepper(
                    "Schedule points \(draft.h3Steps ?? 21)",
                    value: Binding(
                        get: { draft.h3Steps ?? 21 },
                        set: { draft.h3Steps = $0 }
                    ),
                    in: 1...64
                )
                .font(MereRunTheme.captionFont)
            }
            Stepper("Frames \(draft.numFrames) · 17n+5", value: $draft.numFrames, in: 22...600, step: 17)
                .font(MereRunTheme.captionFont)
            Text("Frame rate is fixed at 24 fps.")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            if videoFamily == .miniMaxH3Ref2VA {
                compactH3References
            } else {
                videoAssetRow(title: "End keyframe", path: $draft.endImagePath, type: .image)
            }
        } else {
            if videoFamily == .ltx {
                Picker("Quality", selection: $draft.videoQuality) {
                    Text("Draft").tag(LTXVideoQuality.draft)
                    Text("Final").tag(LTXVideoQuality.final)
                }
                .pickerStyle(.segmented)

                Picker("Output", selection: $draft.videoOutputMode) {
                    Text("Video").tag(LTXVideoOutputMode.videoOnly)
                    Text("Audio + Video").tag(LTXVideoOutputMode.audioVideo)
                }
                .pickerStyle(.segmented)

                videoAssetRow(title: "Source audio", path: $draft.audioPath, type: .audio)
            }
            videoAssetRow(title: "End keyframe", path: $draft.endImagePath, type: .image)
        }

        Toggle("Use duration instead of frame count", isOn: $draft.useDuration)
            .font(MereRunTheme.captionFont)
        if draft.useDuration {
            numberField("Duration seconds", value: $draft.durationSeconds)
        }
        if videoFamily == .ltx, !draft.audioPath.isBlank {
            numberField("Audio start seconds", value: $draft.audioStartTime)
            numberField("Audio max seconds (0 = video)", value: $draft.audioMaxDuration)
            Stepper("A2V steps \(draft.a2vSteps)", value: $draft.a2vSteps, in: 1...100)
                .font(MereRunTheme.captionFont)
            numberField("A2V guidance", value: $draft.a2vGuidanceScale)
            numberField("Video CFG", value: $draft.videoCFGGuidanceScale)
            numberField("Audio CFG", value: $draft.audioCFGGuidanceScale)
            numberField("V2A guidance", value: $draft.v2aGuidanceScale)
        }
        if !draft.endImagePath.isBlank {
            numberField("End image strength", value: $draft.endImageStrength)
        }
        Toggle("Preflight only", isOn: $draft.preflight)
            .font(MereRunTheme.captionFont)
        if !videoFamily.isMiniMaxH3 {
            Toggle("Capture phase timings", isOn: $draft.timings)
                .font(MereRunTheme.captionFont)
        }
    }

    @ViewBuilder
    private var compactH3References: some View {
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
                    Button { moveH3Reference(index, by: 1) } label: { Image(systemName: "arrow.down") }
                        .disabled(index == references.count - 1)
                    Button(role: .destructive) { removeH3Reference(index) } label: { Image(systemName: "trash") }
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

    @ViewBuilder
    private var imageOptions: some View {
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
                    .controlSize(.small)
            }
            .padding(MereRunTheme.Spacing.sm)
            .background {
                RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                    .fill(MereRunTheme.accentSoft.opacity(0.55))
            }
        }

        HStack(spacing: MereRunTheme.Spacing.sm) {
            Text(draft.referenceImagePaths.isBlank
                ? "No extra references"
                : "\(imageReferencePaths.count) reference image\(imageReferencePaths.count == 1 ? "" : "s")")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            Spacer()
            Button("References…", action: chooseImageReferences)
                .controlSize(.small)
            if !draft.referenceImagePaths.isBlank {
                Button {
                    draft.referenceImagePaths = ""
                } label: {
                    Image(systemName: "xmark.circle.fill")
                }
                .buttonStyle(.borderless)
                .accessibilityLabel("Remove reference images")
            }
        }

        Toggle("Keep original aspect for one HiDream reference", isOn: $draft.keepOriginalAspect)
            .font(MereRunTheme.captionFont)

        HStack(spacing: MereRunTheme.Spacing.sm) {
            Text(draft.loraPath.isBlank
                ? "No LoRA adapter"
                : URL(fileURLWithPath: draft.loraPath).lastPathComponent)
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
                .lineLimit(1)
            Spacer()
            Button("LoRA…", action: chooseImageLoRA)
                .controlSize(.small)
            Button("Catalog…", action: onShowAdapters)
                .controlSize(.small)
        }
        TextField("LoRA catalog id or local path", text: $draft.loraPath)
            .mereField()
        numberField("LoRA scale", value: $draft.loraScale)

        Toggle("Expand into a structured JSON prompt", isOn: $draft.structuredPrompt)
            .font(MereRunTheme.captionFont)
        if draft.structuredPrompt {
            TextField("Structured prompt model", text: $draft.structuredPromptModel)
                .mereField()
            Stepper(
                "Prompt max tokens \(draft.structuredPromptMaxTokens)",
                value: $draft.structuredPromptMaxTokens,
                in: 128...16_384,
                step: 128
            )
            .font(MereRunTheme.captionFont)
        }

        DisclosureGroup("Krea tuning") {
            VStack(spacing: MereRunTheme.Spacing.sm) {
                numberField(
                    "Conditioning multiplier",
                    value: $draft.kreaConditioningMultiplier
                )
                TextField("Layer weights, comma-separated", text: $draft.kreaConditioningLayerWeights)
                    .mereField()
                Picker("Base quantization", selection: $draft.kreaBaseQuantizationBits) {
                    Text("Automatic").tag("")
                    Text("4-bit").tag("4")
                    Text("8-bit").tag("8")
                }
            }
            .padding(.top, MereRunTheme.Spacing.xs)
        }

        Toggle("Preflight only", isOn: $draft.preflight)
            .font(MereRunTheme.captionFont)
        if draft.preflight {
            Toggle("JSON preflight report", isOn: $draft.preflightJSON)
                .font(MereRunTheme.captionFont)
        }
        Toggle("Machine-readable progress", isOn: $draft.progressJSON)
            .font(MereRunTheme.captionFont)
    }

    @ViewBuilder
    private var musicOptions: some View {
        Button {
            onShowRealtimeMusic()
        } label: {
            HStack {
                Image(systemName: "waveform.path.ecg")
                    .foregroundStyle(MereRunTheme.accent)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Open realtime performance")
                        .font(.system(size: 12.5, weight: .semibold))
                    Text("Live playback, recording, MIDI, and prompt steering")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                }
                Spacer()
                Image(systemName: "chevron.right")
                    .foregroundStyle(MereRunTheme.textMuted)
            }
            .padding(MereRunTheme.Spacing.sm)
            .background {
                RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                    .fill(MereRunTheme.accentSoft.opacity(0.55))
            }
        }
        .buttonStyle(.plain)

        Picker("Quality", selection: $draft.musicQuality) {
            Text("Draft").tag("draft")
            Text("Song").tag("song")
            Text("Final").tag("final")
            Text("Edit").tag("edit")
        }
        .pickerStyle(.segmented)

        Picker("Task", selection: $draft.musicTask) {
            Text("Create").tag("text2music")
            Text("Cover").tag("cover")
            Text("No-FSQ").tag("cover-nofsq")
            Text("Repaint").tag("repaint")
            Text("Extract").tag("extract")
            Text("Lego").tag("lego")
            Text("Complete").tag("complete")
        }

        if draft.musicTask != "text2music" || draft.musicFlowEdit {
            videoAssetRow(title: "Source audio", path: $draft.musicSourceAudio, type: .audio)
            numberField("Cover strength", value: $draft.musicCoverStrength)
            numberField("Cover noise", value: $draft.musicCoverNoiseStrength)
        }

        HStack(spacing: MereRunTheme.Spacing.sm) {
            Text(draft.musicReferenceAudioPaths.isBlank
                ? "No timbre references"
                : "\(musicReferencePaths.count) timbre reference\(musicReferencePaths.count == 1 ? "" : "s")")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
                .lineLimit(1)
            Spacer()
            Button("References…", action: chooseMusicReferences)
                .controlSize(.small)
        }

        Picker("LM planning", selection: $draft.musicLMMode) {
            Text("Preset").tag("auto")
            Text("On").tag("use")
            Text("Off").tag("disable")
        }
        .pickerStyle(.segmented)
        Toggle("Analyze source metadata", isOn: $draft.musicAnalyzeSourceAudio)
            .font(MereRunTheme.captionFont)
        HStack(spacing: MereRunTheme.Spacing.sm) {
            numberField("LM temperature", value: $draft.musicLMTemperature)
            numberField(
                "LM repetition penalty",
                value: $draft.musicLMRepetitionPenalty
            )
            numberField("LM CFG scale", value: $draft.musicLMCFGScale)
        }
        TextField("LM negative prompt", text: $draft.musicLMNegativePrompt)
            .textFieldStyle(.roundedBorder)
        Toggle("Instrumental", isOn: $draft.musicInstrumental)
            .font(MereRunTheme.captionFont)

        Toggle("Set exact duration", isOn: $draft.useDuration)
            .font(MereRunTheme.captionFont)
        if draft.useDuration {
            numberField("Duration seconds", value: $draft.durationSeconds)
        }
        Toggle("Override preset steps", isOn: $draft.musicOverrideSteps)
            .font(MereRunTheme.captionFont)
        if draft.musicOverrideSteps {
            Stepper("Steps \(draft.steps)", value: $draft.steps, in: 1...200)
                .font(MereRunTheme.captionFont)
        }
        Stepper(
            "Candidates \(draft.musicCandidates == 0 ? "preset" : String(draft.musicCandidates))",
            value: $draft.musicCandidates,
            in: 0...16
        )
        .font(MereRunTheme.captionFont)
        Toggle("Keep all ranked candidates", isOn: $draft.musicKeepCandidates)
            .font(MereRunTheme.captionFont)

        Toggle("Flow edit source toward prompt", isOn: $draft.musicFlowEdit)
            .font(MereRunTheme.captionFont)
        if draft.musicFlowEdit {
            TextField("Source caption", text: $draft.musicSourceCaption)
                .mereField()
            TextField("Source lyrics", text: $draft.musicSourceLyrics, axis: .vertical)
                .lineLimit(1...4)
                .mereField()
        }

        DisclosureGroup("Adapters & delivery") {
            VStack(spacing: MereRunTheme.Spacing.sm) {
                HStack(spacing: MereRunTheme.Spacing.sm) {
                    Text(draft.musicAdapterPaths.isBlank
                        ? "No ACE-Step adapters"
                        : "\(musicAdapterPaths.count) adapter\(musicAdapterPaths.count == 1 ? "" : "s")")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                    Spacer()
                    Button("Adapters…", action: chooseMusicAdapters)
                        .controlSize(.small)
                    Button("Catalog…", action: onShowAdapters)
                        .controlSize(.small)
                }
                Picker("Adapter kind", selection: $draft.musicAdapterKind) {
                    Text("Auto").tag("auto")
                    Text("LoRA").tag("lora")
                    Text("LoKr").tag("lokr")
                }
                .pickerStyle(.segmented)
                TextField("Adapter scales, one per line", text: $draft.musicAdapterScales)
                    .mereField()
                videoAssetRow(title: "LRC lyrics", path: $draft.musicLRCFile, type: .plainText)
                TextField("Stems (Drums,Bass,Vocals)", text: $draft.musicStems)
                    .mereField()
                TextField("DAW bundle directory", text: $draft.musicDAWBundle)
                    .mereField()
                Picker("WAV format", selection: $draft.musicExportFormat) {
                    Text("PCM 16").tag("pcm16")
                    Text("PCM 24").tag("pcm24")
                    Text("Float 32").tag("float32")
                }
                .pickerStyle(.segmented)
                Toggle("Skip recipe sidecar", isOn: $draft.musicNoRecipe)
                    .font(MereRunTheme.captionFont)
            }
            .padding(.top, MereRunTheme.Spacing.xs)
        }
    }

    private func numberField(_ label: String, value: Binding<Double>) -> some View {
        HStack {
            Text(label)
                .font(MereRunTheme.captionFont)
            Spacer()
            TextField(label, value: value, format: .number)
                .frame(width: 90)
                .mereField(cornerRadius: MereRunTheme.Radius.sm)
        }
    }

    private func videoAssetRow(title: String, path: Binding<String>, type: UTType) -> some View {
        HStack(spacing: MereRunTheme.Spacing.sm) {
            Text(path.wrappedValue.isEmpty
                ? "No \(title.lowercased())"
                : URL(fileURLWithPath: path.wrappedValue).lastPathComponent)
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
                .lineLimit(1)
            Spacer()
            Button(title) {
                let panel = NSOpenPanel()
                panel.allowedContentTypes = [type]
                panel.allowsMultipleSelection = false
                panel.canChooseDirectories = false
                if panel.runModal() == .OK, let url = panel.url {
                    path.wrappedValue = url.path
                    if mode == .video && type == .audio {
                        draft.videoQuality = .final
                        draft.videoOutputMode = .audioVideo
                    }
                }
            }
            .controlSize(.small)
            if !path.wrappedValue.isEmpty {
                Button {
                    path.wrappedValue = ""
                } label: {
                    Image(systemName: "xmark.circle.fill")
                }
                .buttonStyle(.borderless)
                .accessibilityLabel("Remove \(title.lowercased())")
            }
        }
    }

    private func normalizeMiniMaxH3Draft() {
        guard mode == .video, videoFamily.isMiniMaxH3 else { return }
        draft.fps = 24
        draft.width = max(32, (draft.width / 32) * 32)
        draft.height = max(32, (draft.height / 32) * 32)
        draft.numFrames = StudioVideoModelFamily.alignedMiniMaxH3FrameCount(draft.numFrames)
        draft.audioPath = ""
        draft.timings = false
        draft.timingsOutputPath = ""
        if videoFamily == .miniMaxH3Ref2VA {
            draft.inputPath = ""
            draft.endImagePath = ""
        } else {
            draft.h3ReferenceInputs = []
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

    private var imageReferencePaths: [String] {
        draft.referenceImagePaths.components(separatedBy: .newlines)
            .flatMap { $0.split(separator: ",", omittingEmptySubsequences: true) }
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    private var musicReferencePaths: [String] {
        separatedPaths(draft.musicReferenceAudioPaths)
    }

    private var musicAdapterPaths: [String] {
        separatedPaths(draft.musicAdapterPaths)
    }

    private var hasImageOutpaint: Bool {
        [
            draft.imageOutpaintTop,
            draft.imageOutpaintRight,
            draft.imageOutpaintBottom,
            draft.imageOutpaintLeft,
        ].contains(where: { $0 > 0 })
    }

    private func separatedPaths(_ raw: String) -> [String] {
        raw.components(separatedBy: .newlines)
            .flatMap { $0.split(separator: ",", omittingEmptySubsequences: true) }
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    private func chooseImageReferences() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        if panel.runModal() == .OK {
            draft.referenceImagePaths = panel.urls.map(\.path).joined(separator: "\n")
        }
    }

    private func chooseImageLoRA() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.data]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        if panel.runModal() == .OK, let url = panel.url {
            draft.loraPath = url.path
        }
    }

    private func chooseMusicReferences() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.audio]
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        if panel.runModal() == .OK {
            draft.musicReferenceAudioPaths = panel.urls.map(\.path).joined(separator: "\n")
        }
    }

    private func chooseMusicAdapters() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.data]
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        if panel.runModal() == .OK {
            draft.musicAdapterPaths = panel.urls.map(\.path).joined(separator: "\n")
        }
    }

    private var secondaryLabel: String? {
        switch mode {
        case .createImage: return "Negative prompt"
        case .chat, .code: return "System"
        case .speak: return "Voice"
        case .music: return "Lyrics"
        case .video: return "Negative prompt"
        default: return nil
        }
    }
}
