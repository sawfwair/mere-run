import AppKit
import MereRunContract
import StudioKit
import SwiftUI
import UniformTypeIdentifiers

/// The composer pinned under the canvas: the attachment well, the prompt, and a strip of the
/// mode's essential parameters as chips, with Run on the right. Slots and chips come from
/// `StudioComposerSchema.swift`; the remaining depth lives in the inspector column, which binds
/// the same draft.
struct StudioComposer: View {
    let mode: StudioMode
    @Binding var draft: StudioDraft
    let isRunning: Bool
    let queuedCount: Int
    let readiness: ModelReadinessState
    var sendBlocked: Bool = false
    /// Every row of `model list`, installed or not; the model chip filters it to the mode.
    let modelInventory: [StudioModelInventoryRow]
    @Environment(\.studioModelTitles) private var titles
    /// The seed of the mode's most recent run, for "Reuse last".
    var lastSeed: String?
    var promptFocus: FocusState<Bool>.Binding
    let onRun: () -> Void
    let onStop: () -> Void
    let onShowModels: () -> Void
    /// Puts a prompt from the page's history in the field (↑, or Recent prompts) as an undo step.
    let onRecallPrompt: (String) -> Void
    /// "Run variations" beside Run; nil hides it (a command without a seed, a conversation).
    var onRunVariations: ((StudioVariationCount) -> Void)?
    /// Whether the composer carries the scope note: only while no side column is open. The
    /// inspector shows it at its top, beside the controls it explains, and the Command view as
    /// its "Not sent" line.
    var showsScopeNote = true

    @EnvironmentObject private var controller: MereRunController
    @Environment(\.studioScopeSource) private var scopeSource
    @Environment(\.studioLibraryItems) private var libraryItems
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
            if mode.showsAttachmentWell(for: draft, source: scopeSource) {
                attachmentWell
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }
            promptEntry
            chipStrip
            if showsScopeNote, let notice = StudioInspectorSchema.notice(for: mode, draft: draft, source: scopeSource) {
                StudioScopeNote(notice: notice)
            }
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
        .animation(MereRunTheme.Motion.standard, value: mode.showsAttachmentWell(for: draft, source: scopeSource))
    }

    // MARK: - Attachment well

    private var visibleSlots: [StudioAttachmentSlot] {
        mode.attachmentSlots(for: draft, source: scopeSource).filter { !$0.isTransient || $0.isFilled(in: draft) }
    }

    private var attachmentWell: some View {
        HStack(alignment: .center, spacing: 8) {
            ForEach(visibleSlots) { slot in
                StudioAttachmentSlotView(slot: slot, draft: $draft, onPick: { pickFiles(for: slot) })
            }
            VStack(alignment: .leading, spacing: 1) {
                ForEach(visibleSlots) { slot in
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

    private func pickFiles(for slot: StudioAttachmentSlot) {
        StudioAttachmentPicker.pick(for: slot, into: &draft)
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
            .studioAttachmentPasteKey(isActive: promptFocus.wrappedValue) {
                StudioAttachmentPaste.paste(
                    into: &draft, slots: mode.attachmentSlots(for: draft, source: scopeSource), allowsText: true
                )
            }
            .studioPromptHistoryKeys(
                isActive: promptFocus.wrappedValue, history: promptHistory, text: draft.prompt, recall: onRecallPrompt
            )
            .accessibilityLabel(mode.promptPlaceholder)
        }
    }

    /// The prompts this page has run, newest first.
    private var promptHistory: [String] {
        mode == .listen ? [] : StudioPromptHistory.prompts(for: mode.task, in: libraryItems)
    }

    // MARK: - Chip strip

    private var chipStrip: some View {
        HStack(alignment: .center, spacing: 6) {
            ForEach(mode.composerChips(for: draft, source: scopeSource)) { chip in
                if let value = chip.fixedValue {
                    fixedChip(chip.kind, value: value, family: chip.fixedBy)
                } else {
                    self.chip(for: chip.kind)
                }
            }
            Spacer(minLength: 8)
            HStack(spacing: 8) {
                if !promptHistory.isEmpty {
                    StudioRecentPromptsMenu(prompts: promptHistory, onPick: recallFromMenu)
                }
                if showsPaperclip { paperclipButton }
                // A batch already runs once per file; variations of a batch are not offered.
                if let onRunVariations, mode.attachmentSlots(for: draft, source: scopeSource).batchRunCount(in: draft) == nil {
                    StudioVariationsRunButton(isEnabled: sendEnabled, disabledReason: sendHelp, run: onRunVariations)
                }
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

    /// A chip whose value the model's family fixes: shown as it runs, without a menu.
    private func fixedChip(_ kind: StudioComposerChipKind, value: String, family: String?) -> some View {
        let title = StudioComposerPresets.fixedTitle(kind, value: value)
        let reason = family.map { "\($0) always runs \(title.lowercased())" } ?? "This model always runs \(title.lowercased())"
        return StudioComposerChipLabel(title: title, leadingSystemImage: "lock", menu: false)
            .fixedSize()
            .help(reason)
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(kind.accessibilityTitle)
            .accessibilityValue("\(title). \(reason).")
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

    /// MiniMax-H3 picks its steps from an adaptive schedule; the chip then edits the schedule's
    /// override, which is what the command line carries, not the step slider other models use.
    private var usesAdaptiveSchedule: Bool {
        guard mode == .video, let scope = scopeSource.scope(mode: mode, draft: draft), scope.family != nil else { return false }
        return scope.allows(CommandFlags.VideoGenerate.h3Acceleration)
    }

    private var stepsChip: some View {
        chipMenu(
            title: usesAdaptiveSchedule
                ? draft.h3Steps.map { $0 == 1 ? "1 step" : "\($0) steps" } ?? "Adaptive steps"
                : StudioComposerPresets.stepsTitle(draft, mode: mode),
            accessibilityLabel: "Steps",
            kind: .steps
        ) {
            if usesAdaptiveSchedule {
                Toggle(isOn: Binding(get: { draft.h3Steps == nil }, set: { _ in draft.h3Steps = nil })) {
                    Text("Adaptive schedule")
                }
                Divider()
            }
            if mode == .music {
                Toggle(isOn: Binding(get: { !draft.musicOverrideSteps }, set: { _ in draft.musicOverrideSteps = false })) {
                    Text("Preset steps")
                }
                Divider()
            }
            ForEach(StudioComposerPresets.steps(for: mode), id: \.self) { steps in
                Toggle(isOn: Binding(
                    get: { usesAdaptiveSchedule ? draft.h3Steps == steps : stepsAreExplicit && draft.steps == steps },
                    set: { _ in setSteps(steps) }
                )) {
                    Text(steps == 1 ? "1 step" : "\(steps) steps")
                }
            }
            Divider()
            Button("Custom steps…") { editingChip = .steps }
        } editor: {
            numberField(
                "Steps",
                value: Binding(get: { usesAdaptiveSchedule ? draft.h3Steps ?? 21 : draft.steps }, set: { setSteps($0) }),
                range: usesAdaptiveSchedule ? 1...64 : 1...200
            )
        }
    }

    private var stepsAreExplicit: Bool {
        mode != .music || draft.musicOverrideSteps
    }

    private func setSteps(_ steps: Int) {
        if usesAdaptiveSchedule {
            draft.h3Steps = steps
            return
        }
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
        switch StudioCommandAdapter.capabilityRequirement(for: .readImage, draft: candidateDraft, source: scopeSource) {
        case .unavailable(let message):
            return message
        case .managedModel(let modelID):
            return controller.modelCapabilitiesByID[modelID]?.unavailableMessage(titles: titles)
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
        StudioModelChip(
            scope: StudioModelScope(mode: mode, readImageAction: draft.readImageAction, source: scopeSource),
            model: $draft.model,
            modelInventory: modelInventory,
            readiness: readiness,
            onShowModels: onShowModels
        )
    }

    // MARK: - Right cluster

    /// A prompt picked from Recent prompts: into the field, with the field focused to edit it.
    private func recallFromMenu(_ prompt: String) {
        onRecallPrompt(prompt)
        promptFocus.wrappedValue = true
    }

    /// Only a collapsed well (Chat's per-turn image) needs the paperclip; declared slots pick
    /// from the well itself.
    private var showsPaperclip: Bool {
        !mode.attachmentSlots(for: draft, source: scopeSource).isEmpty
            && !mode.showsAttachmentWell(for: draft, source: scopeSource)
    }

    private var paperclipButton: some View {
        Button {
            if let slot = mode.attachmentSlots(for: draft, source: scopeSource).first { pickFiles(for: slot) }
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
                    .font(.caption.weight(.bold))
                    .foregroundStyle(MereRunTheme.textPrimary)
            }
            .frame(width: Metrics.sendDiameter, height: Metrics.sendDiameter)
        }
        .buttonStyle(.plain)
        .help(queuedCount == 0 ? "Stop (⌘.) · ⌘↩ queues another run" : "Stop (⌘.) · \(queuedCount) queued")
        .accessibilityLabel("Stop current run")
    }

    @ViewBuilder
    private var sendButton: some View {
        if let count = mode.attachmentSlots(for: draft, source: scopeSource).batchRunCount(in: draft) {
            StudioBatchRunButton(count: count, isEnabled: sendEnabled,
                                 blockedReason: sendEnabled ? nil : sendHelp, action: onRun)
        } else {
            singleSendButton
        }
    }

    private var singleSendButton: some View {
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
        if readiness.blocksRun { return readiness.message(titles: titles) }
        return "Run (⌘↩)"
    }

    /// Spoken explanation of why Run is disabled, so the blocked state is not color-only.
    private var accessibilityRunHint: String {
        if sendBlocked { return "Waiting for the current reply to finish" }
        if readiness.blocksRun { return readiness.message(titles: titles) }
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
                .font(.caption.weight(.medium))
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
/// Accepts a drop (a Finder file or a Library row), a paste (⌘V while focused: a copied file,
/// picture, or sound, routed like a drop), and a click: straight to the open panel, or — when
/// the Library holds a file the slot takes — a menu of From Disk…, From Library…, and, on an
/// audio slot, Record…, which files the recording with the task's domain. A batching slot that
/// holds several files draws their stack instead (`StudioBatchWellTile`).
struct StudioAttachmentSlotView<Draft: StudioAttachmentDraft>: View {
    let slot: StudioAttachmentSlot
    @Binding var draft: Draft
    /// The disk choice: the caller's open panel into the draft.
    let onPick: () -> Void

    @Environment(\.studioTaskScope) private var taskScope
    @Environment(\.studioLibraryItems) private var libraryItems
    @State private var isDropTargeted = false
    @State private var hovering = false
    @State private var isRecording = false
    @State private var isPickingFromLibrary = false

    private var recordingDomain: StudioDomain {
        StudioTask(rawValue: taskScope)?.domain ?? .audio
    }

    private static var side: CGFloat { 48 }
    private static var cornerRadius: CGFloat { MereRunTheme.Radius.base }

    private var paths: [String] { slot.paths(in: draft) }
    private var isFilled: Bool { !paths.isEmpty }

    private var target: StudioAttachTarget { StudioAttachTarget(slot: slot, draft: $draft) }

    private var hasLibraryChoices: Bool {
        StudioLibraryInputs.hasCandidates(in: libraryItems, for: target.requirement)
    }

    @ViewBuilder
    var body: some View {
        if slot.isBatched(in: draft) {
            StudioBatchWellTile(slot: slot, draft: $draft, onAddFromDisk: onPick)
        } else {
            singleTile
        }
    }

    private var singleTile: some View {
        StudioAttachMenu(
            target: target,
            chooseFromDisk: onPick,
            isPickingFromLibrary: $isPickingFromLibrary,
            arrowEdge: .top
        ) { _ in
            tile
        } extraItems: {
            if slot.canRecord {
                Button { isRecording = true } label: { Label("Record…", systemImage: "mic") }
            }
        }
        .overlay(alignment: .topTrailing) {
            if isFilled, hovering {
                Button {
                    slot.clear(in: &draft)
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.callout)
                        .symbolRenderingMode(.palette)
                        .foregroundStyle(MereRunTheme.surface, MereRunTheme.textPrimary)
                }
                .buttonStyle(.plain)
                .padding(3)
                .help("Remove")
                .accessibilityLabel("Remove \(slot.label.lowercased())")
            }
        }
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
        .onPasteCommand(of: [.fileURL, .image, .audio]) { _ in
            StudioAttachmentPaste.paste(into: &draft, slots: [slot], allowsText: false)
        }
        .contextMenu {
            Button("Choose from Disk…", action: onPick)
            if hasLibraryChoices {
                Button("Choose from Library…") { isPickingFromLibrary = true }
            }
            if slot.canRecord {
                Button("Record…") { isRecording = true }
            }
            if isFilled {
                Button("Reveal in Finder") {
                    NSWorkspace.shared.activateFileViewerSelecting(paths.map { URL(fileURLWithPath: $0) })
                }
                Button("Remove") { slot.clear(in: &draft) }
            }
        }
        .popover(isPresented: $isRecording, arrowEdge: .bottom) {
            StudioAudioRecordingPopover(domain: recordingDomain) { url in
                slot.attach([url], to: &draft)
            }
        }
        .help(isFilled ? paths.joined(separator: "\n") : helpText)
        .accessibilityLabel(slot.label)
        .accessibilityValue(isFilled ? slot.caption(in: draft) : "Empty")
        .accessibilityHint(hasLibraryChoices
            ? "Drop a file, paste with Command-V, or click to choose from disk or from the Library"
            : "Drop a file, paste with Command-V, or click to choose")
    }

    private var helpText: String {
        let source = hasLibraryChoices ? " from disk or the Library" : ""
        let record = slot.canRecord ? "; right-click to record" : ""
        return "Drop, paste, or click to add \(slot.label.lowercased())\(source)\(record)"
    }

    private var tile: some View {
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
        .contentShape(RoundedRectangle(cornerRadius: Self.cornerRadius))
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
        if slot.acceptedTypes.contains(.folder) {
            kindGlyph("folder")
        } else {
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
    }

    private func kindGlyph(_ systemImage: String) -> some View {
        ZStack {
            MereRunTheme.surfaceRaised
            Image(systemName: systemImage)
                .font(.system(size: 16, weight: .medium))
                .foregroundStyle(MereRunTheme.accent)
        }
    }
}
