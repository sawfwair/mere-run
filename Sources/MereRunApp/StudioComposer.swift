import AppKit
import MereRunContract
import SwiftUI
import UniformTypeIdentifiers

/// The floating command bar at the bottom of the canvas: one field, one run button, and the
/// depth (attachment, model, options) folded into quiet affordances around it.
struct StudioComposer: View {
    let mode: StudioMode
    var isCompact = false
    @Binding var draft: StudioDraft
    @Binding var showOptions: Bool
    let isRunning: Bool
    let queuedCount: Int
    let readiness: ModelReadinessState
    var sendBlocked: Bool = false
    let installedModels: [StudioModelInventoryRow]
    var promptFocus: FocusState<Bool>.Binding
    let onRun: () -> Void
    let onStop: () -> Void
    let onAttach: () -> Void
    let onPaste: () -> Void
    let onShowModels: () -> Void
    let onShowAdapters: () -> Void
    let onShowRealtimeMusic: () -> Void

    /// Whether the clipboard currently holds an image, so the paste affordance only
    /// appears when it can actually do something (refreshed on appear + on focus).
    @State private var clipboardHasImage = false

    var body: some View {
        VStack(spacing: MereRunTheme.Spacing.xs) {
            if !draft.inputPath.isBlank {
                attachmentChip
                    .transition(.opacity.combined(with: .move(edge: .bottom)))
            }

            composerBar

            if mode.requiresAttachment && draft.inputPath.isBlank {
                Label(attachmentHint, systemImage: "arrow.down.doc")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                    .padding(.leading, 4)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .animation(MereRunTheme.Motion.standard, value: draft.inputPath.isBlank)
        .onAppear(perform: refreshClipboardImageState)
        .onChange(of: promptFocus.wrappedValue) { _, focused in
            if focused { refreshClipboardImageState() }
        }
    }

    private func refreshClipboardImageState() {
        clipboardHasImage = NSPasteboard.general.canReadObject(forClasses: [NSImage.self], options: nil)
    }

    private var attachmentChip: some View {
        HStack(spacing: MereRunTheme.Spacing.xs) {
            attachmentThumbnail
                .frame(width: 30, height: 26)
                .background(MereRunTheme.surface)
                .clipShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.sm))
                .overlay {
                    RoundedRectangle(cornerRadius: MereRunTheme.Radius.sm)
                        .strokeBorder(MereRunTheme.border.opacity(0.6), lineWidth: 1)
                }

            Text(URL(fileURLWithPath: draft.inputPath).lastPathComponent)
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textSecondary)
                .lineLimit(1)
                .truncationMode(.middle)

            Button {
                draft.inputPath = ""
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .font(.system(size: 12))
                    .padding(2)
            }
            .buttonStyle(.mereIcon(tint: MereRunTheme.textMuted))
            .help("Remove attachment")
            .accessibilityLabel("Remove attachment")
        }
        .padding(.horizontal, MereRunTheme.Spacing.xs)
        .padding(.vertical, 5)
        .background {
            Capsule()
                .fill(MereRunTheme.surface.opacity(0.85))
                .overlay {
                    Capsule().strokeBorder(MereRunTheme.border.opacity(0.6), lineWidth: 1)
                }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder
    private var attachmentThumbnail: some View {
        let url = URL(fileURLWithPath: draft.inputPath)
        if StudioOutputFileKind.classify(url) == .image {
            StudioAsyncImagePreview(
                url: url,
                maxPixelSize: 120,
                contentMode: .fill,
                fallbackSystemImage: "photo"
            )
        } else {
            Image(systemName: attachmentIcon(for: url))
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(MereRunTheme.accent)
        }
    }

    private var composerBar: some View {
        Group {
            if isCompact {
                compactComposerContent
            } else {
                regularComposerContent
            }
        }
        .padding(.horizontal, isCompact ? MereRunTheme.Spacing.sm : MereRunTheme.Spacing.md)
        .padding(.vertical, MereRunTheme.Spacing.sm)
        .background {
            RoundedRectangle(cornerRadius: MereRunTheme.Radius.xxl)
                .fill(MereRunTheme.surface)
                .overlay {
                    RoundedRectangle(cornerRadius: MereRunTheme.Radius.xxl)
                        .strokeBorder(MereRunTheme.border.opacity(0.75), lineWidth: 1)
                }
                .mereShadow(radius: 18, y: 8)
        }
        .mereFocusRing(promptFocus.wrappedValue, cornerRadius: MereRunTheme.Radius.xxl)
    }

    private var regularComposerContent: some View {
        HStack(spacing: MereRunTheme.Spacing.sm) {
            attachmentControls
            promptEntry

            HStack(spacing: 6) {
                modelMenu
                optionsButton
                if isRunning { stopButton }
                sendButton
            }
        }
    }

    private var compactComposerContent: some View {
        VStack(spacing: 5) {
            HStack(spacing: MereRunTheme.Spacing.xs) {
                promptEntry
                sendButton
            }

            HStack(spacing: 4) {
                attachmentControls
                modelMenu
                Spacer(minLength: 0)
                optionsButton
                if isRunning { stopButton }
            }
        }
    }

    private var attachmentControls: some View {
        HStack(spacing: 2) {
            Button(action: onAttach) {
                Image(systemName: mode.requiresAttachment ? "paperclip.badge.ellipsis" : "paperclip")
                    .font(.system(size: 16, weight: .semibold))
                    .frame(width: 30, height: 30)
            }
            .buttonStyle(.mereIcon)
            .disabled(mode.acceptedTypes.isEmpty)
            .opacity(mode.acceptedTypes.isEmpty ? 0.35 : 1)
            .help(mode.requiresAttachment ? "Attach required input" : "Attach reference")
            .accessibilityLabel(mode.requiresAttachment ? "Attach required input" : "Attach reference")

            if mode.acceptedTypes.contains(.image), clipboardHasImage {
                Button(action: onPaste) {
                    Image(systemName: "doc.on.clipboard")
                        .font(.system(size: 14, weight: .semibold))
                        .frame(width: 30, height: 30)
                }
                .buttonStyle(.mereIcon)
                .help("Paste image from clipboard (⌘V)")
                .accessibilityLabel("Paste image from clipboard")
                .transition(.opacity.combined(with: .scale(scale: 0.8)))
            }
        }
        .animation(MereRunTheme.Motion.quick, value: clipboardHasImage)
    }

    @ViewBuilder
    private var promptEntry: some View {
        if mode == .listen {
            Text(mode.promptPlaceholder)
                .font(.system(size: 15, weight: .medium))
                .foregroundStyle(MereRunTheme.textMuted)
                .frame(maxWidth: .infinity, alignment: .leading)
        } else {
            TextField(mode.promptPlaceholder, text: $draft.prompt, axis: .vertical)
                .textFieldStyle(.plain)
                .font(.system(size: 15, weight: .medium))
                .lineLimit(1...5)
                .focused(promptFocus)
                .onSubmit(onRun)
        }
    }

    private var optionsButton: some View {
        Button {
            showOptions.toggle()
        } label: {
            Image(systemName: "slider.horizontal.3")
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(showOptions ? MereRunTheme.accent : MereRunTheme.textSecondary)
                .frame(width: 30, height: 30)
        }
        .buttonStyle(.mereIcon)
        .help("Options")
        .accessibilityLabel("Options")
        .popover(isPresented: $showOptions, arrowEdge: .top) {
            StudioOptionsPanel(
                mode: mode,
                draft: $draft,
                onShowAdapters: onShowAdapters,
                onShowRealtimeMusic: onShowRealtimeMusic
            )
        }
    }

    private var stopButton: some View {
        Button(action: onStop) {
            Image(systemName: "stop.fill")
                .font(.system(size: 12, weight: .bold))
                .frame(width: 30, height: 30)
        }
        .buttonStyle(.mereIcon(tint: MereRunTheme.red))
        .help("Stop current run (⌘.)")
        .accessibilityLabel("Stop current run")
    }

    private var sendButton: some View {
        Button(action: onRun) {
            ZStack {
                Circle()
                    .fill(sendEnabled ? MereRunTheme.accent : MereRunTheme.surfaceRaised)
                Image(systemName: "arrow.up")
                    .font(.system(size: 14, weight: .bold))
                    .foregroundStyle(sendEnabled ? MereRunTheme.background : MereRunTheme.textMuted)
            }
            .frame(width: 34, height: 34)
            .overlay(alignment: .topTrailing) {
                if queuedCount > 0 {
                    Text("\(queuedCount)")
                        .font(.system(size: 9, weight: .bold))
                        .foregroundStyle(MereRunTheme.background)
                        .padding(.horizontal, 4)
                        .frame(height: 13)
                        .background { Capsule().fill(MereRunTheme.yellow) }
                        .offset(x: 4, y: -3)
                        .accessibilityHidden(true)
                }
            }
        }
        .buttonStyle(.plain)
        .disabled(!sendEnabled)
        .help(sendHelp)
        .accessibilityLabel(isRunning ? "Queue run" : "Run")
        .accessibilityHint(accessibilityRunHint)
        .keyboardShortcut(.return, modifiers: .command)
    }

    private var sendEnabled: Bool {
        !readiness.blocksRun && !sendBlocked
    }

    private var sendHelp: String {
        if sendBlocked { return "Waiting for the current reply…" }
        if readiness.blocksRun { return readiness.message }
        if isRunning {
            return queuedCount == 0 ? "Queue run" : "Queue run (\(queuedCount) waiting)"
        }
        return "Run (⌘↩)"
    }

    /// Spoken explanation of why Run is disabled, so the blocked state is not color-only.
    private var accessibilityRunHint: String {
        if sendBlocked { return "Waiting for the current reply to finish" }
        if readiness.blocksRun { return readiness.message }
        return ""
    }

    // MARK: - Model quick-picker

    private var modelMenu: some View {
        Menu {
            if installedModels.isEmpty {
                Text("No local models yet")
            } else {
                ForEach(groupedModelCategories, id: \.category) { group in
                    Section(group.category.capitalized) {
                        ForEach(group.rows) { row in
                            Button {
                                draft.model = row.id
                            } label: {
                                if row.id == draft.model {
                                    Label(row.id, systemImage: "checkmark")
                                } else {
                                    Text(row.id)
                                }
                            }
                        }
                    }
                }
            }
            Divider()
            Button("Use mode default") { draft.model = "" }
                .disabled(draft.model.isBlank)
            Button("Browse Models…", action: onShowModels)
        } label: {
            HStack(spacing: 5) {
                Image(systemName: "cpu")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(MereRunTheme.accent)
                Text(modelLabel)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(MereRunTheme.textSecondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .frame(maxWidth: 130)
                    .fixedSize(horizontal: true, vertical: false)
                Image(systemName: "chevron.up.chevron.down")
                    .font(.system(size: 7.5, weight: .semibold))
                    .foregroundStyle(MereRunTheme.textMuted)
            }
            .padding(.horizontal, 8)
            .frame(height: 26)
            .background {
                Capsule()
                    .fill(MereRunTheme.surfaceRaised.opacity(0.8))
                    .overlay {
                        Capsule().strokeBorder(MereRunTheme.border.opacity(0.55), lineWidth: 1)
                    }
            }
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .help(rawModelID.isEmpty ? "Auto — the mode's default model" : "Model: \(rawModelID)")
        .accessibilityLabel("Model")
        .accessibilityValue(rawModelID.isEmpty ? "Automatic" : rawModelID)
    }

    /// The resolved model id for this run: the explicit draft model, else the
    /// mode's template default, else empty (Auto).
    private var rawModelID: String {
        let current = draft.model.trimmingCharacters(in: .whitespacesAndNewlines)
        if !current.isEmpty { return current }
        return CommandCatalog.template(id: mode.defaultTemplateID)?.defaultModel ?? ""
    }

    private var modelLabel: String {
        rawModelID.isEmpty ? "Auto" : Self.displayModelName(rawModelID)
    }

    /// A human-facing label for a model id: drop the modality/category prefix and
    /// title-case the distinctive remainder ("text-agent-deepseek-v4-flash" →
    /// "Deepseek V4 Flash"). The exact id stays in the pill's tooltip, so the
    /// friendly name never hides what the CLI actually expects.
    static func displayModelName(_ id: String) -> String {
        let leaf = id.components(separatedBy: "/").last ?? id
        let prefixes = [
            "text-chat-", "text-agent-", "text-embed-", "text-code-",
            "image-", "video-", "music-", "sfx-", "speech-", "embed-", "vision-", "text-"
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

    private var groupedModelCategories: [(category: String, rows: [StudioModelInventoryRow])] {
        Dictionary(grouping: installedModels, by: \.category)
            .map { (category: $0.key, rows: $0.value.sorted { $0.id < $1.id }) }
            .sorted { $0.category < $1.category }
    }

    private var attachmentHint: String {
        switch mode {
        case .listen: return "Attach an audio file to begin — or drop it anywhere"
        case .track: return "Attach a video to begin — or drop it anywhere"
        default: return "Attach an image to begin — or drop it anywhere"
        }
    }

    private func attachmentIcon(for url: URL) -> String {
        switch StudioOutputFileKind.classify(url) {
        case .audio: return "waveform"
        case .video: return "film"
        default: return "doc"
        }
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

                if mode == .readImage {
                    Picker("Task", selection: $draft.readImageAction) {
                        ForEach(StudioReadImageAction.allCases) { action in
                            Text(action.title)
                                .tag(action)
                                .disabled(readImageActionUnavailableMessage(action) != nil)
                                .help(readImageActionUnavailableMessage(action) ?? action.title)
                        }
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                }

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

                VStack(alignment: .leading, spacing: 4) {
                    Text("Model")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                    TextField("Model", text: $draft.model)
                        .mereField()
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

                if [.createImage, .video].contains(mode) {
                    HStack(spacing: MereRunTheme.Spacing.sm) {
                        Stepper("Width \(draft.width)", value: $draft.width, in: 64...4096, step: 64)
                        Stepper("Height \(draft.height)", value: $draft.height, in: 64...4096, step: 64)
                    }
                    .font(MereRunTheme.captionFont)
                }

                if [.createImage, .sfx].contains(mode) {
                    Stepper("Steps \(draft.steps)", value: $draft.steps, in: 1...80, step: 1)
                        .font(MereRunTheme.captionFont)
                }

                if mode == .sfx {
                    HStack {
                        Text("Duration seconds")
                            .font(MereRunTheme.captionFont)
                        Spacer()
                        TextField("Seconds", value: $draft.durationSeconds, format: .number)
                            .frame(width: 80)
                            .mereField(cornerRadius: MereRunTheme.Radius.sm)
                    }
                }

                if [.createImage, .music, .video, .sfx].contains(mode) {
                    HStack {
                        Text("Seed")
                            .font(MereRunTheme.captionFont)
                        Spacer()
                        TextField("Random", text: $draft.seed)
                            .frame(width: 110)
                            .mereField(cornerRadius: MereRunTheme.Radius.sm)
                    }
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
        }

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

    private func readImageActionUnavailableMessage(_ action: StudioReadImageAction) -> String? {
        guard mode == .readImage else { return nil }
        var candidateDraft = draft
        candidateDraft.readImageAction = action
        let requirement = StudioCommandAdapter.capabilityRequirement(
            for: .readImage,
            draft: candidateDraft
        )

        switch requirement {
        case .unavailable(let message):
            return message
        case .managedModel(let modelID):
            return controller.modelCapabilitiesByID[modelID]?.unavailableMessage
        case nil:
            return nil
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
