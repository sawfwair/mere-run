import AppKit
import SwiftUI

/// The floating command bar at the bottom of the canvas: one field, one run button, and the
/// depth (attachment, model, options) folded into quiet affordances around it.
struct StudioComposer: View {
    let mode: StudioMode
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
        HStack(spacing: MereRunTheme.Spacing.sm) {
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

                if mode.acceptedTypes.contains(.image) {
                    Button(action: onPaste) {
                        Image(systemName: "doc.on.clipboard")
                            .font(.system(size: 14, weight: .semibold))
                            .frame(width: 30, height: 30)
                    }
                    .buttonStyle(.mereIcon)
                    .help("Paste image from clipboard (⌘V)")
                    .accessibilityLabel("Paste image from clipboard")
                }
            }

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

            HStack(spacing: 6) {
                modelMenu

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
                    StudioOptionsPanel(mode: mode, draft: $draft)
                }

                if isRunning {
                    Button(action: onStop) {
                        Image(systemName: "stop.fill")
                            .font(.system(size: 12, weight: .bold))
                            .frame(width: 30, height: 30)
                    }
                    .buttonStyle(.mereIcon(tint: MereRunTheme.red))
                    .help("Stop current run (⌘.)")
                    .accessibilityLabel("Stop current run")
                }

                sendButton
            }
        }
        .padding(.horizontal, MereRunTheme.Spacing.md)
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
        .help("Model for this run")
        .accessibilityLabel("Model")
        .accessibilityValue(modelLabel)
    }

    private var modelLabel: String {
        let current = draft.model.trimmingCharacters(in: .whitespacesAndNewlines)
        if !current.isEmpty { return shortModelName(current) }
        let templateDefault = CommandCatalog.template(id: mode.defaultTemplateID)?.defaultModel ?? ""
        return templateDefault.isEmpty ? "Auto" : shortModelName(templateDefault)
    }

    private func shortModelName(_ id: String) -> String {
        id.components(separatedBy: "/").last ?? id
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
    @EnvironmentObject private var controller: MereRunController
    @State private var voiceProfiles: [StudioVoiceProfile] = []

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

                if [.createImage, .video].contains(mode) {
                    HStack(spacing: MereRunTheme.Spacing.sm) {
                        Stepper("Width \(draft.width)", value: $draft.width, in: 64...4096, step: 64)
                        Stepper("Height \(draft.height)", value: $draft.height, in: 64...4096, step: 64)
                    }
                    .font(MereRunTheme.captionFont)
                }

                if [.createImage, .music, .sfx].contains(mode) {
                    Stepper("Steps \(draft.steps)", value: $draft.steps, in: 1...80, step: 1)
                        .font(MereRunTheme.captionFont)
                }

                if [.music, .sfx].contains(mode) {
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

                if !StudioOptionSchema.fields(for: mode).isEmpty {
                    Divider().overlay(MereRunTheme.border.opacity(0.4))
                    ForEach(StudioOptionSchema.fields(for: mode)) { field in
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
        default: return nil
        }
    }
}
