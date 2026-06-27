import AppKit
import SwiftUI
import UniformTypeIdentifiers

struct StudioRootView: View {
    @EnvironmentObject private var controller: MereRunController
    @StateObject private var library = StudioLibraryStore()
    @State private var mode: StudioMode = .createImage
    @State private var draft = StudioDraft()
    @State private var showLibrary = true
    @State private var showAdvanced = false
    @State private var showOptions = false
    @State private var showModels = false
    @State private var showHelp = false
    @State private var advancedWidth: CGFloat = 560
    @State private var advancedDragStartWidth: CGFloat?
    @State private var advancedDetached = false
    @State private var selectedLibraryID: UUID?
    /// The conversation the canvas/composer targets in chat/code modes. nil means a fresh,
    /// not-yet-sent conversation (so the library has no empty row until the first message).
    @State private var activeConversationID: UUID?
    @State private var pendingPullRefresh: StudioReadinessRefresh?
    @State private var studioError: String?
    @AppStorage("mererun.app.hasCompletedWelcome") private var hasCompletedWelcome = false
    @State private var showWelcome = false

    private var selectedItem: StudioLibraryItem? {
        if let selectedLibraryID, let found = library.items.first(where: { $0.id == selectedLibraryID }) {
            return found
        }
        // Fall back to the most recent run for the active mode so switching modes never
        // leaves an unrelated mode's output on the canvas.
        return library.items.first { $0.mode == mode }
    }

    private var activeConversationItem: StudioLibraryItem? {
        guard let activeConversationID else { return nil }
        return library.items.first { $0.id == activeConversationID && $0.isConversation }
    }

    private var activeConversationLiveText: String? {
        guard let activeConversationID else { return nil }
        return controller.conversationLiveText[activeConversationID]
    }

    private var activeConversationRunning: Bool {
        guard let activeConversationID else { return false }
        return controller.runningConversationIDs.contains(activeConversationID)
    }

    private var readiness: ModelReadinessState {
        controller.readinessByMode[mode] ?? .unknown("Readiness has not been checked yet.")
    }

    private var selectedCapabilityRequirement: StudioCapabilityRequirement? {
        StudioCommandAdapter.capabilityRequirement(for: mode, draft: draft)
    }

    private var selectedCapability: StudioModelCapability? {
        guard let selectedCapabilityRequirement,
              case .managedModel(let modelID) = selectedCapabilityRequirement else {
            return nil
        }
        return controller.modelCapabilitiesByID[modelID]
    }

    private var selectedUnavailableCapabilityMessage: String? {
        guard let selectedCapabilityRequirement,
              case .unavailable(let message) = selectedCapabilityRequirement else {
            return nil
        }
        return message
    }

    private var displayStatus: String {
        guard controller.queuedRunCount > 0 else { return controller.status }
        let suffix = controller.queuedRunCount == 1 ? "1 queued" : "\(controller.queuedRunCount) queued"
        return "\(controller.status) · \(suffix)"
    }

    private var modeCapabilities: [StudioMode: StudioModelCapability] {
        Dictionary(
            uniqueKeysWithValues: StudioMode.allCases.compactMap { candidate in
                var candidateDraft = StudioDraft()
                candidateDraft.reset(for: candidate)
                let requirement = StudioCommandAdapter.capabilityRequirement(
                    for: candidate,
                    draft: candidateDraft
                )
                guard let requirement,
                      case .managedModel(let modelID) = requirement,
                      let capability = controller.modelCapabilitiesByID[modelID] else {
                    return nil
                }
                return (candidate, capability)
            }
        )
    }

    var body: some View {
        HStack(spacing: 0) {
            if showLibrary {
                StudioLibraryPanel(
                    items: library.items,
                    selectedID: $selectedLibraryID,
                    isVisible: $showLibrary,
                    onDelete: { id in
                        library.delete(id: id)
                        if selectedLibraryID == id { selectedLibraryID = nil }
                    },
                    onRename: { id, title in
                        library.rename(id: id, title: title)
                    }
                )
                .frame(width: 292)

                Divider()
                    .overlay(MereRunTheme.border.opacity(0.55))
            }

            VStack(spacing: 0) {
                StudioTopBar(
                    mode: $mode,
                    showLibrary: $showLibrary,
                    showAdvanced: $showAdvanced,
                    readiness: readiness,
                    modeCapabilities: modeCapabilities,
                    resolvedCLI: controller.resolvedCLI,
                    serverStatus: controller.serverStatus,
                    onShowModels: { showModels = true },
                    onShowHelp: { showHelp = true }
                )

                Divider()
                    .overlay(MereRunTheme.border.opacity(0.45))

                if let persistenceError = library.lastPersistenceError {
                    MereBanner(
                        severity: .warning,
                        text: "Run history not saved: \(persistenceError)"
                    )
                    .padding(.horizontal, 24)
                    .padding(.top, 10)
                }

                StudioCanvas(
                    mode: mode,
                    item: selectedItem,
                    conversationItem: activeConversationItem,
                    conversationLiveText: activeConversationLiveText,
                    isConversationRunning: activeConversationRunning,
                    isRunning: controller.isRunning,
                    status: displayStatus,
                    readiness: readiness,
                    error: studioError,
                    logs: controller.logs,
                    liveOutputText: controller.liveOutputText,
                    progress: controller.currentProgress,
                    onOpen: openSelectedOutput,
                    onReveal: revealSelectedOutput,
                    onPullModel: pullModel,
                    onShowDetails: { showAdvanced = true },
                    onNewChat: startNewConversation,
                    onCopy: copyToClipboard,
                    onRetry: retryLastTurn,
                    onEdit: editMessage
                )

                StudioPromptBar(
                    mode: mode,
                    draft: $draft,
                    showOptions: $showOptions,
                    isRunning: controller.isRunning,
                    queuedCount: controller.queuedRunCount,
                    readiness: readiness,
                    sendBlocked: mode.isConversational && activeConversationRunning,
                    onRun: runStudioCommand,
                    onStop: controller.cancel,
                    onAttach: chooseAttachment
                )
                .padding(.horizontal, 24)
                .padding(.bottom, 22)
            }
            .frame(minWidth: 680)
            .dropDestination(for: URL.self) { urls, _ in
                guard !mode.acceptedTypes.isEmpty, let url = urls.first(where: \.isFileURL) else {
                    return false
                }
                draft.inputPath = url.path
                studioError = nil
                return true
            }

            if showAdvanced {
                advancedResizeHandle
                AdvancedControlSurface(docked: true, onDetach: { advancedDetached = true })
                    .frame(width: advancedWidth)
            }
        }
        .background {
            ZStack {
                MereRunTheme.background
                LinearGradient(
                    colors: [
                        MereRunTheme.surfaceRaised.opacity(0.28),
                        MereRunTheme.background,
                        MereRunTheme.surface.opacity(0.2)
                    ],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
            }
            .ignoresSafeArea()
        }
        .sheet(isPresented: $showOptions) {
            StudioOptionsSheet(mode: mode, draft: $draft)
                .frame(width: 500)
        }
        .sheet(isPresented: $showModels) {
            StudioModelsSheet(onModelsChanged: refreshReadiness)
                .environmentObject(controller)
        }
        .sheet(isPresented: $showWelcome) {
            StudioWelcomeSheet(
                resolvedCLI: controller.resolvedCLI,
                onBrowseModels: {
                    hasCompletedWelcome = true
                    showWelcome = false
                    showModels = true
                },
                onDone: {
                    hasCompletedWelcome = true
                    showWelcome = false
                }
            )
        }
        .sheet(isPresented: $showHelp) {
            StudioHelpSheet()
                .environmentObject(controller)
                .frame(width: 720, height: 560)
        }
        .sheet(isPresented: $advancedDetached) {
            VStack(spacing: 0) {
                HStack {
                    Text("Advanced — full control surface")
                        .font(MereRunTheme.sectionFont)
                    Spacer()
                    Button("Dock") { advancedDetached = false }
                        .keyboardShortcut(.defaultAction)
                }
                .padding(16)
                Divider().overlay(MereRunTheme.border.opacity(0.5))
                AdvancedControlSurface(docked: false)
            }
            .frame(width: 1_260, height: 780)
            .environmentObject(controller)
        }
        .task {
            // Poll the local server status for the top-bar pill. status has a 1s probe timeout,
            // so a modest cadence keeps the pill live without hammering the CLI.
            while !Task.isCancelled {
                await controller.refreshServerStatus()
                try? await Task.sleep(nanoseconds: 20 * 1_000_000_000)
            }
        }
        .onAppear {
            draft.reset(for: mode)
            controller.checkReadiness(for: mode, draft: draft)
            if !hasCompletedWelcome {
                showWelcome = true
            }
        }
        .onChange(of: showAdvanced) { _, isShown in
            if isShown { syncAdvancedToStudio() }
        }
        .onChange(of: mode) { _, newMode in
            var nextDraft = StudioDraft()
            nextDraft.reset(for: newMode)
            draft = nextDraft
            studioError = nil
            if newMode.isConversational {
                // Open the most recent thread for this mode (or a fresh one) and reuse its
                // system/model so follow-ups match; the composer starts empty.
                let latest = library.items.first { $0.mode == newMode && $0.isConversation }
                activeConversationID = latest?.id
                selectedLibraryID = latest?.id
                if let latest { applyConversationSettings(from: latest, to: &draft) }
                draft.prompt = ""
            } else {
                activeConversationID = nil
                selectedLibraryID = library.items.first { $0.mode == newMode }?.id
            }
            controller.checkReadiness(for: newMode, draft: nextDraft)
        }
        .onChange(of: selectedLibraryID) { _, id in
            // Selecting a thread of the current conversation mode opens it in the canvas.
            guard mode.isConversational,
                  let id,
                  let item = library.items.first(where: { $0.id == id }),
                  item.isConversation, item.mode == mode,
                  id != activeConversationID else { return }
            activeConversationID = id
            applyConversationSettings(from: item, to: &draft)
            draft.prompt = ""
        }
        .onChange(of: draft.model) { _, _ in
            studioError = nil
            refreshReadiness()
        }
        .onChange(of: draft.readImageAction) { _, _ in
            studioError = nil
            refreshReadiness()
        }
        .onChange(of: draft.inputPath) { _, _ in
            studioError = nil
        }
        .onChange(of: draft.prompt) { _, _ in
            studioError = nil
        }
        .onChange(of: draft.secondaryText) { _, _ in
            studioError = nil
        }
        .onChange(of: controller.cliPath) { _, _ in
            studioError = nil
            refreshReadiness()
        }
        .onChange(of: controller.modelsRoot) { _, _ in
            studioError = nil
            refreshReadiness()
        }
        .onChange(of: controller.hubCache) { _, _ in
            studioError = nil
            refreshReadiness()
        }
        .onReceive(controller.runCompletions) { result in
            // Subscribe to the lossless completion stream, not lastRunResult: two runs finishing
            // in the same runloop turn would coalesce through onChange and drop one.

            // Conversation turns append the assistant reply to the thread instead of taking the
            // single-shot completion path.
            if let conversationID = result.conversationID {
                library.appendAssistant(
                    conversationID: conversationID,
                    content: conversationReplyContent(for: result),
                    exitCode: result.exitCode
                )
                // Only follow selection if this is the thread the user is currently viewing — a
                // background turn completing must not yank selection away from the foreground.
                if mode.isConversational, activeConversationID == conversationID {
                    selectedLibraryID = conversationID
                }
                refreshReadiness()
                return
            }

            let completedLibraryItem = result.requestID != nil
            if let requestID = result.requestID {
                library.complete(
                    id: requestID,
                    exitCode: result.exitCode,
                    outputURL: result.outputURL,
                    outputText: result.outputText,
                    commandPreview: result.commandPreview.maskingAPIKeyValue()
                )
                selectedLibraryID = requestID
            }

            let mutatedModels = result.templateID == .modelPull
                || result.templateID == .modelRemove
                || result.templateID == .modelRepairManifests

            if let pendingPullRefresh, result.templateID == .modelPull {
                self.pendingPullRefresh = nil
                controller.checkReadiness(for: pendingPullRefresh.mode, draft: pendingPullRefresh.draft)
            } else if mutatedModels || completedLibraryItem {
                refreshReadiness()
            }
        }
        .onChange(of: controller.activeRunRequestID) { _, requestID in
            // Conversation turns use a per-turn request id with no matching library row, so the
            // guard keeps the thread selected instead of deselecting it.
            guard let requestID, library.items.contains(where: { $0.id == requestID }) else { return }
            library.markRunning(id: requestID)
            selectedLibraryID = requestID
        }
        .onChange(of: controller.lastOutputURL) { _, outputURL in
            guard let requestID = controller.activeRunRequestID, let outputURL,
                  library.items.contains(where: { $0.id == requestID }) else { return }
            library.updateOutput(id: requestID, outputURL: outputURL)
            selectedLibraryID = requestID
        }
    }

    private func runStudioCommand() {
        studioError = nil

        if let message = selectedUnavailableCapabilityMessage {
            studioError = message
            return
        }

        if let message = selectedCapability?.unavailableMessage {
            studioError = message
            return
        }

        if readiness.blocksRun {
            studioError = readiness.message
            return
        }

        if mode.isConversational {
            sendConversationTurn()
            return
        }

        do {
            let request = try StudioCommandAdapter.makeRequest(mode: mode, draft: draft)
            let preview = controller
                .commandPreview(template: request.template, draft: request.draft, masksSecrets: true)
            let status: StudioLibraryStatus = controller.isRunning || controller.queuedRunCount > 0 ? .queued : .running
            library.start(request: request, commandPreview: preview, status: status)
            selectedLibraryID = request.id
            controller.run(studio: request)
        } catch {
            studioError = error.localizedDescription
        }
    }

    /// Sends one chat/code turn: appends the user message to the thread (creating it on the first
    /// turn), serializes history into the prompt, and runs it routed back to the conversation.
    private func sendConversationTurn() {
        let content = draft.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !content.isEmpty else { return }

        let conversationID = activeConversationID ?? UUID()
        // One in-flight turn per thread keeps turns ordered.
        guard !controller.runningConversationIDs.contains(conversationID) else { return }

        let systemPrompt = draft.secondaryText.trimmingCharacters(in: .whitespacesAndNewlines)
        let model = draft.model.isBlank ? nil : draft.model
        // Vision chat attaches an image to this turn (chat only); persist it so edit/retry resend it.
        let turnImage = (mode == .chat && !draft.inputPath.isBlank) ? draft.inputPath : nil
        let item = library.appendUser(
            conversationID: conversationID,
            mode: mode,
            model: model,
            systemPrompt: systemPrompt.isEmpty ? nil : systemPrompt,
            content: content,
            imagePath: turnImage
        )

        let rendered = ConversationTranscript.render(
            messages: item.messages ?? [],
            systemPrompt: systemPrompt.isEmpty ? nil : systemPrompt
        )

        do {
            var runDraft = draft
            runDraft.prompt = rendered.prompt
            let request = try StudioCommandAdapter.makeRequest(
                mode: mode, draft: runDraft, conversationID: conversationID
            )
            controller.run(studio: request)
            activeConversationID = conversationID
            selectedLibraryID = conversationID
            draft.prompt = ""
            // The image rode with this turn; clear it so the next turn doesn't resend it.
            draft.inputPath = ""
        } catch {
            studioError = error.localizedDescription
        }
    }

    private func copyToClipboard(_ text: String) {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(text, forType: .string)
    }

    /// Re-runs the latest turn: drops the last assistant reply (if any) and re-sends the thread
    /// ending at the last user message, reusing the thread's own system prompt and model.
    private func retryLastTurn() {
        guard let conversationID = activeConversationID,
              !controller.runningConversationIDs.contains(conversationID) else { return }
        library.dropLastAssistant(conversationID: conversationID)
        guard let item = library.items.first(where: { $0.id == conversationID }),
              item.messages?.last?.role == .user else { return }

        let systemPrompt = item.systemPrompt
        let rendered = ConversationTranscript.render(
            messages: item.messages ?? [],
            systemPrompt: systemPrompt
        )
        do {
            var runDraft = draft
            runDraft.prompt = rendered.prompt
            runDraft.secondaryText = systemPrompt ?? ""
            // Re-attach the image the last user turn carried (or none), not the cleared composer's.
            runDraft.inputPath = item.messages?.last?.imagePath ?? ""
            if let model = item.model, !model.isBlank { runDraft.model = model }
            let request = try StudioCommandAdapter.makeRequest(
                mode: item.mode, draft: runDraft, conversationID: conversationID
            )
            controller.run(studio: request)
        } catch {
            studioError = error.localizedDescription
        }
    }

    /// A draggable divider that resizes the docked Advanced column. The panel is on the right, so
    /// dragging left widens it; width is clamped to a usable range.
    private var advancedResizeHandle: some View {
        Rectangle()
            .fill(MereRunTheme.border.opacity(0.55))
            .frame(width: 5)
            .contentShape(Rectangle())
            .onHover { inside in
                if inside { NSCursor.resizeLeftRight.push() } else { NSCursor.pop() }
            }
            .gesture(
                DragGesture()
                    .onChanged { value in
                        let base = advancedDragStartWidth ?? advancedWidth
                        if advancedDragStartWidth == nil { advancedDragStartWidth = base }
                        advancedWidth = min(max(base - value.translation.width, 360), 860)
                    }
                    .onEnded { _ in advancedDragStartWidth = nil }
            )
    }

    /// When Advanced opens, pre-select the template for the active Studio mode and carry over what
    /// the composer already holds — including the shared depth fields — so Advanced deepens the
    /// current task without silently reverting edits.
    private func syncAdvancedToStudio() {
        if let template = CommandCatalog.template(id: mode.defaultTemplateID) {
            controller.select(template)
        }
        controller.draft.prompt = draft.prompt
        if !draft.model.isBlank { controller.draft.model = draft.model }
        if !draft.inputPath.isBlank { controller.draft.inputPath = draft.inputPath }
        // Shared schema depth (WS-3.5): keep the two surfaces aligned after live edits, not only
        // at defaults.
        controller.draft.temperature = draft.temperature
        controller.draft.maxTokens = draft.maxTokens
        controller.draft.cfgScale = draft.cfgScale
        controller.draft.strength = draft.strength
        controller.draft.language = draft.language
        controller.draft.backend = draft.backend
        controller.draft.timestamps = draft.timestamps
        controller.draft.fps = draft.fps
        controller.draft.numFrames = draft.numFrames
        controller.draft.variant = draft.variant
    }

    /// Edits a prior user turn: truncates the thread at that message and loads its text back into
    /// the composer, so sending re-runs the conversation from that point.
    private func editMessage(_ messageID: UUID) {
        guard let conversationID = activeConversationID,
              !controller.runningConversationIDs.contains(conversationID) else { return }
        if let removed = library.truncate(conversationID: conversationID, removingFrom: messageID) {
            draft.prompt = removed.content
            // Restore the turn's attached image so re-sending re-runs vision chat as before.
            if mode == .chat { draft.inputPath = removed.imagePath ?? "" }
        }
        // Editing the first turn empties the thread — drop the now-empty row and act like a new chat.
        if let item = library.items.first(where: { $0.id == conversationID }),
           item.messages?.isEmpty ?? true {
            library.delete(id: conversationID)
            activeConversationID = nil
            selectedLibraryID = nil
        }
    }

    /// Starts a fresh, not-yet-persisted conversation (no library row until the first message).
    private func startNewConversation() {
        activeConversationID = nil
        selectedLibraryID = nil
        studioError = nil
        var fresh = StudioDraft()
        fresh.reset(for: mode)
        fresh.prompt = ""
        draft = fresh
    }

    private func applyConversationSettings(from item: StudioLibraryItem, to draft: inout StudioDraft) {
        if let systemPrompt = item.systemPrompt { draft.secondaryText = systemPrompt }
        if let model = item.model, !model.isBlank { draft.model = model }
    }

    private func conversationReplyContent(for result: MereRunRunResult) -> String {
        let text = result.outputText?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if !text.isEmpty { return text }
        return result.exitCode == 0 ? "(No output.)" : "Run failed (exit code \(result.exitCode))."
    }

    private func pullModel() {
        studioError = nil

        if let message = selectedUnavailableCapabilityMessage {
            studioError = message
            return
        }

        if let message = selectedCapability?.unavailableMessage {
            studioError = message
            return
        }

        guard readiness.canPull else {
            studioError = readiness.message
            return
        }

        do {
            guard let request = try StudioCommandAdapter.pullRequest(for: mode, draft: draft) else {
                studioError = "This mode does not need a managed model."
                return
            }
            pendingPullRefresh = StudioReadinessRefresh(mode: mode, draft: draft)
            controller.readinessByMode[mode] = .checking
            showAdvanced = true
            if !controller.run(studio: request) {
                pendingPullRefresh = nil
                refreshReadiness()
            }
        } catch {
            studioError = error.localizedDescription
        }
    }

    private func refreshReadiness() {
        controller.checkReadiness(for: mode, draft: draft)
    }

    private func chooseAttachment() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = mode.acceptedTypes.isEmpty ? [.item] : mode.acceptedTypes
        if panel.runModal() == .OK, let url = panel.url {
            draft.inputPath = url.path
            studioError = nil
        }
    }

    private func openSelectedOutput() {
        guard let url = selectedItem?.outputURL else { return }
        NSWorkspace.shared.open(url)
    }

    private func revealSelectedOutput() {
        guard let url = selectedItem?.outputURL else { return }
        NSWorkspace.shared.activateFileViewerSelecting([url])
    }
}

private struct StudioReadinessRefresh: Equatable {
    let mode: StudioMode
    let draft: StudioDraft
}

private struct StudioTopBar: View {
    @Binding var mode: StudioMode
    @Binding var showLibrary: Bool
    @Binding var showAdvanced: Bool
    let readiness: ModelReadinessState
    let modeCapabilities: [StudioMode: StudioModelCapability]
    let resolvedCLI: String
    let serverStatus: StudioServerStatus?
    let onShowModels: () -> Void
    let onShowHelp: () -> Void

    var body: some View {
        VStack(spacing: 14) {
            HStack(spacing: 12) {
                Button {
                    withAnimation(.easeOut(duration: 0.18)) {
                        showLibrary.toggle()
                    }
                } label: {
                    Image(systemName: "sidebar.left")
                        .frame(width: 30, height: 30)
                }
                .buttonStyle(.plain)
                .help("Show library")

                VStack(alignment: .leading, spacing: 2) {
                    Text("mere.run")
                        .font(.system(size: 20, weight: .semibold))
                    Text("Create anything. Locally.")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                }

                Spacer()

                StudioStatusPill(
                    title: "Server",
                    detail: serverStatusDetail,
                    systemImage: serverStatus?.isReachable == true ? "bolt.horizontal.circle" : "circle.dashed",
                    color: serverStatusColor
                )

                StudioStatusPill(
                    title: readiness.title,
                    detail: readiness.message,
                    systemImage: readinessStatusImage,
                    color: readinessStatusColor
                )

                StudioStatusPill(
                    title: "CLI",
                    detail: resolvedCLI,
                    systemImage: "terminal",
                    color: MereRunTheme.accent
                )

                Button {
                    onShowModels()
                } label: {
                    Label("Models", systemImage: "shippingbox")
                }
                .buttonStyle(.bordered)

                Button {
                    onShowHelp()
                } label: {
                    Label("Help", systemImage: "questionmark.circle")
                }
                .buttonStyle(.bordered)

                Button {
                    withAnimation(.easeOut(duration: 0.18)) {
                        showAdvanced.toggle()
                    }
                } label: {
                    Label("Advanced", systemImage: "slider.horizontal.3")
                }
                .buttonStyle(.bordered)
            }

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(StudioMode.allCases) { candidate in
                        StudioModeChip(
                            mode: candidate,
                            isSelected: candidate == mode,
                            unavailableMessage: modeCapabilities[candidate]?.unavailableMessage
                        ) {
                            withAnimation(.easeOut(duration: 0.18)) {
                                mode = candidate
                            }
                        }
                    }
                }
                .padding(.horizontal, 2)
            }
        }
        .padding(.horizontal, 22)
        .padding(.top, 16)
        .padding(.bottom, 14)
        .background(VisualEffectBackground())
    }

    private var serverStatusDetail: String {
        guard let serverStatus else { return "checking…" }
        if serverStatus.isReachable {
            let model = serverStatus.loadedModelSummary.map { " · \($0)" } ?? ""
            return "up\(model) · \(serverStatus.installedCount) installed"
        }
        return "offline · \(serverStatus.installedCount) installed"
    }

    private var serverStatusColor: Color {
        guard let serverStatus else { return MereRunTheme.textMuted }
        return serverStatus.isReachable ? MereRunTheme.green : MereRunTheme.textMuted
    }

    private var readinessStatusImage: String {
        if readiness.isChecking { return "hourglass" }
        if readiness.canPull { return "arrow.down.circle" }
        if readiness.blocksRun { return "exclamationmark.triangle" }
        return "checkmark.circle"
    }

    private var readinessStatusColor: Color {
        if readiness.isChecking { return MereRunTheme.yellow }
        if readiness.canPull { return MereRunTheme.yellow }
        if readiness.blocksRun { return MereRunTheme.red }
        return MereRunTheme.green
    }
}

private struct StudioModeChip: View {
    let mode: StudioMode
    let isSelected: Bool
    let unavailableMessage: String?
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 7) {
                Image(systemName: mode.systemImage)
                    .font(.system(size: 12, weight: .semibold))
                Text(mode.title)
                    .font(.system(size: 12, weight: .semibold))
            }
            .foregroundStyle(foregroundColor)
            .padding(.horizontal, 12)
            .frame(height: 32)
            .background {
                Capsule()
                    .fill(backgroundColor)
                    .overlay {
                        Capsule()
                            .strokeBorder(borderColor, lineWidth: 1)
                    }
            }
        }
        .buttonStyle(.plain)
        .disabled(unavailableMessage != nil)
        .opacity(unavailableMessage == nil ? 1 : 0.52)
        .help(unavailableMessage ?? mode.subtitle)
    }

    private var foregroundColor: Color {
        if unavailableMessage != nil {
            return MereRunTheme.textMuted
        }
        return isSelected ? MereRunTheme.background : MereRunTheme.textSecondary
    }

    private var backgroundColor: Color {
        if unavailableMessage != nil {
            return MereRunTheme.surface.opacity(0.55)
        }
        return isSelected ? MereRunTheme.accent : MereRunTheme.surface
    }

    private var borderColor: Color {
        if unavailableMessage != nil {
            return MereRunTheme.border.opacity(0.7)
        }
        return MereRunTheme.border.opacity(isSelected ? 0 : 0.8)
    }
}

private struct StudioStatusPill: View {
    let title: String
    let detail: String
    let systemImage: String
    let color: Color

    var body: some View {
        HStack(spacing: 7) {
            Image(systemName: systemImage)
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(color)
            VStack(alignment: .leading, spacing: 1) {
                Text(title)
                    .font(.system(size: 11, weight: .semibold))
                Text(detail)
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(MereRunTheme.textMuted)
                    .lineLimit(1)
            }
        }
        .padding(.horizontal, 10)
        .frame(height: 38)
        .frame(maxWidth: 210)
        .merePanel(cornerRadius: 18)
    }
}

private struct StudioCanvas: View {
    let mode: StudioMode
    let item: StudioLibraryItem?
    let conversationItem: StudioLibraryItem?
    let conversationLiveText: String?
    let isConversationRunning: Bool
    let isRunning: Bool
    let status: String
    let readiness: ModelReadinessState
    let error: String?
    let logs: [LogLine]
    let liveOutputText: String
    let progress: StudioRunProgress?
    let onOpen: () -> Void
    let onReveal: () -> Void
    let onPullModel: () -> Void
    let onShowDetails: () -> Void
    let onNewChat: () -> Void
    let onCopy: (String) -> Void
    let onRetry: () -> Void
    let onEdit: (UUID) -> Void

    private var visibleLiveOutputText: String? {
        guard isRunning else { return nil }
        let text = liveOutputText
            .replacingOccurrences(of: "\0", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return text.isEmpty ? nil : text
    }

    private var recentLogLines: [String] {
        logs.suffix(4).map(\.text)
    }

    private var shouldShowReadinessOverlay: Bool {
        !isRunning && (readiness.blocksRun || error != nil)
    }

    var body: some View {
        ZStack {
            if mode.isConversational {
                StudioConversationView(
                    item: conversationItem,
                    liveText: conversationLiveText,
                    isRunning: isConversationRunning,
                    mode: mode,
                    onNewChat: onNewChat,
                    onCopy: onCopy,
                    onRetry: onRetry,
                    onEdit: onEdit
                )
                .transition(.opacity)
            } else if let item {
                StudioOutputView(item: item, liveOutputText: visibleLiveOutputText, onOpen: onOpen, onReveal: onReveal)
                    .padding(32)
                    .transition(.opacity.combined(with: .scale(scale: 0.98)))
            } else {
                StudioEmptyState(mode: mode)
                    .padding(32)
            }

            if isRunning && visibleLiveOutputText == nil && !mode.isConversational {
                StudioRunningOverlay(status: status, progress: progress, recentLogs: recentLogLines)
                    .transition(.opacity)
            }

            if shouldShowReadinessOverlay {
                StudioReadinessOverlay(
                    title: error == nil ? readiness.title : "Needs attention",
                    message: error ?? readiness.message,
                    canPull: error == nil && readiness.canPull,
                    isChecking: error == nil && readiness.isChecking,
                    onPullModel: onPullModel,
                    onShowDetails: onShowDetails
                )
                .padding(32)
                .transition(.opacity.combined(with: .scale(scale: 0.98)))
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .animation(.easeOut(duration: 0.18), value: isRunning)
        .animation(.easeOut(duration: 0.18), value: shouldShowReadinessOverlay)
    }
}

private struct StudioEmptyState: View {
    let mode: StudioMode

    var body: some View {
        VStack(spacing: 18) {
            Image(systemName: mode.systemImage)
                .font(.system(size: 48, weight: .semibold))
                .foregroundStyle(MereRunTheme.accent)
                .frame(width: 92, height: 92)
                .background {
                    Circle()
                        .fill(MereRunTheme.surfaceRaised)
                }

            VStack(spacing: 8) {
                Text(mode.emptyTitle)
                    .font(.system(size: 30, weight: .semibold))
                Text(mode.emptyMessage)
                    .font(.system(size: 14, weight: .medium))
                    .foregroundStyle(MereRunTheme.textSecondary)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 460)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

private struct StudioRunningOverlay: View {
    let status: String
    let progress: StudioRunProgress?
    let recentLogs: [String]

    var body: some View {
        VStack(spacing: 14) {
            if let progress, let fraction = progress.fractionCompleted {
                VStack(spacing: 6) {
                    ProgressView(value: fraction)
                        .progressViewStyle(.linear)
                        .tint(MereRunTheme.accent)
                        .frame(width: 320)
                    HStack {
                        Text(progress.label)
                            .lineLimit(1)
                        Spacer()
                        Text("\(Int((fraction * 100).rounded()))%")
                    }
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                    .frame(width: 320)
                    if let detail = progress.detail {
                        Text(detail)
                            .font(MereRunTheme.captionFont)
                            .foregroundStyle(MereRunTheme.textMuted)
                            .lineLimit(1)
                    }
                }
            } else {
                ProgressView()
                    .controlSize(.large)
            }
            Text(progress?.detail != nil && progress?.fractionCompleted == nil
                ? "\(status) · \(progress?.detail ?? "")"
                : status)
                .font(.system(size: 18, weight: .semibold))
                .multilineTextAlignment(.center)
            if !recentLogs.isEmpty {
                VStack(alignment: .leading, spacing: 6) {
                    ForEach(Array(recentLogs.enumerated()), id: \.offset) { _, line in
                        Text(line)
                            .font(MereRunTheme.monoFont)
                            .foregroundStyle(MereRunTheme.textMuted)
                            .lineLimit(2)
                    }
                }
                .padding(12)
                .frame(width: 460, alignment: .leading)
                .background {
                    RoundedRectangle(cornerRadius: 10)
                        .fill(MereRunTheme.background.opacity(0.55))
                }
            }
        }
        .padding(28)
        .merePanel(cornerRadius: 18)
        .mereShadow(radius: 28, y: 12)
    }
}

private struct StudioReadinessOverlay: View {
    let title: String
    let message: String
    let canPull: Bool
    let isChecking: Bool
    let onPullModel: () -> Void
    let onShowDetails: () -> Void

    var body: some View {
        VStack(spacing: 14) {
            Image(systemName: statusImage)
                .font(.system(size: 32, weight: .semibold))
                .foregroundStyle(statusColor)
            Text(title)
                .font(.system(size: 22, weight: .semibold))
            Text(message)
                .font(MereRunTheme.bodyFont)
                .foregroundStyle(MereRunTheme.textSecondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 280)
            HStack(spacing: 10) {
                if canPull {
                    Button {
                        onPullModel()
                    } label: {
                        Label("Pull Model", systemImage: "arrow.down.circle.fill")
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(MereRunTheme.accent)
                }
                Button {
                    onShowDetails()
                } label: {
                    Label("Details", systemImage: "terminal")
                }
                .buttonStyle(.bordered)
            }
        }
        .padding(28)
        .merePanel(cornerRadius: 18)
        .mereShadow(radius: 30, y: 12)
    }

    private var statusImage: String {
        if isChecking { return "hourglass" }
        return canPull ? "arrow.down.circle" : "exclamationmark.triangle"
    }

    private var statusColor: Color {
        if isChecking { return MereRunTheme.yellow }
        return canPull ? MereRunTheme.yellow : MereRunTheme.red
    }
}

private struct StudioOutputView: View {
    let item: StudioLibraryItem
    let liveOutputText: String?
    let onOpen: () -> Void
    let onReveal: () -> Void

    var body: some View {
        VStack(spacing: 18) {
            outputPreview
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(MereRunTheme.surface.opacity(0.45))
                .clipShape(RoundedRectangle(cornerRadius: 18))
                .overlay {
                    RoundedRectangle(cornerRadius: 18)
                        .strokeBorder(MereRunTheme.border.opacity(0.65), lineWidth: 1)
                }

            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(item.mode.title)
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                    Text(item.displayTitle)
                        .font(.system(size: 16, weight: .semibold))
                        .lineLimit(1)
                }
                Spacer()
                statusBadge
                if item.outputURL != nil {
                    Button("Open", action: onOpen)
                        .buttonStyle(.bordered)
                    Button {
                        onReveal()
                    } label: {
                        Image(systemName: "magnifyingglass")
                    }
                    .buttonStyle(.bordered)
                    .help("Reveal in Finder")
                }
            }
        }
    }

    @ViewBuilder
    private var outputPreview: some View {
        if let url = item.outputURL {
            switch StudioOutputFileKind.classify(url) {
            case .image:
                StudioAsyncImagePreview(
                    url: url,
                    maxPixelSize: 2_200,
                    contentMode: .fit,
                    fallbackSystemImage: iconName(for: url)
                )
                .padding(22)
            case .audio:
                StudioAudioPlayerView(url: url)
            case .video:
                StudioVideoPlayerView(url: url)
            case .text:
                StudioTextFilePreview(url: url)
            case .other:
                filePlaceholder(for: url)
            }
        } else if let text = liveOutputText ?? item.outputText, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            ScrollView {
                Text(text)
                    .font(item.mode == .chat ? MereRunTheme.bodyFont : MereRunTheme.monoFont)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(22)
            }
        } else {
            VStack(spacing: 12) {
                Image(systemName: "doc")
                    .font(.system(size: 46, weight: .semibold))
                    .foregroundStyle(MereRunTheme.textMuted)
                Text(item.status == .failed ? "Run did not produce a file." : "Output will appear here.")
                    .font(MereRunTheme.bodyFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            }
        }
    }

    private func filePlaceholder(for url: URL) -> some View {
        VStack(spacing: 12) {
            Image(systemName: iconName(for: url))
                .font(.system(size: 54, weight: .semibold))
                .foregroundStyle(MereRunTheme.accent)
            Text(url.lastPathComponent)
                .font(.system(size: 16, weight: .semibold))
                .lineLimit(1)
            Text(url.deletingLastPathComponent().path)
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
                .lineLimit(1)
        }
        .padding(22)
    }

    private var statusBadge: some View {
        Text(item.status.rawValue.capitalized)
            .font(MereRunTheme.captionFont)
            .foregroundStyle(statusColor)
            .padding(.horizontal, 10)
            .frame(height: 28)
            .background {
                Capsule()
                    .fill(statusColor.opacity(0.14))
            }
    }

    private var statusColor: Color {
        switch item.status {
        case .queued: return MereRunTheme.textMuted
        case .running: return MereRunTheme.yellow
        case .completed: return MereRunTheme.green
        case .failed: return MereRunTheme.red
        }
    }

    private func iconName(for url: URL) -> String {
        switch url.pathExtension.lowercased() {
        case "wav", "mp3", "m4a": return "waveform"
        case "mp4", "mov": return "film"
        case "json": return "curlybraces"
        case "safetensors": return "shippingbox"
        default: return "doc"
        }
    }
}

private enum StudioImageContentMode: Equatable {
    case fit
    case fill
}

private enum StudioImageLoadState {
    case loading
    case loaded(NSImage)
    case unavailable
}

private struct StudioAsyncImagePreview: View {
    let url: URL
    let maxPixelSize: CGFloat
    let contentMode: StudioImageContentMode
    let fallbackSystemImage: String

    @State private var loadState = StudioImageLoadState.loading

    var body: some View {
        Group {
            switch loadState {
            case .loading:
                ProgressView()
                    .controlSize(.small)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            case .loaded(let image):
                if contentMode == .fill {
                    Image(nsImage: image)
                        .resizable()
                        .scaledToFill()
                } else {
                    Image(nsImage: image)
                        .resizable()
                        .scaledToFit()
                }
            case .unavailable:
                Image(systemName: fallbackSystemImage)
                    .font(.system(size: 32, weight: .semibold))
                    .foregroundStyle(MereRunTheme.accent)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .task(id: url) {
            loadState = .loading
            let loaded = await Task.detached(priority: .userInitiated) {
                StudioImagePreviewLoader.downsampledImage(from: url, maxPixelSize: maxPixelSize)
            }.value
            guard !Task.isCancelled else { return }
            if let image = loaded?.image {
                loadState = .loaded(image)
            } else {
                loadState = .unavailable
            }
        }
    }
}

private struct StudioTextFilePreview: View {
    let url: URL

    @State private var text: String?
    @State private var didLoad = false

    var body: some View {
        ScrollView {
            if let text {
                Text(text)
                    .font(MereRunTheme.monoFont)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(22)
            } else if didLoad {
                Text("Text preview unavailable.")
                    .font(MereRunTheme.bodyFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .padding(22)
            } else {
                ProgressView()
                    .controlSize(.small)
                    .frame(maxWidth: .infinity, minHeight: 180)
                    .padding(22)
            }
        }
        .task(id: url) {
            text = nil
            didLoad = false
            let preview = await Task.detached(priority: .userInitiated) {
                StudioTextPreviewReader.previewText(from: url)
            }.value
            guard !Task.isCancelled else { return }
            text = preview
            didLoad = true
        }
    }
}

private struct StudioPromptBar: View {
    let mode: StudioMode
    @Binding var draft: StudioDraft
    @Binding var showOptions: Bool
    let isRunning: Bool
    let queuedCount: Int
    let readiness: ModelReadinessState
    var sendBlocked: Bool = false
    let onRun: () -> Void
    let onStop: () -> Void
    let onAttach: () -> Void

    var body: some View {
        VStack(spacing: 10) {
            if !draft.inputPath.isBlank {
                HStack {
                    Label(URL(fileURLWithPath: draft.inputPath).lastPathComponent, systemImage: "paperclip")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textSecondary)
                    Spacer()
                    Button {
                        draft.inputPath = ""
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                    }
                    .buttonStyle(.plain)
                }
                .padding(.horizontal, 14)
            }

            HStack(spacing: 12) {
                Button(action: onAttach) {
                    Image(systemName: mode.requiresAttachment ? "paperclip.circle.fill" : "paperclip")
                        .font(.system(size: 18, weight: .semibold))
                        .frame(width: 34, height: 34)
                }
                .buttonStyle(.plain)
                .disabled(mode.acceptedTypes.isEmpty)
                .help(mode.requiresAttachment ? "Attach required input" : "Attach reference")
                .accessibilityLabel(mode.requiresAttachment ? "Attach required input" : "Attach reference")

                if mode == .listen {
                    Text(mode.promptPlaceholder)
                        .font(.system(size: 15, weight: .medium))
                        .foregroundStyle(MereRunTheme.textMuted)
                        .frame(maxWidth: .infinity, alignment: .leading)
                } else {
                    TextField(mode.promptPlaceholder, text: $draft.prompt, axis: .vertical)
                        .textFieldStyle(.plain)
                        .font(.system(size: 15, weight: .medium))
                        .lineLimit(1...4)
                        .onSubmit(onRun)
                }

                Button {
                    showOptions = true
                } label: {
                    Image(systemName: "slider.horizontal.3")
                        .frame(width: 34, height: 34)
                }
                .buttonStyle(.plain)
                .help("Options")
                .accessibilityLabel("Options")

                if isRunning {
                    Button(action: onStop) {
                        Image(systemName: "stop.fill")
                            .font(.system(size: 12, weight: .bold))
                            .foregroundStyle(MereRunTheme.red)
                            .frame(width: 34, height: 34)
                    }
                    .buttonStyle(.plain)
                    .help("Stop current run")
                    .accessibilityLabel("Stop current run")
                }

                Button {
                    onRun()
                } label: {
                    Image(systemName: "arrow.up")
                        .font(.system(size: 14, weight: .bold))
                        .foregroundStyle(MereRunTheme.background)
                        .frame(width: 38, height: 38)
                        .background {
                            Circle()
                                .fill(runButtonColor)
                    }
                }
                .buttonStyle(.plain)
                .disabled(readiness.blocksRun || sendBlocked)
                .help(sendBlocked ? "Waiting for the current reply…" : runButtonHelp)
                .accessibilityLabel(isRunning ? "Queue run" : "Run")
                .keyboardShortcut(.return, modifiers: .command)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 12)
            .background {
                RoundedRectangle(cornerRadius: 20)
                    .fill(MereRunTheme.surfaceRaised)
                    .overlay {
                        RoundedRectangle(cornerRadius: 20)
                            .strokeBorder(MereRunTheme.border.opacity(0.75), lineWidth: 1)
                    }
                    .mereShadow(radius: 20, y: 10)
            }
        }
    }

    private var runButtonColor: Color {
        if readiness.blocksRun { return MereRunTheme.yellow }
        return MereRunTheme.accent
    }

    private var runButtonHelp: String {
        if readiness.blocksRun { return readiness.message }
        if isRunning {
            let queueLabel = queuedCount == 0 ? "Queue run" : "Queue run (\(queuedCount) waiting)"
            return queueLabel
        }
        return "Run"
    }
}

private struct StudioWelcomeSheet: View {
    let resolvedCLI: String
    let onBrowseModels: () -> Void
    let onDone: () -> Void

    private let highlights: [(String, String, String)] = [
        ("photo", "Create locally", "Images, video, music, sound effects, and speech — all on-device."),
        ("bubble.left.and.bubble.right", "Chat & code", "Run local language models for chat, code, OCR, and vision."),
        ("lock.shield", "Private by default", "Nothing leaves your Mac. The app drives the public mere.run CLI underneath.")
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            VStack(alignment: .leading, spacing: 6) {
                Text("Welcome to mere.run")
                    .font(.system(size: 26, weight: .bold))
                Text("Create anything. Locally.")
                    .font(MereRunTheme.bodyFont)
                    .foregroundStyle(MereRunTheme.textSecondary)
            }

            VStack(alignment: .leading, spacing: 14) {
                ForEach(Array(highlights.enumerated()), id: \.offset) { _, item in
                    HStack(alignment: .top, spacing: 12) {
                        Image(systemName: item.0)
                            .font(.system(size: 18, weight: .semibold))
                            .foregroundStyle(MereRunTheme.accent)
                            .frame(width: 28)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(item.1)
                                .font(.system(size: 14, weight: .semibold))
                            Text(item.2)
                                .font(MereRunTheme.captionFont)
                                .foregroundStyle(MereRunTheme.textMuted)
                        }
                    }
                }
            }

            HStack(spacing: 8) {
                Image(systemName: "terminal")
                    .foregroundStyle(MereRunTheme.accent)
                Text(resolvedCLI)
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                    .lineLimit(1)
            }
            .padding(10)
            .merePanel()

            Text("First, download a model for the mode you want. Models open the catalog where you can pull and manage local models.")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)

            HStack(spacing: 10) {
                Button {
                    onBrowseModels()
                } label: {
                    Label("Browse models", systemImage: "shippingbox")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(MereRunTheme.accent)

                Button("Get started") {
                    onDone()
                }
                .buttonStyle(.bordered)
                .keyboardShortcut(.defaultAction)
            }
        }
        .padding(28)
        .frame(width: 460)
        .background(MereRunTheme.background)
        .foregroundStyle(MereRunTheme.textPrimary)
    }
}

private struct StudioOptionsSheet: View {
    let mode: StudioMode
    @Binding var draft: StudioDraft
    @EnvironmentObject private var controller: MereRunController
    @Environment(\.dismiss) private var dismiss
    @State private var voiceProfiles: [StudioVoiceProfile] = []

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack {
                Text("\(mode.title) Options")
                    .font(MereRunTheme.titleFont)
                Spacer()
                Button("Done") {
                    dismiss()
                }
                .keyboardShortcut(.defaultAction)
            }

            if mode == .readImage {
                Picker("Task", selection: $draft.readImageAction) {
                    ForEach(StudioReadImageAction.allCases) { action in
                        Text(action.title)
                            .foregroundStyle(readImageActionUnavailableMessage(action) == nil
                                ? MereRunTheme.textPrimary
                                : MereRunTheme.textMuted)
                            .tag(action)
                            .disabled(readImageActionUnavailableMessage(action) != nil)
                            .help(readImageActionUnavailableMessage(action) ?? action.title)
                    }
                }
                .pickerStyle(.segmented)
            }

            if secondaryLabel != nil {
                TextField(secondaryLabel!, text: $draft.secondaryText, axis: .vertical)
                    .textFieldStyle(.plain)
                    .padding(10)
                    .merePanel()
            }

            TextField("Model", text: $draft.model)
                .textFieldStyle(.plain)
                .padding(10)
                .merePanel()

            if mode == .speak {
                Picker("Voice", selection: $draft.voiceMode) {
                    Text("Style").tag("style")
                    Text("Clone").tag("clone")
                }
                .pickerStyle(.segmented)

                if draft.voiceMode == "clone" {
                    Picker("Profile", selection: $draft.voiceProfile) {
                        Text("None").tag("")
                        ForEach(voiceProfiles) { profile in
                            Text(profile.name).tag(profile.id)
                        }
                    }
                    HStack(spacing: 10) {
                        Text(draft.refAudioPath.isEmpty
                            ? "No reference audio"
                            : URL(fileURLWithPath: draft.refAudioPath).lastPathComponent)
                            .font(MereRunTheme.captionFont)
                            .foregroundStyle(MereRunTheme.textMuted)
                            .lineLimit(1)
                        Spacer()
                        Button("Reference audio…") { chooseReferenceAudio() }
                    }
                    TextField("Save as profile (optional)", text: $draft.saveProfileName)
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                }
            }

            if [.createImage, .video].contains(mode) {
                HStack(spacing: 10) {
                    Stepper("Width \(draft.width)", value: $draft.width, in: 64...4096, step: 64)
                    Stepper("Height \(draft.height)", value: $draft.height, in: 64...4096, step: 64)
                }
            }

            if [.createImage, .music, .sfx].contains(mode) {
                Stepper("Steps \(draft.steps)", value: $draft.steps, in: 1...80, step: 1)
            }

            if [.music, .sfx].contains(mode) {
                TextField("Duration seconds", value: $draft.durationSeconds, format: .number)
                    .textFieldStyle(.plain)
                    .padding(10)
                    .merePanel()
            }

            if [.createImage, .music, .video, .sfx].contains(mode) {
                TextField("Seed", text: $draft.seed)
                    .textFieldStyle(.plain)
                    .padding(10)
                    .merePanel()
            }

            if !StudioOptionSchema.fields(for: mode).isEmpty {
                Divider().overlay(MereRunTheme.border.opacity(0.4))
                ForEach(StudioOptionSchema.fields(for: mode)) { field in
                    optionRow(field)
                }
            }

            Spacer()
        }
        .padding(22)
        .background(MereRunTheme.background)
        .foregroundStyle(MereRunTheme.textPrimary)
        .task {
            if mode == .speak { voiceProfiles = await controller.loadVoiceProfiles() }
        }
    }

    /// Renders one schema field as the appropriate control, bound through the draft key path.
    @ViewBuilder
    private func optionRow(_ field: StudioOptionField) -> some View {
        switch field.control {
        case let .int(keyPath, range, step):
            Stepper("\(field.label): \(draft[keyPath: keyPath])", value: binding(keyPath), in: range, step: step)
        case let .double(keyPath):
            HStack {
                Text(field.label)
                Spacer()
                TextField(field.label, value: binding(keyPath), format: .number)
                    .textFieldStyle(.plain)
                    .frame(width: 90)
                    .padding(8)
                    .merePanel()
            }
        case let .bool(keyPath):
            Toggle(field.label, isOn: binding(keyPath))
        case let .text(keyPath, placeholder):
            HStack {
                Text(field.label)
                Spacer()
                TextField(placeholder, text: binding(keyPath))
                    .textFieldStyle(.plain)
                    .frame(width: 170)
                    .padding(8)
                    .merePanel()
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

private struct StudioLibraryPanel: View {
    let items: [StudioLibraryItem]
    @Binding var selectedID: UUID?
    @Binding var isVisible: Bool
    let onDelete: (UUID) -> Void
    let onRename: (UUID, String) -> Void

    @State private var searchText = ""
    @State private var renamingID: UUID?
    @State private var renameText = ""

    private var filteredItems: [StudioLibraryItem] {
        let query = searchText.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !query.isEmpty else { return items }
        return items.filter { item in
            item.displayTitle.lowercased().contains(query)
                || item.mode.title.lowercased().contains(query)
                || item.prompt.lowercased().contains(query)
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header

            Divider()
                .overlay(MereRunTheme.border.opacity(0.55))

            searchField

            if filteredItems.isEmpty {
                emptyState
            } else {
                list
            }
        }
        .background(MereRunTheme.background)
        .alert("Rename run", isPresented: renameBinding) {
            TextField("Name", text: $renameText)
            Button("Save") {
                if let renamingID { onRename(renamingID, renameText) }
                renamingID = nil
            }
            Button("Cancel", role: .cancel) { renamingID = nil }
        }
    }

    private var renameBinding: Binding<Bool> {
        Binding(get: { renamingID != nil }, set: { if !$0 { renamingID = nil } })
    }

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 3) {
                Text("Library")
                    .font(.system(size: 20, weight: .semibold))
                Text("\(items.count) local runs")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            }
            Spacer()
            Button {
                withAnimation(.easeOut(duration: 0.18)) {
                    isVisible = false
                }
            } label: {
                Image(systemName: "xmark")
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Hide library")
        }
        .padding(18)
    }

    private var searchField: some View {
        HStack(spacing: 8) {
            Image(systemName: "magnifyingglass")
                .foregroundStyle(MereRunTheme.textMuted)
            TextField("Search runs", text: $searchText)
                .textFieldStyle(.plain)
                .font(MereRunTheme.bodyFont)
            if !searchText.isEmpty {
                Button {
                    searchText = ""
                } label: {
                    Image(systemName: "xmark.circle.fill")
                }
                .buttonStyle(.plain)
                .foregroundStyle(MereRunTheme.textMuted)
                .accessibilityLabel("Clear search")
            }
        }
        .padding(10)
        .merePanel()
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
    }

    private var emptyState: some View {
        VStack(spacing: 10) {
            Image(systemName: "rectangle.stack")
                .font(.system(size: 30, weight: .semibold))
                .foregroundStyle(MereRunTheme.textMuted)
            Text(items.isEmpty ? "Runs you create will land here." : "No matching runs.")
                .font(MereRunTheme.bodyFont)
                .foregroundStyle(MereRunTheme.textMuted)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(22)
    }

    private var list: some View {
        ScrollView {
            LazyVStack(spacing: 8) {
                ForEach(filteredItems) { item in
                    StudioLibraryRow(
                        item: item,
                        isSelected: selectedID == item.id
                    ) {
                        selectedID = item.id
                    }
                    .contextMenu {
                        Button("Rename") {
                            renameText = item.displayTitle
                            renamingID = item.id
                        }
                        Button("Delete", role: .destructive) {
                            onDelete(item.id)
                        }
                    }
                }
            }
            .padding(12)
        }
    }
}

private struct StudioLibraryRow: View {
    let item: StudioLibraryItem
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 10) {
                thumbnail
                    .frame(width: 46, height: 38)
                    .background(MereRunTheme.surface)
                    .clipShape(RoundedRectangle(cornerRadius: 7))

                VStack(alignment: .leading, spacing: 4) {
                    Text(item.displayTitle)
                        .font(.system(size: 12, weight: .semibold))
                        .lineLimit(1)
                    Text("\(item.mode.title) · \(item.status.rawValue)")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                }
                Spacer()
            }
            .padding(9)
            .background {
                RoundedRectangle(cornerRadius: 9)
                    .fill(isSelected ? MereRunTheme.surfaceRaised : Color.clear)
            }
        }
        .buttonStyle(.plain)
    }

    @ViewBuilder
    private var thumbnail: some View {
        if let url = item.outputURL, StudioOutputFileKind.classify(url) == .image {
            StudioAsyncImagePreview(
                url: url,
                maxPixelSize: 160,
                contentMode: .fill,
                fallbackSystemImage: item.mode.systemImage
            )
        } else {
            Image(systemName: item.mode.systemImage)
                .font(.system(size: 17, weight: .semibold))
                .foregroundStyle(MereRunTheme.accent)
        }
    }
}

private extension String {
    func maskingAPIKeyValue() -> String {
        var words = ShellWords.split(self)
        words = words.maskingSecrets()
        return words.shellQuoted()
    }
}
