import AppKit
import SwiftUI
import UniformTypeIdentifiers

struct StudioRootView: View {
    private static let orderedModes = StudioModeGroup.allCases.flatMap(\.modes)

    @EnvironmentObject private var controller: MereRunController
    @StateObject private var library = StudioLibraryStore()
    // Persisted per scene so relaunch restores the last mode and panel layout.
    @SceneStorage("studio.mode") private var mode: StudioMode = .createImage
    @SceneStorage("studio.showLibrary") private var showLibrary = true
    @SceneStorage("studio.showAdvanced") private var showAdvanced = false
    @State private var layoutClass: StudioLayoutClass = .regular
    @State private var showCompactLibrary = false
    @State private var draft = StudioDraft()
    @State private var showOptions = false
    @State private var showModels = false
    @State private var showHelp = false
    @State private var advancedWidth: CGFloat = 560
    @State private var advancedDragStartWidth: CGFloat?
    @State private var advancedDetached = false
    @State private var isDropTargeted = false
    @State private var selectedLibraryID: UUID?
    /// The conversation the canvas/composer targets in chat/code modes. nil means a fresh,
    /// not-yet-sent conversation (so the library has no empty row until the first message).
    @State private var activeConversationID: UUID?
    @State private var pendingPullRefresh: StudioReadinessRefresh?
    @State private var studioError: String?
    /// Locally installed models feeding the composer's quick-picker.
    @State private var installedModels: [StudioModelInventoryRow] = []
    @AppStorage("mererun.app.hasCompletedWelcome") private var hasCompletedWelcome = false
    @State private var showWelcome = false
    @FocusState private var promptFocused: Bool

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

    // The body is staged (shell → panels → sheets → observers) so each stage stays a small,
    // independently type-checked expression.
    var body: some View {
        observedShell
    }

    private var shell: some View {
        GeometryReader { geometry in
            let layout = StudioLayoutPolicy.layoutClass(for: geometry.size.width)

            adaptiveShell(layout: layout, availableWidth: geometry.size.width)
                .onAppear { updateLayoutClass(layout) }
                .onChange(of: layout) { _, nextLayout in
                    updateLayoutClass(nextLayout)
                }
        }
        .background(MereRunTheme.background.ignoresSafeArea())
    }

    private func adaptiveShell(layout: StudioLayoutClass, availableWidth: CGFloat) -> some View {
        ZStack(alignment: .trailing) {
            HStack(spacing: 0) {
                if !layout.isCompact {
                    regularNavigation

                    Divider()
                        .overlay(MereRunTheme.border.opacity(0.4))
                }

                if !layout.isCompact, showLibrary {
                    regularLibrary

                    Divider()
                        .overlay(MereRunTheme.border.opacity(0.5))
                }

                contentColumn(isCompact: layout.isCompact)
            }

            if layout.isCompact, showCompactLibrary {
                compactDismissLayer { showCompactLibrary = false }

                HStack(spacing: 0) {
                    compactLibraryPanel(availableWidth: availableWidth)
                    Spacer(minLength: 0)
                }
                .transition(.move(edge: .leading).combined(with: .opacity))
                .zIndex(1)
            }

            if showAdvanced {
                advancedPanel(layout: layout, availableWidth: availableWidth)
                    .transition(.move(edge: .trailing).combined(with: .opacity))
                    .zIndex(2)
            }
        }
        .background(MereRunTheme.background.ignoresSafeArea())
    }

    private var regularNavigation: some View {
        StudioSidebar(
            mode: $mode,
            modeCapabilities: modeCapabilities,
            serverStatus: controller.serverStatus,
            resolvedCLI: controller.resolvedCLI,
            onShowModels: { showModels = true },
            onShowHelp: { showHelp = true }
        )
        .frame(width: StudioLayoutPolicy.sidebarWidth)
    }

    private var regularLibrary: some View {
        StudioLibraryPanel(
            items: library.items,
            selectedID: $selectedLibraryID,
            isVisible: $showLibrary,
            onDelete: deleteLibraryItem,
            onRename: library.rename,
            onQuickLook: { QuickLookCoordinator.shared.preview($0) }
        )
        .frame(width: StudioLayoutPolicy.libraryWidth)
    }

    private func compactLibraryPanel(availableWidth: CGFloat) -> some View {
        StudioLibraryPanel(
            items: library.items,
            selectedID: compactLibrarySelection,
            isVisible: $showCompactLibrary,
            onDelete: deleteLibraryItem,
            onRename: library.rename,
            onQuickLook: { QuickLookCoordinator.shared.preview($0) }
        )
        .frame(
            width: StudioLayoutPolicy.compactPanelWidth(
                availableWidth: availableWidth,
                preferredWidth: 320
            )
        )
        .clipShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.xl))
        .mereShadow(radius: 24, y: 0)
        .padding(StudioLayoutPolicy.compactPanelInset)
    }

    private func compactDismissLayer(action: @escaping () -> Void) -> some View {
        MereRunTheme.textPrimary.opacity(0.06)
            .contentShape(Rectangle())
            .onTapGesture(perform: action)
            .transition(.opacity)
            .zIndex(0)
            .accessibilityHidden(true)
    }

    private func advancedPanel(layout: StudioLayoutClass, availableWidth: CGFloat) -> some View {
        Group {
            if layout.isCompact {
                AdvancedControlSurface(
                    docked: true,
                    onClose: { showAdvanced = false }
                )
                    .frame(
                        width: StudioLayoutPolicy.compactPanelWidth(
                            availableWidth: availableWidth,
                            preferredWidth: advancedWidth
                        )
                    )
                    .clipped()
            } else {
                HStack(spacing: 0) {
                    advancedResizeHandle
                    AdvancedControlSurface(
                        docked: true,
                        onDetach: { advancedDetached = true },
                        onClose: { showAdvanced = false }
                    )
                        .frame(width: advancedWidth)
                        .clipped()
                }
            }
        }
        .background(MereRunTheme.background)
        .mereShadow(radius: 24, y: 0)
    }

    private var compactLibrarySelection: Binding<UUID?> {
        Binding(
            get: { selectedLibraryID },
            set: { nextSelection in
                selectedLibraryID = nextSelection
                if nextSelection != nil { showCompactLibrary = false }
            }
        )
    }

    private var visibleLibraryBinding: Binding<Bool> {
        Binding(
            get: { layoutClass.isCompact ? showCompactLibrary : showLibrary },
            set: { isVisible in
                if layoutClass.isCompact {
                    showCompactLibrary = isVisible
                    if isVisible { showAdvanced = false }
                } else {
                    showLibrary = isVisible
                }
            }
        )
    }

    private func deleteLibraryItem(_ id: UUID) {
        library.delete(id: id)
        if selectedLibraryID == id { selectedLibraryID = nil }
    }

    private func updateLayoutClass(_ nextLayout: StudioLayoutClass) {
        guard layoutClass != nextLayout else { return }
        layoutClass = nextLayout
        if nextLayout == .regular { showCompactLibrary = false }
    }

    private func contentColumn(isCompact: Bool) -> some View {
        VStack(spacing: 0) {
            topBar(isCompact: isCompact)

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

            canvas(isCompact: isCompact)

            composer(isCompact: isCompact)
                .padding(.horizontal, isCompact ? 12 : 24)
                .padding(.bottom, isCompact ? 12 : 20)
        }
        .frame(minWidth: isCompact ? 0 : StudioLayoutPolicy.minimumCanvasWidth)
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
        .dropDestination(for: URL.self) { urls, _ in
            guard !mode.acceptedTypes.isEmpty, let url = urls.first(where: \.isFileURL) else {
                return false
            }
            draft.inputPath = url.path
            studioError = nil
            return true
        } isTargeted: { targeted in
            withAnimation(MereRunTheme.Motion.quick) {
                isDropTargeted = targeted && !mode.acceptedTypes.isEmpty
            }
        }
        .overlay {
            if isDropTargeted {
                RoundedRectangle(cornerRadius: MereRunTheme.Radius.xl)
                    .strokeBorder(MereRunTheme.accent, style: StrokeStyle(lineWidth: 2, dash: [8, 6]))
                    .padding(8)
                    .allowsHitTesting(false)
                    .transition(.opacity)
            }
        }
        .onPasteCommand(of: [.image]) { _ in pasteImageFromClipboard() }
    }

    private func canvas(isCompact: Bool) -> some View {
        StudioCanvas(
            mode: mode,
            isCompact: isCompact,
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
            onQuickLook: {
                if let url = selectedItem?.outputURL { QuickLookCoordinator.shared.preview(url) }
            },
            onPullModel: pullModel,
            onShowDetails: { showAdvanced = true },
            onNewChat: startNewConversation,
            onCopy: copyToClipboard,
            onRetry: retryLastTurn,
            onEdit: editMessage,
            onStop: controller.cancel,
            onUseExample: useExamplePrompt,
            onAttach: chooseAttachment
        )
    }

    private func composer(isCompact: Bool) -> some View {
        StudioComposer(
            mode: mode,
            isCompact: isCompact,
            draft: $draft,
            showOptions: $showOptions,
            isRunning: controller.isRunning,
            queuedCount: controller.queuedRunCount,
            readiness: readiness,
            sendBlocked: mode.isConversational && activeConversationRunning,
            installedModels: installedModels,
            promptFocus: $promptFocused,
            onRun: runStudioCommand,
            onStop: controller.cancel,
            onAttach: chooseAttachment,
            onPaste: pasteImageFromClipboard,
            onShowModels: { showModels = true }
        )
    }

    private var sheetedShell: some View {
        shell
            .sheet(isPresented: $showModels) {
                StudioModelsSheet(onModelsChanged: {
                    refreshReadiness()
                    refreshInstalledModels()
                })
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
            .focusedSceneValue(\.showLibrary, visibleLibraryBinding)
            .focusedSceneValue(\.showAdvanced, $showAdvanced)
            .focusedSceneValue(\.showModels, $showModels)
    }

    private var lifecycleShell: some View {
        sheetedShell
        .task {
            // Poll the local server status for the sidebar status cluster. status has a 1s probe
            // timeout, so a modest cadence keeps it live without hammering the CLI.
            while !Task.isCancelled {
                await controller.refreshServerStatus()
                try? await Task.sleep(nanoseconds: 20 * 1_000_000_000)
            }
        }
        .onAppear {
            draft = freshDraft(for: mode)
            // mode may be restored from @SceneStorage (e.g. chat) — onChange(of:mode) doesn't fire
            // for the initial value, so replicate the conversational setup here to open the latest
            // thread instead of leaving the canvas blank.
            if mode.isConversational {
                let latest = library.items.first { $0.mode == mode && $0.isConversation }
                activeConversationID = latest?.id
                selectedLibraryID = latest?.id
                if let latest { applyConversationSettings(from: latest, to: &draft) }
                draft.prompt = ""
            } else {
                selectedLibraryID = library.items.first { $0.mode == mode }?.id
            }
            controller.checkReadiness(for: mode, draft: draft)
            refreshInstalledModels()
            if !hasCompletedWelcome {
                showWelcome = true
            } else if mode != .listen {
                promptFocused = true
            }
        }
    }

    private var navigationObservedShell: some View {
        lifecycleShell
        .onChange(of: showAdvanced) { _, isShown in
            if isShown {
                if layoutClass.isCompact { showCompactLibrary = false }
                syncAdvancedToStudio()
            }
        }
        .onChange(of: mode) { _, newMode in
            var nextDraft = freshDraft(for: newMode)
            studioError = nil
            if newMode.isConversational {
                // Open the most recent thread for this mode (or a fresh one) and reuse its
                // system/model so follow-ups match; the composer starts empty.
                let latest = library.items.first { $0.mode == newMode && $0.isConversation }
                activeConversationID = latest?.id
                selectedLibraryID = latest?.id
                if let latest { applyConversationSettings(from: latest, to: &nextDraft) }
                nextDraft.prompt = ""
            } else {
                activeConversationID = nil
                selectedLibraryID = library.items.first { $0.mode == newMode }?.id
            }
            draft = nextDraft
            controller.checkReadiness(for: newMode, draft: draft)
            if newMode != .listen { promptFocused = true }
        }
        .onChange(of: controller.recommendedChatModelID) { _, _ in
            guard mode == .chat, activeConversationID == nil else { return }
            controller.applyRecommendedDefaults(to: &draft, for: mode)
            refreshReadiness()
        }
        .onChange(of: controller.recommendedCodeModelID) { _, _ in
            guard mode == .code, activeConversationID == nil else { return }
            controller.applyRecommendedDefaults(to: &draft, for: mode)
            refreshReadiness()
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
    }

    private var validationObservedShell: some View {
        navigationObservedShell
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
    }

    private var observedShell: some View {
        validationObservedShell
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

            if mutatedModels {
                refreshInstalledModels()
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

    /// The slim, contextual header for the content column: which mode you're in, and the two
    /// panel toggles. Machine-wide status lives in the sidebar; blocking states own the canvas.
    private func topBar(isCompact: Bool) -> some View {
        HStack(spacing: MereRunTheme.Spacing.sm) {
            if isCompact {
                compactModePicker
            } else {
                Image(systemName: mode.systemImage)
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(MereRunTheme.accent)

                VStack(alignment: .leading, spacing: 1) {
                    Text(mode.title)
                        .font(.system(size: 15, weight: .semibold))
                    Text(mode.subtitle)
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(MereRunTheme.textMuted)
                }
            }

            Spacer()

            Button {
                withAnimation(MereRunTheme.Motion.standard) {
                    toggleLibrary(isCompact: isCompact)
                }
            } label: {
                Image(systemName: "rectangle.stack")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(
                        libraryIsVisible(isCompact: isCompact)
                            ? MereRunTheme.accent
                            : MereRunTheme.textSecondary
                    )
                    .frame(width: 30, height: 30)
            }
            .buttonStyle(.mereIcon)
            .help(libraryIsVisible(isCompact: isCompact) ? "Hide library (⌃⌘L)" : "Show library (⌃⌘L)")
            .accessibilityLabel(libraryIsVisible(isCompact: isCompact) ? "Hide library" : "Show library")

            Button {
                withAnimation(MereRunTheme.Motion.standard) {
                    if isCompact { showCompactLibrary = false }
                    showAdvanced.toggle()
                }
            } label: {
                Image(systemName: "slider.horizontal.3")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(showAdvanced ? MereRunTheme.accent : MereRunTheme.textSecondary)
                    .frame(width: 30, height: 30)
            }
            .buttonStyle(.mereIcon)
            .help(showAdvanced ? "Hide Advanced (⌃⌘E)" : "Show Advanced (⌃⌘E)")
            .accessibilityLabel(showAdvanced ? "Hide Advanced" : "Show Advanced")
        }
        .padding(.horizontal, isCompact ? MereRunTheme.Spacing.md : MereRunTheme.Spacing.lg)
        .frame(height: 52)
        .background(VisualEffectBackground())
    }

    private var compactModePicker: some View {
        Menu {
            ForEach(StudioModeGroup.allCases) { group in
                Section(group.rawValue) {
                    ForEach(group.modes) { candidate in
                        Button {
                            mode = candidate
                        } label: {
                            if candidate == mode {
                                Label(candidate.title, systemImage: "checkmark")
                            } else {
                                Label(candidate.title, systemImage: candidate.systemImage)
                            }
                        }
                        .disabled(modeCapabilities[candidate]?.unavailableMessage != nil)
                        .modifier(
                            StudioModeShortcut(
                                number: Self.orderedModes.firstIndex(of: candidate).flatMap { index in
                                    index < 9 ? index + 1 : nil
                                }
                            )
                        )
                    }
                }
            }
        } label: {
            HStack(spacing: 8) {
                Image(systemName: mode.systemImage)
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(MereRunTheme.accent)
                    .frame(width: 20)
                Text(mode.title)
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(MereRunTheme.textPrimary)
                    .lineLimit(1)
                Image(systemName: "chevron.up.chevron.down")
                    .font(.system(size: 8, weight: .semibold))
                    .foregroundStyle(MereRunTheme.textMuted)
            }
            .padding(.horizontal, 10)
            .frame(height: 30)
            .background {
                RoundedRectangle(cornerRadius: MereRunTheme.Radius.md)
                    .fill(MereRunTheme.surface.opacity(0.62))
                    .overlay {
                        RoundedRectangle(cornerRadius: MereRunTheme.Radius.md)
                            .strokeBorder(MereRunTheme.border.opacity(0.45), lineWidth: 1)
                    }
            }
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .help("Switch creation mode")
        .accessibilityLabel("Mode")
        .accessibilityValue(mode.title)
    }

    private func libraryIsVisible(isCompact: Bool) -> Bool {
        isCompact ? showCompactLibrary : showLibrary
    }

    private func toggleLibrary(isCompact: Bool) {
        if isCompact {
            showAdvanced = false
            showCompactLibrary.toggle()
        } else {
            showLibrary.toggle()
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
            let status: StudioLibraryStatus = controller.isRunning || controller.queuedRunCount > 0
                ? .queued
                : .running
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
            .frame(width: 4)
            .contentShape(Rectangle().inset(by: -3))
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

    private func freshDraft(for mode: StudioMode) -> StudioDraft {
        var nextDraft = StudioDraft()
        nextDraft.reset(for: mode)
        controller.applyRecommendedDefaults(to: &nextDraft, for: mode)
        return nextDraft
    }

    /// Fills the composer from an empty-state example and hands it focus — never auto-runs.
    private func useExamplePrompt(_ example: String) {
        draft.prompt = example
        studioError = nil
        promptFocused = true
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
            promptFocused = true
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
        promptFocused = true
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
            // The canvas running overlay shows pull progress (bytes, speed, cancel) in place,
            // so the Advanced console no longer needs to open for a pull.
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

    /// Refreshes the composer's installed-model quick-picker from `model list`.
    private func refreshInstalledModels() {
        Task {
            let result = await controller.utilityCommandResult(args: ["model", "list"])
            guard result.exitCode == 0 else { return }
            installedModels = StudioModelInventoryParser.rows(from: result.stdout).filter(\.isInstalled)
        }
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

    /// Pastes an image from the clipboard into the attachment (Edit ▸ Paste / ⌘V when the canvas,
    /// not a text field, holds focus). Prefers a pasted image file; otherwise writes the pasted
    /// bitmap to a temporary PNG. Only acts in modes that accept an image.
    private func pasteImageFromClipboard() {
        guard mode.acceptedTypes.contains(.image) else { return }
        let pasteboard = NSPasteboard.general

        if let urls = pasteboard.readObjects(
            forClasses: [NSURL.self],
            options: [.urlReadingFileURLsOnly: true]
        ) as? [URL], let imageURL = urls.first(where: { StudioOutputFileKind.classify($0) == .image }) {
            draft.inputPath = imageURL.path
            studioError = nil
            return
        }

        guard let image = NSImage(pasteboard: pasteboard), let data = image.pngDataRepresentation() else {
            studioError = "The clipboard has no image to paste."
            return
        }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("pasted-\(UUID().uuidString).png")
        do {
            try data.write(to: url)
            draft.inputPath = url.path
            studioError = nil
        } catch {
            studioError = "Could not paste image: \(error.localizedDescription)"
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

private extension String {
    func maskingAPIKeyValue() -> String {
        var words = ShellWords.split(self)
        words = words.maskingSecrets()
        return words.shellQuoted()
    }
}

private extension NSImage {
    /// PNG encoding of the image (for persisting a pasted bitmap to a temp file).
    func pngDataRepresentation() -> Data? {
        guard let tiff = tiffRepresentation,
              let bitmap = NSBitmapImageRep(data: tiff) else { return nil }
        return bitmap.representation(using: .png, properties: [:])
    }
}
