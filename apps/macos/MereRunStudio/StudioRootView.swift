import AppKit
import SwiftUI
import UniformTypeIdentifiers

/// The Studio shell: a native split view whose sidebar lists the domains and whose detail area
/// renders the current task — the prompt workspace for the twelve composer modes, or one of the
/// former specialist sheets re-hosted inline. Navigation is one value (`NavigationModel`), and
/// the only sheets left are true tasks (terms, the mask editor, Relay sign-in, rename, the Guide).
struct StudioRootView: View {
    @EnvironmentObject private var controller: MereRunController
    @EnvironmentObject private var library: StudioLibraryStore
    @EnvironmentObject private var navigation: NavigationModel
    @Environment(\.openWindow) private var openWindow
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    // Persisted per scene so relaunch restores the last place, the last prompt mode, and the panel
    // layout. `studio.mode` keeps its v1 meaning (the last prompt mode) so drafts and readiness
    // stay attached to it while a System or Lab task is shown.
    @SceneStorage("studio.destination") private var storedDestination: StudioDestination = .default
    @SceneStorage("studio.mode") private var lastPromptMode: StudioMode = .createImage
    @SceneStorage("studio.showLibrary") private var storedShowLibrary = true
    @SceneStorage("studio.libraryScope") private var libraryScope: StudioLibraryScope = .domain
    @State private var columnVisibility: NavigationSplitViewVisibility = .all
    /// The prompt mode the composer, canvas, and readiness were last set up for, so the
    /// destination observer activates a mode exactly once however the destination arrived.
    @State private var activatedMode: StudioMode?
    @State private var draft = StudioDraft()
    @State private var showOptions = false
    @State private var isDropTargeted = false
    /// The conversation the canvas/composer targets in chat/code modes. nil means a fresh,
    /// not-yet-sent conversation (so the library has no empty row until the first message).
    @State private var activeConversationID: UUID?
    @State private var pendingPullRefresh: StudioReadinessRefresh?
    @State private var pendingRestrictedPull: StudioRunRequest?
    @State private var studioError: String?
    /// Every `model list` row, installed or not, feeding the composer's model chip.
    @State private var modelInventory: [StudioModelInventoryRow] = []
    @State private var modelUsageTermsByID: [String: StudioModelUsageTerms] = [:]
    @State private var imageDatasetTask: StudioUtilityTask = .datasetDiscovery
    @AppStorage("mererun.app.hasCompletedWelcome") private var hasCompletedWelcome = false
    @FocusState private var promptFocused: Bool
    /// Drafts a prompt mode starts with instead of its defaults. The snapshot harness stages
    /// sample content this way; the app passes nothing.
    private let seededDrafts: [StudioMode: StudioDraft]

    init(seededDrafts: [StudioMode: StudioDraft] = [:]) {
        self.seededDrafts = seededDrafts
    }

    private var destination: StudioDestination { navigation.destination }

    /// The prompt mode the composer and canvas belong to. While a non-prompt task is shown this is
    /// the last prompt mode, so its draft, readiness, and conversation survive the detour.
    private var mode: StudioMode {
        destination.task.mode ?? lastPromptMode
    }

    private var showsPromptWorkspace: Bool {
        destination.task.mode != nil
    }

    /// The Library column belongs to domains with a prompt workspace; System pages and the
    /// form-shaped domains (3D, Earth, Text) use the full width.
    private var showsLibraryColumn: Bool {
        navigation.showLibrary && destination.domain.hasPromptWorkspace
    }

    private var selectedItem: StudioLibraryItem? {
        if let selectedLibraryID = navigation.selectedLibraryID,
           let found = library.items.first(where: { $0.id == selectedLibraryID }),
           found.mode == mode {
            return found
        }
        if mode.isConversational {
            guard let activeConversationID else { return nil }
            return library.items.first { $0.id == activeConversationID && $0.isConversation }
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

    /// The seed the mode's most recent run was queued with, for the seed chip's "Reuse last".
    private var lastSeed: String? {
        library.items.lazy
            .filter { $0.mode == mode }
            .compactMap { $0.commandDraft?.seed.trimmingCharacters(in: .whitespacesAndNewlines) }
            .first { !$0.isEmpty }
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

    /// Domains whose default task needs a managed model this machine cannot run.
    private var domainUnavailableMessages: [StudioDomain: String] {
        var messages: [StudioDomain: String] = [:]
        for domain in StudioDomain.allCases {
            guard let candidate = domain.defaultTask.mode else { continue }
            var candidateDraft = StudioDraft()
            candidateDraft.reset(for: candidate)
            let requirement = StudioCommandAdapter.capabilityRequirement(for: candidate, draft: candidateDraft)
            guard let requirement,
                  case .managedModel(let modelID) = requirement,
                  let message = controller.modelCapabilitiesByID[modelID]?.unavailableMessage else {
                continue
            }
            messages[domain] = message
        }
        return messages
    }

    private var showLibraryBinding: Binding<Bool> {
        Binding(
            get: { navigation.showLibrary },
            set: { navigation.showLibrary = $0 }
        )
    }

    private var domainBinding: Binding<StudioDomain> {
        Binding(
            get: { destination.domain },
            set: { navigation.open(domain: $0) }
        )
    }

    private var taskBinding: Binding<StudioTask> {
        Binding(
            get: { destination.task },
            set: { navigation.open(task: $0) }
        )
    }

    // The body is staged (shell → presentation → observers) so each stage stays a small,
    // independently type-checked expression.
    var body: some View {
        observedShell
    }

    // MARK: - Shell

    private var shell: some View {
        NavigationSplitView(columnVisibility: $columnVisibility) {
            StudioSidebar(
                selectedDomain: domainBinding,
                unavailableMessages: domainUnavailableMessages,
                serverStatus: controller.serverStatus,
                resolvedCLI: controller.resolvedCLI,
                onShowServer: { navigation.open(domain: .server) },
                onShowModels: { navigation.open(task: .modelsInstalled) }
            )
        } detail: {
            detailArea
                .toolbar { toolbarContent }
        }
        .background(MereRunTheme.background.ignoresSafeArea())
    }

    private var detailArea: some View {
        VStack(spacing: 0) {
            banners

            HStack(spacing: 0) {
                if showsLibraryColumn {
                    libraryColumn
                        .frame(width: StudioLayoutPolicy.libraryWidth)
                        .transition(reduceMotion ? .identity : .move(edge: .leading).combined(with: .opacity))

                    Divider()
                        .overlay(MereRunTheme.border.opacity(0.5))
                }

                domainContent
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .background(MereRunTheme.background.ignoresSafeArea())
        .foregroundStyle(MereRunTheme.textPrimary)
    }

    @ViewBuilder
    private var banners: some View {
        if let persistenceError = library.lastPersistenceError {
            MereBanner(
                severity: .warning,
                text: "Run history not saved: \(persistenceError)"
            )
            .padding(.horizontal, MereRunTheme.Spacing.lg)
            .padding(.top, MereRunTheme.Spacing.sm)
        }

        if !hasCompletedWelcome {
            MereBanner(
                severity: .info,
                text: "Nothing leaves your Mac. Pick a domain, get its model once, and create.",
                systemImage: "lock.shield",
                onDismiss: { hasCompletedWelcome = true }
            )
            .padding(.horizontal, MereRunTheme.Spacing.lg)
            .padding(.top, MereRunTheme.Spacing.sm)
        }
    }

    private var libraryColumn: some View {
        StudioLibraryPanel(
            items: library.items,
            domain: destination.domain,
            scope: $libraryScope,
            progressByID: controller.progressByRequestID,
            selectedID: $navigation.selectedLibraryID,
            onSelect: selectLibraryItem,
            onDelete: deleteLibraryItem,
            onRename: library.rename,
            onQuickLook: { QuickLookCoordinator.shared.preview($0) },
            onRetry: retryLibraryItem,
            onEdit: editLibraryItem
        )
    }

    // MARK: - Toolbar

    @ToolbarContentBuilder
    private var toolbarContent: some ToolbarContent {
        ToolbarItem(placement: .navigation) {
            domainTitle
        }

        ToolbarItem(placement: .principal) {
            taskControl
        }

        ToolbarItemGroup(placement: .primaryAction) {
            if destination.domain.hasPromptWorkspace {
                Button {
                    toggleLibrary()
                } label: {
                    Image(systemName: "sidebar.squares.left")
                        .foregroundStyle(navigation.showLibrary ? MereRunTheme.accent : MereRunTheme.textSecondary)
                }
                .help(navigation.showLibrary ? "Hide Library (⌥⌘L)" : "Show Library (⌥⌘L)")
                .accessibilityLabel(navigation.showLibrary ? "Hide Library" : "Show Library")
            }

            Button {
                openConsole()
            } label: {
                Image(systemName: "terminal")
            }
            .help("Command Console (⌥⌘C)")
            .accessibilityLabel("Command Console")
        }
    }

    private var domainTitle: some View {
        HStack(spacing: MereRunTheme.Spacing.sm) {
            Image(systemName: destination.domain.systemImage)
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(MereRunTheme.accent)
                .frame(width: 22)

            VStack(alignment: .leading, spacing: 1) {
                Text(destination.domain.title)
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(MereRunTheme.textPrimary)
                Text(destination.domain.subtitle)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(MereRunTheme.textMuted)
                    .lineLimit(1)
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(.isHeader)
    }

    @ViewBuilder
    private var taskControl: some View {
        switch StudioTaskControlStyle.style(for: destination.domain) {
        case .none:
            EmptyView()
        case .segmented:
            Picker("Task", selection: taskBinding) {
                ForEach(destination.domain.tasks) { task in
                    Text(task.title).tag(task)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .fixedSize()
            .accessibilityLabel("\(destination.domain.title) task")
        case .menu:
            Picker("Task", selection: taskBinding) {
                ForEach(destination.domain.tasks) { task in
                    Text(task.title).tag(task)
                }
            }
            .pickerStyle(.menu)
            .labelsHidden()
            .fixedSize()
            .accessibilityLabel("\(destination.domain.title) task")
        }
    }

    // MARK: - Domain content

    @ViewBuilder
    private var domainContent: some View {
        switch destination.task {
        case .imageGenerate, .videoGenerate, .musicCompose, .soundGenerate, .voiceSpeak,
             .chatChat, .chatCode, .visionRead, .visionFind, .visionSegment, .visionTrack,
             .audioTranscribe:
            promptWorkspace
        case .imageDatasets:
            StudioUtilityLabView(
                task: $imageDatasetTask,
                tasks: [.datasetDiscovery, .imageValidation, .runPlan],
                showsTaskPicker: true
            )
        case .imageTrain:
            StudioTrainingView(kind: .image)
        case .chatTrain:
            StudioTrainingView(kind: .text)
        case .musicTrain:
            StudioTrainingView(kind: .music)
        case .videoSubjects:
            StudioSCAILView()
        case .musicRealtime:
            StudioRealtimeMusicView(initialDraft: draft)
        case .musicAnalyze, .musicTranscribe:
            StudioMusicToolsView(tool: musicToolBinding, tools: [.analyze, .transcribe])
        case .musicSeparate:
            StudioAudioToolsView(tool: .constant(.separate))
        case .soundFoley, .soundCondition, .soundEncode, .soundDecode, .soundScore:
            StudioSFXLabView(
                task: sfxTaskBinding,
                tasks: [.video, .condition, .encode, .decode, .score],
                initialDraft: draft
            )
        case .voiceClone, .voiceVoices:
            StudioVoiceView(task: voiceTaskBinding, tasks: [.synthesize, .profiles], initialDraft: draft)
        case .threeDFromImage:
            Studio3DCreationView()
        case .visionDepth, .visionPose, .visionFaces, .visionFlow, .visionGeometry, .visionLive:
            StudioVisionLabView(task: visionLabBinding)
        case .audioWhoSpoke, .audioLive:
            StudioVoiceView(task: voiceTaskBinding, tasks: [.diarize, .listen], initialDraft: draft)
        case .audioEnhance, .audioSeparate:
            StudioAudioToolsView(tool: audioToolBinding)
        case .textEmbeddings, .textAnonymize:
            StudioUtilityLabView(
                task: utilityTaskBinding,
                tasks: [.embeddings, .anonymize],
                showsTaskPicker: false
            )
        case .earthFlood, .earthFire, .earthTessera, .earthOlmoEarth:
            StudioGeoLabView(tool: geoToolBinding)
        case .modelsInstalled:
            StudioModelsView(onModelsChanged: {
                refreshReadiness()
                refreshInstalledModels()
            })
        case .modelsLocations:
            StudioModelLocationsView(onLocationsChanged: {
                refreshReadiness()
                refreshInstalledModels()
            })
        case .modelsHealth:
            StudioModelHealthView(scope: .health, onModelsChanged: {
                refreshReadiness()
                refreshInstalledModels()
            })
        case .modelsBenchmarks:
            StudioModelHealthView(scope: .benchmarks, onModelsChanged: {
                refreshReadiness()
                refreshInstalledModels()
            })
        case .modelsAdapters:
            StudioAdaptersView(
                activeModelID: draft.model,
                onUse: applyAdapter,
                onUseLocal: applyLocalAdapter,
                onTrain: openTraining
            )
        case .serverServing:
            StudioServingConsoleView()
        case .serverMusic:
            StudioMusicToolsView(tool: .constant(.serve), tools: [.serve])
        case .runsRuns:
            StudioOperationsView()
        case .pluginsCatalog:
            StudioPluginsView()
        }
    }

    // MARK: Task bindings for re-hosted views

    private var musicToolBinding: Binding<StudioMusicTool> {
        Binding(
            get: {
                switch destination.task {
                case .musicTranscribe: return .transcribe
                default: return .analyze
                }
            },
            set: { tool in
                switch tool {
                case .analyze: navigation.open(task: .musicAnalyze)
                case .transcribe: navigation.open(task: .musicTranscribe)
                case .serve: navigation.open(task: .serverMusic)
                }
            }
        )
    }

    private var audioToolBinding: Binding<StudioAudioTool> {
        Binding(
            get: { destination.task == .audioSeparate ? .separate : .enhance },
            set: { navigation.open(task: $0 == .separate ? .audioSeparate : .audioEnhance) }
        )
    }

    private var sfxTaskBinding: Binding<StudioSFXTask> {
        Binding(
            get: {
                switch destination.task {
                case .soundCondition: return .condition
                case .soundEncode: return .encode
                case .soundDecode: return .decode
                case .soundScore: return .score
                default: return .video
                }
            },
            set: { task in
                switch task {
                case .generate: navigation.open(task: .soundGenerate)
                case .video: navigation.open(task: .soundFoley)
                case .condition: navigation.open(task: .soundCondition)
                case .encode: navigation.open(task: .soundEncode)
                case .decode: navigation.open(task: .soundDecode)
                case .score: navigation.open(task: .soundScore)
                }
            }
        )
    }

    private var voiceTaskBinding: Binding<StudioVoiceTask> {
        Binding(
            get: {
                switch destination.task {
                case .voiceVoices: return .profiles
                case .audioWhoSpoke: return .diarize
                case .audioLive: return .listen
                case .audioTranscribe: return .transcribe
                default: return .synthesize
                }
            },
            set: { task in
                switch task {
                case .synthesize: navigation.open(task: .voiceClone)
                case .profiles: navigation.open(task: .voiceVoices)
                case .transcribe: navigation.open(task: .audioTranscribe)
                case .diarize: navigation.open(task: .audioWhoSpoke)
                case .listen: navigation.open(task: .audioLive)
                }
            }
        )
    }

    private var utilityTaskBinding: Binding<StudioUtilityTask> {
        Binding(
            get: { destination.task == .textAnonymize ? .anonymize : .embeddings },
            set: { task in
                switch task {
                case .embeddings: navigation.open(task: .textEmbeddings)
                case .anonymize: navigation.open(task: .textAnonymize)
                case .imageValidation, .datasetDiscovery, .runPlan:
                    imageDatasetTask = task
                    navigation.open(task: .imageDatasets)
                }
            }
        )
    }

    private var geoToolBinding: Binding<StudioGeoTool> {
        Binding(
            get: {
                switch destination.task {
                case .earthFire: return .fire
                case .earthTessera: return .tessera
                case .earthOlmoEarth: return .olmoEarth
                default: return .flood
                }
            },
            set: { tool in
                switch tool {
                case .flood: navigation.open(task: .earthFlood)
                case .fire: navigation.open(task: .earthFire)
                case .tessera: navigation.open(task: .earthTessera)
                case .olmoEarth: navigation.open(task: .earthOlmoEarth)
                }
            }
        )
    }

    private var visionLabBinding: Binding<StudioVisionTask> {
        Binding(
            get: { navigation.visionLabTask },
            set: { navigation.selectVisionLabVariant($0) }
        )
    }

    // MARK: - Prompt workspace

    private var promptWorkspace: some View {
        VStack(spacing: 0) {
            canvas

            composer
        }
        .frame(minWidth: StudioLayoutPolicy.minimumCanvasWidth)
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
            // A file dropped anywhere on the canvas lands in the first well slot that takes it.
            guard draft.attach(dropped: urls, for: mode) else { return false }
            studioError = nil
            return true
        } isTargeted: { targeted in
            withAnimation(MereRunTheme.Motion.quick) {
                isDropTargeted = targeted && !mode.attachmentSlots.isEmpty
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

    private var canvas: some View {
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
            progress: selectedItem.flatMap { controller.progressByRequestID[$0.id] }
                ?? controller.currentProgress,
            onOpen: openSelectedOutput,
            onReveal: revealSelectedOutput,
            onQuickLook: {
                if let url = selectedItem?.outputURL { QuickLookCoordinator.shared.preview(url) }
            },
            onPullModel: pullModel,
            onShowDetails: { openConsole() },
            onNewChat: startNewConversation,
            onCopy: copyToClipboard,
            onRetry: retryLastTurn,
            onRerunItem: retryLibraryItem,
            onEditRun: editLibraryItem,
            onEdit: editMessage,
            onStop: controller.cancel,
            onUseExample: useExamplePrompt,
            onAttach: chooseAttachment
        )
    }

    private var composer: some View {
        StudioComposer(
            mode: mode,
            draft: $draft,
            showOptions: $showOptions,
            isRunning: controller.isRunning,
            queuedCount: controller.queuedRunCount,
            readiness: readiness,
            sendBlocked: mode.isConversational && activeConversationRunning,
            modelInventory: modelInventory,
            lastSeed: lastSeed,
            promptFocus: $promptFocused,
            onRun: runStudioCommand,
            onStop: controller.cancel,
            onShowModels: { navigation.open(task: .modelsInstalled) },
            onShowAdapters: {
                showOptions = false
                navigation.open(task: .modelsAdapters)
            },
            onShowRealtimeMusic: {
                showOptions = false
                navigation.open(task: .musicRealtime)
            }
        )
    }

    // MARK: - Presentation

    private var presentedShell: some View {
        shell
            .sheet(isPresented: $navigation.showGuide) {
                StudioHelpSheet()
                    .environmentObject(controller)
                    .frame(width: 720, height: 560)
            }
            .alert(
                "Accept third-party model terms",
                isPresented: Binding(
                    get: { pendingRestrictedPull != nil },
                    set: { if !$0 { pendingRestrictedPull = nil } }
                ),
                presenting: pendingRestrictedPull
            ) { request in
                Button("Cancel", role: .cancel) {
                    pendingRestrictedPull = nil
                }
                Button("Accept & Download") {
                    pendingRestrictedPull = nil
                    startPull(request, acknowledgingUsageTerms: true)
                }
            } message: { request in
                let usageTerms = modelUsageTermsByID[request.draft.model]
                Text(
                    """
                    \(usageTerms?.summary ?? "This model has third-party usage terms.")

                    By continuing, you confirm that you reviewed and accept the listed terms and agree to comply with them. \
                    Mere does not determine whether your intended use is permitted. You are responsible for compliance.
                    """
                )
            }
            .alert(
                "Couldn’t open MereRun link",
                isPresented: Binding(
                    get: { navigation.deepLinkError != nil },
                    set: { if !$0 { navigation.deepLinkError = nil } }
                )
            ) {
                Button("OK") { navigation.deepLinkError = nil }
            } message: {
                Text(navigation.deepLinkError ?? "The MereRun link is invalid.")
            }
            .focusedSceneValue(\.studioActions, sceneActions)
    }

    private var sceneActions: StudioSceneActions {
        StudioSceneActions(
            destination: destination,
            showLibrary: showLibraryBinding,
            canShowLibrary: destination.domain.hasPromptWorkspace,
            open: { navigation.open(destination: $0) },
            openDomain: { navigation.open(domain: $0) },
            newChat: startNewConversation,
            canNewChat: showsPromptWorkspace && mode.isConversational,
            runComposer: runStudioCommand,
            canRun: showsPromptWorkspace && !readiness.blocksRun
                && !(mode.isConversational && activeConversationRunning),
            stop: controller.cancel,
            canStop: controller.isRunning,
            openConsole: { openConsole() },
            showGuide: { navigation.showGuide = true },
            importReceipt: importReceipt
        )
    }

    private var lifecycleShell: some View {
        presentedShell
        .task {
            // Poll the local server status for the sidebar status cluster. status has a 1s probe
            // timeout, so a modest cadence keeps it live without hammering the CLI.
            while !Task.isCancelled {
                await controller.refreshServerStatus()
                try? await Task.sleep(nanoseconds: 20 * 1_000_000_000)
            }
        }
        .onAppear {
            navigation.showLibrary = storedShowLibrary
            // Restore goes through the model so remembered tasks and the Vision Lab variant learn
            // the persisted destination; the reconciled prompt mode replaces a stale studio.mode.
            let restoredMode = navigation.restore(destination: storedDestination, lastPromptMode: lastPromptMode)
            lastPromptMode = restoredMode
            activateMode(restoredMode)
            refreshInstalledModels()
        }
        .onOpenURL { url in
            navigation.open(deepLink: url, library: library)
        }
    }

    private var navigationObservedShell: some View {
        lifecycleShell
        .onChange(of: navigation.destination) { _, next in
            storedDestination = next
            guard let nextMode = next.task.mode else { return }
            lastPromptMode = nextMode
            if nextMode != activatedMode {
                activateMode(nextMode)
            } else if nextMode != .listen {
                promptFocused = true
            }
        }
        .onChange(of: navigation.showLibrary) { _, isShown in
            storedShowLibrary = isShown
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
        .onChange(of: navigation.selectedLibraryID) { _, id in
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
                    navigation.selectedLibraryID = conversationID
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
                    commandPreview: result.commandPreview.maskingAPIKeyValue(),
                    artifactURLs: result.artifactURLs
                )
                navigation.selectedLibraryID = requestID
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
            navigation.selectedLibraryID = requestID
        }
        .onChange(of: controller.lastOutputURL) { _, outputURL in
            guard let requestID = controller.activeRunRequestID, let outputURL,
                  library.items.contains(where: { $0.id == requestID }) else { return }
            library.updateOutput(id: requestID, outputURL: outputURL)
            navigation.selectedLibraryID = requestID
        }
    }

    // MARK: - Navigation

    /// Switches the composer, canvas, readiness, and Library selection to `newMode`. Honors a
    /// Library row the user just picked (so selecting a row of another domain lands on that row),
    /// otherwise opens the most recent item or thread of the mode.
    private func activateMode(_ newMode: StudioMode) {
        activatedMode = newMode
        var nextDraft = seededDrafts[newMode] ?? freshDraft(for: newMode)
        studioError = nil
        let preferred = navigation.selectedLibraryID.flatMap { id in
            library.items.first { $0.id == id && $0.mode == newMode }
        }
        if newMode.isConversational {
            // Open the picked or most recent thread for this mode (or a fresh one) and reuse its
            // system/model so follow-ups match; the composer starts empty.
            let thread = preferred?.isConversation == true
                ? preferred
                : library.items.first { $0.mode == newMode && $0.isConversation }
            activeConversationID = thread?.id
            navigation.selectedLibraryID = thread?.id
            if let thread { applyConversationSettings(from: thread, to: &nextDraft) }
            nextDraft.prompt = ""
        } else {
            activeConversationID = nil
            navigation.selectedLibraryID = preferred?.id ?? library.items.first { $0.mode == newMode }?.id
        }
        draft = nextDraft
        controller.checkReadiness(for: newMode, draft: draft)
        if newMode != .listen { promptFocused = true }
    }

    /// A Library row the user clicked. Rows of another mode switch the destination first;
    /// `activateMode` then keeps the clicked row selected.
    private func selectLibraryItem(_ item: StudioLibraryItem) {
        navigation.selectedLibraryID = item.id
        guard item.mode != mode || !showsPromptWorkspace else { return }
        navigation.open(destination: item.mode.destination)
    }

    private func deleteLibraryItem(_ id: UUID) {
        library.delete(id: id)
        if navigation.selectedLibraryID == id { navigation.selectedLibraryID = nil }
    }

    /// Opens the Command Console window with the composer's draft carried into the Advanced
    /// template for the current mode, so the console deepens the current task. An already-open
    /// console is only raised: its edits and run state are never reset from here.
    private func openConsole(syncingComposer: Bool = true) {
        if navigation.shouldSyncComposerToConsole(requested: syncingComposer) {
            controller.syncAdvanced(to: mode, from: draft)
        }
        openWindow(id: StudioConsoleWindow.id)
    }

    private func toggleLibrary() {
        if reduceMotion {
            navigation.showLibrary.toggle()
        } else {
            withAnimation(MereRunTheme.Motion.standard) {
                navigation.showLibrary.toggle()
            }
        }
    }

    /// File ▸ Import Receipt…: the same validated path as the `mererun://library/import` link.
    private func importReceipt() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.json]
        panel.title = "Import receipt"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            let item = try library.importReceipt(at: url)
            navigation.open(libraryItem: item.id, mode: item.mode)
        } catch {
            navigation.deepLinkError = error.localizedDescription
        }
    }

    // MARK: - Running

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
            navigation.selectedLibraryID = request.id
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
            navigation.selectedLibraryID = conversationID
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

    private func retryLibraryItem(_ item: StudioLibraryItem) {
        guard let templateID = item.templateID,
              let commandDraft = item.commandDraft,
              let template = CommandCatalog.template(id: templateID) else {
            studioError = "This older Library item does not include a replayable command."
            return
        }
        let request = StudioRunRequest(
            mode: item.mode,
            templateID: templateID,
            template: template,
            draft: commandDraft
        )
        let preview = controller.commandPreview(template: template, draft: commandDraft, masksSecrets: true)
        let status: StudioLibraryStatus = controller.isRunning || controller.queuedRunCount > 0
            ? .queued
            : .running
        library.start(request: request, commandPreview: preview, status: status)
        navigation.selectedLibraryID = request.id
        _ = controller.run(studio: request)
    }

    /// Library ▸ Edit command: loads the row's exact command into the Console window.
    private func editLibraryItem(_ item: StudioLibraryItem) {
        guard let templateID = item.templateID,
              let commandDraft = item.commandDraft,
              let template = CommandCatalog.template(id: templateID) else {
            studioError = "This older Library item does not include editable command settings."
            return
        }
        controller.select(template)
        controller.draft = commandDraft
        openConsole(syncingComposer: false)
    }

    // MARK: - Adapters and training

    private func applyAdapter(_ adapter: StudioAdapterRow) {
        applyAdapterReference(adapter.path ?? adapter.id)
    }

    private func applyLocalAdapter(_ path: String) {
        applyAdapterReference(path)
    }

    /// Applies an adapter to the last prompt mode's draft and returns there. Modes whose adapters
    /// are only typed in the raw command (for example SCAIL-2) open the Console instead.
    private func applyAdapterReference(_ reference: String) {
        switch mode {
        case .music:
            let existing = draft.musicAdapterPaths
                .components(separatedBy: .newlines)
                .filter { !$0.isBlank }
            if !existing.contains(reference) {
                draft.musicAdapterPaths = (existing + [reference]).joined(separator: "\n")
            }
            navigation.open(destination: mode.destination)
        case .createImage, .chat, .code:
            draft.loraPath = reference
            navigation.open(destination: mode.destination)
        default:
            openConsole()
        }
    }

    private func openTraining(_ templateID: CommandTemplateID) {
        switch templateID {
        case .imageTrainLoRA:
            navigation.open(task: .imageTrain)
        case .textTrainLoRA:
            navigation.open(task: .chatTrain)
        case .musicTrainAdapter:
            navigation.open(task: .musicTrain)
        default:
            openConsole()
        }
    }

    // MARK: - Drafts and conversations

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
            navigation.selectedLibraryID = nil
        }
    }

    /// Starts a fresh, not-yet-persisted conversation (no library row until the first message).
    private func startNewConversation() {
        activeConversationID = nil
        navigation.selectedLibraryID = nil
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

    // MARK: - Readiness and models

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
            if modelUsageTermsByID[request.draft.model] != nil {
                pendingRestrictedPull = request
            } else {
                startPull(request)
            }
        } catch {
            studioError = error.localizedDescription
        }
    }

    private func startPull(
        _ request: StudioRunRequest,
        acknowledgingUsageTerms: Bool = false
    ) {
        var commandDraft = request.draft
        commandDraft.acceptModelLicense = acknowledgingUsageTerms
        let effectiveRequest = StudioRunRequest(
            id: request.id,
            mode: request.mode,
            templateID: request.templateID,
            template: request.template,
            draft: commandDraft,
            createdAt: request.createdAt,
            conversationID: request.conversationID
        )
        pendingPullRefresh = StudioReadinessRefresh(mode: request.mode, draft: draft)
        controller.readinessByMode[request.mode] = .checking
        // The canvas running overlay shows pull progress (bytes, speed, cancel) in place.
        if !controller.run(studio: effectiveRequest) {
            pendingPullRefresh = nil
            refreshReadiness()
        }
    }

    private func refreshReadiness() {
        controller.checkReadiness(for: mode, draft: draft)
    }

    /// Refreshes the composer's model chip from `model list`.
    private func refreshInstalledModels() {
        Task {
            let result = await controller.utilityCommandResult(args: ["model", "list"])
            guard result.exitCode == 0 else { return }
            let rows = StudioModelInventoryParser.rows(from: result.stdout)
            modelInventory = rows
            modelUsageTermsByID = Dictionary(
                uniqueKeysWithValues: rows.compactMap { row in
                    row.usageTerms.map { (row.id, $0) }
                }
            )
        }
    }

    // MARK: - Attachments

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

    /// Pastes an image from the clipboard into the well (Edit ▸ Paste / ⌘V when the canvas, not a
    /// text field, holds focus): the first empty image slot, else the first image slot. Prefers a
    /// pasted image file; otherwise writes the pasted bitmap to a temporary PNG.
    private func pasteImageFromClipboard() {
        guard let slot = mode.pastedImageSlot(in: draft) else { return }
        let pasteboard = NSPasteboard.general
        let urls = StudioAttachmentPasteboard.fileURLs(from: pasteboard, for: slot)
        if !urls.isEmpty {
            slot.attach(urls, to: &draft)
            studioError = nil
            return
        }
        do {
            guard let url = try StudioAttachmentPasteboard.writePastedImage(from: pasteboard) else {
                studioError = "The clipboard has no image to paste."
                return
            }
            slot.attach([url], to: &draft)
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

/// Which Library rows the column shows: the current domain's, or everything.
enum StudioLibraryScope: String, CaseIterable, Identifiable {
    case domain
    case all

    var id: String { rawValue }
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
