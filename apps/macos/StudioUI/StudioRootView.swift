import AppKit
import StudioKit
import SwiftUI
import UniformTypeIdentifiers

/// The Studio shell: a native split view whose sidebar lists the domains and whose detail area
/// renders the current task — the prompt workspace for the twelve composer modes, or one of the
/// former specialist sheets re-hosted inline. Navigation is one value (`NavigationModel`), and
/// the only sheets left are true tasks (terms, the mask editor, Relay sign-in, rename, the Guide).
package struct StudioRootView: View {
    @EnvironmentObject private var controller: MereRunController
    @EnvironmentObject private var library: StudioLibraryStore
    @EnvironmentObject private var navigation: NavigationModel
    private let seededDrafts: [StudioMode: StudioDraft]

    package init(seededDrafts: [StudioMode: StudioDraft] = [:]) {
        self.seededDrafts = seededDrafts
    }

    package var body: some View {
        StudioWorkspaceView(controller: controller, library: library, navigation: navigation, seededDrafts: seededDrafts)
    }
}

/// Scene composition owns the prompt controller's lifetime without putting it in global app state.
private struct StudioWorkspaceView: View {
    @ObservedObject private var controller: MereRunController
    @ObservedObject private var library: StudioLibraryStore
    @ObservedObject private var navigation: NavigationModel
    @State private var prompt: StudioPromptTaskController
    @Environment(\.openWindow) private var openWindow
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    // Persisted per scene so relaunch restores the last place, the last prompt mode, and the panel
    // layout. `studio.mode` keeps its v1 meaning (the last prompt mode) so drafts and readiness
    // stay attached to it while a System or Lab task is shown.
    @SceneStorage("studio.destination") private var storedDestination: StudioDestination = .default
    @SceneStorage("studio.mode") private var lastPromptMode: StudioMode = .createImage
    @SceneStorage("studio.showLibrary") private var storedShowLibrary = true
    @SceneStorage("studio.libraryScope") private var libraryScope: StudioLibraryScope = .domain
    @SceneStorage("studio.libraryView") private var libraryViewMode: StudioLibraryViewMode = .list
    @SceneStorage("studio.libraryKind") private var libraryKind: StudioLibraryKind = .all
    @SceneStorage("studio.libraryFavorites") private var libraryFavoritesOnly = false
    /// The prompt tasks whose inspector stays open, as `StudioInspectorTaskMemory` encodes them.
    @SceneStorage("studio.inspectorTasks") private var storedInspectorTasks = ""
    /// The unsent prompt, system text, and attachment of every prompt task, as
    /// `StudioDraftMemory` encodes them, so a relaunch resumes mid-sentence.
    @SceneStorage("studio.drafts") private var storedDrafts = ""
    @State private var detailWidth: CGFloat = 1024
    @State private var libraryOverlay = false
    @State private var columnVisibility: NavigationSplitViewVisibility = .all
    /// The status probe never answered within its grace period, so the footer says so.
    @State private var probeTimedOut = false
    @State private var isDropTargeted = false
    /// Which jobs exist; the feed re-derives its cards when one starts or finishes.
    @StateObject private var jobMonitor = StudioJobMonitor()
    /// The card of the Library row the user just picked, outlined for a moment.
    @State private var highlightedCardID: UUID?
    @State private var highlightReset: Task<Void, Never>?
    /// A run of this mode that finished while its card was off-screen ("New result ↓").
    @State private var newResultID: UUID?
    @State private var pendingPullRefresh: StudioReadinessRefresh?
    @State private var pendingRestrictedPull: StudioRunRequest?
    @State private var studioError: String?
    /// A run whose user-visible destination could not be created, explained once per launch.
    @State private var outputFallbackNotice: String?
    @State private var outputFallbackAnnounced = false
    /// Every `model list` row, installed or not, feeding the composer's model chip.
    @State private var modelInventory: [StudioModelInventoryRow] = []
    /// What Models ▸ Installed last reported, so the content header's subtitle shows the real inventory.
    @State private var modelInventorySummary: StudioModelInventorySummary?
    @State private var modelUsageTermsByID: [String: StudioModelUsageTerms] = [:]
    @State private var imageDatasetTask: StudioUtilityTask = .datasetDiscovery
    @AppStorage("mererun.app.hasCompletedWelcome") private var hasCompletedWelcome = false
    @FocusState private var promptFocused: Bool
    init(controller: MereRunController, library: StudioLibraryStore, navigation: NavigationModel,
         seededDrafts: [StudioMode: StudioDraft]) {
        self.controller = controller
        self.library = library
        self.navigation = navigation
        _prompt = State(initialValue: StudioPromptTaskController(controller: controller, library: library, seededDrafts: seededDrafts))
    }

    private var draft: StudioDraft {
        get { prompt.draft }
        nonmutating set { prompt.draft = newValue }
    }
    private var activatedMode: StudioMode? { prompt.activatedMode }
    private var activeConversationID: UUID? { prompt.activeConversationID }

    private var destination: StudioDestination { navigation.destination }

    /// The prompt mode the composer and canvas belong to. While a non-prompt task is shown this is
    /// the last prompt mode, so its draft, readiness, and conversation survive the detour.
    private var mode: StudioMode {
        destination.task.mode ?? lastPromptMode
    }

    private var showsPromptWorkspace: Bool {
        destination.task.mode != nil
    }

    /// The Library column belongs to prompt tasks only: Subjects, Realtime, Models, and the other
    /// Project, Session, and Manage tasks take the full width even inside a Create domain.
    private var showsLibraryColumn: Bool {
        navigation.showLibrary && destination.task.isPromptTask
    }

    private var showsInspectorColumn: Bool {
        showsPromptWorkspace && navigation.showsInspector(for: destination.task)
    }

    private var showsCommandColumn: Bool {
        navigation.showsCommandColumn(for: destination.task)
    }

    /// The feed's cards for the current mode: the Library rows plus the jobs still alive.
    private var feedCards: [StudioFeedCard] {
        _ = jobMonitor.generation
        return StudioFeedCardBuilder.cards(items: library.items, mode: mode, job: jobMonitor.job(requestID:))
    }

    private var runningFeedJob: Job? {
        feedCards.last { $0.kind == .running }?.job
    }

    private var queuedFeedCount: Int {
        feedCards.filter { $0.kind == .queued }.count
    }

    /// Whether the composer shows Stop instead of Run: only a conversation turn in flight. A
    /// generation in flight keeps Run available (the next one queues) and has Cancel on its card.
    private var isModeRunning: Bool {
        mode.isConversational && activeConversationRunning
    }

    private var showsConversation: Bool {
        mode.isConversational && (selectedItem == nil || selectedItem?.isConversation == true)
    }

    /// The `model pull` in flight for this mode's model, so the readiness card shows its progress.
    private var activePullJob: Job? {
        _ = jobMonitor.generation
        return jobMonitor.pullJob(for: StudioCommandAdapter.requiredModel(for: mode, draft: draft))
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

    private var showInspectorBinding: Binding<Bool> {
        Binding(
            get: { navigation.showsInspector(for: destination.task) },
            set: { shown in
                if shown != navigation.showsInspector(for: destination.task) { toggleInspector() }
            }
        )
    }

    private var showCommandBinding: Binding<Bool> {
        Binding(
            get: { navigation.showsCommandColumn(for: destination.task) },
            set: { shown in
                if shown != navigation.showsCommandColumn(for: destination.task) { toggleCommand() }
            }
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
            .environment(\.studioTaskSessions, controller.taskSessions)
            .environment(\.studioTaskScope, destination.task.rawValue)
    }

    // MARK: - Shell

    private var shell: some View {
        NavigationSplitView(columnVisibility: $columnVisibility) {
            StudioSidebar(
                selectedDomain: domainBinding,
                unavailableMessages: domainUnavailableMessages,
                status: machineStatus,
                runningJobs: runningJobCount,
                isActivityOpen: $navigation.showActivity
            )
            .toolbar(removing: .sidebarToggle)
        } detail: {
            detailArea
        }
        .toolbar(removing: .sidebarToggle)
        .background(MereRunTheme.background.ignoresSafeArea())
        // The controller publishes nil until the status probe answers; a probe that never answers
        // must still resolve, so the footer stops saying "Checking…" after the grace period.
        .task(id: controller.serverStatus == nil) {
            guard controller.serverStatus == nil else {
                probeTimedOut = false
                return
            }
            try? await Task.sleep(for: .seconds(StudioMachineStatus.checkingGracePeriod))
            guard !Task.isCancelled, controller.serverStatus == nil else { return }
            probeTimedOut = true
        }
        .overlay(alignment: .bottomLeading) { activityOverlay }
    }

    private var machineStatus: StudioMachineStatus {
        StudioMachineStatus(serverStatus: controller.serverStatus, probeTimedOut: probeTimedOut)
    }

    /// How many jobs the footer pill counts: the user's work, never a readiness probe.
    private var runningJobCount: Int {
        _ = jobMonitor.generation
        return StudioActivity.lanes.reduce(0) { $0 + controller.jobs.running(in: $1).count }
    }

    /// The Activity popover, drawn over the whole window rather than inside the sidebar column: it
    /// is 340pt wide and would be clipped by the column, and it must float over the Library.
    @ViewBuilder
    private var activityOverlay: some View {
        if navigation.showActivity {
            ZStack(alignment: .bottomLeading) {
                // Anywhere else in the window dismisses it, the way a popover does.
                Color.clear
                    .contentShape(Rectangle())
                    .onTapGesture { navigation.showActivity = false }
                StudioActivityPopover(
                    jobs: controller.jobs,
                    status: machineStatus,
                    appVersion: controller.appVersion,
                    cliVersion: controller.cliVersion,
                    modelsRoot: controller.modelsRoot,
                    resolvedCLI: controller.resolvedCLI,
                    onOpenServer: {
                        navigation.showActivity = false
                        navigation.open(domain: .server)
                    },
                    onOpenModels: {
                        navigation.showActivity = false
                        navigation.open(task: .modelsInstalled)
                    }
                )
                .padding(.leading, 10)
                .padding(.bottom, 56)
                Button("Close Activity") { navigation.showActivity = false }
                    .keyboardShortcut(.cancelAction)
                    .frame(width: 0, height: 0)
                    .opacity(0)
                    .accessibilityHidden(true)
            }
            .transition(reduceMotion ? .identity : .opacity)
        }
    }

    // The detail area runs to the top of the window (the title bar is hidden and nothing sits in
    // the window toolbar): the Library column on the left with its own header, and beside it the
    // content column whose first row is the 52pt domain header. With the sidebar collapsed the
    // traffic lights land over the detail's top-left corner, so whichever
    // header is first leaves room for them.
    private var layout: StudioLayoutPolicy.Presentation {
        StudioLayoutPolicy.presentation(width: detailWidth, library: showsLibraryColumn,
                                        inspector: showsInspectorColumn, command: showsCommandColumn)
    }

    private var detailArea: some View {
        GeometryReader { geometry in
            let policy = StudioLayoutPolicy.presentation(width: geometry.size.width, library: showsLibraryColumn,
                                                        inspector: showsInspectorColumn, command: showsCommandColumn)
            HStack(spacing: 0) {
                if policy.showsLibrary {
                    historyColumn.frame(width: StudioLayoutPolicy.libraryWidth)
                    Divider()
                }
                VStack(spacing: 0) {
                    contentHeader
                    banners
                    HStack(spacing: 0) {
                        domainContent.frame(maxWidth: .infinity, maxHeight: .infinity)
                        if policy.panelIsInline {
                            auxiliaryPanel.frame(width: policy.panelWidth)
                        }
                    }
                    .overlay(alignment: .trailing) {
                        if !policy.panelIsInline && (showsInspectorColumn || showsCommandColumn) {
                            ZStack(alignment: .trailing) {
                                Color.black.opacity(0.12).onTapGesture(perform: closeAuxiliaryPanel)
                                auxiliaryPanel.frame(width: policy.panelWidth)
                                    .shadow(color: .black.opacity(0.16), radius: 16, x: -6)
                            }
                        }
                    }
                }
            }
            .overlay(alignment: .leading) {
                if libraryOverlay && !policy.showsLibrary {
                    ZStack(alignment: .leading) {
                        Color.black.opacity(0.12).onTapGesture { libraryOverlay = false }
                        VStack(spacing: 0) {
                            Button("Close Library", systemImage: "xmark") { libraryOverlay = false }
                                .buttonStyle(.plain).padding(12)
                            historyColumn
                        }
                        .frame(width: min(320, geometry.size.width))
                        .background(MereRunTheme.background)
                        .shadow(color: .black.opacity(0.16), radius: 16, x: 6)
                    }
                }
            }
            .onChange(of: geometry.size.width, initial: true) { _, width in detailWidth = width }
        }
        .ignoresSafeArea(.container, edges: .top)
        .background(MereRunTheme.background.ignoresSafeArea())
        .foregroundStyle(MereRunTheme.textPrimary)
    }

    @ViewBuilder private var historyColumn: some View {
        if destination.domain == .chat { threadListColumn } else { libraryColumn }
    }

    @ViewBuilder private var auxiliaryPanel: some View {
        if showsInspectorColumn { inspectorColumn } else if showsCommandColumn { commandColumn }
    }

    private func closeAuxiliaryPanel() {
        if showsCommandColumn { toggleCommand() } else if showsInspectorColumn { toggleInspector() }
    }

    /// Space for the traffic lights while the sidebar is collapsed.
    private var windowChromeInset: CGFloat {
        columnVisibility == .detailOnly ? StudioContentHeader.collapsedSidebarInset : 0
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

        if let persistenceError = controller.taskSessions.lastPersistenceError {
            MereBanner(severity: .warning, text: persistenceError)
                .padding(.horizontal, MereRunTheme.Spacing.lg)
                .padding(.top, MereRunTheme.Spacing.sm)
        }

        if let outputFallbackNotice {
            MereBanner(
                severity: .warning,
                text: outputFallbackNotice,
                systemImage: "folder.badge.questionmark",
                onDismiss: { self.outputFallbackNotice = nil }
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
            items: StudioThreadListPresenter.mediaItems(in: library.items),
            domain: destination.domain,
            scope: $libraryScope,
            viewMode: $libraryViewMode,
            kind: $libraryKind,
            favoritesOnly: $libraryFavoritesOnly,
            progressByID: controller.progressByRequestID,
            selectedID: $navigation.selectedLibraryID,
            onSelect: selectLibraryItem,
            onDelete: deleteLibraryItems,
            onRename: library.rename,
            onToggleFavorite: toggleLibraryFavorite,
            onQuickLook: { QuickLookCoordinator.shared.preview($0) },
            onReveal: revealInFinder,
            onExport: exportLibraryItems,
            onRetry: retryLibraryItem,
            onEdit: editLibraryItem,
            leadingInset: windowChromeInset
        )
    }

    private var threadListColumn: some View {
        StudioThreadList(
            items: library.items,
            selectedID: activeConversationID,
            onSelect: openThread,
            onNewThread: startNewConversation,
            onDelete: deleteLibraryItem,
            onRename: library.rename,
            leadingInset: windowChromeInset
        )
    }

    // MARK: - Content header

    /// The 52pt header row at the top of the content column: domain glyph, title, and subtitle
    /// leading; the task control in the middle; Library, Inspector, and Command toggles trailing.
    private var contentHeader: some View {
        StudioContentHeader(
            domain: destination.domain,
            subtitle: domainSubtitle,
            task: taskBinding,
            showsPanelToggles: destination.task.isPromptTask,
            isLibraryShown: layout.showsLibrary || libraryOverlay,
            isInspectorShown: navigation.showsInspector(for: destination.task),
            isCommandShown: navigation.showsCommandColumn(for: destination.task),
            isSidebarShown: columnVisibility != .detailOnly,
            onToggleSidebar: {
                columnVisibility = columnVisibility == .detailOnly ? .all : .detailOnly
            },
            leadingInset: layout.showsLibrary ? 0 : windowChromeInset,
            onToggleLibrary: toggleLibrary,
            onToggleInspector: toggleInspector,
            onToggleCommand: toggleCommand
        )
    }

    // MARK: - Inspector and Command view

    private var inspectorColumn: some View {
        StudioInspector(
            mode: mode,
            draft: $prompt.draft,
            baseline: freshDraft(for: mode),
            modelInventory: modelInventory,
            readiness: readiness,
            lastSeed: lastSeed,
            onShowModels: { navigation.open(task: .modelsInstalled) },
            onShowAdapters: { navigation.open(task: .modelsAdapters) },
            onClose: toggleInspector
        )
    }

    private var baseTaskRequest: StudioRunRequest? {
        if showsPromptWorkspace {
            return try? StudioCommandAdapter.makeRequest(mode: mode, draft: draft, validating: false)
        }
        let key = destination.task.rawValue
        let chosen = controller.taskSessions.value(for: key + ".commandTemplate", default: Optional<CommandTemplateID>.none)
        guard let template = chosen.flatMap(CommandCatalog.template(id:)) ?? destination.task.commandTemplates.first else { return nil }
        let command = controller.taskSessions.value(for: key + ".commandDraft", default: template.defaultDraft())
        return StudioRunRequest(mode: template.libraryMode, templateID: template.id, template: template, draft: command)
    }

    private func commandForm(for request: StudioRunRequest) -> StudioConsoleDraft {
        controller.taskSessions.commandForm(for: request)
    }

    private func resolvedCommand(_ base: StudioRunRequest) -> StudioRunRequest {
        controller.taskSessions.resolving(base)
    }

    @ViewBuilder
    private var commandColumn: some View {
        if let request = baseTaskRequest {
            StudioTaskCommandView(template: request.template, seed: request.draft, form: Binding(
                get: { commandForm(for: request) },
                set: { edited in
                    if showsPromptWorkspace {
                        edited.applyingChanges(from: commandForm(for: request), to: &draft,
                                               mode: mode, templateID: request.templateID)
                    }
                    let source = baseTaskRequest ?? request
                    controller.taskSessions.set(StudioTaskCommandState(templateID: request.templateID,
                        sourceArguments: source.template.arguments(from: source.draft), form: edited),
                        for: request.templateID.studioTask.rawValue + ".commandOverride")
                }
            ), onRun: runStudioCommand, onClose: toggleCommand,
               canRun: !showsPromptWorkspace || (!readiness.blocksRun && !(mode.isConversational && activeConversationRunning)))
        }
    }

    /// Models reports its installed count and store size; every other domain keeps its tagline.
    private var domainSubtitle: String {
        if destination.domain == .models, let modelInventorySummary {
            return modelInventorySummary.subtitle
        }
        return destination.domain.subtitle
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
            StudioModelsView(
                onModelsChanged: {
                    refreshReadiness()
                    refreshInstalledModels()
                },
                onInventoryChanged: { modelInventorySummary = $0 },
                adapterTargetTitle: mode.destination.domain.title,
                onUseAdapter: applyAdapter,
                onTrain: openTraining
            )
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
            StudioServingConsoleView(monitor: controller.servingMonitor)
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
            if let selection = focusedResult, let item = library.items.first(where: { $0.id == selection.itemID }) {
                StudioResultWorkspaceView(item: item, url: selection.url, items: library.items,
                    onClose: { focusedResult = nil }, onVary: varyLibraryItem,
                    onSave: saveOutput, onContinue: continueResult)
            } else if mode.isConversational {
                converseSurface
            } else {
                canvas
            }

            if focusedResult == nil { composer }

            if let studioError {
                MereBanner(severity: .error, text: studioError, onDismiss: { self.studioError = nil })
                    .padding(.horizontal, 24)
                    .padding(.bottom, 16)
                    .padding(.top, -8)
                    .transition(.opacity.combined(with: .move(edge: .bottom)))
            }
        }
        .animation(reduceMotion ? nil : MereRunTheme.Motion.standard, value: studioError)
        .frame(minWidth: min(StudioLayoutPolicy.minimumCanvasWidth, detailWidth))
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

    @ViewBuilder
    private var canvas: some View {
        if showsConversation {
            StudioConversationView(
                item: activeConversationItem,
                liveText: activeConversationLiveText,
                isRunning: activeConversationRunning,
                mode: mode,
                onNewChat: startNewConversation,
                onCopy: copyToClipboard,
                onRetry: retryLastTurn,
                onEdit: editMessage,
                onUseExample: useExamplePrompt
            )
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else if let archetype = destination.task.analyzeArchetype {
            StudioAnalyzeCanvas(
                archetype: archetype,
                mode: mode,
                cards: feedCards,
                selectedID: navigation.selectedLibraryID,
                inputPath: draft.inputPath,
                readiness: readiness,
                pullJob: activePullJob,
                actions: feedActions,
                analyze: analyzeActions
            )
        } else {
            StudioFeedCanvas(
                mode: mode,
                cards: feedCards,
                readiness: readiness,
                pullJob: activePullJob,
                highlightedID: highlightedCardID,
                newResultID: $newResultID,
                actions: feedActions
            )
        }
    }

    private var focusedResult: StudioResultSelection? {
        get {
            let selection = controller.taskSessions.value(for: destination.task.rawValue + ".focus", default: Optional<StudioResultSelection>.none)
            return selection.flatMap { selected in library.items.contains(where: { $0.id == selected.itemID }) ? selected : nil }
        }
        nonmutating set { controller.taskSessions.set(newValue, for: destination.task.rawValue + ".focus") }
    }

    private func focusResult(_ item: StudioLibraryItem, _ url: URL) {
        guard StudioOutputFileKind.classify(url) == .image else {
            QuickLookCoordinator.shared.preview(url)
            return
        }
        focusedResult = StudioResultSelection(itemID: item.id, url: url)
    }

    private func continueResult(_ action: StudioResultContinuation, _ item: StudioLibraryItem, _ url: URL) {
        guard prompt.continueResult(action, item: item, url: url) else { return }
        navigation.selectedLibraryID = nil
        navigation.open(task: action.task)
        promptFocused = true
    }

    private var analyzeActions: StudioAnalyzeActions {
        StudioAnalyzeActions(
            replaceInput: chooseAttachment,
            openTask: openSiblingTask,
            save: saveAnalyzeResult
        )
    }

    private var feedActions: StudioFeedActions {
        StudioFeedActions(
            vary: varyLibraryItem,
            rerun: retryLibraryItem,
            useAsInput: useOutputAsInput,
            saveTo: saveOutput,
            cancel: { jobMonitor.cancel($0) },
            remove: removeQueued,
            retry: retryLibraryItem,
            delete: { deleteLibraryItem($0.id) },
            pullModel: pullModel,
            showDetails: {
                if !navigation.showsCommandColumn(for: destination.task) { toggleCommand() }
            },
            useExample: useExamplePrompt,
            attach: chooseAttachment,
            focus: focusResult
        )
    }

    /// The Converse archetype: thread header, transcript, and readiness in place of the canvas.
    private var converseSurface: some View {
        StudioConverseView(
            mode: mode,
            item: activeConversationItem,
            liveText: activeConversationLiveText,
            isRunning: activeConversationRunning,
            readiness: readiness,
            error: studioError,
            budgetChars: prompt.conversationBudgetChars(inventory: modelInventory),
            modelInventory: modelInventory,
            model: $prompt.draft.model,
            systemPrompt: $prompt.draft.secondaryText,
            onPullModel: pullModel,
            onShowDetails: { openConsole() },
            onShowModels: { navigation.open(task: .modelsInstalled) },
            onCopy: copyToClipboard,
            onRetry: retryLastTurn,
            onEdit: editMessage,
            onBranch: branchFromMessage,
            onUseExample: useExamplePrompt
        )
    }

    private var composer: some View {
        StudioComposer(
            mode: mode,
            draft: $prompt.draft,
            isRunning: isModeRunning,
            queuedCount: mode.isConversational ? 0 : queuedFeedCount,
            readiness: readiness,
            modelInventory: modelInventory,
            lastSeed: lastSeed,
            promptFocus: $promptFocused,
            onRun: runStudioCommand,
            onStop: stopModeRun,
            onShowModels: { navigation.open(task: .modelsInstalled) }
        )
    }

    // MARK: - Presentation

    private var presentedShell: some View {
        shell
            .sheet(isPresented: $navigation.showGuide) {
                StudioHelpSheet()
                    .environmentObject(controller)
                    .frame(width: 960, height: 680)
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
            canShowLibrary: destination.task.isPromptTask,
            showInspector: showInspectorBinding,
            canShowInspector: destination.task.isPromptTask,
            showCommand: showCommandBinding,
            canShowCommand: baseTaskRequest != nil,
            open: { navigation.open(destination: $0) },
            openDomain: { navigation.open(domain: $0) },
            newChat: startNewConversation,
            canNewChat: showsPromptWorkspace && mode.isConversational,
            runComposer: runStudioCommand,
            canRun: showsPromptWorkspace
                ? !readiness.blocksRun && !(mode.isConversational && activeConversationRunning)
                : baseTaskRequest != nil,
            stop: stopCurrentRun,
            canStop: currentTaskJob != nil,
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
            navigation.inspectorTasks = StudioInspectorTaskMemory.decode(storedInspectorTasks)
            jobMonitor.attach(controller.jobs)
            // Restore goes through the model so remembered tasks and the Vision Lab variant learn
            // the persisted destination; the reconciled prompt mode replaces a stale studio.mode.
            library.observe(controller: controller)
            let restoredMode = navigation.restore(destination: storedDestination, lastPromptMode: lastPromptMode)
            lastPromptMode = restoredMode
            prompt.importLegacyDrafts(storedDrafts)
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
        .onChange(of: navigation.inspectorTasks) { _, tasks in
            storedInspectorTasks = StudioInspectorTaskMemory.encode(tasks)
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
            if showsPromptWorkspace, activatedMode == mode {
                if let item = library.items.first(where: { $0.id == id }),
                   item.mode == mode || (mode.isConversational && item.isConversation) {
                    controller.taskSessions.rememberSelection(id, for: mode)
                } else if id == nil {
                    controller.taskSessions.rememberSelection(nil, for: mode)
                }
            }
            // A thread selected while Converse is shown (a deep link, or the list) opens in the
            // transcript; a thread of the other preset switches the task control to match.
            guard showsPromptWorkspace, activatedMode == mode, mode.isConversational,
                  let id,
                  let item = library.items.first(where: { $0.id == id }),
                  item.isConversation,
                  id != activeConversationID else { return }
            guard item.mode == mode else {
                navigation.open(task: item.mode.task)
                return
            }
            restoreConversation(item)
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
                // Only follow selection if this is the thread the user is currently viewing — a
                // background turn completing must not yank selection away from the foreground.
                if mode.isConversational, activeConversationID == conversationID {
                    navigation.selectedLibraryID = conversationID
                }
                refreshReadiness()
                return
            }

            // The card updates in place; selection stays where the user left it, and a finished
            // card that is scrolled out of view announces itself with the "New result" pill.
            let completedLibraryItem = result.requestID != nil
            if let requestID = result.requestID {
                if result.exitCode == 0, library.items.first(where: { $0.id == requestID })?.mode == mode {
                    newResultID = requestID
                }
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

    }

    // MARK: - Navigation

    /// Switches the composer, canvas, readiness, and Library selection to `newMode`. Honors a
    /// Library row the user just picked (so selecting a row of another domain lands on that row),
    /// otherwise opens the most recent item or thread of the mode.
    private func activateMode(_ newMode: StudioMode) {
        let activation = prompt.activate(newMode, preferredID: navigation.selectedLibraryID)
        studioError = nil
        navigation.selectedLibraryID = activation.selectedLibraryID
        lastPromptMode = activation.mode
        if activation.mode != newMode { navigation.open(task: activation.mode.task) }
        controller.checkReadiness(for: activation.mode, draft: draft)
        if activation.mode != .listen { promptFocused = true }
    }

    /// A Library row the user clicked. Rows of another mode switch the destination first;
    /// `activateMode` then keeps the clicked row selected. The feed scrolls to the row's card
    /// and outlines it briefly.
    private func selectLibraryItem(_ item: StudioLibraryItem) {
        libraryOverlay = false
        navigation.selectedLibraryID = item.id
        highlightCard(item.id)
        guard item.mode != mode || !showsPromptWorkspace else {
            if destination.task.isAnalyzeTask { prompt.selectAnalyzeInput(from: item) }
            return
        }
        navigation.open(destination: item.mode.destination)
    }

    private func highlightCard(_ id: UUID) {
        highlightReset?.cancel()
        highlightedCardID = id
        highlightReset = Task { @MainActor in
            try? await Task.sleep(nanoseconds: 1_800_000_000)
            guard !Task.isCancelled, highlightedCardID == id else { return }
            highlightedCardID = nil
        }
    }

    private func toggleInspector() {
        if reduceMotion {
            navigation.toggleInspector(for: destination.task)
        } else {
            withAnimation(MereRunTheme.Motion.standard) {
                navigation.toggleInspector(for: destination.task)
            }
        }
    }

    /// Every task exposes its current editable command in the same workspace.
    private func toggleCommand() {
        if reduceMotion {
            navigation.toggleCommandColumn(for: destination.task)
        } else {
            withAnimation(MereRunTheme.Motion.standard) {
                navigation.toggleCommandColumn(for: destination.task)
            }
        }
    }

    /// A thread picked in the Converse list. Threads of the other preset switch the task first
    /// (the selection observer then opens the thread); same-preset threads open directly.
    private func openThread(_ thread: StudioLibraryItem) {
        libraryOverlay = false
        guard thread.id != activeConversationID else { return }
        navigation.selectedLibraryID = thread.id
        guard thread.mode == mode else {
            navigation.open(task: thread.mode.task)
            return
        }
        restoreConversation(thread)
    }

    private func restoreConversation(_ thread: StudioLibraryItem) {
        prompt.restoreConversation(thread)
        studioError = nil
        promptFocused = true
    }

    private func deleteLibraryItem(_ id: UUID) {
        deleteLibraryItems([id], trashingFiles: false)
    }

    /// Library ▸ Delete, for one row or a whole batch. The files follow the rows into the Trash
    /// only when the user chose that in the confirmation.
    private func deleteLibraryItems(_ ids: Set<UUID>, trashingFiles: Bool) {
        let failures = library.delete(ids: ids, trashingFiles: trashingFiles)
        prompt.forgetConversations(ids)
        if let first = failures.first {
            studioError = failures.count == 1
                ? "Could not move \(first.lastPathComponent) to the Trash."
                : "Could not move \(failures.count) files to the Trash."
        }
        if let selected = navigation.selectedLibraryID, ids.contains(selected) {
            navigation.selectedLibraryID = nil
        }
    }

    private func toggleLibraryFavorite(_ id: UUID) {
        guard let item = library.items.first(where: { $0.id == id }) else { return }
        library.setFavorite(id: id, isFavorite: !item.isStarred)
    }

    private func revealInFinder(_ urls: [URL]) {
        let existing = urls.filter { FileManager.default.fileExists(atPath: $0.path) }
        guard !existing.isEmpty else {
            studioError = "Those files are no longer on disk."
            return
        }
        NSWorkspace.shared.activateFileViewerSelecting(existing)
    }

    /// "Save to…": copies a row's artifacts (or a whole batch's) somewhere the user picks, leaving
    /// the originals — and the Library rows that point at them — untouched.
    private func exportLibraryItems(_ selected: [StudioLibraryItem]) {
        let urls = selected.flatMap(\.allArtifactURLs)
            .filter { FileManager.default.fileExists(atPath: $0.path) }
        guard !urls.isEmpty else {
            studioError = "Those runs have no files to save."
            return
        }

        if urls.count == 1, let source = urls.first {
            let panel = NSSavePanel()
            panel.nameFieldStringValue = source.lastPathComponent
            panel.canCreateDirectories = true
            guard panel.runModal() == .OK, let destination = panel.url else { return }
            copyExport(from: [source], into: destination.deletingLastPathComponent(), names: [destination.lastPathComponent])
            return
        }

        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        panel.prompt = "Save"
        panel.message = "Choose a folder for \(urls.count) files."
        guard panel.runModal() == .OK, let directory = panel.url else { return }
        copyExport(from: urls, into: directory, names: urls.map(\.lastPathComponent))
    }

    private func copyExport(from sources: [URL], into directory: URL, names: [String]) {
        let fileManager = FileManager.default
        var failures = 0
        for (source, name) in zip(sources, names) {
            var destination = directory.appendingPathComponent(name)
            var counter = 2
            let stem = destination.deletingPathExtension().lastPathComponent
            let ext = destination.pathExtension
            while fileManager.fileExists(atPath: destination.path), counter < 1_000 {
                let candidate = ext.isEmpty ? "\(stem)-\(counter)" : "\(stem)-\(counter).\(ext)"
                destination = directory.appendingPathComponent(candidate)
                counter += 1
            }
            do {
                try fileManager.copyItem(at: source, to: destination)
            } catch {
                failures += 1
            }
        }
        if failures > 0 {
            studioError = failures == 1 ? "One file could not be saved." : "\(failures) files could not be saved."
        }
    }

    /// Opens the Command Console window with the composer's draft carried into the Advanced
    /// template for the current mode, so the console deepens the current task. An already-open
    /// console is only raised: its edits and run state are never reset from here.
    private func openConsole(syncingComposer: Bool = true) {
        if navigation.shouldSyncComposerToConsole(requested: syncingComposer), let request = baseTaskRequest {
            controller.select(request.template)
            controller.draft = request.draft
            controller.consoleSeedArguments = resolvedCommand(request).execution?.arguments
                ?? request.template.arguments(from: request.draft)
        }
        openWindow(id: StudioConsoleWindow.id)
    }

    private func toggleLibrary() {
        if !layout.showsLibrary && detailWidth < StudioLayoutPolicy.minimumCanvasWidth + StudioLayoutPolicy.libraryWidth + 1 {
            libraryOverlay.toggle()
            return
        }
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
        do {
            if !showsPromptWorkspace {
                if let base = baseTaskRequest { _ = try prompt.runTask(base, task: destination.task) }
                return
            }
            guard let submission = try prompt.runPrompt(inventory: modelInventory) else { return }
            if let reason = submission.outputFallbackReason, !outputFallbackAnnounced {
                outputFallbackAnnounced = true
                outputFallbackNotice = "\(reason) Saving to \(StudioOutputLocation.abbreviate(StudioOutputLocation.appOutputsRoot())) instead."
            }
            navigation.selectedLibraryID = submission.request.conversationID ?? submission.request.id
        } catch {
            studioError = error.localizedDescription
        }
    }

    /// The composer's Stop: the run of this mode in flight, or the thread's turn.
    private func stopModeRun() {
        stopCurrentRun()
    }

    /// Takes a queued run out of the queue and drops its row; a stale queued row from an earlier
    /// launch has no job and is simply dropped.
    private func removeQueued(_ card: StudioFeedCard) {
        if let job = card.job { jobMonitor.cancel(job) }
        deleteLibraryItem(card.id)
    }

    /// Runs the same command again with a fresh, recorded seed, so the variation is repeatable.
    private func varyLibraryItem(_ item: StudioLibraryItem) {
        guard var commandDraft = item.commandDraft else {
            studioError = "This older Library item does not include a replayable command."
            return
        }
        commandDraft.seed = String(Int.random(in: 1...Int(Int32.max)))
        runLibraryItem(item, draft: commandDraft)
    }

    /// A contextual next step on an Analyze result: opens the sibling task with this run's input
    /// and prompt carried over, so "Segment these" continues from the same picture.
    private func openSiblingTask(_ task: StudioTask) {
        if task.mode != nil {
            prompt.prepareAnalyzeHandoff(to: task)
            navigation.selectedLibraryID = nil
        }
        // A task without a composer reads this same draft, so its input is already carried.
        navigation.open(destination: task.destination)
    }

    /// The Analyze result column's "Save…" steps.
    private func saveAnalyzeResult(_ kind: StudioAnalyzeSaveKind) {
        guard let item = analyzeResultItem else { return }
        switch kind {
        case .json:
            guard let url = StudioAnalyzeDocumentSource.url(for: item) else {
                studioError = "This run did not write a result document."
                return
            }
            saveOutput(url)
        case .media:
            guard let url = item.outputURL else {
                studioError = "This run did not write an output file."
                return
            }
            saveOutput(url)
        case .text:
            if let url = StudioAnalyzeDocumentSource.url(for: item), url.pathExtension != "json" {
                saveOutput(url)
                return
            }
            saveText(item.outputText, suggestedName: "transcript.txt")
        }
    }

    /// The run the Analyze canvas is showing: the picked Library row, else this mode's newest.
    private var analyzeResultItem: StudioLibraryItem? {
        let finished = feedCards.filter { $0.kind == .generation }
        if let selectedLibraryID = navigation.selectedLibraryID,
           let picked = finished.first(where: { $0.id == selectedLibraryID }) {
            return picked.item
        }
        return finished.last?.item
    }

    private func saveText(_ text: String?, suggestedName: String) {
        guard let text, !text.isBlank else {
            studioError = "This run left no text to save."
            return
        }
        guard let destination = StudioSpecialistFiles.saveFile(
            title: "Save result",
            suggestedName: suggestedName
        ) else { return }
        do {
            try text.write(to: destination, atomically: true, encoding: .utf8)
        } catch {
            studioError = "Could not save \(destination.lastPathComponent): \(error.localizedDescription)"
        }
    }

    /// Loads an output into the composer's well as the next run's input.
    private func useOutputAsInput(_ url: URL) {
        guard draft.attach(dropped: [url], for: mode) else {
            studioError = "\(mode.title) does not take \(url.lastPathComponent) as an input."
            return
        }
        studioError = nil
        promptFocused = true
    }

    /// Copies an output to a location the user picks.
    private func saveOutput(_ url: URL) {
        guard let destination = StudioSpecialistFiles.saveFile(
            title: "Save output",
            suggestedName: url.lastPathComponent
        ) else { return }
        do {
            try StudioFileExport.copy(url, to: destination)
        } catch {
            studioError = "Could not save \(url.lastPathComponent): \(error.localizedDescription)"
        }
    }

    /// Stops what the composer's Stop circle points at: the streaming turn of the open thread
    /// in Converse, otherwise the foreground run.
    private var currentTaskJob: Job? {
        _ = jobMonitor.generation
        return prompt.currentJob(for: destination.task)
    }

    private func stopCurrentRun() {
        prompt.stop(task: destination.task)
    }

    private func copyToClipboard(_ text: String) {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(text, forType: .string)
    }

    /// Re-runs the latest turn: drops the last assistant reply (if any) and re-sends the thread
    /// ending at the last user message, reusing the thread's own system prompt and model.
    private func retryLastTurn() {
        do {
            if try prompt.retryLastTurn(inventory: modelInventory) != nil { studioError = nil }
        } catch {
            studioError = error.localizedDescription
            promptFocused = true
        }
    }

    private func retryLibraryItem(_ item: StudioLibraryItem) {
        guard let commandDraft = item.commandDraft else {
            studioError = "This older Library item does not include a replayable command."
            return
        }
        runLibraryItem(item, draft: commandDraft)
    }

    /// Submits a Library row's command again as a new row, with `draft` in place of its own.
    private func runLibraryItem(_ item: StudioLibraryItem, draft commandDraft: CommandDraft) {
        do {
            let variationSeed = commandDraft.seed != item.commandDraft?.seed ? commandDraft.seed : nil
            navigation.selectedLibraryID = try prompt.replay(item, variationSeed: variationSeed).id
        } catch {
            studioError = error.localizedDescription
        }
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
        // A console run recorded the argv it launched; reopen on that rather than on the draft,
        // which cannot carry an option the console form edited but the draft has no field for.
        controller.consoleSeedArguments = item.commandArguments
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
            if !navigation.showsInspector(for: mode.task) { navigation.toggleInspector(for: mode.task) }
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
        prompt.freshDraft(for: mode)
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
        guard prompt.editMessage(messageID) else { return }
        navigation.selectedLibraryID = activeConversationID
        promptFocused = true
    }

    /// Branches a new thread at a turn. From a user turn: the thread up to (not including) that
    /// turn, with its text loaded into the composer so the edit runs in the branch and the
    /// original keeps its history. From an assistant turn: the thread through that reply.
    private func branchFromMessage(_ messageID: UUID) {
        guard let activation = prompt.branchFromMessage(messageID) else { return }
        navigation.selectedLibraryID = activation.selectedLibraryID
        if activation.mode != mode { navigation.open(task: activation.mode.task) }
        studioError = nil
        promptFocused = true
    }

    /// Starts a fresh, not-yet-persisted conversation (no library row until the first message).
    private func startNewConversation() {
        libraryOverlay = false
        guard activeConversationID != nil else { promptFocused = true; return }
        prompt.startNewConversation()
        navigation.selectedLibraryID = nil
        studioError = nil
        promptFocused = true
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
        // The feed's readiness card shows the pull's own progress (bytes, speed, Cancel).
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
            let result = await controller.utilityCommandResult(args: ["model", "list", "--json"])
            guard result.exitCode == 0 else { return }
            let rows = StudioModelInventoryParser.rows(from: result.stdout)
            modelInventory = rows
            modelInventorySummary = StudioModelInventorySummary(
                installedCount: rows.filter(\.isInstalled).count,
                storageBytes: modelInventorySummary?.storageBytes
            )
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

}

private struct StudioReadinessRefresh: Equatable {
    let mode: StudioMode
    let draft: StudioDraft
}
