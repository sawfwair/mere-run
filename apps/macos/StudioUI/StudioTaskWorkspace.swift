import AppKit
import StudioKit
import SwiftUI
import UniformTypeIdentifiers

private struct StudioTaskRunnerKey: EnvironmentKey {
    static let defaultValue: StudioTaskRunner? = nil
}

extension EnvironmentValues {
    /// The one submission path for task drafts, set by the window root beside `studioTaskSessions`.
    var studioTaskRunner: StudioTaskRunner? {
        get { self[StudioTaskRunnerKey.self] }
        set { self[StudioTaskRunnerKey.self] = newValue }
    }
}

/// The counterpart of the root's prompt workspace for a mode-less Generate or Analyze task: the
/// archetype's canvas (the Analyze board or the generation feed, filtered to the task's rows)
/// over `StudioTaskComposer`, both bound to the task's `StudioTaskDraft`. The draft lives in
/// `StudioTaskSessions` under `"<task>.taskDraft"` — the same value the root's inspector column
/// and Command view edit — so the workspace owns no draft state of its own and survives being
/// replaced.
///
/// The root routes every contract-backed Generate or Analyze task here.
struct StudioTaskWorkspace: View {
    @Environment(\.studioScopeSource) private var scopeSource
    let task: StudioTask

    @EnvironmentObject private var controller: MereRunController
    @EnvironmentObject private var library: StudioLibraryStore
    @EnvironmentObject private var navigation: NavigationModel
    @Environment(\.studioTaskRunner) private var runner
    @Environment(\.studioTaskSessions) private var sessions
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @ObservedObject private var models: StudioModelStore
    @StateObject private var jobMonitor = StudioJobMonitor()
    @FocusState private var promptFocused: Bool
    @State private var error: String?
    /// What Run again or Vary left out of a Library row's recorded command.
    @State private var replayNotice: StudioScopeNotice?
    @State private var highlightedCardID: UUID?
    @State private var newResultID: UUID?
    @State private var isDropTargeted = false

    init(task: StudioTask, models: StudioModelStore) {
        self.task = task
        _models = ObservedObject(wrappedValue: models)
    }

    // MARK: Draft

    /// The task's draft: the parked one, else the page draft imported once, else a fresh one on
    /// the task's first template. The session store memoizes the decoded value and every reader
    /// (this view, the root's inspector column, the Command view) goes through it, so writes
    /// land in one place and a re-render never decodes the same bytes twice.
    private var draft: StudioTaskDraft {
        get { sessions?.taskDraft(for: task) ?? StudioTaskDraft(task: task) ?? StudioTaskDraft(templateID: .custom) }
        nonmutating set { sessions?.setTaskDraft(newValue, for: task) }
    }

    private var draftBinding: Binding<StudioTaskDraft> {
        Binding(get: { draft }, set: { draft = $0 })
    }

    private var presentation: StudioTaskPresentation {
        task.presentation.attaching(StudioTaskSchema.primarySlot(for: draft.templateID))
    }

    private var readiness: ModelReadinessState {
        controller.readiness(for: task)
    }

    private var feedCards: [StudioFeedCard] {
        _ = jobMonitor.generation
        return StudioFeedCardBuilder.cards(items: library.items, task: task, job: jobMonitor.job(requestID:))
    }

    private var queuedCount: Int {
        feedCards.filter { $0.kind == .queued }.count
    }

    private var activePullJob: Job? {
        _ = jobMonitor.generation
        return jobMonitor.pullJob(for: StudioTaskSchema.modelID(for: draft, source: scopeSource))
    }

    private var focusedResult: StudioResultSelection? {
        get { sessions?.focusedResult(for: task, items: library.items) }
        nonmutating set { sessions?.setFocus(newValue, for: task) }
    }

    /// The typed input of a `.text` Analyze task is the composer's prompt, edited on the canvas.
    private var analyzeInputKind: StudioAnalyzeInputKind? {
        task.analyzeArchetype?.inputKind(for: draft.templateID)
    }

    // MARK: Body

    var body: some View {
        VStack(spacing: 0) {
            if let compared = comparedItems {
                StudioCompareView(items: compared,
                    onClose: { sessions?.setComparison(nil, for: task); promptFocused = true },
                    onKeep: { library.setFavorite(id: $0.id, isFavorite: !$0.isStarred) }, onUseSettings: useSettings)
            } else if let selection = focusedResult, let item = library.items.first(where: { $0.id == selection.itemID }) {
                StudioResultWorkspaceView(item: item, url: selection.url, items: library.items,
                    onClose: { focusedResult = nil; promptFocused = true }, onVary: vary,
                    onSave: saveOutput, onContinue: { _, _, _ in })
            } else {
                canvas
            }

            if focusedResult == nil && comparedItems == nil { composer }

            if let error {
                MereBanner(severity: .error, text: error, onDismiss: { self.error = nil })
                    .padding(.horizontal, 24)
                    .padding(.bottom, 16)
                    .padding(.top, -8)
                    .transition(.opacity.combined(with: .move(edge: .bottom)))
            } else if let replayNotice {
                MereBanner(
                    severity: .info, text: replayNotice.accessibilityLabel, systemImage: "eye.slash",
                    onDismiss: { self.replayNotice = nil }
                )
                .padding(.horizontal, 24)
                .padding(.bottom, 16)
                .padding(.top, -8)
                .transition(.opacity.combined(with: .move(edge: .bottom)))
            }
        }
        .animation(reduceMotion ? nil : MereRunTheme.Motion.standard, value: error)
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
            var next = draft
            guard next.attach(dropped: urls, slots: next.slots(source: scopeSource)) else { return false }
            draft = next
            error = nil
            return true
        } isTargeted: { targeted in
            withAnimation(MereRunTheme.Motion.quick) {
                isDropTargeted = targeted && !draft.slots(source: scopeSource).isEmpty
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
        .onPasteCommand(of: [.fileURL, .image, .audio]) { _ in
            // The canvas takes a paste the way it takes a drop: into the first slot that fits.
            StudioAttachmentPaste.paste(into: &draft, slots: draft.slots(source: scopeSource), allowsText: false)
        }
        .onAppear {
            jobMonitor.attach(controller.jobs)
            refreshReadiness()
            takeComposerFocusRequest()
        }
        .onChange(of: navigation.composerFocusRequest) { _, _ in takeComposerFocusRequest() }
        .onChange(of: StudioTaskSchema.requirement(for: draft, source: scopeSource)) { _, _ in
            error = nil
            refreshReadiness()
        }
        .onChange(of: draft.templateID) { _, _ in
            error = nil
            refreshReadiness()
        }
        .onChange(of: navigation.selectedLibraryID) { _, id in
            guard let id, let item = library.items.first(where: { $0.id == id }) else { return }
            // A pick of another variant's run shows that variant, so the Analyze canvas can draw it.
            if let templateID = item.templateID, templateID.studioTask == task, templateID != draft.templateID {
                var next = draft
                next.switchTemplate(to: templateID)
                draft = next
            }
            highlightCard(id)
        }
        .onReceive(controller.runCompletions) { result in
            guard let requestID = result.requestID, result.exitCode == 0,
                  library.items.first(where: { $0.id == requestID })?.templateID?.studioTask == task else { return }
            newResultID = requestID
            refreshReadiness()
        }
    }

    // MARK: Canvas

    @ViewBuilder
    private var canvas: some View {
        if let archetype = task.analyzeArchetype {
            StudioAnalyzeCanvas(
                archetype: archetype,
                presentation: presentation,
                cards: feedCards,
                selectedID: navigation.selectedLibraryID,
                inputPath: draft.primaryInputPath,
                readiness: readiness,
                pullJob: activePullJob,
                actions: feedActions,
                readinessActions: readinessActions,
                analyze: analyzeActions,
                templateID: draft.templateID,
                textInput: analyzeInputKind == .text ? draftBinding.prompt : nil
            )
        } else {
            StudioFeedCanvas(
                presentation: presentation,
                cards: feedCards,
                readiness: readiness,
                pullJob: activePullJob,
                highlightedID: highlightedCardID,
                newResultID: $newResultID,
                actions: feedActions,
                readinessActions: readinessActions
            )
        }
    }

    private var composer: some View {
        // Like a generation in the prompt workspace, a run in flight keeps Run available (the
        // next one queues) and has Cancel on its card; only a conversation turn shows Stop.
        StudioTaskComposer(
            task: task,
            draft: draftBinding,
            isRunning: false,
            queuedCount: queuedCount,
            readiness: readiness,
            modelInventory: models.rows,
            showsPrompt: analyzeInputKind != .text,
            promptFocus: $promptFocused,
            onRun: run,
            onStop: { runner?.stop(task: task) },
            onShowModels: { navigation.open(task: .modelsInstalled) },
            onRunVariations: StudioVariations.applies(to: draft, source: scopeSource) ? { runVariations($0) } : nil,
            showsScopeNote: !navigation.showCommandColumn && !(task.showsPromptChrome && navigation.showsInspector(for: task))
        )
    }

    // MARK: Actions

    private var feedActions: StudioFeedActions {
        StudioFeedActions(
            vary: vary,
            rerun: rerun,
            saveTo: saveOutput,
            cancel: { _ = jobMonitor.cancel($0) },
            remove: removeQueued,
            retry: rerun,
            delete: { delete($0.id) },
            useSettings: useSettings,
            pullModel: { pull(modelID: $0) },
            useExample: { example in
                var next = draft
                next.prompt = example
                draft = next
                promptFocused = true
            },
            attach: inputTarget,
            focus: focusResult,
            runVariations: replayVariations,
            compare: { sessions?.setComparison($0, for: task) }
        )
    }

    private var readinessActions: StudioReadinessActions {
        StudioReadinessActions(
            scope: StudioTaskSchema.modelScope(for: draft, source: scopeSource),
            model: draftBinding.model,
            modelInventory: models.rows,
            pullModel: { pull(modelID: StudioTaskSchema.modelID(for: draft, source: scopeSource)) },
            openModels: { navigation.open(task: .modelsInstalled) },
            recheck: refreshReadiness
        )
    }

    private var analyzeActions: StudioAnalyzeActions {
        StudioAnalyzeActions(
            input: inputTarget,
            openTask: { target, _ in navigation.open(destination: target.destination) },
            save: saveAnalyzeResult
        )
    }

    private func run() {
        error = nil
        guard let runner else { return }
        do {
            let request = try runner.run(draft, task: task)
            navigation.selectedLibraryID = request.id
        } catch {
            self.error = error.localizedDescription
        }
    }

    /// The comparison this task's page shows in place of its canvas.
    private var comparedItems: [StudioLibraryItem]? {
        sessions?.comparison(for: task, items: library.items)
    }

    /// The composer's "Run variations": the draft once per new seed, as one group.
    private func runVariations(_ count: StudioVariationCount) {
        error = nil
        guard let runner else { return }
        do {
            let requests = try runner.runVariations(draft, task: task, seeds: StudioVariations.seeds(count: count.rawValue))
            navigation.selectedLibraryID = requests.last?.id
        } catch {
            self.error = error.localizedDescription
        }
    }

    /// "Run variations" on a result: its recorded command once per new seed, as one group.
    private func replayVariations(_ item: StudioLibraryItem, _ count: StudioVariationCount) {
        error = nil
        guard let runner else { return }
        do {
            let requests = try runner.replayVariations(of: item, seeds: StudioVariations.seeds(count: count.rawValue))
            navigation.selectedLibraryID = requests.last?.id
            replayNotice = StudioLibraryReplay.notice(for: item, source: scopeSource)
        } catch {
            self.error = error.localizedDescription
        }
    }

    private func refreshReadiness() {
        controller.checkReadiness(for: task, requirement: StudioTaskSchema.requirement(for: draft, source: scopeSource))
    }

    /// The primary slot, for the empty state's Choose… and the Analyze canvas's Replace: a pick
    /// from disk or the Library lands where the well would put it.
    private var inputTarget: StudioAttachTarget? {
        guard let slot = StudioTaskSchema.primarySlot(for: draft.templateID) else { return nil }
        return StudioAttachTarget(requirement: StudioAttachmentRequirement(slot: slot)) { urls in
            var next = draft
            slot.attach(urls, to: &next)
            draft = next
            error = nil
        }
    }

    /// Takes focus for the composer when Use as input or Send to asked for this task's.
    private func takeComposerFocusRequest() {
        guard navigation.composerFocusRequest == task else { return }
        navigation.composerFocusRequest = nil
        error = nil
        // A workspace that has just appeared is not in the window yet; focus lands next turn.
        Task { @MainActor in promptFocused = true }
    }

    /// Library ▸ "Use these settings" on one of this task's rows: the recorded command becomes
    /// the draft.
    private func useSettings(_ item: StudioLibraryItem) {
        guard let restored = StudioLibraryDraftRestoration.taskDraft(from: item, source: scopeSource) else {
            error = "This run's command can't be loaded into the composer. Use Edit command… to change it."
            return
        }
        var next = draft
        next.adopt(restored)
        let overrideKey = task.rawValue + ".commandOverride"
        // The draft's setter writes through these sessions, so without them there is nothing to write.
        sessions?.undoably(StudioPromptTaskController.useSettingsUndoName,
                           keys: [StudioTaskSessions.taskDraftKey(task), overrideKey]) {
            draft = next
            sessions?.set(Optional<StudioTaskCommandState>.none, for: overrideKey)
        }
        error = nil
        navigation.selectedLibraryID = item.id
        promptFocused = true
    }

    private func rerun(_ item: StudioLibraryItem) {
        replay(item, variationSeed: nil)
    }

    private func vary(_ item: StudioLibraryItem) {
        replay(item, variationSeed: String(Int.random(in: 1...Int(Int32.max))))
    }

    /// Runs a row's recorded command again as a new row, never the current draft's edits.
    private func replay(_ item: StudioLibraryItem, variationSeed: String?) {
        guard let runner,
              let request = StudioLibraryReplay.request(for: item, variationSeed: variationSeed, source: scopeSource) else {
            error = "This older Library item does not include a replayable command."
            return
        }
        do {
            navigation.selectedLibraryID = try runner.run(request: request, task: task).id
            replayNotice = StudioLibraryReplay.notice(for: item, source: scopeSource)
        } catch {
            self.error = error.localizedDescription
        }
    }

    private func removeQueued(_ card: StudioFeedCard) {
        if let job = card.job { _ = jobMonitor.cancel(job) }
        delete(card.id)
    }

    private func delete(_ id: UUID) {
        let failures = library.delete(ids: [id], trashingFiles: false)
        sessions?.forgetLibraryItems([id])
        if let first = failures.first { error = "Could not move \(first.lastPathComponent) to the Trash." }
        if navigation.selectedLibraryID == id { navigation.selectedLibraryID = nil }
    }

    private func focusResult(_ item: StudioLibraryItem, _ url: URL) {
        guard item.allArtifactURLs.contains(url), FileManager.default.fileExists(atPath: url.path) else {
            error = "That result is no longer on disk."
            return
        }
        guard StudioOutputFileKind.classify(url) == .image else {
            QuickLookCoordinator.shared.preview(url)
            return
        }
        navigation.selectedLibraryID = item.id
        focusedResult = StudioResultSelection(itemID: item.id, url: url)
    }

    private func highlightCard(_ id: UUID) {
        highlightedCardID = id
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: 1_800_000_000)
            if highlightedCardID == id { highlightedCardID = nil }
        }
    }

    /// Gets a managed model through the same `model pull` job the readiness card reports.
    private func pull(modelID: String) {
        guard !modelID.isBlank, let template = CommandCatalog.template(id: .modelPull) else { return }
        var commandDraft = template.defaultDraft()
        commandDraft.model = modelID
        let request = StudioRunRequest(mode: template.libraryMode, templateID: .modelPull, template: template, draft: commandDraft)
        if !models.startPull(request) {
            error = controller.status
            refreshReadiness()
        }
    }

    private func saveOutput(_ url: URL) {
        guard let destination = StudioFilePanels.saveFile(title: "Save output", suggestedName: url.lastPathComponent) else {
            return
        }
        do {
            try StudioFileExport.copy(url, to: destination)
        } catch {
            self.error = "Could not save \(url.lastPathComponent): \(error.localizedDescription)"
        }
    }

    /// The run the Analyze canvas is showing: the picked Library row, else this task's newest run
    /// of the chosen variant (the canvas's own reading).
    private var analyzeResultItem: StudioLibraryItem? {
        let finished = feedCards.filter { $0.kind == .generation && $0.item.templateID == draft.templateID }
        if let selectedLibraryID = navigation.selectedLibraryID,
           let picked = finished.first(where: { $0.id == selectedLibraryID }) {
            return picked.item
        }
        return finished.last?.item
    }

    private func saveAnalyzeResult(_ kind: StudioAnalyzeSaveKind) {
        guard let item = analyzeResultItem else { return }
        switch kind {
        case .json:
            guard let url = StudioAnalyzeDocumentSource.url(for: item) else {
                error = "This run did not write a result document."
                return
            }
            saveOutput(url)
        case .media:
            guard let url = item.outputURL else {
                error = "This run did not write an output file."
                return
            }
            saveOutput(url)
        case .text:
            if let url = StudioAnalyzeDocumentSource.url(for: item), url.pathExtension != "json" {
                saveOutput(url)
                return
            }
            guard let text = item.outputText.flatMap(Self.savedText), !text.isBlank else {
                error = "This run left no text to save."
                return
            }
            guard let destination = StudioFilePanels.saveFile(title: "Save result", suggestedName: "result.txt") else { return }
            do {
                try text.write(to: destination, atomically: true, encoding: .utf8)
            } catch {
                self.error = "Could not save \(destination.lastPathComponent): \(error.localizedDescription)"
            }
        }
    }

    /// The words "Save text" writes from a run's captured output: the protected text of an
    /// anonymization (its JSON is the Save JSON action's), else the output as printed.
    private static func savedText(from output: String) -> String {
        if case .anonymization(let document) = StudioAnalyzeDocument.decode(Data(output.utf8)) {
            return document.protectedText
        }
        return output
    }
}
